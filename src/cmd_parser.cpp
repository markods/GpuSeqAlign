#include "cmd_parser.hpp"
#include "defer.hpp"
#include "io.hpp"
#include "nw_algorithm.hpp"
#include "run_types.hpp"
#include <algorithm>
#include <iostream>

static NwStat setStringArgOnce(
    const int argc,
    const char* argv[],
    int& i,
    std::optional<std::string>& arg,
    const std::string& arg_name)
{
    if (arg.has_value())
    {
        std::cerr << "error: parameter already set: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }
    if (i + 1 >= argc)
    {
        std::cerr << "error: expected parameter value: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }

    arg.emplace(argv[++i]);
    return NwStat::success;
}

static NwStat setStringVectArg(
    const int argc,
    const char* argv[],
    int& i,
    std::optional<std::vector<std::string>>& arg,
    const std::string& arg_name)
{
    if (i + 1 >= argc)
    {
        std::cerr << "error: expected parameter value: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }

    if (!arg.has_value())
    {
        arg.emplace(std::vector<std::string> {});
    }

    arg.value().push_back(argv[++i]);
    return NwStat::success;
}

static NwStat setIntArgOnce(
    const int argc,
    const char* argv[],
    int& i,
    std::optional<int>& arg,
    const std::string& arg_name)
{
    if (arg.has_value())
    {
        std::cerr << "error: parameter already set: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }
    if (i + 1 >= argc)
    {
        std::cerr << "error: expected parameter value: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }

    try
    {
        arg.emplace(std::stoi(argv[++i]));
    }
    catch (const std::invalid_argument&)
    {
        std::cerr << "error: parameter value should be int: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }
    catch (const std::out_of_range&)
    {
        std::cerr << "error: parameter value is out-of-range for int: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }

    return NwStat::success;
}

static NwStat setSwitchArgOnce(
    std::optional<bool>& arg,
    const std::string& arg_name)
{
    if (arg.has_value())
    {
        std::cerr << "error: parameter already set: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }

    arg.emplace(true);
    return NwStat::success;
}

template <typename T>
static NwStat expectNonEmptyArg(std::optional<T>& arg, const std::string& arg_name)
{
    if (!arg.has_value())
    {
        std::cerr << "error: expected parameter: \"" << arg_name << "\"\n";
        return NwStat::errorInvalidValue;
    }
    return NwStat::success;
}

template <typename T>
static void setDefaultIfArgEmpty(std::optional<T>& arg, const T& value)
{
    if (!arg.has_value())
    {
        arg.emplace(value);
    }
}

static void print_cmd_usage(std::ostream& os)
{
    os << "nw --algParamPath \"path\" --seqPath \"path\" [params]\n"
          "\n"
          "Parameters:\n"
          "-b, --substPath <path>     Path of JSON substitution matrices file, defaults to \"./resrc/subst.json\".\n"
          "-r, --algParamPath <path>  Path of JSON algorithm parameters file.\n"
          "-s, --seqPath <path>       Path of FASTA file with sequences to be aligned.\n"
          "-p, --seqPairPath <path>   Path of TXT file with sequence pairs to be aligned. Each line has the format \"seqY seqX\",\n"
          "                           where \"seqY\" and \"seqX\" are sequence ids. It's possible to specify a substring\n"
          "                           e.g. \"seqX[l:r]\", starting from element \"l\" inclusive until element \"r\" exclusive.\n"
          "                           The start/end of the interval can be omitted: \"[l:]\", \"[:r]\", \"[:]\".\n"
          "                           If the TXT file is not specified, then all sequences in the FASTA file except the first\n"
          "                           are aligned to the first sequence. In that case, there must be two or more sequences\n"
          "                           in the FASTA file.\n"
          "-o, --resPath <path>       Path of TSV test bench results file, defaults to \"./logs/%{datetime}.tsv\".\n"
          "\n"
          "--substName <name>         Specify which substitution matrix from the \"subst\" file will be used. Defaults to\n"
          "                           \"blosum62\".\n"
          "--gapoCost <cost>          Gap open cost. Integer, defaults to -11.\n"
          "--gapeCost <cost>          Unused. Gap extend cost. Integer, defaults to 0.\n"
          "--algName <name>           Specify which algorithm from the \"algParam\" JSON file will be used. Can be specified\n"
          "                           multiple times, in which case those algorithms will be used, in that order.\n"
          "                           If not specified, all algorithms in the \"algParam\" JSON file are used, in that order.\n"
          "--refAlgName <name>        Specify the algorithm name which should be considered as the source of truth.\n"
          "                           If not specified, defaults to the first algorithm in the \"algParam\" JSON file.\n"
          "--warmupPerAlign <num>     Number of warmup runs per alignments. Nonnegative integer, defaults to 0.\n"
          "--samplesPerAlign <num>    Number of runs per alignment. Nonnegative integer, defaults to 1.\n"
          "\n"
          "--fCalcTrace               Should the trace be calculated. Defaults to false.\n"
          "--fCalcScoreHash           Should the score matrix hash be calculated. Used to verify correctness with the reference\n"
          "                           algorithm implementation. Defaults to false.\n"
          "--fWriteProgress           Should progress be printed on stdout. Defaults to false.\n"
          "--debugPath <path>         For debug purposes, path of the TXT file where score matrices/traces will be\n"
          "                           written to, once per alignment. Defaults to \"%{datetime}_debug.txt\" if\n"
          "                           fPrintScore or fPrintTrace are passed, otherwise \"\".\n"
          "--fPrintScore              Should the score matrix be printed. Defaults to false.\n"
          "--fPrintTrace              Should the trace be printed. Defaults to false.\n"
          "\n"
          "-h, --help                 Print help and exit.\n";
}

NwStat parseCmdArgs(const int argc, const char* argv[], NwCmdArgs& cmdArgs)
{
    if (NwStat stat = isoDatetimeAsString(cmdArgs.isoDateTime); stat != NwStat::success)
    {
        std::cerr << "error: failed to get local time\n";
        return stat;
    }

    if (argc == 1)
    {
        print_cmd_usage(std::cout);
        std::cerr << "error: expected command parameters\n";
        return NwStat::errorInvalidValue;
    }

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-b" || arg == "--substPath")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.substPath, arg));
        }
        else if (arg == "-r" || arg == "--algParamPath")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.algParamPath, arg));
        }
        else if (arg == "-s" || arg == "--seqPath")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.seqPath, arg));
        }
        else if (arg == "-p" || arg == "--seqPairPath")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.seqPairPath, arg));
        }
        else if (arg == "-o" || arg == "--resPath")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.resPath, arg));
        }
        else if (arg == "--substName")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.substName, arg));
        }
        else if (arg == "--gapoCost")
        {
            ZIG_TRY(NwStat::success, setIntArgOnce(argc, argv, i, cmdArgs.gapoCost, arg));
        }
        else if (arg == "--gapeCost")
        {
            ZIG_TRY(NwStat::success, setIntArgOnce(argc, argv, i, cmdArgs.gapeCost, arg));
        }
        else if (arg == "--algName")
        {
            ZIG_TRY(NwStat::success, setStringVectArg(argc, argv, i, cmdArgs.algNames, arg));
        }
        else if (arg == "--refAlgName")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.refAlgName, arg));
        }
        else if (arg == "--warmupPerAlign")
        {
            ZIG_TRY(NwStat::success, setIntArgOnce(argc, argv, i, cmdArgs.warmupPerAlign, arg));
            if (cmdArgs.warmupPerAlign.value() < 0)
            {
                std::cerr << "error: parameter must be nonnegative integer: \"" << arg << "\"\n";
                NwStat::errorInvalidValue;
            }
        }
        else if (arg == "--samplesPerAlign")
        {
            ZIG_TRY(NwStat::success, setIntArgOnce(argc, argv, i, cmdArgs.samplesPerAlign, arg));
            if (cmdArgs.samplesPerAlign.value() <= 0)
            {
                std::cerr << "error: parameter must be positive integer: \"" << arg << "\"\n";
                NwStat::errorInvalidValue;
            }
        }
        else if (arg == "--fCalcTrace")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fCalcTrace, arg));
        }
        else if (arg == "--fCalcScoreHash")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fCalcScoreHash, arg));
        }
        else if (arg == "--fWriteProgress")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fWriteProgress, arg));
        }
        else if (arg == "--debugPath")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.debugPath, arg));
        }
        else if (arg == "--fPrintScore")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fPrintScore, arg));
        }
        else if (arg == "--fPrintTrace")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fPrintTrace, arg));
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_cmd_usage(std::cout);
            return NwStat::helpMenuRequested;
        }
        else
        {
            print_cmd_usage(std::cout);
            std::cout << '\n';
            std::cerr << "error: unknown parameter: \"" << arg << "\"\n";
            return NwStat::errorInvalidValue;
        }
    }

    // Required.
    ZIG_TRY(NwStat::success, expectNonEmptyArg(cmdArgs.algParamPath, "--algParamPath"));
    ZIG_TRY(NwStat::success, expectNonEmptyArg(cmdArgs.seqPath, "--seqPath"));
    if ((cmdArgs.fPrintScore.has_value() || cmdArgs.fPrintTrace.has_value()) && !cmdArgs.debugPath.has_value())
    {
        setDefaultIfArgEmpty(cmdArgs.debugPath, std::string("./logs/") + cmdArgs.isoDateTime + std::string("_debug.txt"));
    }

    // Defaults.
    setDefaultIfArgEmpty(cmdArgs.substPath, std::string("./resrc/subst.json"));
    // Required.
    cmdArgs.algParamPath;
    cmdArgs.seqPath;
    setDefaultIfArgEmpty(cmdArgs.seqPairPath, std::string {});
    setDefaultIfArgEmpty(cmdArgs.resPath, std::string("./logs/") + cmdArgs.isoDateTime + std::string(".tsv"));

    setDefaultIfArgEmpty(cmdArgs.substName, std::string("blosum62"));
    setDefaultIfArgEmpty(cmdArgs.gapoCost, -11);
    setDefaultIfArgEmpty(cmdArgs.gapeCost, 0);
    // Handled when the algParam file is read.
    cmdArgs.algNames;
    cmdArgs.refAlgName;
    setDefaultIfArgEmpty(cmdArgs.warmupPerAlign, 0);
    setDefaultIfArgEmpty(cmdArgs.samplesPerAlign, 1);

    setDefaultIfArgEmpty(cmdArgs.fCalcTrace, false);
    setDefaultIfArgEmpty(cmdArgs.fCalcScoreHash, false);
    setDefaultIfArgEmpty(cmdArgs.fWriteProgress, false);
    setDefaultIfArgEmpty(cmdArgs.debugPath, std::string {});
    setDefaultIfArgEmpty(cmdArgs.fPrintScore, false);
    setDefaultIfArgEmpty(cmdArgs.fPrintTrace, false);

    return NwStat::success;
}

static NwStat parseSubstFile(const std::string& substPath, NwSubstData& substData)
{
    std::string error_msg {};
    if (NwStat::success != readFromJsonFile(substPath, substData, error_msg))
    {
        std::cerr << "error: could not open/parse json from substPath: \"" << substPath << "\"\n";
        std::cerr << error_msg << '\n';
        return NwStat::errorIoStream;
    }

    int letter_cnt = (int)substData.letterMap.size();
    int letter_idx = 0;
    for (const auto& letter_tuple : substData.letterMap)
    {
        if (letter_tuple.first.size() != 1)
        {
            std::cerr << "error: substitution matrix letters must be a character: \"" << letter_tuple.first << "\"\n";
            return NwStat::errorInvalidFormat;
        }

        if (letter_tuple.second != letter_idx)
        {
            std::cerr << "error: substitution matrix letter's index must start from zero and increase by 1: \"" << letter_tuple.first << "\": " << letter_tuple.second << '\n';
            return NwStat::errorInvalidFormat;
        }

        letter_idx++;
    }

    for (const auto& subst_tuple : substData.substMap)
    {
        if (subst_tuple.second.size() != letter_cnt * letter_cnt)
        {
            std::cerr << "error: substitution matrix must have exactly letter_cnt^2 elements: \"" << subst_tuple.first << "\"\n";
            return NwStat::errorInvalidFormat;
        }
    }

    return NwStat::success;
}

static NwStat parseAlgParamsFile(const std::string& algParamPath, NwAlgParamsData& algParamsData)
{
    std::string error_msg {};
    if (NwStat::success != readFromJsonFile(algParamPath, algParamsData, error_msg))
    {
        std::cerr << "error: could not open/parse json from algParamPath: \"" << algParamPath << "\"\n";
        std::cerr << error_msg << '\n';
        return NwStat::errorIoStream;
    }

    return NwStat::success;
}

static NwStat verifyAndSetAlgNames(NwCmdArgs& cmdArgs, const NwCmdData& cmdData)
{
    Dict<std::string, NwAlgorithm> algMap {};
    getNwAlgorithmMap(algMap);

    std::vector<std::string> providedAlgNames = cmdData.algParamsData.paramMap.keys();
    for (const auto& algName : providedAlgNames)
    {
        if (algMap.find(algName) == algMap.cend())
        {
            std::cerr << "error: unknown algorithm in algParam file: \"" << algName << "\"\n";
            return NwStat::errorInvalidValue;
        }
    }

    if (cmdArgs.algNames.has_value())
    {
        std::vector<std::string> selectedAlgNames = cmdArgs.algNames.value();
        for (const auto& algName : selectedAlgNames)
        {
            if (algMap.find(algName) == algMap.cend())
            {
                std::cerr << "error: unknown algorithm on command line: \"" << algName << "\"\n";
                return NwStat::errorInvalidValue;
            }

            if (std::find(providedAlgNames.cbegin(), providedAlgNames.cend(), algName) == providedAlgNames.cend())
            {
                std::cerr << "error: selected algorithm not present in algParam file: \"" << algName << "\"\n";
                return NwStat::errorInvalidValue;
            }
        }
    }
    setDefaultIfArgEmpty(cmdArgs.algNames, providedAlgNames);

    if (cmdArgs.refAlgName.has_value())
    {
        std::string algName = cmdArgs.refAlgName.value();
        if (algMap.find(algName) == algMap.cend())
        {
            std::cerr << "error: unknown referent algorithm on command line: \"" << algName << "\"\n";
            return NwStat::errorInvalidValue;
        }

        auto selectedAlgNames = cmdArgs.algNames.value();
        if (std::find(selectedAlgNames.cbegin(), selectedAlgNames.cend(), algName) == selectedAlgNames.cend())
        {
            std::cerr << "error: selected referent algorithm not present in algParam file: \"" << algName << "\"\n";
            return NwStat::errorInvalidValue;
        }
    }
    setDefaultIfArgEmpty(cmdArgs.refAlgName, cmdArgs.algNames.value()[0]);

    return NwStat::success;
}

static NwStat parseSeqFile(const std::string& seqPath, const Dict<std::string, int>& letterMap, NwSeqData& seqData)
{
    std::ifstream ifs;
    if (NwStat stat = openInFile(seqPath, ifs); stat != NwStat::success)
    {
        std::cerr << "error: could not open fasta file from seqPath: \"" << seqPath << "\"\n";
        return NwStat::errorIoStream;
    }

    std::string error_msg {};
    if (NwStat stat = readFromFastaFormat(seqPath, ifs, seqData, letterMap, error_msg); stat != NwStat::success)
    {
        std::cerr << "error: invalid fasta format on seqPath: \"" << seqPath << "\"\n";
        std::cerr << error_msg << '\n';
        return stat;
    }

    return NwStat::success;
}

static NwStat parseSeqPairFile(const std::string& seqPairPath, NwSeqPairData& seqPairData, const Dict<std::string, NwSeq>& seqMap)
{
    std::ifstream ifs;
    if (NwStat stat = openInFile(seqPairPath, ifs); stat != NwStat::success)
    {
        std::cerr << "error: could not open text file from seqPairPath: \"" << seqPairPath << "\"\n";
        return NwStat::errorIoStream;
    }

    std::string error_msg {};
    if (NwStat stat = readFromSeqPairFormat(seqPairPath, ifs, seqPairData, seqMap, error_msg); stat != NwStat::success)
    {
        std::cerr << "error: invalid text format on seqPairPath: \"" << seqPairPath << "\"\n";
        std::cerr << error_msg << '\n';
        return stat;
    }

    return NwStat::success;
}

// By default, align second and following sequences (Y) to the first (X).
static NwStat initSeqPairData(const NwSeqData& seqData, NwSeqPairData& seqPairData)
{
    const auto& seqX_tuple = *(seqData.seqMap.cbegin());
    auto& seqX_id = seqX_tuple.first;
    auto& seqX = seqX_tuple.second;

    for (const auto& seqY_tuple : seqData.seqMap)
    {
        auto& seqY_id = seqY_tuple.first;
        auto& seqY = seqY_tuple.second;
        if (seqY_id == seqX_id)
        {
            continue;
        }

        NwSeqPair pair {};
        pair.seqY_id = seqY.id;
        pair.seqX_id = seqX.id;
        pair.seqY_range.l = 0;
        pair.seqY_range.r = seqY_tuple.second.seq.size() - 1 /*header*/;
        pair.seqX_range.l = 0;
        pair.seqX_range.r = seqX_tuple.second.seq.size() - 1 /*header*/;
        seqPairData.pairList.push_back(pair);
    }

    if (seqPairData.pairList.size() == 0)
    {
        std::cerr << "error: since seqPairPath is empty, at least two sequences are necessary for default alignment\n";
        return NwStat::errorInvalidFormat;
    }

    return NwStat::success;
}

NwStat initCmdData(NwCmdArgs& cmdArgs, NwCmdData& cmdData)
{
    ZIG_TRY(NwStat::success, parseSubstFile(cmdArgs.substPath.value(), cmdData.substData));
    ZIG_TRY(NwStat::success, parseAlgParamsFile(cmdArgs.algParamPath.value(), cmdData.algParamsData));
    ZIG_TRY(NwStat::success, verifyAndSetAlgNames(cmdArgs, cmdData));
    ZIG_TRY(NwStat::success, parseSeqFile(cmdArgs.seqPath.value(), cmdData.substData.letterMap, cmdData.seqData));

    if (!cmdArgs.seqPairPath.value().empty())
    {
        ZIG_TRY(NwStat::success, parseSeqPairFile(cmdArgs.seqPairPath.value(), cmdData.seqPairData, cmdData.seqData.seqMap));
    }
    else
    {
        initSeqPairData(cmdData.seqData, cmdData.seqPairData);
    }

    if (NwStat::success != openOutFile(cmdArgs.resPath.value(), cmdData.resOfs))
    {
        std::cerr << "error: could not open resPath: \"" << cmdArgs.resPath.value() << "\"\n";
        return NwStat::errorIoStream;
    }

    if (cmdArgs.debugPath.value() != "" && NwStat::success != openOutFile(cmdArgs.debugPath.value(), cmdData.debugOfs))
    {
        std::cerr << "error: could not open debugPath: \"" << cmdArgs.debugPath.value() << "\"\n";
        return NwStat::errorIoStream;
    }

    return NwStat::success;
}
