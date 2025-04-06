#include "defer.hpp"
#include "fmt_guard.hpp"
#include "json.hpp"
#include "nwalign.hpp"
#include "print_mat.hpp"
#include "run_types.hpp"
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>

// get the current time as an ISO string
std::string isoDatetimeAsString()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::tm tm_struct {};
    if (localtime_s(&tm_struct, &time) != 0)
    {
        throw std::runtime_error("Failed to get local time.");
    }

    std::stringstream strs;
    strs << std::put_time(&tm_struct, "%Y%m%d_%H%M%S");
    return strs.str();
}

// open output file stream
NwStat openOutFile(const std::string& path, std::ofstream& ofs)
{
    try
    {
        // Create directories if they don't exist
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());

        ofs.open(path, std::ios_base::out);
        ofs.exceptions(std::ios_base::goodbit);
        if (!ofs)
        {
            return NwStat::errorIoStream;
        }
    }
    catch (const std::exception&)
    {
        return NwStat::errorIoStream;
    }

    return NwStat::success;
}

// read a json file into a variable
template <typename T>
NwStat readFromJsonFile(const std::string& path, T& res)
{
    std::ifstream ifs;

    ifs.open(path, std::ios_base::in);
    ifs.exceptions(std::ios_base::goodbit);
    if (!ifs)
    {
        return NwStat::errorIoStream;
    }

    auto defer1 = make_defer([&]() noexcept
    {
        ifs.close();
    });

    // NOTE: the parser doesn't allow for trailing commas
    auto json = nlohmann::ordered_json::parse(
        ifs,
        /*callback*/ nullptr,
        /*allow_exceptions*/ false,
        /*ignore_comments*/ true);

    if (json.is_discarded())
    {
        return NwStat::errorInvalidFormat;
    }

    try
    {
        res = json;
    }
    catch (const std::exception&)
    {
        return NwStat::errorInvalidFormat;
    }

    return NwStat::success;
}

// the Needleman-Wunsch algorithm implementations
class NwAlgorithm
{
public:
    using NwAlignFn = NwStat (*)(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
    using NwTraceFn = NwStat (*)(const NwAlgInput& nw, NwAlgResult& res);
    using NwHashFn = NwStat (*)(const NwAlgInput& nw, NwAlgResult& res);
    using NwPrintFn = NwStat (*)(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);

public:
    NwAlgorithm();

    NwAlgorithm(
        NwAlignFn alignFn,
        NwTraceFn traceFn,
        NwHashFn hashFn,
        NwPrintFn printFn);

    void init(NwAlgParams& alignPr);

    NwAlgParams& alignPr();

    NwStat align(NwAlgInput& nw, NwAlgResult& res);
    NwStat trace(const NwAlgInput& nw, NwAlgResult& res);
    NwStat hash(const NwAlgInput& nw, NwAlgResult& res);
    NwStat print(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);

private:
    NwAlignFn _alignFn;
    NwTraceFn _traceFn;
    NwHashFn _hashFn;
    NwPrintFn _printFn;

    NwAlgParams _alignPr;
};

NwAlgorithm::NwAlgorithm()
{
    _alignFn = {};
    _traceFn = {};
    _hashFn = {};
    _printFn = {};

    _alignPr = {};
}

NwAlgorithm::NwAlgorithm(
    NwAlgorithm::NwAlignFn alignFn,
    NwAlgorithm::NwTraceFn traceFn,
    NwAlgorithm::NwHashFn hashFn,
    NwAlgorithm::NwPrintFn printFn)
{
    _alignFn = alignFn;
    _traceFn = traceFn;
    _hashFn = hashFn;
    _printFn = printFn;

    _alignPr = {};
}

void NwAlgorithm::init(NwAlgParams& alignPr)
{
    _alignPr = alignPr;
}

NwAlgParams& NwAlgorithm::alignPr()
{
    return _alignPr;
}

NwStat NwAlgorithm::align(NwAlgInput& nw, NwAlgResult& res)
{
    return _alignFn(_alignPr, nw, res);
}
NwStat NwAlgorithm::trace(const NwAlgInput& nw, NwAlgResult& res)
{
    return _traceFn(nw, res);
}
NwStat NwAlgorithm::hash(const NwAlgInput& nw, NwAlgResult& res)
{
    return _hashFn(nw, res);
}
NwStat NwAlgorithm::print(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res)
{
    return _printFn(os, nw, res);
}

// input file formats
struct NwSubstData
{
    std::map<std::string, int> letterMap;
    std::map<std::string, std::vector<int>> substMap;
};

struct NwParamData
{
    std::map<std::string, NwAlgParams> paramMap;
};

struct NwSeqData
{
    std::string substName;
    int indel = 0;
    // repeat each comparison this many times
    int repeat = 0;
    // each sequence will be an int vector and have a header (zeroth) element
    std::vector<std::string> seqList;
};

// conversion to object from json
void from_json(const nlohmann::ordered_json& j, NwSubstData& substData)
{
    j.at("letterMap").get_to(substData.letterMap);
    j.at("substMap").get_to(substData.substMap);
}
void from_json(const nlohmann::ordered_json& j, NwAlgParam& param)
{
    j.get_to(param._values);
}
void from_json(const nlohmann::ordered_json& j, NwAlgParams& params)
{
    j.get_to(params._params);
}
void from_json(const nlohmann::ordered_json& j, NwParamData& paramData)
{
    j.get_to(paramData.paramMap);
}
void from_json(const nlohmann::ordered_json& j, NwSeqData& seqData)
{
    j.at("substName").get_to(seqData.substName);
    j.at("indel").get_to(seqData.indel);
    j.at("repeat").get_to(seqData.repeat);
    j.at("seqList").get_to(seqData.seqList);
}

// conversion to json from object
void to_json(nlohmann::ordered_json& j, const NwSubstData& substData)
{
    j["letterMap"] = substData.letterMap;
    j["substMap"] = substData.substMap;
}
void to_json(nlohmann::ordered_json& j, const NwAlgParam& param)
{
    j = param._values;
}
void to_json(nlohmann::ordered_json& j, const NwAlgParams& params)
{
    j = params._params;
}
void to_json(nlohmann::ordered_json& j, const NwParamData& paramData)
{
    j["paramMap"] = paramData.paramMap;
}
void to_json(nlohmann::ordered_json& j, const NwSeqData& seqData)
{
    j["substName"] = seqData.substName;
    j["indel"] = seqData.indel;
    j["repeat"] = seqData.repeat;
    j["seqList"] = seqData.seqList;
}

static void lapTimeToTsv(std::ostream& os, float lapTime)
{
    FormatFlagsGuard fg {os};

    os << std::fixed << std::setprecision(3) << lapTime;
}

// conversion to tsv from object
void writeResultHeaderToTsv(std::ostream& os)
{
    FormatFlagsGuard fg {os};
    os.fill(' ');

    os << "algName" << "\t";
    os << "iY" << "\t";
    os << "iX" << "\t";
    os << "reps" << "\t";

    os << "lenY" << "\t";
    os << "lenX" << "\t";

    os << "algParams" << "\t";

    os << "step" << "\t";
    os << "stat" << "\t";
    os << "cuda" << "\t";

    os << "score_hash" << "\t";
    os << "trace_hash" << "\t";

    os << "align.alloc" << "\t";
    os << "align.cpy_dev" << "\t";
    os << "align.init_hdr" << "\t";
    os << "align.calc_init" << "\t";
    os << "align.calc" << "\t";
    os << "align.cpy_host" << "\t";

    os << "trace.alloc" << "\t";
    os << "trace.calc" << "\t";

    os << '\n';
}
void writeResultLineToTsv(std::ostream& os, const NwAlgResult& res)
{
    FormatFlagsGuard fg {os};
    {
        os.fill(' ');

        os << res.algName << "\t";
        os << res.iY << "\t";
        os << res.iX << "\t";
        os << res.reps << "\t";

        os << res.seqY_len << "\t";
        os << res.seqX_len << "\t";

        nlohmann::ordered_json algParamsJson = res.algParams;

        os << algParamsJson.dump() << "\t";

        os << res.errstep << "\t";
        os << int(res.stat) << "\t";
        os << int(res.cudaStat) << "\t";

        // TODO: print align score

        os.fill('0');
        os << std::setw(10) << res.score_hash << "\t";
        os << std::setw(10) << res.trace_hash << "\t";
    }
    fg.restore();

    lapTimeToTsv(os, res.sw_align.get_or_default("align.alloc"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.cpy_dev"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.init_hdr"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.calc_init"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.calc"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("align.cpy_host"));
    os << "\t";

    // TODO: only print if requested
    lapTimeToTsv(os, res.sw_hash.get_or_default("hash.calc"));
    os << "\t";

    lapTimeToTsv(os, res.sw_trace.get_or_default("trace.alloc"));
    os << "\t";
    lapTimeToTsv(os, res.sw_trace.get_or_default("trace.calc"));
    os << "\t";
}

// convert the sequence string to a vector using a character map
// + NOTE: add the header (zeroth) element if requested
std::vector<int> seqStrToVect(const std::string& str, const std::map<std::string, int>& map, const bool addHeader)
{
    // preallocate the requred amount of elements
    std::vector<int> vect {};

    // initialize the zeroth element if requested
    if (addHeader)
    {
        vect.push_back(0);
    }

    // for all characters of the string
    for (char c : str)
    {
        // add them to the vector
        std::string cs {c};
        int val = map.at(cs);
        vect.push_back(val);
    }

    return vect;
}

// algorithm map
// structs used to verify that the algorithms' results are correct
struct NwCompareKey
{
    int iY;
    int iX;

    friend bool operator<(const NwCompareKey& l, const NwCompareKey& r);
};
struct NwCompareRes
{
    unsigned score_hash;
    unsigned trace_hash;

    friend bool operator==(const NwCompareRes& l, const NwCompareRes& r);
    friend bool operator!=(const NwCompareRes& l, const NwCompareRes& r);
};
struct NwCompareData
{
    std::map<NwCompareKey, NwCompareRes> compareMap;
    int calcErrors;
};

// structs used to verify that the algorithms' results are correct
bool operator<(const NwCompareKey& l, const NwCompareKey& r)
{
    bool res =
        (l.iY < r.iY) ||
        (l.iY == r.iY && l.iX < r.iX);
    return res;
}

bool operator==(const NwCompareRes& l, const NwCompareRes& r)
{
    bool res =
        l.score_hash == r.score_hash &&
        l.trace_hash == r.trace_hash;
    return res;
}
bool operator!=(const NwCompareRes& l, const NwCompareRes& r)
{
    bool res =
        l.score_hash != r.score_hash ||
        l.trace_hash != r.trace_hash;
    return res;
}

// check that the result hashes match the hashes calculated by the first algorithm (the gold standard)
NwStat setOrVerifyResult(const NwAlgResult& res, NwCompareData& compareData)
{
    std::map<NwCompareKey, NwCompareRes>& compareMap = compareData.compareMap;
    NwCompareKey key {
        res.iY, // iY;
        res.iX  // iX;
    };
    NwCompareRes calcVal {
        res.score_hash, // score_hash;
        res.trace_hash  // trace_hash;
    };

    // if this is the first time the two sequences have been aligned
    auto compareRes = compareMap.find(key);
    if (compareRes == compareMap.end())
    {
        // add the calculated (gold) values to the map
        compareMap[key] = calcVal;
        return NwStat::success;
    }

    // if the calculated value is not the same as the expected value
    NwCompareRes& expVal = compareMap[key];
    if (calcVal != expVal)
    {
        // the current algorithm probably made a mistake during calculation
        return NwStat::errorInvalidResult;
    }

    // the current and gold algoritm agree on the results
    return NwStat::success;
}

// combine results from many repetitions into one
NwAlgResult combineResults(std::vector<NwAlgResult>& resList)
{
    // if the result list is empty, return a default initialized result
    if (resList.empty())
    {
        return NwAlgResult {};
    }

    // get the stopwatches from multiple repeats as lists
    std::vector<Stopwatch> swAlignList {};
    std::vector<Stopwatch> swHashList {};
    std::vector<Stopwatch> swTraceList {};
    for (auto& curr : resList)
    {
        swAlignList.push_back(curr.sw_align);
        swHashList.push_back(curr.sw_hash);
        swTraceList.push_back(curr.sw_trace);
    }

    // copy on purpose here -- don't modify the given result list
    // +   take the last result since it might have an error (if it errored it is definitely the last result)
    NwAlgResult res = resList[resList.size() - 1];
    // combine the stopwatches from many repeats into one
    res.sw_align = Stopwatch::combineStopwatches(swAlignList);
    res.sw_hash = Stopwatch::combineStopwatches(swHashList);
    res.sw_trace = Stopwatch::combineStopwatches(swTraceList);

    return res;
}

void print_cmd_usage(std::ostream& os)
{
    os << "nw --algParamPath \"path\" --seqPath \"path\" [params]\n"
          "\n"
          "Parameters:\n"
          "-b, --substPath <path>     Path of JSON substitution matrices file, defaults to \"./resrc/subst.json\".\n"
          "-r, --algParamPath <path>  Path of JSON algorithm parameters file.\n"
          "-s, --seqPath <path>       Path of FASTA file with sequences to be aligned.\n"
          "-p, --pairPath <path>      Path of TXT file with sequence pairs to be aligned. Each line is in the format\n"
          "                           \"seqA[0:42] seqB\", where \"seqA\" and \"seqB\" are sequence ids, and \"[a:b]\" specifies the\n"
          "                           substring starting from element \"a\" (inclusive) until element \"b\" (exclusive).\n"
          "                           It's possible to omit the start/end of the interval, like so: \"[a:b]\", \"[a:]\", \"[:b]\".\n"
          "                           If the TXT file is not specified, then all sequences in the FASTA file except the first\n"
          "                           are aligned to the first sequence. If there is only one sequence in the FASTA file,\n"
          "                           it's aligned with itself.\n"
          "-o, --resPath <path>       Path of JSON test bench results file, defaults to \"./logs/<datetime>.json\".\n"
          "\n"
          "--substName <name>         Specify which substitution matrix from the \"subst\" file will be used. Defaults to\n"
          "                           \"blosum62\".\n"
          "--gapoCost <cost>          Gap open cost. Nonnegative integer, defaults to 11.\n"
          "--gapeCost <cost>          Unused. Gap extend cost. Nonnegative integer, defaults to 0.\n"
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
          "                           written to, once per alignment. Defaults to \"\".\n"
          "--fPrintScore              Should the score matrix be printed. Defaults to false.\n"
          "--fPrintTrace              Should the trace be printed. Defaults to false.\n"
          "\n"
          "-h, --help                 Print help and exit.\n";
}

struct NwCmdArgs
{
    std::optional<std::string> substPath;
    std::optional<std::string> algParamPath;
    std::optional<std::string> seqPath;
    std::optional<std::string> pairPath;
    std::optional<std::string> resPath;

    std::optional<std::string> substName;
    std::optional<int> gapoCost;
    std::optional<int> gapeCost;
    std::optional<std::vector<std::string>> algName;
    std::optional<std::string> refAlgName;
    std::optional<int> warmupPerAlign;
    std::optional<int> samplesPerAlign;

    std::optional<bool> fCalcTrace;
    std::optional<bool> fCalcScoreHash;
    std::optional<bool> fWriteProgress;
    std::optional<std::string> debugPath;
    std::optional<bool> fPrintScore;
    std::optional<bool> fPrintTrace;
};

NwStat setStringArgOnce(
    const int argc,
    const char* argv[],
    int& i,
    std::optional<std::string>& arg,
    const std::string& arg_name /*must not be a rvalue*/)
{
    if (arg.has_value())
    {
        std::cerr << "error: parameter already set: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }
    if (i >= argc)
    {
        std::cerr << "error: expected parameter value: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }

    arg.emplace(argv[++i]);
    return NwStat::success;
}

NwStat setStringVectArg(
    const int argc,
    const char* argv[],
    int& i,
    std::optional<std::vector<std::string>>& arg,
    const std::string& arg_name /*must not be a rvalue*/)
{
    if (i >= argc)
    {
        std::cerr << "error: expected parameter value: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }

    if (!arg.has_value())
    {
        arg.emplace(std::vector<std::string> {});
    }

    arg.value().push_back(argv[++i]);
    return NwStat::success;
}

NwStat setIntArgOnce(
    const int argc,
    const char* argv[],
    int& i,
    std::optional<int>& arg,
    const std::string& arg_name /*must not be a rvalue*/)
{
    if (arg.has_value())
    {
        std::cerr << "error: parameter already set: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }
    if (i >= argc)
    {
        std::cerr << "error: expected parameter value: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }

    try
    {
        arg.emplace(std::stoi(argv[++i]));
    }
    catch (const std::invalid_argument&)
    {
        std::cerr << "error: parameter value should be int: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }
    catch (const std::out_of_range&)
    {
        std::cerr << "error: parameter value is out-of-range for int: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }

    return NwStat::success;
}

NwStat setSwitchArgOnce(
    std::optional<bool>& arg,
    const std::string& arg_name /*must not be a rvalue*/)
{
    if (arg.has_value())
    {
        std::cerr << "error: parameter already set: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }

    arg.emplace(true);
    return NwStat::success;
}

template <typename T>
NwStat expectNonEmptyArg(std::optional<T>& arg, const std::string& arg_name)
{
    if (!arg.has_value())
    {
        std::cerr << "error: expected parameter: \"" << arg_name << "\"";
        return NwStat::errorInvalidValue;
    }
    return NwStat::success;
}

template <typename T>
void setDefaultIfArgEmpty(std::optional<T>& arg, const T& value)
{
    if (!arg.has_value())
    {
        arg.emplace(value);
    }
}

NwStat parseCmdArgs(const int argc, const char* argv[], NwCmdArgs& cmdArgs)
{
    if (argc == 1)
    {
        print_cmd_usage(std::cout);
        std::cerr << "error: expected command parameters";
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
        else if (arg == "-p" || arg == "--pairPath")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.pairPath, arg));
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
            ZIG_TRY(NwStat::success, setStringVectArg(argc, argv, i, cmdArgs.algName, arg));
        }
        else if (arg == "--refAlgName")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.refAlgName, arg));
        }
        else if (arg == "--warmupPerAlign")
        {
            ZIG_TRY(NwStat::success, setIntArgOnce(argc, argv, i, cmdArgs.warmupPerAlign, arg));
        }
        else if (arg == "--samplesPerAlign")
        {
            ZIG_TRY(NwStat::success, setIntArgOnce(argc, argv, i, cmdArgs.samplesPerAlign, arg));
        }
        else if (arg == "--calcTrace")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fCalcTrace, arg));
        }
        else if (arg == "--calcScoreHash")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fCalcScoreHash, arg));
        }
        else if (arg == "--writeProgress")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fWriteProgress, arg));
        }
        else if (arg == "--debugPath")
        {
            ZIG_TRY(NwStat::success, setStringArgOnce(argc, argv, i, cmdArgs.debugPath, arg));
        }
        else if (arg == "--printScore")
        {
            ZIG_TRY(NwStat::success, setSwitchArgOnce(cmdArgs.fPrintScore, arg));
        }
        else if (arg == "--printTrace")
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
            std::cerr << "error: unknown parameter: \"" << arg << "\"";
            return NwStat::errorInvalidValue;
        }
    }

    ZIG_TRY(NwStat::success, expectNonEmptyArg(cmdArgs.algParamPath, "--algParamPath"));
    ZIG_TRY(NwStat::success, expectNonEmptyArg(cmdArgs.seqPath, "--seqPath"));

    setDefaultIfArgEmpty(cmdArgs.substPath, std::string("./resrc/subst.json"));
    setDefaultIfArgEmpty(cmdArgs.resPath, std::string("./logs/") + isoDatetimeAsString() + std::string(".tsv"));

    setDefaultIfArgEmpty(cmdArgs.substName, std::string("blosum62"));
    setDefaultIfArgEmpty(cmdArgs.gapoCost, 11);
    setDefaultIfArgEmpty(cmdArgs.gapeCost, 0);
    cmdArgs.refAlgName; // TODO
    setDefaultIfArgEmpty(cmdArgs.warmupPerAlign, 0);
    setDefaultIfArgEmpty(cmdArgs.samplesPerAlign, 0);

    setDefaultIfArgEmpty(cmdArgs.fCalcTrace, false);
    setDefaultIfArgEmpty(cmdArgs.fCalcScoreHash, false);
    setDefaultIfArgEmpty(cmdArgs.fWriteProgress, false);
    cmdArgs.debugPath = std::nullopt;
    setDefaultIfArgEmpty(cmdArgs.fPrintScore, false);
    setDefaultIfArgEmpty(cmdArgs.fPrintTrace, false);

    return NwStat::success;
}

int main(const int argc, const char* argv[])
{
    NwCmdArgs cmdArgs {};
    if (NwStat stat = parseCmdArgs(argc, argv, cmdArgs); stat != NwStat::success)
    {
        if (stat == NwStat::helpMenuRequested)
        {
            return 0;
        }
        return -1;
    }

    std::ofstream ofsRes {};
    cudaError_t cudaStatus {cudaSuccess};

    std::vector<NwAlgResult> resultList {};
    std::map<std::string, NwAlgorithm> algMap {
        /*algMap:*/ {
                     {"NwAlign_Cpu1_St_Row", {NwAlign_Cpu1_St_Row, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Cpu2_St_Diag", {NwAlign_Cpu2_St_Diag, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Cpu3_St_DiagRow", {NwAlign_Cpu3_St_DiagRow, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Cpu4_Mt_DiagRow", {NwAlign_Cpu4_Mt_DiagRow, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Gpu1_Ml_Diag", {NwAlign_Gpu1_Ml_Diag, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Gpu2_Ml_DiagRow2Pass", {NwAlign_Gpu2_Ml_DiagRow2Pass, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Gpu3_Ml_DiagDiag", {NwAlign_Gpu3_Ml_DiagDiag, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Gpu4_Ml_DiagDiag2Pass", {NwAlign_Gpu4_Ml_DiagDiag2Pass, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Gpu5_Coop_DiagDiag", {NwAlign_Gpu5_Coop_DiagDiag, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Gpu6_Coop_DiagDiag2Pass", {NwAlign_Gpu6_Coop_DiagDiag2Pass, NwTrace1_Plain, NwHash1_Plain, NwPrint1_Plain}},
                     {"NwAlign_Gpu7_Mlsp_DiagDiag", {NwAlign_Gpu7_Mlsp_DiagDiag, NwTrace2_Sparse, NwHash2_Sparse, NwPrint2_Sparse}},
                     {"NwAlign_Gpu8_Mlsp_DiagDiag", {NwAlign_Gpu8_Mlsp_DiagDiag, NwTrace2_Sparse, NwHash2_Sparse, NwPrint2_Sparse}},
                     {"NwAlign_Gpu9_Mlsp_DiagDiagDiag", {NwAlign_Gpu9_Mlsp_DiagDiagDiag, NwTrace2_Sparse, NwHash2_Sparse, NwPrint2_Sparse}},
                     },
    };

    NwSubstData substData {};
    NwParamData paramData {};
    NwSeqData seqData {};

    if (NwStat::success != readFromJsonFile(cmdArgs.substPath.value(), substData))
    {
        std::cerr << "error: could not open/read json from substs file";
        return -1;
    }
    if (NwStat::success != readFromJsonFile(cmdArgs.algParamPath.value(), paramData))
    {
        std::cerr << "error: could not open/read json from params file";
        return -1;
    }
    if (NwStat::success != readFromJsonFile(cmdArgs.seqPath.value(), seqData))
    {
        std::cerr << "error: could not open/read json from seqs file";
        return -1;
    }

    if (NwStat::success != openOutFile(cmdArgs.resPath.value(), ofsRes))
    {
        std::cerr << "error: could not open tsv results file";
        return -1;
    }
    auto defer1 = make_defer([&]() noexcept
    {
        ofsRes.close();
    });

    // get the device properties
    cudaDeviceProp deviceProps {};
    if (cudaSuccess != (cudaStatus = cudaGetDeviceProperties(&deviceProps, 0 /*deviceId*/)))
    {
        std::cerr << "error: could not get device properties";
        return -1;
    }

    // number of streaming multiprocessors (sm-s) and threads in a warp
    const int sm_count = deviceProps.multiProcessorCount;
    const int warpsz = deviceProps.warpSize;
    const int maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;

    NwAlgInput nw {
        ////// host specific memory
        // subst;   <-- once
        // seqX;    <-- loop-inited
        // seqY;    <-- loop-inited
        // score;   <-- algorithm-reserved

        ////// device specific memory
        // subst_gpu;   <-- once
        // seqX_gpu;    <-- algorithm-reserved
        // seqY_gpu;    <-- algorithm-reserved
        // score_gpu;   <-- algorithm-reserved
        // // sparse representation of the score matrix
        // tileHrowMat_gpu;   <-- algorithm-reserved
        // tileHcolMat_gpu;   <-- algorithm-reserved

        ////// alignment parameters
        // substsz;   <-- once
        // adjrows;   <-- loop-inited
        // adjcols;   <-- loop-inited
        // indel;   <-- once
        // // sparse representation of the score matrix
        // tileHdrMatRows;   <-- algorithm-reserved
        // tileHdrMatCols;   <-- algorithm-reserved
        // tileHrowLen;   <-- algorithm-reserved
        // tileHcolLen;   <-- algorithm-reserved

        ////// device parameters
        // sm_count;
        // warpsz;
        // maxThreadsPerBlock;
    };

    // initialize the device parameters
    nw.sm_count = sm_count;
    nw.warpsz = warpsz;
    nw.maxThreadsPerBlock = maxThreadsPerBlock;

    auto defer3 = make_defer([&]() noexcept
    {
        nw.resetAllocsBenchmarkEnd();
    });

    // initialize the substitution matrix on the cpu and gpu
    {
        nw.subst = substData.substMap[seqData.substName];
        nw.substsz = (int)std::sqrt(nw.subst.size());

        // reserve space in the gpu global memory
        try
        {
            nw.subst_gpu.init(nw.substsz * nw.substsz);
        }
        catch (const std::exception&)
        {
            std::cerr << "error: could not reserve space for the substitution matrix in the gpu";
            return -1;
        }

        // transfer the substitution matrix to the gpu global memory
        if (cudaSuccess != (cudaStatus = memTransfer(nw.subst_gpu, nw.subst, nw.substsz * nw.substsz)))
        {
            std::cerr << "error: could not transfer substitution matrix to the gpu";
            return -1;
        }
    }

    // initialize the indel cost
    nw.indel = seqData.indel;
    // initialize the letter map
    std::map<std::string, int> letterMap = substData.letterMap;

    // initialize the sequence map
    std::vector<std::vector<int>> seqList {};
    for (auto& charSeq : seqData.seqList)
    {
        auto seq = seqStrToVect(charSeq, letterMap, true /*addHeader*/);
        seqList.push_back(seq);
    }

    // initialize the gold result map (as calculated by the first algorithm)
    NwCompareData compareData {};

    // write the tsv file's header
    writeResultHeaderToTsv(ofsRes);
    ofsRes.flush();

    // for all algorithms which have parameters in the param map
    for (auto& paramTuple : paramData.paramMap)
    {
        // if the current algorithm doesn't exist, skip it
        const std::string& algName = paramTuple.first;
        if (algMap.find(algName) == algMap.end())
        {
            continue;
        }

        std::cout << algName << ":";

        // get the current algorithm and initialize its parameters
        NwAlgorithm& alg = algMap[algName];
        alg.init(paramTuple.second /*algParams*/);

        // for all Y sequences + for all X sequences (also compare every sequence with itself)
        for (int iY = 0; iY < seqList.size(); iY++)
        {
            std::cout << std::endl
                      << "|";

            for (int iX = iY; iX < seqList.size(); iX++)
            {
                // get the Y sequence
                // NOTE: the padding (zeroth element) was already added to the sequence
                nw.seqY = seqList[iY];
                nw.adjrows = (int)nw.seqY.size();

                // get the X sequence
                // NOTE: the padding (zeroth element) was already added to the sequence
                nw.seqX = seqList[iX];
                nw.adjcols = (int)nw.seqX.size();

                // if the number of columns is less than the number of rows, swap them and the sequences
                if (nw.adjcols < nw.adjrows)
                {
                    std::swap(nw.adjcols, nw.adjrows);
                    std::swap(nw.seqX, nw.seqY);
                }

                // for all parameter combinations
                for (; alg.alignPr().hasCurr(); alg.alignPr().next())
                {
                    // results from multiple repetitions
                    std::vector<NwAlgResult> resList {};

                    // for all requested repeats
                    for (int iR = 0; iR < seqData.repeat; iR++)
                    {
                        // initialize the result in the result list
                        resList.push_back(NwAlgResult {
                            algName,              // algName;
                            alg.alignPr().copy(), // algParams;

                            nw.seqX.size(), // seqX_len;
                            nw.seqY.size(), // seqY_len;

                            iX,             // iX;
                            iY,             // iY;
                            seqData.repeat, // reps;

                            {}, // sw_align;
                            {}, // sw_hash;
                            {}, // sw_trace;

                            {}, // score_hash;
                            {}, // trace_hash;

                            {}, // errstep;   // 0 for success
                            {}, // stat;      // 0 for success
                            {}, // cudaerr;   // 0 for success
                        });
                        // get the result from the list
                        NwAlgResult& res = resList.back();

                        auto defer4 = make_defer([&]() noexcept
                        {
                            nw.resetAllocsBenchmarkCycle();
                        });

                        // compare the sequences, hash and trace the score matrices, and verify the soundness of the results
                        if (!res.errstep && NwStat::success != (res.stat = alg.align(nw, res)))
                        {
                            if (res.stat == NwStat::errorInvalidValue)
                            {
                                res.errstep = 1;
                            }
                            else
                            {
                                res.errstep = 2;
                            }
                        }
                        if (!res.errstep && NwStat::success != (res.stat = alg.hash(nw, res)))
                        {
                            res.errstep = 3;
                        }
                        if (!res.errstep && NwStat::success != (res.stat = alg.trace(nw, res)))
                        {
                            res.errstep = 4;
                        }
                        if (!res.errstep && NwStat::success != (res.stat = setOrVerifyResult(res, compareData)))
                        {
                            res.errstep = 5;
                            compareData.calcErrors++;
                        }

                        // if the result is successful, print a dot, otherwise an x
                        if (res.stat == NwStat::success)
                        {
                            std::cout << '.' << std::flush;
                        }
                        else
                        {
                            std::cout << res.errstep << std::flush;
                        }

                        // clear cuda non-sticky errors and get possible cuda sticky errors
                        // note: repeat twice, since sticky errors cannot be cleared
                        cudaStatus = cudaGetLastError();
                        cudaStatus = cudaGetLastError();
                        if (cudaStatus != cudaSuccess)
                        {
                            std::cerr << "error: corrupted cuda context";
                            return -1;
                        }
                    }

                    // add the result to the results list
                    resultList.push_back(combineResults(resList));
                    NwAlgResult& res = resultList.back();
                    // reset the multiple repetition list
                    resList.clear();

                    // print the result as a tsv line to the tsv output file
                    writeResultLineToTsv(ofsRes, res);
                    ofsRes << '\n';
                    ofsRes.flush();
                }

                // reset the algorithm parameters
                alg.alignPr().reset();

                // seqX-seqY comparison separator
                std::cout << '|' << std::flush;
            }
        }

        // algorithm separator
        std::cout << std::endl
                  << std::endl;
    }

    // print the number of calculation errors
    if (compareData.calcErrors > 0)
    {
        std::cerr << "error: " << compareData.calcErrors << " calculation error(s)";
        return -1;
    }
}
