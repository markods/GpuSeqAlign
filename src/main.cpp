#include "defer.hpp"
#include "file_formats.hpp"
#include "fmt_guard.hpp"
#include "io.hpp"
#include "json.hpp"
#include "nwalign.hpp"
#include "print_mat.hpp"
#include "run_types.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <optional>
#include <string>

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
    res.sw_align = Stopwatch::combine(swAlignList);
    res.sw_hash = Stopwatch::combine(swHashList);
    res.sw_trace = Stopwatch::combine(swTraceList);

    return res;
}

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
            std::cerr << "error: unknown parameter: \"" << arg << "\"";
            return NwStat::errorInvalidValue;
        }
    }

    // Required.
    ZIG_TRY(NwStat::success, expectNonEmptyArg(cmdArgs.algParamPath, "--algParamPath"));
    ZIG_TRY(NwStat::success, expectNonEmptyArg(cmdArgs.seqPath, "--seqPath"));
    if (cmdArgs.fPrintScore.has_value() || cmdArgs.fPrintTrace.has_value())
    {
        ZIG_TRY(NwStat::success, expectNonEmptyArg(cmdArgs.debugPath, "--debugPath"));
    }

    // Defaults.
    setDefaultIfArgEmpty(cmdArgs.substPath, std::string("./resrc/subst.json"));
    cmdArgs.algParamPath;
    cmdArgs.seqPath;
    setDefaultIfArgEmpty(cmdArgs.pairPath, std::string {});
    setDefaultIfArgEmpty(cmdArgs.resPath, std::string("./logs/") + isoDatetimeAsString() + std::string(".tsv"));

    setDefaultIfArgEmpty(cmdArgs.substName, std::string("blosum62"));
    setDefaultIfArgEmpty(cmdArgs.gapoCost, 11);
    setDefaultIfArgEmpty(cmdArgs.gapeCost, 0);
    // TODO
    // Handled when the algParam file is read.
    cmdArgs.algName;
    cmdArgs.refAlgName;
    setDefaultIfArgEmpty(cmdArgs.warmupPerAlign, 0);
    setDefaultIfArgEmpty(cmdArgs.samplesPerAlign, 1);

    setDefaultIfArgEmpty(cmdArgs.fCalcTrace, false);
    setDefaultIfArgEmpty(cmdArgs.fCalcScoreHash, false);
    setDefaultIfArgEmpty(cmdArgs.fWriteProgress, false);
    setDefaultIfArgEmpty(cmdArgs.debugPath, std::string {}); // TODO: required ako ima fPrintScore ili fPrintTrace
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
        ////// sparse representation of the score matrix
        // tileHrowMat_gpu;   <-- algorithm-reserved
        // tileHcolMat_gpu;   <-- algorithm-reserved

        ////// alignment parameters
        // substsz;   <-- once
        // adjrows;   <-- loop-inited
        // adjcols;   <-- loop-inited
        // indel;   <-- once
        ////// sparse representation of the score matrix
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

    auto defer1 = make_defer([&]() noexcept
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
    writeResultHeaderToTsv(ofsRes, cmdArgs.fCalcScoreHash.value(), cmdArgs.fCalcTrace.value());
    if (cmdArgs.fWriteProgress.value())
    {
        // When we write to progress immediately, then write to file immediately.
        ofsRes.flush();
    }

    // for all algorithms which have parameters in the param map
    for (auto& paramTuple : paramData.paramMap)
    {
        // if the current algorithm doesn't exist, skip it
        const std::string& algName = paramTuple.first;
        if (algMap.find(algName) == algMap.end())
        {
            continue;
        }

        if (cmdArgs.fWriteProgress.value())
        {
            std::cout << algName << ":";
        }

        // get the current algorithm and initialize its parameters
        NwAlgorithm& alg = algMap[algName];
        alg.init(paramTuple.second /*algParams*/);

        // for all Y sequences + for all X sequences (also compare every sequence with itself)
        for (int iY = 0; iY < seqList.size(); iY++)
        {
            if (cmdArgs.fWriteProgress.value())
            {
                std::cout << "\n"
                          << std::flush
                          << "|";
            }

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
                        auto defer2 = make_defer([&]() noexcept
                        {
                            nw.resetAllocsBenchmarkCycle();
                        });

                        resList.push_back(NwAlgResult {});
                        NwAlgResult& res = resList.back();

                        res.algName = algName;
                        res.algParams = alg.alignPr().copy();
                        //
                        res.iX = iX;
                        res.iY = iY;
                        res.reps = seqData.repeat;
                        //
                        res.seqX_len = nw.seqX.size();
                        res.seqY_len = nw.seqY.size();

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

                        if (cmdArgs.fWriteProgress.value())
                        {
                            // if the result is successful, print a dot, otherwise an x
                            if (res.stat == NwStat::success)
                            {
                                std::cout << '.' << std::flush;
                            }
                            else
                            {
                                std::cout << res.errstep << std::flush;
                            }
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
                    writeResultLineToTsv(ofsRes, res, cmdArgs.fCalcScoreHash.value(), cmdArgs.fCalcTrace.value());
                    if (cmdArgs.fWriteProgress.value())
                    {
                        // When we write to progress immediately, then write to file immediately.
                        ofsRes.flush();
                    }
                }

                // reset the algorithm parameters
                alg.alignPr().reset();

                if (cmdArgs.fWriteProgress.value())
                {
                    // seqX-seqY comparison separator
                    std::cout << '|' << std::flush;
                }
            }
        }

        if (cmdArgs.fWriteProgress.value())
        {
            // algorithm separator
            std::cout << "\n\n"
                      << std::flush;
        }
    }

    // print the number of calculation errors
    if (compareData.calcErrors > 0)
    {
        std::cerr << "error: " << compareData.calcErrors << " calculation error(s)";
        return -1;
    }
}
