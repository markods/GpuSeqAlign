#include "common.hpp"
#include "json.hpp"
#include "algorithm.hpp"
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

void print_cmd_usage(std::ostream& os)
{
    os << "nw --algParamPath \"path\" --seqPath \"path\" [params]\n"
          "\n"
          "Parameters:\n"
          "-b, --substPath            Path of JSON substitution matrices file, defaults to \"./resrc/subst.json\".\n"
          "-r, --algParamPath         Path of JSON algorithm parameters file.\n"
          "-s, --seqPath              Path of FASTA file with sequences to be aligned.\n"
          "-p, --pairPath             Path of TXT file with sequence pairs to be aligned. Each line is in the format\n"
          "                           \"seqA[0:42] seqB\", where \"seqA\" and \"seqB\" are sequence ids, and \"[a:b]\" specifies the\n"
          "                           substring starting from element \"a\" (inclusive) until element \"b\" (exclusive).\n"
          "                           It's possible to omit the start/end of the interval, like so: \"[a:b]\", \"[a:]\", \"[:b]\".\n"
          "                           If the TXT file is not specified, then all sequences in the FASTA file except the first\n"
          "                           are aligned to the first sequence. If there is only one sequence in the FASTA file,\n"
          "                           it's aligned with itself.\n"
          "-o, --resPath              Path of JSON test bench results file, defaults to \"./logs/<datetime>.json\".\n"
          "\n"
          "--substName                Specify which substitution matrix from the \"subst\" file will be used. Defaults to\n"
          "                           \"blosum62\".\n"
          "--gapoCost                 Gap open cost, defaults to 11.\n"
          "--gapeCost                 Unused, defaults to 0.\n"
          "--algName                  Specify which algorithm from the \"algParam\" JSON file will be used. Can be specified\n"
          "                           multiple times, in which case those algorithms will be used, in that order.\n"
          "                           If not specified, all algorithms in the \"algParam\" JSON file are used, in that order.\n"
          "--refAlgName               Specify the algorithm name which should be considered as the source of truth.\n"
          "                           If not specified, defaults to the first algorithm in the \"algParam\" JSON file.\n"
          "--warmupPerAlign           Number of warmup runs per alignments. Defaults to 0.\n"
          "--samplesPerAlign          Number of runs per alignment. Defaults to 1.\n"
          "\n"
          "--calcTrace                Should the trace be calculated. Defaults to true.\n"
          "--calcScoreHash            Should the score matrix hash be calculated. Used to verify correctness with the reference\n"
          "                           algorithm implementation. Defaults to false.\n"
          "--debugPath                For debug purposes, path of the TXT file where score matrices/traces will be\n"
          "                           written to, once per alignment. Defaults to \"\".\n"
          "--printScore               Should the score matrix be printed. Defaults to false.\n"
          "--printTrace               Should the trace be printed. Defaults to false.\n"
          "\n"
          "-h, --help                 Print help and exit.\n"
          "\n";
}

// get the current time as an ISO string
std::string IsoTime()
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
NwStat readFromJson(const std::string& path, T& res)
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

struct NwAlgorithmData
{
    std::map<std::string, NwAlgorithm> algMap;
};
extern NwAlgorithmData algData;

// input file formats
struct NwSubstData
{
    std::map<std::string, int> letterMap;
    std::map<std::string, std::vector<int>> substMap;
};

struct NwParamData
{
    std::map<std::string, NwParams> paramMap;
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

struct NwResData
{
    // directories
    std::string projPath;
    std::string resrcPath;
    std::string resPath;

    // filenames
    std::string isoTime;
    std::string substFname;
    std::string paramFname;
    std::string seqFname;
    std::string resFname;

    // result list
    std::vector<NwResult> resList;
};

// all algorithms
NwAlgorithmData algData {
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

// conversion to object from json
void from_json(const nlohmann::ordered_json& j, NwSubstData& substData)
{
    j.at("letterMap").get_to(substData.letterMap);
    j.at("substMap").get_to(substData.substMap);
}
void from_json(const nlohmann::ordered_json& j, NwParam& param)
{
    j.get_to(param._values);
}
void from_json(const nlohmann::ordered_json& j, NwParams& params)
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
void to_json(nlohmann::ordered_json& j, const NwParam& param)
{
    j = param._values;
}
void to_json(nlohmann::ordered_json& j, const NwParams& params)
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
void resHeaderToTsv(std::ostream& os)
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

    os << "alloc" << "\t";
    os << "cpy-dev" << "\t";
    os << "init-hdr" << "\t";
    os << "calc-init" << "\t";
    os << "calc" << "\t";
    os << "cpy-host" << "\t";

    os << "trace-alloc" << "\t";
    os << "trace-calc" << "\t";

    os << '\n';
}
void nwResultToTsv(std::ostream& os, const NwResult& res)
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
        os << int(res.cudaerr) << "\t";

        os.fill('0');
        os << std::setw(10) << res.score_hash << "\t";
        os << std::setw(10) << res.trace_hash << "\t";
    }
    fg.restore();

    lapTimeToTsv(os, res.sw_align.get_or_default("alloc"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("cpy-dev"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("init-hdr"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("calc-init"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("calc"));
    os << "\t";
    lapTimeToTsv(os, res.sw_align.get_or_default("cpy-host"));
    os << "\t";

    lapTimeToTsv(os, res.sw_trace.get_or_default("trace-alloc"));
    os << "\t";
    lapTimeToTsv(os, res.sw_trace.get_or_default("trace-calc"));
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

int main(int argc, char* argv[])
{
    extern NwAlgorithmData algData;
    NwSubstData substData;
    NwParamData paramData;
    NwSeqData seqData;
    NwResData resData;
    std::ofstream ofsRes;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-b" || arg == "--substPath")
        {
        }
        else if (arg == "-r" || arg == "--algParamPath")
        {
        }
        else if (arg == "-s" || arg == "--seqPath")
        {
        }
        else if (arg == "-p" || arg == "--pairPath")
        {
        }
        else if (arg == "-o" || arg == "--resPath")
        {
        }
        else if (arg == "--substName")
        {
        }
        else if (arg == "--gapoCost")
        {
        }
        else if (arg == "--gapeCost")
        {
        }
        else if (arg == "--algName")
        {
        }
        else if (arg == "--refAlgName")
        {
        }
        else if (arg == "--warmupPerAlign")
        {
        }
        else if (arg == "--samplesPerAlign")
        {
        }
        else if (arg == "--calcTrace")
        {
        }
        else if (arg == "--calcScoreHash")
        {
        }
        else if (arg == "--debugPath")
        {
        }
        else if (arg == "--printScore")
        {
        }
        else if (arg == "--printTrace")
        {
        }
        else if (arg == "--substPath")
        {
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_cmd_usage(std::cout);
            return 0;
        }
        else
        {
            std::cerr << "Unknown parameter: \"" << arg << "\"";
            return -1;
        }
    }

    resData.projPath = std::filesystem::current_path().string() + "/../../";
    resData.resrcPath = resData.projPath + "resrc/";
    resData.resPath = resData.projPath + "log/";

    resData.isoTime = IsoTime();
    resData.substFname = argv[1];
    resData.paramFname = argv[2];
    resData.seqFname = argv[3];
    resData.resFname = resData.isoTime + ".tsv";

    // read data from input .json files
    // +   also open the output file
    {
        std::string substPath = resData.resrcPath + resData.substFname;
        std::string paramPath = resData.resrcPath + resData.paramFname;
        std::string seqPath = resData.resrcPath + resData.seqFname;
        std::string resPath = resData.resPath + resData.resFname;

        if (NwStat::success != readFromJson(substPath, substData))
        {
            std::cerr << "ERR - could not open/read json from substs file";
            return -1;
        }
        if (NwStat::success != readFromJson(paramPath, paramData))
        {
            std::cerr << "ERR - could not open/read json from params file";
            return -1;
        }
        if (NwStat::success != readFromJson(seqPath, seqData))
        {
            std::cerr << "ERR - could not open/read json from seqs file";
            return -1;
        }
        if (NwStat::success != openOutFile(resPath, ofsRes))
        {
            std::cerr << "ERR - could not open tsv results file";
            return -1;
        }
    }
    auto defer1 = make_defer([&]() noexcept
    {
        ofsRes.close();
    });

    // get the device properties
    cudaDeviceProp deviceProps;
    if (cudaSuccess != (cudaStatus = cudaGetDeviceProperties(&deviceProps, 0 /*deviceId*/)))
    {
        std::cerr << "ERR - could not get device properties";
        return -1;
    }

    // number of streaming multiprocessors (sm-s) and threads in a warp
    const int sm_count = deviceProps.multiProcessorCount;
    const int warpsz = deviceProps.warpSize;
    const int maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;

    NwInput nw {
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
            std::cerr << "ERR - could not reserve space for the substitution matrix in the gpu";
            return -1;
        }

        // transfer the substitution matrix to the gpu global memory
        if (cudaSuccess != (cudaStatus = memTransfer(nw.subst_gpu, nw.subst, nw.substsz * nw.substsz)))
        {
            std::cerr << "ERR - could not transfer substitution matrix to the gpu";
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
    resHeaderToTsv(ofsRes);
    ofsRes.flush();

    // for all algorithms which have parameters in the param map
    for (auto& paramTuple : paramData.paramMap)
    {
        // if the current algorithm doesn't exist, skip it
        const std::string& algName = paramTuple.first;
        if (algData.algMap.find(algName) == algData.algMap.end())
        {
            continue;
        }

        std::cout << algName << ":";

        // get the current algorithm and initialize its parameters
        NwAlgorithm& alg = algData.algMap[algName];
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
                    std::vector<NwResult> resList {};

                    // for all requested repeats
                    for (int iR = 0; iR < seqData.repeat; iR++)
                    {
                        // initialize the result in the result list
                        resList.push_back(NwResult {
                            algName,                  // algName;
                            alg.alignPr().snapshot(), // algParams;

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
                        NwResult& res = resList.back();

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
                            res.cudaerr = cudaStatus;
                        }
                        if (!res.errstep && NwStat::success != (res.stat = alg.hash(nw, res)))
                        {
                            res.errstep = 3;
                            res.cudaerr = cudaStatus;
                        }
                        if (!res.errstep && NwStat::success != (res.stat = alg.trace(nw, res)))
                        {
                            res.errstep = 4;
                            res.cudaerr = cudaStatus;
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
                            std::cerr << "ERR - corrupted cuda context";
                            return -1;
                        }
                    }

                    // add the result to the results list
                    resData.resList.push_back(combineResults(resList));
                    NwResult& res = resData.resList.back();
                    // reset the multiple repetition list
                    resList.clear();

                    // print the result as a tsv line to the tsv output file
                    nwResultToTsv(ofsRes, res);
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
        std::cerr << "ERR - " << compareData.calcErrors << " calculation error(s)";
        return -1;
    }
}
