#include "nw-algorithm.hpp"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

// align functions implemented in other files
NwStat NwAlign_Cpu1_St_Row(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Cpu2_St_Diag(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Cpu3_St_DiagRow(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Cpu4_Mt_DiagRow(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu1_Ml_Diag(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu2_Ml_DiagRow2Pass(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu3_Ml_DiagDiag(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu4_Ml_DiagDiag2Pass(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu5_Coop_DiagDiag(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu6_Coop_DiagDiag2Pass(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu7_Mlsp_DiagDiag(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu8_Mlsp_DiagDiag(NwParams& pr, NwInput& nw, NwResult& res);
NwStat NwAlign_Gpu9_Mlsp_DiagDiagDiag(NwParams& pr, NwInput& nw, NwResult& res);

// traceback, hash and print functions implemented in other files
NwStat NwTrace1_Plain(const NwInput& nw, NwResult& res);
NwStat NwTrace2_Sparse(const NwInput& nw, NwResult& res);
NwStat NwHash1_Plain(const NwInput& nw, NwResult& res);
NwStat NwHash2_Sparse(const NwInput& nw, NwResult& res);
NwStat NwPrint1_Plain(std::ostream& os, const NwInput& nw, NwResult& res);
NwStat NwPrint2_Sparse(std::ostream& os, const NwInput& nw, NwResult& res);

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
void from_json(const nlohmann::ordered_json& j, NwParamData& paramData)
{
    j.get_to(paramData.paramMap);
}
void from_json(const nlohmann::ordered_json& j, NwParams& params)
{
    j.get_to(params._params);
}
void from_json(const nlohmann::ordered_json& j, NwParam& param)
{
    j.get_to(param._values);
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
    j = nlohmann::ordered_json {
        {"letterMap", substData.letterMap},
        {"substMap",  substData.substMap }
    };
}
void to_json(nlohmann::ordered_json& j, const NwParamData& paramData)
{
    j = nlohmann::ordered_json {
        {"paramMap", paramData.paramMap},
    };
}
void to_json(nlohmann::ordered_json& j, const NwParams& params)
{
    j = params._params;
}
void to_json(nlohmann::ordered_json& j, const NwParam& param)
{
    j = param._values;
}
void to_json(nlohmann::ordered_json& j, const NwSeqData& seqData)
{
    j = nlohmann::ordered_json {
        {"substName", seqData.substName},
        {"indel",     seqData.indel    },
        {"repeat",    seqData.repeat   },
        {"seqList",   seqData.seqList  }
    };
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
NwStat setOrVerifyResult(const NwResult& res, NwCompareData& compareData)
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
NwResult combineResults(std::vector<NwResult>& resList)
{
    // if the result list is empty, return a default initialized result
    if (resList.empty())
    {
        return NwResult {};
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
    NwResult res = resList[resList.size() - 1];
    // combine the stopwatches from many repeats into one
    res.sw_align = Stopwatch::combineStopwatches(swAlignList);
    res.sw_hash = Stopwatch::combineStopwatches(swHashList);
    res.sw_trace = Stopwatch::combineStopwatches(swTraceList);

    return res;
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
