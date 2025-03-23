#ifndef INCLUDE_NW_ALGORITHM_HPP
#define INCLUDE_NW_ALGORITHM_HPP

#include "common.hpp"
#include "json.hpp"
#include <iostream>
#include <map>
#include <string>
#include <vector>
using json = nlohmann::ordered_json;

using NwAlignFn = NwStat (*)(NwParams& pr, NwInput& nw, NwResult& res);
using NwTraceFn = NwStat (*)(const NwInput& nw, NwResult& res);
using NwHashFn = NwStat (*)(const NwInput& nw, NwResult& res);
using NwPrintFn = NwStat (*)(std::ostream& os, const NwInput& nw, NwResult& res);

// the Needleman-Wunsch algorithm implementations
class NwAlgorithm
{
public:
    NwAlgorithm()
    {
        _alignFn = {};
        _traceFn = {};
        _hashFn = {};
        _printFn = {};

        _alignPr = {};
    }

    NwAlgorithm(
        NwAlignFn alignFn,
        NwTraceFn traceFn,
        NwHashFn hashFn,
        NwPrintFn printFn)
    {
        _alignFn = alignFn;
        _traceFn = traceFn;
        _hashFn = hashFn;
        _printFn = printFn;

        _alignPr = {};
    }

    void init(NwParams& alignPr)
    {
        _alignPr = alignPr;
    }

    NwParams& alignPr()
    {
        return _alignPr;
    }

    NwStat align(NwInput& nw, NwResult& res)
    {
        return _alignFn(_alignPr, nw, res);
    }
    NwStat trace(const NwInput& nw, NwResult& res)
    {
        return _traceFn(nw, res);
    }
    NwStat hash(const NwInput& nw, NwResult& res)
    {
        return _hashFn(nw, res);
    }
    NwStat print(std::ostream& os, const NwInput& nw, NwResult& res)
    {
        return _printFn(os, nw, res);
    }

private:
    NwAlignFn _alignFn;
    NwTraceFn _traceFn;
    NwHashFn _hashFn;
    NwPrintFn _printFn;

    NwParams _alignPr;
};

// algorithm map
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

// conversion to object from json
void from_json(const json& j, NwSubstData& substData);
void from_json(const json& j, NwParamData& paramData);
void from_json(const json& j, NwParams& params);
void from_json(const json& j, NwParam& param);
void from_json(const json& j, NwSeqData& seqData);

// conversion to json from object
void to_json(json& j, const NwSubstData& substData);
void to_json(json& j, const NwParamData& paramData);
void to_json(json& j, const NwParams& params);
void to_json(json& j, const NwParam& param);
void to_json(json& j, const NwSeqData& seqData);

// conversion to csv from object
void resHeaderToCsv(std::ostream& os, const NwResData& resData);
void to_csv(std::ostream& os, const NwResult& res);
void to_csv(std::ostream& os, const Stopwatch& sw);
void paramsToCsv(std::ostream& os, const std::map<std::string, int>& paramMap);
void lapTimeToCsv(std::ostream& os, float lapTime);

// convert the sequence string to a vector using a character map
// + NOTE: add the header (zeroth) element if requested
std::vector<int> seqStrToVect(const std::string& str, const std::map<std::string, int>& map, const bool addHeader);

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

// check that the result hashes match the hashes calculated by the first algorithm (the gold standard)
NwStat setOrVerifyResult(const NwResult& res, NwCompareData& compareData);
// combine results from many repetitions into one
NwResult combineResults(std::vector<NwResult>& resList);

// get the current time as an ISO string
std::string IsoTime();

// open output file stream
NwStat openOutFile(const std::string& path, std::ofstream& ofs);

// read a json file into a variable
template <typename T>
NwStat readFromJson(const std::string& path, T& var)
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

    try
    {
        // NOTE: the parser doesn't allow for trailing commas
        var = json::parse(
            ifs,
            /*callback*/ nullptr,
            /*allow_exceptions*/ true,
            /*ignore_comments*/ true);
    }
    catch (const std::exception&)
    {
        NwStat::errorInvalidFormat;
    }

    return NwStat::success;
}

#endif // INCLUDE_NW_ALGORITHM_HPP
