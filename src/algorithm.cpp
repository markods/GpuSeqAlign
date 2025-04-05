#include "algorithm.hpp"

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
