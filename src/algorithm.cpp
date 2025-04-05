#include "algorithm.hpp"
#include "fmt_guard.hpp"

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

void NwAlgorithm::init(NwParams& alignPr)
{
    _alignPr = alignPr;
}

NwParams& NwAlgorithm::alignPr()
{
    return _alignPr;
}

NwStat NwAlgorithm::align(NwInput& nw, NwResult& res)
{
    return _alignFn(_alignPr, nw, res);
}
NwStat NwAlgorithm::trace(const NwInput& nw, NwResult& res)
{
    return _traceFn(nw, res);
}
NwStat NwAlgorithm::hash(const NwInput& nw, NwResult& res)
{
    return _hashFn(nw, res);
}
NwStat NwAlgorithm::print(std::ostream& os, const NwInput& nw, NwResult& res)
{
    return _printFn(os, nw, res);
}

// print results
void NwPrintVect(
    std::ostream& os,
    const int* const vect,
    const int len)
{
    FormatFlagsGuard fg {os, 4};

    for (int i = 0; i < len; i++)
    {
        std::cout << std::setw(4) << vect[i] << ",";
    }
    std::cout << std::endl;
}

void NwPrintMat(
    std::ostream& os,
    const int* const mat,
    const int rows,
    const int cols)
{
    FormatFlagsGuard fg {os, 4};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            os << std::setw(4) << el(mat, cols, i, j) << ',';
        }
        os << std::endl;
    }
}

void NwPrintTiledMat(
    std::ostream& os,
    const int* const mat,
    const int rows,
    const int cols,
    const int tileWid /*without header column*/,
    const int tileHei /*without header row*/)
{
    FormatFlagsGuard fg {os, 4};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            os << std::setw(4) << el(mat, cols, i, j) << ',';
            if ((j - 1) % tileWid == 0)
            {
                os << "   ";
            }
        }

        os << std::endl;
        if ((i - 1) % tileHei == 0)
        {
            os << "   ";
        }
    }
}

void NwPrintHdrMat(
    std::ostream& os,
    const int* const tileHdrMat,
    const int rows,
    const int cols,
    const int hdrLen)
{
    FormatFlagsGuard fg {os, 4};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < hdrLen; k++)
            {
                int kHdrElem = (i * cols + j) * hdrLen + k;
                os << std::setw(4) << tileHdrMat[kHdrElem] << ',';
            }
            os << " // " << i << " " << j << std::endl;
        }
    }
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
