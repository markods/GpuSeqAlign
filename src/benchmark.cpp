#include "benchmark.hpp"
#include "defer.hpp"
#include "dict.hpp"
#include "file_formats.hpp"
#include "nw_algorithm.hpp"
#include "nw_fns.hpp"
#include "run_types.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

// convert the sequence string to a vector using a character map
// + NOTE: add the header (zeroth) element if requested
static std::vector<int> seqStrToVect(const std::string& str, const Dict<std::string, int>& map, const bool addHeader)
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

// Structs used to verify that the algorithms' results are correct.
struct NwCompareKey
{
    std::string seqY_id;
    std::string seqX_id;

    friend bool operator==(const NwCompareKey& l, const NwCompareKey& r);
    friend bool operator!=(const NwCompareKey& l, const NwCompareKey& r);
    friend bool operator<(const NwCompareKey& l, const NwCompareKey& r);
};
struct NwCompareRes
{
    int align_cost;
    unsigned score_hash;
    unsigned trace_hash;

    friend bool operator==(const NwCompareRes& l, const NwCompareRes& r);
    friend bool operator!=(const NwCompareRes& l, const NwCompareRes& r);
};

bool operator<(const NwCompareKey& l, const NwCompareKey& r)
{
    bool res =
        (l.seqY_id < r.seqY_id) ||
        (l.seqY_id == r.seqY_id && l.seqX_id < r.seqX_id);
    return res;
}
bool operator==(const NwCompareKey& l, const NwCompareKey& r)
{
    bool res =
        l.seqY_id == r.seqY_id &&
        l.seqX_id == r.seqX_id;
    return res;
}
bool operator!=(const NwCompareKey& l, const NwCompareKey& r)
{
    bool res =
        l.seqY_id != r.seqY_id ||
        l.seqX_id != r.seqX_id;
    return res;
}

namespace std
{
template <>
struct hash<NwCompareKey>
{
    size_t operator()(const NwCompareKey& key) const
    {
        size_t h1 = std::hash<std::string> {}(key.seqY_id);
        size_t h2 = std::hash<std::string> {}(key.seqX_id);

        // Taken from boost.
        if constexpr (sizeof(size_t) >= 8)
        {
            h1 ^= h2 + 0x517cc1b727220a95 + (h1 << 6) + (h1 >> 2);
        }
        else
        {
            h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        }
        return h1;
    }
};
} // namespace std

bool operator==(const NwCompareRes& l, const NwCompareRes& r)
{
    bool res =
        l.align_cost == r.align_cost &&
        l.score_hash == r.score_hash &&
        l.trace_hash == r.trace_hash;
    return res;
}
bool operator!=(const NwCompareRes& l, const NwCompareRes& r)
{
    bool res =
        l.align_cost != r.align_cost ||
        l.score_hash != r.score_hash ||
        l.trace_hash != r.trace_hash;
    return res;
}

// check that the result hashes match the hashes calculated by the first algorithm (the gold standard)
static NwStat setOrVerifyResult(const NwAlgResult& res, Dict<NwCompareKey, NwCompareRes>& compareMap)
{
    NwCompareKey key;
    key.seqY_id = res.seqY_id;
    key.seqX_id = res.seqX_id;

    NwCompareRes calcVal {};
    calcVal.align_cost = res.align_cost;
    calcVal.score_hash = res.score_hash;
    calcVal.trace_hash = res.trace_hash;

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
static NwAlgResult combineResults(std::vector<NwAlgResult>& resList)
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

static NwStat initNwInput(const NwCmdArgs& cmdArgs, const NwCmdData& cmdData, NwAlgInput& nw)
{
    // initialize the device parameters
    cudaDeviceProp deviceProps {};
    if (auto cudaStatus = cudaGetDeviceProperties(&deviceProps, 0 /*deviceId*/); cudaSuccess != cudaStatus)
    {
        std::cerr << "error: could not get device properties\n";
        return NwStat::errorCudaGeneral;
    }

    nw.sm_count = deviceProps.multiProcessorCount;
    nw.warpsz = deviceProps.warpSize;
    nw.maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;

    // initialize the substitution matrix on the cpu and gpu
    try
    {
        nw.subst = cmdData.substData.substMap.at(cmdArgs.substName.value());
        nw.substsz = (int)std::sqrt(nw.subst.size());
    }
    catch (const std::exception&)
    {
        std::cerr << "error: unknown substitution matrix name\n";
        return NwStat::errorInvalidValue;
    }

    // reserve space in the gpu global memory
    try
    {
        nw.subst_gpu.init(nw.substsz * nw.substsz);
    }
    catch (const std::exception&)
    {
        std::cerr << "error: could not reserve space for the substitution matrix in the gpu\n";
        return NwStat::errorMemoryAllocation;
    }

    // transfer the substitution matrix to the gpu global memory
    if (auto cudaStatus = memTransfer(nw.subst_gpu, nw.subst, nw.substsz * nw.substsz); cudaSuccess != cudaStatus)
    {
        std::cerr << "error: could not transfer substitution matrix to the gpu\n";
        return NwStat::errorMemoryTransfer;
    }

    // initialize the gapoCost cost
    nw.gapoCost = cmdArgs.gapoCost.value();

    return NwStat::success;
}

NwStat benchmarkAlgs(const NwCmdArgs& cmdArgs, NwCmdData& cmdData, NwBenchmarkData& benchData)
{
    Dict<NwCompareKey, NwCompareRes> compareMap {};
    Dict<std::string, NwAlgorithm> algMap {};
    getNwAlgorithmMap(algMap);

    NwAlgInput nw {};
    auto defer1 = make_defer([&]() noexcept
    {
        nw.resetAllocsBenchmarkEnd();
    });
    initNwInput(cmdArgs, cmdData, nw);

    // initialize the letter map
    Dict<std::string, int>& letterMap = cmdData.substData.letterMap;

    // initialize the sequence map
    std::vector<std::vector<int>> seqList {};
    for (auto& charSeq : cmdData.seqData.seqList)
    {
        auto seq = seqStrToVect(charSeq, letterMap, true /*addHeader*/);
        seqList.push_back(seq);
    }

    // write the tsv file's header
    writeResultHeaderToTsv(cmdData.resOfs, cmdArgs.fCalcScoreHash.value(), cmdArgs.fCalcTrace.value());
    if (cmdArgs.fWriteProgress.value())
    {
        // Since we write to progress immediately, write to file immediately as well.
        cmdData.resOfs.flush();
    }

    // for all algorithms which have parameters in the param map
    for (auto& paramTuple : cmdData.algParamsData.paramMap)
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

                // for all parameter combinations
                for (; alg.alignPr().hasCurr(); alg.alignPr().next())
                {
                    // results from multiple repetitions
                    std::vector<NwAlgResult> resList {};

                    // for all requested repeats
                    for (int iR = -cmdArgs.warmupPerAlign.value(); iR < cmdArgs.samplesPerAlign.value(); iR++)
                    {
                        auto defer2 = make_defer([&]() noexcept
                        {
                            nw.resetAllocsBenchmarkCycle();
                        });

                        resList.push_back(NwAlgResult {});
                        NwAlgResult& res = resList.back();

                        res.algName = algName;
                        res.algParams = alg.alignPr().copy();
                        res.seqY_id = std::to_string(iY);
                        res.seqX_id = std::to_string(iX);
                        //
                        res.seqY_len = nw.seqY.size();
                        res.seqX_len = nw.seqX.size();
                        res.substName = cmdArgs.substName.value();
                        res.gapoCost = cmdArgs.gapoCost.value();
                        res.warmup_runs = cmdArgs.warmupPerAlign.value();
                        res.sample_runs = cmdArgs.samplesPerAlign.value();

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
                        if (cmdArgs.fCalcScoreHash.value() && !res.errstep && NwStat::success != (res.stat = alg.hash(nw, res)))
                        {
                            res.errstep = 3;
                        }
                        if (cmdArgs.fCalcTrace.value() && !res.errstep && NwStat::success != (res.stat = alg.trace(nw, res)))
                        {
                            res.errstep = 4;
                        }
                        if (!res.errstep && NwStat::success != (res.stat = setOrVerifyResult(res, compareMap)))
                        {
                            res.errstep = 5;
                            benchData.calcErrors++;
                        }

                        if (iR < 0)
                        {
                            // Discard warmup runs.
                            resList.pop_back();
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

                        // Clear cuda non-sticky errors and get possible cuda sticky errors.
                        // Since sticky errors cannot be cleared, so repeat twice.
                        if (auto cudaStatus = (cudaGetLastError(), cudaGetLastError()); cudaStatus != cudaSuccess)
                        {
                            std::cerr << "error: corrupted cuda context\n";
                            return NwStat::errorCudaGeneral;
                        }
                    }

                    // add the result to the results list
                    benchData.resultList.push_back(combineResults(resList));
                    NwAlgResult& res = benchData.resultList.back();
                    // reset the multiple repetition list
                    resList.clear();

                    // print the result as a tsv line to the tsv output file
                    writeResultLineToTsv(cmdData.resOfs, res, cmdArgs.fCalcScoreHash.value(), cmdArgs.fCalcTrace.value());
                    if (cmdArgs.fWriteProgress.value())
                    {
                        // Since we write to progress immediately, write to file immediately as well.
                        cmdData.resOfs.flush();
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
    if (benchData.calcErrors > 0)
    {
        std::cerr << "error: " << benchData.calcErrors << " calculation error(s)\n";
        return NwStat::errorInvalidResult;
    }

    return NwStat::success;
}
