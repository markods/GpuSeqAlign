#include "cmd_parser.hpp"
#include "defer.hpp"
#include "fmt_guard.hpp"
#include "nwalign.hpp"
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
    int align_cost;
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
NwStat setOrVerifyResult(const NwAlgResult& res, NwCompareData& compareData)
{
    std::map<NwCompareKey, NwCompareRes>& compareMap = compareData.compareMap;
    NwCompareKey key {
        res.iY, // iY;
        res.iX  // iX;
    };
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

    NwCmdData cmdData {};
    if (NwStat stat = initCmdData(cmdArgs, cmdData); stat != NwStat::success)
    {
        return -1;
    }

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

    // get the device properties
    cudaDeviceProp deviceProps {};
    if (auto cudaStatus = cudaGetDeviceProperties(&deviceProps, 0 /*deviceId*/); cudaSuccess != cudaStatus)
    {
        std::cerr << "error: could not get device properties\n";
        return -1;
    }

    // number of streaming multiprocessors (sm-s) and threads in a warp
    const int sm_count = deviceProps.multiProcessorCount;
    const int warpsz = deviceProps.warpSize;
    const int maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;

    NwAlgInput nw {};
    auto defer1 = make_defer([&]() noexcept
    {
        nw.resetAllocsBenchmarkEnd();
    });

    // initialize the device parameters
    nw.sm_count = sm_count;
    nw.warpsz = warpsz;
    nw.maxThreadsPerBlock = maxThreadsPerBlock;

    // initialize the substitution matrix on the cpu and gpu
    {
        nw.subst = cmdData.substData.substMap[cmdArgs.substName.value()];
        nw.substsz = (int)std::sqrt(nw.subst.size());

        // reserve space in the gpu global memory
        try
        {
            nw.subst_gpu.init(nw.substsz * nw.substsz);
        }
        catch (const std::exception&)
        {
            std::cerr << "error: could not reserve space for the substitution matrix in the gpu\n";
            return -1;
        }

        // transfer the substitution matrix to the gpu global memory
        if (auto cudaStatus = memTransfer(nw.subst_gpu, nw.subst, nw.substsz * nw.substsz); cudaSuccess != cudaStatus)
        {
            std::cerr << "error: could not transfer substitution matrix to the gpu\n";
            return -1;
        }
    }

    // initialize the gapoCost cost
    nw.gapoCost = cmdArgs.gapoCost.value();
    // initialize the letter map
    std::map<std::string, int>& letterMap = cmdData.substData.letterMap;

    // initialize the sequence map
    std::vector<std::vector<int>> seqList {};
    for (auto& charSeq : cmdData.seqData.seqList)
    {
        auto seq = seqStrToVect(charSeq, letterMap, true /*addHeader*/);
        seqList.push_back(seq);
    }

    // initialize the gold result map (as calculated by the first algorithm)
    NwCompareData compareData {};

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
                        res.iX = iX;
                        res.iY = iY;
                        //
                        res.seqX_len = nw.seqX.size();
                        res.seqY_len = nw.seqY.size();
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
                        if (!res.errstep && NwStat::success != (res.stat = setOrVerifyResult(res, compareData)))
                        {
                            res.errstep = 5;
                            compareData.calcErrors++;
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
                            return -1;
                        }
                    }

                    // add the result to the results list
                    resultList.push_back(combineResults(resList));
                    NwAlgResult& res = resultList.back();
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
    if (compareData.calcErrors > 0)
    {
        std::cerr << "error: " << compareData.calcErrors << " calculation error(s)\n";
        return -1;
    }
}
