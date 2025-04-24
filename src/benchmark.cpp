#include "benchmark.hpp"
#include "defer.hpp"
#include "dict.hpp"
#include "file_formats.hpp"
#include "fmt_guard.hpp"
#include "nw_algorithm.hpp"
#include "nw_fns.hpp"
#include "run_types.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

template <typename T>
static NwStat vectorSubstringWithHeader(const std::vector<T>& vect, const NwRange& range, std::vector<T>& res)
{
    int64_t vectSizeNoHeader = (int64_t)vect.size() - 1 /*without header*/;

    if ((range.l < 0 || range.l >= vectSizeNoHeader) || (range.r <= range.l || range.r > vectSizeNoHeader))
    {
        return NwStat::errorInvalidValue;
    }

    if ((!range.lNotDefault || range.l == 0) && (!range.rNotDefault || range.r == vectSizeNoHeader))
    {
        res = vect;
        return NwStat::success;
    }

    res = std::vector<T> {};
    res.reserve(1 + (range.r - range.l));
    res.push_back(0); // Header.
    res.insert(res.end(), 1 /*header*/ + vect.begin() + range.l, 1 /*header*/ + vect.begin() + range.r);

    return NwStat::success;
}

// Struct used to verify that the algorithms' results are correct.
struct NwCompareKey
{
    std::string seqY_id;
    std::string seqX_id;
    NwRange seqY_range;
    NwRange seqX_range;

    friend bool operator==(const NwCompareKey& l, const NwCompareKey& r);
    friend bool operator!=(const NwCompareKey& l, const NwCompareKey& r);
};
// Struct used to verify that the algorithms' results are correct.
struct NwCompareRes
{
    int align_cost;
    unsigned score_hash;
    unsigned trace_hash;

    friend bool operator==(const NwCompareRes& l, const NwCompareRes& r);
    friend bool operator!=(const NwCompareRes& l, const NwCompareRes& r);
};
bool operator==(const NwCompareKey& l, const NwCompareKey& r)
{
    bool res =
        l.seqY_id == r.seqY_id &&
        l.seqX_id == r.seqX_id &&
        l.seqY_range == r.seqY_range &&
        l.seqX_range == r.seqX_range;
    return res;
}
bool operator!=(const NwCompareKey& l, const NwCompareKey& r)
{
    bool res =
        l.seqY_id != r.seqY_id ||
        l.seqX_id != r.seqX_id ||
        l.seqY_range != r.seqY_range ||
        l.seqX_range != r.seqX_range;
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

// Check that the result hashes match the hashes calculated by the reference algorithm (calculated first).
static NwStat setOrVerifyResult(const NwAlgResult& res, Dict<NwCompareKey, NwCompareRes>& compareMap)
{
    NwCompareKey key {};
    key.seqY_id = res.seqY_id;
    key.seqX_id = res.seqX_id;
    key.seqY_range = res.seqY_range;
    key.seqX_range = res.seqX_range;

    NwCompareRes calcVal {};
    calcVal.align_cost = res.align_cost;
    calcVal.score_hash = res.score_hash;
    calcVal.trace_hash = res.trace_hash;

    auto compareRes = compareMap.find(key);
    if (compareRes == compareMap.end())
    {
        compareMap[key] = calcVal;
        return NwStat::success;
    }

    NwCompareRes& expVal = compareMap[key];
    if (calcVal != expVal)
    {
        return NwStat::errorInvalidResult;
    }

    return NwStat::success;
}

static NwAlgResult combineRepResults(std::vector<NwAlgResult>& resList)
{
    if (resList.empty())
    {
        return NwAlgResult {};
    }

    std::vector<Stopwatch> swAlignList {};
    std::vector<Stopwatch> swHashList {};
    std::vector<Stopwatch> swTraceList {};
    for (auto& curr : resList)
    {
        swAlignList.push_back(curr.sw_align);
        swHashList.push_back(curr.sw_hash);
        swTraceList.push_back(curr.sw_trace);
    }

    // Take the last result - if it errored it is definitely the last result.
    NwAlgResult res = resList[resList.size() - 1]; // Copy on purpose.
    res.sw_align = Stopwatch::combine(swAlignList);
    res.sw_hash = Stopwatch::combine(swHashList);
    res.sw_trace = Stopwatch::combine(swTraceList);

    return res;
}

static NwStat initNwInput(const NwCmdArgs& cmdArgs, const NwCmdData& cmdData, NwAlgInput& nw)
{
    // Initialize the device parameters.
    cudaDeviceProp deviceProps {};
    if (auto cudaStatus = cudaGetDeviceProperties(&deviceProps, 0 /*deviceId*/); cudaSuccess != cudaStatus)
    {
        std::cerr << "error: could not get device properties\n";
        return NwStat::errorCudaGeneral;
    }

    nw.sm_count = deviceProps.multiProcessorCount;
    nw.warpsz = deviceProps.warpSize;
    nw.maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;

    // Initialize the substitution matrix on the cpu and gpu.
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

    // Reserve space in the gpu global memory.
    try
    {
        nw.subst_gpu.init(nw.substsz * nw.substsz);
    }
    catch (const std::exception&)
    {
        std::cerr << "error: could not reserve space for the substitution matrix in the gpu\n";
        return NwStat::errorMemoryAllocation;
    }

    // Transfer the substitution matrix to the gpu global memory.
    if (auto cudaStatus = memTransfer(nw.subst_gpu, nw.subst, nw.substsz * nw.substsz); cudaSuccess != cudaStatus)
    {
        std::cerr << "error: could not transfer substitution matrix to the gpu\n";
        return NwStat::errorMemoryTransfer;
    }

    // Initialize the gapoCost cost.
    nw.gapoCost = cmdArgs.gapoCost.value();

    return NwStat::success;
}

void breakProgressLine(const NwCmdArgs& cmdArgs, bool& write_progress_newline)
{
    if (cmdArgs.fWriteProgress.value() && write_progress_newline)
    {
        std::cout << '\n'
                  << std::flush;
        write_progress_newline = false;
    }
}

static NwStat printBenchReportLine(
    const NwCmdArgs& cmdArgs,
    NwCmdData& cmdData,
    const NwAlgorithm& alg,
    const NwAlgInput& nw,
    NwAlgResult& repResCombined,
    bool& write_progress_newline)
{
    // Print progress to stdout.
    if (cmdArgs.fWriteProgress.value())
    {
        if (repResCombined.stat == NwStat::success)
        {
            std::cout << '.' << std::flush;
        }
        else
        {
            std::cout << repResCombined.errstep << std::flush;
        }
        write_progress_newline = true;
    }

    // Print the result to the tsv output file.
    {
        TsvPrintCtl printCtl {};
        printCtl.writeValue = 1;
        printCtl.fPrintScoreStats = cmdArgs.fCalcScoreHash.value();
        printCtl.fPrintTraceStats = cmdArgs.fCalcTrace.value();
        if (NwStat stat = writeNwResultToTsv(cmdData.resOfs, repResCombined, printCtl); stat != NwStat::success)
        {
            breakProgressLine(cmdArgs, write_progress_newline);
            std::cerr << "error: could not write output tsv result line\n";
            return stat;
        }
    }
    if (cmdArgs.fWriteProgress.value())
    {
        // Since we write to progress immediately, write to file immediately as well.
        cmdData.resOfs.flush();
    }

    // Print the result to the debug output file.
    if (cmdArgs.fPrintScore.value() || cmdArgs.fPrintTrace.value())
    {
        TsvPrintCtl printCtl {};
        printCtl.fPrintScoreStats = cmdArgs.fCalcScoreHash.value();
        printCtl.fPrintTraceStats = cmdArgs.fCalcTrace.value();

        cmdData.debugOfs << ">results\n";

        printCtl.writeColName = 1;
        printCtl.writeValue = 0;
        if (NwStat stat = writeNwResultToTsv(cmdData.debugOfs, repResCombined /*unused*/, printCtl); stat != NwStat::success)
        {
            breakProgressLine(cmdArgs, write_progress_newline);
            std::cerr << "error: could not write debug output tsv header\n";
            return stat;
        }

        printCtl.writeColName = 0;
        printCtl.writeValue = 1;
        if (NwStat stat = writeNwResultToTsv(cmdData.debugOfs, repResCombined, printCtl); stat != NwStat::success)
        {
            breakProgressLine(cmdArgs, write_progress_newline);
            std::cerr << "error: could not write debug output tsv result line\n";
            return stat;
        }

        if (cmdArgs.fPrintTrace.value())
        {
            cmdData.debugOfs << "+\n>edit_trace\n";
            alg.printTrace(cmdData.debugOfs, nw, repResCombined);
            cmdData.debugOfs << '\n';
        }

        if (cmdArgs.fPrintScore.value())
        {
            cmdData.debugOfs << "+\n>score_matrix\n";
            alg.printScore(cmdData.debugOfs, nw, repResCombined);
        }

        cmdData.debugOfs << "\n\n";

        if (cmdArgs.fWriteProgress.value())
        {
            // Since we write to progress immediately, write to file immediately as well.
            cmdData.debugOfs.flush();
        }
    }

    return NwStat::success;
}

NwStat benchmarkAlgs(const NwCmdArgs& cmdArgs, NwCmdData& cmdData, NwBenchmarkData& benchData)
{
    Dict<NwCompareKey, NwCompareRes> compareMap {};
    Dict<std::string, NwAlgorithm> algMap {};
    getNwAlgorithmMap(algMap);

    NwAlgInput nw {};
    bool write_progress_newline {};
    auto defer1 = make_defer([&nw, &benchData, &cmdArgs, &write_progress_newline]() noexcept
    {
        nw.resetAllocsBenchmarkEnd();

        if (benchData.calcErrors > 0)
        {
            // Must not throw in defer.
            FormatFlagsGuard fg {std::cerr};
            std::cerr.exceptions(std::ios_base::goodbit);
            breakProgressLine(cmdArgs, write_progress_newline);
            std::cerr << "error: " << benchData.calcErrors << " calculation error(s)\n";
        }
    });
    initNwInput(cmdArgs, cmdData, nw);

    auto seqList = cmdData.seqData.seqMap.values();
    auto& seqPairList = cmdData.seqPairData.pairList;

    Dict<std::string, int> seqIdxMap {};
    {
        int iSeq = 0;
        for (const auto& seq : cmdData.seqData.seqMap)
        {
            seqIdxMap.insert(seq.second.id, iSeq);
            iSeq++;
        }
    }

    // Write tsv header.
    {
        TsvPrintCtl printCtl {};
        printCtl.writeColName = 1;
        printCtl.fPrintScoreStats = cmdArgs.fCalcScoreHash.value();
        printCtl.fPrintTraceStats = cmdArgs.fCalcTrace.value();
        NwAlgResult res {};
        if (NwStat stat = writeNwResultToTsv(cmdData.resOfs, res /*unused*/, printCtl); stat != NwStat::success)
        {
            breakProgressLine(cmdArgs, write_progress_newline);
            std::cerr << "error: could not write output tsv header\n";
            return stat;
        }
    }
    if (cmdArgs.fWriteProgress.value())
    {
        // Since we write to progress immediately, write to file immediately as well.
        cmdData.resOfs.flush();
    }

    auto algNames = cmdArgs.algNames.value(); // Copy on purpose.
    std::string refAlgName = cmdArgs.refAlgName.value();
    if (auto iter = std::find(algNames.begin(), algNames.end(), refAlgName); iter != algNames.end())
    {
        algNames.erase(iter);
        algNames.insert(algNames.begin(), refAlgName);
    }

    // For all selected algorithms.
    for (auto& algName : algNames)
    {
        if (cmdArgs.fWriteProgress.value())
        {
            std::cout << algName << ":\n"
                      << std::flush;
            write_progress_newline = false;
        }

        NwAlgorithm& alg = algMap[algName];
        NwAlgParams algParams = cmdData.algParamsData.paramMap.at(algName); // Copy on purpose.

        // For all sequence pairs.
        for (const NwSeqPair& seqPair : seqPairList)
        {
            int iY = seqIdxMap.at(seqPair.seqY_id);
            int iX = seqIdxMap.at(seqPair.seqX_id);

            if (NwStat stat = vectorSubstringWithHeader(seqList[iY].seq, seqPair.seqY_range, nw.seqY); stat != NwStat::success)
            {
                breakProgressLine(cmdArgs, write_progress_newline);
                std::cerr << "error: cannot take substring from seqY\n";
                return stat;
            }
            if (NwStat stat = vectorSubstringWithHeader(seqList[iX].seq, seqPair.seqX_range, nw.seqX); stat != NwStat::success)
            {
                breakProgressLine(cmdArgs, write_progress_newline);
                std::cerr << "error: cannot take substring from seqX\n";
                return stat;
            }

            // Already includes header (zeroth) element.
            nw.adjrows = (int)nw.seqY.size();
            nw.adjcols = (int)nw.seqX.size();

            // For all parameter combinations.
            for (; algParams.hasCurr(); algParams.next())
            {
                std::vector<NwAlgResult> repResList {};

                // For all requested repeats.
                for (int iR = -cmdArgs.warmupPerAlign.value(); iR < cmdArgs.samplesPerAlign.value(); iR++)
                {
                    auto defer2 = make_defer([&nw]() noexcept
                    {
                        nw.resetAllocsBenchmarkCycle();
                    });

                    repResList.push_back(NwAlgResult {});
                    NwAlgResult& res = repResList.back();

                    res.algName = algName;
                    res.algParams = algParams.copy();
                    res.seqY_idx = iY;
                    res.seqX_idx = iX;
                    res.seqY_id = seqList[iY].id;
                    res.seqX_id = seqList[iX].id;
                    res.seqY_range = seqPair.seqY_range;
                    res.seqX_range = seqPair.seqX_range;
                    //
                    res.seqY_len = nw.seqY.size() - 1 /*header*/;
                    res.seqX_len = nw.seqX.size() - 1 /*header*/;
                    res.substName = cmdArgs.substName.value();
                    res.gapoCost = cmdArgs.gapoCost.value();
                    res.warmup_runs = cmdArgs.warmupPerAlign.value();
                    res.sample_runs = cmdArgs.samplesPerAlign.value();
                    res.last_run_idx = iR;
                    //
                    res.sm_count = nw.sm_count;

                    // Clear cuda non-sticky errors and get possible cuda sticky errors.
                    // Since sticky errors cannot be cleared, repeat twice.
                    if (auto cudaStatus = (cudaGetLastError(), cudaGetLastError()); cudaStatus != cudaSuccess)
                    {
                        breakProgressLine(cmdArgs, write_progress_newline);
                        std::cerr << "error: corrupted cuda context\n";
                        return NwStat::errorCudaGeneral;
                    }

                    // Compare the sequences, hash and trace the score matrices, and verify the soundness of the results.
                    if (!res.errstep && NwStat::success != (res.stat = alg.align(algParams, nw, res)))
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
                    if (cmdArgs.fCalcTrace.value() && !res.errstep && NwStat::success != (res.stat = alg.trace(nw, res, cmdArgs.fPrintTrace.value())))
                    {
                        res.errstep = 4;
                    }
                    if (!res.errstep && NwStat::success != (res.stat = setOrVerifyResult(res, compareMap)))
                    {
                        res.errstep = 5;
                        benchData.calcErrors++;
                    }

                    if (iR < 0 && res.stat == NwStat::success)
                    {
                        // Discard successful warmup runs.
                        repResList.pop_back();
                    }

                    // Last iteration.
                    if (iR == cmdArgs.samplesPerAlign.value() - 1 || res.stat != NwStat::success)
                    {
                        benchData.resultList.push_back(combineRepResults(repResList));
                        NwAlgResult& repResCombined = benchData.resultList.back();
                        repResList.clear();

                        ZIG_TRY(NwStat::success, printBenchReportLine(cmdArgs, cmdData, alg, nw, repResCombined, write_progress_newline));
                    }

                    if (res.stat != NwStat::success)
                    {
                        break;
                    }
                }
            }

            algParams.reset();
        }

        if (cmdArgs.fWriteProgress.value())
        {
            // Algorithm separator.
            std::cout << "\n\n"
                      << std::flush;
            write_progress_newline = false;
        }
    }

    if (benchData.calcErrors > 0)
    {
        // Error message written in defer.
        return NwStat::errorInvalidResult;
    }

    return NwStat::success;
}
