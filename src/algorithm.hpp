#ifndef INCLUDE_NW_ALGORITHM_HPP
#define INCLUDE_NW_ALGORITHM_HPP

#include "common.hpp"
#include <map>
#include <string>
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

// the Needleman-Wunsch algorithm implementations
class NwAlgorithm
{
public:
    using NwAlignFn = NwStat (*)(NwParams& pr, NwInput& nw, NwResult& res);
    using NwTraceFn = NwStat (*)(const NwInput& nw, NwResult& res);
    using NwHashFn = NwStat (*)(const NwInput& nw, NwResult& res);
    using NwPrintFn = NwStat (*)(std::ostream& os, const NwInput& nw, NwResult& res);

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

#endif // INCLUDE_NW_ALGORITHM_HPP
