#include "nw_algorithm.hpp"
#include "nw_fns.hpp"
#include "run_types.hpp"

NwAlgorithm::NwAlgorithm()
    : _alignFn(),
      _traceFn(),
      _hashFn(),
      _printScoreFn(),
      _printTraceFn()
{ }

NwAlgorithm::NwAlgorithm(
    NwAlgorithm::NwAlignFn alignFn,
    NwAlgorithm::NwTraceFn traceFn,
    NwAlgorithm::NwHashFn hashFn,
    NwAlgorithm::NwPrintScoreFn printScoreFn,
    NwAlgorithm::NwPrintTraceFn printTraceFn)
    : _alignFn(alignFn),
      _traceFn(traceFn),
      _hashFn(hashFn),
      _printScoreFn(printScoreFn),
      _printTraceFn(printTraceFn)
{ }

// Align seqX to seqY (seqX becomes seqY).
NwStat NwAlgorithm::align(const NwAlgParams& algParams, NwAlgInput& nw, NwAlgResult& res) const
{
    return _alignFn(algParams, nw, res);
}
NwStat NwAlgorithm::trace(NwAlgInput& nw, NwAlgResult& res) const
{
    return _traceFn(nw, res);
}
NwStat NwAlgorithm::hash(const NwAlgInput& nw, NwAlgResult& res) const
{
    return _hashFn(nw, res);
}
NwStat NwAlgorithm::printScore(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res) const
{
    return _printScoreFn(os, nw, res);
}
NwStat NwAlgorithm::printTrace(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res) const
{
    return _printTraceFn(os, nw, res);
}

void getNwAlgorithmMap(Dict<std::string, NwAlgorithm>& algMap)
{
    Dict<std::string, NwAlgorithm> algMapTmp {
        {
         {"NwAlign_Cpu1_St_Row", {NwAlign_Cpu1_St_Row, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Cpu2_St_Diag", {NwAlign_Cpu2_St_Diag, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Cpu3_St_DiagRow", {NwAlign_Cpu3_St_DiagRow, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Cpu4_Mt_DiagRow", {NwAlign_Cpu4_Mt_DiagRow, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu1_Ml_Diag", {NwAlign_Gpu1_Ml_Diag, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu2_Ml_DiagRow2Pass", {NwAlign_Gpu2_Ml_DiagRow2Pass, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu3_Ml_DiagDiag", {NwAlign_Gpu3_Ml_DiagDiag, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu4_Ml_DiagDiag2Pass", {NwAlign_Gpu4_Ml_DiagDiag2Pass, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu5_Coop_DiagDiag", {NwAlign_Gpu5_Coop_DiagDiag, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu6_Coop_DiagDiag2Pass", {NwAlign_Gpu6_Coop_DiagDiag2Pass, NwTrace1_Plain, NwHash1_Plain, NwPrintScore1_Plain, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu7_Mlsp_DiagDiag", {NwAlign_Gpu7_Mlsp_DiagDiag, NwTrace2_Sparse, NwHash2_Sparse, NwPrintScore2_Sparse, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu8_Mlsp_DiagDiag", {NwAlign_Gpu8_Mlsp_DiagDiag, NwTrace2_Sparse, NwHash2_Sparse, NwPrintScore2_Sparse, NwPrintTrace1_Plain}},
         {"NwAlign_Gpu9_Mlsp_DiagDiagDiag", {NwAlign_Gpu9_Mlsp_DiagDiagDiag, NwTrace2_Sparse, NwHash2_Sparse, NwPrintScore2_Sparse, NwPrintTrace1_Plain}},
         },
    };

    std::swap(algMap, algMapTmp);
}
