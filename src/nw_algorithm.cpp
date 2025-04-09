#include "nw_algorithm.hpp"
#include "nw_fns.hpp"
#include "run_types.hpp"

NwAlgorithm::NwAlgorithm()
    : _alignFn(),
      _traceFn(),
      _hashFn(),
      _printFn(),
      _alignPr()
{ }

NwAlgorithm::NwAlgorithm(
    NwAlgorithm::NwAlignFn alignFn,
    NwAlgorithm::NwTraceFn traceFn,
    NwAlgorithm::NwHashFn hashFn,
    NwAlgorithm::NwPrintFn printFn)
    : _alignFn(alignFn),
      _traceFn(traceFn),
      _hashFn(hashFn),
      _printFn(printFn),
      _alignPr()
{ }

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

void getNwAlgorithmMap(std::map<std::string, NwAlgorithm>& algMap)
{
    std::map<std::string, NwAlgorithm> algMapTmp {
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

    std::swap(algMap, algMapTmp);
}
