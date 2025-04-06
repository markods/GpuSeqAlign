#ifndef INCLUDE_NWALIGN_HPP
#define INCLUDE_NWALIGN_HPP

#include "run_types.hpp"

// align functions implemented in other files
NwStat NwAlign_Cpu1_St_Row(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Cpu2_St_Diag(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Cpu3_St_DiagRow(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Cpu4_Mt_DiagRow(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu1_Ml_Diag(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu2_Ml_DiagRow2Pass(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu3_Ml_DiagDiag(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu4_Ml_DiagDiag2Pass(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu5_Coop_DiagDiag(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu6_Coop_DiagDiag2Pass(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu7_Mlsp_DiagDiag(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu8_Mlsp_DiagDiag(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu9_Mlsp_DiagDiagDiag(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);

// traceback, hash and print functions implemented in other files
NwStat NwTrace1_Plain(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwTrace2_Sparse(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwHash1_Plain(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwHash2_Sparse(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwPrint1_Plain(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);
NwStat NwPrint2_Sparse(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);

#endif // INCLUDE_NWALIGN_HPP
