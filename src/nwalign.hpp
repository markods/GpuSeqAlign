#ifndef INCLUDE_NWALIGN_HPP
#define INCLUDE_NWALIGN_HPP

#include "run_types.hpp"

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

#endif // INCLUDE_NWALIGN_HPP
