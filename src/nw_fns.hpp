#ifndef INCLUDE_NW_FNS_HPP
#define INCLUDE_NW_FNS_HPP

#include "run_types.hpp"
#include <iostream>
#include <vector>

NwStat NwAlign_Cpu1_St_Row(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Cpu2_St_Diag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Cpu3_St_DiagRow(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Cpu4_Mt_DiagRow(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu1_Ml_Diag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu2_Ml_DiagRow2Pass(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu3_Ml_DiagDiag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu4_Ml_DiagDiag2Pass(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu5_Coop_DiagDiag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu6_Coop_DiagDiag2Pass(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu7_Mlsp_DiagDiag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu8_Mlsp_DiagDiag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
NwStat NwAlign_Gpu9_Mlsp_DiagDiagDiag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);

NwStat NwTrace1_Plain(NwAlgInput& nw, NwAlgResult& res);
NwStat NwHash1_Plain(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwPrintScore1_Plain(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res);
NwStat NwPrintTrace1_Plain(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res);

struct TileAndElemIJ
{
    int iTile;
    int jTile;
    int iTileElem;
    int jTileElem;
};

NwStat NwTrace2_Sparse(NwAlgInput& nw, NwAlgResult& res);
NwStat NwHash2_Sparse(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwPrintScore2_Sparse(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res);
void NwTrace2_AlignTile(std::vector<int>& tile, const NwAlgInput& nw, const TileAndElemIJ& co);
void NwTrace2_GetTileAndElemIJ(const NwAlgInput& nw, int i, int j, TileAndElemIJ& co);

#endif // INCLUDE_NW_FNS_HPP
