#ifndef INCLUDE_NW_FNS_HPP
#define INCLUDE_NW_FNS_HPP

#include "run_types.hpp"
#include <vector>

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

NwStat NwTrace1_Plain(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwHash1_Plain(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwPrint1_Plain(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);

struct TileAndElemIJ
{
    int iTile;
    int jTile;
    int iTileElem;
    int jTileElem;
};

NwStat NwTrace2_Sparse(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwHash2_Sparse(const NwAlgInput& nw, NwAlgResult& res);
NwStat NwPrint2_Sparse(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);
void NwTrace2_AlignTile(std::vector<int>& tile, const NwAlgInput& nw, int iTile, int jTile);
void NwTrace2_GetTileAndElemIJ(const NwAlgInput& nw, int i, int j, TileAndElemIJ& co);

#endif // INCLUDE_NW_FNS_HPP
