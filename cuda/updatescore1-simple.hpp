#pragma once
#include "common.hpp"

// update the score given the current score matrix and position
inline void UpdateScore1_Simple(
   const int* const seqX,
   const int* const seqY,
   int* const score,
   const int rows,
   const int cols,
   const int insdelcost,
   const int i,
   const int j )
   noexcept
{
   const int p1 = el(score,cols, i-1,j-1) + subst[ seqY[i] ][ seqX[j] ];
   const int p2 = el(score,cols, i-1,j  ) - insdelcost;
   const int p3 = el(score,cols, i  ,j-1) - insdelcost;
   el(score,cols, i,j) = max3( p1, p2, p3 );
}
