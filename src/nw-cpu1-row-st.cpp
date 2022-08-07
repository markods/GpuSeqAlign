#include "common.hpp"

// sequential cpu implementation of the Needleman Wunsch algorithm
void Nw_Cpu1_Row_St( NwInput& nw, NwMetrics& res )
{
   // start the timer
   res.sw.lap( "cpu-start" );

   // initialize the first row and column of the score matrix
   for( int i = 0; i < nw.adjrows; i++ ) el(nw.score,nw.adjcols, i,0) = -i*nw.insdelcost;
   for( int j = 0; j < nw.adjcols; j++ ) el(nw.score,nw.adjcols, 0,j) = -j*nw.insdelcost;

   // the dimensions of the matrix without its row and column header
   const int rows = -1 + nw.adjrows;
   const int cols = -1 + nw.adjcols;

   //  x x x x x x
   //  x / / / / /
   //  x / / / / /
   //  x / / / / /
   for( int i = 0; i < rows; i++ )
   for( int j = 0; j < cols; j++ )
   {
      UpdateScore( nw, 1+i, 1+j );
   }

   // stop the timer
   res.sw.lap( "cpu-end" );
   res.Tcpu = res.sw.dt( "cpu-end", "cpu-start" );
}







