#include "common.hpp"

// sequential cpu implementation of the Needleman Wunsch algorithm
void Nw_Cpu2_Diag_St( NwInput& nw, NwMetrics& res )
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
   //  x / / / . .
   //  x / / . . .
   //  x / . . . .
   for( int s = 0; s < rows; s++ )
   for( int t = 0; t <= s; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( nw.seqX, nw.seqY, nw.score, nw.subst, nw.adjcols, nw.insdelcost, i, j );
   }

   //  x x x x x x
   //  x . . . / /
   //  x . . / / .
   //  x . / / . .
   // if the matrix is not square shaped
   if( rows != cols )
   {
      for( int s = rows; s < cols; s++ )
      for( int t = 0; t <= rows-1; t++ )
      {
         int i = 1 +     t;
         int j = 1 + s - t;
         UpdateScore( nw.seqX, nw.seqY, nw.score, nw.subst, nw.adjcols, nw.insdelcost, i, j );
      }
   }

   //  x x x x x x
   //  x . . . . .|/ /
   //  x . . . . /|/
   //  x . . . / /|
   for( int s = cols; s < cols-1 + rows; s++ )
   for( int t = s-cols+1; t <= rows-1; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( nw.seqX, nw.seqY, nw.score, nw.subst, nw.adjcols, nw.insdelcost, i, j );
   }

   // stop the timer
   res.sw.lap( "cpu-end" );
   res.Tcpu = res.sw.dt( "cpu-end", "cpu-start" );
}





