#include <cstdio>
#include "omp.h"
#include "Common.h"


// sequential cpu implementation of the Needleman Wunsch algorithm
void CpuSequential( NWArgs& nw, NWResult& res )
{
   // start the timer
   res.Tcpu = omp_get_wtime();


   // skip the first row and first column in the next calculation
   nw.rows--; nw.cols--;

   // initialize the first row and column of the score matrix
   for( int i = 0; i < 1+nw.rows; i++ ) el(nw.score,nw.adjcols, i,0) = -i*nw.insdelcost;
   for( int j = 0; j < 1+nw.cols; j++ ) el(nw.score,nw.adjcols, 0,j) = -j*nw.insdelcost;

   //  / / / . .
   //  / / . . .
   //  / . . . .
   // printf("   - processing top-left triangle + first diagonal of the score matrix\n");
   for( int s = 0; s < nw.rows; s++ )
   for( int t = 0; t <= s; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( nw.seqX, nw.seqY, nw.score, nw.adjrows, nw.adjcols, nw.insdelcost, i, j );
   }

   //  . . . / /
   //  . . / / .
   //  . / / . .
   // if the matrix is not square shaped
   if( nw.rows != nw.cols )
   {
      // printf("   - processing all other diagonals of the score matrix\n");
      for( int s = nw.rows; s < nw.cols; s++ )
      for( int t = 0; t <= nw.rows-1; t++ )
      {
         int i = 1 +     t;
         int j = 1 + s - t;
         UpdateScore( nw.seqX, nw.seqY, nw.score, nw.adjrows, nw.adjcols, nw.insdelcost, i, j );
      }
   }

   //  . . . . .|/ /
   //  . . . . /|/
   //  . . . / /|
   // printf("   - processing bottom-right triangle of the score matrix\n");
   for( int s = nw.cols; s < nw.cols-1 + nw.rows; s++ )
   for( int t = s-nw.cols+1; t <= nw.rows-1; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( nw.seqX, nw.seqY, nw.score, nw.adjrows, nw.adjcols, nw.insdelcost, i, j );
   }

   // restore the original row and column count
   nw.rows++; nw.cols++;

   // stop the timer
   res.Tcpu = ( omp_get_wtime() - res.Tcpu );
}





