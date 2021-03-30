#include <cstdio>
#include <cmath>
#include "omp.h"
#include "Common.h"


// parallel cpu implementation of the Needleman Wunsch algorithm
int CpuParallel( const int* seqX, const int* seqY, int* score, int rows, int cols, int adjrows, int adjcols, int insdelcost, float* time )
{
   // skip the first row and first column in the next calculations
   rows--; cols--;

   // initialize the first row and column of the score matrix
   #pragma omp parallel
   {
      #pragma omp for schedule( static ) nowait
      for( int i = 0; i < 1+rows; i++ ) el(score,cols, i,0) = -i*insdelcost;
      #pragma omp for schedule( static )
      for( int j = 0; j < 1+cols; j++ ) el(score,cols, 0,j) = -j*insdelcost;

      #pragma omp single
      {
         printf("   - processing score matrix in a blocky diagonal fashion\n");
      }
      
      // size of block that will be a unit of work
      // +   8*16 ints on standard architectures, or 8 cache lines
      const int bsize = 8 * 64/*B*//sizeof( int );
      
      // number of blocks in a row and column (rounded up)
      const int brows = ceil( 1.*rows / bsize );
      const int bcols = ceil( 1.*cols / bsize );


      //  / / / . .   +   . . . / /   +   . . . . .|/ /
      //  / / . . .   +   . . / / .   +   . . . . /|/
      //  / . . . .   +   . / / . .   +   . . . / /|
      for( int s = 0; s < bcols-1 + brows; s++ )
      {
         int tbeg = max2( 0, s - (bcols-1) );
         int tend = min2( s, brows-1 );

         #pragma omp for schedule( static )
         for( int t = tbeg; t <= tend; t++ )
         {
            // calculate the block boundaries
            int ibeg = 1 + (   t )*bsize;
            int jbeg = 1 + ( s-t )*bsize;

            int iend = min2( ibeg + bsize, 1+rows );
            int jend = min2( jbeg + bsize, 1+cols );

            // process the block
            for( int i = ibeg; i < iend; i++ )
            for( int j = jbeg; j < jend; j++ )
            {
               UpdateScore( seqX, seqY, score, rows, cols, insdelcost, i, j );
            }
         }
      }
   }

   // restore the original row and column count
   rows++; cols++;
}
