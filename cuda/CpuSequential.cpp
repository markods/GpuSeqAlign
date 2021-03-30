#include <cstdio>
#include "omp.h"
#include "Common.h"


// sequential cpu implementation of the Needleman Wunsch algorithm
int CpuSequential( const int* seqX, const int* seqY, int* score, int rows, int cols, int adjrows, int adjcols, int insdelcost, float* time )
{
   // check if the given input is valid, if not return
   if( !seqX || !seqY || !score || !time ) return false;
   if( rows <= 1 || cols <= 1 ) return false;

   // start the timer
   *time = omp_get_wtime();


   // skip the first row and first column in the next calculation
   rows--; cols--;

   // initialize the first row and column of the score matrix
   for( int i = 0; i < 1+rows; i++ ) el(score,adjcols, i,0) = -i*insdelcost;
   for( int j = 0; j < 1+cols; j++ ) el(score,adjcols, 0,j) = -j*insdelcost;

   //  / / / . .
   //  / / . . .
   //  / . . . .
   printf("   - processing top-left triangle + first diagonal of the score matrix\n");
   for( int s = 0; s < rows; s++ )
   for( int t = 0; t <= s; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( seqX, seqY, score, adjrows, adjcols, insdelcost, i, j );
   }

   //  . . . / /
   //  . . / / .
   //  . / / . .
   // if the matrix is not square shaped
   if( rows != cols )
   {
      printf("   - processing all other diagonals of the score matrix\n");
      for( int s = rows; s < cols; s++ )
      for( int t = 0; t <= rows-1; t++ )
      {
         int i = 1 +     t;
         int j = 1 + s - t;
         UpdateScore( seqX, seqY, score, adjrows, adjcols, insdelcost, i, j );
      }
   }

   //  . . . . .|/ /
   //  . . . . /|/
   //  . . . / /|
   printf("   - processing bottom-right triangle of the score matrix\n");
   for( int s = cols; s < cols-1 + rows; s++ )
   for( int t = s-cols+1; t <= rows-1; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( seqX, seqY, score, adjrows, adjcols, insdelcost, i, j );
   }

   // restore the original row and column count
   rows++; cols++;

   // stop the timer
   *time = ( omp_get_wtime() - *time );
   // return that the operation is successful
   return true;
}





