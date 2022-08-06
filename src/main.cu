/*
./build -run  2048  2048
./build -run  6144  6144
./build -run 16384 16384
./build -run 22528 22528
*/

#include <cstdio>
#include <string>
#include <map>

#include "common.hpp"
// === NW CPU ===
#include "nw-cpu1-row-st.hpp"
#include "nw-cpu2-diag-st.hpp"
#include "nw-cpu3-diagrow-st.hpp"
#include "nw-cpu4-diagrow-mt.hpp"
// === NW GPU ===
// #include "nw-gpu1.hpp"
// #include "nw-gpu2.hpp"
#include "nw-gpu3-diagdiag-coop.cuh"
// #include "nw-gpu4.hpp"
// === UTILS ===
#include "trace1-diag.hpp"
#include "updatescore1-simple.hpp"
#include "updatescore2-incremental.hpp"


// TODO: read from file
// block substitution matrix
#define SUBSTSZ 24
static int subst_tmp[SUBSTSZ*SUBSTSZ] =
{
    4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4,
   -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4,
   -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4,
   -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4,
    0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4,
   -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4,
   -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
    0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4,
   -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4,
   -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4,
   -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4,
   -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4,
   -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4,
   -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4,
   -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4,
    1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4,
    0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4,
   -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4,
   -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4,
    0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4,
   -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4,
   -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
    0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4,
   -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1,
};


// call in case of invalid command line arguments
void PrintHelpInfo( char* argv[] )
{
   fprintf( stderr,
      "nw m n\n"
      "   m    - length of the first generated sequence\n"
      "   n    - length of the second generated sequence\n"
   );
   fflush( stderr );
   exit( 0 );
}


// main program
int main( int argc, char *argv[] )
{
   fflush( stdout );
   if( argc != 3 ) PrintHelpInfo( argv );

   // number of rows, number of columns and insdelcost
   int rows = atoi( argv[1] );
   int cols = atoi( argv[2] );
   int insdelcost = 4; // TODO: should this be input, or not?
   // add the padding (zeroth row and column) to the matrix
   rows++; cols++;
   // if the number of columns is less than the number of rows, swap them
   if( cols < rows ) { std::swap( rows, cols ); }

   // allocate memory for the sequences which will be compared and the score matrix
   int* const seqX  = ( int* ) malloc( cols * sizeof( int ) );
   int* const seqY  = ( int* ) malloc( rows * sizeof( int ) );
   int* const score = ( int* ) malloc( rows*cols * sizeof( int ) );

   // if memory hasn't been allocated
   if( !seqX || !seqY || !score )
   {
      fprintf(stderr, "Error: memory allocation failed\n");
      fflush(stderr);

      // free allocated memory
      free( seqX ); free( seqY ); free( score );
      exit(1);
   }

   // seed the random generator
   unsigned int seed = time( NULL );
// unsigned int seed = 1605868371;
   srand( seed );

   // initialize the sequences A and B to random values in the range [0, SUBSTSIZE-1]
   // +   also initialize the padding with zeroes
   seqX[0] = 0;
   seqY[0] = 0;
   for( int j = 1; j < cols; j++ ) seqX[j] = ( j < cols ) ? ( rand() % SUBSTSZ ) : 0;
   for( int i = 1; i < rows; i++ ) seqY[i] = ( i < rows ) ? ( rand() % SUBSTSZ ) : 0;

   Stopwatch sw {};

   std::map<std::string, NwVariant> variants {
      { "Nw_Cpu1_Row_St", Nw_Cpu1_Row_St },
      { "Nw_Cpu2_Diag_St", Nw_Cpu2_Diag_St },
      { "Nw_Cpu3_DiagRow_St", Nw_Cpu3_DiagRow_St },
      { "Nw_Cpu4_DiagRow_Mt", Nw_Cpu4_DiagRow_Mt },
      { "Nw_Gpu3_DiagDiag_Coop", Nw_Gpu3_DiagDiag_Coop },
   };

   NwInput nw {
      seqX,
      seqY,
      score,
      subst_tmp,

      rows,
      cols,

      insdelcost,
   };

   // variables for storing the calculation hashes
   unsigned prevhash = 10;
   // if the test was successful
   bool firstIter = true;
   bool success = true;

   for( auto& variant_iter: variants )
   {
      const std::string& name = variant_iter.first;
      NwVariant& variant = variant_iter.second;

      NwMetrics res {};

      printf( "%-22s:   ", name.c_str() );
      fflush( stdout );

      variant( nw, res );

      Trace1_Diag( nw, res );
      if( firstIter )
      {
         prevhash = res.hash;
         firstIter = false;
      }
      else if( prevhash != res.hash )
      {
         success = false;
      }

      printf( "hash=%10u   Tcpu=%6.3fs   Tgpu=%6.3fs\n", res.hash, res.Tcpu, res.Tgpu );
      fflush( stdout );
   }

   // +   compare the implementations
   if( success ) printf( "TEST PASSED\n" );
   else          printf( "TEST FAILED\n" );
   fflush(stdout);

   // free allocated memory
   free( seqX ); free( seqY ); free( score );
}


