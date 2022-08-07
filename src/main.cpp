/*
./build -run     1     1
./build -run     6     5
./build -run    32    32
./build -run    32    60
./build -run    33    61
./build -run  2050  2048
./build -run  6144  6150
./build -run 16384 16390
./build -run 22528 22550
*/

#include <cstdio>
#include <string>
#include <map>

#include "common.hpp"


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
   if( argc != 3 ) PrintHelpInfo( argv );

   // initialize the needleman-wunsch algorithm inputs
   NwInput nw {
      // seqX,
      // seqY,
      // score,
      // subst,

      // adjrows,
      // adjcols,

      // insdelcost,
   };

   // number of rows, number of columns and insdelcost
   // add the padding (zeroth row and column) to the matrix
   nw.adjrows = 1+ atoi( argv[1] );
   nw.adjcols = 1+ atoi( argv[2] );
   nw.insdelcost = 4; // TODO: should this be input, or not?
   // if the number of columns is less than the number of rows, swap them
   if( nw.adjcols < nw.adjrows )
   {
      std::swap( nw.adjrows, nw.adjcols );
      // std::swap( nw.inscost, nw.delcost );   // TODO: fix once insert and delete costs are added, instead of them being the same
   }

   // allocate memory for the sequences which will be compared and the score matrix
   nw.seqX  = ( int* ) malloc( nw.adjcols * sizeof( int ) );
   nw.seqY  = ( int* ) malloc( nw.adjrows * sizeof( int ) );
   nw.score = ( int* ) malloc( nw.adjrows*nw.adjcols * sizeof( int ) );
   nw.subst = subst_tmp;

   // if memory hasn't been allocated
   if( !nw.seqX || !nw.seqY || !nw.score )
   {
      fprintf(stderr, "Error: memory allocation failed\n");
      fflush(stderr);

      // free allocated memory
      free( nw.seqX ); free( nw.seqY ); free( nw.score );
      exit(1);
   }



   // seed the random generator
   unsigned int seed = time( NULL );
// unsigned int seed = 1605868371;
   srand( seed );

   // initialize the sequences A and B to random values in the range [0, SUBSTSIZE-1]
   // +   also initialize the padding with zeroes
   nw.seqX[0] = 0;
   nw.seqY[0] = 0;
   for( int j = 1; j < nw.adjcols; j++ ) nw.seqX[j] = rand() % SUBSTSZ;
   for( int i = 1; i < nw.adjrows; i++ ) nw.seqY[i] = rand() % SUBSTSZ;


   // the tested nw implementations
   std::map<std::string, NwVariant> variants {
      { "Nw_Cpu1_Row_St", Nw_Cpu1_Row_St },
      { "Nw_Cpu2_Diag_St", Nw_Cpu2_Diag_St },
      { "Nw_Cpu3_DiagRow_St", Nw_Cpu3_DiagRow_St },
      { "Nw_Cpu4_DiagRow_Mt", Nw_Cpu4_DiagRow_Mt },
      { "Nw_Gpu3_DiagDiag_Coop", Nw_Gpu3_DiagDiag_Coop },
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

      ZeroOutMatrix( nw.score, nw.adjrows, nw.adjcols );
      variant( nw, res );
      // PrintMatrix( nw.score, nw.adjrows, nw.adjcols );

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
   free( nw.seqX ); free( nw.seqY ); free( nw.score );
}


