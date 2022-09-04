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
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <map>
#include "common.hpp"
#include "json.hpp"
using json = nlohmann::json;


// call in case of invalid command line arguments
void PrintHelpInfo( FILE* stream, char* argv[] )
{
   fprintf( stream,
      "nw m n\n"
      "   m    - length of the first generated sequence\n"
      "   n    - length of the second generated sequence\n"
   );
}

// get the current time as an ISO string
std::string IsoTime()
{
   auto now = std::chrono::system_clock::now();
   auto time_t = std::chrono::system_clock::to_time_t( now );

   std::stringstream ss;
   // https://en.cppreference.com/w/cpp/io/manip/put_time
   ss << std::put_time( std::localtime( &time_t ), "%Y%m%d_%H%M%S" );
   return ss.str();
}


// main program
int main( int argc, char *argv[] )
{
   if( argc != 3 )
   {
      PrintHelpInfo( stderr, argv );
      exit(-1);
   }

   std::map<std::string, std::vector<int>> subst_mats;

   // read the substitution matrixes from memory
   {
      std::ifstream ifs;
      ifs.open( "D:/---/.Marko/GpuNW/resrc/blosum.json", std::ios_base::in );
      
         // NOTE: the parser doesn't allow for trailing commas
         subst_mats = json::parse(
            ifs,
            /*callback*/ nullptr,
            /*allow exceptions*/ true,
            /*ignore_comments*/ true
         );
         
      ifs.close();
   }

   
   // initialize the needleman-wunsch algorithm inputs
   NwInput nw {
      // seqX,
      // seqY,
      // score,
      // subst,

      // adjrows,
      // adjcols,
      // substsz,

      // indelcost,
   };

   // number of rows, number of columns and indelcost
   // add the padding (zeroth row and column) to the matrix
   nw.adjrows = 1+ atoi( argv[1] );   if( nw.adjrows < 1 ) nw.adjrows = 1;
   nw.adjcols = 1+ atoi( argv[2] );   if( nw.adjcols < 1 ) nw.adjcols = 1;
   nw.substsz = sqrt( subst_mats["blosum62"].size() );
   nw.indelcost = -10;
   // if the number of columns is less than the number of rows, swap them
   if( nw.adjcols < nw.adjrows )
   {
      std::swap( nw.adjrows, nw.adjcols );
   }

   // allocate memory for the sequences which will be compared and the score matrix
   nw.seqX  = ( int* ) malloc( nw.adjcols * sizeof( int ) );
   nw.seqY  = ( int* ) malloc( nw.adjrows * sizeof( int ) );
   nw.score = ( int* ) malloc( nw.adjrows*nw.adjcols * sizeof( int ) );
   nw.subst = &subst_mats["blosum62"][0];

   // if memory hasn't been allocated
   if( !nw.seqX || !nw.seqY || !nw.score )
   {
      fprintf(stderr, "Error: memory allocation failed\n");
      fflush(stderr);

      // free allocated memory
      free( nw.seqX ); free( nw.seqY ); free( nw.score );
      exit(-1);
   }



   // seed the random generator
   unsigned int seed = time( NULL );
// unsigned int seed = 1605868371;
   srand( seed );

   // initialize the sequences A and B to random values in the range [0, SUBSTSIZE-1]
   // +   also initialize the padding with zeroes
   nw.seqX[0] = 0;
   nw.seqY[0] = 0;
   for( int j = 1; j < nw.adjcols; j++ ) nw.seqX[j] = rand() % nw.substsz;
   for( int i = 1; i < nw.adjrows; i++ ) nw.seqY[i] = rand() % nw.substsz;


   // output file
   // FILE* stream = fopen( ("../../log/" + IsoTime() + ".log" ).c_str(), "w" );

   // the tested nw implementations
   std::map<std::string, NwVariant> variants {
      { "Nw_Cpu1_Row_St", Nw_Cpu1_Row_St },
      { "Nw_Cpu2_Diag_St", Nw_Cpu2_Diag_St },
      { "Nw_Cpu3_DiagRow_St", Nw_Cpu3_DiagRow_St },
      { "Nw_Cpu4_DiagRow_Mt", Nw_Cpu4_DiagRow_Mt },
      { "Nw_Gpu1_Diag_Ml", Nw_Gpu1_Diag_Ml },
      { "Nw_Gpu2_DiagRow_Ml2K", Nw_Gpu2_DiagRow_Ml2K },
      { "Nw_Gpu3_DiagDiag_Coop", Nw_Gpu3_DiagDiag_Coop },
      { "Nw_Gpu4_DiagDiag_Coop2K", Nw_Gpu4_DiagDiag_Coop2K },
   // { "Nw_Gpu5_DiagDiagDiag_Ml", Nw_Gpu5_DiagDiagDiag_Ml },
   };

   // variables for storing the calculation hashes
   unsigned firsthash = 10;
   // if the test was successful
   bool firstIter = true;
   bool success = true;

   for( auto& variant_iter: variants )
   {
      const std::string& name = variant_iter.first;
      NwVariant& variant = variant_iter.second;

      NwMetrics res {};

      printf( "%-23s:   ", name.c_str() );
      fflush( stdout );

      variant( nw, res );
      // Trace1_Diag( nw, res );
      // PrintMatrix( stream, nw.score, nw.adjrows, nw.adjcols );
      HashAndZeroOutMatrix( nw.score, nw.adjrows, nw.adjcols, res.hash );

      if( firstIter )
      {
         firsthash = res.hash;
         firstIter = false;
      }
      else if( firsthash != res.hash )
      {
         success = false;
      }

      printf( "hash=%10u   Tcpu=%6.3fs   Tgpu=%6.3fs\n", res.hash, res.Tcpu, res.Tgpu );
      fflush( stdout );
   }
   // fclose( stream );

   // +   compare the implementations
   if( success ) printf( "TEST PASSED\n" );
   else          printf( "TEST FAILED\n" );
   fflush(stdout);

   // free allocated memory
   free( nw.seqX ); free( nw.seqY ); free( nw.score );
   exit( ( success == true ) ? 0 : 1 );
}


