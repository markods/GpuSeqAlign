/*
./build-v2 -2
./build-v2 -3
./build-v2 -4
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <map>
#include "json.hpp"
#include <cuda_runtime.h>

#include "common.hpp"
#include "nw-algorithm.hpp"
using json = nlohmann::ordered_json;
using namespace std::string_literals;



// main program
int main( int argc, char *argv[] )
{
   // check if the arguments are valid
   if( argc != 4 )
   {
      std::cerr <<
         "ERR - invalid arguments\n"
         "nw fsubsts fparams fseqs\n"
         "   fsubst     json file with substitution matrices\n"
         "   fparams    json file with nw parameters\n"
         "   fseqs      json file with sequences to be compared\n";

      exit( -1 );
   }

   std::string projPath  = "../../";
   std::string logPath   = projPath + "log/" + IsoTime() + ".log";
   std::string substPath = projPath + "resrc/" + argv[ 1 ];
   std::string paramPath = projPath + "resrc/" + argv[ 2 ];
   std::string seqPath   = projPath + "resrc/" + argv[ 3 ];

// AlgorithmMap algorithmMap;
   NwSubstMap substMap;
   NwParamMap paramMap;
   NwSeqMap seqMap;
   std::ofstream ofsLog;

   if( NwStat::success != readFromJson( substPath, substMap ) )
   {
      std::cerr << "ERR - could not open/read json from substs file"; exit( -1 );
   }
   if( NwStat::success != readFromJson( paramPath, paramMap ) )
   {
      std::cerr << "ERR - could not open/read json from params file"; exit( -1 );
   }
   if( NwStat::success != readFromJson( seqPath,   seqMap   ) )
   {
      std::cerr << "ERR - could not open/read json from seqs file"; exit( -1 );
   }
   if( NwStat::success != openOutFile ( logPath,   ofsLog   ) )
   {
      std::cerr << "ERR - could not open output log file"; exit( -1 );
   }
   
   // get the device properties
   cudaDeviceProp deviceProps;
   if( cudaSuccess != ( cudaStatus = cudaGetDeviceProperties( &deviceProps, 0/*deviceId*/ ) ) )
   {
      std::cerr << "ERR - could not get device properties"; exit( -1 );
   }

   // number of streaming multiprocessors (sm-s) and threads in a warp
   const int MPROCS = deviceProps.multiProcessorCount;   // 28 on GTX 1080Ti
   const int WARPSZ = deviceProps.warpSize;              // 32 on GTX 1080Ti


   NwInput nw
   {
      ////// host specific memory
      // subst;   <-- once
      // seqX;    <-- loop-inited
      // seqY;    <-- loop-inited
      // score;   <-- algorithm-reserved
      
      ////// device specific memory
      // subst_gpu;   <-- once
      // seqX_gpu;    <-- algorithm-reserved
      // seqY_gpu;    <-- algorithm-reserved
      // score_gpu;   <-- algorithm-reserved

      ////// alignment parameters
      // substsz;   <-- once
      // adjrows;   <-- loop-inited
      // adjcols;   <-- loop-inited

      // indel;   <-- once

      ////// device parameters
      // MPROCS;
      // WARPSZ;
   };

   // initialize the device parameters
   nw.MPROCS = MPROCS;
   nw.WARPSZ = WARPSZ;

   // initialize the substitution matrix on the cpu and gpu
   {
      nw.subst = substMap.substs[ seqMap.substName ];
      nw.substsz = std::sqrt( nw.subst.size() );

      // reserve space in the gpu global memory
      try
      {
         nw.subst_gpu.init( nw.substsz*nw.substsz );
      }
      catch( const std::exception& ex )
      {
         std::cerr << "ERR - could not reserve space for the substitution matrix in the gpu"; exit( -1 );
      }

      // transfer the substitution matrix to the gpu global memory
      if( cudaSuccess != ( cudaStatus = memTransfer( nw.subst_gpu, nw.subst, nw.substsz*nw.substsz ) ) )
      {
         std::cerr << "ERR - could not transfer substitution matrix to the gpu"; exit( -1 );
      }
   }
   
   // initialize the indel cost
   nw.indel = seqMap.indel;
   // initialize the letter map
   std::map<std::string, int> letterMap = substMap.letterMap;



   // for all algorithms
   for( auto& algTuple: algorithmMap )
   {

      // get the current algorithm
      std::string algName = algTuple.first;
      NwAlgorithm alg     = algTuple.second;
      // get the algorithm params
      NwParams pr = paramMap.params[algName];

      // for all X sequences
      for( int iX = 0; iX < seqMap.seqs.size(); iX++ )
      {

         // get the X sequence
         nw.seqX = seqStrToVect( seqMap.seqs[ iX ], letterMap, true/*addHeader*/ );
         // NOTE: the padding (zeroth element) was already added to the sequence
         nw.adjcols = nw.seqX.size();

         // for all Y sequences
         for( int iY = iX+1; iY < seqMap.seqs.size(); iY++ )
         {
            // get the Y sequence
            nw.seqY = seqStrToVect( seqMap.seqs[ iY ], letterMap, true/*addHeader*/ );
            // NOTE: the padding (zeroth element) was already added to the sequence
            nw.adjrows = nw.seqY.size();

            // if the number of columns is less than the number of rows, swap them and the sequences
            if( nw.adjcols < nw.adjrows )
            {
               std::swap( nw.adjcols, nw.adjrows );
               std::swap( nw.seqX, nw.seqY );
            }

            // initialize the algorithm parameters
            alg.init( paramMap.params[ algName ] );

            // for all parameter combinations
            // TODO: restore
         // for( ;   alg.alignPr().hasCurr();   alg.alignPr().next() )
            {
               // initialize the result
               NwResult res {};
               
               // compare the sequences
               int errstep = 0;
               NwStat stat = NwStat::success;
               if( !errstep && NwStat::success != ( stat = alg.align( nw, res ) ) ) { errstep = 1; }
               if( !errstep && NwStat::success != ( stat = alg.hash ( nw, res ) ) ) { errstep = 2; }
               if( !errstep && NwStat::success != ( stat = alg.trace( nw, res ) ) ) { errstep = 3; }

               // TODO: print results to .json file
               // print the algorithm name and info
               FormatFlagsGuard fg { std::cout };
               std::cout << std::setw( 2) << std::right << iX << " "
                         << std::setw( 2) << std::right << iY << "   "
                         << std::setw(15) << std::left  << algName << "   ";

               if( !errstep ) { std::cout << std::setw(10) << std::right << res.score_hash                       << std::endl; }
               else           { std::cout << "<STEP_" << errstep << " FAILED WITH STAT_" << ( (int)stat ) << ">" << std::endl; }

               // reset allocations
               nw.resetAllocs();
            }

            // reset the algorithm parameters
            alg.alignPr().reset();
         }
      }
   }

   exit( 0 );
}


