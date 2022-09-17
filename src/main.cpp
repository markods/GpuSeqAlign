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

// AlgorithmData algData;
   NwSubstData substData;
   NwParamData paramData;
   NwSeqData seqData;
   std::ofstream ofsLog;

   if( NwStat::success != readFromJson( substPath, substData ) )
   {
      std::cerr << "ERR - could not open/read json from substs file"; exit( -1 );
   }
   if( NwStat::success != readFromJson( paramPath, paramData ) )
   {
      std::cerr << "ERR - could not open/read json from params file"; exit( -1 );
   }
   if( NwStat::success != readFromJson( seqPath,   seqData   ) )
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
   const int sm_count = deviceProps.multiProcessorCount;   // 28 on GTX 1080Ti
   const int warpsz   = deviceProps.warpSize;              // 32 on GTX 1080Ti


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
      // sm_count;
      // warpsz;
   };

   // initialize the device parameters
   nw.sm_count = sm_count;
   nw.warpsz = warpsz;

   // initialize the substitution matrix on the cpu and gpu
   {
      nw.subst = substData.substs[ seqData.substName ];
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
   nw.indel = seqData.indel;
   // initialize the letter map
   std::map<std::string, int> letterMap = substData.letterMap;
   
   // initialize the sequence map
   std::vector< std::vector<int> > seqMap {};
   for( auto& charSeq : seqData.seqs )
   {
      auto seq = seqStrToVect( charSeq, letterMap, true/*addHeader*/ );
      seqMap.push_back( seq );
   }



   // for all algorithms which have parameters in the param map
   for( auto& paramTuple: paramData.params )
   {
      // get the current algorithm parameters
      const std::string& algName = paramTuple.first;
      NwParams algParams = paramTuple.second;

      // if the current algorithm doesn't exist, skip it
      if( algData.algs.find( algName ) == algData.algs.end() )
      {
         continue;
      }

      // get the current algorithm and initialize its parameters
      NwAlgorithm alg = algData.algs[ algName ];
      alg.init( algParams );


      // for all Y sequences
      for( int iY = 0; iY < seqMap.size(); iY++ )
      {
         // get the Y sequence
         nw.seqY = seqMap[ iY ];
         // NOTE: the padding (zeroth element) was already added to the sequence
         nw.adjrows = nw.seqY.size();


         // for all X sequences (also compare every sequence with itself)
         for( int iX = iY; iX < seqMap.size(); iX++ )
         {
            // get the X sequence
            nw.seqX = seqMap[ iX ];
            // NOTE: the padding (zeroth element) was already added to the sequence
            nw.adjcols = nw.seqX.size();

            // if the number of columns is less than the number of rows, swap them and the sequences
            if( nw.adjcols < nw.adjrows )
            {
               std::swap( nw.adjcols, nw.adjrows );
               std::swap( nw.seqX, nw.seqY );
            }


            // for all parameter combinations
            for( ;   alg.alignPr().hasCurr();   alg.alignPr().next() )
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
               std::cout << std::setw( 2) << std::right << iY << " "
                         << std::setw( 2) << std::right << iX << "   "
                         << std::setw(20) << std::left  << algName << "   ";

               if( !errstep ) { std::cout << std::setw(10) << std::right << res.score_hash                       << std::endl; }
               else           { std::cout << "<STEP_" << errstep << " FAILED WITH STAT_" << ( (int)stat ) << ">" << std::endl; }

               // clear cuda non-sticky errors
               cudaStatus = cudaGetLastError();

               // get possible cuda sticky errors
               cudaStatus = cudaGetLastError();
               if( cudaStatus != cudaSuccess )
               {
                  std::cerr << "ERR - corrupted cuda context"; exit( -1 );
               }

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


