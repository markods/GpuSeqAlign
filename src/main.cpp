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

   extern NwAlgorithmData algData;
   NwSubstData substData;
   NwParamData paramData;
   NwSeqData seqData;
   NwResData resData;
   std::ofstream ofsRes;

   resData.projPath  = "../../";
   resData.resrcPath = resData.projPath + "resrc/";
   resData.resPath   = resData.projPath + "log/";

   resData.isoTime    = IsoTime();
   resData.substFname = argv[ 1 ];
   resData.paramFname = argv[ 2 ];
   resData.seqFname   = argv[ 3 ];
   resData.resFname   = resData.isoTime + ".csv";

   // read data from input .json files
   // +   also open the output file
   {
      std::string substPath = resData.resrcPath + resData.substFname;
      std::string paramPath = resData.resrcPath + resData.paramFname;
      std::string seqPath   = resData.resrcPath + resData.seqFname;
      std::string resPath = resData.resPath + resData.resFname;

      if( NwStat::success != readFromJson( substPath, substData ) )
      {
         std::cerr << "ERR - could not open/read json from substs file"; exit( -1 );
      }
      if( NwStat::success != readFromJson( paramPath, paramData ) )
      {
         std::cerr << "ERR - could not open/read json from params file"; exit( -1 );
      }
      if( NwStat::success != readFromJson( seqPath, seqData ) )
      {
         std::cerr << "ERR - could not open/read json from seqs file"; exit( -1 );
      }
      if( NwStat::success != openOutFile( resPath, ofsRes ) )
      {
         std::cerr << "ERR - could not open csv results file"; exit( -1 );
      }
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
      nw.subst = substData.substMap[ seqData.substName ];
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
   std::vector< std::vector<int> > seqList {};
   for( auto& charSeq : seqData.seqList )
   {
      auto seq = seqStrToVect( charSeq, letterMap, true/*addHeader*/ );
      seqList.push_back( seq );
   }

   // write the csv file's header
   resHeaderToCsv( ofsRes, resData );
   // if the csv header should be written to stderr
   bool writeCsvHeaderToStderr = true;



   // for all algorithms which have parameters in the param map
   for( auto& paramTuple: paramData.paramMap )
   {
      // get the current algorithm parameters
      const std::string& algName = paramTuple.first;
      NwParams& algParams = paramTuple.second;

      // if the current algorithm doesn't exist, skip it
      if( algData.algMap.find( algName ) == algData.algMap.end() )
      {
         continue;
      }

      // get the current algorithm and initialize its parameters
      NwAlgorithm& alg = algData.algMap[ algName ];
      alg.init( algParams );


      // for all Y sequences + for all X sequences (also compare every sequence with itself)
      for( int iY = 0; iY < seqList.size(); iY++ )
      for( int iX = iY; iX < seqList.size(); iX++ )
      {
         // get the Y sequence
         // NOTE: the padding (zeroth element) was already added to the sequence
         nw.seqY = seqList[ iY ];
         nw.adjrows = nw.seqY.size();

         // get the X sequence
         // NOTE: the padding (zeroth element) was already added to the sequence
         nw.seqX = seqList[ iX ];
         nw.adjcols = nw.seqX.size();

         // if the number of columns is less than the number of rows, swap them and the sequences
         if( nw.adjcols < nw.adjrows )
         {
            std::swap( nw.adjcols, nw.adjrows );
            std::swap( nw.seqX, nw.seqY );
         }


         // for all parameter combinations + for all requested repeats
         for( ;   alg.alignPr().hasCurr();   alg.alignPr().next() )
         for( int iR = 0; iR < seqData.repeat; iR++ )
         {
            // initialize the result in the result list
            resData.resList.push_back( NwResult {
               algName,   // algName;
               alg.alignPr().snapshot(), // algParams;

               nw.seqX.size(), // seqX_len;
               nw.seqY.size(), // seqY_len;

               iX, // iX;
               iY, // iY;
               iR, // iR;

               {}, // sw_align;
               {}, // sw_hash;
               {}, // sw_trace;

               {}, // score_hash;
               {}, // trace_hash;

               {}, // stat;
               {}, // errstep;   // 0 for success
            });
            // get the result from the list
            NwResult& res = resData.resList.back();

            // compare the sequences, hash the score matrices and trace the score matrices
            if( !res.errstep && NwStat::success != ( res.stat = alg.align( nw, res ) ) ) { res.errstep = 1; }
            if( !res.errstep && NwStat::success != ( res.stat = alg.hash ( nw, res ) ) ) { res.errstep = 2; }
            if( !res.errstep && NwStat::success != ( res.stat = alg.trace( nw, res ) ) ) { res.errstep = 3; }

            // verify that the hashes match the first ever calculated hash
            if( !res.errstep )
            {
               // // TODO: verify that the hashes match
               // res.errstep = 4;
            }

            // if there is an error in any step
            if( res.errstep )
            {
               // add the csv header to stderr
               if( writeCsvHeaderToStderr )
               {
                  // write the csv file's header
                  resHeaderToCsv( std::cerr, resData );
                  writeCsvHeaderToStderr = false;
               }

               // print the result to stderr
               to_csv( std::cerr, res ); std::cerr << std::endl;
            }
            
            // print the result as a csv line to the csv output file
            to_csv( ofsRes, res ); ofsRes << '\n';

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

   exit( 0 );
}


