#include "common.hpp"

// sequential cpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Cpu1_Row_St( NwParams& pr, NwInput& nw, NwResult& res )
{
   // the dimensions of the matrix without its row and column header
   const int rows = -1 + nw.adjrows;
   const int cols = -1 + nw.adjcols;

   // start the timer
   Stopwatch& sw = res.sw_align;
   sw.start();


   // reserve space in the ram
   try
   {
      nw.score.init( nw.adjrows*nw.adjcols );
   }
   catch( const std::exception& ex )
   {
      return NwStat::errorMemoryAllocation;
   }

   // measure allocation time
   sw.lap( "alloc" );


   // initialize the first row and column of the score matrix
   for( int i = 0; i < nw.adjrows; i++ ) el(nw.score,nw.adjcols, i,0) = i*nw.indel;
   for( int j = 0; j < nw.adjcols; j++ ) el(nw.score,nw.adjcols, 0,j) = j*nw.indel;

   // measure header initialization time
   sw.lap( "init-hdr" );


   //  x x x x x x
   //  x / / / / /
   //  x / / / / /
   //  x / / / / /
   for( int i = 0; i < rows; i++ )
   for( int j = 0; j < cols; j++ )
   {
      UpdateScore( nw, 1+i, 1+j );
   }

   // measure calculation time
   sw.lap( "calc-1" );

   return NwStat::success;
}







