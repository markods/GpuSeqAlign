#include "common.hpp"

// sequential cpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Cpu2_Diag_St( NwParams& pr, NwInput& nw, NwResult& res )
{
   // the dimensions of the matrix without its row and column header
   const int rows = -1 + nw.adjrows;
   const int cols = -1 + nw.adjcols;

   // start the timer
   res.sw.start();


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
   res.sw.lap( "alloc" );


   // initialize the first row and column of the score matrix
   for( int i = 0; i < nw.adjrows; i++ ) el(nw.score,nw.adjcols, i,0) = i*nw.indel;
   for( int j = 0; j < nw.adjcols; j++ ) el(nw.score,nw.adjcols, 0,j) = j*nw.indel;

   // measure header initialization time
   res.sw.lap( "init-hdr" );


   //  x x x x x x
   //  x / / / . .
   //  x / / . . .
   //  x / . . . .
   for( int s = 0; s < rows; s++ )
   for( int t = 0; t <= s; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( nw, i, j );
   }

   // measure calculation time for the upper triangle
   res.sw.lap( "calc-upper" );


   //  x x x x x x
   //  x . . . / /
   //  x . . / / .
   //  x . / / . .
   // if the matrix is not square shaped
   if( rows != cols )
   {
      for( int s = rows; s < cols; s++ )
      for( int t = 0; t <= rows-1; t++ )
      {
         int i = 1 +     t;
         int j = 1 + s - t;
         UpdateScore( nw, i, j );
      }
   }

   // measure calculation time for the central parallelogram
   res.sw.lap( "calc-parallel" );


   //  x x x x x x
   //  x . . . . .|/ /
   //  x . . . . /|/
   //  x . . . / /|
   for( int s = cols; s < cols-1 + rows; s++ )
   for( int t = s-cols+1; t <= rows-1; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( nw, i, j );
   }

   // measure calculation time for the lower triangle
   res.sw.lap( "calc-lower" );

   return NwStat::success;
}





