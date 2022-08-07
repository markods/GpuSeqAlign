#include "common.hpp"

// parallel cpu implementation of the Needleman Wunsch algorithm
void Nw_Cpu3_DiagRow_St( NwInput& nw, NwMetrics& res )
{
   // size of square block (that will be a unit of work)
   // +   8*16 ints on standard architectures, or 8 cache lines
   const int blocksz = 8 * 64/*B*//sizeof( int );


   // start the timer
   res.sw.lap( "cpu-start" );

   // initialize the first row and column of the score matrix
   for( int i = 0; i < nw.adjrows; i++ ) el(nw.score,nw.adjcols, i,0) = -i*nw.insdelcost;
   for( int j = 0; j < nw.adjcols; j++ ) el(nw.score,nw.adjcols, 0,j) = -j*nw.insdelcost;

   // the dimensions of the matrix without its row and column header
   const int rows = -1 + nw.adjrows;
   const int cols = -1 + nw.adjcols;

   // number of blocks in a row and column (rounded up)
   const int rowblocks = ceil( float( rows ) / blocksz );
   const int colblocks = ceil( float( cols ) / blocksz );

   //  / / / . .       . . . / /       . . . . .|/ /
   //  / / . . .   +   . . / / .   +   . . . . /|/
   //  / . . . .       . / / . .       . . . / /|
   for( int s = 0; s < colblocks-1 + rowblocks; s++ )
   {
      int tbeg = max2( 0, s - (colblocks-1) );
      int tend = min2( s, rowblocks-1 );

      for( int t = tbeg; t <= tend; t++ )
      {
         // calculate the block boundaries
         int ibeg = 1 + (   t )*blocksz;
         int jbeg = 1 + ( s-t )*blocksz;

         int iend = min2( ibeg + blocksz, 1+rows );
         int jend = min2( jbeg + blocksz, 1+cols );

         // process the block
         for( int i = ibeg; i < iend; i++ )
         for( int j = jbeg; j < jend; j++ )
         {
            UpdateScore( nw, i, j );
         }
      }
   }

   // stop the timer
   res.sw.lap( "cpu-end" );
   res.Tcpu = res.sw.dt( "cpu-end", "cpu-start" );
}





