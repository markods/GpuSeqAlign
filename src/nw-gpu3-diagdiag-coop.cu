#include <cooperative_groups.h>
#include "common.hpp"


// cuda kernel for the parallel implementation
__global__ static void Nw_Gpu3_Kernel(
   // nw input
   const int* const seqX_gpu,
   const int* const seqY_gpu,
         int* const score_gpu,
   const int* const subst_gpu,
// const int adjrows,   // can be calculated as 1 + trows*tileAy
// const int adjcols,   // can be calculated as 1 + tcols*tileAx
   const int substsz,
   const int indel,
   const int warpsz,
   // tile size
   const int trows,
   const int tcols,
   const unsigned tileAx,
   const unsigned tileAy
)
{
   extern __shared__ int shmem[/* substsz*substsz + tileAx + tileAy + (1+tileAy)*(1+tileAx) */];
   // the substitution matrix and relevant parts of the two sequences
   // TODO: align allocations to 0-th shared memory bank?
   int* const subst/*[substsz*substsz]*/      = shmem + 0;
   int* const seqX/*[tileAx]*/                = subst + substsz*substsz;
   int* const seqY/*[tileAy]*/                = seqX  + tileAx;
   int* const tile/*[(1+tileAy)*(1+tileAx)]*/ = seqY  + tileAy;

   // initialize the substitution shared memory copy
   {
      // map the threads from the thread block onto the substitution matrix elements
      int i = threadIdx.x;
      // while the current thread maps onto an element in the matrix
      while( i < substsz*substsz )
      {
         // copy the current element from the global substitution matrix
         el(subst,substsz, 0,i) = el(subst_gpu,substsz, 0,i);
         // map this thread to the next element with stride equal to the number of threads in this block
         i += blockDim.x;
      }
   }

   // initialize the global score's header row and column
   {
      // the number of rows in the score matrix
      int adjrows = 1 + trows*tileAy;
      // the number of columns in the score matrix
      int adjcols = 1 + tcols*tileAx;

      // map the threads from the thread grid onto the global score's header row elements
      // +   stride equal to the number of threads in this grid
      int j  = blockIdx.x*blockDim.x + threadIdx.x;
      int dj = gridDim.x *blockDim.x;
      // while the current thread maps onto an element in the header row
      while( j < adjcols )
      {
         // initialize that header row element
         el(score_gpu,adjcols, 0,j) = j*indel;

         // map this thread to the next element
         j += dj;
      }

      // map the threads from the thread grid onto the global score's header column elements
      // +   stride equal to the number of threads in this grid
      // NOTE: the zeroth element of the header column is skipped since it is already initialized
      int i  = 1+ blockIdx.x*blockDim.x + threadIdx.x;
      int di = gridDim.x *blockDim.x;
      // while the current thread maps onto an element in the header column
      while( i < adjrows )
      {
         // initialize that header column element
         el(score_gpu,adjcols, i,0) = i*indel;

         // map this thread to the next element
         i += di;
      }
   }

   // all threads in this grid should finish initializing their substitution shared memory + the global score's header row and column
   cooperative_groups::this_grid().sync();



   //  / / / . .       . . . / /       . . . . .|/ /
   //  / / . . .   +   . . / / .   +   . . . . /|/
   //  / . . . .       . / / . .       . . . / /|

   // for all diagonals of tiles in the grid of tiles (score matrix)
   for( int s = 0;   s < tcols-1 + trows;   s++ )
   {
      // (s,t) -- tile coordinates in the grid of tiles (score matrix)
      int tbeg = max( 0, s - (tcols-1) );
      int tend = min( s, trows-1 );


      // map a tile on the current diagonal of tiles to this thread block
      // +   then go to the next tile on the diagonal with stride equal to the number of thread blocks in the thread grid
      for( int t = tbeg + blockIdx.x;   t <= tend;   t += gridDim.x )
      {


         // initialize the tile's window into the global X and Y sequences
         {
            //       x x x x x
            //       | | | | |
            //     h h h h h h     // note the x and y seqences on this schematic
            // y --h u . . . .     // +   they don't! need to be extended by 1 to the left and by 1 to the top
            // y --h . . . . .
            // y --h . . . . .
            // position of the top left uninitialized! element <u> of the current tile in the score matrix
            // +   only the uninitialized elements will be calculated, and they need the corresponding global sequence X and Y elements
            int ibeg = 1 + (   t )*tileAy;
            int jbeg = 1 + ( s-t )*tileAx;

            // map the threads from the thread block onto the global X sequence's elements (which will be used in this tile)
            int j = threadIdx.x;
            // while the current thread maps onto an element in the tile's X sequence
            while( j < tileAx )
            {
               // initialize that element in the X seqence's shared window
               seqX[ j ] = seqX_gpu[ jbeg + j ];

               // map this thread to the next element with stride equal to the number of threads in this block
               j += blockDim.x;
            }

            // map the threads from the thread block onto the global Y sequence's elements (which will be used in this tile)
            int i = threadIdx.x;
            // while the current thread maps onto an element in the tile's Y sequence
            while( i < tileAy )
            {
               // initialize that element in the Y seqence's shared window
               seqY[ i ] = seqY_gpu[ ibeg + i ];

               // map this thread to the next element with stride equal to the number of threads in this block
               i += blockDim.x;
            }
         }

         // initialize the tile's header row and column
         {
            //       x x x x x
            //       | | | | |
            //     p h h h h h
            // y --h . . . . .
            // y --h . . . . .
            // y --h . . . . .
            // position of the top left element <p> of the current tile in the score matrix
            // +   start indexes from the header, since the tile header (<h>) should be copied from the global score matrix
            int ibeg = ( 1 + (   t )*tileAy ) - 1/*header*/;
            int jbeg = ( 1 + ( s-t )*tileAx ) - 1/*header*/;
            // the number of columns in the score matrix
            int adjcols = 1 + tcols*tileAx;

            // map the threads from the thread block onto the tile's header row (stored in the global score matrix)
            int j = threadIdx.x;
            // while the current thread maps onto an element in the tile's header row (stored in the global score matrix)
            while( j < 1+tileAx )
            {
               // initialize that element in the tile's shared memory
               el(tile,1+tileAx, 0,j) = el(score_gpu,adjcols, ibeg+0,jbeg+j);

               // map this thread to the next element with stride equal to the number of threads in this block
               j += blockDim.x;
            }

            // map the threads from the thread block onto the tile's header column (stored in the global score matrix)
            // +   skip the zeroth element since it is already initialized
            int i = 1+ threadIdx.x;
            // while the current thread maps onto an element in the tile's header column (stored in the global score matrix)
            while( i < 1+tileAy )
            {
               // initialize that element in the tile's shared memory
               el(tile,1+tileAx, i,0) = el(score_gpu,adjcols, ibeg+i,jbeg+0);

               // map this thread to the next element with stride equal to the number of threads in this block
               i += blockDim.x;
            }
         }

         // make sure that all threads have finished initializing their corresponding elements in the shared X and Y sequences, and the tile's header row and column sequences
         __syncthreads();
         
         // initialize the score matrix tile
         {
            //       x x x x x
            //       | | | | |
            //     p h h h h h
            // y --h . . . . .
            // y --h . . . . .
            // y --h . . . . .
            // position of the top left element <p> of the current tile in the score matrix
            
            // current thread position in the tile
            int i = threadIdx.x / tileAx;
            int j = threadIdx.x % tileAx;
            // stride on the current thread position in the tile, equal to the number of threads in this thread block
            // +   it is split into row and column increments for the thread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
            int di = blockDim.x / tileAx;
            int dj = blockDim.x % tileAx;
            
            // while the current thread maps onto an element in the tile
            while( i < tileAy )
            {
               // use the substitution matrix to partially calculate the score matrix element value
               // +   increase the value by insert delete cost, since then the formula for calculating the actual element value later on becomes simpler
               el(tile,1+tileAx, 1+i,1+j) = el(subst,substsz, seqY[i],seqX[j]) - indel;

               // map the current thread to the next tile element
               i += di; j += dj;
               // if the column index is out of bounds, increase the row index by one and wrap around the column index
               if( j >= tileAx ) { i++; j -= tileAx; }
            }
         }

         // all threads in this block should finish initializing this tile in shared memory
         __syncthreads();
         
         // calculate the tile elements
         // +   only threads in the first warp from this block are active here, other warps have to wait
         if( threadIdx.x < warpsz )
         {
            // the number of rows and columns in the tile without its first row and column (the part of the tile to be calculated)
            int rows = tileAy;
            int cols = tileAx;

            //  x x x x x x       x x x x x x       x x x x x x
            //  x / / / . .       x . . . / /       x . . . . .|/ /
            //  x / / . . .   +   x . . / / .   +   x . . . . /|/
            //  x / . . . .       x . / / . .       x . . . / /|

            // for all diagonals in the tile without its first row and column
            for( int d = 0;   d < cols-1 + rows;   d++ )
            {
               // (d,p) -- element coordinates in the tile
               int pbeg = max( 0, d - (cols-1) );
               int pend = min( d, rows-1 );
               // position of the current thread's element on the tile diagonal
               int p = pbeg + threadIdx.x;

               // if the thread maps onto an element on the current tile diagonal
               if( p <= pend )
               {
                  // position of the current element
                  int i = 1 + (   p );
                  int j = 1 + ( d-p );
                  
                  // calculate the current element's value
                  // +   always subtract the insert delete cost from the result, since the kernel A added that value to each element of the score matrix
                  int temp1              =      el(tile,1+tileAx, i-1,j-1) + el(tile,1+tileAx, i  ,j  );
                  int temp2              = max( el(tile,1+tileAx, i-1,j  ) , el(tile,1+tileAx, i  ,j-1) );
                  el(tile,1+tileAx, i,j) = max( temp1, temp2 ) + indel;
               }

               // all threads in this warp should finish calculating the tile's current diagonal
               __syncwarp();
            }
         }
         
         // all threads in this block should finish calculating this tile
         __syncthreads();
         

         // save the score matrix tile
         {
            // position of the first (top left) calculated element of the current tile in the score matrix
            int ibeg = ( 1 + (   t )*tileAy );
            int jbeg = ( 1 + ( s-t )*tileAx );
            // the number of columns in the score matrix
            int adjcols = 1 + tcols*tileAx;

            // current thread position in the tile
            int i = threadIdx.x / tileAx;
            int j = threadIdx.x % tileAx;
            // stride on the current thread position in the tile, equal to the number of threads in this thread block
            // +   it is split into row and column increments for the thread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
            int di = blockDim.x / tileAx;
            int dj = blockDim.x % tileAx;
            
            // while the current thread maps onto an element in the tile
            while( i < tileAy )
            {
               // copy the current element from the tile to the global score matrix
               el(score_gpu,adjcols, ibeg+i,jbeg+j) = el(tile,1+tileAx, 1+i,1+j );

               // map the current thread to the next tile element
               i += di; j += dj;
               // if the column index is out of bounds, increase the row index by one and wrap around the column index
               if( j >= tileAx ) { i++; j -= tileAx; }
            }
         }
         
         // all threads in this block should finish saving this tile
         __syncthreads();
      }

      // all threads in this grid should finish calculating the diagonal of tiles
      cooperative_groups::this_grid().sync();
   }
}



// parallel gpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Gpu3_DiagDiag_Coop( NwParams& pr, NwInput& nw, NwResult& res )
{
   // tile size for the kernel
   // +   tile A must have one dimension fixed to the number of threads in a warp
   unsigned tileAx;
   unsigned tileAy;

   // get the parameter values
   try
   {
      tileAx = pr["tileAx"].curr();
      tileAy = pr["tileAy"].curr();
   }
   catch( const std::out_of_range& ex )
   {
      return NwStat::errorInvalidValue;
   }

   // adjusted gpu score matrix dimensions
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile A size (in order to be evenly divisible)
   int adjrows = 1 + tileAy*ceil( float( nw.adjrows-1 )/tileAy );
   int adjcols = 1 + tileAx*ceil( float( nw.adjcols-1 )/tileAx );
   // special case when very small and very large sequences are compared
   if( adjrows == 1 ) { adjrows = 1 + tileAy; }
   if( adjcols == 1 ) { adjcols = 1 + tileAx; }

   // start the timer
   Stopwatch& sw = res.sw_align;
   sw.start();


   // reserve space in the ram and gpu global memory
   try
   {
      nw.seqX_gpu .init(         adjcols );
      nw.seqY_gpu .init( adjrows         );
      nw.score_gpu.init( adjrows*adjcols );

      nw.score    .init( nw.adjrows*nw.adjcols );
   }
   catch( const std::exception& ex )
   {
      return NwStat::errorMemoryAllocation;
   }

   // measure allocation time
   sw.lap( "alloc" );

   
   // copy data from host to device
   if( cudaSuccess != ( cudaStatus = memTransfer( nw.seqX_gpu, nw.seqX, nw.adjcols ) ) )
   {
      return NwStat::errorMemoryTransfer;
   }
   if( cudaSuccess != ( cudaStatus = memTransfer( nw.seqY_gpu, nw.seqY, nw.adjrows ) ) )
   {
      return NwStat::errorMemoryTransfer;
   }
   // also initialize padding, since it is used to access elements in the substitution matrix
   if( cudaSuccess != ( cudaStatus = memSet( nw.seqX_gpu, nw.adjcols, 0/*value*/ ) ) )
   {
      return NwStat::errorMemoryTransfer;
   }
   if( cudaSuccess != ( cudaStatus = memSet( nw.seqY_gpu, nw.adjrows, 0/*value*/ ) ) )
   {
      return NwStat::errorMemoryTransfer;
   }

   // measure memory transfer time
   sw.lap( "cpy-dev" );



   // launch kernel
   {
      // grid and block dimensions for kernel
      dim3 gridA {};
      dim3 blockA {};
      // the number of tiles per row and column of the score matrix
      int trows = ceil( float( adjrows-1 )/tileAy );
      int tcols = ceil( float( adjcols-1 )/tileAx );

      // calculate size of shared memory per block in bytes
      int shmemsz = (
         /*subst[]*/ nw.substsz*nw.substsz *sizeof( int )
         /*seqX[]*/ + tileAx *sizeof( int )
         /*seqY[]*/ + tileAy *sizeof( int )
         /*tile[]*/ + ( 1+tileAy )*( 1+tileAx ) *sizeof( int )
      );
      
      // calculate grid and block dimensions for kernel
      {
         // take the number of threads on the largest diagonal of the tile
         // +   multiply by the number of half warps in the larger dimension for faster writing to global gpu memory
         blockA.x = nw.warpsz * ceil( max( tileAy, tileAx )*2./nw.warpsz );

         // the maximum number of parallel blocks on a streaming multiprocessor
         int maxBlocksPerSm = 0;
         // number of threads per block that the kernel will be launched with
         int numThreads = blockA.x;

         // calculate the max number of parallel blocks per streaming multiprocessor
         if( cudaSuccess != ( cudaStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerSm, Nw_Gpu3_Kernel, numThreads, shmemsz ) ) )
         {
            return NwStat::errorKernelFailure;
         }
         // take the number of tiles on the largest score matrix diagonal as the only dimension
         // +   the number of cooperative blocks launched must not exceed the maximum possible number of parallel blocks on the device
         gridA.x = min( min( trows, tcols ), nw.sm_count*maxBlocksPerSm );
      }


      // create variables for gpu arrays in order to be able to take their addresses
      int* seqX_gpu = nw.seqX_gpu.data();
      int* seqY_gpu = nw.seqY_gpu.data();
      int* score_gpu = nw.score_gpu.data();
      int* subst_gpu = nw.subst_gpu.data();

      // group arguments to be passed to kernel
      void* kargs[]
      {
         &seqX_gpu,
         &seqY_gpu,
         &score_gpu,
         &subst_gpu,
         /*&adjrows,*/
         /*&adjcols,*/
         &nw.substsz,
         &nw.indel,
         &nw.warpsz,
         &trows,
         &tcols,
         &tileAx,
         &tileAy
      };
      
      // launch the kernel in the given stream (don't statically allocate shared memory)
      if( cudaSuccess != ( cudaStatus = cudaLaunchCooperativeKernel( ( void* )Nw_Gpu3_Kernel, gridA, blockA, kargs, shmemsz, nullptr/*stream*/ ) ) )
      {
         return NwStat::errorKernelFailure;
      }
   }

   // wait for the gpu to finish before going to the next step
   if( cudaSuccess != ( cudaStatus = cudaDeviceSynchronize() ) )
   {
      return NwStat::errorKernelFailure;
   }

   // measure calculation time
   sw.lap( "calc-1" );


   // save the calculated score matrix
   if( cudaSuccess != ( cudaStatus = memTransfer( nw.score, nw.score_gpu, nw.adjrows, nw.adjcols, adjcols ) ) )
   {
      return NwStat::errorMemoryTransfer;
   }

   // measure memory transfer time
   sw.lap( "cpy-host" );

   return NwStat::success;
}





