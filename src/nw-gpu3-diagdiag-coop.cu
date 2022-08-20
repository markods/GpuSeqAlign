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
   const int indelcost,
   // tile size
   const int trows,
   const int tcols,
   const unsigned tileAx,
   const unsigned tileAy
)
{
   extern __shared__ int shmem[/* max2( substsz*substsz + tileAx + tileAy, (1+tileAy)*(1+tileAx) ) */];
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
         el(score_gpu,adjcols, 0,j) = -j*indelcost;

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
         el(score_gpu,adjcols, i,0) = -i*indelcost;

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
               el(tile,1+tileAx, 1+i,1+j) = el(subst,substsz, seqY[i],seqX[j]) + indelcost;

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
         if( threadIdx.x < WARPSZ )
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
                  el(tile,1+tileAx, i,j) = max( temp1, temp2 ) - indelcost;
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



// parallel gpu implementation of the Needleman Wunsch algorithm
void Nw_Gpu3_DiagDiag_Coop( NwInput& nw, NwMetrics& res )
{
   // tile size for the kernel
   // +   tile A must have one dimension fixed to the number of threads in a warp
   unsigned tileAx = 40;
   unsigned tileAy = WARPSZ;

   // substitution matrix, sequences which will be compared and the score matrix stored in gpu global memory
   NwInput nw_gpu = {
      // nw.seqX,
      // nw.seqY,
      // nw.score,
      // nw.subst,

      // nw.adjrows,
      // nw.adjcols,
      // nw.substsz,

      // nw.indelcost,
   };

   // adjusted gpu score matrix dimensions
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile A size (in order to be evenly divisible)
   nw_gpu.adjrows = 1 + tileAy*ceil( float( nw.adjrows-1 )/tileAy );
   nw_gpu.adjcols = 1 + tileAx*ceil( float( nw.adjcols-1 )/tileAx );
   nw_gpu.substsz = nw.substsz;
   nw_gpu.indelcost = nw.indelcost;

   // allocate space in the gpu global memory
   cudaMalloc( &nw_gpu.seqX,  nw_gpu.adjcols                * sizeof( int ) );
   cudaMalloc( &nw_gpu.seqY,  nw_gpu.adjrows                * sizeof( int ) );
   cudaMalloc( &nw_gpu.score, nw_gpu.adjrows*nw_gpu.adjcols * sizeof( int ) );
   cudaMalloc( &nw_gpu.subst, nw_gpu.substsz*nw_gpu.substsz * sizeof( int ) );
   // create events for measuring kernel execution time
   cudaEvent_t start, stop;
   cudaEventCreate( &start );
   cudaEventCreate( &stop );

   // start the host timer and initialize the gpu timer
   res.sw.lap( "cpu-start" );
   res.Tgpu = 0;

   // copy data from host to device
   // +   gpu padding remains uninitialized, but this is not an issue since padding is only used to simplify kernel code (optimization)
   cudaMemcpy( nw_gpu.seqX,  nw.seqX,  nw.adjcols * sizeof( int ), cudaMemcpyHostToDevice );
   cudaMemcpy( nw_gpu.seqY,  nw.seqY,  nw.adjrows * sizeof( int ), cudaMemcpyHostToDevice );
   cudaMemcpy( nw_gpu.subst, nw.subst, nw_gpu.substsz*nw_gpu.substsz * sizeof( int ), cudaMemcpyHostToDevice );


   // launch kernel
   {
      // grid and block dimensions for kernel
      dim3 gridA {};
      dim3 blockA {};
      // the number of tiles per row and column of the score matrix
      int trows = ceil( float( nw_gpu.adjrows-1 )/tileAy );
      int tcols = ceil( float( nw_gpu.adjcols-1 )/tileAx );

      // calculate size of shared memory per block in bytes
      int shmemsz = (
         /*subst[]*/ nw_gpu.substsz*nw_gpu.substsz *sizeof( int )
         /*seqX[]*/ + tileAx                       *sizeof( int )
         /*seqY[]*/ + tileAy                       *sizeof( int )
         /*tile[]*/ + ( 1+tileAy )*( 1+tileAx )    *sizeof( int )
      );
      
      // calculate grid and block dimensions for kernel
      {
         // take the number of threads on the largest diagonal of the tile
         // +   multiply by the number of half warps in the larger dimension for faster writing to global gpu memory
         blockA.x = WARPSZ * ceil( max( tileAy, tileAx )*2./WARPSZ );
         // take the number of tiles on the largest score matrix diagonal as the only dimension
         gridA.x = min( trows, tcols );

         // the maximum number of parallel blocks on a streaming multiprocessor
         int maxBlocksPerSm = 0;
         // number of threads per block that the kernel will be launched with
         int numThreads = blockA.x;

         // calculate the max number of parallel blocks per streaming multiprocessor
         cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerSm, Nw_Gpu3_Kernel, numThreads, shmemsz );
         // the number of cooperative blocks launched must not exceed the maximum possible number of parallel blocks on the device
         gridA.x = min( gridA.x, MPROCS*maxBlocksPerSm );
      }


      // group arguments to be passed to kernel
      void* kargs[] { &nw_gpu.seqX, &nw_gpu.seqY, &nw_gpu.score, &nw_gpu.subst, /*&nw_gpu.adjrows,*/ /*&nw_gpu.adjcols,*/ &nw_gpu.substsz, &nw_gpu.indelcost, &trows, &tcols, &tileAx, &tileAy };
      
      // launch the kernel in the given stream (don't statically allocate shared memory)
      // +   capture events around kernel launch as well
      // +   update the stop event when the kernel finishes
      cudaEventRecord( start, 0/*stream*/ );
      cudaLaunchCooperativeKernel( ( void* )Nw_Gpu3_Kernel, gridA, blockA, kargs, shmemsz, 0/*stream*/ );
      cudaEventRecord( stop, 0/*stream*/ );
      cudaEventSynchronize( stop );
      
      // kernel execution time
      float ktime {};
      // calculate the time between the given events
      cudaEventElapsedTime( &ktime, start, stop ); ktime /= 1000./*ms*/;
      // update the total kernel execution time
      res.Tgpu += ktime;
   }

   // wait for the gpu to finish before going to the next step
   cudaDeviceSynchronize();

   // \brief Copies data between host and device
   // 
   // Copies a matrix (\p height rows of \p width bytes each) from the memory
   // area pointed to by \p src to the memory area pointed to by \p dst, where
   // \p kind specifies the direction of the copy, and must be one of
   // ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
   // ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
   // ::cudaMemcpyDefault is recommended, in which case the type of transfer is
   // inferred from the pointer values. However, ::cudaMemcpyDefault is only
   // allowed on systems that support unified virtual addressing. \p dpitch and
   // \p spitch are the widths in memory in bytes of the 2D arrays pointed to by
   // \p dst and \p src, including any padding added to the end of each row. The
   // memory areas may not overlap. \p width must not exceed either \p dpitch or
   // \p spitch. Calling ::cudaMemcpy2D() with \p dst and \p src pointers that do
   // not match the direction of the copy results in an undefined behavior.
   // ::cudaMemcpy2D() returns an error if \p dpitch or \p spitch exceeds
   // the maximum allowed.

   // save the calculated score matrix
   // +   starts an async data copy from device to host, then waits for the copy to finish
   cudaMemcpy2D(
       nw    .score,                     // dst    - Destination memory address
       nw    .adjcols * sizeof( int ),   // dpitch - Pitch of destination memory (padded row size in bytes; in other words distance between the starting points of two rows)
       nw_gpu.score,                     // src    - Source memory address
       nw_gpu.adjcols * sizeof( int ),   // spitch - Pitch of source memory (padded row size in bytes)
       
       nw.adjcols * sizeof( int ),       // width  - Width of matrix transfer (non-padding row size in bytes)
       nw.adjrows,                       // height - Height of matrix transfer (#rows)
       cudaMemcpyDeviceToHost            // kind   - Type of transfer
   );      

   // stop the cpu timer
   res.sw.lap( "cpu-end" );
   res.Tcpu = res.sw.dt( "cpu-end", "cpu-start" );

   
   // free allocated space in the gpu global memory
   cudaFree( nw_gpu.seqX );
   cudaFree( nw_gpu.seqY );
   cudaFree( nw_gpu.score );
   cudaFree( nw_gpu.subst );
   // free events' memory
   cudaEventDestroy( start );
   cudaEventDestroy( stop );
}





