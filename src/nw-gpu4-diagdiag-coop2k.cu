#include <cooperative_groups.h>
#include "common.hpp"


// cuda kernel A for the parallel implementation
// +   initializes the score matrix in the gpu
__global__ static void Nw_Gpu4_KernelA(
   const int* const seqX_gpu,
   const int* const seqY_gpu,
         int* const score_gpu,
   const int* const subst_gpu,
   const int adjrows,
   const int adjcols,
   const int substsz,
   const int indelcost
)
{
   extern __shared__ int shmem[/* substsz*substsz + tileAx + tileAy */];
   // the substitution matrix and relevant parts of the two sequences
   // +   stored in shared memory for faster random access
   // TODO: align allocations to 0-th shared memory bank?
   int* const subst/*[substsz*substsz]*/ = shmem + 0;
   int* const seqX/*[tileAx]*/           = subst + substsz*substsz;
   int* const seqY/*[tileAy]*/           = seqX  + blockDim.x;

   // initialize the substitution shared memory copy
   {
      // map the threads from the thread block onto the substitution matrix elements
      int i = threadIdx.y*substsz + threadIdx.x;
      // while the current thread maps onto an element in the matrix
      while( i < substsz*substsz )
      {
         // copy the current element from the global substitution matrix
         el(subst,substsz, 0,i) = el(subst_gpu,substsz, 0,i);
         // map this thread to the next element with stride equal to the number of threads in this block
         i += blockDim.y*blockDim.x;
      }
   }

   // initialize the X and Y sequences' shared memory copies
   {
      // position of the current thread in the global X and Y sequences
      int x = blockIdx.x*blockDim.x;
      int y = blockIdx.y*blockDim.y;
      // map the threads from the first            row  to the shared X sequence part
      // map the threads from the second and later rows to the shared Y sequence part
      int iX = ( threadIdx.y     )*blockDim.x + threadIdx.x;
      int iY = ( threadIdx.y - 1 )*blockDim.x + threadIdx.x;

      // if the current thread maps to the first row, initialize the corresponding element
      if( iX < blockDim.x )        seqX[ iX ] = seqX_gpu[ x + iX ];
      // otherwise, remap it to the first column and initialize the corresponding element
      else if( iY < blockDim.y )   seqY[ iY ] = seqY_gpu[ y + iY ];
   }
   
   // make sure that all threads have finished initializing their corresponding elements
   __syncthreads();

   // initialize the score matrix in global memory
   {
      // position of the current thread in the score matrix
      int i = blockIdx.y*blockDim.y + threadIdx.y;
      int j = blockIdx.x*blockDim.x + threadIdx.x;
      // position of the current thread in the sequences
      int iX = threadIdx.x;
      int iY = threadIdx.y;
      // the current element value
      int elem = 0;
      
      // if the current thread is outside the score matrix, return
      if( i >= adjrows || j >= adjcols ) return;

      // if the current thread is not in the first row or column of the score matrix
      // +   use the substitution matrix to calculate the score matrix element value
      // +   increase the value by insert delete cost, since then the formula for calculating the actual element value in kernel B becomes simpler
      if( i > 0 && j > 0 ) { elem = el(subst,substsz, seqY[iY],seqX[iX]) - indelcost; }
      // otherwise, if the current thread is in the first row or column
      // +   update the score matrix element using the insert delete cost
      else                 { elem = ( i|j )*indelcost; }
      
      // update the corresponding element in global memory
      // +   fully coallesced memory access
      el(score_gpu,adjcols, i,j) = elem;
   }
}


// cuda kernel B for the parallel implementation
// +   calculates the score matrix in the gpu using the initialized score matrix from kernel A
// +   the given matrix minus the padding (zeroth row and column) must be evenly divisible by the tile B
__global__ static void Nw_Gpu4_KernelB(
         int* const score_gpu,
   const int indelcost,
   const int trows,
   const int tcols,
   const unsigned tileBx,
   const unsigned tileBy
)
{
   extern __shared__ int shmem[/* (1+tileBy)*(1+tileBx) */];
   // matrix tile which this thread block maps onto
   // +   stored in shared memory for faster random access
   int* const tile/*[(1+tileBy)*(1+tileBx)]*/ = shmem + 0;

   
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
         // initialize the score matrix tile
         {
            // position of the top left element of the current tile in the score matrix
            int ibeg = ( 1 + (   t )*tileBy ) - 1;
            int jbeg = ( 1 + ( s-t )*tileBx ) - 1;
            // the number of columns in the score matrix
            int adjcols = 1 + tcols*tileBx;

            // current thread position in the tile
            int i = threadIdx.x / ( tileBx+1 );
            int j = threadIdx.x % ( tileBx+1 );
            // stride on the current thread position in the tile, equal to the number of threads in this thread block
            // +   it is split into row and column increments for the thread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
            int di = blockDim.x / ( tileBx+1 );
            int dj = blockDim.x % ( tileBx+1 );
            
            // while the current thread maps onto an element in the tile
            while( i < ( 1+tileBy ) )
            {
               // copy the current element from the global score matrix to the tile
               el(tile,1+tileBx, i,j) = el(score_gpu,adjcols, ibeg+i,jbeg+j);

               // map the current thread to the next tile element
               i += di; j += dj;
               // if the column index is out of bounds, increase the row index by one and wrap around the column index
               if( j >= ( 1+tileBx ) ) { i++; j -= ( 1+tileBx ); }
            }
         }

         // all threads in this block should finish initializing this tile in shared memory
         __syncthreads();
         
         // calculate the tile elements
         // +   only threads in the first warp from this block are active here, other warps have to wait
         if( threadIdx.x < WARPSZ )
         {
            // the number of rows and columns in the tile without its first row and column (the part of the tile to be calculated)
            int rows = tileBy;
            int cols = tileBx;

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
                  int temp1              =      el(tile,1+tileBx, i-1,j-1) + el(tile,1+tileBx, i  ,j  );
                  int temp2              = max( el(tile,1+tileBx, i-1,j  ) , el(tile,1+tileBx, i  ,j-1) );
                  el(tile,1+tileBx, i,j) = max( temp1, temp2 ) + indelcost;
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
            int ibeg = ( 1 + (   t )*tileBy );
            int jbeg = ( 1 + ( s-t )*tileBx );
            // the number of columns in the score matrix
            int adjcols = 1 + tcols*tileBx;

            // current thread position in the tile
            int i = threadIdx.x / tileBx;
            int j = threadIdx.x % tileBx;
            // stride on the current thread position in the tile, equal to the number of threads in this thread block
            // +   it is split into row and column increments for the thread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
            int di = blockDim.x / tileBx;
            int dj = blockDim.x % tileBx;
            
            // while the current thread maps onto an element in the tile
            while( i < tileBy )
            {
               // copy the current element from the tile to the global score matrix
               el(score_gpu,adjcols, ibeg+i,jbeg+j) = el(tile,1+tileBx, 1+i,1+j );

               // map the current thread to the next tile element
               i += di; j += dj;
               // if the column index is out of bounds, increase the row index by one and wrap around the column index
               if( j >= tileBx ) { i++; j -= tileBx; }
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
void Nw_Gpu4_DiagDiag_Coop2K( NwInput& nw, NwMetrics& res )
{
   // tile sizes for kernels A and B
   // +   tile A should have one dimension be a multiple of the warp size for full memory coallescing
   // +   tile B must have one dimension fixed to the number of threads in a warp
   unsigned tileAx = 1*WARPSZ;
   unsigned tileAy = 32;
   unsigned tileBx = 60;
   unsigned tileBy = WARPSZ;

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
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile B size (in order to be evenly divisible)
   nw_gpu.adjrows = 1 + tileBy*ceil( float( nw.adjrows-1 )/tileBy );
   nw_gpu.adjcols = 1 + tileBx*ceil( float( nw.adjcols-1 )/tileBx );
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


   // launch kernel A
   {
      // calculate grid dimensions for kernel A
      dim3 gridA {};
      gridA.y = ceil( float( nw_gpu.adjrows )/tileAy );
      gridA.x = ceil( float( nw_gpu.adjcols )/tileAx );
      // block dimensions for kernel A
      dim3 blockA { tileAx, tileAy };

      // calculate size of shared memory per block in bytes
      int shmemsz = (
         /*subst[][]*/ nw_gpu.substsz*nw_gpu.substsz *sizeof( int )
         /*seqX[]*/ + tileAx *sizeof( int )
         /*seqY[]*/ + tileAy *sizeof( int )
      );
      
      // group arguments to be passed to kernel A
      void* kargs[] { &nw_gpu.seqX, &nw_gpu.seqY, &nw_gpu.score, &nw_gpu.subst, &nw_gpu.adjrows, &nw_gpu.adjcols, &nw_gpu.substsz, &nw_gpu.indelcost };

      // launch the kernel in the given stream (don't statically allocate shared memory)
      // +   capture events around kernel launch as well
      // +   update the stop event when the kernel finishes
      cudaEventRecord( start, 0/*stream*/ );
      cudaLaunchKernel( ( void* )Nw_Gpu4_KernelA, gridA, blockA, kargs, shmemsz, 0/*stream*/ );
      cudaEventRecord( stop, 0/*stream*/ );
      cudaEventSynchronize( stop );
      
      // kernel A execution time
      float ktime {};
      // calculate the time between the given events
      cudaEventElapsedTime( &ktime, start, stop ); ktime /= 1000./*ms*/;
      // update the total kernel execution time
      res.Tgpu += ktime;
   }
   
   // wait for the gpu to finish before going to the next step
   cudaDeviceSynchronize();


   // launch kernel B
   {
      // grid and block dimensions for kernel B
      dim3 gridB {};
      dim3 blockB {};
      // the number of tiles per row and column of the score matrix
      int trows = ceil( float( nw_gpu.adjrows-1 )/tileBy );
      int tcols = ceil( float( nw_gpu.adjcols-1 )/tileBx );
      
      // calculate size of shared memory per block in bytes
      int shmemsz = (
         /*tile[]*/ ( 1+tileBy )*( 1+tileBx ) *sizeof( int )
      );
      
      // calculate grid and block dimensions for kernel B
      {
         // take the number of threads on the largest diagonal of the tile
         // +   multiply by the number of half warps in the larger dimension for faster writing to global gpu memory
         blockB.x = WARPSZ * ceil( max( tileBy, tileBx )*2./WARPSZ );
         // take the number of tiles on the largest score matrix diagonal as the only dimension
         gridB.x = min( trows, tcols );

         // the maximum number of parallel blocks on a streaming multiprocessor
         int maxBlocksPerSm = 0;
         // number of threads per block that the kernel will be launched with
         int numThreads = blockB.x;

         // calculate the max number of parallel blocks per streaming multiprocessor
         cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerSm, Nw_Gpu4_KernelB, numThreads, shmemsz );
         // the number of cooperative blocks launched must not exceed the maximum possible number of parallel blocks on the device
         gridB.x = min( gridB.x, MPROCS*maxBlocksPerSm );
      }


      // group arguments to be passed to kernel B
      void* kargs[] { &nw_gpu.score, &nw_gpu.indelcost, &trows, &tcols, &tileBx, &tileBy };
      
      // launch the kernel in the given stream (don't statically allocate shared memory)
      // +   capture events around kernel launch as well
      // +   update the stop event when the kernel finishes
      cudaEventRecord( start, 0/*stream*/ );
      cudaLaunchCooperativeKernel( ( void* )Nw_Gpu4_KernelB, gridB, blockB, kargs, shmemsz, 0/*stream*/ );
      cudaEventRecord( stop, 0/*stream*/ );
      cudaEventSynchronize( stop );
      
      // kernel B execution time
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





