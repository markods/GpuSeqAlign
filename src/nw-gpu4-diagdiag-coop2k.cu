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
   const int indel
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
      if( i > 0 && j > 0 ) { elem = el(subst,substsz, seqY[iY],seqX[iX]) - indel; }
      // otherwise, if the current thread is in the first row or column
      // +   update the score matrix element using the insert delete cost
      else                 { elem = ( i|j )*indel; }
      
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
   const int indel,
   const int WARPSZ,
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
                  el(tile,1+tileBx, i,j) = max( temp1, temp2 ) + indel;
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



// parallel gpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Gpu4_DiagDiag_Coop2K( NwParams& pr, NwInput& nw, NwResult& res )
{
   // tile sizes for kernels A and B
   // +   tile A should have one dimension be a multiple of the warp size for full memory coallescing
   // +   tile B must have one dimension fixed to the number of threads in a warp
   unsigned tileAx;
   unsigned tileAy;
   unsigned tileBx;
   unsigned tileBy;

   // get the parameter values (this can throw)
   try
   {
      tileAx = pr["tileAx"].curr();
      tileAy = pr["tileAy"].curr();
      tileBx = pr["tileBx"].curr();
      tileBy = pr["tileBy"].curr();
   }
   catch( const std::out_of_range& ex )
   {
      return NwStat::errorInvalidValue;
   }

   // adjusted gpu score matrix dimensions
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile B size (in order to be evenly divisible)
   int adjrows = 1 + tileBy*ceil( float( nw.adjrows-1 )/tileBy );
   int adjcols = 1 + tileBx*ceil( float( nw.adjcols-1 )/tileBx );

   // clear cuda non-sticky errors
   cudaGetLastError();

   // start the timer
   res.sw.start();


   // reserve space in the ram and gpu global memory (this can throw)
   try
   {
      nw.seqX_gpu .init(              nw.adjcols );
      nw.seqY_gpu .init( nw.adjrows              );
      nw.score_gpu.init( nw.adjrows * nw.adjcols );
      nw.score    .init( nw.adjrows * nw.adjcols );
   }
   catch( const std::exception& ex )
   {
      return NwStat::errorMemoryAllocation;
   }

   // measure allocation time
   res.sw.lap( "alloc" );

   
   // copy data from host to device
   // +   gpu padding remains uninitialized, but this is not an issue since padding is only used to simplify kernel code (optimization)
   if( !memTransfer( nw.seqX_gpu, nw.seqX, nw.adjcols ) ) return NwStat::errorMemoryTransfer;
   if( !memTransfer( nw.seqY_gpu, nw.seqY, nw.adjrows ) ) return NwStat::errorMemoryTransfer;

   // measure memory transfer time
   res.sw.lap( "mem-to-device" );



   // launch kernel A
   {
      // calculate grid dimensions for kernel A
      dim3 gridA {};
      gridA.y = ceil( float( adjrows )/tileAy );
      gridA.x = ceil( float( adjcols )/tileAx );
      // block dimensions for kernel A
      dim3 blockA { tileAx, tileAy };

      // calculate size of shared memory per block in bytes
      int shmemsz = (
         /*subst[][]*/ nw.substsz*nw.substsz *sizeof( int )
         /*seqX[]*/ + tileAx *sizeof( int )
         /*seqY[]*/ + tileAy *sizeof( int )
      );
      

      // create variables for gpu arrays in order to be able to take their addresses
      int* seqX_gpu = nw.seqX_gpu.data();
      int* seqY_gpu = nw.seqY_gpu.data();
      int* score_gpu = nw.score_gpu.data();
      int* subst_gpu = nw.subst_gpu.data();

      // group arguments to be passed to kernel A
      void* kargs[]
      {
         &seqX_gpu,
         &seqY_gpu,
         &score_gpu,
         &subst_gpu,
         &adjrows,
         &adjcols,
         &nw.substsz,
         &nw.indel
      };

      // launch the kernel in the given stream (don't statically allocate shared memory)
      if( cudaSuccess != cudaLaunchKernel( ( void* )Nw_Gpu4_KernelA, gridA, blockA, kargs, shmemsz, nullptr/*stream*/ ) ) return NwStat::errorKernelFailure;
   }
   
   // wait for the gpu to finish before going to the next step
   if( cudaSuccess != cudaDeviceSynchronize() ) return NwStat::errorKernelFailure;

   // measure calculation init time
   res.sw.lap( "calc-init" );



   // launch kernel B
   {
      // grid and block dimensions for kernel B
      dim3 gridB {};
      dim3 blockB {};
      // the number of tiles per row and column of the score matrix
      int trows = ceil( float( adjrows-1 )/tileBy );
      int tcols = ceil( float( adjcols-1 )/tileBx );
      
      // calculate size of shared memory per block in bytes
      int shmemsz = (
         /*tile[]*/ ( 1+tileBy )*( 1+tileBx ) *sizeof( int )
      );
      
      // calculate grid and block dimensions for kernel B
      {
         // take the number of threads on the largest diagonal of the tile
         // +   multiply by the number of half warps in the larger dimension for faster writing to global gpu memory
         blockB.x = nw.WARPSZ * ceil( max( tileBy, tileBx )*2./nw.WARPSZ );
         // take the number of tiles on the largest score matrix diagonal as the only dimension
         gridB.x = min( trows, tcols );

         // the maximum number of parallel blocks on a streaming multiprocessor
         int maxBlocksPerSm = 0;
         // number of threads per block that the kernel will be launched with
         int numThreads = blockB.x;

         // calculate the max number of parallel blocks per streaming multiprocessor
         if( cudaSuccess != cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerSm, Nw_Gpu4_KernelB, numThreads, shmemsz ) ) return NwStat::errorKernelFailure;
         // the number of cooperative blocks launched must not exceed the maximum possible number of parallel blocks on the device
         gridB.x = min( gridB.x, nw.MPROCS*maxBlocksPerSm );
      }


      // create variables for gpu arrays in order to be able to take their addresses
      int* score_gpu = nw.score_gpu.data();

      // group arguments to be passed to kernel B
      void* kargs[]
      {
         &score_gpu,
         &nw.indel,
         &nw.WARPSZ,
         &trows,
         &tcols,
         &tileBx,
         &tileBy
      };
      
      // launch the kernel in the given stream (don't statically allocate shared memory)
      if( cudaSuccess != cudaLaunchCooperativeKernel( ( void* )Nw_Gpu4_KernelB, gridB, blockB, kargs, shmemsz, nullptr/*stream*/ ) ) return NwStat::errorKernelFailure;
   }

   // wait for the gpu to finish before going to the next step
   if( cudaSuccess != cudaDeviceSynchronize() ) return NwStat::errorKernelFailure;

   // measure calculation time
   res.sw.lap( "calc" );


   // save the calculated score matrix
   if( !memTransfer( nw.score, nw.score_gpu, nw.adjrows, nw.adjcols, adjcols ) ) return NwStat::errorMemoryTransfer;

   // measure memory transfer time
   res.sw.lap( "mem-to-host" );

   return NwStat::success;
}





