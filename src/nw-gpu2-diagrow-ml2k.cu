#include "common.hpp"


// cuda kernel A for the parallel implementation
// +   initializes the score matrix's header row and column in the gpu
__global__ static void Nw_Gpu2_KernelA(
         int* const score_gpu,
   const int adjrows,
   const int adjcols,
   const int indelcost
)
{
   int j = ( blockDim.x*blockIdx.x + threadIdx.x );
   if( j < adjcols )
   {
      el(score_gpu,adjcols, 0,j) = j*indelcost;
   }

   // skip the zeroth element in the zeroth column, since it is already initialized
   int i = 1+ j;
   if( i < adjrows )
   {
      el(score_gpu,adjcols, i,0) = i*indelcost;
   }
}


// cuda kernel B for the parallel implementation
__global__ static void Nw_Gpu2_KernelB(
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
   const unsigned tileBx,
   const unsigned tileBy,
   const int d   // the current minor tile diagonal in the score matrix (exclude the header row and column)
)
{
   extern __shared__ int shmem[/* substsz*substsz */];
   // the substitution matrix and relevant parts of the two sequences
   int* const subst/*[substsz*substsz]*/ = shmem + 0;

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

   // all threads should finish initializing their substitution shared memory
   __syncthreads();

   // the number of columns in the score matrix
   int adjcols = 1 + tcols*tileBx;
   
   // (d,p) -- tile coordinates on the score matrix diagonal
   int pbeg = max( 0, d - (tcols-1) );
   int pend = min( d, trows-1 );
   // position of the current thread's tile on the matrix diagonal
   int p = pbeg + ( blockDim.x*blockIdx.x + threadIdx.x );

   //  x x x x x x       x x x x x x       x x x x x x
   //  x / / / . .       x . . . / /       x . . . . .|/ /
   //  x / / . . .   +   x . . / / .   +   x . . . . /|/
   //  x / . . . .       x . / / . .       x . . . / /|

   // if the thread maps onto a tile on the current matrix diagonal
   if( p <= pend )
   {
      // position of the top left tile element in the current tile diagonal in the score matrix
      int ibeg = 1 + (   p )*tileBy;
      int jbeg = 1 + ( d-p )*tileBx;

      //      x x x x x
      //      | | | | |
      // y -- o . . . .
      // y -- . . . . .
      // y -- . . . . .
      // calculate the tile (one thread per tile, therefore small tiles)
      for( int i = ibeg; i < ibeg + tileBy; i++ )
      for( int j = jbeg; j < jbeg + tileBx; j++ )
      {
         // calculate the current element's value
         // +   always subtract the insert delete cost from the result, since that value was added to the initial temporary
         int p0 = el(subst,substsz, seqY_gpu[i], seqX_gpu[j]) - indelcost;
         
         int p1 =      el(score_gpu,adjcols, i-1,j-1) + p0;     // MOVE DOWN-RIGHT
         int p2 = max( el(score_gpu,adjcols, i-1,j  ) , p1 );   // MOVE DOWN
         int p3 = max( el(score_gpu,adjcols, i  ,j-1) , p2 );   // MOVE RIGHT
         el(score_gpu,adjcols, i,j) = p3 + indelcost;
      }
   }
}



// parallel gpu implementation of the Needleman Wunsch algorithm
void Nw_Gpu2_DiagRow_Ml2K( NwInput& nw, NwMetrics& res )
{
   // tile size for the kernel B
   unsigned tileBx = 8;
   unsigned tileBy = 4;
   // number of threads per block for kernels A and B
   unsigned threadsPerBlockA = 16*WARPSZ;
   unsigned threadsPerBlockB = 8*WARPSZ;


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



   //  x x x x x x
   //  x . . . . .
   //  x . . . . .
   //  x . . . . .
   // launch kernel A to initialize the score matrix's header row and column
   {
      // grid and block dimensions for kernel A
      dim3 gridA {};
      dim3 blockA {};

      // calculate size of shared memory per block in bytes
      int shmemsz = (
         0
      );

      // calculate grid and block dimensions for kernel A
      {
         // take the number of threads per block as the only dimension
         blockA.x = threadsPerBlockA;
         // take the number of blocks on the score matrix diagonal as the only dimension
         gridA.x = ceil( float( max2( nw_gpu.adjrows, nw_gpu.adjcols ) ) / threadsPerBlockA )*threadsPerBlockA;
      }


      // group arguments to be passed to kernel A
      void* kargs[] { &nw_gpu.score, &nw_gpu.adjrows, &nw_gpu.adjcols, &nw_gpu.indelcost };
      
      // launch the kernel A in the given stream (don't statically allocate shared memory)
      // +   capture events around kernel launch as well
      // +   update the stop event when the kernel finishes
      cudaEventRecord( start, 0/*stream*/ );
      cudaLaunchKernel( ( void* )Nw_Gpu2_KernelA, gridA, blockA, kargs, shmemsz, 0/*stream*/ );
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

   //  x x x x x x       x x x x x x       x x x x x x
   //  x / / / . .       x . . . / /       x . . . . .|/ /
   //  x / / . . .   +   x . . / / .   +   x . . . . /|/
   //  x / . . . .       x . / / . .       x . . . / /|
   // launch kernel B for each minor diagonal of the score matrix
   {
      // grid and block dimensions for kernel B
      dim3 gridB {};
      dim3 blockB {};
      // the number of tiles per row and column of the score matrix
      int trows = ceil( float( nw_gpu.adjrows-1 )/tileBy );
      int tcols = ceil( float( nw_gpu.adjcols-1 )/tileBx );

      // calculate size of shared memory per block in bytes
      int shmemsz = (
         /*subst[]*/ nw_gpu.substsz*nw_gpu.substsz *sizeof( int )
      );

      // for all minor diagonals in the score matrix (excluding the header row and column)
      for( int d = 0;   d < tcols-1 + trows;   d++ )
      {
         // calculate grid and block dimensions for kernel B
         {
            int pbeg = max( 0, d - (tcols-1) );
            int pend = min( d, trows-1 );
            
            // the number of elements on the current diagonal
            int dsize = pend-pbeg + 1;

            // take the number of threads per block as the only dimension
            blockB.x = threadsPerBlockB;
            // take the number of blocks on the current score matrix diagonal as the only dimension
            gridB.x = ceil( float( dsize ) / threadsPerBlockB )*threadsPerBlockB;
         }


         // group arguments to be passed to kernel B
         void* kargs[] { &nw_gpu.seqX, &nw_gpu.seqY, &nw_gpu.score, &nw_gpu.subst, /*&nw_gpu.adjrows,*/ /*&nw_gpu.adjcols,*/ &nw_gpu.substsz, &nw_gpu.indelcost, &trows, &tcols, &tileBx, &tileBy, &d };
         
         // launch the kernel B in the given stream (don't statically allocate shared memory)
         // +   capture events around kernel launch as well
         // +   update the stop event when the kernel finishes
         cudaEventRecord( start, 0/*stream*/ );
         cudaLaunchKernel( ( void* )Nw_Gpu2_KernelB, gridB, blockB, kargs, shmemsz, 0/*stream*/ );
         cudaEventRecord( stop, 0/*stream*/ );
         cudaEventSynchronize( stop );
         
         // kernel execution time
         float ktime {};
         // calculate the time between the given events
         cudaEventElapsedTime( &ktime, start, stop ); ktime /= 1000./*ms*/;
         // update the total kernel execution time
         res.Tgpu += ktime;
      }
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





