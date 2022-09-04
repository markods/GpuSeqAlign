#include "common.hpp"


// cuda kernel for the parallel implementation
__global__ static void Nw_Gpu1_Kernel(
   const int* const seqX_gpu,
   const int* const seqY_gpu,
         int* const score_gpu,
   const int* const subst_gpu,
   const int adjrows,
   const int adjcols,
   const int substsz,
   const int indelcost,
   const int d   // the current minor diagonal in the score matrix (exclude the header row and column)
)
{
   // the dimensions of the matrix without its row and column header
   const int rows = -1 + adjrows;
   const int cols = -1 + adjcols;
   

   //  x x x x x x       x x x x x x       x x x x x x
   //  x / / / . .       x . . . / /       x . . . . .|/ /
   //  x / / . . .   +   x . . / / .   +   x . . . . /|/
   //  x / . . . .       x . / / . .       x . . . / /|

   // (d,p) -- element coordinates on the score matrix diagonal
   int pbeg = max( 0, d - (cols-1) );
   int pend = min( d, rows-1 );
   // position of the current thread's element on the matrix diagonal
   int p = pbeg + ( blockDim.x*blockIdx.x + threadIdx.x );


   // if the thread maps onto an element on the current matrix diagonal
   if( p <= pend )
   {
      // position of the current element
      int i = 1 + (   p );
      int j = 1 + ( d-p );


      // if the thread maps onto the start of the diagonal
      if( d < cols && p == 0 )
      {
         // initialize TOP header element
         el(score_gpu,adjcols, 0,j) = j*indelcost;
         // if this is also the zeroth diagonal (with only one element on it)
         if( d == 0 )
         {
            // initialize TOP-LEFT header element
            el(score_gpu,adjcols, 0,0) = 0*indelcost;
         }
      }
      // if the thread maps onto the end of the diagonal
      if( d < rows && p == pend )
      {
         // initialize LEFT header element
         el(score_gpu,adjcols, i,0) = i*indelcost;
      }

      // calculate the current element's value
      // +   always subtract the insert delete cost from the result, since that value was added to the initial temporary
      int p0 = el(subst_gpu,substsz, seqY_gpu[i], seqX_gpu[j]) - indelcost;
      
      int p1 =      el(score_gpu,adjcols, i-1,j-1) + p0;     // MOVE DOWN-RIGHT
      int p2 = max( el(score_gpu,adjcols, i-1,j  ) , p1 );   // MOVE DOWN
      int p3 = max( el(score_gpu,adjcols, i  ,j-1) , p2 );   // MOVE RIGHT
      el(score_gpu,adjcols, i,j) = p3 + indelcost;
   }
}



// parallel gpu implementation of the Needleman Wunsch algorithm
void Nw_Gpu1_Diag_Ml( NwInput& nw, NwMetrics& res )
{
   // number of threads per block
   // +   the tile is one-dimensional
   unsigned threadsPerBlock = 8*WARPSZ;


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
   nw_gpu.adjrows = nw.adjrows;
   nw_gpu.adjcols = nw.adjcols;
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



   // the dimensions of the matrix without its row and column header
   const int rows = -1 + nw.adjrows;
   const int cols = -1 + nw.adjcols;

   //  x x x x x x       x x x x x x       x x x x x x
   //  x / / / . .       x . . . / /       x . . . . .|/ /
   //  x / / . . .   +   x . . / / .   +   x . . . . /|/
   //  x / . . . .       x . / / . .       x . . . / /|
   // launch kernel for each minor diagonal of the score matrix
   {
      // grid and block dimensions for kernel
      dim3 gridA {};
      dim3 blockA {};

      // calculate size of shared memory per block in bytes
      int shmemsz = (
         0
      );

      // for all minor diagonals in the score matrix (excluding the header row and column)
      for( int d = 0;   d < cols-1 + rows;   d++ )
      {
         // calculate grid and block dimensions for kernel
         {
            int pbeg = max( 0, d - (cols-1) );
            int pend = min( d, rows-1 );
            
            // the number of elements on the current diagonal
            int dsize = pend-pbeg + 1;

            // take the number of threads per block as the only dimension
            blockA.x = threadsPerBlock;
            // take the number of blocks on the current score matrix diagonal as the only dimension
            gridA.x = ceil( float( dsize ) / threadsPerBlock )*threadsPerBlock;
         }


         // group arguments to be passed to kernel
         void* kargs[] { &nw_gpu.seqX, &nw_gpu.seqY, &nw_gpu.score, &nw_gpu.subst, &nw_gpu.adjrows, &nw_gpu.adjcols, &nw_gpu.substsz, &nw_gpu.indelcost, &d };
         
         // launch the kernel in the given stream (don't statically allocate shared memory)
         // +   capture events around kernel launch as well
         // +   update the stop event when the kernel finishes
         cudaEventRecord( start, 0/*stream*/ );
         cudaLaunchKernel( ( void* )Nw_Gpu1_Kernel, gridA, blockA, kargs, shmemsz, 0/*stream*/ );
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
   // Copies \p count bytes from the memory area pointed to by \p src to the
   // memory area pointed to by \p dst, where \p kind specifies the direction
   // of the copy, and must be one of ::cudaMemcpyHostToHost,
   // ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
   // ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
   // ::cudaMemcpyDefault is recommended, in which case the type of transfer is
   // inferred from the pointer values. However, ::cudaMemcpyDefault is only
   // allowed on systems that support unified virtual addressing. Calling
   // ::cudaMemcpy() with dst and src pointers that do not match the direction of
   // the copy results in an undefined behavior.

   // save the calculated score matrix
   // +   starts an async data copy from device to host, then waits for the copy to finish
   cudaMemcpy(
      nw    .score,                          // dst    - Destination memory address
      nw_gpu.score,                          // src    - Source memory address
      nw.adjrows*nw.adjcols * sizeof( int ), // count  - Size in bytes to copy
      cudaMemcpyDeviceToHost                 // kind   - Type of transfer
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





