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
   const int indel,
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
         el(score_gpu,adjcols, 0,j) = j*indel;
         // if this is also the zeroth diagonal (with only one element on it)
         if( d == 0 )
         {
            // initialize TOP-LEFT header element
            el(score_gpu,adjcols, 0,0) = 0*indel;
         }
      }
      // if the thread maps onto the end of the diagonal
      if( d < rows && p == pend )
      {
         // initialize LEFT header element
         el(score_gpu,adjcols, i,0) = i*indel;
      }

      // calculate the current element's value
      // +   always subtract the insert delete cost from the result, since that value was added to the initial temporary
      int p0 = el(subst_gpu,substsz, seqY_gpu[i], seqX_gpu[j]) - indel;
      
      int p1 =      el(score_gpu,adjcols, i-1,j-1) + p0;     // MOVE DOWN-RIGHT
      int p2 = max( el(score_gpu,adjcols, i-1,j  ) , p1 );   // MOVE DOWN
      int p3 = max( el(score_gpu,adjcols, i  ,j-1) , p2 );   // MOVE RIGHT
      el(score_gpu,adjcols, i,j) = p3 + indel;
   }
}



// parallel gpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Gpu1_Diag_Ml( NwParams& pr, NwInput& nw, NwResult& res )
{
   // number of threads per block
   // +   the tile is one-dimensional
   unsigned threadsPerBlock;

   // get the parameter values
   try
   {
      threadsPerBlock = pr["threadsPerBlock"].curr();
   }
   catch( const std::out_of_range& ex )
   {
      return NwStat::errorInvalidValue;
   }
 
   // adjusted gpu score matrix dimensions
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile A size (in order to be evenly divisible)
   int adjrows = nw.adjrows;
   int adjcols = nw.adjcols;
   // special case when very small and very large sequences are compared
   if( adjrows == 1 ) { adjrows = 2; }
   if( adjcols == 1 ) { adjcols = 2; }
   // the dimensions of the matrix without its row and column header
   int rows = -1 + adjrows;
   int cols = -1 + adjcols;

   // start the timer
   res.sw.start();


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
   res.sw.lap( "alloc" );

   
   // copy data from host to device
   // +   gpu padding remains uninitialized, but this is not an issue since padding is only used to simplify kernel code (optimization)
   if( cudaSuccess != ( cudaStatus = memTransfer( nw.seqX_gpu, nw.seqX, nw.adjcols ) ) )
   {
      return NwStat::errorMemoryTransfer;
   }
   if( cudaSuccess != ( cudaStatus = memTransfer( nw.seqY_gpu, nw.seqY, nw.adjrows ) ) )
   {
      return NwStat::errorMemoryTransfer;
   }

   // measure memory transfer time
   res.sw.lap( "mem-to-device" );



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
            // +   launch at least one block on the x axis
            gridA.x = ceil( float( dsize ) / threadsPerBlock );
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
            &adjrows,
            &adjcols,
            &nw.substsz,
            &nw.indel,
            &d
         };
         
         // launch the kernel in the given stream (don't statically allocate shared memory)
         if( cudaSuccess != ( cudaStatus = cudaLaunchKernel( ( void* )Nw_Gpu1_Kernel, gridA, blockA, kargs, shmemsz, nullptr/*stream*/ ) ) )
         {
            return NwStat::errorKernelFailure;
         }
      }
   }

   // wait for the gpu to finish before going to the next step
   if( cudaSuccess != ( cudaStatus = cudaDeviceSynchronize() ) )
   {
      return NwStat::errorKernelFailure;
   }

   // measure calculation time
   res.sw.lap( "calc" );


   // save the calculated score matrix
   if( cudaSuccess != ( cudaStatus = memTransfer( nw.score, nw.score_gpu, nw.adjrows, nw.adjcols, adjcols ) ) )
   {
      return NwStat::errorMemoryTransfer;
   }

   // measure memory transfer time
   res.sw.lap( "mem-to-host" );

   return NwStat::success;
}





