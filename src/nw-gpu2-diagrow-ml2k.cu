#include "common.hpp"


// cuda kernel A for the parallel implementation
// +   initializes the score matrix's header row and column in the gpu
__global__ static void Nw_Gpu2_KernelA(
         int* const score_gpu,
   const int adjrows,
   const int adjcols,
   const int indel
)
{
   int j = ( blockDim.x*blockIdx.x + threadIdx.x );
   if( j < adjcols )
   {
      el(score_gpu,adjcols, 0,j) = j*indel;
   }

   // skip the zeroth element in the zeroth column, since it is already initialized
   int i = 1+ j;
   if( i < adjrows )
   {
      el(score_gpu,adjcols, i,0) = i*indel;
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
   const int indel,
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
         int p0 = el(subst,substsz, seqY_gpu[i], seqX_gpu[j]) - indel;
         
         int p1 =      el(score_gpu,adjcols, i-1,j-1) + p0;     // MOVE DOWN-RIGHT
         int p2 = max( el(score_gpu,adjcols, i-1,j  ) , p1 );   // MOVE DOWN
         int p3 = max( el(score_gpu,adjcols, i  ,j-1) , p2 );   // MOVE RIGHT
         el(score_gpu,adjcols, i,j) = p3 + indel;
      }
   }
}



// parallel gpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Gpu2_DiagRow_Ml2K( NwParams& pr, NwInput& nw, NwResult& res )
{
   // tile size for the kernel B
   unsigned tileBx;
   unsigned tileBy;
   // number of threads per block for kernels A and B
   unsigned threadsPerBlockA;
   unsigned threadsPerBlockB;

   // get the parameter values
   try
   {
      tileBx = pr["tileBx"].curr();
      tileBy = pr["tileBy"].curr();
      threadsPerBlockA = pr["threadsPerBlock"].curr();
      threadsPerBlockB = pr["threadsPerBlock"].curr();
   }
   catch( const std::out_of_range& )
   {
      return NwStat::errorInvalidValue;
   }

   // adjusted gpu score matrix dimensions
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile A size (in order to be evenly divisible)
   int adjrows = 1 + tileBy*ceil( float( nw.adjrows-1 )/tileBy );
   int adjcols = 1 + tileBx*ceil( float( nw.adjcols-1 )/tileBx );

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
         gridA.x = ceil( float( max2( adjrows, adjcols ) ) / threadsPerBlockA )*threadsPerBlockA;
      }


      // create variables for gpu arrays in order to be able to take their addresses
      int* score_gpu = nw.score_gpu.data();

      // group arguments to be passed to kernel A
      void* kargs[]
      {
         &score_gpu,
         &adjrows,
         &adjcols,
         &nw.indel
      };
      
      // launch the kernel A in the given stream (don't statically allocate shared memory)
      if( cudaSuccess != ( cudaStatus = cudaLaunchKernel( ( void* )Nw_Gpu2_KernelA, gridA, blockA, kargs, shmemsz, nullptr/*stream*/ ) ) )
      {
         return NwStat::errorKernelFailure;
      }
   }

   // wait for the gpu to finish before going to the next step
   if( cudaSuccess != ( cudaStatus = cudaDeviceSynchronize() ) )
   {
      return NwStat::errorKernelFailure;
   }

   // measure header initialization time
   res.sw.lap( "init-hdr" );



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
      int trows = ceil( float( adjrows-1 )/tileBy );
      int tcols = ceil( float( adjcols-1 )/tileBx );

      // calculate size of shared memory per block in bytes
      int shmemsz = (
         /*subst[]*/ nw.substsz*nw.substsz *sizeof( int )
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
            // +   launch at least one block on the x axis
            gridB.x = ceil( float( dsize ) / threadsPerBlockB );
            if( gridB.x < 1 ) { gridB.x = 1; }
         }


         // create variables for gpu arrays in order to be able to take their addresses
         int* seqX_gpu = nw.seqX_gpu.data();
         int* seqY_gpu = nw.seqY_gpu.data();
         int* score_gpu = nw.score_gpu.data();
         int* subst_gpu = nw.subst_gpu.data();

         // group arguments to be passed to kernel B
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
            &trows,
            &tcols,
            &tileBx,
            &tileBy,
            &d
         };
         
         // launch the kernel B in the given stream (don't statically allocate shared memory)
         if( cudaSuccess != ( cudaStatus = cudaLaunchKernel( ( void* )Nw_Gpu2_KernelB, gridB, blockB, kargs, shmemsz, nullptr/*stream*/ ) ) )
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





