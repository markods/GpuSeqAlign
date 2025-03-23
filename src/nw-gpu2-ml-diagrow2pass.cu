#include "common.hpp"

// cuda kernel A for the parallel implementation
// +   initializes the score matrix's header row and column in the gpu
__global__ static void Nw_Gpu2_KernelA(
    int* const score_gpu,
    const int adjrows,
    const int adjcols,
    const int indel)
{
    int j = (blockDim.x * blockIdx.x + threadIdx.x);
    if (j < adjcols)
    {
        el(score_gpu, adjcols, 0, j) = j * indel;
    }

    // skip the zeroth element in the zeroth column, since it is already initialized
    int i = 1 + j;
    if (i < adjrows)
    {
        el(score_gpu, adjcols, i, 0) = i * indel;
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
    const int tileBx,
    const int tileBy,
    const int d // the current minor tile diagonal in the score matrix (exclude the header row and column)
)
{
    extern __shared__ int shmem[/* substsz*substsz */];
    // the substitution matrix and relevant parts of the two sequences
    int* const subst /*[substsz*substsz]*/ = shmem + 0;

    // initialize the substitution shared memory copy
    {
        // map the threads from the thread block onto the substitution matrix elements
        int i = threadIdx.x;
        // while the current thread maps onto an element in the matrix
        while (i < substsz * substsz)
        {
            // copy the current element from the global substitution matrix
            el(subst, substsz, 0, i) = el(subst_gpu, substsz, 0, i);
            // map this thread to the next element with stride equal to the number of threads in this block
            i += blockDim.x;
        }
    }

    // all threads should finish initializing their substitution shared memory
    __syncthreads();

    // the number of columns in the score matrix
    int adjcols = 1 + tcols * tileBx;

    // (d,p) -- tile coordinates on the score matrix diagonal
    int pbeg = max(0, d - (tcols - 1));
    int pend = min(d + 1, trows);
    // position of the current thread's tile on the matrix diagonal
    int p = pbeg + (blockDim.x * blockIdx.x + threadIdx.x);

    //  x x x x x x       x x x x x x       x x x x x x
    //  x / / / . .       x . . . / /       x . . . . .|/ /
    //  x / / . . .   +   x . . / / .   +   x . . . . /|/
    //  x / . . . .       x . / / . .       x . . . / /|

    // if the thread maps onto a tile on the current matrix diagonal
    if (p < pend)
    {
        // position of the top left tile element in the current tile diagonal in the score matrix
        int ibeg = 1 + (p)*tileBy;
        int jbeg = 1 + (d - p) * tileBx;

        //      x x x x x
        //      | | | | |
        // y -- o . . . .
        // y -- . . . . .
        // y -- . . . . .
        // calculate the tile (one thread per tile, therefore small tiles)
        for (int i = ibeg; i < ibeg + tileBy; i++)
        {
            for (int j = jbeg; j < jbeg + tileBx; j++)
            {
                // calculate the current element's value
                // +   always subtract the insert delete cost from the result, since that value was added to the initial temporary
                int p0 = el(subst, substsz, seqY_gpu[i], seqX_gpu[j]) - indel;

                int p1 = el(score_gpu, adjcols, i - 1, j - 1) + p0; // MOVE DOWN-RIGHT
                int p2 = max(el(score_gpu, adjcols, i - 1, j), p1); // MOVE DOWN
                int p3 = max(el(score_gpu, adjcols, i, j - 1), p2); // MOVE RIGHT
                el(score_gpu, adjcols, i, j) = p3 + indel;
            }
        }
    }
}

// parallel gpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Gpu2_Ml_DiagRow2Pass(NwParams& pr, NwInput& nw, NwResult& res)
{
    // tile size for the kernel B
    int tileBx = {};
    int tileBy = {};
    // number of threads per block for kernels A and B
    int threadsPerBlockA = {};
    int threadsPerBlockB = {};

    // get the parameter values
    try
    {
        tileBx = pr["tileBx"].curr();
        tileBy = pr["tileBy"].curr();
        threadsPerBlockA = pr["threadsPerBlock"].curr();
        threadsPerBlockB = pr["threadsPerBlock"].curr();
    }
    catch (const std::out_of_range&)
    {
        return NwStat::errorInvalidValue;
    }

    // adjusted gpu score matrix dimensions
    // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile A size (in order to be evenly divisible)
    int adjrows = 1 + tileBy * (int)ceil(float(nw.adjrows - 1) / tileBy);
    int adjcols = 1 + tileBx * (int)ceil(float(nw.adjcols - 1) / tileBx);
    // special case when very small and very large sequences are compared
    if (adjrows == 1)
    {
        adjrows = 1 + tileBy;
    }
    if (adjcols == 1)
    {
        adjcols = 1 + tileBx;
    }

    // start the timer
    Stopwatch& sw = res.sw_align;
    sw.start();

    // reserve space in the ram and gpu global memory
    try
    {
        nw.seqX_gpu.init(adjcols);
        nw.seqY_gpu.init(adjrows);
        nw.score_gpu.init(adjrows * adjcols);

        nw.score.init(nw.adjrows * nw.adjcols);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // measure allocation time
    sw.lap("alloc");

    // copy data from host to device
    if (cudaSuccess != (cudaStatus = memTransfer(nw.seqX_gpu, nw.seqX, nw.adjcols)))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (cudaStatus = memTransfer(nw.seqY_gpu, nw.seqY, nw.adjrows)))
    {
        return NwStat::errorMemoryTransfer;
    }
    // also initialize padding, since it is used to access elements in the substitution matrix
    if (cudaSuccess != (cudaStatus = memSet(nw.seqX_gpu, nw.adjcols, 0 /*value*/)))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (cudaStatus = memSet(nw.seqY_gpu, nw.adjrows, 0 /*value*/)))
    {
        return NwStat::errorMemoryTransfer;
    }

    // measure memory transfer time
    sw.lap("cpy-dev");

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
        int shmemsz = (0);

        // calculate grid and block dimensions for kernel A
        {
            // take the number of threads per block as the only dimension
            blockA.x = threadsPerBlockA;
            // take the number of blocks on the score matrix diagonal as the only dimension
            gridA.x = (int)ceil(float(max2(adjrows, adjcols)) / threadsPerBlockA);
        }

        // create variables for gpu arrays in order to be able to take their addresses
        int* score_gpu = nw.score_gpu.data();

        // group arguments to be passed to kernel A
        void* kargs[] {
            &score_gpu,
            &adjrows,
            &adjcols,
            &nw.indel};

        // launch the kernel A in the given stream (don't statically allocate shared memory)
        if (cudaSuccess != (cudaStatus = cudaLaunchKernel((void*)Nw_Gpu2_KernelA, gridA, blockA, kargs, shmemsz, cudaStreamDefault)))
        {
            return NwStat::errorKernelFailure;
        }
    }

    // wait for the gpu to finish before going to the next step
    if (cudaSuccess != (cudaStatus = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // measure header initialization time
    sw.lap("init-hdr");

    //  x x x x x x       x x x x x x       x x x x x x
    //  x / / / . .       x . . . / /       x . . . . .|/ /
    //  x / / . . .   +   x . . / / .   +   x . . . . /|/
    //  x / . . . .       x . / / . .       x . . . / /|
    // launch kernel B for each minor diagonal of the score matrix
    {
        cudaStream_t stream;
        if (cudaSuccess != (cudaStatus = cudaStreamCreate(&stream)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer1 = make_defer([&]() noexcept
        {
            cudaStreamDestroy(stream);
        });

        cudaGraph_t graph;
        if (cudaSuccess != (cudaStatus = cudaGraphCreate(&graph, 0)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer2 = make_defer([&]() noexcept
        {
            cudaGraphDestroy(graph);
        });

        // start capturing kernel launches by this thread
        if (cudaSuccess != (cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)))
        {
            return NwStat::errorKernelFailure;
        }

        // grid and block dimensions for kernel B
        dim3 gridB {};
        dim3 blockB {};
        // the number of tiles per row and column of the score matrix
        int trows = (int)ceil(float(adjrows - 1) / tileBy);
        int tcols = (int)ceil(float(adjcols - 1) / tileBx);

        // calculate size of shared memory per block in bytes
        int shmemsz = (
            /*subst[]*/ nw.substsz * nw.substsz * sizeof(int));

        // for all minor diagonals in the score matrix (excluding the header row and column)
        for (int d = 0; d < tcols - 1 + trows; d++)
        {
            // calculate grid and block dimensions for kernel B
            {
                int pbeg = max(0, d - (tcols - 1));
                int pend = min(d + 1, trows);

                // the number of elements on the current diagonal
                int dsize = pend - pbeg;

                // take the number of threads per block as the only dimension
                blockB.x = threadsPerBlockB;
                // take the number of blocks on the current score matrix diagonal as the only dimension
                // +   launch at least one block on the x axis
                gridB.x = (int)ceil(float(dsize) / threadsPerBlockB);
            }

            // create variables for gpu arrays in order to be able to take their addresses
            int* seqX_gpu = nw.seqX_gpu.data();
            int* seqY_gpu = nw.seqY_gpu.data();
            int* score_gpu = nw.score_gpu.data();
            int* subst_gpu = nw.subst_gpu.data();

            // group arguments to be passed to kernel B
            void* kargs[] {
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
                &d};

            // launch the kernel B in the given stream (don't statically allocate shared memory)
            if (cudaSuccess != (cudaStatus = cudaLaunchKernel((void*)Nw_Gpu2_KernelB, gridB, blockB, kargs, shmemsz, stream)))
            {
                return NwStat::errorKernelFailure;
            }
        }

        // collect kernel launches from this thread
        if (cudaSuccess != (cudaStatus = cudaStreamEndCapture(stream, &graph)))
        {
            return NwStat::errorKernelFailure;
        }

        cudaGraphExec_t graphExec;
        if (cudaSuccess != (cudaStatus = cudaGraphInstantiate(&graphExec, graph, nullptr /*pErrorNode*/, nullptr /*pLogBuffer*/, 0 /*bufferSize*/)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer3 = make_defer([&]() noexcept
        {
            cudaGraphExecDestroy(graphExec);
        });

        // actually execute the kernels
        if (cudaSuccess != (cudaStatus = cudaGraphLaunch(graphExec, cudaStreamDefault)))
        {
            return NwStat::errorKernelFailure;
        }
    }

    // wait for the gpu to finish before going to the next step
    if (cudaSuccess != (cudaStatus = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // measure calculation time
    sw.lap("calc-1");

    // save the calculated score matrix
    if (cudaSuccess != (cudaStatus = memTransfer(nw.score, nw.score_gpu, nw.adjrows, nw.adjcols, adjcols)))
    {
        return NwStat::errorMemoryTransfer;
    }

    // measure memory transfer time
    sw.lap("cpy-host");

    return NwStat::success;
}
