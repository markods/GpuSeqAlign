#include "common.hpp"
#include "lang.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

// cuda kernel A for the parallel implementation
// +   initializes the score matrix's header row and column in the gpu
__global__ static void Nw_Gpu3_KernelA(
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

// cuda kernel for the parallel implementation
__global__ static void Nw_Gpu3_KernelB(
    // nw input
    const int* const seqX_gpu,
    const int* const seqY_gpu,
    int* const score_gpu,
    const int* const subst_gpu,
    // const int adjrows,   // can be calculated as 1 + trows*tileBy
    // const int adjcols,   // can be calculated as 1 + tcols*tileBx
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
    extern __shared__ int shmem[/* substsz*substsz + tileBx + tileBy + (1+tileBy)*(1+tileBx) */];
    // the substitution matrix and relevant parts of the two sequences
    // NOTE: should we align allocations to 0-th shared memory bank?
    int* const subst /*[substsz*substsz]*/ = shmem + 0;
    int* const seqX /*[tileBx]*/ = subst + substsz * substsz;
    int* const seqY /*[tileBy]*/ = seqX + tileBx;
    int* const tile /*[(1+tileBy)*(1+tileBx)]*/ = seqY + tileBy;

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

    // all threads in this block should finish initializing their substitution shared memory
    __syncthreads();

    //  / / / . .       . . . / /       . . . . .|/ /
    //  / / . . .   +   . . / / .   +   . . . . /|/
    //  / . . . .       . / / . .       . . . / /|
    // map a tile on the current tile diagonal to this thread block
    // (s,t) -- tile coordinates in the grid of tiles (score matrix)
    int tbeg = max(0, d - (tcols - 1));
    int tend = min(d + 1, trows);

    // map a tile on the current diagonal of tiles to this thread block
    int t = tbeg + blockIdx.x;

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
        int ibeg = 1 + (t)*tileBy;
        int jbeg = 1 + (d - t) * tileBx;

        // map the threads from the thread block onto the global X sequence's elements (which will be used in this tile)
        int j = threadIdx.x;
        // while the current thread maps onto an element in the tile's X sequence
        while (j < tileBx)
        {
            // initialize that element in the X seqence's shared window
            seqX[j] = seqX_gpu[jbeg + j];

            // map this thread to the next element with stride equal to the number of threads in this block
            j += blockDim.x;
        }

        // map the threads from the thread block onto the global Y sequence's elements (which will be used in this tile)
        int i = threadIdx.x;
        // while the current thread maps onto an element in the tile's Y sequence
        while (i < tileBy)
        {
            // initialize that element in the Y seqence's shared window
            seqY[i] = seqY_gpu[ibeg + i];

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
        int ibeg = (1 + (t)*tileBy) - 1 /*header*/;
        int jbeg = (1 + (d - t) * tileBx) - 1 /*header*/;
        // the number of columns in the score matrix
        int adjcols = 1 + tcols * tileBx;

        // map the threads from the thread block onto the tile's header row (stored in the global score matrix)
        int j = threadIdx.x;
        // while the current thread maps onto an element in the tile's header row (stored in the global score matrix)
        while (j < 1 + tileBx)
        {
            // initialize that element in the tile's shared memory
            el(tile, 1 + tileBx, 0, j) = el(score_gpu, adjcols, ibeg + 0, jbeg + j);

            // map this thread to the next element with stride equal to the number of threads in this block
            j += blockDim.x;
        }

        // map the threads from the thread block onto the tile's header column (stored in the global score matrix)
        // +   skip the zeroth element since it is already initialized
        int i = 1 + threadIdx.x;
        // while the current thread maps onto an element in the tile's header column (stored in the global score matrix)
        while (i < 1 + tileBy)
        {
            // initialize that element in the tile's shared memory
            el(tile, 1 + tileBx, i, 0) = el(score_gpu, adjcols, ibeg + i, jbeg + 0);

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
        int i = threadIdx.x / tileBx;
        int j = threadIdx.x % tileBx;
        // stride on the current thread position in the tile, equal to the number of threads in this thread block
        // +   it is split into row and column increments for the thread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
        int di = blockDim.x / tileBx;
        int dj = blockDim.x % tileBx;

        // while the current thread maps onto an element in the tile
        while (i < tileBy)
        {
            // use the substitution matrix to partially calculate the score matrix element value
            // +   increase the value by insert delete cost, since then the formula for calculating the actual element value later on becomes simpler
            el(tile, 1 + tileBx, 1 + i, 1 + j) = el(subst, substsz, seqY[i], seqX[j]) - indel;

            // map the current thread to the next tile element
            i += di;
            j += dj;
            // if the column index is out of bounds, increase the row index by one and wrap around the column index
            if (j >= tileBx)
            {
                i++;
                j -= tileBx;
            }
        }
    }

    // all threads in this block should finish initializing this tile in shared memory
    __syncthreads();

    // calculate the tile elements
    // +   only threads in the first warp from this block are active here, other warps have to wait
    if (threadIdx.x < warpSize)
    {
        // the number of rows and columns in the tile without its first row and column (the part of the tile to be calculated)
        int rows = tileBy;
        int cols = tileBx;

        //  x x x x x x       x x x x x x       x x x x x x
        //  x / / / . .       x . . . / /       x . . . . .|/ /
        //  x / / . . .   +   x . . / / .   +   x . . . . /|/
        //  x / . . . .       x . / / . .       x . . . / /|

        // for all diagonals in the tile without its first row and column
        for (int d = 0; d < cols - 1 + rows; d++)
        {
            // (d,p) -- element coordinates in the tile
            int pbeg = max(0, d - (cols - 1));
            int pend = min(d + 1, rows);
            // position of the current thread's element on the tile diagonal
            int p = pbeg + threadIdx.x;

            // if the thread maps onto an element on the current tile diagonal
            if (p < pend)
            {
                // position of the current element
                int i = 1 + (p);
                int j = 1 + (d - p);

                // calculate the current element's value
                // +   always subtract the insert delete cost from the result, since the kernel A added that value to each element of the score matrix
                int temp1 = el(tile, 1 + tileBx, i - 1, j - 1) + el(tile, 1 + tileBx, i, j);
                int temp2 = max(el(tile, 1 + tileBx, i - 1, j), el(tile, 1 + tileBx, i, j - 1));
                el(tile, 1 + tileBx, i, j) = max(temp1, temp2) + indel;
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
        int ibeg = (1 + (t)*tileBy);
        int jbeg = (1 + (d - t) * tileBx);
        // the number of columns in the score matrix
        int adjcols = 1 + tcols * tileBx;

        // current thread position in the tile
        int i = threadIdx.x / tileBx;
        int j = threadIdx.x % tileBx;
        // stride on the current thread position in the tile, equal to the number of threads in this thread block
        // +   it is split into row and column increments for the thread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
        int di = blockDim.x / tileBx;
        int dj = blockDim.x % tileBx;

        // while the current thread maps onto an element in the tile
        while (i < tileBy)
        {
            // copy the current element from the tile to the global score matrix
            el(score_gpu, adjcols, ibeg + i, jbeg + j) = el(tile, 1 + tileBx, 1 + i, 1 + j);

            // map the current thread to the next tile element
            i += di;
            j += dj;
            // if the column index is out of bounds, increase the row index by one and wrap around the column index
            if (j >= tileBx)
            {
                i++;
                j -= tileBx;
            }
        }
    }
}

// parallel gpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Gpu3_Ml_DiagDiag(NwParams& pr, NwInput& nw, NwResult& res)
{
    // tile size for the kernel
    int tileBx = {};
    int tileBy = nw.warpsz;
    // number of threads per block for kernels A and B
    int threadsPerBlockA = {};

    // get the parameter values
    try
    {
        threadsPerBlockA = pr["threadsPerBlockA"].curr();
        tileBx = pr["tileBx"].curr();
    }
    catch (const std::out_of_range&)
    {
        return NwStat::errorInvalidValue;
    }

    // adjusted gpu score matrix dimensions
    // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile B size (in order to be evenly divisible)
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
    sw.lap("align.alloc");

    // copy data from host to device
    if (cudaSuccess != (res.cudaStat = memTransfer(nw.seqX_gpu, nw.seqX, nw.adjcols)))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (res.cudaStat = memTransfer(nw.seqY_gpu, nw.seqY, nw.adjrows)))
    {
        return NwStat::errorMemoryTransfer;
    }
    // also initialize padding, since it is used to access elements in the substitution matrix
    if (cudaSuccess != (res.cudaStat = memSet(nw.seqX_gpu, nw.adjcols, 0 /*value*/)))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (res.cudaStat = memSet(nw.seqY_gpu, nw.adjrows, 0 /*value*/)))
    {
        return NwStat::errorMemoryTransfer;
    }

    // measure memory transfer time
    sw.lap("align.cpy_dev");

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
        if (cudaSuccess != (res.cudaStat = cudaLaunchKernel((void*)Nw_Gpu3_KernelA, gridA, blockA, kargs, shmemsz, cudaStreamDefault)))
        {
            return NwStat::errorKernelFailure;
        }
    }

    // wait for the gpu to finish before going to the next step
    if (cudaSuccess != (res.cudaStat = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // measure header initialization time
    sw.lap("align.init_hdr");

    //  x x x x x x       x x x x x x       x x x x x x
    //  x / / / . .       x . . . / /       x . . . . .|/ /
    //  x / / . . .   +   x . . / / .   +   x . . . . /|/
    //  x / . . . .       x . / / . .       x . . . / /|
    // launch kernel B for each minor tile diagonal of the score matrix
    {
        cudaStream_t stream;
        if (cudaSuccess != (res.cudaStat = cudaStreamCreate(&stream)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer1 = make_defer([&]() noexcept
        {
            cudaStreamDestroy(stream);
        });

        cudaGraph_t graph;
        if (cudaSuccess != (res.cudaStat = cudaGraphCreate(&graph, 0)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer2 = make_defer([&]() noexcept
        {
            cudaGraphDestroy(graph);
        });

        // start capturing kernel launches by this thread
        if (cudaSuccess != (res.cudaStat = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)))
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
            /*subst[]*/ nw.substsz * nw.substsz * sizeof(int)
            /*seqX[]*/
            + tileBx * sizeof(int)
            /*seqY[]*/
            + tileBy * sizeof(int)
            /*tile[]*/
            + (1 + tileBy) * (1 + tileBx) * sizeof(int));

        // for all minor tile diagonals in the score matrix (excluding the header row and column)
        for (int d = 0; d < tcols - 1 + trows; d++)
        {
            // calculate grid and block dimensions for kernel B
            {
                int pbeg = max2(0, d - (tcols - 1));
                int pend = min2(d + 1, trows);

                // the number of elements on the current diagonal
                int dsize = pend - pbeg;

                // take the number of threads on the largest diagonal of the tile
                // +   multiply by the number of half warps in the larger dimension for faster writing to global gpu memory
                blockB.x = nw.warpsz * (int)ceil(max2(tileBy, tileBx) * 2. / nw.warpsz);

                // take the number of blocks on the current score matrix diagonal as the only dimension
                // +   launch at least one block on the x axis
                gridB.x = dsize;
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
            if (cudaSuccess != (res.cudaStat = cudaLaunchKernel((void*)Nw_Gpu3_KernelB, gridB, blockB, kargs, shmemsz, stream)))
            {
                return NwStat::errorKernelFailure;
            }
        }

        // collect kernel launches from this thread
        if (cudaSuccess != (res.cudaStat = cudaStreamEndCapture(stream, &graph)))
        {
            return NwStat::errorKernelFailure;
        }

        cudaGraphExec_t graphExec;
        if (cudaSuccess != (res.cudaStat = cudaGraphInstantiate(&graphExec, graph, nullptr /*pErrorNode*/, nullptr /*pLogBuffer*/, 0 /*bufferSize*/)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer3 = make_defer([&]() noexcept
        {
            cudaGraphExecDestroy(graphExec);
        });

        // actually execute the kernels
        if (cudaSuccess != (res.cudaStat = cudaGraphLaunch(graphExec, cudaStreamDefault)))
        {
            return NwStat::errorKernelFailure;
        }
    }

    // wait for the gpu to finish before going to the next step
    if (cudaSuccess != (res.cudaStat = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // measure calculation time
    sw.lap("align.calc");

    // save the calculated score matrix
    if (cudaSuccess != (res.cudaStat = memTransfer(nw.score, nw.score_gpu, nw.adjrows, nw.adjcols, adjcols)))
    {
        return NwStat::errorMemoryTransfer;
    }

    // measure memory transfer time
    sw.lap("align.cpy_host");

    return NwStat::success;
}
