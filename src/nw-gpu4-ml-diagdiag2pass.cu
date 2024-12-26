#include "common.hpp"

// cuda kernel A for the parallel implementation
// +   initializes the score matrix in the gpu
__global__ static void Nw_Gpu4_KernelA(
    const int *const seqX_gpu,
    const int *const seqY_gpu,
    int *const score_gpu,
    const int *const subst_gpu,
    const int adjrows,
    const int adjcols,
    const int substsz,
    const int indel,
    const unsigned tileAx,
    const unsigned tileAy)
{
    extern __shared__ int shmem[/* substsz*substsz + tileAx + tileAy */];
    // the substitution matrix and relevant parts of the two sequences
    // +   stored in shared memory for faster random access
    // NOTE: should we align allocations to 0-th shared memory bank?
    int *const subst /*[substsz*substsz]*/ = shmem + 0;
    int *const seqX /*[tileAx]*/ = subst + substsz * substsz;
    int *const seqY /*[tileAy]*/ = seqX + tileAx;

    // start position of the block in the global X and Y sequences
    int ibeg = blockIdx.y * tileAy;
    int jbeg = blockIdx.x * tileAx;
    // real tile size (since the score matrix is not evenly divisible by tileA, but is instead by tileB)
    int realAy = min(tileAy, adjrows - ibeg);
    int realAx = min(tileAx, adjcols - jbeg);

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

    // initialize the X and Y sequences' shared memory copies
    {
        // map the threads from the thread block onto the global X sequence's elements (which will be used in this tile)
        int j = threadIdx.x;
        // while the current thread maps onto an element in the tile's X sequence
        for (; j < realAx; j += blockDim.x)
        {
            // initialize that element in the X seqence's shared window
            seqX[j] = seqX_gpu[jbeg + j];
        }
        // while the current thread maps onto the padding of the tile's X sequence
        for (; j < tileAx; j += blockDim.x)
        {
            // initialize that element in the X seqence's padding
            seqX[j] = 0;
        }

        // map the threads from the thread block onto the global Y sequence's elements (which will be used in this tile)
        int i = threadIdx.x;
        // while the current thread maps onto an element in the tile's Y sequence
        for (; i < realAy; i += blockDim.x)
        {
            // initialize that element in the Y seqence's shared window
            seqY[i] = seqY_gpu[ibeg + i];
        }
        // while the current thread maps onto the padding of the tile's Y sequence
        for (; i < tileAy; i += blockDim.x)
        {
            // initialize that element in the Y seqence's padding
            seqY[i] = seqY_gpu[ibeg + i];
        }
    }

    // make sure that all threads have finished initializing their corresponding elements
    __syncthreads();

    // initialize the score matrix in global memory
    {
        // the current element value
        int elem = 0;

        // while this thread maps onto an element in the current tile in the global score matrix
        // +   blockDim.x is the number of threads in the block (in other words stride)
        int i = threadIdx.x / realAx;
        int di = blockDim.x / realAx;
        int j = threadIdx.x % realAx;
        int dj = blockDim.x % realAx;
        while (i < realAy)
        {
            // the position of the element this thread currently maps to in the score matrix
            int ipos = ibeg + i;
            int jpos = jbeg + j;
            // if the current thread is not in the first row or column of the score matrix
            // +   use the substitution matrix to calculate the score matrix element value
            // +   increase the value by insert delete cost, since then the formula for calculating the actual element value in kernel B becomes simpler
            if (ipos > 0 && jpos > 0)
            {
                elem = el(subst, substsz, seqY[i], seqX[j]) - indel;
            }
            // otherwise, if the current thread is in the first row or column
            // +   update the score matrix element using the insert delete cost
            else
            {
                elem = (ipos | jpos) * indel;
            }

            // update the corresponding element in global memory
            // +   fully coallesced memory access
            el(score_gpu, adjcols, ipos, jpos) = elem;

            // map the current thread to the next tile element
            i += di;
            j += dj;
            // if the column index is out of bounds, increase the row index by one and wrap around the column index
            if (j >= realAx)
            {
                i++;
                j -= realAx;
            }
        }
    }
}

// cuda kernel B for the parallel implementation
// +   calculates the score matrix in the gpu using the initialized score matrix from kernel A
// +   the given matrix minus the padding (zeroth row and column) must be evenly divisible by the tile B
__global__ static void Nw_Gpu4_KernelB(
    int *const score_gpu,
    const int indel,
    const int warpsz,
    const int trows,
    const int tcols,
    const unsigned tileBx,
    const unsigned tileBy,
    const int d // the current minor tile diagonal in the score matrix (exclude the header row and column)
)
{
    extern __shared__ int shmem[/* (1+tileBy)*(1+tileBx) */];
    // matrix tile which this thread block maps onto
    // +   stored in shared memory for faster random access
    int *const tile /*[(1+tileBy)*(1+tileBx)]*/ = shmem + 0;

    //  / / / . .       . . . / /       . . . . .|/ /
    //  / / . . .   +   . . / / .   +   . . . . /|/
    //  / . . . .       . / / . .       . . . / /|

    // map a tile on the current tile diagonal to this thread block
    {
        // (s,t) -- tile coordinates in the grid of tiles (score matrix)
        int tbeg = max(0, d - (tcols - 1));
        int tend = min(d, trows - 1);

        // map a tile on the current diagonal of tiles to this thread block
        // +   then go to the next tile on the diagonal with stride equal to the number of thread blocks in the thread grid
        int t = tbeg + blockIdx.x;
        // initialize the score matrix tile
        {
            // position of the top left element of the current tile in the score matrix
            int ibeg = (1 + (t)*tileBy) - 1;
            int jbeg = (1 + (d - t) * tileBx) - 1;
            // the number of columns in the score matrix
            int adjcols = 1 + tcols * tileBx;

            // current thread position in the tile
            int i = threadIdx.x / (tileBx + 1);
            int j = threadIdx.x % (tileBx + 1);
            // stride on the current thread position in the tile, equal to the number of threads in this thread block
            // +   it is split into row and column increments for the thread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
            int di = blockDim.x / (tileBx + 1);
            int dj = blockDim.x % (tileBx + 1);

            // while the current thread maps onto an element in the tile
            while (i < (1 + tileBy))
            {
                // copy the current element from the global score matrix to the tile
                el(tile, 1 + tileBx, i, j) = el(score_gpu, adjcols, ibeg + i, jbeg + j);

                // map the current thread to the next tile element
                i += di;
                j += dj;
                // if the column index is out of bounds, increase the row index by one and wrap around the column index
                if (j >= (1 + tileBx))
                {
                    i++;
                    j -= (1 + tileBx);
                }
            }
        }

        // all threads in this block should finish initializing this tile in shared memory
        __syncthreads();

        // calculate the tile elements
        // +   only threads in the first warp from this block are active here, other warps have to wait
        if (threadIdx.x < warpsz)
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
                int pend = min(d, rows - 1);
                // position of the current thread's element on the tile diagonal
                int p = pbeg + threadIdx.x;

                // if the thread maps onto an element on the current tile diagonal
                if (p <= pend)
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

        // all threads in this block should finish saving this tile
        __syncthreads();
    }
}

// parallel gpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Gpu4_Ml_DiagDiag2Pass(NwParams &pr, NwInput &nw, NwResult &res)
{
    // tile sizes for kernels A and B
    // +   tile A should have one dimension be a multiple of the warp size for full memory coallescing
    // +   tile B must have one dimension fixed to the number of threads in a warp
    unsigned tileAx;
    unsigned tileAy;
    unsigned tileBx;
    unsigned tileBy;

    // get the parameter values
    try
    {
        tileAx = pr["tileAx"].curr();
        tileAy = pr["tileAy"].curr();
        tileBx = pr["tileBx"].curr();
        tileBy = pr["tileBy"].curr();
    }
    catch (const std::out_of_range &ex)
    {
        return NwStat::errorInvalidValue;
    }

    if (tileBx != nw.warpsz && tileBy != nw.warpsz)
    {
        return NwStat::errorInvalidValue;
    }

    // adjusted gpu score matrix dimensions
    // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile B size (in order to be evenly divisible)
    int adjrows = 1 + tileBy * ceil(float(nw.adjrows - 1) / tileBy);
    int adjcols = 1 + tileBx * ceil(float(nw.adjcols - 1) / tileBx);
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
    Stopwatch &sw = res.sw_align;
    sw.start();

    // reserve space in the ram and gpu global memory
    try
    {
        nw.seqX_gpu.init(adjcols);
        nw.seqY_gpu.init(adjrows);
        nw.score_gpu.init(adjrows * adjcols);

        nw.score.init(nw.adjrows * nw.adjcols);
    }
    catch (const std::exception &ex)
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

    // launch kernel A
    {
        // calculate grid dimensions for kernel A
        dim3 gridA{};
        gridA.y = ceil(float(adjrows) / tileAy);
        gridA.x = ceil(float(adjcols) / tileAx);
        // block dimensions for kernel A
        unsigned threadsPerBlockA = min(nw.maxThreadsPerBlock, tileAy * tileAx);
        dim3 blockA{threadsPerBlockA};

        // calculate size of shared memory per block in bytes
        int shmemsz = (
            /*subst[][]*/ nw.substsz * nw.substsz * sizeof(int)
            /*seqX[]*/
            + tileAx * sizeof(int)
            /*seqY[]*/
            + tileAy * sizeof(int));

        // create variables for gpu arrays in order to be able to take their addresses
        int *seqX_gpu = nw.seqX_gpu.data();
        int *seqY_gpu = nw.seqY_gpu.data();
        int *score_gpu = nw.score_gpu.data();
        int *subst_gpu = nw.subst_gpu.data();

        // group arguments to be passed to kernel A
        void *kargs[]{
            &seqX_gpu,
            &seqY_gpu,
            &score_gpu,
            &subst_gpu,
            &adjrows,
            &adjcols,
            &nw.substsz,
            &nw.indel,
            &tileAx,
            &tileAy};

        // launch the kernel in the given stream (don't statically allocate shared memory)
        if (cudaSuccess != (cudaStatus = cudaLaunchKernel((void *)Nw_Gpu4_KernelA, gridA, blockA, kargs, shmemsz, nullptr /*stream*/)))
        {
            return NwStat::errorKernelFailure;
        }
    }

    // wait for the gpu to finish before going to the next step
    if (cudaSuccess != (cudaStatus = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // measure calculation init time
    sw.lap("calc-1");

    //  x x x x x x       x x x x x x       x x x x x x
    //  x / / / . .       x . . . / /       x . . . . .|/ /
    //  x / / . . .   +   x . . / / .   +   x . . . . /|/
    //  x / . . . .       x . / / . .       x . . . / /|
    // launch kernel B for each minor tile diagonal of the score matrix
    {
        // grid and block dimensions for kernel B
        dim3 gridB{};
        dim3 blockB{};
        // the number of tiles per row and column of the score matrix
        int trows = ceil(float(adjrows - 1) / tileBy);
        int tcols = ceil(float(adjcols - 1) / tileBx);

        // calculate size of shared memory per block in bytes
        int shmemsz = (
            /*tile[]*/ (1 + tileBy) * (1 + tileBx) * sizeof(int));

        // for all minor tile diagonals in the score matrix (excluding the header row and column)
        for (int d = 0; d < tcols - 1 + trows; d++)
        {
            // calculate grid and block dimensions for kernel B
            {
                int pbeg = max(0, d - (tcols - 1));
                int pend = min(d, trows - 1);

                // the number of elements on the current diagonal
                int dsize = pend - pbeg + 1;

                // take the number of threads on the largest diagonal of the tile
                // +   multiply by the number of half warps in the larger dimension for faster writing to global gpu memory
                blockB.x = nw.warpsz * ceil(max(tileBy, tileBx) * 2. / nw.warpsz);

                // take the number of blocks on the current score matrix diagonal as the only dimension
                // +   launch at least one block on the x axis
                gridB.x = dsize;
            }

            // create variables for gpu arrays in order to be able to take their addresses
            int *seqX_gpu = nw.seqX_gpu.data();
            int *seqY_gpu = nw.seqY_gpu.data();
            int *score_gpu = nw.score_gpu.data();
            int *subst_gpu = nw.subst_gpu.data();

            // group arguments to be passed to kernel B
            void* kargs[]{
                &score_gpu,
                &nw.indel,
                &nw.warpsz,
                &trows,
                &tcols,
                &tileBx,
                &tileBy,
                &d };

            // launch the kernel B in the given stream (don't statically allocate shared memory)
            if (cudaSuccess != (cudaStatus = cudaLaunchKernel((void *)Nw_Gpu4_KernelB, gridB, blockB, kargs, shmemsz, nullptr /*stream*/)))
            {
                return NwStat::errorKernelFailure;
            }
        }
    }

    // wait for the gpu to finish before going to the next step
    if (cudaSuccess != (cudaStatus = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // measure calculation time
    sw.lap("calc-2");

    // save the calculated score matrix
    if (cudaSuccess != (cudaStatus = memTransfer(nw.score, nw.score_gpu, nw.adjrows, nw.adjcols, adjcols)))
    {
        return NwStat::errorMemoryTransfer;
    }

    // measure memory transfer time
    sw.lap("cpy-host");

    return NwStat::success;
}
