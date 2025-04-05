#include "common.hpp"

// Initialize the score matrix's header row and column.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
__global__ static void Nw_Gpu7_KernelA(
    int* const tileHrowMat_gpu,
    int* const tileHcolMat_gpu,
    const int trows,
    const int tcols,
    const int tileBx,
    const int tileBy,
    const int indel)
{
    int tid = (blockDim.x * blockIdx.x + threadIdx.x);

    // Initialize score matrix header row.
    {
        // The tile header row and column have an extra zeroth element that needs initializing.
        // That's why we divide by (1 + ...).
        int jTile = tid / (1 + tileBx);
        int iTile = 0;

        if (jTile < tcols)
        {
            int jTileElem = tid % (1 + tileBx);
            int j = jTile * tileBx + jTileElem;

            int kHrow = tcols * iTile + jTile;
            int kHrowElem = kHrow * (1 + tileBx) + 0 + jTileElem;

            tileHrowMat_gpu[kHrowElem] = j * indel;
        }
    }

    // Initialize score matrix header column.
    {
        // The tile header row and column have an extra zeroth element that needs initializing.
        // That's why we divide by (1 + ...).
        int jTile = 0;
        int iTile = tid / (1 + tileBy);

        if (iTile < trows)
        {
            int iTileElem = tid % (1 + tileBy);
            int i = iTile * tileBy + iTileElem;

            int kHcol = tcols * iTile + jTile; // row-major
            int kHcolElem = kHcol * (1 + tileBy) + 0 + iTileElem;

            tileHcolMat_gpu[kHcolElem] = i * indel;
        }
    }
}

// Calculate the score matrix.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
__global__ static void Nw_Gpu7_KernelB(
    // standard params
    const int* const seqX_gpu,
    const int* const seqY_gpu,
    int* const tileHrowMat_gpu,
    int* const tileHcolMat_gpu,
    const int* const subst_gpu,
    const int substsz,
    const int indel,
    // params related to tile B
    const int trows,
    const int tcols,
    const int tileBx,
    const int tileBy,
    const int d)
{
    extern __shared__ int shmem[/* substsz*substsz + tileBx + tileBy + (1+tileBy)*(1+tileBx) */];
    int* const subst /*[substsz*substsz]*/ = shmem + 0;
    int* const seqX /*[tileBx]*/ = subst + substsz * substsz;
    int* const seqY /*[tileBy]*/ = seqX + tileBx;
    int* const tile /*[(1+tileBy)*(1+tileBx)]*/ = seqY + tileBy;

    // Initialize the substitution matrix in shared memory.
    {
        int i = threadIdx.x;
        while (i < substsz * substsz)
        {
            el(subst, substsz, 0, i) = el(subst_gpu, substsz, 0, i);
            i += blockDim.x;
        }
    }

    // Block should finish initializing substitution matrix in shared memory.
    __syncthreads();

    // Tile schematic:
    //       x x x x x
    //       | | | | |
    //     h h h h h h
    // y --h . . . . .
    // y --h . . . . .
    // y --h . . . . .
    // Observe that the X and Y seqences don't need to be extended by 1 to the left and by 1 to the top.

    // Map a tile on the current tile diagonal to this thread block.
    // (d,t) -- tile coordinates in the grid of tiles (score matrix).
    int tbeg = max(0, d - (tcols - 1));
    int tend = min(d + 1, trows);
    int t = tbeg + blockIdx.x;

    // Initialize the tile's window into the global X sequence.
    {
        int jbeg = 1 + (d - t) * tileBx;

        int j = threadIdx.x;
        while (j < tileBx)
        {
            seqX[j] = seqX_gpu[jbeg + j];
            j += blockDim.x;
        }
    }

    // Initialize the tile's window into the global Y sequence.
    {
        int ibeg = 1 + (t)*tileBy;

        int i = threadIdx.x;
        while (i < tileBy)
        {
            seqY[i] = seqY_gpu[ibeg + i];
            i += blockDim.x;
        }
    }

    // Initialize the tile's header row in shared memory.
    {
        int iTile = t;
        int jTile = d - t;
        int kHrow = tcols * iTile + jTile;
        int jbeg = kHrow * (1 + tileBx);

        int j = threadIdx.x;
        while (j < 1 + tileBx)
        {
            el(tile, 1 + tileBx, 0, j) = tileHrowMat_gpu[jbeg + j];
            j += blockDim.x;
        }
    }

    // Initialize the tile's header column in shared memory.
    {
        int iTile = t;
        int jTile = d - t;
        int kHcol = tcols * iTile + jTile; // row-major
        int ibeg = kHcol * (1 + tileBy);

        int i = threadIdx.x;
        while (i < 1 + tileBy)
        {
            el(tile, 1 + tileBx, i, 0) = tileHcolMat_gpu[ibeg + i];
            i += blockDim.x;
        }
    }

    // Block should finish initializing the tile's windows into the X and Y sequences and its header row and column.
    __syncthreads();

    // Partially calculate the tile's elements by doing only neighbour-independent work.
    // The expression for calculating the final element value becomes simpler later on.
    {
        // Current thread position in the tile.
        int i = threadIdx.x / tileBx;
        int j = threadIdx.x % tileBx;
        // Position increment is split into row and column increments to avoid using division and modulo operator in the inner loop.
        int di = blockDim.x / tileBx;
        int dj = blockDim.x % tileBx;

        while (i < tileBy)
        {
            el(tile, 1 + tileBx, 1 + i, 1 + j) = el(subst, substsz, seqY[i], seqX[j]) - indel;

            i += di;
            j += dj;
            if (j >= tileBx)
            {
                i++;
                j -= tileBx;
            }
        }
    }

    // Block should finish partially calculating this tile's elements.
    __syncthreads();

    // Calculate the tile elements.
    // Only threads in the first warp from this block are active here, other warps have to wait.
    if (threadIdx.x < warpSize)
    {
        // Number of rows and columns in the tile (without its header row and column).
        int rows = tileBy;
        int cols = tileBx;

        // Cases:     1.                2.                3.
        //  x x x x x x       x x x x x x       x x x x x x
        //  x / / . . .       x . . / / /       x . . . . .|/ /
        //  x / . . . .   +   x . / / / .   +   x . . . . /|/
        //  x . . . . .       x / / / . .       x . . . / /|

        // For all element (minor) diagonals in the tile.
        // (s,p) -- element coordinates in the tile (when traversing in minor-diagonal order).
        for (int s = 0; s < rows + cols - 1; s++)
        {
            int pbeg = max(0, s - (cols - 1));
            int pend = min(s + 1, rows);
            int p = pbeg + threadIdx.x;

            // Only if the thread maps onto an element on the current tile diagonal (case 3.).
            if (p < pend)
            {
                // Element coordinates in the tile extended with header row and column (when traversing in minor-diagonal order).
                int i = 1 + (p);
                int j = 1 + (s - p);

                // Calculate the current element's value.
                // Subtract the insert delete cost from the result, since the kernel A added that value in all movement directions implicitly.
                int temp1 = el(tile, 1 + tileBx, i - 1, j - 1) + el(tile, 1 + tileBx, i, j);
                int temp2 = max(el(tile, 1 + tileBx, i - 1, j), el(tile, 1 + tileBx, i, j - 1));
                el(tile, 1 + tileBx, i, j) = max(temp1, temp2) + indel;
            }

            // Warp should finish calculating the current element diagonal.
            __syncwarp();
        }
    }

    // Block should finish calculating this tile.
    __syncthreads();

    // Save the tile last row to the tile header row matrix.
    {
        int iTileBelow = (t) + 1;
        int jTileBelow = (d - t);

        // Bottommost tile should not save its row.
        if (iTileBelow < trows)
        {
            int kHrowBelow = tcols * iTileBelow + jTileBelow;
            int jbeg = kHrowBelow * (1 + tileBx);

            int j = threadIdx.x;
            while (j < 1 + tileBx)
            {
                tileHrowMat_gpu[jbeg + j] = el(tile, 1 + tileBx, tileBy /*last row*/, j);
                j += blockDim.x;
            }
        }
    }

    // Save the tile last column to the tile header column matrix.
    {
        int iTileRight = (t);
        int jTileRight = (d - t) + 1;

        // Rightmost tile should not save its column.
        if (jTileRight < tcols)
        {
            int kHcol = tcols * iTileRight + jTileRight; // row-major
            int ibeg = kHcol * (1 + tileBy);

            int i = threadIdx.x;
            while (i < 1 + tileBy)
            {
                tileHcolMat_gpu[ibeg + i] = el(tile, 1 + tileBx, i, tileBx /*last column*/);
                i += blockDim.x;
            }
        }
    }
}

// Parallel gpu implementation of the Needleman-Wunsch algorithm.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
//
// Assumes that the row sequence (X) is longer or equal in length to the column sequence (Y).
NwStat NwAlign_Gpu7_Mlsp_DiagDiag(NwParams& pr, NwInput& nw, NwResult& res)
{
    // Number of threads per block for kernel A.
    int threadsPerBlockA = {};
    // Tile B must have one dimension fixed to the number of threads in a warp.
    int tileBx = {};
    int tileBy = nw.warpsz;
    // Reduce the number of warps in the thread block in kernel B.
    int warpDivFactorB = {};

    try
    {
        threadsPerBlockA = pr["threadsPerBlockA"].curr();
        tileBx = pr["tileBx"].curr();
        warpDivFactorB = pr["warpDivFactorB"].curr();
    }
    catch (const std::out_of_range&)
    {
        return NwStat::errorInvalidValue;
    }

    // Adjusted gpu score matrix dimensions.
    // The matrix dimensions are rounded up to 1 + <the nearest multiple of the tile B size>.
    int adjrows = 1 + tileBy * (int)ceil(float(nw.adjrows - 1) / tileBy);
    int adjcols = 1 + tileBx * (int)ceil(float(nw.adjcols - 1) / tileBx);
    // Special case when very small and very large sequences are compared.
    if (adjrows == 1)
    {
        adjrows = 1 + tileBy;
    }
    if (adjcols == 1)
    {
        adjcols = 1 + tileBx;
    }
    // The number of tiles per row and column of the score matrix.
    int trows = (int)ceil(float(adjrows - 1) / tileBy);
    int tcols = (int)ceil(float(adjcols - 1) / tileBx);

    // Start the timer.
    Stopwatch& sw = res.sw_align;
    sw.start();

    // Allocate space in the ram and gpu global memory.
    try
    {
        nw.seqX_gpu.init(adjcols);
        nw.seqY_gpu.init(adjrows);
        nw.tileHrowMat_gpu.init(trows * tcols * (1 + tileBx));
        nw.tileHcolMat_gpu.init(trows * tcols * (1 + tileBy));

        nw.tileHrowMat.init(trows * tcols * (1 + tileBx));
        nw.tileHcolMat.init(trows * tcols * (1 + tileBy));
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // Measure allocation time.
    sw.lap("align.alloc");

    // Copy data from host to device.
    if (cudaSuccess != (res.cudaStat = memTransfer(nw.seqX_gpu, nw.seqX, nw.adjcols)))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (res.cudaStat = memTransfer(nw.seqY_gpu, nw.seqY, nw.adjrows)))
    {
        return NwStat::errorMemoryTransfer;
    }
    // Also initialize padding, since it is used to access elements in the substitution matrix.
    if (cudaSuccess != (res.cudaStat = memSet(nw.seqX_gpu, nw.adjcols, 0 /*value*/)))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (res.cudaStat = memSet(nw.seqY_gpu, nw.adjrows, 0 /*value*/)))
    {
        return NwStat::errorMemoryTransfer;
    }

    // Measure memory transfer time.
    sw.lap("align.cpy_dev");

    //  x x x x x x
    //  x . . . . .
    //  x . . . . .
    //  x . . . . .
    // Launch kernel A to initialize the score matrix's header row and column.
    // The score matrix is represented as two matrices (row-major order):
    // + tile header row matrix,
    // + tile header column matrix.
    {
        // Size of shared memory per block in bytes.
        int shmemByteSize = (0);

        dim3 blockDim {};
        blockDim.x = threadsPerBlockA;

        // Calculate the necessary number of blocks to cover the larger score matrix dimension.
        dim3 gridDim {};
        {
            int tileHrowMat_RowElemCount = tcols * (1 + tileBx);
            int tileHcolMat_ColElemCount = trows * (1 + tileBy);
            int largerDimElemCount = max2(tileHrowMat_RowElemCount, tileHcolMat_ColElemCount);
            gridDim.x = (int)ceil(float(largerDimElemCount) / threadsPerBlockA);
        }

        int* tileHrowMat_gpu = nw.tileHrowMat_gpu.data();
        int* tileHcolMat_gpu = nw.tileHcolMat_gpu.data();

        void* kargs[] {
            &tileHrowMat_gpu,
            &tileHcolMat_gpu,
            &trows,
            &tcols,
            &tileBx,
            &tileBy,
            &nw.indel};

        if (cudaSuccess != (res.cudaStat = cudaLaunchKernel((void*)Nw_Gpu7_KernelA, gridDim, blockDim, kargs, shmemByteSize, cudaStreamDefault)))
        {
            return NwStat::errorKernelFailure;
        }
    }

    // Wait for the gpu to finish before going to the next step.
    if (cudaSuccess != (res.cudaStat = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // Measure header initialization time.
    sw.lap("align.init_hdr");

    //  x x x x x x       x x x x x x       x x x x x x
    //  x / / / . .       x . . . / /       x . . . . .|/ /
    //  x / / . . .   +   x . . / / .   +   x . . . . /|/
    //  x / . . . .       x . / / . .       x . . . / /|
    // Launch kernel B for each (minor) tile diagonal of the score matrix.
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

        // Size of shared memory per block in bytes.
        int shmemsz = (
            /*subst[]*/ nw.substsz * nw.substsz * sizeof(int)
            /*seqX[]*/
            + tileBx * sizeof(int)
            /*seqY[]*/
            + tileBy * sizeof(int)
            /*tile[]*/
            + (1 + tileBy) * (1 + tileBx) * sizeof(int));

        // The number of threads should be divisible by the warp size.
        // But for performance reasons, we don't need all those single-use warps, just half of them (or some other fraction).
        // That way the thread block can be smaller while doing the same amount of work.
        dim3 blockB {};
        {
            int warps = (int)ceil(float(max(tileBx, tileBx)) / nw.warpsz / warpDivFactorB);
            blockB.x = nw.warpsz * warps;
        }

        // For all (minor) tile diagonals in the score matrix.
        for (int d = 0; d < tcols - 1 + trows; d++)
        {
            dim3 gridB {};
            {
                int tbeg = max(0, d - (tcols - 1));
                int tend = min(d + 1, trows);
                // Number of tiles on the current (minor) tile diagonal.
                int dsize = tend - tbeg;

                gridB.x = dsize;
            }

            int* seqX_gpu = nw.seqX_gpu.data();
            int* seqY_gpu = nw.seqY_gpu.data();
            int* tileHrowMat_gpu = nw.tileHrowMat_gpu.data();
            int* tileHcolMat_gpu = nw.tileHcolMat_gpu.data();
            int* subst_gpu = nw.subst_gpu.data();

            void* kargs[] {
                // standard params
                &seqX_gpu,
                &seqY_gpu,
                &tileHrowMat_gpu,
                &tileHcolMat_gpu,
                &subst_gpu,
                &nw.substsz,
                &nw.indel,
                // params related to tile B
                &trows,
                &tcols,
                &tileBx,
                &tileBy,
                &d};

            if (cudaSuccess != (res.cudaStat = cudaLaunchKernel((void*)Nw_Gpu7_KernelB, gridB, blockB, kargs, shmemsz, stream)))
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

    // Wait for the gpu to finish before going to the next step.
    if (cudaSuccess != (res.cudaStat = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // Measure calculation time.
    sw.lap("align.calc");

    // Save the calculated score matrix.
    nw.tileHdrMatRows = trows;
    nw.tileHdrMatCols = tcols;
    nw.tileHrowLen = 1 + tileBx;
    nw.tileHcolLen = 1 + tileBy;

    if (cudaSuccess != (res.cudaStat = memTransfer(nw.tileHrowMat, nw.tileHrowMat_gpu, trows * tcols * (1 + tileBx))))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (res.cudaStat = memTransfer(nw.tileHcolMat, nw.tileHcolMat_gpu, trows * tcols * (1 + tileBy))))
    {
        return NwStat::errorMemoryTransfer;
    }

    // Measure memory transfer time.
    sw.lap("align.cpy_host");

    return NwStat::success;
}
