#include "common.hpp"

// Initialize the score matrix's header row and column.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
__global__ static void Nw_Gpu9_KernelA(
    int *const tileHrowMat_gpu,
    int *const tileHcolMat_gpu,
    const int trows,
    const int tcols,
    const unsigned tileBx,
    const unsigned tileBy,
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
__global__ static void Nw_Gpu9_KernelB(
    // standard params
    const int *const seqX_gpu,
    const int *const seqY_gpu,
    int *const tileHrowMat_gpu,
    int *const tileHcolMat_gpu,
    const int *const subst_gpu,
    const int substsz,
    const int indel,
    const int warpsz,
    // params related to tile B
    const int trows,
    const int tcols,
    const unsigned tileBx,
    const unsigned tileBy,
    const int d)
{
    extern __shared__ int shmem[/* substsz*substsz + tileBx + tileBy + (1+tileBx) + (1+tileBy) */];
    int *const subst /*[substsz*substsz]*/ = shmem + 0;
    int *const seqX /*[tileBx]*/ = subst + substsz * substsz;
    int *const seqY /*[tileBy]*/ = seqX + tileBx;
    int *const tileHrow /*[(1+tileBx)]*/ = seqY + tileBy;
    int *const tileHcol /*[(1+tileBy)]*/ = tileHrow + (1 + tileBx);

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
            tileHrow[j] = tileHrowMat_gpu[jbeg + j];
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
            tileHcol[i] = tileHcolMat_gpu[ibeg + i];
            i += blockDim.x;
        }
    }

    // Block should finish initializing the tile's windows into the X and Y sequences and its header row and column.
    __syncthreads();

    // Calculate the tile elements.
    // Only threads in the first warp from this block are active here, other warps have to wait.
    if (threadIdx.x < warpsz)
    {
        // Tile shematic:
        //               |h  h  h  h  h  h  h  h  h  |.  .  .  .  .
        //             . |h  .  .  .  .  x  x  o  .  |.  .  .  .
        //          .  . |h  .  .  . |ul u| o  .  .  |.  .  .
        //       .  .  . |h  .  .  x |l  c| .  .  .  |.  .
        //    .  .  .  . |h  .  x  x  o  .  .  .  .  |.
        // .  .  .  .  . |h  .  x  o  .  .  .  .  .  |
        // Observe that we only need three elements from the previous two diagonals, to slide the calculation window to the right each iteration.
        // Therefore each thread keeps them in its registers.
        //
        // Warp thread 0 will read from the tile header row each iteration, and share that value with the other threads.
        // Warp thread 31 will write its current value to the last tile row (reusing tile header row for this purpose).
        // Each warp thread, upon reaching the tile right boundary, will write its current value to the last tile column (reusing tile header column for this purpose).

        // Save the up-left and left element before any calculation to simplify the algorithm.
        // The thread now only needs to worry how it's going to get its 'up' element.
        //
        // Observe that we initialized the elements as if we slid once to the left (in the reverse direction).
        // That's because the code below first slides to the right, and then calculates the current element.
        int upleft = 0;
        int left = 0;
        int up = tileHcol[1 + (threadIdx.x - 1)];
        int curr = tileHcol[1 + (threadIdx.x)];

        // The current thread's position in the tile.
        const int i = threadIdx.x;
        int j = 0 - threadIdx.x;
        // All threads have to calculate the same number of elements. Otherwise, warp sync would deadlock.
        // "Artificial" elements outside the tile boundaries are of no concern for correctness.
        int jend = j + (tileBx + (tileBy - 1));

        // Zeroth warp thread should update the upper-left corner header elements.
        // These will not be updated during the calculation, because none of the threads will map to them.
        if (i == 0 && j == 0)
        {
            tileHrow[0] = tileHcol[1 + (tileBy - 1)];
            tileHcol[0] = tileHrow[1 + (tileBx - 1)];
        }

        while (j < jend)
        {
            // Prevent losing the 'left' header element before the first "real" calculation.
            if (j >= 0 && j < tileBx)
            {
                upleft = up;
                left = curr;
            }

            // Initialize 'up' elements for all warp threads except the zeroth.
            // (Copies from a lane with lower thread id relative to the caller thread id.
            // Also syncs the threads in the warp.)
            up = __shfl_up_sync(/*mask*/ 0xffffffff, /*var*/ curr, /*delta*/ 1, /*width*/ warpsz);

            // Initialize 'up' element for the zeroth thread.
            // For "artificial" elements, initialize to 0 so that behavior is deterministic.
            if (i == 0)
            {
                up = (j >= 0 && j < tileBx) ? tileHrow[1 + j] : 0;
            }

            if (i >= 0 && i < tileBy && j >= 0 && j < tileBx)
            {
                curr = upleft + el(subst, substsz, seqY[i], seqX[j]); // MOVE DOWN-RIGHT
                curr = max(curr, up + indel);                         // MOVE DOWN
                curr = max(curr, left + indel);                       // MOVE RIGHT

                if (j == tileBx - 1)
                {
                    // When a warp thread reaches the right tile boundary, save that element to the next header column.
                    tileHcol[1 + i] = curr;
                }

                if (i == tileBy - 1)
                {
                    // Last warp thread should save its current element to the next header row.
                    tileHrow[1 + j] = curr;
                }
            }

            j++;
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
                tileHrowMat_gpu[jbeg + j] = tileHrow[j];
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
                tileHcolMat_gpu[ibeg + i] = tileHcol[i];
                i += blockDim.x;
            }
        }
    }

    // Block should finish saving tile last row/column to tile header row/column matrix.
    __syncthreads();
}

// Parallel gpu implementation of the Needleman-Wunsch algorithm.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
//
// Assumes that the row sequence (X) is longer or equal in length to the column sequence (Y).
NwStat NwAlign_Gpu9_Mlsp_DiagDiagDiag(NwParams &pr, NwInput &nw, NwResult &res)
{
    // Number of threads per block for kernel A.
    int threadsPerBlockA = {};
    // Tile B is a multiple of subtiles B in both dimensions.
    int tileBx = {};
    int tileBy = {};
    int subtileRows = {};
    int subtileCols = {};
    // Subtile B must have one dimension be a multiple of the number of threads in a warp.
    int subtileBx = {};
    int subtileBy = nw.warpsz;

    try
    {
        threadsPerBlockA = pr["threadsPerBlockA"].curr();
        subtileRows = pr["subtileRows"].curr();
        subtileCols = pr["subtileCols"].curr();
        subtileBx = pr["subtileBx"].curr();
    }
    catch (const std::out_of_range &)
    {
        return NwStat::errorInvalidValue;
    }

    tileBx = subtileCols * subtileBx;
    tileBy = subtileRows * subtileBy;

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
    Stopwatch &sw = res.sw_align;
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
    catch (const std::exception &)
    {
        return NwStat::errorMemoryAllocation;
    }

    // Measure allocation time.
    sw.lap("alloc");

    // Copy data from host to device.
    if (cudaSuccess != (cudaStatus = memTransfer(nw.seqX_gpu, nw.seqX, nw.adjcols)))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (cudaStatus = memTransfer(nw.seqY_gpu, nw.seqY, nw.adjrows)))
    {
        return NwStat::errorMemoryTransfer;
    }
    // Also initialize padding, since it is used to access elements in the substitution matrix.
    if (cudaSuccess != (cudaStatus = memSet(nw.seqX_gpu, nw.adjcols, 0 /*value*/)))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (cudaStatus = memSet(nw.seqY_gpu, nw.adjrows, 0 /*value*/)))
    {
        return NwStat::errorMemoryTransfer;
    }

    // Measure memory transfer time.
    sw.lap("cpy-dev");

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

        dim3 blockDim{};
        blockDim.x = threadsPerBlockA;

        // Calculate the necessary number of blocks to cover the larger score matrix dimension.
        dim3 gridDim{};
        {
            int tileHrowMat_RowElemCount = tcols * (1 + tileBx);
            int tileHcolMat_ColElemCount = trows * (1 + tileBy);
            int largerDimElemCount = max2(tileHrowMat_RowElemCount, tileHcolMat_ColElemCount);
            gridDim.x = (int)ceil(float(largerDimElemCount) / threadsPerBlockA);
        }

        int *tileHrowMat_gpu = nw.tileHrowMat_gpu.data();
        int *tileHcolMat_gpu = nw.tileHcolMat_gpu.data();

        void *kargs[]{
            &tileHrowMat_gpu,
            &tileHcolMat_gpu,
            &trows,
            &tcols,
            &tileBx,
            &tileBy,
            &nw.indel};

        if (cudaSuccess != (cudaStatus = cudaLaunchKernel((void *)Nw_Gpu9_KernelA, gridDim, blockDim, kargs, shmemByteSize, nullptr /*stream*/)))
        {
            return NwStat::errorKernelFailure;
        }
    }

    // Wait for the gpu to finish before going to the next step.
    if (cudaSuccess != (cudaStatus = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // Measure header initialization time.
    sw.lap("init-hdr");

    //  x x x x x x       x x x x x x       x x x x x x
    //  x / / / . .       x . . . / /       x . . . . .|/ /
    //  x / / . . .   +   x . . / / .   +   x . . . . /|/
    //  x / . . . .       x . / / . .       x . . . / /|
    // Launch kernel B for each (minor) tile diagonal of the score matrix.
    {
        // Size of shared memory per block in bytes.
        int shmemsz = (
            /*subst[]*/ nw.substsz * nw.substsz * sizeof(int)
            /*seqX[]*/
            + tileBx * sizeof(int)
            /*seqY[]*/
            + tileBy * sizeof(int)
            /*tileHrow[]*/
            + (1 + tileBx) * sizeof(int)
            /*tileHcol[]*/
            + (1 + tileBy) * sizeof(int));

        dim3 blockB{};
        {
            blockB.x = nw.warpsz * subtileRows;
        }

        // For all (minor) tile diagonals in the score matrix.
        for (int d = 0; d < tcols - 1 + trows; d++)
        {
            dim3 gridB{};
            {
                int tbeg = max(0, d - (tcols - 1));
                int tend = min(d + 1, trows);
                // Number of tiles on the current (minor) tile diagonal.
                int dsize = tend - tbeg;

                gridB.x = dsize;
            }

            int *seqX_gpu = nw.seqX_gpu.data();
            int *seqY_gpu = nw.seqY_gpu.data();
            int *tileHrowMat_gpu = nw.tileHrowMat_gpu.data();
            int *tileHcolMat_gpu = nw.tileHcolMat_gpu.data();
            int *subst_gpu = nw.subst_gpu.data();

            void *kargs[]{
                // standard params
                &seqX_gpu,
                &seqY_gpu,
                &tileHrowMat_gpu,
                &tileHcolMat_gpu,
                &subst_gpu,
                &nw.substsz,
                &nw.indel,
                &nw.warpsz,
                // params related to tile B
                &trows,
                &tcols,
                &tileBx,
                &tileBy,
                &d};

            if (cudaSuccess != (cudaStatus = cudaLaunchKernel((void *)Nw_Gpu9_KernelB, gridB, blockB, kargs, shmemsz, nullptr /*stream*/)))
            {
                return NwStat::errorKernelFailure;
            }
        }
    }

    // Wait for the gpu to finish before going to the next step.
    if (cudaSuccess != (cudaStatus = cudaDeviceSynchronize()))
    {
        return NwStat::errorKernelFailure;
    }

    // Measure calculation time.
    sw.lap("calc-1");

    // Save the calculated score matrix.
    nw.tileHdrMatRows = trows;
    nw.tileHdrMatCols = tcols;
    nw.tileHrowLen = 1 + tileBx;
    nw.tileHcolLen = 1 + tileBy;

    if (cudaSuccess != (cudaStatus = memTransfer(nw.tileHrowMat, nw.tileHrowMat_gpu, trows * tcols * (1 + tileBx))))
    {
        return NwStat::errorMemoryTransfer;
    }
    if (cudaSuccess != (cudaStatus = memTransfer(nw.tileHcolMat, nw.tileHcolMat_gpu, trows * tcols * (1 + tileBy))))
    {
        return NwStat::errorMemoryTransfer;
    }

    // Measure memory transfer time.
    sw.lap("cpy-host");

    return NwStat::success;
}
