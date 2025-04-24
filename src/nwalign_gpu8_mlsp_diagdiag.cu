#include "defer.hpp"
#include "math.hpp"
#include "nw_fns.hpp"
#include "nwalign_shared.hpp"
#include "run_types.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// Initialize the score matrix's header row and column.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
__global__ static void Nw_Gpu8_KernelA(
    int* const tileHrowMat_gpu,
    int* const tileHcolMat_gpu,
    const int trows,
    const int tcols,
    const int tileBx,
    const int tileBy,
    const int gapoCost)
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

            tileHrowMat_gpu[kHrowElem] = j * gapoCost;
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

            tileHcolMat_gpu[kHcolElem] = i * gapoCost;
        }
    }
}

// Calculate the score matrix.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
__global__ static void Nw_Gpu8_KernelB(
    // standard params
    const int* const seqX_gpu,
    const int* const seqY_gpu,
    int* const tileHrowMat_gpu,
    int* const tileHcolMat_gpu,
    const int* const subst_gpu,
    const int substsz,
    const int gapoCost,
    // params related to tile B
    const int trows,
    const int tcols,
    const int tileBx,
    const int tileBy,
    const int d)
{
    extern __shared__ int shmem[/* substsz*substsz + tileBx + tileBy + (1+tileBx) + (1+tileBy) */];
    int* const subst /*[substsz*substsz]*/ = shmem + 0;
    int* const seqX /*[tileBx]*/ = subst + substsz * substsz;
    int* const seqY /*[tileBy]*/ = seqX + tileBx;
    int* const tileHrow /*[(1+tileBx)]*/ = seqY + tileBy;
    int* const tileHcol /*[(1+tileBy)]*/ = tileHrow + (1 + tileBx);

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
    if (threadIdx.x < warpSize)
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
            up = __shfl_up_sync(0xffffffff /*mask*/, curr /*var*/, 1 /*delta*/, warpSize /*width*/);

            // Initialize 'up' element for the zeroth thread.
            // For "artificial" elements, initialize to 0 so that behavior is deterministic.
            if (i == 0)
            {
                up = (j >= 0 && j < tileBx) ? tileHrow[1 + j] : 0;
            }

            if (/*i >= 0 && i < tileBy && */ j >= 0 && j < tileBx)
            {
                curr = upleft + el(subst, substsz, seqY[i], seqX[j]); // MOVE DOWN-RIGHT
                curr = max(curr, up + gapoCost);                      // MOVE DOWN
                curr = max(curr, left + gapoCost);                    // MOVE RIGHT

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
}

// Parallel gpu implementation of the Needleman-Wunsch algorithm.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
//
// Assumes that the row sequence (X) is longer or equal in length to the column sequence (Y).
NwStat NwAlign_Gpu8_Mlsp_DiagDiag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res)
{
    // Number of threads per block for kernel A.
    int threadsPerBlockA {};
    // Tile B must have one dimension fixed to the number of threads in a warp.
    int tileBx {};
    int tileBy {nw.warpsz};
    // Reduce the number of warps in the thread block in kernel B.
    int warpDivFactorB {};

    // Get parameters.
    try
    {
        threadsPerBlockA = pr.at("threadsPerBlockA").curr();
        tileBx = pr.at("tileBx").curr();
        warpDivFactorB = pr.at("warpDivFactorB").curr();

        if ((threadsPerBlockA < nw.warpsz || threadsPerBlockA > nw.maxThreadsPerBlock) ||
            (tileBx < 1 || warpDivFactorB < 1))
        {
            return NwStat::errorInvalidValue;
        }
    }
    catch (const std::exception&)
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

    // Allocate.
    try
    {
        nw.seqX_gpu.init(adjcols);
        nw.seqY_gpu.init(adjrows);
        nw.tileHrowMat_gpu.init(trows * tcols * (1 + tileBx));
        nw.tileHcolMat_gpu.init(trows * tcols * (1 + tileBy));

        nw.tileHrowMat.init(trows * tcols * (1 + tileBx));
        nw.tileHcolMat.init(trows * tcols * (1 + tileBy));

        std::vector<int> tmpTile((1 + tileBy) * (1 + tileBx), 0);
        std::swap(nw.tile, tmpTile);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    updateNwAlgPeakMemUsage(nw, res);

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
        dim3 blockA {};
        dim3 gridA {};
        size_t shmemsz {};

        blockA.x = threadsPerBlockA;

        cudaFuncAttributes attr {};
        if (cudaSuccess != (res.cudaStat = cudaFuncGetAttributes(&attr, (void*)Nw_Gpu8_KernelA)))
        {
            return NwStat::errorKernelFailure;
        }

        int maxActiveBlocksPerSm = 0;
        if (cudaSuccess != (res.cudaStat = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSm, (void*)Nw_Gpu8_KernelA, blockA.x, shmemsz)))
        {
            return NwStat::errorKernelFailure;
        }

        // Calculate the necessary number of blocks to cover the larger score matrix dimension.
        {
            int tileHrowMat_RowElemCount = tcols * (1 + tileBx);
            int tileHcolMat_ColElemCount = trows * (1 + tileBy);
            int largerDimElemCount = max2(tileHrowMat_RowElemCount, tileHcolMat_ColElemCount);
            gridA.x = (int)ceil(float(largerDimElemCount) / threadsPerBlockA);
        }

        int maxActiveBlocksActual = min2(maxActiveBlocksPerSm * nw.sm_count, (int)gridA.x);
        updateNwAlgPeakMemUsage(nw, res, &attr, maxActiveBlocksActual, blockA.x, shmemsz);

        int* tileHrowMat_gpu = nw.tileHrowMat_gpu.data();
        int* tileHcolMat_gpu = nw.tileHcolMat_gpu.data();

        void* kargs[] {
            &tileHrowMat_gpu,
            &tileHcolMat_gpu,
            &trows,
            &tcols,
            &tileBx,
            &tileBy,
            &nw.gapoCost};

        if (cudaSuccess != (res.cudaStat = cudaLaunchKernel((void*)Nw_Gpu8_KernelA, gridA, blockA, kargs, shmemsz, cudaStreamDefault)))
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
        cudaStream_t stream {};
        if (cudaSuccess != (res.cudaStat = cudaStreamCreate(&stream)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer1 = make_defer([&stream]() noexcept
        {
            cudaStreamDestroy(stream);
        });

        cudaGraph_t graph {};
        if (cudaSuccess != (res.cudaStat = cudaGraphCreate(&graph, 0)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer2 = make_defer([&graph]() noexcept
        {
            cudaGraphDestroy(graph);
        });

        // start capturing kernel launches by this thread
        if (cudaSuccess != (res.cudaStat = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)))
        {
            return NwStat::errorKernelFailure;
        }
        cudaError_t cudaStreamEndCapture_stat = cudaSuccess;
        auto defer3_cudaStreamEndCapture = make_defer([&cudaStreamEndCapture_stat, &stream, &graph]() noexcept
        {
            cudaStreamEndCapture_stat = cudaStreamEndCapture(stream, &graph);
        });

        dim3 blockB {};
        dim3 gridB {};

        {
            // The number of threads should be divisible by the warp size.
            // But for performance reasons, we don't need all those single-use warps, just half of them (or some other fraction).
            // That way the thread block can be smaller while doing the same amount of work.
            int warps = (int)ceil(float(max2(tileBx, tileBy)) / nw.warpsz / warpDivFactorB);
            blockB.x = nw.warpsz * warps;
        }

        // Size of shared memory per block in bytes.
        size_t shmemsz =
            /*subst[]*/ nw.substsz * nw.substsz * sizeof(int)
            /*seqX[]*/
            + tileBx * sizeof(int)
            /*seqY[]*/
            + tileBy * sizeof(int)
            /*tileHrow[]*/
            + (1 + tileBx) * sizeof(int)
            /*tileHcol[]*/
            + (1 + tileBy) * sizeof(int);

        cudaFuncAttributes attr {};
        if (cudaSuccess != (res.cudaStat = cudaFuncGetAttributes(&attr, (void*)Nw_Gpu8_KernelB)))
        {
            return NwStat::errorKernelFailure;
        }

        int maxActiveBlocksPerSm = 0;
        if (cudaSuccess != (res.cudaStat = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSm, (void*)Nw_Gpu8_KernelB, blockB.x, shmemsz)))
        {
            return NwStat::errorKernelFailure;
        }

        // For all (minor) tile diagonals in the score matrix.
        for (int d = 0; d < tcols - 1 + trows; d++)
        {
            {
                int tbeg = max2(0, d - (tcols - 1));
                int tend = min2(d + 1, trows);
                // Number of tiles on the current (minor) tile diagonal.
                int dsize = tend - tbeg;

                gridB.x = dsize;
            }

            int maxActiveBlocksActual = min2(maxActiveBlocksPerSm * nw.sm_count, (int)gridB.x);
            updateNwAlgPeakMemUsage(nw, res, &attr, maxActiveBlocksActual, blockB.x, shmemsz);

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
                &nw.gapoCost,
                // params related to tile B
                &trows,
                &tcols,
                &tileBx,
                &tileBy,
                &d};

            if (cudaSuccess != (res.cudaStat = cudaLaunchKernel((void*)Nw_Gpu8_KernelB, gridB, blockB, kargs, shmemsz, stream)))
            {
                return NwStat::errorKernelFailure;
            }
        }

        // collect kernel launches from this thread
        defer3_cudaStreamEndCapture();
        if (cudaSuccess != (res.cudaStat = cudaStreamEndCapture_stat))
        {
            return NwStat::errorKernelFailure;
        }

        cudaGraphExec_t graphExec;
        if (cudaSuccess != (res.cudaStat = cudaGraphInstantiate(&graphExec, graph, nullptr /*pErrorNode*/, nullptr /*pLogBuffer*/, 0 /*bufferSize*/)))
        {
            return NwStat::errorKernelFailure;
        }
        auto defer4 = make_defer([&graphExec]() noexcept
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

    TileAndElemIJ co;
    NwTrace2_GetTileAndElemIJ(nw, nw.adjrows - 1 /*last valid i pos*/, nw.adjcols - 1 /*last valid j pos*/, co);
    NwTrace2_AlignTile(nw.tile, nw, co);
    res.align_cost = el(nw.tile, 1 + tileBx, co.iTileElem, co.jTileElem);

    // Increment calculation time.
    sw.lap("align.calc");

    return NwStat::success;
}
