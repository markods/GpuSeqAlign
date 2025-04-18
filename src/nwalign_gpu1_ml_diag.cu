#include "defer.hpp"
#include "fmt_guard.hpp"
#include "math.hpp"
#include "nwalign_shared.hpp"
#include "run_types.hpp"
#include <cuda_runtime.h>

// cuda kernel for the parallel implementation
__global__ static void Nw_Gpu1_KernelAB(
    const int* const seqX_gpu,
    const int* const seqY_gpu,
    int* const score_gpu,
    const int* const subst_gpu,
    const int adjrows,
    const int adjcols,
    const int substsz,
    const int gapoCost,
    const int d // the current minor diagonal in the score matrix (exclude the header row and column)
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
    int pbeg = max(0, d - (cols - 1));
    int pend = min(d + 1, rows);
    // position of the current thread's element on the matrix diagonal
    int p = pbeg + (blockDim.x * blockIdx.x + threadIdx.x);

    // if the thread maps onto an element on the current matrix diagonal
    if (p < pend)
    {
        // position of the current element
        int i = 1 + (p);
        int j = 1 + (d - p);

        // if the thread maps onto the start of the diagonal
        if (d < cols && p == 0)
        {
            // initialize TOP header element
            el(score_gpu, adjcols, 0, j) = j * gapoCost;
            // if this is also the zeroth diagonal (with only one element on it)
            if (d == 0)
            {
                // initialize TOP-LEFT header element
                el(score_gpu, adjcols, 0, 0) = 0 * gapoCost;
            }
        }
        // if the thread maps onto the end of the diagonal
        if (d < rows && p == pend - 1)
        {
            // initialize LEFT header element
            el(score_gpu, adjcols, i, 0) = i * gapoCost;
        }

        // calculate the current element's value
        // +   always subtract the insert delete cost from the result, since that value was added to the initial temporary
        int p0 = el(subst_gpu, substsz, seqY_gpu[i], seqX_gpu[j]) - gapoCost;

        int p1 = el(score_gpu, adjcols, i - 1, j - 1) + p0; // MOVE DOWN-RIGHT
        int p2 = max(el(score_gpu, adjcols, i - 1, j), p1); // MOVE DOWN
        int p3 = max(el(score_gpu, adjcols, i, j - 1), p2); // MOVE RIGHT
        el(score_gpu, adjcols, i, j) = p3 + gapoCost;
    }
}

// parallel gpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Gpu1_Ml_Diag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res)
{
    // number of threads per block
    // +   the tile is one-dimensional
    int threadsPerBlock {};

    // Get parameters.
    try
    {
        threadsPerBlock = pr.at("threadsPerBlock").curr();
        if (threadsPerBlock < nw.warpsz || threadsPerBlock > nw.maxThreadsPerBlock)
        {
            return NwStat::errorInvalidValue;
        }
    }
    catch (const std::exception&)
    {
        return NwStat::errorInvalidValue;
    }

    // adjusted gpu score matrix dimensions
    // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile A size (in order to be evenly divisible)
    int adjrows = nw.adjrows;
    int adjcols = nw.adjcols;
    // special case when very small and very large sequences are compared
    if (adjrows == 1)
    {
        adjrows = 2;
    }
    if (adjcols == 1)
    {
        adjcols = 2;
    }
    // the dimensions of the matrix without its row and column header
    int rows = -1 + adjrows;
    int cols = -1 + adjcols;

    // start the timer
    Stopwatch& sw = res.sw_align;
    sw.start();

    // Allocate.
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

    updateNwAlgPeakMemUsage(nw, res);

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

    //  x x x x x x       x x x x x x       x x x x x x
    //  x / / / . .       x . . . / /       x . . . . .|/ /
    //  x / / . . .   +   x . . / / .   +   x . . . . /|/
    //  x / . . . .       x . / / . .       x . . . / /|
    // launch kernel for each minor diagonal of the score matrix
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

        // grid and block dimensions for kernel
        dim3 gridA {};
        dim3 blockA {};
        // calculate size of shared memory per block in bytes
        size_t shmemsz {};

        // take the number of threads per block as the only dimension
        blockA.x = threadsPerBlock;

        cudaFuncAttributes attr {};
        if (cudaSuccess != (res.cudaStat = cudaFuncGetAttributes(&attr, (void*)Nw_Gpu1_KernelAB)))
        {
            return NwStat::errorKernelFailure;
        }

        int maxActiveBlocksPerSm = 0;
        if (cudaSuccess != (res.cudaStat = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSm, (void*)Nw_Gpu1_KernelAB, blockA.x, shmemsz)))
        {
            return NwStat::errorKernelFailure;
        }

        // for all minor diagonals in the score matrix (excluding the header row and column)
        for (int d = 0; d < cols - 1 + rows; d++)
        {
            // calculate grid and block dimensions for kernel
            {
                int pbeg = max2(0, d - (cols - 1));
                int pend = min2(d + 1, rows);

                // the number of elements on the current diagonal
                int dsize = pend - pbeg;

                // take the number of blocks on the current score matrix diagonal as the only dimension
                // +   launch at least one block on the x axis
                gridA.x = (int)ceil(float(dsize) / threadsPerBlock);
            }

            int maxActiveBlocksActual = min2(maxActiveBlocksPerSm * nw.sm_count, (int)gridA.x);
            updateNwAlgPeakMemUsage(nw, res, &attr, maxActiveBlocksActual, blockA.x, shmemsz);

            // create variables for gpu arrays in order to be able to take their addresses
            int* seqX_gpu = nw.seqX_gpu.data();
            int* seqY_gpu = nw.seqY_gpu.data();
            int* score_gpu = nw.score_gpu.data();
            int* subst_gpu = nw.subst_gpu.data();

            // group arguments to be passed to kernel
            void* kargs[] {
                &seqX_gpu,
                &seqY_gpu,
                &score_gpu,
                &subst_gpu,
                &adjrows,
                &adjcols,
                &nw.substsz,
                &nw.gapoCost,
                &d};

            // launch the kernel in the given stream (don't statically allocate shared memory)
            if (cudaSuccess != (res.cudaStat = cudaLaunchKernel((void*)Nw_Gpu1_KernelAB, gridA, blockA, kargs, shmemsz, stream)))
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

    res.align_cost = el(nw.score, nw.adjcols, nw.adjrows - 1, nw.adjcols - 1);

    // measure memory transfer time
    sw.lap("align.cpy_host");

    return NwStat::success;
}
