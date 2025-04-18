#include "nwalign_shared.hpp"
#include "run_types.hpp"

static void UpdateScore(NwAlgInput& nw, int i, int j) noexcept
{
    int p1 = el(nw.score, nw.adjcols, i - 1, j - 1) + el(nw.subst, nw.substsz, nw.seqY[i], nw.seqX[j]); // MOVE DOWN-RIGHT
    int p2 = el(nw.score, nw.adjcols, i - 1, j) + nw.gapoCost;                                          // MOVE DOWN
    int p3 = el(nw.score, nw.adjcols, i, j - 1) + nw.gapoCost;                                          // MOVE RIGHT
    el(nw.score, nw.adjcols, i, j) = max3(p1, p2, p3);
}

NwStat NwAlign_Cpu4_Mt_DiagRow(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res)
{
    // Size of square block that will be a unit of work.
    int blocksz = {};

    // Get parameters.
    try
    {
        blocksz = pr.at("blocksz").curr();
        if (blocksz < 1)
        {
            return NwStat::errorInvalidValue;
        }
    }
    catch (const std::exception&)
    {
        return NwStat::errorInvalidValue;
    }

    // The dimensions of the matrix without its row and column header.
    const int rows = -1 + nw.adjrows;
    const int cols = -1 + nw.adjcols;

    // Number of blocks in a row and column (rounded up).
    const int rowblocks = (int)ceil(float(rows) / blocksz);
    const int colblocks = (int)ceil(float(cols) / blocksz);

    Stopwatch& sw = res.sw_align;
    sw.start();

    // Allocate.
    try
    {
        nw.score.init(nw.adjrows * nw.adjcols);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    updateNwAlgPeakMemUsage(nw, res);

    sw.lap("align.alloc");

    #pragma omp parallel
    {
        // Initialize the first row and column of the score matrix.
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < nw.adjrows; i++)
        {
            el(nw.score, nw.adjcols, i, 0) = i * nw.gapoCost;
        }
        #pragma omp for schedule(static)
        for (int j = 0; j < nw.adjcols; j++)
        {
            el(nw.score, nw.adjcols, 0, j) = j * nw.gapoCost;
        }

        #pragma omp single
        {
            sw.lap("align.init_hdr");
        }

        //  / / / . .       . . . / /       . . . . .|/ /
        //  / / . . .   +   . . / / .   +   . . . . /|/
        //  / . . . .       . / / . .       . . . / /|
        for (int s = 0; s < colblocks - 1 + rowblocks; s++)
        {
            int tbeg = max2(0, s - (colblocks - 1));
            int tend = min2(s + 1, rowblocks);

            #pragma omp for schedule(static)
            for (int t = tbeg; t < tend; t++)
            {
                // Calculate the block boundaries.
                int ibeg = 1 + (t)*blocksz;
                int jbeg = 1 + (s - t) * blocksz;

                int iend = min2(ibeg + blocksz, 1 + rows);
                int jend = min2(jbeg + blocksz, 1 + cols);

                // Process the block.
                for (int i = ibeg; i < iend; i++)
                {
                    for (int j = jbeg; j < jend; j++)
                    {
                        UpdateScore(nw, i, j);
                    }
                }
            }
        }
    }

    res.align_cost = el(nw.score, nw.adjcols, nw.adjrows - 1, nw.adjcols - 1);

    sw.lap("align.calc");

    return NwStat::success;
}
