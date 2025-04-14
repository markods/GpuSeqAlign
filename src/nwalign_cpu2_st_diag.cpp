#include "run_types.hpp"

static void UpdateScore(NwAlgInput& nw, int i, int j) noexcept
{
    int p1 = el(nw.score, nw.adjcols, i - 1, j - 1) + el(nw.subst, nw.substsz, nw.seqY[i], nw.seqX[j]); // MOVE DOWN-RIGHT
    int p2 = el(nw.score, nw.adjcols, i - 1, j) + nw.gapoCost;                                          // MOVE DOWN
    int p3 = el(nw.score, nw.adjcols, i, j - 1) + nw.gapoCost;                                          // MOVE RIGHT
    el(nw.score, nw.adjcols, i, j) = max3(p1, p2, p3);
}

NwStat NwAlign_Cpu2_St_Diag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res)
{
    (void)pr;

    // The dimensions of the matrix without its row and column header.
    const int rows = -1 + nw.adjrows;
    const int cols = -1 + nw.adjcols;

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

    sw.lap("align.alloc");

    // Initialize the first row and column of the score matrix.
    for (int i = 0; i < nw.adjrows; i++)
    {
        el(nw.score, nw.adjcols, i, 0) = i * nw.gapoCost;
    }
    for (int j = 0; j < nw.adjcols; j++)
    {
        el(nw.score, nw.adjcols, 0, j) = j * nw.gapoCost;
    }

    sw.lap("align.init_hdr");

    //  / / / . .       . . . / /       . . . . .|/ /
    //  / / . . .   +   . . / / .   +   . . . . /|/
    //  / . . . .       . / / . .       . . . / /|
    for (int s = 0; s < cols - 1 + rows; s++)
    {
        int tbeg = max2(0, s - (cols - 1));
        int tend = min2(s + 1, rows);

        for (int t = tbeg; t < tend; t++)
        {
            // calculate the element boundaries
            int ibeg = 1 + t;
            int jbeg = 1 + s - t;

            int iend = min2(ibeg + 1, 1 + rows);
            int jend = min2(jbeg + 1, 1 + cols);

            for (int i = ibeg; i < iend; i++)
            {
                for (int j = jbeg; j < jend; j++)
                {
                    UpdateScore(nw, i, j);
                }
            }
        }
    }

    res.align_cost = el(nw.score, nw.adjcols, nw.adjrows - 1, nw.adjcols - 1);

    sw.lap("align.calc");

    return NwStat::success;
}
