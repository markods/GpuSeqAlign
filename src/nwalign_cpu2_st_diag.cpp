#include "run_types.hpp"

// update the score given the current score matrix and position
// NOTE: gapoCost and most elements in the substitution matrix are negative, therefore find the maximum of them (instead of the minimum)
static void UpdateScore(NwAlgInput& nw, int i, int j) noexcept
{
    int p1 = el(nw.score, nw.adjcols, i - 1, j - 1) + el(nw.subst, nw.substsz, nw.seqY[i], nw.seqX[j]); // MOVE DOWN-RIGHT
    int p2 = el(nw.score, nw.adjcols, i - 1, j) + nw.gapoCost;                                          // MOVE DOWN
    int p3 = el(nw.score, nw.adjcols, i, j - 1) + nw.gapoCost;                                          // MOVE RIGHT
    el(nw.score, nw.adjcols, i, j) = max3(p1, p2, p3);
}

// sequential cpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Cpu2_St_Diag(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res)
{
    (void)pr;

    // the dimensions of the matrix without its row and column header
    const int rows = -1 + nw.adjrows;
    const int cols = -1 + nw.adjcols;

    // start the timer
    Stopwatch& sw = res.sw_align;
    sw.start();

    // reserve space in the ram
    try
    {
        nw.score.init(nw.adjrows * nw.adjcols);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // measure allocation time
    sw.lap("align.alloc");

    // initialize the first row and column of the score matrix
    for (int i = 0; i < nw.adjrows; i++)
    {
        el(nw.score, nw.adjcols, i, 0) = i * nw.gapoCost;
    }
    for (int j = 0; j < nw.adjcols; j++)
    {
        el(nw.score, nw.adjcols, 0, j) = j * nw.gapoCost;
    }

    // measure header initialization time
    sw.lap("align.init_hdr");

    //  x x x x x x
    //  x / / / . .
    //  x / / . . .
    //  x / . . . .
    for (int s = 0; s < rows; s++)
    {
        for (int t = 0; t <= s; t++)
        {
            int i = 1 + t;
            int j = 1 + s - t;
            UpdateScore(nw, i, j);
        }
    }

    //  x x x x x x
    //  x . . . / /
    //  x . . / / .
    //  x . / / . .
    // if the matrix is not square shaped
    if (rows != cols)
    {
        for (int s = rows; s < cols; s++)
        {
            for (int t = 0; t <= rows - 1; t++)
            {
                int i = 1 + t;
                int j = 1 + s - t;
                UpdateScore(nw, i, j);
            }
        }
    }

    //  x x x x x x
    //  x . . . . .|/ /
    //  x . . . . /|/
    //  x . . . / /|
    for (int s = cols; s < cols - 1 + rows; s++)
    {
        for (int t = s - cols + 1; t <= rows - 1; t++)
        {
            int i = 1 + t;
            int j = 1 + s - t;
            UpdateScore(nw, i, j);
        }
    }

    res.align_cost = el(nw.score, nw.adjcols, nw.adjrows - 1, nw.adjcols - 1);

    // measure calculation time
    sw.lap("align.calc");

    return NwStat::success;
}
