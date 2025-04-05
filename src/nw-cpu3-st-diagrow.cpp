#include "common.hpp"

// parallel cpu implementation of the Needleman-Wunsch algorithm
NwStat NwAlign_Cpu3_St_DiagRow(NwParams& pr, NwInput& nw, NwResult& res)
{
    // size of square block (that will be a unit of work)
    // +   8*(16 ints) on standard architectures, or 8 cache lines
    int blocksz = {};

    // get the parameter values
    try
    {
        blocksz = pr["blocksz"].curr();
    }
    catch (const std::out_of_range&)
    {
        return NwStat::errorInvalidValue;
    }

    // the dimensions of the matrix without its row and column header
    const int rows = -1 + nw.adjrows;
    const int cols = -1 + nw.adjcols;

    // number of blocks in a row and column (rounded up)
    const int rowblocks = (int)ceil(float(rows) / blocksz);
    const int colblocks = (int)ceil(float(cols) / blocksz);

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
        el(nw.score, nw.adjcols, i, 0) = i * nw.indel;
    }
    for (int j = 0; j < nw.adjcols; j++)
    {
        el(nw.score, nw.adjcols, 0, j) = j * nw.indel;
    }

    // measure header initialization time
    sw.lap("align.init_hdr");

    //  / / / . .       . . . / /       . . . . .|/ /
    //  / / . . .   +   . . / / .   +   . . . . /|/
    //  / . . . .       . / / . .       . . . / /|
    for (int s = 0; s < colblocks - 1 + rowblocks; s++)
    {
        int tbeg = max2(0, s - (colblocks - 1));
        int tend = min2(s + 1, rowblocks);

        for (int t = tbeg; t < tend; t++)
        {
            // calculate the block boundaries
            int ibeg = 1 + (t)*blocksz;
            int jbeg = 1 + (s - t) * blocksz;

            int iend = min2(ibeg + blocksz, 1 + rows);
            int jend = min2(jbeg + blocksz, 1 + cols);

            // process the block
            for (int i = ibeg; i < iend; i++)
            {
                for (int j = jbeg; j < jend; j++)
                {
                    UpdateScore(nw, i, j);
                }
            }
        }
    }

    // measure calculation time
    sw.lap("align.calc");

    return NwStat::success;
}
