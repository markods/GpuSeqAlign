#include "common.hpp"
#include <limits>

static void NwLoadHeaderAndAlignTile(std::vector<int>& tile, const NwInput& nw, int iTile, int jTile)
{
    //  x x x x x x
    //  x / / / / /
    //  x / / / / /
    //  x / / / / /

    // Load the tile's header row.
    {
        int kHrow = nw.tileHdrMatCols * iTile + jTile;
        int kHrowElem = kHrow * nw.tileHrowLen;

        for (int j = 0; j < nw.tileHrowLen; j++)
        {
            el(tile, nw.tileHrowLen, 0, j) = nw.tileHrowMat[kHrowElem + j];
        }
    }

    // Load the tile's header column.
    {
        int kHcol = nw.tileHdrMatCols * iTile + jTile; // row-major
        int kHcolElem = kHcol * nw.tileHcolLen;

        for (int i = 0; i < nw.tileHcolLen; i++)
        {
            el(tile, nw.tileHrowLen, i, 0) = nw.tileHcolMat[kHcolElem + i];
        }
    }

    // Calculate remaining elements in tile.
    {
        // Don't add header column because it's added in the inner loop - loops start from 1.
        int ibeg = 0 + iTile * (nw.tileHcolLen - 1);
        int jbeg = 0 + jTile * (nw.tileHrowLen - 1);

        for (int i = 1; i < nw.tileHcolLen; i++)
        {
            for (int j = 1; j < nw.tileHrowLen; j++)
            {
                if (ibeg + i >= nw.adjrows || jbeg + j >= nw.adjcols)
                {
                    el(tile, nw.tileHrowLen, i, j) = 0;
                    continue;
                }

                int p1 = el(tile, nw.tileHrowLen, i - 1, j - 1) + el(nw.subst, nw.substsz, nw.seqY[ibeg + i], nw.seqX[jbeg + j]); // MOVE DOWN-RIGHT
                int p2 = el(tile, nw.tileHrowLen, i - 1, j) + nw.indel;                                                           // MOVE DOWN
                int p3 = el(tile, nw.tileHrowLen, i, j - 1) + nw.indel;                                                           // MOVE RIGHT
                el(tile, nw.tileHrowLen, i, j) = max3(p1, p2, p3);
            }
        }
    }
}

// Trace one of the optimal alignments in the score matrix.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
NwStat NwTrace2_Sparse(const NwInput& nw, NwResult& res)
{
    std::vector<int> trace;
    std::vector<int> tile;

    // Start the timer.
    Stopwatch& sw = res.sw_trace;
    sw.start();

    // Allocate memory.
    try
    {
        trace.reserve(nw.adjrows + nw.adjcols - 1);
        std::vector<int> tmpTile(nw.tileHcolLen * nw.tileHrowLen, 0);
        std::swap(tile, tmpTile);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // Measure allocation time.
    sw.lap("alloc");

    // ---------------
    // |h  h  h  h  h| h
    // |h  .  .  .  .| .
    // |h  .  .  .  .| .
    // ---------------
    //  h  .  .  .  .  .
    // We're using the same tiles as before (the '.' in the schematic), which we extend with their header row and column,
    // but subtract their last row and column. Meaning the tile dimensions stay the same.
    // The last row and column are in fact the header row/column of the neighbouring tile, so no need to calculate them.

    // Don't subtract 1 from i and j like so: "(i-1)/..." or "(j-1)/..." !
    int iTile = (nw.adjrows - 1) / (nw.tileHcolLen - 1);
    int jTile = (nw.adjcols - 1) / (nw.tileHrowLen - 1);
    int i = (nw.adjrows - 1) % (nw.tileHcolLen - 1);
    int j = (nw.adjcols - 1) % (nw.tileHrowLen - 1);

    // Small adjustment for the last score matrix row and column:
    // saturate iTile and jTile, so that we don't end up with impossible tile coordinates.
    if (iTile == nw.tileHdrMatRows)
    {
        iTile -= 1;
        i += (nw.tileHcolLen - 1);
    }
    if (jTile == nw.tileHdrMatCols)
    {
        jTile -= 1;
        j += (nw.tileHrowLen - 1);
    }

    // Load last tile.
    NwLoadHeaderAndAlignTile(tile, nw, iTile, jTile);

    // While there are elements on one of the optimal paths.
    while (true)
    {
        int currElem = el(tile, nw.tileHrowLen, i, j);
        trace.push_back(currElem);

        int max = std::numeric_limits<int>::min();
        int di = 0;
        int dj = 0;

        // Up-left movement if possible and better.
        if (i > 0 && j > 0 && max < el(tile, nw.tileHrowLen, i - 1, j - 1))
        {
            max = el(tile, nw.tileHrowLen, i - 1, j - 1);
            di = -1;
            dj = -1;
        }
        // Up movement if possible and better.
        if (i > 0 && max < el(tile, nw.tileHrowLen, i - 1, j))
        {
            max = el(tile, nw.tileHrowLen, i - 1, j);
            di = -1;
            dj = 0;
        }
        // Left movement if possible and better.
        if (j > 0 && max < el(tile, nw.tileHrowLen, i, j - 1))
        {
            max = el(tile, nw.tileHrowLen, i, j - 1);
            di = 0;
            dj = -1;
        }
        i += di;
        j += dj;

        // Load up / left / up-left tile if possible and we ended up in the header row / header column / intersection of the header row and header column.
        int diTile = -(i == 0 && iTile > 0);
        int djTile = -(j == 0 && jTile > 0);
        if (diTile != 0 || djTile != 0)
        {
            iTile += diTile;
            jTile += djTile;

            // If we are in the tile header row/column, set coordinates as if we're in the next tile's last row/column.
            if (i == 0 && di != 0)
            {
                i = nw.tileHcolLen - 1;
            }
            if (j == 0 && dj != 0)
            {
                j = nw.tileHrowLen - 1;
            }

            NwLoadHeaderAndAlignTile(tile, nw, iTile, jTile);
        }

        if (di == 0 && dj == 0)
        {
            break;
        }
    }

    // Reverse the trace, so it starts from the top-left corner of the matrix instead of the bottom-right.
    std::reverse(trace.begin(), trace.end());

    // Measure trace time.
    sw.lap("calc-1");

    // Calculate the trace hash.
    // http://www.cse.yorku.ca/~oz/hash.html
    unsigned hash = 5381;
    for (auto& curr : trace)
    {
        hash = ((hash << 5) + hash) ^ curr;
    }

    // Save the trace hash.
    res.trace_hash = hash;

    return NwStat::success;
}

// Hash the score matrix.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
NwStat NwHash2_Sparse(const NwInput& nw, NwResult& res)
{
    // http://www.cse.yorku.ca/~oz/hash.html
    unsigned hash = 5381;
    std::vector<int> currRow;
    std::vector<int> prevRow;

    // Start the timer.
    Stopwatch& sw = res.sw_hash;
    sw.start();

    // Allocate memory.
    try
    {
        std::vector<int> tmpCurrRow(nw.adjcols, 0);
        std::vector<int> tmpPrevRow(nw.adjcols, 0);
        std::swap(currRow, tmpCurrRow);
        std::swap(prevRow, tmpPrevRow);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // Measure allocation time.
    sw.lap("alloc");

    // Calculate the score matrix hash.
    for (int i = 0; i < nw.adjrows; i++)
    {
        for (int j = 0; j < nw.adjcols; j++)
        {
            // ---------------
            // |h  h  h  h  h| h
            // |h  .  .  .  .| .
            // |h  .  .  .  .| .
            // ---------------
            //  h  .  .  .  .  .
            // We're using the same tiles as before (the '.' in the schematic), which we extend with their header row and column,
            // but subtract their last row and column. Meaning the tile dimensions stay the same.
            // The last row and column are in fact the header row/column of the neighbouring tile, so no need to calculate them.

            // Don't subtract 1 from i and j like so: "(i-1)/..." or "(j-1)/..." !
            int iTile = i / (nw.tileHcolLen - 1);
            int jTile = j / (nw.tileHrowLen - 1);
            int iTileElem = i % (nw.tileHcolLen - 1);
            int jTileElem = j % (nw.tileHrowLen - 1);

            // Small adjustment for the last score matrix row and column:
            // saturate iTile and jTile, so that we don't end up with impossible tile coordinates.
            if (iTile == nw.tileHdrMatRows)
            {
                iTile -= 1;
                iTileElem += (nw.tileHcolLen - 1);
            }
            if (jTile == nw.tileHdrMatCols)
            {
                jTile -= 1;
                jTileElem += (nw.tileHrowLen - 1);
            }

            int currElem = 0;
            // Load the header row element instead of calculating it.
            // Don't attempt to load the last header row, since it isn't stored in the tile header row matrix.
            if (iTileElem == 0 && i != nw.adjrows - 1)
            {
                int kHrowElemBeg = (nw.tileHdrMatCols * iTile + jTile) * nw.tileHrowLen + 0;
                currElem = nw.tileHrowMat[kHrowElemBeg + jTileElem];
            }
            // Load the header column element instead of calculating it.
            // Don't attempt to load the last header column, since it isn't stored in the tile header column matrix.
            else if (jTileElem == 0 && j != nw.adjcols - 1)
            {
                int kHcolElemBeg = (nw.tileHdrMatCols * iTile + jTile) * nw.tileHcolLen + 0;
                currElem = nw.tileHcolMat[kHcolElemBeg + iTileElem];
            }
            else if (i > 0 && j > 0)
            {
                int p1 = prevRow[/*i - 1,*/ j - 1] + el(nw.subst, nw.substsz, nw.seqY[i], nw.seqX[j]); // MOVE DOWN-RIGHT
                int p2 = prevRow[/*i - 1,*/ j] + nw.indel;                                             // MOVE DOWN
                int p3 = currRow[/*i,*/ j - 1] + nw.indel;                                             // MOVE RIGHT
                currElem = max3(p1, p2, p3);
            }
            else if (i > 0)
            {
                currElem = prevRow[/*i - 1,*/ j] + nw.indel; // MOVE DOWN
            }
            else if (j > 0)
            {
                currElem = currRow[/*i,*/ j - 1] + nw.indel; // MOVE RIGHT
            }

            currRow[j] = currElem;

            // Add the current element to the hash.
            hash = ((hash << 5) + hash) ^ currElem;
        }

        std::swap(currRow, prevRow);
    }

    // Save the resulting hash.
    res.score_hash = hash;

    // Measure hash time.
    sw.lap("calc-1");

    return NwStat::success;
}

// Print the score matrix.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
NwStat NwPrint2_Sparse(std::ostream& os, const NwInput& nw, NwResult& res)
{
    (void)res;

    FormatFlagsGuard fg {os, 4};

    std::vector<int> currRow;
    std::vector<int> prevRow;

    // Allocate memory.
    try
    {
        std::vector<int> tmpCurrRow(nw.adjcols, 0);
        std::vector<int> tmpPrevRow(nw.adjcols, 0);
        std::swap(currRow, tmpCurrRow);
        std::swap(prevRow, tmpPrevRow);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // Write the score matrix.
    for (int i = 0; i < nw.adjrows; i++)
    {
        for (int j = 0; j < nw.adjcols; j++)
        {
            // ---------------
            // |h  h  h  h  h| h
            // |h  .  .  .  .| .
            // |h  .  .  .  .| .
            // ---------------
            //  h  .  .  .  .  .
            // We're using the same tiles as before (the '.' in the schematic), which we extend with their header row and column,
            // but subtract their last row and column. Meaning the tile dimensions stay the same.
            // The last row and column are in fact the header row/column of the neighbouring tile, so no need to calculate them.

            // Don't subtract 1 from i and j like so: "(i-1)/..." or "(j-1)/..." !
            int iTile = i / (nw.tileHcolLen - 1);
            int jTile = j / (nw.tileHrowLen - 1);
            int iTileElem = i % (nw.tileHcolLen - 1);
            int jTileElem = j % (nw.tileHrowLen - 1);

            // Small adjustment for the last score matrix row and column:
            // saturate iTile and jTile, so that we don't end up in impossible tile coordinates.
            if (iTile == nw.tileHdrMatRows)
            {
                iTile -= 1;
                iTileElem += (nw.tileHcolLen - 1);
            }
            if (jTile == nw.tileHdrMatCols)
            {
                jTile -= 1;
                jTileElem += (nw.tileHrowLen - 1);
            }

            int currElem = 0;
            // Load the header row element instead of calculating it.
            // Don't attempt to load the last header row, since it isn't stored in the tile header row matrix.
            if (iTileElem == 0 && i != nw.adjrows - 1)
            {
                int kHrowElemBeg = (nw.tileHdrMatCols * iTile + jTile) * nw.tileHrowLen + 0;
                currElem = nw.tileHrowMat[kHrowElemBeg + jTileElem];
            }
            // Load the header column element instead of calculating it.
            // Don't attempt to load the last header colum, since it isn't stored in the tile header column matrix.
            else if (jTileElem == 0 && j != nw.adjcols - 1)
            {
                int kHcolElemBeg = (nw.tileHdrMatCols * iTile + jTile) * nw.tileHcolLen + 0;
                currElem = nw.tileHcolMat[kHcolElemBeg + iTileElem];
            }
            else if (i > 0 && j > 0)
            {
                int p1 = prevRow[/*i - 1,*/ j - 1] + el(nw.subst, nw.substsz, nw.seqY[i], nw.seqX[j]); // MOVE DOWN-RIGHT
                int p2 = prevRow[/*i - 1,*/ j] + nw.indel;                                             // MOVE DOWN
                int p3 = currRow[/*i,*/ j - 1] + nw.indel;                                             // MOVE RIGHT
                currElem = max3(p1, p2, p3);
            }
            else if (i > 0)
            {
                currElem = prevRow[/*i - 1,*/ j] + nw.indel; // MOVE DOWN
            }
            else if (j > 0)
            {
                currElem = currRow[/*i,*/ j - 1] + nw.indel; // MOVE RIGHT
            }

            currRow[j] = currElem;

            // Write the current element to the output stream.
            os << std::setw(4) << currElem << ',';
        }

        std::swap(currRow, prevRow);
        os << std::endl;
    }

    return NwStat::success;
}
