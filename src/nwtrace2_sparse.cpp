#include "fmt_guard.hpp"
#include "nw_fns.hpp"
#include "nwalign_shared.hpp"
#include "run_types.hpp"
#include <limits>
#include <vector>

void NwTrace2_GetTileAndElemIJ(const NwAlgInput& nw, int i, int j, TileAndElemIJ& co)
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

    // Don't subtract header element from i and j!
    co.iTile = i / (nw.tileHcolLen - 1);
    co.jTile = j / (nw.tileHrowLen - 1);
    co.iTileElem = i % (nw.tileHcolLen - 1);
    co.jTileElem = j % (nw.tileHrowLen - 1);

    // Small adjustment for the last score matrix row and column:
    // saturate iTile and jTile, so that we don't end up in impossible tile coordinates.
    if (co.iTile == nw.tileHdrMatRows)
    {
        co.iTile -= 1;
        co.iTileElem += (nw.tileHcolLen - 1);
    }
    if (co.jTile == nw.tileHdrMatCols)
    {
        co.jTile -= 1;
        co.jTileElem += (nw.tileHrowLen - 1);
    }
}

void NwTrace2_AlignTile(std::vector<int>& tile, const NwAlgInput& nw, const TileAndElemIJ& co)
{
    //  x x x x x x
    //  x / / / / /
    //  x / / / / /
    //  x / / / / /

    // Load the tile's header row.
    {
        int kHrow = nw.tileHdrMatCols * co.iTile + co.jTile;
        int kHrowElem = kHrow * nw.tileHrowLen;

        for (int j = 0; j < nw.tileHrowLen; j++)
        {
            el(tile, nw.tileHrowLen, 0, j) = nw.tileHrowMat[kHrowElem + j];
        }
    }

    // Load the tile's header column.
    {
        int kHcol = nw.tileHdrMatCols * co.iTile + co.jTile; // row-major
        int kHcolElem = kHcol * nw.tileHcolLen;

        for (int i = 0; i < nw.tileHcolLen; i++)
        {
            el(tile, nw.tileHrowLen, i, 0) = nw.tileHcolMat[kHcolElem + i];
        }
    }

    // Calculate remaining elements in tile.
    {
        // Don't add header column because it's added in the inner loop - loops start from 1.
        int ibeg = 0 + co.iTile * (nw.tileHcolLen - 1);
        int jbeg = 0 + co.jTile * (nw.tileHrowLen - 1);

        int iend = min2(nw.tileHcolLen, co.iTileElem + 1 /*exclusive*/);
        int jend = min2(nw.tileHrowLen, co.jTileElem + 1 /*exclusive*/);

        for (int i = 1; i < iend; i++)
        {
            for (int j = 1; j < jend; j++)
            {
                if (ibeg + i >= nw.adjrows || jbeg + j >= nw.adjcols)
                {
                    // Write 0 for artificial elements.
                    el(tile, nw.tileHrowLen, i, j) = 0;
                    continue;
                }

                int p1 = el(tile, nw.tileHrowLen, i - 1, j - 1) + el(nw.subst, nw.substsz, nw.seqY[ibeg + i], nw.seqX[jbeg + j]); // MOVE DOWN-RIGHT
                int p2 = el(tile, nw.tileHrowLen, i - 1, j) + nw.gapoCost;                                                        // MOVE DOWN
                int p3 = el(tile, nw.tileHrowLen, i, j - 1) + nw.gapoCost;                                                        // MOVE RIGHT
                el(tile, nw.tileHrowLen, i, j) = max3(p1, p2, p3);
            }
        }
    }
}

// Trace one of the optimal alignments in the score matrix.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
NwStat NwTrace2_Sparse(NwAlgInput& nw, NwAlgResult& res, bool calcDebugTrace)
{
    Stopwatch& sw = res.sw_trace;
    sw.start();

    // Allocate.
    try
    {
        std::vector<int> tmpTile(nw.tileHcolLen * nw.tileHrowLen, 0);
        std::swap(nw.tile, tmpTile);
        res.edit_trace.reserve(nw.adjrows - 1 + nw.adjcols); // Longest possible path.
        if (calcDebugTrace)
        {
            nw.trace.reserve(nw.adjrows - 1 + nw.adjcols);
        }
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    updateNwAlgPeakMemUsage(nw, res);

    sw.lap("trace.alloc");

    // Load last tile.
    int i = nw.adjrows - 1 /*last valid i pos*/;
    int j = nw.adjcols - 1 /*last valid j pos*/;
    TileAndElemIJ co;
    NwTrace2_GetTileAndElemIJ(nw, i, j, co);
    NwTrace2_AlignTile(nw.tile, nw, co);

    int same_letter_cnt = 1;
    char edit = '\0';
    char prev_edit = '\0';
    while (true)
    {
        if (calcDebugTrace)
        {
            int currElem = el(nw.tile, nw.tileHrowLen, co.iTileElem, co.jTileElem);
            nw.trace.push_back(currElem);
        }

        int max = std::numeric_limits<int>::min();
        int di = 0;
        int dj = 0;

        if (co.iTileElem > 0 && co.jTileElem > 0)
        {
            max = el(nw.tile, nw.tileHrowLen, co.iTileElem - 1, co.jTileElem - 1);
            di = -1;
            dj = -1;
            if (nw.seqX[j] == nw.seqY[i])
            {
                // DIAGONAL^-1 -- match.
                edit = '=';
            }
            else
            {
                // DIAGONAL^-1 -- mismatch.
                edit = 'X';
            }
        }
        if (co.iTileElem > 0 && max < el(nw.tile, nw.tileHrowLen, co.iTileElem - 1, co.jTileElem))
        {
            max = el(nw.tile, nw.tileHrowLen, co.iTileElem - 1, co.jTileElem);
            di = -1;
            dj = 0;
            // DOWN^-1 -- align a gap in seqX to a letter in seqY. INSERTION in seqX.
            edit = 'I';
        }
        if (co.jTileElem > 0 && max < el(nw.tile, nw.tileHrowLen, co.iTileElem, co.jTileElem - 1))
        {
            max = el(nw.tile, nw.tileHrowLen, co.iTileElem, co.jTileElem - 1);
            di = 0;
            dj = -1;
            // RIGHT^-1 -- align a letter in seqX to a gap in seqY. DELETION in seqX.
            edit = 'D';
        }
        i += di;
        j += dj;
        co.iTileElem += di;
        co.jTileElem += dj;

        // Load up / left / up-left tile if possible and we ended up in the header row / header column / intersection of the header row and header column.
        int diTile = -(co.iTileElem == 0 && co.iTile > 0);
        int djTile = -(co.jTileElem == 0 && co.jTile > 0);
        if (diTile != 0 || djTile != 0)
        {
            co.iTile += diTile;
            co.jTile += djTile;

            // If we are in the tile header row/column, set coordinates as if we're in the UP/LEFT/UP-LEFT tile's last row/column.
            if (co.iTileElem == 0 && di != 0)
            {
                co.iTileElem = nw.tileHcolLen - 1;
            }
            if (co.jTileElem == 0 && dj != 0)
            {
                co.jTileElem = nw.tileHrowLen - 1;
            }

            NwTrace2_AlignTile(nw.tile, nw, co);
        }

        if (edit != prev_edit && prev_edit != '\0')
        {
            std::string letter_cnt_str {std::to_string(same_letter_cnt)};
            std::reverse(letter_cnt_str.begin(), letter_cnt_str.end());
            res.edit_trace.push_back(prev_edit);
            res.edit_trace.append(letter_cnt_str);
            same_letter_cnt = 1;
        }
        else if (edit == prev_edit)
        {
            same_letter_cnt++;
        }
        prev_edit = edit;
        edit = '\0';

        if (di == 0 && dj == 0)
        {
            break;
        }
    }

    // Reverse the edit trace, so it starts from the top-left corner of the matrix instead of the bottom-right.
    std::reverse(res.edit_trace.begin(), res.edit_trace.end());

    sw.lap("trace.calc");

    if (calcDebugTrace)
    {
        // Reverse the trace, so it starts from the top-left corner of the matrix.
        std::reverse(nw.trace.begin(), nw.trace.end());
    }

    // http://www.cse.yorku.ca/~oz/hash.html
    unsigned hash = 5381;

    for (auto& curr : res.edit_trace)
    {
        hash = ((hash << 5) + hash) ^ curr;
    }
    if (calcDebugTrace)
    {
        for (auto& curr : nw.trace)
        {
            hash = ((hash << 5) + hash) ^ curr;
        }
    }

    res.trace_hash = hash;

    return NwStat::success;
}

// Hash the score matrix.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
NwStat NwHash2_Sparse(NwAlgInput& nw, NwAlgResult& res)
{
    // http://www.cse.yorku.ca/~oz/hash.html
    unsigned hash = 5381;

    Stopwatch& sw = res.sw_hash;
    sw.start();

    // Allocate.
    try
    {
        std::vector<int> tmpCurrRow(nw.adjcols, 0);
        std::vector<int> tmpPrevRow(nw.adjcols, 0);
        std::swap(nw.currRow, tmpCurrRow);
        std::swap(nw.prevRow, tmpPrevRow);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    updateNwAlgPeakMemUsage(nw, res);

    sw.lap("align.alloc");

    for (int i = 0; i < nw.adjrows; i++)
    {
        for (int j = 0; j < nw.adjcols; j++)
        {
            TileAndElemIJ co;
            NwTrace2_GetTileAndElemIJ(nw, nw.adjrows - 1 /*last valid i pos*/, nw.adjcols - 1 /*last valid j pos*/, co);

            int currElem = 0;
            // Load the header row element instead of calculating it.
            // Don't attempt to load the last header row, since it isn't stored in the tile header row matrix.
            if (co.iTileElem == 0 && i != nw.adjrows - 1)
            {
                int kHrowElemBeg = (nw.tileHdrMatCols * co.iTile + co.jTile) * nw.tileHrowLen + 0;
                currElem = nw.tileHrowMat[kHrowElemBeg + co.jTileElem];
            }
            // Load the header column element instead of calculating it.
            // Don't attempt to load the last header column, since it isn't stored in the tile header column matrix.
            else if (co.jTileElem == 0 && j != nw.adjcols - 1)
            {
                int kHcolElemBeg = (nw.tileHdrMatCols * co.iTile + co.jTile) * nw.tileHcolLen + 0;
                currElem = nw.tileHcolMat[kHcolElemBeg + co.iTileElem];
            }
            else if (i > 0 && j > 0)
            {
                int p1 = nw.prevRow[/*i - 1,*/ j - 1] + el(nw.subst, nw.substsz, nw.seqY[i], nw.seqX[j]); // MOVE DOWN-RIGHT
                int p2 = nw.prevRow[/*i - 1,*/ j] + nw.gapoCost;                                          // MOVE DOWN
                int p3 = nw.currRow[/*i,*/ j - 1] + nw.gapoCost;                                          // MOVE RIGHT
                currElem = max3(p1, p2, p3);
            }
            else if (i > 0)
            {
                currElem = nw.prevRow[/*i - 1,*/ j] + nw.gapoCost; // MOVE DOWN
            }
            else if (j > 0)
            {
                currElem = nw.currRow[/*i,*/ j - 1] + nw.gapoCost; // MOVE RIGHT
            }

            nw.currRow[j] = currElem;

            // Add the current element to the hash.
            hash = ((hash << 5) + hash) ^ currElem;
        }

        std::swap(nw.currRow, nw.prevRow);
    }

    res.score_hash = hash;

    sw.lap("hash.calc");

    return NwStat::success;
}

// Print the score matrix.
// The score matrix is represented as two matrices (row-major order):
// + tile header row matrix,
// + tile header column matrix.
NwStat NwPrintScore2_Sparse(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res)
{
    (void)res;

    FormatFlagsGuard fg {os, 4};

    std::vector<int> currRow;
    std::vector<int> prevRow;

    // Allocate.
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

    updateNwAlgPeakMemUsage(nw, res);

    for (int i = 0; i < nw.adjrows; i++)
    {
        for (int j = 0; j < nw.adjcols; j++)
        {
            TileAndElemIJ co;
            NwTrace2_GetTileAndElemIJ(nw, nw.adjrows - 1 /*last valid i pos*/, nw.adjcols - 1 /*last valid j pos*/, co);

            int currElem = 0;
            // Load the header row element instead of calculating it.
            // Don't attempt to load the last header row, since it isn't stored in the tile header row matrix.
            if (co.iTileElem == 0 && i != nw.adjrows - 1)
            {
                int kHrowElemBeg = (nw.tileHdrMatCols * co.iTile + co.jTile) * nw.tileHrowLen + 0;
                currElem = nw.tileHrowMat[kHrowElemBeg + co.jTileElem];
            }
            // Load the header column element instead of calculating it.
            // Don't attempt to load the last header colum, since it isn't stored in the tile header column matrix.
            else if (co.jTileElem == 0 && j != nw.adjcols - 1)
            {
                int kHcolElemBeg = (nw.tileHdrMatCols * co.iTile + co.jTile) * nw.tileHcolLen + 0;
                currElem = nw.tileHcolMat[kHcolElemBeg + co.iTileElem];
            }
            else if (i > 0 && j > 0)
            {
                int p1 = prevRow[/*i - 1,*/ j - 1] + el(nw.subst, nw.substsz, nw.seqY[i], nw.seqX[j]); // MOVE DOWN-RIGHT
                int p2 = prevRow[/*i - 1,*/ j] + nw.gapoCost;                                          // MOVE DOWN
                int p3 = currRow[/*i,*/ j - 1] + nw.gapoCost;                                          // MOVE RIGHT
                currElem = max3(p1, p2, p3);
            }
            else if (i > 0)
            {
                currElem = prevRow[/*i - 1,*/ j] + nw.gapoCost; // MOVE DOWN
            }
            else if (j > 0)
            {
                currElem = currRow[/*i,*/ j - 1] + nw.gapoCost; // MOVE RIGHT
            }

            currRow[j] = currElem;

            // Write the current element to the output stream.
            os << std::setw(4) << currElem << ',';
        }

        std::swap(currRow, prevRow);
        os << '\n';
    }

    return NwStat::success;
}
