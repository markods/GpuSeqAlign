#include "nwalign_shared.hpp"
#include "print_mat.hpp"
#include "run_types.hpp"
#include <limits>

// TODO: smith waterman
// TODO: affine gap

// TODO: remove comments in nwalign algorithms (part 3)

NwStat NwTrace1_Plain(NwAlgInput& nw, NwAlgResult& res, bool calcDebugTrace)
{
    Stopwatch& sw = res.sw_trace;
    sw.start();

    // Allocate.
    try
    {
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

    int i = nw.adjrows - 1;
    int j = nw.adjcols - 1;
    int same_letter_cnt = 1;
    char edit = '\0';
    char prev_edit = '\0';
    while (true)
    {
        if (calcDebugTrace)
        {
            int curr = el(nw.score, nw.adjcols, i, j);
            nw.trace.push_back(curr);
        }

        int max = std::numeric_limits<int>::min();
        int di = 0;
        int dj = 0;

        if (i > 0 && j > 0)
        {
            max = el(nw.score, nw.adjcols, i - 1, j - 1);
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
        if (i > 0 && max < el(nw.score, nw.adjcols, i - 1, j))
        {
            max = el(nw.score, nw.adjcols, i - 1, j);
            // DOWN^-1 -- align a gap in seqX to a letter in seqY. INSERTION in seqX.
            di = -1;
            dj = 0;
            edit = 'I';
        }
        if (j > 0 && max < el(nw.score, nw.adjcols, i, j - 1))
        {
            max = el(nw.score, nw.adjcols, i, j - 1);
            di = 0;
            dj = -1;
            // RIGHT^-1 -- align a letter in seqX to a gap in seqY. DELETION in seqX.
            edit = 'D';
        }
        i += di;
        j += dj;

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

    // Reverse the edit trace, so it starts from the top-left corner of the matrix.
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

NwStat NwHash1_Plain(NwAlgInput& nw, NwAlgResult& res)
{
    // http://www.cse.yorku.ca/~oz/hash.html
    unsigned hash = 5381;

    Stopwatch& sw = res.sw_hash;
    sw.start();

    for (int i = 0; i < nw.adjrows; i++)
    {
        for (int j = 0; j < nw.adjcols; j++)
        {
            int curr = el(nw.score, nw.adjcols, i, j);
            hash = ((hash << 5) + hash) ^ curr;
        }
    }
    res.score_hash = hash;

    sw.lap("hash.calc");

    return NwStat::success;
}

NwStat NwPrintScore1_Plain(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res)
{
    (void)res;
    NwPrintMat(os, nw.score.data(), nw.adjrows, nw.adjcols);
    return NwStat::success;
}

NwStat NwPrintTrace1_Plain(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res)
{
    (void)res;
    NwPrintVect(os, nw.trace.data(), nw.trace.size());
    return NwStat::success;
}
