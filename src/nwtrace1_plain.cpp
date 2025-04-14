#include "print_mat.hpp"
#include "run_types.hpp"
#include <limits>

// TODO: don't calculate default trace (numbers) unless asked
// TODO: same_letter_cnt

// TODO: smith waterman
// TODO: affine gap

// TODO: remove comments in nwalign algorithms (part 3)
// TODO: fix: Exception thrown at 0x00007FF83227016C in nw.exe: Microsoft C++ exception: std::out_of_range at memory location 0x000000A7E93FB4C8.

NwStat NwTrace1_Plain(NwAlgInput& nw, NwAlgResult& res)
{
    Stopwatch& sw = res.sw_trace;
    sw.start();

    try
    {
        nw.trace.reserve(nw.adjrows - 1 + nw.adjcols); // Longest possible path.
        res.edit_trace.reserve(nw.adjrows - 1 + nw.adjcols);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    sw.lap("trace.alloc");

    int i = nw.adjrows - 1;
    int j = nw.adjcols - 1;
    while (true)
    {
        int curr = el(nw.score, nw.adjcols, i, j);
        nw.trace.push_back(curr);

        int max = std::numeric_limits<int>::min();
        int di = 0;
        int dj = 0;
        char edit = '\0';

        if (i > 0 && j > 0)
        {
            max = el(nw.score, nw.adjcols, i - 1, j - 1);
            di = -1;
            dj = -1;
            if (nw.seqX[j] == nw.seqY[i])
            {
                // DIAGONAL^-1 -- match.
                edit = 'M';
            }
            else
            {
                // DIAGONAL^-1 -- substitution.
                edit = 'S';
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

        if (di == 0 && dj == 0)
        {
            break;
        }
        res.edit_trace.push_back(edit);
    }

    // Reverse the edit trace, so it starts from the top-left corner of the matrix.
    std::reverse(res.edit_trace.begin(), res.edit_trace.end());

    sw.lap("trace.calc");

    // Reverse the trace, so it starts from the top-left corner of the matrix.
    std::reverse(nw.trace.begin(), nw.trace.end());

    // http://www.cse.yorku.ca/~oz/hash.html
    unsigned hash = 5381;

    for (auto& curr : nw.trace)
    {
        hash = ((hash << 5) + hash) ^ curr;
    }
    for (auto& curr : res.edit_trace)
    {
        hash = ((hash << 5) + hash) ^ curr;
    }

    res.trace_hash = hash;

    return NwStat::success;
}

NwStat NwHash1_Plain(const NwAlgInput& nw, NwAlgResult& res)
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

NwStat NwPrintScore1_Plain(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res)
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
