#include "print_mat.hpp"
#include "run_types.hpp"
#include <limits>

// TODO: don't calculate default trace (numbers) unless asked
// TODO: same_letter_cnt

// get one of the optimal matching paths to a file
NwStat NwTrace1_Plain(NwAlgInput& nw, NwAlgResult& res)
{
    // start the timer
    Stopwatch& sw = res.sw_trace;
    sw.start();

    // reserve space in the ram
    try
    {
        nw.trace.reserve(nw.adjrows - 1 + nw.adjcols); // Longest possible path.
        res.edit_trace.reserve(nw.adjrows - 1 + nw.adjcols);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // measure allocation time
    sw.lap("trace.alloc");

    int i = nw.adjrows - 1;
    int j = nw.adjcols - 1;
    // for all elements on one of the optimal paths
    while (true)
    {
        int currElem = el(nw.score, nw.adjcols, i, j);
        nw.trace.push_back(currElem);

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

    // reverse the trace, so it starts from the top-left corner of the matrix
    std::reverse(nw.trace.begin(), nw.trace.end());
    std::reverse(res.edit_trace.begin(), res.edit_trace.end());

    // measure trace time
    sw.lap("trace.calc");

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

// hash the score matrix
NwStat NwHash1_Plain(const NwAlgInput& nw, NwAlgResult& res)
{
    // variable used to calculate the hash function
    // http://www.cse.yorku.ca/~oz/hash.html
    // the starting value is a magic constant
    unsigned hash = 5381;

    // Start the timer.
    Stopwatch& sw = res.sw_hash;
    sw.start();

    for (int i = 0; i < nw.adjrows; i++)
    {
        for (int j = 0; j < nw.adjcols; j++)
        {
            // add the current element to the hash
            int curr = el(nw.score, nw.adjcols, i, j);
            hash = ((hash << 5) + hash) ^ curr;
        }
    }

    // save the resulting hash
    res.score_hash = hash;

    // Measure hash time.
    sw.lap("hash.calc");

    return NwStat::success;
}

// print the score matrix
NwStat NwPrintScore1_Plain(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res)
{
    (void)res;
    NwPrintMat(os, nw.score.data(), nw.adjrows, nw.adjcols);
    return NwStat::success;
}

// print the edit trace
NwStat NwPrintTrace1_Plain(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res)
{
    (void)res;
    NwPrintVect(os, nw.trace.data(), nw.trace.size());
    return NwStat::success;
}
