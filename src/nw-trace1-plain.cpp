#include "common.hpp"
#include <limits>

// get one of the optimal matching paths to a file
NwStat NwTrace1_Plain(const NwInput& nw, NwResult& res)
{
    // variable used to calculate the hash function
    // http://www.cse.yorku.ca/~oz/hash.html
    // the starting value is a magic constant
    unsigned hash = 5381;
    // vector containing the trace
    std::vector<int> trace;

    // start the timer
    Stopwatch& sw = res.sw_trace;
    sw.start();

    // reserve space in the ram
    try
    {
        trace.reserve(nw.adjrows - 1 + nw.adjcols);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // measure trace time
    sw.lap("alloc");

    // for all elements on one of the optimal paths
    bool loop = true;
    for (int i = nw.adjrows - 1, j = nw.adjcols - 1; loop;)
    {
        // add the current element to the trace
        int curr = el(nw.score, nw.adjcols, i, j);
        trace.push_back(curr);

        int max = std::numeric_limits<int>::min(); // maximum value of the up, left and diagonal neighbouring elements
        int dir = '-';                             // the current movement direction is unknown

        if (i > 0 && j > 0 && max < el(nw.score, nw.adjcols, i - 1, j - 1))
        {
            max = el(nw.score, nw.adjcols, i - 1, j - 1);
            dir = 'i';
        } // diagonal movement if possible
        if (i > 0 && max < el(nw.score, nw.adjcols, i - 1, j))
        {
            max = el(nw.score, nw.adjcols, i - 1, j);
            dir = 'u';
        } // up       movement if possible
        if (j > 0 && max < el(nw.score, nw.adjcols, i, j - 1))
        {
            max = el(nw.score, nw.adjcols, i, j - 1);
            dir = 'l';
        } // left     movement if possible

        // move to the neighbour with the maximum value
        switch (dir)
        {
        case 'i':
            i--;
            j--;
            break;
        case 'u':
            i--;
            break;
        case 'l':
            j--;
            break;
        default:
            loop = false;
            break;
        }
    }

    // reverse the trace, so it starts from the top-left corner of the matrix
    std::reverse(trace.begin(), trace.end());

    // measure trace time
    sw.lap("calc-1");

    // calculate the hash value
    for (auto& curr : trace)
    {
        hash = ((hash << 5) + hash) ^ curr;
    }

    // save the hash value
    res.trace_hash = hash;

    // measure trace time
    sw.lap("calc-2");

    return NwStat::success;
}

// hash the score matrix
NwStat NwHash1_Plain(const NwInput& nw, NwResult& res)
{
    // variable used to calculate the hash function
    // http://www.cse.yorku.ca/~oz/hash.html
    // the starting value is a magic constant
    unsigned hash = 5381;

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

    return NwStat::success;
}

// print the score matrix
NwStat NwPrint1_Plain(std::ostream& os, const NwInput& nw, NwResult& res)
{
    (void)res;
    NwPrintMat(os, nw.score.data(), nw.adjrows, nw.adjcols);
    return NwStat::success;
}
