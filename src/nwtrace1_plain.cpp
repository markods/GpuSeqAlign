#include "print_mat.hpp"
#include "run_types.hpp"
#include <limits>

// get one of the optimal matching paths to a file
NwStat NwTrace1_Plain(NwAlgInput& nw, NwAlgResult& res)
{
    // variable used to calculate the hash function
    // http://www.cse.yorku.ca/~oz/hash.html
    // the starting value is a magic constant
    unsigned hash = 5381;

    // start the timer
    Stopwatch& sw = res.sw_trace;
    sw.start();

    // reserve space in the ram
    try
    {
        nw.trace.reserve(nw.adjrows - 1 + nw.adjcols);
    }
    catch (const std::exception&)
    {
        return NwStat::errorMemoryAllocation;
    }

    // measure allocation time
    sw.lap("trace.alloc");

    // for all elements on one of the optimal paths
    bool loop = true;
    for (int i = nw.adjrows - 1, j = nw.adjcols - 1; loop;)
    {
        // add the current element to the trace
        int curr = el(nw.score, nw.adjcols, i, j);
        nw.trace.push_back(curr);

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
    std::reverse(nw.trace.begin(), nw.trace.end());

    // measure trace time
    sw.lap("trace.calc");

    // calculate the hash value
    for (auto& curr : nw.trace)
    {
        hash = ((hash << 5) + hash) ^ curr;
    }

    // save the hash value
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
