#ifndef INCLUDE_NW_ALGORITHM_HPP
#define INCLUDE_NW_ALGORITHM_HPP

#include "dict.hpp"
#include "run_types.hpp"
#include <string>

class NwAlgorithm
{
public:
    using NwAlignFn = NwStat (*)(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
    using NwTraceFn = NwStat (*)(NwAlgInput& nw, NwAlgResult& res, bool calcDebugTrace);
    using NwHashFn = NwStat (*)(NwAlgInput& nw, NwAlgResult& res);
    using NwPrintScoreFn = NwStat (*)(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res);
    using NwPrintTraceFn = NwStat (*)(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res);

public:
    NwAlgorithm();

    NwAlgorithm(
        NwAlignFn alignFn,
        NwTraceFn traceFn,
        NwHashFn hashFn,
        NwPrintScoreFn printScoreFn,
        NwPrintTraceFn printTraceFn);

    // Align seqX to seqY (seqX becomes seqY).
    NwStat align(const NwAlgParams& algParams, NwAlgInput& nw, NwAlgResult& res) const;
    NwStat trace(NwAlgInput& nw, NwAlgResult& res, bool calcDebugTrace) const;
    NwStat hash(NwAlgInput& nw, NwAlgResult& res) const;
    NwStat printScore(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res) const;
    NwStat printTrace(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res) const;

private:
    NwAlignFn _alignFn;
    NwTraceFn _traceFn;
    NwHashFn _hashFn;
    NwPrintScoreFn _printScoreFn;
    NwPrintTraceFn _printTraceFn;
};

void getNwAlgorithmMap(Dict<std::string, NwAlgorithm>& algMap);

#endif // INCLUDE_NW_ALGORITHM_HPP
