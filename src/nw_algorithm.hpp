#ifndef INCLUDE_NW_ALGORITHM_HPP
#define INCLUDE_NW_ALGORITHM_HPP

#include "dict.hpp"
#include "run_types.hpp"
#include <string>

class NwAlgorithm
{
public:
    using NwAlignFn = NwStat (*)(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
    using NwTraceFn = NwStat (*)(NwAlgInput& nw, NwAlgResult& res);
    using NwHashFn = NwStat (*)(const NwAlgInput& nw, NwAlgResult& res);
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

    NwStat align(const NwAlgParams& algParams, NwAlgInput& nw, NwAlgResult& res);
    NwStat trace(NwAlgInput& nw, NwAlgResult& res);
    NwStat hash(const NwAlgInput& nw, NwAlgResult& res);
    NwStat printScore(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res);
    NwStat printTrace(std::ostream& os, const NwAlgInput& nw, const NwAlgResult& res);

private:
    NwAlignFn _alignFn;
    NwTraceFn _traceFn;
    NwHashFn _hashFn;
    NwPrintScoreFn _printScoreFn;
    NwPrintTraceFn _printTraceFn;
};

void getNwAlgorithmMap(Dict<std::string, NwAlgorithm>& algMap);

#endif // INCLUDE_NW_ALGORITHM_HPP
