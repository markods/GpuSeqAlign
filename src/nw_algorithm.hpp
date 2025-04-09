#ifndef INCLUDE_NW_ALGORITHM_HPP
#define INCLUDE_NW_ALGORITHM_HPP

#include "dict.hpp"
#include "run_types.hpp"
#include <string>

class NwAlgorithm
{
public:
    using NwAlignFn = NwStat (*)(const NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
    using NwTraceFn = NwStat (*)(const NwAlgInput& nw, NwAlgResult& res);
    using NwHashFn = NwStat (*)(const NwAlgInput& nw, NwAlgResult& res);
    using NwPrintFn = NwStat (*)(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);

public:
    NwAlgorithm();

    NwAlgorithm(
        NwAlignFn alignFn,
        NwTraceFn traceFn,
        NwHashFn hashFn,
        NwPrintFn printFn);

    NwStat align(const NwAlgParams& algParams, NwAlgInput& nw, NwAlgResult& res);
    NwStat trace(const NwAlgInput& nw, NwAlgResult& res);
    NwStat hash(const NwAlgInput& nw, NwAlgResult& res);
    NwStat print(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);

private:
    NwAlignFn _alignFn;
    NwTraceFn _traceFn;
    NwHashFn _hashFn;
    NwPrintFn _printFn;
};

void getNwAlgorithmMap(Dict<std::string, NwAlgorithm>& algMap);

#endif // INCLUDE_NW_ALGORITHM_HPP
