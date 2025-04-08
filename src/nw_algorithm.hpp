#ifndef INCLUDE_NW_ALGORITHM_HPP
#define INCLUDE_NW_ALGORITHM_HPP

#include "run_types.hpp"
#include <map>
#include <string>

// the Needleman-Wunsch algorithm implementations
class NwAlgorithm
{
public:
    using NwAlignFn = NwStat (*)(NwAlgParams& pr, NwAlgInput& nw, NwAlgResult& res);
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

    void init(NwAlgParams& alignPr);

    NwAlgParams& alignPr();

    NwStat align(NwAlgInput& nw, NwAlgResult& res);
    NwStat trace(const NwAlgInput& nw, NwAlgResult& res);
    NwStat hash(const NwAlgInput& nw, NwAlgResult& res);
    NwStat print(std::ostream& os, const NwAlgInput& nw, NwAlgResult& res);

private:
    NwAlignFn _alignFn;
    NwTraceFn _traceFn;
    NwHashFn _hashFn;
    NwPrintFn _printFn;

    NwAlgParams _alignPr;
};

void getNwAlgorithmMap(std::map<std::string, NwAlgorithm>& algMap);

#endif // INCLUDE_NW_ALGORITHM_HPP
