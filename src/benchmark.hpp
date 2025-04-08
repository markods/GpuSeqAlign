#ifndef INCLUDE_BENCHMARK_HPP
#define INCLUDE_BENCHMARK_HPP

#include "cmd_parser.hpp"
#include "run_types.hpp"
#include <vector>

struct NwBenchmarkData
{
    std::vector<NwAlgResult> resultList;
    int calcErrors;
};

NwStat benchmarkAlgs(const NwCmdArgs& cmdArgs, NwCmdData& cmdData, NwBenchmarkData& benchData);

#endif // INCLUDE_BENCHMARK_HPP
