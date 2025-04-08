#include "benchmark.hpp"
#include "cmd_parser.hpp"

int main(const int argc, const char* argv[])
{
    NwCmdArgs cmdArgs {};
    if (NwStat stat = parseCmdArgs(argc, argv, cmdArgs); stat != NwStat::success)
    {
        if (stat == NwStat::helpMenuRequested)
        {
            return 0;
        }
        return -1;
    }

    NwCmdData cmdData {};
    if (NwStat stat = initCmdData(cmdArgs, cmdData); stat != NwStat::success)
    {
        return -1;
    }

    NwBenchmarkData benchData {};
    if (NwStat stat = benchmarkAlgs(cmdArgs, cmdData, benchData); stat != NwStat::success)
    {
        return -1;
    }
}
