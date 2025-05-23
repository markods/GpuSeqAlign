#ifndef INCLUDE_CMD_PARSER_HPP
#define INCLUDE_CMD_PARSER_HPP

#include "file_formats.hpp"
#include <fstream>
#include <optional>
#include <string>
#include <vector>

struct NwCmdArgs
{
    std::optional<std::string> substPath;
    std::optional<std::string> algParamPath;
    std::optional<std::string> seqPath;
    std::optional<std::string> seqPairPath;
    std::optional<std::string> resPath;

    std::optional<std::string> substName;
    std::optional<int> gapoCost;
    std::optional<int> gapeCost;
    std::optional<std::vector<std::string>> algNames;
    std::optional<std::string> refAlgName;
    std::optional<int> warmupPerAlign;
    std::optional<int> samplesPerAlign;

    std::optional<bool> fCalcTrace;
    std::optional<bool> fCalcScoreHash;
    std::optional<bool> fWriteProgress;
    std::optional<std::string> debugPath;
    std::optional<bool> fPrintScore;
    std::optional<bool> fPrintTrace;

    std::string isoDateTime;
};

struct NwCmdData
{
    NwSubstData substData;
    NwAlgParamsData algParamsData;
    NwSeqData seqData;
    NwSeqPairData seqPairData;
    std::ofstream resOfs;

    std::ofstream debugOfs;
};

NwStat parseCmdArgs(const int argc, const char* argv[], NwCmdArgs& cmdArgs);
NwStat initCmdData(NwCmdArgs& cmdArgs, NwCmdData& cmdData);

#endif // INCLUDE_CMD_PARSER_HPP
