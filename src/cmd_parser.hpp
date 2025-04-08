#ifndef INCLUDE_CMD_PARSER_HPP
#define INCLUDE_CMD_PARSER_HPP

#include "file_formats.hpp"
#include <fstream>

struct NwCmdArgs
{
    std::optional<std::string> substPath;
    std::optional<std::string> algParamPath;
    std::optional<std::string> seqPath;  // TODO: fasta format
    std::optional<std::string> pairPath; // TODO: parse and use
    std::optional<std::string> resPath;

    std::optional<std::string> substName;
    std::optional<int> gapoCost;
    std::optional<int> gapeCost;
    std::optional<std::vector<std::string>> algNames; // TODO: check
    std::optional<std::string> refAlgName;            // TODO: first one from algNames by default
    std::optional<int> warmupPerAlign;
    std::optional<int> samplesPerAlign;

    std::optional<bool> fCalcTrace;
    std::optional<bool> fCalcScoreHash;
    std::optional<bool> fWriteProgress;
    std::optional<std::string> debugPath; // TODO: use
    std::optional<bool> fPrintScore;      // TODO: use
    std::optional<bool> fPrintTrace;      // TODO: use
};

struct NwCmdData
{
    NwSubstData substData;
    NwAlgParamsData algParamsData;
    NwSeqData seqData;
    // NwPairData pairData; // TODO
    std::ofstream resOfs;

    std::ofstream debugOfs;
};

NwStat parseCmdArgs(const int argc, const char* argv[], NwCmdArgs& cmdArgs);
NwStat initCmdData(const NwCmdArgs& cmdArgs, NwCmdData& cmdData);

#endif // INCLUDE_CMD_PARSER_HPP
