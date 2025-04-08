#ifndef INCLUDE_FILE_FORMATS_HPP
#define INCLUDE_FILE_FORMATS_HPP

#include "json.hpp"
#include "run_types.hpp"
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct NwSubstData
{
    std::map<std::string, int> letterMap;
    std::map<std::string, std::vector<int>> substMap;
};

// TODO: order maps by insertion order, not by string
struct NwAlgParamsData
{
    std::map<std::string, NwAlgParams> paramMap;
};

struct NwSeqData
{
    // Sequence has a header (zeroth) element.
    std::vector<std::string> seqList;
};

void from_json(const nlohmann::ordered_json& j, NwSubstData& substData);
void from_json(const nlohmann::ordered_json& j, NwAlgParam& param);
void from_json(const nlohmann::ordered_json& j, NwAlgParams& params);
void from_json(const nlohmann::ordered_json& j, NwAlgParamsData& paramData);
void from_json(const nlohmann::ordered_json& j, NwSeqData& seqData);

void writeResultHeaderToTsv(std::ostream& os, bool fPrintScoreStats, bool fPrintTraceStats);
void writeResultLineToTsv(std::ostream& os, const NwAlgResult& res, bool fPrintScoreStats, bool fPrintTraceStats);

#endif // INCLUDE_FILE_FORMATS_HPP
