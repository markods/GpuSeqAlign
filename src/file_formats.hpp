#ifndef INCLUDE_FILE_FORMATS_HPP
#define INCLUDE_FILE_FORMATS_HPP

#include "dict.hpp"
#include "json.hpp"
#include "run_types.hpp"
#include <iostream>
#include <string>
#include <vector>

struct NwSubstData
{
    Dict<std::string, int> letterMap;
    Dict<std::string, std::vector<int>> substMap;
};

struct NwAlgParamsData
{
    Dict<std::string, NwAlgParams> paramMap;
};

struct NwSeqData
{
    // Sequence has a header (zeroth) element.
    std::vector<std::string> seqList;
};

namespace nlohmann
{
template <typename K, typename V, typename Hash, typename KeyEqual, typename Allocator>
struct adl_serializer<Dict<K, V, Hash, KeyEqual, Allocator>>
{
    static void to_json(nlohmann::ordered_json& j, const Dict<K, V, Hash, KeyEqual, Allocator>& dict)
    {
        nlohmann::ordered_json json_entries = nlohmann::ordered_json::object();

        for (const auto& entry : dict)
        {
            json_entries[entry.first] = entry.second;
        }

        j = json_entries;
    }

    static void from_json(const nlohmann::ordered_json& j, Dict<K, V, Hash, KeyEqual, Allocator>& dict)
    {
        dict.clear();

        for (const auto& element : j.items())
        {
            K key = element.key();
            V value = element.value();
            dict.insert(std::move(key), std::move(value));
        }
    }
};
} // namespace nlohmann

void from_json(const nlohmann::ordered_json& j, NwSubstData& substData);
void from_json(const nlohmann::ordered_json& j, NwAlgParam& param);
void from_json(const nlohmann::ordered_json& j, NwAlgParams& params);
void from_json(const nlohmann::ordered_json& j, NwAlgParamsData& paramData);
void from_json(const nlohmann::ordered_json& j, NwSeqData& seqData);

void to_json(nlohmann::ordered_json& j, const NwSubstData& substData);
void to_json(nlohmann::ordered_json& j, const NwAlgParam& param);
void to_json(nlohmann::ordered_json& j, const NwAlgParams& params);
void to_json(nlohmann::ordered_json& j, const NwAlgParamsData& paramData);
void to_json(nlohmann::ordered_json& j, const NwSeqData& seqData);

void writeResultHeaderToTsv(std::ostream& os, bool fPrintScoreStats, bool fPrintTraceStats);
void writeResultLineToTsv(std::ostream& os, const NwAlgResult& res, bool fPrintScoreStats, bool fPrintTraceStats);

#endif // INCLUDE_FILE_FORMATS_HPP
