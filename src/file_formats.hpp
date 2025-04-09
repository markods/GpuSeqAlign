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

    static void to_json(nlohmann::ordered_json& j, const Dict<K, V, Hash, KeyEqual, Allocator>& dict)
    {
        nlohmann::ordered_json json_entries = nlohmann::ordered_json::object();

        for (const auto& entry : dict)
        {
            json_entries[entry.first] = entry.second;
        }

        j = json_entries;
    }
};

template <>
struct adl_serializer<NwSubstData>
{
    static void from_json(const nlohmann::ordered_json& j, NwSubstData& substData)
    {
        if (j.size() != 2)
        {
            throw std::exception("Expected a JSON object with exactly two keys: \"letterMap\", \"substMap\"");
        }
        j.at("letterMap").get_to(substData.letterMap);
        j.at("substMap").get_to(substData.substMap);
    }
    static void to_json(nlohmann::ordered_json& j, const NwSubstData& substData)
    {
        j["letterMap"] = substData.letterMap;
        j["substMap"] = substData.substMap;
    }
};

template <>
struct adl_serializer<NwAlgParam>
{
    static void from_json(const nlohmann::ordered_json& j, NwAlgParam& param)
    {
        j.get_to(param._values);
    }
    static void to_json(nlohmann::ordered_json& j, const NwAlgParam& param)
    {
        j = param._values;
    }
};

template <>
struct adl_serializer<NwAlgParams>
{
    static void from_json(const nlohmann::ordered_json& j, NwAlgParams& params)
    {
        j.get_to(params._params);
    }
    static void to_json(nlohmann::ordered_json& j, const NwAlgParams& params)
    {
        j = params._params;
    }
};

template <>
struct adl_serializer<NwAlgParamsData>
{
    static void from_json(const nlohmann::ordered_json& j, NwAlgParamsData& paramData)
    {
        j.get_to(paramData.paramMap);
    }
    static void to_json(nlohmann::ordered_json& j, const NwAlgParamsData& paramData)
    {
        j["paramMap"] = paramData.paramMap;
    }
};

template <>
struct adl_serializer<NwSeqData>
{
    static void from_json(const nlohmann::ordered_json& j, NwSeqData& seqData)
    {
        if (j.size() != 1)
        {
            throw std::exception("Expected a JSON object with exactly one key: \"seqList\"");
        }
        j.at("seqList").get_to(seqData.seqList);
    }
    static void to_json(nlohmann::ordered_json& j, const NwSeqData& seqData)
    {
        j["seqList"] = seqData.seqList;
    }
};

} // namespace nlohmann

void writeResultHeaderToTsv(std::ostream& os, bool fPrintScoreStats, bool fPrintTraceStats);
void writeResultLineToTsv(std::ostream& os, const NwAlgResult& res, bool fPrintScoreStats, bool fPrintTraceStats);

#endif // INCLUDE_FILE_FORMATS_HPP
