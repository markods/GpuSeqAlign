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

struct NwSeq
{
    std::string id;
    std::string info;
    // Sequence has a header (zeroth) element.
    std::vector<int> seq;
};

struct NwSeqData
{
    Dict<std::string, NwSeq> seqMap;
};

struct NwSeqPair
{
    std::string seqY_id;
    std::string seqX_id;
    NwRange seqY_range;
    NwRange seqX_range;
};

struct NwSeqPairData
{
    std::vector<NwSeqPair> pairList;
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

} // namespace nlohmann

NwStat readFromFastaFormat(
    const std::string& path,
    std::istream& is,
    NwSeqData& seqData,
    const Dict<std::string, int>& letterMap,
    std::string& error_msg);

NwStat readFromSeqPairFormat(
    const std::string& path,
    std::istream& is,
    NwSeqPairData& seqPairData,
    const Dict<std::string, NwSeq>& seqMap,
    std::string& error_msg);

struct TsvPrintCtl
{
    // Enum class turned out to be too much hassle.

    unsigned writeColName : 1;
    unsigned writeValue : 1;
    unsigned fPrintScoreStats : 1;
    unsigned fPrintTraceStats : 1;
};

NwStat writeNwResultToTsv(std::ostream& os, const NwAlgResult& res, const TsvPrintCtl printCtl);

#endif // INCLUDE_FILE_FORMATS_HPP
