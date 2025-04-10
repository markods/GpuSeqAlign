#ifndef INCLUDE_FS_HPP
#define INCLUDE_FS_HPP

#include "defer.hpp"
#include "json.hpp"
#include "run_types.hpp"
#include <fstream>
#include <iostream>
#include <string>

// get the current time as an ISO string
std::string isoDatetimeAsString();

// open input file stream
NwStat openInFile(const std::string& path, std::ifstream& ifs);

// open output file stream
NwStat openOutFile(const std::string& path, std::ofstream& ofs);

// read a json file into a variable
template <typename T>
NwStat readFromJsonFile(const std::string& path, T& res)
{
    std::ifstream ifs;
    ZIG_TRY(NwStat::success, openInFile(path, ifs));

    // The parser doesn't allow trailing commas.
    auto json = nlohmann::ordered_json::parse(
        ifs,
        /*callback*/ nullptr,
        /*allow_exceptions*/ false,
        /*ignore_comments*/ true);

    if (json.is_discarded())
    {
        return NwStat::errorInvalidFormat;
    }

    try
    {
        res = json;
    }
    catch (const std::exception&)
    {
        return NwStat::errorInvalidFormat;
    }

    return NwStat::success;
}

#endif // INCLUDE_FS_HPP
