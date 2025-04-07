#ifndef INCLUDE_FS_HPP
#define INCLUDE_FS_HPP

#include "json.hpp"
#include "run_types.hpp"
#include <fstream>
#include <iostream>
#include <string>

// get the current time as an ISO string
std::string isoDatetimeAsString();

// open output file stream
NwStat openOutFile(const std::string& path, std::ofstream& ofs);

// read a json file into a variable
template <typename T>
NwStat readFromJsonFile(const std::string& path, T& res)
{
    std::ifstream ifs;

    ifs.exceptions(std::ios_base::goodbit);
    ifs.open(path, std::ios_base::in);
    if (!ifs)
    {
        return NwStat::errorIoStream;
    }

    auto defer1 = make_defer([&]() noexcept
    {
        ifs.close();
    });

    // NOTE: the parser doesn't allow for trailing commas
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
