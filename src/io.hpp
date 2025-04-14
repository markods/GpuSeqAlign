#ifndef INCLUDE_FS_HPP
#define INCLUDE_FS_HPP

#include "defer.hpp"
#include "json.hpp"
#include "run_types.hpp"
#include <fstream>
#include <iostream>
#include <string>

// get the current time as an ISO string
NwStat isoDatetimeAsString(std::string& res);

// open input file stream
NwStat openInFile(const std::string& path, std::ifstream& ifs);

// open output file stream
NwStat openOutFile(const std::string& path, std::ofstream& ofs);

// read a json file into a variable
template <typename T>
NwStat readFromJsonFile(const std::string& path, T& res, std::string& error_msg)
{
    std::ifstream ifs;
    if (NwStat stat = openInFile(path, ifs); stat != NwStat::success)
    {
        error_msg = std::string("could not open file on path: \"" + path + "\"");
        return stat;
    }

    try
    {
        // The parser doesn't allow trailing commas.
        auto json = nlohmann::ordered_json::parse(
            ifs,
            /*callback*/ nullptr,
            /*allow_exceptions*/ true,
            /*ignore_comments*/ true);

        if (json.is_discarded())
        {
            return NwStat::errorInvalidFormat;
        }

        res = json;
    }
    catch (const std::exception& ex)
    {
        error_msg = ex.what();
        return NwStat::errorInvalidFormat;
    }

    return NwStat::success;
}

#endif // INCLUDE_FS_HPP
