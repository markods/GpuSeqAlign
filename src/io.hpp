#ifndef INCLUDE_FS_HPP
#define INCLUDE_FS_HPP

#include "defer.hpp"
#include "json.hpp"
#include "run_types.hpp"
#include <fstream>
#include <iostream>
#include <string>

NwStat isoDatetimeAsString(std::string& res);

NwStat openInFile(const std::string& path, std::ifstream& ifs);
NwStat openOutFile(const std::string& path, std::ofstream& ofs);

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
            nullptr /*callback*/,
            true /*allow_exceptions*/,
            true /*ignore_comments*/);

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
