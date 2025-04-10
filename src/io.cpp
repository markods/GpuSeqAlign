#include "io.hpp"
#include <chrono>
#include <filesystem>
#include <sstream>

// get the current time as an ISO string
std::string isoDatetimeAsString()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::tm tm_struct {};
    if (localtime_s(&tm_struct, &time) != 0)
    {
        throw std::runtime_error("Failed to get local time.");
    }

    std::stringstream strs;
    strs << std::put_time(&tm_struct, "%Y%m%d_%H%M%S");
    return strs.str();
}

// open input file stream
NwStat openInFile(const std::string& path, std::ifstream& ifs)
{
    ifs.exceptions(std::ios_base::goodbit);
    ifs.open(path, std::ios_base::in);
    if (!ifs)
    {
        return NwStat::errorIoStream;
    }

    return NwStat::success;
}

// open output file stream
NwStat openOutFile(const std::string& path, std::ofstream& ofs)
{
    try
    {
        // Create directories if they don't exist
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());

        ofs.exceptions(std::ios_base::goodbit);
        ofs.open(path, std::ios_base::out);
        if (!ofs)
        {
            return NwStat::errorIoStream;
        }
    }
    catch (const std::exception&)
    {
        return NwStat::errorIoStream;
    }

    return NwStat::success;
}
