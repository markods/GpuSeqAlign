#include "io.hpp"
#include <chrono>
#include <filesystem>
#include <sstream>

NwStat isoDatetimeAsString(std::string& res)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::tm tm_struct {};

#if defined(_WIN32) || defined(_WIN64)
    if (localtime_s(&tm_struct, &time) != 0)
    {
        return NwStat::errorInvalidValue;
    }
#elif defined(__linux__)
    if (localtime_r(&time, &tm_struct) == nullptr)
    {
        return NwStat::errorInvalidValue;
    }
#else
    #error "error: unknown os"
#endif

    std::stringstream strs {};
    strs << std::put_time(&tm_struct, "%Y%m%d_%H%M%S");
    res = strs.str();

    return NwStat::success;
}

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

// Create directories if they don't exist
NwStat openOutFile(const std::string& path, std::ofstream& ofs)
{
    try
    {
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
