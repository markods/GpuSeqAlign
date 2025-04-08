#include "stopwatch.hpp"

// Combine many stopwatches into one.
Stopwatch Stopwatch::combine(std::vector<Stopwatch>& swList)
{
    // if the stopwatch list is empty, return a default initialized stopwatch
    if (swList.empty())
    {
        return Stopwatch {};
    }

    // copy on purpose here -- don't modify the given stopwatch list
    Stopwatch res {};
    // the number of times the lap was found in the stopwatches
    std::map<std::string, int> lapCount {};

    // for all stopwatches + for all laps in a stopwatch, get the average lap time
    // +   for the average, don't count non-existent values in the denominator
    for (auto& sw : swList)
    {
        for (auto& lapTuple : sw._laps)
        {
            const std::string& lapName = lapTuple.first;
            float lapTime = lapTuple.second;

            // insert or set default value! -- no initialization necessary before addition
            res._laps[lapName] += lapTime;
            lapCount[lapName]++;
        }
    }

    // for all laps in the result; divide them by the number of their occurences
    for (auto& lapTuple : res._laps)
    {
        // divide the total lap time by the number of lap occurences
        const std::string& lapName = lapTuple.first;
        res._laps[lapName] /= lapCount[lapName];
    }

    return res;
}

void Stopwatch::start()
{
    _start = Clock::now();
}
// If the lap time with the specified name already exists, increment it.
void Stopwatch::lap(std::string lap_name)
{
    auto curr = Clock::now();
    float diff = float(std::chrono::duration_cast<std::chrono::nanoseconds>(curr - _start).count()) / 1000000;
    _start = curr;

    _laps[lap_name] = _laps[lap_name] + diff;
}
void Stopwatch::reset() noexcept
{
    _start = {};
    _laps.clear();
}

float Stopwatch::total() const
{
    float sum = 0;
    for (auto& lap : _laps)
    {
        sum += lap.second;
    }
    return sum;
}

float Stopwatch::get_or_default(const std::string& lap_name) const
{
    try
    {
        return _laps.at(lap_name);
    }
    catch (const std::exception&)
    {
        return 0;
    }
}
const std::map<std::string, float>& Stopwatch::laps() const
{
    return _laps;
}
