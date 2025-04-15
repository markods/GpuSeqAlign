#include "stopwatch.hpp"

// Combine many stopwatches into one.
Stopwatch Stopwatch::combine(std::vector<Stopwatch>& swList)
{
    if (swList.empty())
    {
        return Stopwatch {};
    }

    // Copy on purpose here.
    Stopwatch res {};
    Dict<std::string, int> lapCount {};

    for (auto& sw : swList)
    {
        for (auto& lapTuple : sw._laps)
        {
            const std::string& lapName = lapTuple.first;
            float lapTime = lapTuple.second;

            // Insert or set default value (no initialization necessary before addition).
            res._laps[lapName] += lapTime;
            lapCount[lapName]++;
        }
    }

    // Don't count non-existent values in the denominator.
    for (auto& lapTuple : res._laps)
    {
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
    if (_laps.contains(lap_name))
    {
        return _laps.at(lap_name);
    }
    return 0;
}
const Dict<std::string, float>& Stopwatch::laps() const
{
    return _laps;
}
