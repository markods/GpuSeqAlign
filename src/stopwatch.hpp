#ifndef INCLUDE_STOPWATCH_HPP
#define INCLUDE_STOPWATCH_HPP

#include <chrono>
#include <map>
#include <string>
#include <vector>

// Measure time between events in milliseconds.
class Stopwatch
{
public:
    // Combine many stopwatches into one.
    static Stopwatch combine(std::vector<Stopwatch>& swList);

public:
    void start();
    // If the lap time with the specified name already exists, increment it.
    void lap(std::string lap_name);
    void reset() noexcept;

    float total() const;

    float get_or_default(const std::string& lap_name) const;
    const std::map<std::string, float>& laps() const;

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint _start;
    std::map<std::string, float> _laps;
};

#endif // INCLUDE_STOPWATCH_HPP
