#ifndef INCLUDE_COMMON_HPP
#define INCLUDE_COMMON_HPP

#include "fmt_guard.hpp"
#include "memory.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

// defer execution to scope exit
template <typename F>
class Defer
{
    // static_assert(std::is_nothrow_invocable_r_v<void, F>, "F must be a callable type with signature void() noexcept");

public:
    explicit Defer(F _func) noexcept
        : func {std::move(_func)}, active {true}
    { }

    Defer(const Defer&) = delete;
    Defer& operator=(const Defer&) = delete;

    Defer(Defer&& other) noexcept
        : func {std::move(other.func)}, active {other.active}
    {
        // The moved-from object must not run the function.
        other.active = false;
    }
    Defer& operator=(Defer&& other) noexcept
    {
        if (this != &other)
        {
            func = std::move(other.func);
            active = other.active;
            other.active = false;
        }
        return *this;
    }

    void operator()() noexcept
    {
        doDefer();
    }

    ~Defer() noexcept
    {
        doDefer();
    }

private:
    void doDefer() noexcept
    {
        if (active)
        {
            active = false;
            func();
        }
    }

private:
    F func;
    bool active;
};

template <typename F>
Defer<F> make_defer(F _func)
{
    return Defer<F>(std::move(_func));
}

// get the specified element from the given linearized matrix
#define el(mat, cols, i, j) (mat[(cols) * (i) + (j)])

// calculate the minimum of two numbers
inline const int& min2(const int& a, const int& b) noexcept
{
    return (a <= b) ? a : b;
}
// calculate the maximum of two numbers
inline const int& max2(const int& a, const int& b) noexcept
{
    return (a >= b) ? a : b;
}
// calculate the minimum of three numbers
inline const int& min3(const int& a, const int& b, const int& c) noexcept
{
    return (a <= b) ? ((a <= c) ? a : c) : ((b <= c) ? b : c);
}
// calculate the maximum of three numbers
inline const int& max3(const int& a, const int& b, const int& c) noexcept
{
    return (a >= b) ? ((a >= c) ? a : c) : ((b >= c) ? b : c);
}

// Needleman-Wunsch status
enum class NwStat : int
{
    success = 0,
    errorMemoryAllocation = 1,
    errorMemoryTransfer = 2,
    errorKernelFailure = 3,
    errorIoStream = 4,
    errorInvalidFormat = 5,
    errorInvalidValue = 6,
    errorInvalidResult = 7,
};

// measure time between events
class Stopwatch
{
public:
    // combine many stopwatches into one
    static Stopwatch combineStopwatches(std::vector<Stopwatch>& swList)
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

public:
    void start()
    {
        _start = Clock::now();
    }
    void lap(std::string lap_name)
    {
        auto curr = Clock::now();
        float diff = float(std::chrono::duration_cast<Micros>(curr - _start).count()) / 1000;
        _start = curr;

        _laps.insert_or_assign(lap_name, diff);
    }
    void reset() noexcept
    {
        _start = {};
        _laps.clear();
    }

    float total() const
    {
        float sum = 0;
        for (auto& lap : _laps)
        {
            sum += lap.second;
        }
        return sum;
    }

    float get_or_default(const std::string& lap_name) const
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
    const std::map<std::string, float>& laps() const
    {
        return _laps;
    }

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Micros = std::chrono::nanoseconds;

    TimePoint _start;
    std::map<std::string, float> _laps;
};

// parameter that takes values from a vector
struct NwParam
{
public:
    NwParam() = default;
    NwParam(std::vector<int> values)
    {
        _values = values;
        _currIdx = 0;
    }

    int curr() const
    {
        return _values[_currIdx];
    }
    bool hasCurr() const
    {
        return _currIdx < _values.size();
    }
    void next()
    {
        _currIdx++;
    }
    void reset()
    {
        _currIdx = 0;
    }

    std::vector<int> _values;
    int _currIdx = 0;
};

// parameters for the Needleman-Wunsch algorithm variant
struct NwParams
{
    NwParams()
    {
        _params = {};
        _isEnd = false;
    }
    NwParams(std::map<std::string, NwParam> params)
    {
        _params = params;
        // always allow the inital iteration, even if there are no params
        _isEnd = false;
    }

    NwParam& operator[](const std::string name)
    {
        return _params.at(name);
    }

    bool hasCurr() const
    {
        return !_isEnd;
    }
    void next() // updates starting from the last parameter and so on
    {
        for (auto iter = _params.rbegin(); iter != _params.rend(); iter++)
        {
            auto& param = iter->second;
            param.next();

            if (param.hasCurr())
            {
                return;
            }
            param.reset();
        }
        _isEnd = true;
    }
    void reset()
    {
        for (auto iter = _params.rbegin(); iter != _params.rend(); iter++)
        {
            auto& param = iter->second;
            param.reset();
        }
        _isEnd = false;
    }

    std::map<std::string, int> snapshot() const
    {
        std::map<std::string, int> res;
        for (const auto& paramTuple : _params)
        {
            const std::string& paramName = paramTuple.first;
            int paramValue = paramTuple.second.curr();

            res[paramName] = paramValue;
        }

        return res;
    }

    std::map<std::string, NwParam> _params;
    bool _isEnd;
};

// input for the Needleman-Wunsch algorithm variant
struct NwInput
{
    // IMPORTANT: dont't use .size() on vectors to get the number of elements, since it is not accurate
    // +   instead, use the alignment parameters below

    ////// host specific memory
    std::vector<int> subst;
    std::vector<int> seqX;
    std::vector<int> seqY;
    HostArray<int> score;
    // sparse representation of the score matrix
    HostArray<int> tileHrowMat;
    HostArray<int> tileHcolMat;

    ////// device specific memory
    DeviceArray<int> subst_gpu;
    DeviceArray<int> seqX_gpu;
    DeviceArray<int> seqY_gpu;
    DeviceArray<int> score_gpu;
    // sparse representation of the score matrix
    DeviceArray<int> tileHrowMat_gpu;
    DeviceArray<int> tileHcolMat_gpu;

    // alignment parameters
    int substsz;
    int adjrows;
    int adjcols;
    int indel;
    // sparse representation of the score matrix
    int tileHdrMatRows;
    int tileHdrMatCols;
    int tileHrowLen;
    int tileHcolLen;

    // device parameters
    int sm_count;
    int warpsz;
    int maxThreadsPerBlock;

    // free all memory allocated by the Needleman-Wunsch algorithms
    void resetAllocsBenchmarkCycle()
    {
        // NOTE: first free device memory, since there is less of it for other algorithms

        ////// device specific memory
        // subst_gpu.clear();
        seqX_gpu.clear();
        seqY_gpu.clear();
        score_gpu.clear();
        ////// sparse representation of score matrix
        tileHrowMat_gpu.clear();
        tileHcolMat_gpu.clear();

        ////// host specific memory
        // subst.clear();
        // seqX.clear();
        // seqY.clear();
        score.clear();
        ////// sparse representation of score matrix
        tileHrowMat.clear();
        tileHcolMat.clear();
    }

    // free all remaining memory not cleared by resetAllocs
    void resetAllocsBenchmarkEnd()
    {
        // NOTE: first free device memory, since there is less of it for other algorithms

        ////// device specific memory
        subst_gpu.clear();

        ////// host specific memory
        subst.clear();
        seqX.clear();
        seqY.clear();
    }
};

// results which the Needleman-Wunsch algorithm variant returns
struct NwResult
{
    std::string algName;
    std::map<std::string, int> algParams;

    size_t seqX_len;
    size_t seqY_len;

    int iX;
    int iY;
    int reps;

    Stopwatch sw_align;
    Stopwatch sw_hash;
    Stopwatch sw_trace;

    unsigned score_hash;
    unsigned trace_hash;

    int errstep;          // 0 for success
    NwStat stat;          // 0 for success
    cudaError_t cudaStat; // 0 for success
};

// update the score given the current score matrix and position
// NOTE: indel and most elements in the substitution matrix are negative, therefore find the maximum of them (instead of the minimum)
inline void UpdateScore(NwInput& nw, int i, int j) noexcept
{
    int p1 = el(nw.score, nw.adjcols, i - 1, j - 1) + el(nw.subst, nw.substsz, nw.seqY[i], nw.seqX[j]); // MOVE DOWN-RIGHT
    int p2 = el(nw.score, nw.adjcols, i - 1, j) + nw.indel;                                             // MOVE DOWN
    int p3 = el(nw.score, nw.adjcols, i, j - 1) + nw.indel;                                             // MOVE RIGHT
    el(nw.score, nw.adjcols, i, j) = max3(p1, p2, p3);
}

#endif // INCLUDE_COMMON_HPP
