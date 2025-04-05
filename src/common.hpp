#ifndef INCLUDE_COMMON_HPP
#define INCLUDE_COMMON_HPP

#include "math.hpp"
#include "memory.hpp"
#include "stopwatch.hpp"
#include <map>
#include <string>
#include <utility>
#include <vector>

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

    std::map<std::string, int> copy() const
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

#endif // INCLUDE_COMMON_HPP
