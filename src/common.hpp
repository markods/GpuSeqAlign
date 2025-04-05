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
    NwParam();
    NwParam(std::vector<int> values);

    int curr() const;
    bool hasCurr() const;
    void next();
    void reset();

    std::vector<int> _values;
    int _currIdx = 0;
};

// parameters for the Needleman-Wunsch algorithm variant
struct NwParams
{
    NwParams();
    NwParams(std::map<std::string, NwParam> params);

    NwParam& operator[](const std::string name);

    bool hasCurr() const;
    // updates starting from the last parameter and so on
    void next();
    void reset();

    std::map<std::string, int> copy() const;

    std::map<std::string, NwParam> _params;
    bool _isEnd;
};

// input for the Needleman-Wunsch algorithm variant
struct NwInput
{
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
    // prefer using them instead of vector.size()
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
    void resetAllocsBenchmarkCycle();

    // free all remaining memory not cleared by resetAllocs
    void resetAllocsBenchmarkEnd();
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
