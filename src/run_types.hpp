#ifndef INCLUDE_RUN_TYPES_HPP
#define INCLUDE_RUN_TYPES_HPP

#include "dict.hpp"
#include "math.hpp"
#include "memory.hpp"
#include "stopwatch.hpp"
#include <vector>

// Needleman-Wunsch status
enum class NwStat : int
{
    success,
    helpMenuRequested,
    errorCudaGeneral,
    errorMemoryAllocation,
    errorMemoryTransfer,
    errorKernelFailure,
    errorIoStream,
    errorInvalidFormat,
    errorInvalidValue,
    errorInvalidResult,
};

// parameter that takes values from a vector
struct NwAlgParam
{
    NwAlgParam();
    NwAlgParam(std::vector<int> values);

    int curr() const;
    bool hasCurr() const;
    void next();
    void reset();

    std::vector<int> _values;
    int _currIdx = 0;
};

// parameters for the Needleman-Wunsch algorithm variant
struct NwAlgParams
{
    NwAlgParams();
    NwAlgParams(Dict<std::string, NwAlgParam> params);

    NwAlgParam& at(const std::string name);
    const NwAlgParam& at(const std::string name) const;

    bool hasCurr() const;
    // updates starting from the last parameter and so on
    void next();
    void reset();

    Dict<std::string, int> copy() const;

    Dict<std::string, NwAlgParam> _params;
    bool _isEnd;
};

// input for the Needleman-Wunsch algorithm variant
struct NwAlgInput
{
    ////// host specific memory
    std::vector<int> subst;
    std::vector<int> seqX;
    std::vector<int> seqY;
    HostArray<int> score;
    std::vector<int> trace; // TODO: char vector of Match/Mismatch, Insertion, Deletion
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
    int gapoCost;
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
struct NwAlgResult
{
    std::string algName;
    Dict<std::string, int> algParams;
    std::string seqY_id;
    std::string seqX_id;

    int errstep;          // 0 for success
    NwStat stat;          // 0 for success
    cudaError_t cudaStat; // 0 for success

    size_t seqY_len;
    size_t seqX_len;
    std::string substName;
    int gapoCost;
    int warmup_runs;
    int sample_runs;

    int align_cost;
    unsigned score_hash;
    unsigned trace_hash;

    Stopwatch sw_align;
    Stopwatch sw_hash;
    Stopwatch sw_trace;
};

#endif // INCLUDE_RUN_TYPES_HPP
