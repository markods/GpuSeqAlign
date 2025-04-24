#ifndef INCLUDE_RUN_TYPES_HPP
#define INCLUDE_RUN_TYPES_HPP

#include "dict.hpp"
#include "math.hpp"
#include "memory.hpp"
#include "stopwatch.hpp"
#include <cstdint>
#include <string>
#include <vector>

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

struct NwRange
{
    bool lNotDefault;
    bool rNotDefault;
    int64_t l; // Inclusive.
    int64_t r; // Exclusive.

    friend bool operator==(const NwRange& l, const NwRange& r);
    friend bool operator!=(const NwRange& l, const NwRange& r);
};

struct NwAlgParam
{
    NwAlgParam();
    NwAlgParam(std::vector<int> values);

    int curr() const;
    bool hasCurr() const;
    void next();
    void reset();

    std::vector<int> _values;
    size_t _currIdx;
};

struct NwAlgParams
{
    NwAlgParams();
    NwAlgParams(Dict<std::string, NwAlgParam> params);

    NwAlgParam& at(const std::string name);
    const NwAlgParam& at(const std::string name) const;

    bool hasCurr() const;
    // Updates last parameter, then on iteration loop second-to-last, etc.
    void next();
    void reset();

    Dict<std::string, int> copy() const;

    Dict<std::string, NwAlgParam> _params;
    bool _isEnd;
};

struct NwAlgInput
{
    std::vector<int> subst;
    // Align seqX to seqY (seqX becomes seqY).
    std::vector<int> seqX;
    std::vector<int> seqY;
    HostArray<int> score;
    HostArray<int> tileHrowMat;
    HostArray<int> tileHcolMat;

    std::vector<int> trace;
    std::vector<int> tile;
    std::vector<int> currRow;
    std::vector<int> prevRow;

    DeviceArray<int> subst_gpu;
    DeviceArray<int> seqX_gpu;
    DeviceArray<int> seqY_gpu;
    DeviceArray<int> score_gpu;
    DeviceArray<int> tileHrowMat_gpu;
    DeviceArray<int> tileHcolMat_gpu;

    // Prefer using these instead of vector.size().
    int substsz;
    int adjrows;
    int adjcols;
    int gapoCost;
    int tileHdrMatRows;
    int tileHdrMatCols;
    int tileHrowLen;
    int tileHcolLen;

    int sm_count;
    int warpsz;
    int maxThreadsPerBlock;

    size_t measureHostAllocations() const;
    size_t measureDeviceAllocations() const;
    void resetAllocsBenchmarkCycle();
    void resetAllocsBenchmarkEnd();
};

struct NwAlgResult
{
    std::string algName;
    Dict<std::string, int> algParams;
    int seqY_idx;
    int seqX_idx;
    std::string seqY_id;
    std::string seqX_id;
    NwRange seqY_range;
    NwRange seqX_range;

    int errstep;          // 0 for success.
    NwStat stat;          // 0 for success.
    cudaError_t cudaStat; // 0 for success.

    size_t seqY_len;
    size_t seqX_len;
    std::string substName;
    int gapoCost;
    int warmup_runs;
    int sample_runs;
    int last_run_idx;

    int align_cost;
    unsigned score_hash;
    unsigned trace_hash;

    size_t sm_count;
    size_t ramPeakAllocs;       // In bytes.
    size_t globalMemPeakAllocs; // In bytes.
    size_t sharedMemPeakAllocs; // In bytes.
    size_t localMemPeakAllocs;  // In bytes.
    size_t regMemPeakAllocs;    // In bytes.

    Stopwatch sw_align;
    Stopwatch sw_hash;
    Stopwatch sw_trace;

    std::string edit_trace;
};

#endif // INCLUDE_RUN_TYPES_HPP
