#ifndef INCLUDE_NWALIGN_SHARED_HPP
#define INCLUDE_NWALIGN_SHARED_HPP

#include "run_types.hpp"

void updateNwAlgPeakMemUsage(
    const NwAlgInput& nw,
    NwAlgResult& res,
    const cudaFuncAttributes* const attr = nullptr,
    const size_t maxActiveBlocks = 0,
    const size_t threadsPerBlockRequested = 0,
    const size_t shmemDynamicPerBlock = 0);

#endif // INCLUDE_NWALIGN_SHARED_HPP
