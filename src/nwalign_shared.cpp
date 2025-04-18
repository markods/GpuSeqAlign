#include "nwalign_shared.hpp"
#include "math.hpp"
#include <cuda_runtime.h>

void updateNwAlgPeakMemUsage(
    const NwAlgInput& nw,
    NwAlgResult& res,
    const cudaFuncAttributes* const attr /*= nullptr*/,
    const size_t maxActiveBlocks /*= 0*/,
    const size_t threadsPerBlockRequested /*= 0*/,
    const size_t shmemDynamicPerBlock /*= 0*/)
{
    res.ramPeakAllocs = max2(res.ramPeakAllocs, nw.measureHostAllocations());

    if (attr)
    {
        const size_t warpsPerBlock = (threadsPerBlockRequested + (nw.warpsz - 1) /*round up*/) / nw.warpsz;
        const size_t threadsPerBlock = nw.warpsz * warpsPerBlock;

        res.globalMemPeakAllocs = max2(res.globalMemPeakAllocs, nw.measureDeviceAllocations());
        res.sharedMemPeakAllocs = max2(res.sharedMemPeakAllocs, (attr->sharedSizeBytes /*static*/ + shmemDynamicPerBlock /*dynamic*/) * maxActiveBlocks);
        res.localMemPeakAllocs = max2(res.localMemPeakAllocs, attr->localSizeBytes * threadsPerBlock * maxActiveBlocks);
        res.regMemPeakAllocs = max2(res.regMemPeakAllocs, (size_t)attr->numRegs * 4 /*B*/ * threadsPerBlock * maxActiveBlocks);
    }
}
