#include <cuda_runtime.h>
#include "common.hpp"


// cuda status, used for getting the return status of cuda functions
thread_local cudaError_t cudaStatus = cudaSuccess;
