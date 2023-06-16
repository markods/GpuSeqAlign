#include "common.hpp"
#include <cuda_runtime.h>

// cuda status, used for getting the return status of cuda functions
thread_local cudaError_t cudaStatus = cudaSuccess;
