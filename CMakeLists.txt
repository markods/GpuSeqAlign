cmake_minimum_required (VERSION 3.20)

project("GpuSeqAlign" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


set(CPP_SOURCES
    "src/common.cpp"
    "src/main.cpp"
    "src/nw-algorithm.cpp"
    "src/nw-cpu1-st-row.cpp"
    "src/nw-cpu2-st-diag.cpp"
    "src/nw-cpu3-st-diagrow.cpp"
    "src/nw-cpu4-mt-diagrow.cpp"
    "src/nw-trace1-plain.cpp"
    "src/nw-trace2-sparse.cpp"
)

set(CUDA_SOURCES
    "src/nw-gpu1-ml-diag.cu"
    "src/nw-gpu2-ml-diagrow2pass.cu"
    "src/nw-gpu3-ml-diagdiag.cu"
    "src/nw-gpu4-ml-diagdiag2pass.cu"
    "src/nw-gpu5-coop-diagdiag.cu"
    "src/nw-gpu6-coop-diagdiag2pass.cu"
    "src/nw-gpu7-mlsp-diagdiag.cu"
    "src/nw-gpu8-mlsp-diagdiag.cu"
    "src/nw-gpu9-mlsp-diagdiagdiag.cu"
)

set(HEADERS
    "src/common.hpp"
    "src/json.hpp"
    "src/nw-algorithm.hpp"
)

add_executable(nw ${CPP_SOURCES} ${CUDA_SOURCES})

target_include_directories(nw PRIVATE src)

set_target_properties(nw PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
    CUDA_MAX_REGISTERS 32
)

target_compile_options(nw PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-openmp -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-warn-spills>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:RelWithDebInfo>:-O2 -pg>
)

target_link_options(nw PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:RelWithDebInfo>:-O2 -pg>
)
