cmake_minimum_required (VERSION 3.20)

project("GpuSeqAlign" LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# message(STATUS "CUDA include dirs: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


set(CPP_SOURCES
    "src/benchmark.cpp"
    "src/cmd_parser.cpp"
    "src/file_formats.cpp"
    "src/io.cpp"
    "src/main.cpp"
    "src/nw_algorithm.cpp"
    "src/nwalign_cpu1_st_row.cpp"
    "src/nwalign_cpu2_st_diag.cpp"
    "src/nwalign_cpu3_st_diagrow.cpp"
    "src/nwalign_cpu4_mt_diagrow.cpp"
    "src/nwalign_shared.cpp"
    "src/nwtrace1_plain.cpp"
    "src/nwtrace2_sparse.cpp"
    "src/run_types.cpp"
    "src/stopwatch.cpp"
)

set(CUDA_SOURCES
    "src/nwalign_gpu1_ml_diag.cu"
    "src/nwalign_gpu2_ml_diagrow2pass.cu"
    "src/nwalign_gpu3_ml_diagdiag.cu"
    "src/nwalign_gpu4_ml_diagdiag2pass.cu"
    "src/nwalign_gpu5_coop_diagdiag.cu"
    "src/nwalign_gpu6_coop_diagdiag2pass.cu"
    "src/nwalign_gpu7_mlsp_diagdiag.cu"
    "src/nwalign_gpu8_mlsp_diagdiag.cu"
    "src/nwalign_gpu9_mlsp_diagdiagdiag.cu"
)

add_executable(nw ${CPP_SOURCES} ${CUDA_SOURCES})

target_include_directories(nw PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    "src"
)

set_target_properties(nw PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 86   # min supported is 68
    # CUDA_SEPARABLE_COMPILATION ON
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON  # ensures correct linking of device symbols

    # Workaround for mismatched C runtime libraries, otherwise:
    # 1. we link with msvcrt.dll (dynamic), but also link with
    # 2. cudart_static.lib, which uses LIBCMT (static) and is included through
    #    <AdditionalLibraryDirectories>...NVIDIA GPU Computing Toolkit/CUDA/<VERSION>/lib/x64...</AdditionalLibraryDirectories>
    CUDA_RUNTIME_LIBRARY "Shared"
)

# Enable Link-Time optimization.
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

if(ipo_supported)
    foreach(config RELEASE RELWITHDEBINFO)
        set_property(TARGET nw PROPERTY INTERPROCEDURAL_OPTIMIZATION_${config} TRUE)
    endforeach()
else()
    message(WARNING "IPO/LTO is not supported: ${ipo_error}")
endif()


target_compile_options(nw PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:
            /openmp
            /MP   # enable parallel compilation
            -W4   # not the highest level - cuda 12.8 libraries make a lot of warnings
            /Oy-   # keep frame pointers
            /permissive-   # conformance mode
        >
        $<$<CXX_COMPILER_ID:GNU>:
            -fopenmp
            -Wall -Wextra -Wpedantic
            -fno-omit-frame-pointer
        >
        $<$<CXX_COMPILER_ID:Clang>:
            -fopenmp
            -Wall -Wextra -Wpedantic
            -fno-omit-frame-pointer
        >
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug,RelWithDebInfo>:-G>   # kernel debug symbols
        -maxrregcount 32
        --ptxas-options=-warn-spills
    >
)

target_link_libraries(nw PRIVATE
    ${OpenMP_CXX_LIB_NAMES}
)
