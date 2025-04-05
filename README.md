## GpuSeqAlignment - a benchmark for dynamic-programming-based GPU sequence alignment algorithms
This project compares different CPU and GPU implementations of the Needleman-Wunsch and Smith-Waterman (🔍) algorithms for efficiency and memory consumption.

For the GPU algorithms, the bulk of the work is calculating the dynamic-programming score matrix (not necessarily square). Another concern is its transfer to main memory. Most optimizations apply to the score matrix calculation.

Present algorithms:
| Algorithm                     | NW_LG | NW_AG         | SW_LG | SW_AG         | Description                                                                                                                                                     |
| ----------------------------- | ----- | ------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| cpu1-st-row                   | ✅     | 🔍             | 🔍     | 🔍             | Row-major calculation of score matrix.                                                                                                                          |
| cpu2-st-diag                  | ✅     | 🔍             | 🔍     | 🔍             | Minor diagonal calculation of score matrix.                                                                                                                     |
| cpu3-st-diagrow               | ✅     | 🔍             | 🔍     | 🔍             | Divide matrix into tiles. Tiles are visited in minor diagonal order, inside the tile visit elements in row-major order.                                         |
| cpu4-mt-diagrow               | ✅     | 🔍             | 🔍     | 🔍             | Multi-threaded variant of cpu3-st-diagrow. One thread per rectangular tile. Static visitation schedule for tile diagonal.                                       |
| ---                           | ---   | ---           | ---   | ---           | ---                                                                                                                                                             |
| gpu1-ml-diag                  | ✅     | 🔍             | 🔍     | 🔍             | Launch kernel per each minor diagonal. One thread per element.                                                                                                  |
| gpu2-ml-diagrow2pass          | ✅     | 🔍             | 🔍     | 🔍             | Like gpu1-ml-diag, but one thread per tile. Two-pass, first does neighbour-independent work.                                                                    |
| gpu3-ml-diagdiag              | ✅     | 🔍             | 🔍     | 🔍             | Kernel per each minor tile diagonal. Multiple threads per tile - one per tile row. Threads sync on each minor diagonal in tile.                                 |
| gpu4-ml-diagdiag2pass         | ✅     | 🔍             | 🔍     | 🔍             | Like gpu3-ml-diagdiag, but two-pass like in gpu2-ml-diagrow2pass.                                                                                               |
| gpu5-coop-diagdiag            | ✅     | ❌<sup>1</sup> | 🔍     | ❌<sup>1</sup> | Like gpu3-ml-diagdiag, but use grid sync instead of multi-launching kernels.                                                                                    |
| gpu6-coop-diagdiag2pass       | ✅     | ❌<sup>1</sup> | 🔍     | ❌<sup>1</sup> | Like gpu4-ml-diagdiag2pass, but use grid sync instead of multi-launching kernels.                                                                               |
| ---                           | ---   | ---           | ---   | ---           | ---                                                                                                                                                             |
| gpu7-mlsp-diagdiag            | ✅     | 🔍             | 🔍     | 🔍             | Like gpu3-ml-diagdiag, but represents the score matrix as a tile header row matrix and tile header column. Transfers back only those.                           |
| gpu8-mlsp-diagdiag            | ✅     | 🔍             | 🔍     | 🔍             | Like gpu7-mlsp-diagdiag, but stores the tile completely in registers, instead of in shared memory.                                                              |
| gpu9-mlsp-diagdiagdiagskew    | ✅     | 🔍             | 🔍     | 🔍             | Like gpu8-mlsp-diagdiag, but divides the rectangular tile into parallelogram-shaped subtiles (skewed). Diagonal of subtiles is visited in minor-diagonal order. |
| gpu10-mlsppt-diagdiagdiagskew | 🔍     | 🔍             | 🔍     | 🔍             | Like gpu9-mlsp-diagdiagdiagskew, but does tile diagonal transfer parallel to score matrix calculation.                                                          |

Table terms:
- 🔍 - means that the combination may be implemented in the future.  
- ⚠️ - work-in-progress.
- `NW` - Needleman-Wunsch implementation present.
- `SW` - Smith-Waterman implemented present.
- `LG` - Linear gap penaulty function used.
- `AG` - Affine gap penaulty function used.

Algorithm terms:
- `st` - single-threaded.
- `mt` - multi-threaded, uses OpenMP for concurrency.
- `ml` - multi-launch, the same kernel gets launched multiple times but on different data.
- `coop` - cuda cooperative launch, these algorithms use whole grid synchronization instead of launching multiple kernels.
- `mlsp` - multi-launch sparse - sparse here means keeping only portions of the score matrix in gpu shared memory and main memory. Necessitates modified traceback algorithm.
- `mlsppt` - multi-launch sparse with parallel transfer, where memory transfer back to the host is done in parallel with the calculation.
- ===
- `diag` - minor-diagonal order of calculating matrix elements/tiles.
- `row` - row-major order of calculating matrix elements/tiles.
- `2pass` - uses two kernels, the first partially initializes the matrix doing neighbour-independent work, and the second does the remaning bulk of the work.
- `skew` - skewed tile/subtile, parallelogram.

<sup>1</sup> These algorithms, for maximum Cuda occupancy, require that a maximum of 32 registers be used at once. With the Affine gap penaulty function, this would not be possible.

The algorithms are written in Cuda and C++.

This work is part of my master's thesis (todo link). The idea is to start with the simplest GPU implementation and progressively optimize it, trying out different techniques. Then, pick overall good implementations, and compare to existing tools for exact alignment (todo).

## Assumptions
1. There is a niche where exact alignment is done, where speed and resource consumption is critial.
2. Returning the score matrix calculated on the GPU is necessary (e.g. we would like to trace the edits).
3. Focus on pairwise alignment, and use linear/affine gap penalty functions.

## Prerequisites
The project is run on Windows. The algorithms themselves are platform-agnostic.  
Minimum Cuda supported architecture is `sm6_8`<sup>1</sup>.

1. Install [Visual Studio Community Edition 2022](https://visualstudio.microsoft.com/vs/community/).

2. Install [Cuda for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

## Build
The build is done in Visual Studio Community Edition 2022. Build output is in the `/build` directory.

## Benchmark
Benchmark files are in `.json` format, in the `/resrc` directory. Sequences inside benchmark files are aligned each-with-each without repeating, the specified number of times. The average time for each step is reported. Results are written to the `log` directory.

Run existing benchmarks from the project root:

```shell
# Calibrate optimal algoritm parameters on your system. See the results in the '/logs' directory.
./build/windows-x64-vs-release/nw.exe "./resrc/subst-blosum.json" "./resrc/param-optimize.json" "./resrc/seq-optimize.json"

# Run a quick test to verify all algorithms work on your system.
# Run the sequences used in profiling reports.
./build/windows-x64-vs-release/nw.exe "./resrc/subst-blosum.json" "./resrc/param-best.json" "./resrc/seq-debug.json"
./build/windows-x64-vs-release/nw.exe "./resrc/subst-blosum.json" "./resrc/param-best.json" "./resrc/seq-profile.json"

# Small test - sequences up to 0.1k base pairs.
# Medium test - sequences up to 1k base pairs.
# Large test - sequences up to 10k base pairs.
./build/windows-x64-vs-release/nw.exe "./resrc/subst-blosum.json" "./resrc/param-best.json" "./resrc/seq-gen1.json"
./build/windows-x64-vs-release/nw.exe "./resrc/subst-blosum.json" "./resrc/param-best.json" "./resrc/seq-gen2.json"
./build/windows-x64-vs-release/nw.exe "./resrc/subst-blosum.json" "./resrc/param-best.json" "./resrc/seq-gen3.json"
```

## Results
todo
