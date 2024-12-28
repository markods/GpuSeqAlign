// #include "common.hpp"

// // get the number of potential chunks on the tile
// // +   take into account that the chunk is diagonally shaped
// __device__ static int num_of_chunks_in_tile(int tile_wid, int chunk_wid, int chunk_hei)
// {
//     return (tile_wid + (chunk_hei - 1) + /*homemade ceil function*/ chunk_wid - 1) / chunk_wid;
// }

// // cuda kernel for the parallel implementation
// __global__ static void Nw_Gpu10_Kernel(
//     // nw input
//     const int *const seqX_gpu,
//     const int *const seqY_gpu,
//     const int *const hrow_gpu, // header row for the score matrix
//     const int *const hcol_gpu, // header column for the score matrix
//     const int *const subst_gpu,
//     // const int adjrows,   // can be calculated as 1 + trows*tileAy
//     // const int adjcols,   // can be calculated as 1 + tcols*tileAx
//     const int substsz,
//     const int indel,
//     const int warpsz,
//     // tile size and miscellaneous
//     int *const hrowTDi_gpu, // input header_row tile_diagonal, consisting of the header rows for all tiles on the current tile diagonal
//     int *const hcolTDi_gpu, // input header_col tile_diagonal, consisting of the header columns for all tiles on the current tile diagonal
//     int *const hrowTDo_gpu, // output header_row tile_diagonal, consisting of the header rows for all tiles on the current tile diagonal
//     int *const hcolTDo_gpu, // output header_col tile_diagonal, consisting of the header columns for all tiles on the current tile diagonal
//     const int trows,
//     const int tcols,
//     const unsigned tileAx,
//     const unsigned tileAy,
//     const int chunksz, // the number of elements the last thread in a warp must calculate before the next block sync
//     const int s        // the current minor tile diagonal in the score matrix (exclude the header row and column)
//                        // +   note: s >= 0 are normal tile diagonals, s = -1 when we want to initialize the header_row and header_col for the zeroth tile diagonal
// )
// {
//     extern __shared__ int shmem[/* substsz*substsz + tileAx + tileAy + (1+tileAx) + (1+tileAy) */];
//     // the substitution matrix and relevant parts of the two sequences
//     // TODO: align allocations to 0-th shared memory bank?
//     int *const subst /*[substsz*substsz]*/ = shmem + (0);
//     int *const seqX /*[tileAx]*/ = subst + (substsz * substsz);
//     int *const seqY /*[tileAy]*/ = seqX + (tileAx);
//     // the header row and column for the tile; they will become the header row for the below tile, and the header column for the right tile
//     int *const hrow /*[1+tileAx]*/ = seqY + (tileAy);
//     int *const hcol /*[1+tileAy]*/ = hrow + (1 + tileAx);

//     // initialize the substitution shared memory copy
//     if (s >= 0)
//     {
//         // map the threads from the thread block onto the substitution matrix elements
//         int i = threadIdx.x;
//         // while the current thread maps onto an element in the matrix
//         while (i < substsz * substsz)
//         {
//             // copy the current element from the global substitution matrix
//             el(subst, substsz, 0, i) = el(subst_gpu, substsz, 0, i);
//             // map this thread to the next element with stride equal to the number of threads in this block
//             i += blockDim.x;
//         }
//     }

//     // initialize the tile's window into the global X and Y sequences
//     if (s >= 0)
//     {
//         //  / / / . .       . . . / /       . . . . .|/ /
//         //  / / . . .   +   . . / / .   +   . . . . /|/
//         //  / . . . .       . / / . .       . . . / /|

//         // (s,t) -- tile coordinates on the score matrix diagonal
//         int tbeg = max(0, s - (tcols - 1));
//         // int tend = min( s + 1, trows );
//         // position of the current thread's tile on the score matrix diagonal
//         int t = tbeg + blockIdx.x;

//         // unnecessary question, since the number of launched blocks will always be the same as the number of tiles on the diagonal
//         // if( t < tend )

//         //       x x x x x
//         //       | | | | |
//         //     h h h h h h     // note the x and y seqences on this schematic
//         // y --h c . . . .     // +   they don't! need to be extended by 1 to the left and by 1 to the top
//         // y --h . . . . .
//         // y --h . . . . .
//         // position of the top left not-yet-calculated! element <c> of the current tile in the score matrix
//         // +   only the not-yet-calculated elements will be calculated, and they need the corresponding global sequence X and Y elements
//         int ibeg = 1 + (t)*tileAy;
//         int jbeg = 1 + (s - t) * tileAx;

//         // map the threads from the thread block onto the global X sequence's elements (which will be used in this tile)
//         int j = threadIdx.x;
//         // while the current thread maps onto an element in the tile's X sequence
//         while (j < tileAx)
//         {
//             // initialize that element in the X seqence's shared window
//             seqX[j] = seqX_gpu[jbeg + j];

//             // map this thread to the next element with stride equal to the number of threads in this block
//             j += blockDim.x;
//         }

//         // map the threads from the thread block onto the global Y sequence's elements (which will be used in this tile)
//         int i = threadIdx.x;
//         // while the current thread maps onto an element in the tile's Y sequence
//         while (i < tileAy)
//         {
//             // initialize that element in the Y seqence's shared window
//             seqY[i] = seqY_gpu[ibeg + i];

//             // map this thread to the next element with stride equal to the number of threads in this block
//             i += blockDim.x;
//         }
//     }

//     //   MOVE DOWN             d=0                              d=1                              d=2                              d=3                         d=4, end
//     //   ~   ~ ~ a   . . .   . . .        .   . . ~   ~ ~ b   . . .        .   . . .   . . ~   ~ ~ c        .   . . .   . . .   . . .        .   . . .   . . .   . . .
//     //
//     //   .   1 1 1   . . .   . . .        .   . . .   1 1 1   . . .        .   . . .   . . .   1 1 1        .   . . .   . . .   . . .        .   . . .   . . .   . . .
//     //   .   1 1 1   . . .   . . .        _   _ _ a   1 1 1   . . .        .   . . _   _ _ b   1 1 1        .   . . .   . . _   _ _ _        .   . . .   . . .   . . .
//     //
//     //   .   . . .   . . .   . . .        .   2 2 2   . . .   . . .        .   . . .   2 2 2   . . .        .   . . .   . . .   1 1 1        .   . . .   . . .   . . .
//     //   .   . . .   . . .   . . .        .   2 2 2   . . .   . . .        _   _ _ a   2 2 2   . . .        .   . . _   _ _ _   1 1 1        .   . . .   . . _   _ _ _
//     //
//     //                   ||  a ...                     ||  b  a ...                  ||  c  b  a ...                     ||  c  b ...                        ||  c ...   // add before start, remove from end
//     //                       0                             0  1                          0  1  2                             1  2                                2
//     //                               ->                               ->                               ->                               ->
//     //   MOVE RIGHT
//     //   ~   . . .   . . .   . . .        .   . . |   . . .   . . .        .   . . .   . . |   . . .        .   . . .   . . .   . . |        .   . . .   . . .   . . .
//     //
//     //   ~   1 1 1   . . .   . . .        .   . . |   1 1 1   . . .        .   . . .   . . |   1 1 1        .   . . .   . . .   . . |        .   . . .   . . .   . . .
//     //   a   1 1 1   . . .   . . .        ~   . . a   1 1 1   . . .        .   . . |   . . a   1 1 1        .   . . .   . . |   . . a        .   . . .   . . .   . . |
//     //
//     //   .   . . .   . . .   . . .        ~   2 2 2   . . .   . . .        .   . . |   2 2 2   . . .        .   . . .   . . |   1 1 1        .   . . .   . . .   . . |
//     //   .   . . .   . . .   . . .        b   2 2 2   . . .   . . .        .   . . b   2 2 2   . . .        .   . . .   . . b   1 1 1        .   . . .   . . .   . . b
//     //
//     //                   ||  a ...                     ||  a  b ...                     ||  a  b ...                     ||  a  b ...                        ||  b ...   // add after end, remove from start
//     //                       0                             1  0                             2  1                             3  2                                3

//     // read the current tile's header_row and header_col from the input <header_row, header_col> tile_diagonal
//     // +   use the previous tile_diagonal's results unmodified!
//     if (s >= 0)
//     {
//         // s=0   1     2       3      4
//         // ┌─────╔═════┌─────  ╔═════┌─────    5
//         // │     ║  A  │       ║  A  │       ║ <-- x
//         //  ╔════╝┌────┘  ╔════╝┌────┘  ╔════╝
//         //  ║  B  │       ║  B  │       ║  A  │ 6
//         //   ┌────┘  ╔════╝┌────┘  ╔════╝┌────┘
//         //   │       ║  C  │       ║  B  │     │
//         //      ═════╝─────┘  ═════╝─────┘─────┘
//         //                    ^-- y
//         //
//         // reading the current diagonal (double-lined)

//         // the index of the tile's header_row and header_col in the current tile_diagonal
//         // +   skip reading the <zeroth header_col element> (marked with x) in the <lower right triangle>, since it is not a part of the zeroth tile (tile A)
//         // +   skip reading the <last header_row element> in the <middle parallelogram and bottom right triange> automatically,
//         //     since there is no thread block to try to read it (no thread block C on that diagonal)
//         int t_hr = blockIdx.x;
//         int t_hc = blockIdx.x + (s >= tcols);
//         // the starting location of the header_row in the header_row tile_diagonal
//         // the starting location of the header_col in the header_col tile_diagonal
//         int jbeg = (1 + tileAx) * t_hr + 0;
//         int ibeg = (1 + tileAy) * t_hc + 0;

//         // map the threads from the thread block onto the header_row in the header_row tile_diagonal
//         int j = threadIdx.x;

//         // while the current thread maps onto an element in the corresponding header_row
//         while (j < 1 + tileAx)
//         {
//             // copy the current element from the corresponding input header_row
//             hrow[j] = hrowTDi_gpu[jbeg + j];
//             // map this thread to the next element with stride equal to the number of threads in this block
//             j += blockDim.x;
//         }

//         // map the threads from the thread block onto the header_col in the header_col tile_diagonal
//         int i = threadIdx.x;

//         // while the current thread maps onto an element in the corresponding header_col
//         while (i < 1 + tileAy)
//         {
//             // copy the current element from the corresponding input header_col
//             hcol[i] = hcolTDi_gpu[ibeg + i];
//             // map this thread to the next element with stride equal to the number of threads in this block
//             i += blockDim.x;
//         }
//     }

//     if (s >= 0)
//     {
//         // all threads in this block should finish initializing their substitution shared memory, corresponding parts of the global X and Y sequences, and the tile's header_row and header_col
//         __syncthreads();
//     }

//     // map a tile on the current diagonal of tiles to this thread block
//     if (s >= 0)
//     {
//         // INITIAL IDEA, BUT NOT COMPLETELY CORRECT
//         //       chunksz           chunksz           chunksz           chunksz
//         //     x                     x x x x x       x x x x x x       x x x x x x
//         //  y
//         //       . . . . . .    y  . . / / 1 .       . . . . . .       . . . . . .     warpsz
//         //       . . . . . .    y  . / / 1 . .       . . . . . .       . . . . . .
//         //       . . . . . .    y  / / 1 . . .       . . . . . .       . . . . . .
//         //       . . . . . .    y  / 1 . . . .       . . . . . .       . . . . . .
//         //                              ^----
//         //             x x x       x   <----- issue
//         //       . . . . . /    y  / 2 . . . .       . . . . . .       . . . . . .     warpsz
//         //  y    . . . . / /    y  2 . . . . .       . . . . . .       . . . . . .
//         //  y    . . . / / 2       . . . . . .       . . . . . .       . . . . . .
//         //  y    . . . / 2 .       . . . . . .       . . . . . .       . . . . . .
//         //
//         //       x x x x
//         //  y   3. . . . . .       . . . . . .       . . . . . .       . . . . . .     warpsz
//         //  y   3. . . . . .       . . . . . .       . . . . . .       . . . . . .
//         //  y   3. . . . . .       . . . . . .       . . . . . .       . . . . . .
//         //  y   3. . . . . .       . . . . . .       . . . . . .       . . . . . .

//         // REFINED IDEA, CORRECT
//         //      ____________________________________________        ____________________________________________        _______________________________________________
//         //     / . . . / . . / . . . . . . . . . . . . . . .       / . . . . . . / . . . / . . / . . . . . . . .       /. . . . . . . . / . . . . . . / . . . / . . / .
//         //   / |1A . / .1B / . . . . . . . . . . . . . . . .     / | . . . . . / .2A . / .2B / . . . . . . . . .     / |. . . . . . . / . . . . . . / .3A . / .3B / . .   ... after all warps except the last one are initialized,
//         // / _ | _ / _ _ / . . . . . . . . . . . . . . . . .   / _ | _ / _ _ / _ _ _ / _ _ / . . . . . . . . . .   / _ |_ _ _ / _ _ / _ _ _ / _ _ / _ _ _ / _ _ / . . .       only do chunk C (c elements per thread, instead of c+w in two chunks, A and B)
//         //     | . . . . . . . . . . . . . . . . . . . . . .       / . . . / . . / . . . . . . . . . . . . . . .       /. . . . . . . . / . . . / . . / . . . . . . . .
//         //     | . c+w -w. . . . . . . . . . . . . . . . . .     / |2A . / .2B / . . . c . . . . . . . . . . . .     / |. . . . . . . / .3A . / .3B / . . . c . . . . .       c - chunksz
//         //     | . . . . . . . . . . . . . . . . . . . . . .   / _ | _ / _ _ / . . . . . . . . . . . . . . . . .   / _ |_ _ _ / _ _ / _ _ _ / _ _ / . . . . . . . . . .       w - warpsz
//         //     | . . . . . . . . . . . . . . . . . . . . . .       | . . . . . . . . . . . . . . . . . . . . . .       /. . . . . / . . / . . . . . . . . . . . . . . .
//         //     | . . . . . . . . . . . . . . . . . . . . . .       | . c+w -w. . . . . . . . . . . . . . . . . .     / |. .3A . / .3B / . . . c . . . . . . . . . . . .       note: chunk A has to be done before chunk B, since e.g. 3B would require 3A from above to be finished
//         //     | . . . . . . . . . . . . . . . . . . . . . .       | . . . . . . . . . . . . . . . . . . . . . .   / _ |_ _ _ / _ _ / . . . . . . . . . . . . . . . . .       note: a diagonal consists of either chunks A+B or only chunk C later on, therefore the width of the diagonal is not fixed
//         //     | . . . . . . . . . . . . . . . . . . . . . .       | . . . . . . . . . . . . . . . . . . . . . .       |. . . . . . . . . . . . . . . . . . . . . . . .
//         //     | . . . . . . . . . . . . . . . . . . . . . .       | . . . . . . . . . . . . . . . . . . . . . .       |. . . c+w -w. . . . . . . . . . . . . . . . . .       important: c >= w for the algorithm to work (chunk A before chunk B),
//         //     | . . . . . . . . . . . . . . . . . . . . . .       | . . . . . . . . . . . . . . . . . . . . . .       |. . . . . . . . . . . . . . . . . . . . . . . .                  if c < w swap their values

//         // the thread's warp and index inside the warp, as well as the number of warps in the tile
//         const int warpIdx = threadIdx.x / warpsz;
//         const int warpThreadIdx = threadIdx.x % warpsz;
//         const int warpcnt = tileAy / warpsz;

//         // size of chunk per chunk type
//         // +   A has to be bigger than B, to avoid a race condition after A and before B
//         // +   C is equal to the requested chunk size
//         const int chunkAsz = max(chunksz, warpcnt);
//         const int chunkBsz = min(chunksz, warpcnt);
//         const int chunkABsz = chunksz + warpcnt;
//         const int chunkCsz = chunksz;

//         // idea: think of the matrix as only composed of chunk C
//         // +   calculate the number of chunk diagonals for that matrix
//         // +   now widen some chunks C in the upper left triangle, and then add "artificial" sync tiles (___) after the matrix, to keep the number of chunk diagonals constant
//         //
//         // case 1.                       AB                                   C                                   S2
//         //                              |aaabb    aaabb    aaabb              ccc      ccc      |ccc              ___      ___      ___      ___     ___
//         //                             a|aabb    aaabb    aaabb              ccc      ccc      c|cc              ___      ___      ___      ___     ___
//         //                           /         /        /                 /        /        /                 /        /        /        /        /
//         //                      __ _    |aaabb    aaabb              ccc      ccc      ccc               cc|c     ___      ___      ___      ___
//         //         S0          __ _    a|aabb    aaabb              ccc      ccc      ccc               ccc|     ___      ___      ___      ___
//         //                  /        /         /                 /        /        /                 /        /        /        /        /
//         //             __ _     __ _    |aaabb              ccc      ccc      ccc               ccc      ccc      c|cc     ___      ___
//         //            __ _     __ _    a|aabb              ccc      ccc      ccc               ccc      ccc      cc|c     ___      ___
//         //         /        /        /                  /        /        /                /        /        /        /        /
//         //    __ _     __ _     __ _              |ccc      ccc      ccc              ccc      ccc      ccc      ccc      |ccc    = 11 diagonals = (warpcnt-1) + ceil( float( tileAx + (warpsz-1) )/chunkCsz )
//         //   __ _     __ _     __ _              c|cc      ccc      ccc              ccc      ccc      ccc      ccc      c|cc                                                          [1]
//         //                                    ==[1]                                                                     [1]
//         //
//         // case 2.                       AB                                    S2
//         //                              |aaabb    aaabb    |aaabb              ___      ___      ___      ___
//         //                             a|aabb    aaabb    a|aabb              ___      ___      ___      ___
//         //                           /         /        /            C     /        /        /        /
//         //                      __ _    |aaabb    aaabb              |ccc      ___      ___      ___
//         //         S0          __ _    a|aabb    aaabb              c|cc      ___      ___      ___
//         //                  /        /         /                 /        /        /        /
//         //             __ _     __ _    |aaabb              ccc      cc|c      ___      ___
//         //            __ _     __ _    a|aabb              ccc      ccc|     ___      ___
//         //         /        /        /                  /        /        /        /
//         //    __ _     __ _     __ _              |ccc      ccc      ccc      ccc            = 7 diagonals = (warpcnt-1) + ceil( float( tileAx + (warpsz-1) )/chunkCsz )
//         //   __ _     __ _     __ _              c|cc      ccc      ccc      ccc
//         //
//         //
//         // case 3.                       AB        S1                          S2
//         //                              |aaab|b    __ _     __ _               ___      ___
//         //                             a|aabb|    __ _     __ _               ___      ___
//         //                           /         /        /                  /        /
//         //                      __ _    |aaab|b    __ _              ___       ___
//         //         S0          __ _    a|aabb|    __ _              ___       ___
//         //                  /        /         /                 /        /
//         //             __ _     __ _    |aaab|b              ccc     ___
//         //            __ _     __ _    a|aabb|              ccc     ___
//         //         /        /        /                  /        /
//         //    __ _     __ _     __ _              |ccc      ccc         = 5 diagonals = (warpcnt-1) + ceil( float( tileAx + (warpsz-1) )/chunkCsz )
//         //   __ _     __ _     __ _              c|cc      ccc          NOTE: the "artificial" part of the upper left triangle (__ _ in S1) has to sync twice! (to prevent a deadlock)
//         //

//         // the total number of chunk diagonals
//         const int chunks = (warpcnt - 1) + num_of_chunks_in_tile(tileAx, chunkCsz, warpsz);
//         // remaining number of chunks per chunk type, for this! warp
//         // +   note: sync twice in the S0 and S1 chunk, since we'll sync twice in the AB chunk
//         int chunksS0 = 2 * warpIdx; // number of artificial chunks before the AB chunks
//         int chunksAB = min(
//             (warpcnt - 1 - warpIdx),                         // expected number of AB chunks
//             num_of_chunks_in_tile(tileAx, chunkABsz, warpsz) // maximum possible number of AB chunks on the tile horizontal
//         );
//         int chunksS1 = 2 * ((warpcnt - 1 - warpIdx) - chunksAB); // expected number of artificial S1 chunks between AB and C chunks
//         int chunksC = max(
//             num_of_chunks_in_tile(tileAx - chunksAB * chunkABsz, chunkCsz, warpsz), // expected number of C chunks
//             0                                                                       // minimum number of C chunks
//         );
//         int chunksS2 = chunks - (chunksS0 + chunksAB + chunksS1 + chunksC); // number of artificial chunks after C chunks

//         bool inChunkAB = false; // if we are in the superchunk AB (between chunks A and B)

//         //               |h  h  h  h  h  h  h  h  h |hx hx|h  h  h  h  h  h  h |  .  .  .  .  .
//         //             . |h  .  .  .  .  .  .  .  x |x  o |.  .  .  .  .  .  . |  .  .  .  .
//         //          .  . |h  .  .  .  .  .  . |ul u| o  .  .  .  .  .  .  .  . |  .  .  .
//         //       .  .  . |h  .  .  .  .  .  x |l  c| .  .  .  .  .  .  .  .  . |  .  .
//         //    .  .  .  . |h  .  .  .  .  x  x  o  .  .  .  .  .  .  .  .  .  . |  .
//         // .  .  .  .  . |h  .  .  .  .  x  o  .  .  .  .  .  .  .  .  .  .  . |

//         // the elements needed in order to slide the calculation window to the right each iteration
//         // +   save the left element before any calculation (to simplify the algorithm; the thread now only needs to worry how it's going to get its 'up' element)
//         int up = 0; // <thread 0 in the warp>: read from hrow on each iteration
//         int upleft = hcol[1 + (threadIdx.x - 1)];
//         int left = hcol[1 + (threadIdx.x)];
//         int curr = 0; // stores the value of the newly calculated element

//         // slide the tile once in the reverse direction (to the left), since the below algorithm first <slides the tile to the right> and then! <calculates the current element>
//         up = upleft;
//         upleft = 0;
//         curr = left;
//         left = 0;

//         // the current thread's position in the tile (also the position of the 'curr' element of the calculation window)
//         // +    on the x axis, offset the thread by its "artificial" elements in the zeroth chunk
//         const int i = threadIdx.x;
//         int j = -warpThreadIdx;
//         // the current chunk's end position in the tile
//         int jcend = j;

//         ////// calculation //////
//         while (true)
//         {
//             if (chunksS0 > 0)
//             {
//                 chunksS0--;
//                 // synchronize twice for each AB chunk on this chunk diagonal
//                 // +   wait until the header_row for the warp is ready
//             }
//             else if (chunksAB > 0 && !inChunkAB)
//             {
//                 // calculate the current chunk A
//                 jcend += chunkAsz;

//                 inChunkAB = true;
//             }
//             else if (chunksAB > 0 && inChunkAB)
//             {
//                 // calculate the current chunk B
//                 jcend += chunkBsz;

//                 inChunkAB = false;
//                 chunksAB--;
//             }
//             else if (chunksS1 > 0)
//             {
//                 chunksS1--;
//                 // synchronize twice for each AB chunk on this chunk diagonal
//             }
//             else if (chunksC > 0)
//             {
//                 // calculate the current chunk
//                 jcend += chunkCsz;

//                 chunksC--;
//             }
//             else if (chunksS2 > 0)
//             {
//                 chunksS2--;
//                 // synchronize once for each C chunk on this chunk diagonal
//             }
//             // finish the calculation
//             else
//             {
//                 break;
//             }

//             // while the warp is in the current chunk (if at all)
//             while (j < jcend)
//             {
//                 //               |h  h  h  h  h  h  h  h  h |hx hx|h  h  h  h  h  h  h |  .  .  .  .  .
//                 //             . |h  .  .  .  .  .  .  .  x |x  o |.  .  .  .  .  .  . |  .  .  .  .
//                 //          .  . |h  .  .  .  .  .  . |ul u| o  .  .  .  .  .  .  .  . |  .  .  .
//                 //       .  .  . |h  .  .  .  .  .  x |l  c| .  .  .  .  .  .  .  .  . |  .  .
//                 //    .  .  .  . |h  .  .  .  .  x  x  o  .  .  .  .  .  .  .  .  .  . |  .
//                 // .  .  .  .  . |h  .  .  .  .  x  o  .  .  .  .  .  .  .  .  .  .  . |

//                 // slide the calculation window to the right
//                 // +   also synchronize the warp minor diagonal as many times as necessary (even "artificially")
//                 // +   the 'upleft', 'left' and 'up' elements will be initialized at the thread's start of the AB chunk
//                 {
//                     // if the current element is not "artificial", slide the tile to the right
//                     // +   this prevents losing the left header element before the first calculation
//                     if (j >= 0 && j < tileAx)
//                     {
//                         upleft = up;
//                         left = curr;
//                     }

//                     // copy from a lane with lower id relative to the caller; also sync the threads in the warp
//                     // +   IMPORTANT: always! sync with the warp, even for "artificial" elements
//                     // +   NOTE: 'curr' is unused after this step (meaning, its value is ready to be calculated below)
//                     up = __shfl_up_sync(0xffff /*mask*/, curr /*var*/, 1 /*delta,*/ /*width=warpSize*/);

//                     // initialize the <zeroth thread in a warp>'s 'up' element with the next 'up' element from the header_row
//                     // +   for "artificial" elements, initialize to 0 so that possible errors are deterministic
//                     if (warpThreadIdx == 0)
//                     {
//                         up = (j >= 0 && j < tileAx) ? hrow[1 + (j)] : 0;
//                     }
//                 }

//                 // calculate the current element's value and save the results to the header_row and header_col
//                 // +   only if the current element is not "artificial"
//                 if (j >= 0 && j < tileAx)
//                 {
//                     // calculate the current element's value
//                     curr = upleft + el(subst, substsz, seqY[i], seqX[j]); // MOVE DOWN-RIGHT
//                     curr = max(curr, up + indel);                         // MOVE DOWN
//                     curr = max(curr, left + indel);                       // MOVE RIGHT

//                     // if this is the last thread in the warp,
//                     // always save its current element to the header_row
//                     if (warpThreadIdx == warpsz - 1)
//                     {
//                         hrow[1 + (j)] = curr;
//                     }

//                     // if this is the last element in the thread's row,
//                     // save that element to the header_col
//                     if (j == tileAx - 1)
//                     {
//                         hcol[1 + (i)] = curr;

//                         // last thread in block: save the zeroth elements in the header_row and header_col
//                         // +   these elements were not updated during the calculation, because none of the threads map to them (they are in the upper left header corner)
//                         if (threadIdx.x == blockDim.x - 1)
//                         {
//                             hrow[0] = hcol[1 + (tileAy - 1)];
//                             hcol[0] = hrow[1 + (tileAx - 1)];
//                         }
//                     }
//                 }

//                 // move to the next element in the thread's row
//                 j++;
//             }

//             // synchronize with the thread block
//             __syncthreads();
//         }
//     }

//     // no block synchronization necessary here, since it's also done after the last chunk in the calculation
//     // if( s >= 0 )
//     // {
//     //    __syncthreads();
//     // }

//     // store the results to the next tile diagonal
//     if (s >= 0 || s == -1)
//     {
//         // s=-1   0    1       2      3    .-- x
//         // ┌─────┌─────┌─────  ┌─────┌─────
//         // │     │  A  ║  .    │  A  ║  .    │ 4
//         //  ┌────┘╔════╝  ┌────┘╔════╝  ┌────┘
//         //  │  B  ║  .    │  B  ║  .    │  A  ║ 5
//         //   ═════╝  ┌────┘╔════╝  ┌────┘╔════╝
//         // > │  .    │  C  ║  .    │  B  ║  .  │ 6
//         // |    ─────┘═════╝  ─────┘═════╝─────┘
//         // | y
//         //
//         // writing the next diagonal (now double-lined)
//         // +   the double-lined header_rows ═══ have moved down  .  by one tile,
//         //     the double-lined header_cols ║   have moved right -> by one tile

//         // if this is not a special diagonal (not the -1st)
//         if (s >= 0)
//         {
//             // the index of the tile's header_row and header_col in the next tile_diagonal
//             // +   leave space for the right tile's header_row at the beginning [────, ═══, ═══, ═══), but only inside the <upper left triangle + middle parallelogram>
//             // +   leave space for the below tile's header_col at the end [║, ║, │) automatically, but only inside the <upper left triangle>
//             int t_hr = blockIdx.x + (s + 1 < tcols);
//             int t_hc = blockIdx.x;
//             // the starting location of the header_row in the header_row tile_diagonal
//             // the starting location of the header_col in the header_col tile_diagonal
//             int jbeg = (1 + tileAx) * t_hr + 0;
//             int ibeg = (1 + tileAy) * t_hc + 0;

//             // map the threads from the thread block onto the header_row in the header_row tile_diagonal
//             int j = threadIdx.x;
//             // while the current thread maps onto an element in the corresponding header_row
//             while (j < (1 + tileAx))
//             {
//                 // copy the current element from the corresponding output header_row
//                 hrowTDo_gpu[jbeg + j] = hrow[j];
//                 // map this thread to the next element with stride equal to the number of threads in this block
//                 j += blockDim.x;
//             }

//             // map the threads from the thread block onto the header_col in the header_col tile_diagonal
//             int i = threadIdx.x;
//             // while the current thread maps onto an element in the corresponding header_col
//             while (i < (1 + tileAy))
//             {
//                 // copy the current element from the corresponding output header_col
//                 hcolTDo_gpu[ibeg + i] = hcol[i];
//                 // map this thread to the next element with stride equal to the number of threads in this block
//                 i += blockDim.x;
//             }
//         }

//         // if this is the zeroth block in the grid and the tile_diagonal touches the top of the matrix
//         // +   initialize the header_row for the right tile (this also supports the special -1st diagonal)
//         if (blockIdx.x == 0 && (s + 1 < tcols))
//         {
//             // the starting location of the header_row in the header_row tile_diagonal
//             // the starting location of the header_row in the matrix's header row
//             int jbegTD = (1 + tileAx) * blockIdx.x + 0;
//             int jbegM = (1 + (tileAx) * (s + 1)) - 1;

//             // map the threads from the thread block onto the header_row in the header_row tile_diagonal
//             int j = threadIdx.x;
//             // while the current thread maps onto an element in the corresponding header_row in the matrix header_row
//             while (j < (1 + tileAx))
//             {
//                 // copy the current element from the corresponding matrix header_row
//                 hrowTDo_gpu[jbegTD + j] = hrow_gpu[jbegM + j];
//                 // map this thread to the next element with stride equal to the number of threads in this block
//                 j += blockDim.x;
//             }
//         }
//         // if this is the last block in the grid and the tile_diagonal touches the left of the matrix
//         // +   initialize the header_col for the below tile (this also supports the special -1st diagonal)
//         if (blockIdx.x == gridDim.x - 1 && (s + 1 < trows))
//         {
//             // the starting location of the header_col in the header_col tile_diagonal
//             // the starting location of the header_col in the matrix's header column
//             int ibegTD = (1 + tileAy) * blockIdx.x + 0;
//             int ibegM = (1 + (tileAy) * (s + 1)) - 1;

//             // map the threads from the thread block onto the header_col in the header_col tile_diagonal
//             int i = threadIdx.x;
//             // while the current thread maps onto an element in the corresponding header_col in the matrix header_col
//             while (i < (1 + tileAy))
//             {
//                 // copy the current element from the corresponding matrix header_col
//                 hcolTDo_gpu[ibegTD + i] = hcol_gpu[ibegM + i];
//                 // map this thread to the next element with stride equal to the number of threads in this block
//                 i += blockDim.x;
//             }
//         }
//     }
// }

// // parallel gpu implementation of the Needleman-Wunsch algorithm
// NwStat NwAlign_Gpu10_Mlsp_DiagDiagDiagSkew2(NwParams &pr, NwInput &nw, NwResult &res)
// {
//     // TODO
//     return NwStat::errorInvalidValue;

//     // // tile size for the kernel
//     // unsigned tileAx;
//     // unsigned tileAy; // TODO: must be a multiple of the warp size
//     // // horizontal size of the chunk in the tile
//     // int chunksz;
//     // int warpcnt;

//     // // get the parameter values
//     // try
//     // {
//     //     tileAx = pr["tileAx"].curr();
//     //     tileAy = pr["tileAy"].curr();
//     //     chunksz = pr["chunksz"].curr();
//     //     warpcnt = tileAy / nw.warpsz;
//     // }
//     // catch (const std::out_of_range &ex)
//     // {
//     //     return NwStat::errorInvalidValue;
//     // }

//     // // adjusted gpu score matrix dimensions
//     // // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile A size (in order to be evenly divisible)
//     // int adjrows = 1 + tileAy * ceil(float(nw.adjrows - 1) / tileAy);
//     // int adjcols = 1 + tileAx * ceil(float(nw.adjcols - 1) / tileAx);
//     // // special case when one of the sequences is very short
//     // if (adjrows == 1)
//     // {
//     //     adjrows = 1 + tileAy;
//     // }
//     // if (adjcols == 1)
//     // {
//     //     adjcols = 1 + tileAx;
//     // }

//     // // start the timer
//     // Stopwatch &sw = res.sw_align;
//     // sw.start();

//     // // s=0   1     2       3      4
//     // // ┌─────╔═════┌─────  ╔═════┌─────    5
//     // // │     ║  A  │       ║  A  │       ║ <-- x
//     // //  ╔════╝┌────┘  ╔════╝┌────┘  ╔════╝
//     // //  ║  B  │       ║  B  │       ║  A  │ 6
//     // //   ┌────┘  ╔════╝┌────┘  ╔════╝┌────┘
//     // //   │       ║  C  │       ║  B  │     │
//     // //      ═════╝─────┘  ═════╝─────┘─────┘
//     // //                    ^-- y
//     // //
//     // // reading the current diagonal (double-lined)

//     // // TODO: chunk size >= 1
//     // // TODO: special case if (s == -1)
//     // // TODO: allocate header row and column memory
//     // // TODO: transfer appropriate memory to GPU
//     // // TODO: initialize the header row and column for the score matrix
//     // // TODO: create kernel stream and memcpy stream
//     // // TODO: initialize the padding in the global X and Y sequences, as well as the matrix header_row and header_col

//     // // reserve space in the ram and gpu global memory
//     // try
//     // {
//     //     ////// device specific memory
//     //     nw.subst_gpu.init();
//     //     nw.seqX_gpu.init();
//     //     nw.seqY_gpu.init();
//     //     nw.score_gpu.init();
//     //     // sparse representation of the score matrix
//     //     nw.hrow_gpu.init();
//     //     nw.hcol_gpu.init();
//     //     nw.hrowTDi_gpu.init();
//     //     nw.hcolTDi_gpu.init();
//     //     nw.hrowTDo_gpu.init();
//     //     nw.hcolTDo_gpu.init();

//     //     ////// host specific memory
//     //     nw.subst.init();
//     //     nw.seqX.init();
//     //     nw.seqY.init();
//     //     nw.score.init();
//     //     // sparse representation of the score matrix
//     //     nw.hrowM_di.init(); // diag index array into hrowM
//     //     nw.hrowM.init();
//     //     nw.hcolM_di.init(); // diag index arrya into hcolM
//     //     nw.hcolM.init();

//     //     ////// device specific memory
//     //     nw.seqX_gpu.init(adjcols);
//     //     nw.seqY_gpu.init(adjrows);
//     //     // sparse representation of score matrix
//     //     nw.hrow_gpu.init(adjcols);
//     //     nw.hcol_gpu.init(adjrows);
//     //     nw.hrowTDi_gpu.init();
//     //     nw.hcolTDi_gpu.init();
//     //     nw.hrowTDo_gpu.init();
//     //     nw.hcolTDo_gpu.init();

//     //     ////// host specific memory
//     //     nw.subst.init();
//     //     nw.seqX.init();
//     //     nw.seqY.init();
//     //     nw.score.init();
//     //     // sparse representation of score matrix
//     //     nw.hrowM.init();
//     //     nw.hcolM.init();
//     // }
//     // catch (const std::exception &ex)
//     // {
//     //     return NwStat::errorMemoryAllocation;
//     // }

//     // // measure allocation time
//     // sw.lap("alloc");

//     // // copy data from host to device
//     // if (cudaSuccess != (cudaStatus = memTransfer(nw.seqX_gpu, nw.seqX, nw.adjcols)))
//     // {
//     //     return NwStat::errorMemoryTransfer;
//     // }
//     // if (cudaSuccess != (cudaStatus = memTransfer(nw.seqY_gpu, nw.seqY, nw.adjrows)))
//     // {
//     //     return NwStat::errorMemoryTransfer;
//     // }
//     // // also initialize padding, since it is used to access elements in the substitution matrix
//     // if (cudaSuccess != (cudaStatus = memSet(nw.seqX_gpu, nw.adjcols, 0 /*value*/)))
//     // {
//     //     return NwStat::errorMemoryTransfer;
//     // }
//     // if (cudaSuccess != (cudaStatus = memSet(nw.seqY_gpu, nw.adjrows, 0 /*value*/)))
//     // {
//     //     return NwStat::errorMemoryTransfer;
//     // }

//     // // measure memory transfer time
//     // sw.lap("cpy-dev");

//     // //  x x x x x x       x x x x x x       x x x x x x
//     // //  x / / / . .       x . . . / /       x . . . . .|/ /
//     // //  x / / . . .   +   x . . / / .   +   x . . . . /|/
//     // //  x / . . . .       x . / / . .       x . . . / /|
//     // // launch kernel for each minor tile diagonal of the score matrix
//     // {
//     //     // grid and block dimensions for kernel
//     //     dim3 gridA{};
//     //     dim3 blockA{};
//     //     // the number of tiles per row and column of the score matrix
//     //     int trows = ceil(float(adjrows - 1) / tileAy);
//     //     int tcols = ceil(float(adjcols - 1) / tileAx);

//     //     // calculate size of shared memory per block in bytes
//     //     int shmemsz = (
//     //         /*subst[]*/ nw.substsz * nw.substsz * sizeof(int)
//     //         /*seqX[]*/
//     //         + tileAx * sizeof(int)
//     //         /*seqY[]*/
//     //         + tileAy * sizeof(int)
//     //         /*hrow[]*/
//     //         + (1 + tileAx) * sizeof(int)
//     //         /*hcol[]*/
//     //         + (1 + tileAy) * sizeof(int)
//     //         /*warpstate*/
//     //         + warpcnt * sizeof(int));

//     //     // for all minor diagonals in the score matrix (excluding the header row and column)
//     //     for (int d = 0; d < tcols - 1 + trows; d++)
//     //     {
//     //         // calculate grid and block dimensions for kernel B
//     //         {
//     //             int pbeg = max(0, d - (tcols - 1));
//     //             int pend = min(d, trows - 1);

//     //             // the number of elements on the current diagonal
//     //             int dsize = pend - pbeg + 1;

//     //             // take the number of threads per tile as the only dimension
//     //             blockA.x = tileAy;
//     //             // take the number of blocks on the current score matrix diagonal as the only dimension
//     //             // +   launch at least one block on the x axis
//     //             gridA.x = ceil(float(dsize) / tileAy);
//     //         }

//     //         // create variables for gpu arrays in order to be able to take their addresses
//     //         int *seqX_gpu = nw.seqX_gpu.data();
//     //         int *seqY_gpu = nw.seqY_gpu.data();
//     //         int *hrow_gpu = nw.hrow_gpu.data();
//     //         int *hcol_gpu = nw.hcol_gpu.data();
//     //         int *subst_gpu = nw.subst_gpu.data();
//     //         // tile size and miscellaneous
//     //         int *hrowTDi_gpu = nw.hrowTDi_gpu.data();
//     //         int *hcolTDi_gpu = nw.hcolTDi_gpu.data();
//     //         int *hrowTDo_gpu = nw.hrowTDo_gpu.data();
//     //         int *hcolTDo_gpu = nw.hcolTDo_gpu.data();

//     //         // group arguments to be passed to kernel
//     //         void *kargs[]{
//     //             // nw input
//     //             seqX_gpu,
//     //             seqY_gpu,
//     //             hrow_gpu,
//     //             hcol_gpu,
//     //             subst_gpu,
//     //             /*&nw.adjrows,*/
//     //             /*&nw.adjcols,*/
//     //             &nw.substsz,
//     //             &nw.indel,
//     //             &nw.warpsz,
//     //             // tile size and miscellaneous
//     //             hrowTDi_gpu,
//     //             hcolTDi_gpu,
//     //             hrowTDo_gpu,
//     //             hcolTDo_gpu,
//     //             &trows,
//     //             &tcols,
//     //             &tileAx,
//     //             &tileAy,
//     //             &chunksz,
//     //             &d};

//     //         // launch the kernel B in the given stream (don't statically allocate shared memory)
//     //         if (cudaSuccess != (cudaStatus = cudaLaunchKernel((void *)Nw_Gpu10_Kernel, gridA, blockA, kargs, shmemsz, nullptr /*stream*/)))
//     //         {
//     //             return NwStat::errorKernelFailure;
//     //         }
//     //     }
//     // }

//     // // wait for the gpu to finish before going to the next step
//     // if (cudaSuccess != (cudaStatus = cudaDeviceSynchronize()))
//     // {
//     //     return NwStat::errorKernelFailure;
//     // }

//     // // measure calculation time
//     // sw.lap("calc-1");

//     // // save the calculated score matrix
//     // if (cudaSuccess != (cudaStatus = memTransfer(nw.score, nw.score_gpu, nw.adjrows, nw.adjcols, adjcols)))
//     // {
//     //     return NwStat::errorMemoryTransfer;
//     // }

//     // // measure memory transfer time
//     // sw.lap("cpy-host");

//     // return NwStat::success;
// }
