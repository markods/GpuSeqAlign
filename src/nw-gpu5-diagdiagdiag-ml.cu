// #include "common.hpp"

// // presync   active   postsync   stop
// // 0      -> 1     -> 2       -> 3
// enum {
//    presync = 0,
//    active = 1,
//    postsync = 2,
//    stop = 3,
// };


// // cuda kernel for the parallel implementation
// __global__ static void Nw_Gpu5_Kernel(
//    // nw input
//    const int* const seqX_gpu,
//    const int* const seqY_gpu,
//    const int* const hrow_gpu,      // header row for the score matrix
//    const int* const hcol_gpu,      // header column for the score matrix
//    const int* const subst_gpu,
// // const int adjrows,   // can be calculated as 1 + trows*tileAy
// // const int adjcols,   // can be calculated as 1 + tcols*tileAx
//    const int substsz,
//    const int indel,
//    // tile size and miscellaneous
//          int* const hrowTD_gpui,   // input header_row tile_diagonal, consisting of the header rows for all tiles on the current tile diagonal
//          int* const hcolTD_gpui,   // input header_column tile_diagonal, consisting of the header columns for all tiles on the current tile diagonal
//          int* const hrowTD_gpuo,   // output header_row tile_diagonal, consisting of the header rows for all tiles on the current tile diagonal
//          int* const hcolTD_gpuo,   // output header_column tile_diagonal, consisting of the header columns for all tiles on the current tile diagonal
//    const int trows,
//    const int tcols,
//    const unsigned tileAx,
//    const unsigned tileAy,
//    const int chunksz,   // the number of elements the last thread in a warp must calculate before the next block sync
//    const int s   // the current minor tile diagonal in the score matrix (exclude the header row and column)
// )
// {
//    extern __shared__ int shmem[/* substsz*substsz + tileAx + tileAy + (1+tileAx) + (1+tileAy) */];
//    // the substitution matrix and relevant parts of the two sequences
//    // TODO: align allocations to 0-th shared memory bank?
//    int* const subst/*[substsz*substsz]*/      = shmem + ( 0 );
//    int* const seqX/*[tileAx]*/                = subst + ( substsz*substsz );
//    int* const seqY/*[tileAy]*/                = seqX  + ( tileAx );
//    // the header row and column for the tile; they will become the header row for the below tile, and the header column for the right tile
//    int* const hrow/*[1+tileAx]*/              = seqY  + ( tileAy );
//    int* const hcol/*[1+tileAy]*/              = hrow  + ( 1+tileAx );
//    // the warps' state, used to synchronize warps during chunk calculation
//    // sync_before   calc_n_sync   sync_after   stop
//    // 0          -> 1          -> 2         -> 3
//    int* const warpstate/*[warpcnt]*/          = hcol  + ( 1+tileAy );

//    // initialize the substitution shared memory copy
//    {
//       // map the threads from the thread block onto the substitution matrix elements
//       int i = threadIdx.x;
//       // while the current thread maps onto an element in the matrix
//       while( i < substsz*substsz )
//       {
//          // copy the current element from the global substitution matrix
//          el(subst,substsz, 0,i) = el(subst_gpu,substsz, 0,i);
//          // map this thread to the next element with stride equal to the number of threads in this block
//          i += blockDim.x;
//       }
//    }

//    // initialize the tile's window into the global X and Y sequences
//    {
//       //  / / / . .       . . . / /       . . . . .|/ /
//       //  / / . . .   +   . . / / .   +   . . . . /|/
//       //  / . . . .       . / / . .       . . . / /|

//       // (s,t) -- tile coordinates on the score matrix diagonal
//       int tbeg = max( 0, s - (tcols-1) );
//    // int tend = min( s, trows-1 );
//       // position of the current thread's tile on the score matrix diagonal
//       int t = tbeg + blockIdx.x;

//       // unnecessary question, since the number of launched blocks will always be the same as the number of tiles on the diagonal
//    // if( t <= tend )

//       //       x x x x x
//       //       | | | | |
//       //     h h h h h h     // note the x and y seqences on this schematic
//       // y --h c . . . .     // +   they don't! need to be extended by 1 to the left and by 1 to the top
//       // y --h . . . . .
//       // y --h . . . . .
//       // position of the top left not-yet-calculated! element <c> of the current tile in the score matrix
//       // +   only the not-yet-calculated elements will be calculated, and they need the corresponding global sequence X and Y elements
//       int ibeg = 1 + (   t )*tileAy;
//       int jbeg = 1 + ( s-t )*tileAx;

//       // map the threads from the thread block onto the global X sequence's elements (which will be used in this tile)
//       int j = threadIdx.x;
//       // while the current thread maps onto an element in the tile's X sequence
//       while( j < tileAx )
//       {
//          // initialize that element in the X seqence's shared window
//          seqX[ j ] = seqX_gpu[ jbeg + j ];

//          // map this thread to the next element with stride equal to the number of threads in this block
//          j += blockDim.x;
//       }

//       // map the threads from the thread block onto the global Y sequence's elements (which will be used in this tile)
//       int i = threadIdx.x;
//       // while the current thread maps onto an element in the tile's Y sequence
//       while( i < tileAy )
//       {
//          // initialize that element in the Y seqence's shared window
//          seqY[ i ] = seqY_gpu[ ibeg + i ];

//          // map this thread to the next element with stride equal to the number of threads in this block
//          i += blockDim.x;
//       }
//    }

//    //   MOVE DOWN             d=0                              d=1                              d=2                              d=3                              end
//    //   ~   ~ ~ ~   . . .   . . .        .   . . ~   ~ ~ ~   . . .        .   . . .   . . ~   ~ ~ ~        .   . . .   . . .   . . .        .   . . .   . . .   . . .
//    //                                                                                                                                                                
//    //   .   1 1 1   . . .   . . .        .   . . .   1 1 1   . . .        .   . . .   . . .   1 1 1        .   . . .   . . .   . . .        .   . . .   . . .   . . .
//    //   .   1 1 1   . . .   . . .        _   _ _ _   1 1 1   . . .        .   . . _   _ _ _   1 1 1        .   . . .   . . _   _ _ _        .   . . .   . . .   . . .
//    //                                                                                                                                                                
//    //   .   . . .   . . .   . . .        .   2 2 2   . . .   . . .        .   . . .   2 2 2   . . .        .   . . .   . . .   1 1 1        .   . . .   . . .   . . .
//    //   .   . . .   . . .   . . .        .   2 2 2   . . .   . . .        _   _ _ _   2 2 2   . . .        .   . . _   _ _ _   1 1 1        .   . . .   . . _   _ _ _
//    //                                                                                                                                                                
//    //                   || ~a ...                     || ~b -a ...                  || ~c -b -a ...                     || -c -b ...                        || -c ...   // prepend before start, and then remove from end
//    //                       0                             0  1                          0  1  2                             0  1                                0    
//    //                               ->                               ->                               ->                               ->                            
//    //   MOVE RIGHT                                                                                                                                                   
//    //   ~   . . .   . . .   . . .        .   . . |   . . .   . . .        .   . . .   . . |   . . .        .   . . .   . . .   . . |        .   . . .   . . .   . . .
//    //                                                                                                                                                                
//    //   ~   1 1 1   . . .   . . .        .   . . |   1 1 1   . . .        .   . . .   . . |   1 1 1        .   . . .   . . .   . . |        .   . . .   . . .   . . .
//    //   ~   1 1 1   . . .   . . .        ~   . . |   1 1 1   . . .        .   . . |   . . |   1 1 1        .   . . .   . . |   . . |        .   . . .   . . .   . . |
//    //                                                                                                                                                                
//    //   .   . . .   . . .   . . .        ~   2 2 2   . . .   . . .        .   . . |   2 2 2   . . .        .   . . .   . . |   1 1 1        .   . . .   . . .   . . |
//    //   .   . . .   . . .   . . .        ~   2 2 2   . . .   . . .        .   . . |   2 2 2   . . .        .   . . .   . . |   1 1 1        .   . . .   . . .   . . |
//    //                                                                                                                                                                
//    //                   || ~a ...                     || -a ~b ...                     || -a -b ...                     || -a -b ...                        || -b ...   // append to the end, and then remove from start
//    //                       0                             0  1                             0  1                             0  1                                0    

//    // load the current tile's header_row and header_column from the input header_row + header_column tile_diagonals
//    {
//       TODO;

//       // (s,t) -- tile coordinates on the score matrix diagonal
//       int tbeg = max( 0, s - (tcols-1) );
//       int tend = min( s, trows-1 );

//       // the starting location of the header_row in the header_row tile_diagonal
//       int jbeg = (1+tileAx)*( blockIdx.x );
//       // the starting location of the header_column in the header_column tile_diagonal
//       // +   skip the zeroth tile in the header_column tile_diagonal's tiles if the current diagonal is in the lower-right tile_triangle (when the current diagonal loses the previous diagonal's first tile)
//       int ibeg = (1+tileAy)*( blockIdx.x + (s >= trows) );

//       // map the threads from the thread block onto the header_row in the header_row tile_diagonal
//       int j = threadIdx.x;

//       // while the current thread maps onto an element in the corresponding header_row
//       while( j < 1+tileAx )
//       {
//          // copy the current element from the corresponding input header_row
//          hrow[ j ] = hrowTD_gpui[ jbeg + j ];
//          // map this thread to the next element with stride equal to the number of threads in this block
//          j += blockDim.x;
//       }

//       // map the threads from the thread block onto the header_column in the header_column tile_diagonal
//       int i = threadIdx.x;

//       // while the current thread maps onto an element in the corresponding header_column
//       while( i < 1+tileAy )
//       {
//          // copy the current element from the corresponding input header_column
//          hcol[ i ] = hcolTD_gpui[ ibeg + i ];
//          // map this thread to the next element with stride equal to the number of threads in this block
//          i += blockDim.x;
//       }
//    }

//    // initialize the warps' state
//    {
//       // the thread's warp and index inside the warp
//       const int warpIdx       = threadIdx.x / WARPSZ;
//       const int warpThreadIdx = threadIdx.x % WARPSZ;

//       // if this is the first thread in the warp
//       if( warpThreadIdx == 0 )
//       {
//          // presync   active   postsync   stop
//          // 0      -> 1     -> 2       -> 3

//          // initialize the warp state
//          // +   only the first warp is active, the other warps pre-sync
//          warpstate[ warpIdx ] = ( warpIdx == 0 ) ? active : presync;
//       }
//    }

//    // all threads in this block should finish initializing their substitution shared memory, corresponding parts of the global X and Y sequences, the tile's header_row and header_column and the warp's state
//    __syncthreads();

//    // map a tile on the current diagonal of tiles to this thread block
//    {
//       // INITIAL IDEA, BUT NOT COMPLETELY CORRECT
//       //       chunksz           chunksz           chunksz           chunksz    
//       //     x                     x x x x x       x x x x x x       x x x x x x
//       //  y                                                                     
//       //       . . . . . .    y  . . / / 1 .       . . . . . .       . . . . . .     WARPSZ
//       //       . . . . . .    y  . / / 1 . .       . . . . . .       . . . . . .
//       //       . . . . . .    y  / / 1 . . .       . . . . . .       . . . . . .
//       //       . . . . . .    y  / 1 . . . .       . . . . . .       . . . . . .
//       //                              ^----                                     
//       //             x x x       x   <----- issue                               
//       //       . . . . . /    y  / 2 . . . .       . . . . . .       . . . . . .     WARPSZ
//       //  y    . . . . / /    y  2 . . . . .       . . . . . .       . . . . . .
//       //  y    . . . / / 2       . . . . . .       . . . . . .       . . . . . .
//       //  y    . . . / 2 .       . . . . . .       . . . . . .       . . . . . .
//       //                                                                        
//       //       x x x x                                                          
//       //  y   3. . . . . .       . . . . . .       . . . . . .       . . . . . .     WARPSZ
//       //  y   3. . . . . .       . . . . . .       . . . . . .       . . . . . .
//       //  y   3. . . . . .       . . . . . .       . . . . . .       . . . . . .
//       //  y   3. . . . . .       . . . . . .       . . . . . .       . . . . . .

//       // REFINED IDEA, CORRECT
//       //  ________________________________________________   ________________________________________________   ________________________________________________
//       //  |. . . . . / . . / . . . . . . . . . . . . . . .   |. . . . . . . . / . . . / . . / . . . . . . . .   |. . . . . . . . / . . . . . . / . . . / . . / .
//       //  |. .1A . / .1B / . . . . . . . . . . . . . . . .   |. . . . . . . / .2A . / .2B / . . . . . . . . .   |. . . . . . . / . . . . . . / .3A . / .3B / . .   ... after all warps except the last one are initialized,
//       //  |_ _ _ / _ _ / . . . . . . . . . . . . . . . . .   |_ _ _ / _ _ / _ _ _ / _ _ / . . . . . . . . . .   |_ _ _ / _ _ / _ _ _ / _ _ / _ _ _ / _ _ / . . .       only do phase A (c elements per thread, instead of c+w in two phases, A and B)
//       //  |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . / . . / . . . . . . . . . . . . . . .   |. . . . . . . . / . . . / . . / . . . . . . . .
//       //  |. . . c+w . . . . . . . . . . . . . . . . . . .   |. .2A . / .2B / . . . c . . . . . . . . . . . .   |. . . . . . . / .3A . / .3B / . . . c . . . . .       c - chunksz
//       //  |. . . . . . . . . . . . . . . . . . . . . . . .   |_ _ _ / _ _ / . . . . . . . . . . . . . . . . .   |_ _ _ / _ _ / _ _ _ / _ _ / . . . . . . . . . .       w - warpsz
//       //  |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . / . . / . . . . . . . . . . . . . . .
//       //  |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . c+w . . . . . . . . . . . . . . . . . . .   |. .3A . / .3B / . . . c . . . . . . . . . . . .       note: phase A has to be done before phase B, since e.g. 3B would require 3A from above to be finished
//       //  |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . . . . . . . . . . . . . . . . . . . .   |_ _ _ / _ _ / . . . . . . . . . . . . . . . . .       note: a chunk is phases A+B or phase A, therefore actual chunk size is not fixed
//       //  |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . . . . . . . . . . . . . . . . . . . .
//       //  |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . c+w . . . . . . . . . . . . . . . . . . .       important: c > w for the algorithm to work (phase A before phase B),
//       //  |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . . . . . . . . . . . . . . . . . . . .   |. . . . . . . . . . . . . . . . . . . . . . . .                  if c < w swap them

//       // the thread's warp and index inside the warp
//       const int warpIdx       = threadIdx.x / WARPSZ;
//       const int warpThreadIdx = threadIdx.x % WARPSZ;

//       // the elements needed in order to slide the calculation window to the right each iteration
//       // +   save the left element before any calculation (to simplify the algorithm; the thread now only needs to worry how it's going to get its 'up' element)
//       int up     = 0;   // <thread 0 in the warp>: read from hrow on each iteration
//       int upleft = hcol[ 1+ (threadIdx.x - 1) ];
//       int left   = hcol[ 1+ (threadIdx.x    ) ];
//       int curr   = 0;   // stores the value of the newly calculated element

//       // the width of phase A and B
//       // +   phase A's width has to be longer than phase B's width in order for this warp to never access not-yet-initialized! header_row elements (see the refined idea schematic)
//       const int phAwid = max( chunksz, WARPSZ );
//       const int phBwid = min( chunksz, WARPSZ );

//       // wait until the header_row for the warp is ready, and then wait for the warp's diagonal to reach the current thread
//       // +   also initialize the thread's up element
//       {
//          // wait until the above warp has finished its first chunk (the header_row for this warp is ready)
//          while( warpstate[ warpIdx ] == presync )
//          {
//             // presync   active   postsync   stop
//             // 0      -> 1     -> 2       -> 3
//             __syncthreads();
//          }

//          //    |/ / . . . .             . . / / / /             . . . . . .|/ /
//          //   /|/ . . . . . ...  +  ... . / / / / . ...  +  ... . . . . . /|/
//          // / /|. . . . . .             / / / / . .             . . . . / /|

//          // if the thread is the zeroth thread in the warp
//          if( warpThreadIdx == 0 )
//          {
//             // initialize its 'up' element from the header_row
//             up = hrow[ 1+ (0) ];
//          }
//          // otherwise, wait for the warp's diagonal to reach the current thread
//          else for( int i = 0; i < warpThreadIdx; i++ )
//          {
//             // used only for synchronization here, except for the last shuffle which truly initializes the 'up' element
//             up = __shfl_up_sync( 0xffff/*mask*/, curr/*var*/, 1/*delta,*/ /*width=warpSize*/ );
//          }
//       }

//       // current thread: start calculation
//       {
//          // the current element to be calculated
//          int i = threadIdx.x;
//          int j = 0;

//          // for all elements in the thread's tile row
//          while( j < tileAx )
//          {
//             // calculate the current element's value
//             {
//                curr = upleft + el(subst,substsz, seqY[i],seqX[j]);  // MOVE DOWN-RIGHT
//                curr = max( curr, up   + indel );   // MOVE DOWN
//                curr = max( curr, left + indel );   // MOVE RIGHT
//             }

//             // save the results to the header_row and header_column
//             {
//                // if this is the last thread in the warp
//                if( warpThreadIdx == WARPSZ-1 )
//                {
//                   // always save its current element to the header_row
//                   hrow[ 1+ (j) ] = curr;
//                }

//                // if this is the last element in the tile row
//                if( j == tileAx-1 )
//                {
//                   // save that element to the header_column
//                   hcol[ 1+ (i) ] = curr;
//                }
//             }

//             // slide the calculation window to the right
//             // +   also synchronize the warp minor diagonal
//             {
//                // update the elements in the calculation window
//                upleft = up;
//                left   = curr;
//                // copy from a lane with lower id relative to the caller; also sync the threads in the warp
//                // +   NOTE: 'curr' is unused after this step
//                up     = __shfl_up_sync( 0xffff/*mask*/, curr/*var*/, 1/*delta,*/ /*width=warpSize*/ );

//                // if the thread is the zeroth thread in the warp and the next 'up' element exists
//                // +   NOTE: 'up' is not initialized after! the last iteration (which is ok)
//                if( warpThreadIdx == 0 && (j+1) < tileAx )
//                {
//                   // initialize the 'up' element with the next 'up' element from the header_row
//                   up  = hrow[ 1+ (j+1) ];
//                }

//                // move to the next element in the thread's tile row
//                j++;
//             }

//             // synchronize the warp with the thread block, after the last! thread in the warp finishes its chunk row
//             {
//                TODO;
//                __syncthreads();
//             }
//          }

//          // last thread: save the zeroth elements in the header_row and header_column
//          if( threadIdx.x == blockDim.x-1 )
//          {
//             hrow[ 0 ] = hcol[ 1+ (tileAy-1) ];
//             hcol[ 0 ] = hrow[ 1+ (tileAx-1) ];
//          }

//          // wait until the last warp finishes calculating its last chunk stripe
//          // also update the next warp's status
//          TODO
//          for( int i = 0; i < warpIdx; i++ )
//          {
//             __syncthreads();
//          }
//       }
//    }
   
//    // all threads in this block should finish calculating the below tile's header_row and right tile's header_column
//    __syncthreads();

//    //   MOVE DOWN             d=0                              d=1                              d=2                              d=3                              end
//    //   ~   ~ ~ ~   . . .   . . .        .   . . ~   ~ ~ ~   . . .        .   . . .   . . ~   ~ ~ ~        .   . . .   . . .   . . .        .   . . .   . . .   . . .
//    //                                                                                                                                                                
//    //   .   1 1 1   . . .   . . .        .   . . .   1 1 1   . . .        .   . . .   . . .   1 1 1        .   . . .   . . .   . . .        .   . . .   . . .   . . .
//    //   .   1 1 1   . . .   . . .        _   _ _ _   1 1 1   . . .        .   . . _   _ _ _   1 1 1        .   . . .   . . _   _ _ _        .   . . .   . . .   . . .
//    //                                                                                                                                                                
//    //   .   . . .   . . .   . . .        .   2 2 2   . . .   . . .        .   . . .   2 2 2   . . .        .   . . .   . . .   1 1 1        .   . . .   . . .   . . .
//    //   .   . . .   . . .   . . .        .   2 2 2   . . .   . . .        _   _ _ _   2 2 2   . . .        .   . . _   _ _ _   1 1 1        .   . . .   . . _   _ _ _
//    //                                                                                                                                                                
//    //                   || ~a ...                     || ~b -a ...                  || ~c -b -a ...                     || -c -b ...                        || -c ...   // prepend before start, and then remove from end
//    //                       0                             0  1                          0  1  2                             0  1                                0    
//    //                               ->                               ->                               ->                               ->                            
//    //   MOVE RIGHT                                                                                                                                                   
//    //   ~   . . .   . . .   . . .        .   . . |   . . .   . . .        .   . . .   . . |   . . .        .   . . .   . . .   . . |        .   . . .   . . .   . . .
//    //                                                                                                                                                                
//    //   ~   1 1 1   . . .   . . .        .   . . |   1 1 1   . . .        .   . . .   . . |   1 1 1        .   . . .   . . .   . . |        .   . . .   . . .   . . .
//    //   ~   1 1 1   . . .   . . .        ~   . . |   1 1 1   . . .        .   . . |   . . |   1 1 1        .   . . .   . . |   . . |        .   . . .   . . .   . . |
//    //                                                                                                                                                                
//    //   .   . . .   . . .   . . .        ~   2 2 2   . . .   . . .        .   . . |   2 2 2   . . .        .   . . .   . . |   1 1 1        .   . . .   . . .   . . |
//    //   .   . . .   . . .   . . .        ~   2 2 2   . . .   . . .        .   . . |   2 2 2   . . .        .   . . .   . . |   1 1 1        .   . . .   . . .   . . |
//    //                                                                                                                                                                
//    //                   || ~a ...                     || -a ~b ...                     || -a -b ...                     || -a -b ...                        || -b ...   // append to the end, and then remove from start
//    //                       0                             0  1                             0  1                             0  1                                0    

//    // store the below tile's header_row and right tile's header_column to the output header_row + header_column tile_diagonal
//    {
//       TODO;

//       // (s,t) -- tile coordinates on the score matrix diagonal
//       int tbeg = max( 0, s - (tcols-1) );
//       int tend = min( s, trows-1 );

//       // the starting location of the header_row in the header_row tile_diagonal
//       // +   shift the header_row tile_diagonal's tiles by one tile to the right if the next zeroth tile on the tile_diagonal touches the matrix header row (should be initialized by the gpu)
//       int jbeg = (1+tileAx)*( blockIdx.x + (s < tcols) );
//       // the starting location of the header_column in the header_column tile_diagonal
//       int ibeg = (1+tileAy)*( blockIdx.x );

//       // map the threads from the thread block onto the header_row in the header_row tile_diagonal
//       int j = threadIdx.x;
//       // while the current thread maps onto an element in the corresponding header_row
//       while( j < (1+tileAx) )
//       {
//          // copy the current element from the corresponding output header_row
//          hrowTD_gpuo[ jbeg + j ] = hrow[ j ];
//          // map this thread to the next element with stride equal to the number of threads in this block
//          j += blockDim.x;
//       }

//       // map the threads from the thread block onto the header_column in the header_column tile_diagonal
//       int i = threadIdx.x;
//       // while the current thread maps onto an element in the corresponding header_column
//       while( i < (1+tileAy) )
//       {
//          // copy the current element from the corresponding output header_column
//          hcolTD_gpuo[ ibeg + i ] = hcol[ i ];
//          // map this thread to the next element with stride equal to the number of threads in this block
//          i += blockDim.x;
//       }
//    }
// }



// // parallel gpu implementation of the Needleman-Wunsch algorithm
// NwStat NwAlign_Gpu5_DiagDiagDiag_Coop( NwParams& pr, NwInput& nw, NwResult& res )
// {
//    // TODO: allocate header row and column memory
//    // TODO: initialize the header row and column for the score matrix
//    // hrowM
//    // hcolM

//    // tile size for the kernel
//    unsigned tileAx = 320;
//    unsigned tileAy = 4*WARPSZ;   // must be a multiple of the warp size
//    int chunksz = 32;

//    // substitution matrix, sequences which will be compared and the score matrix stored in gpu global memory
//    NwInput nw_gpu = {
//       // nw.seqX,
//       // nw.seqY,
//       // nw.score,
//       // nw.subst,

//       // nw.adjrows,
//       // nw.adjcols,
//       // nw.substsz,

//       // nw.indel,
//    };

//    // adjusted gpu score matrix dimensions
//    // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile A size (in order to be evenly divisible)
//    nw_gpu.adjrows = 1 + tileAy*ceil( float( nw.adjrows-1 )/tileAy );
//    nw_gpu.adjcols = 1 + tileAx*ceil( float( nw.adjcols-1 )/tileAx );
//    nw_gpu.substsz = nw.substsz;
//    nw_gpu.indel = nw.indel;

//    // allocate space in the gpu global memory
//    cudaMalloc( &nw_gpu.seqX,  nw_gpu.adjcols                * sizeof( int ) );
//    cudaMalloc( &nw_gpu.seqY,  nw_gpu.adjrows                * sizeof( int ) );
//    cudaMalloc( &nw_gpu.score, nw_gpu.adjrows*nw_gpu.adjcols * sizeof( int ) );
//    cudaMalloc( &nw_gpu.subst, nw_gpu.substsz*nw_gpu.substsz * sizeof( int ) );
//    // create events for measuring kernel execution time
//    cudaEvent_t start, stop;
//    cudaEventCreate( &start );
//    cudaEventCreate( &stop );

//    // start the host timer and initialize the gpu timer
//    res.sw.lap( "cpu-start" );
//    res.Tgpu = 0;

//    // copy data from host to device
//    // +   gpu padding remains uninitialized, but this is not an issue since padding is only used to simplify kernel code (optimization)
//    cudaMemcpy( nw_gpu.seqX,  nw.seqX,  nw.adjcols * sizeof( int ), cudaMemcpyHostToDevice );
//    cudaMemcpy( nw_gpu.seqY,  nw.seqY,  nw.adjrows * sizeof( int ), cudaMemcpyHostToDevice );
//    cudaMemcpy( nw_gpu.subst, nw.subst, nw_gpu.substsz*nw_gpu.substsz * sizeof( int ), cudaMemcpyHostToDevice );


//    // launch kernel
//    {
//       // grid and block dimensions for kernel
//       dim3 gridA {};
//       dim3 blockA {};
//       // the number of tiles per row and column of the score matrix
//       int trows = ceil( float( nw_gpu.adjrows-1 )/tileAy );
//       int tcols = ceil( float( nw_gpu.adjcols-1 )/tileAx );

//       // calculate size of shared memory per block in bytes
//       int shmemsz = (
//          /*subst[]*/ nw_gpu.substsz*nw_gpu.substsz *sizeof( int )
//          /*seqX[]*/ + tileAx                       *sizeof( int )
//          /*seqY[]*/ + tileAy                       *sizeof( int )
//          /*hrow[]*/ + (1+tileAx)                   *sizeof( int )
//          /*hcol[]*/ + (1+tileAy)                   *sizeof( int )
//       );
      
//       // calculate grid and block dimensions for kernel
//       {
//          // take the number of threads on the largest diagonal of the tile
//          // +   multiply by the number of half warps in the larger dimension for faster writing to global gpu memory
//          blockA.x = WARPSZ * ceil( max( tileAy, tileAx )*2./WARPSZ );
//          // take the number of tiles on the largest score matrix diagonal as the only dimension
//          gridA.x = min( trows, tcols );

//          // the maximum number of parallel blocks on a streaming multiprocessor
//          int maxBlocksPerSm = 0;
//          // number of threads per block that the kernel will be launched with
//          int numThreads = blockA.x;

//          // calculate the max number of parallel blocks per streaming multiprocessor
//          cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerSm, Nw_Gpu5_Kernel, numThreads, shmemsz );
//          // the number of cooperative blocks launched must not exceed the maximum possible number of parallel blocks on the device
//          gridA.x = min( gridA.x, MPROCS*maxBlocksPerSm );
//       }


//       // group arguments to be passed to kernel
//       void* kargs[] { &nw_gpu.seqX, &nw_gpu.seqY, &nw_gpu.score, &nw_gpu.subst, /*&nw_gpu.adjrows,*/ /*&nw_gpu.adjcols,*/ &nw_gpu.substsz, &nw_gpu.indel, &trows, &tcols, &tileAx, &tileAy, &chunksz };
      
//       // launch the kernel in the given stream (don't statically allocate shared memory)
//       // +   capture events around kernel launch as well
//       // +   update the stop event when the kernel finishes
//       cudaEventRecord( start, nullptr/*stream*/ );
//       cudaLaunchCooperativeKernel( ( void* )Nw_Gpu5_Kernel, gridA, blockA, kargs, shmemsz, nullptr/*stream*/ );
//       cudaEventRecord( stop, nullptr/*stream*/ );
//       cudaEventSynchronize( stop );
      
//       // kernel execution time
//       float ktime {};
//       // calculate the time between the given events
//       cudaEventElapsedTime( &ktime, start, stop ); ktime /= 1000./*ms*/;
//       // update the total kernel execution time
//       res.Tgpu += ktime;
//    }

//    // wait for the gpu to finish before going to the next step
//    cudaDeviceSynchronize();

//    // \brief Copies data between host and device
//    // 
//    // Copies a matrix (\p height rows of \p width bytes each) from the memory
//    // area pointed to by \p src to the memory area pointed to by \p dst, where
//    // \p kind specifies the direction of the copy, and must be one of
//    // ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
//    // ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
//    // ::cudaMemcpyDefault is recommended, in which case the type of transfer is
//    // inferred from the pointer values. However, ::cudaMemcpyDefault is only
//    // allowed on systems that support unified virtual addressing. \p dpitch and
//    // \p spitch are the widths in memory in bytes of the 2D arrays pointed to by
//    // \p dst and \p src, including any padding added to the end of each row. The
//    // memory areas may not overlap. \p width must not exceed either \p dpitch or
//    // \p spitch. Calling ::cudaMemcpy2D() with \p dst and \p src pointers that do
//    // not match the direction of the copy results in an undefined behavior.
//    // ::cudaMemcpy2D() returns an error if \p dpitch or \p spitch exceeds
//    // the maximum allowed.

//    // save the calculated score matrix
//    // +   starts an async data copy from device to host, then waits for the copy to finish
//    cudaMemcpy2D(
//       nw    .score,                     // dst    - Destination memory address
//       nw    .adjcols * sizeof( int ),   // dpitch - Pitch of destination memory (padded row size in bytes; in other words distance between the starting points of two rows)
//       nw_gpu.score,                     // src    - Source memory address
//       nw_gpu.adjcols * sizeof( int ),   // spitch - Pitch of source memory (padded row size in bytes)
      
//       nw.adjcols * sizeof( int ),       // width  - Width of matrix transfer (non-padding row size in bytes)
//       nw.adjrows,                       // height - Height of matrix transfer (#rows)
//       cudaMemcpyDeviceToHost            // kind   - Type of transfer
//    );      

//    // stop the cpu timer
//    res.sw.lap( "cpu-end" );
//    res.Tcpu = res.sw.dt( "cpu-end", "cpu-start" );

   
//    // free allocated space in the gpu global memory
//    cudaFree( nw_gpu.seqX );
//    cudaFree( nw_gpu.seqY );
//    cudaFree( nw_gpu.score );
//    cudaFree( nw_gpu.subst );
//    // free events' memory
//    cudaEventDestroy( start );
//    cudaEventDestroy( stop );
// }





