// missing cpp file on purpose, since whole program optimization is disabled
#pragma once


// stream used through the rest of the program
#define STREAM_ID 0
// number of streaming multiprocessors (sm-s) and cores per sm
#define MPROCS 28
#define CORES 128
// number of threads in warp
#define WARPSZ 32
// tile sizes for kernels A and B
// +   tile A should have one dimension be a multiple of the warp size for full memory coallescing
// +   tile B must have one dimension fixed to the number of threads in a warp
const int tileAx = 1*WARPSZ;
const int tileAy = 32;
const int tileBx = 60;
const int tileBy = WARPSZ;

// get the specified element from the given linearized matrix
#define el( mat, cols, i, j ) ( mat[(i)*(cols) + (j)] )


// block substitution matrix
#define BLOSUMSZ 24
static int blosum62[BLOSUMSZ][BLOSUMSZ] =
{
   {  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4 },
   { -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4 },
   { -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4 },
   { -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4 },
   {  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4 },
   { -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4 },
   { -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4 },
   {  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4 },
   { -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4 },
   { -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4 },
   { -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4 },
   { -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4 },
   { -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4 },
   { -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4 },
   { -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4 },
   {  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4 },
   {  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4 },
   { -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4 },
   { -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4 },
   {  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4 },
   { -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4 },
   { -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4 },
   {  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4 },
   { -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1 }
};


// calculate the maximum of two numbers
[[nodiscard]] constexpr const int& max2( const int& a, const int& b ) noexcept { return ( a >= b ) ? a : b; }
// calculate the minimum of two numbers
[[nodiscard]] constexpr const int& min2( const int& a, const int& b ) noexcept { return ( a < b ) ? a : b; }
// calculate the maximum of three numbers
[[nodiscard]] constexpr const int& max3( const int& a, const int& b, const int& c ) noexcept
{
   if( a >= b ) { return ( a >= c ) ? a : c; }
   else         { return ( b >= c ) ? b : c; }
}


// update the score given the current score matrix and position
inline void UpdateScore( const int* seqX, const int* seqY, int* score, int rows, int cols, int insdelcost, int i, int j )
{
   int p1 = el(score,cols, i-1,j-1) + blosum62[ seqY[i] ][ seqX[j] ];
   int p2 = el(score,cols, i-1,j  ) - insdelcost;
   int p3 = el(score,cols, i  ,j-1) - insdelcost;
   el(score,cols, i,j) = max3( p1, p2, p3 );
}


// sequential implementation of the Needleman Wunsch algorithm
int CpuSequential( const int* seqX, const int* seqY, int* score, int rows, int cols, int adjrows, int adjcols, int insdelcost, float* time );
// parallel cpu implementation of the Needleman Wunsch algorithm
int CpuParallel( const int* seqX, const int* seqY, int* score, int rows, int cols, int adjrows, int adjcols, int insdelcost, float* time );
// parallel implementation of the Needleman Wunsch algorithm (fast)
int GpuParallel1( const int* seqX, const int* seqY, int* score, int rows, int cols, int adjrows, int adjcols, int insdelcost, float* time, float* ktime );
// parallel implementation of the Needleman Wunsch algorithm (medium)
int GpuParallel2( const int* seqX, const int* seqY, int* score, int rows, int cols, int adjrows, int adjcols, int insdelcost, float* time, float* ktime );





