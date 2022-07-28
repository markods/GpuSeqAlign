// missing Common.cpp file on purpose, since whole program optimization is disabled
#pragma once

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

// TODO: test performance of min2, max2 and max3 without branching
// +   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

// calculate the minimum of two numbers
inline const int& min2( const int& a, const int& b ) noexcept
{
   return ( a < b ) ? a : b;
}
// calculate the maximum of two numbers
inline const int& max2( const int& a, const int& b ) noexcept
{
   return ( a >= b ) ? a : b;
}
// calculate the maximum of three numbers
inline const int& max3( const int& a, const int& b, const int& c ) noexcept
{
   if( a >= b ) { return ( a >= c ) ? a : c; }
   else         { return ( b >= c ) ? b : c; }
}

// update the score given the current score matrix and position
inline void UpdateScore(
   const int* const seqX,
   const int* const seqY,
   int* const score,
   const int rows,
   const int cols,
   const int insdelcost,
   const int i,
   const int j )
   noexcept
{
   const int p1 = el(score,cols, i-1,j-1) + blosum62[ seqY[i] ][ seqX[j] ];
   const int p2 = el(score,cols, i-1,j  ) - insdelcost;
   const int p3 = el(score,cols, i  ,j-1) - insdelcost;
   el(score,cols, i,j) = max3( p1, p2, p3 );
}



// arguments for the Needleman-Wunsch algorithm variants
struct NWArgs
{
   const int* seqX = nullptr;
   const int* seqY = nullptr;
   
   int* score = nullptr;
   int rows = 0;
   int cols = 0;

   int adjrows = 0;
   int adjcols = 0;

   int insdelcost = 0;
};

// results that the Needleman-Wunsch algorithm variants return
struct NWResult
{
   const char* algname = "alg";
   const char* fpath = "./out.txt";
   unsigned hash = 0;
   
   float Tcpu = 0;
   float Tgpu = 0;
};

// Needleman-Wunsch template algorithm
template<typename T>
class NWVariant
{
public:
   void run( NWArgs& args, NWResult& res )
   {
      static_cast<T*>( this )->impl( args );
   }

   // define the actual implementation in subclasses
// void impl() = 0;
};


using Variant = void (*)( NWArgs& args, NWResult& res );








