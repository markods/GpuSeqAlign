#ifndef INCLUDE_COMMON_HPP
#define INCLUDE_COMMON_HPP

#include <cstdio>
#include <chrono>
#include <memory>
#include <limits>
#include <unordered_map>


// TODO: test performance of min2, max2 and max3 without branching
// +   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

// calculate the minimum of two numbers
inline const int& min2( const int& a, const int& b ) noexcept
{
   return ( a <= b ) ? a : b;
}
// calculate the maximum of two numbers
inline const int& max2( const int& a, const int& b ) noexcept
{
   return ( a >= b ) ? a : b;
}
// calculate the minimum of three numbers
inline const int& min3( const int& a, const int& b, const int& c ) noexcept
{
   return ( a <= b ) ? ( ( a <= c ) ? a : c ):
                       ( ( b <= c ) ? b : c );
}
// calculate the maximum of three numbers
inline const int& max3( const int& a, const int& b, const int& c ) noexcept
{
   return ( a >= b ) ? ( ( a >= c ) ? a : c ):
                       ( ( b >= c ) ? b : c );
}



// number of streaming multiprocessors (sm-s) and cores per sm
constexpr int MPROCS = 28;
constexpr int CORES = 128;
// number of threads in warp
constexpr int WARPSZ = 32;

// get the specified element from the given linearized matrix
#define el( mat, cols, i, j ) ( mat[(cols)*(i) + (j)] )

// for diagnostic purposes
inline void PrintMatrix(
   const int* const mat,
   const int rows,
   const int cols
)
{
   printf( "\n" );
   for( int i = 0; i < rows; i++ )
   {
      for( int j = 0; j < cols; j++ )
      {
         printf( "%4d ", el(mat,cols, i,j) );
      }
      printf( "\n" );
   }
   fflush(stdout);
}

// for diagnostic purposes
inline void ZeroOutMatrix(
   int* const mat,
   const int rows,
   const int cols
) noexcept
{
   for( int i = 0; i < rows; i++ )
   for( int j = 0; j < cols; j++ )
   {
      el(mat,cols, i,j) = 0;
   }
}



class Stopwatch
{
public:
   void lap( std::string lap_name )
   {
      laps.insert_or_assign( lap_name, Clock::now() );
   }

   void reset() noexcept
   {
      laps.clear();
   }


   float dt( std::string lap1_name, std::string lap2_name )
   {
      auto p1_iter = laps.find( lap1_name );
      auto p2_iter = laps.find( lap2_name );

      auto p1 = p1_iter->second;
      auto p2 = p2_iter->second;
      return std::chrono::duration_cast<Resolution>( p1 - p2 ).count() / 1000.;
   }

private:
   using Clock = std::chrono::steady_clock;
   using Resolution = std::chrono::milliseconds;

   std::unordered_map< std::string, std::chrono::time_point<Clock> > laps;
};



// arguments for the Needleman-Wunsch algorithm variants
struct NwInput
{
   int* seqX;
   int* seqY;
   int* score;
   int* subst;

   int adjrows;
   int adjcols;
   int substsz;

   int indelcost;
};

// results which the Needleman-Wunsch algorithm variants return
struct NwMetrics
{
   Stopwatch sw;
   float Tcpu;
   float Tgpu;
   std::vector<int> trace;
   unsigned hash;
};


using NwVariant = void (*)( NwInput& nw, NwMetrics& res );
void Nw_Cpu1_Row_St( NwInput& nw, NwMetrics& res );
void Nw_Cpu2_Diag_St( NwInput& nw, NwMetrics& res );
void Nw_Cpu3_DiagRow_St( NwInput& nw, NwMetrics& res );
void Nw_Cpu4_DiagRow_Mt( NwInput& nw, NwMetrics& res );
void Nw_Gpu2_DiagDiag_Coop( NwInput& nw, NwMetrics& res );
void Nw_Gpu3_DiagDiag_Coop2K( NwInput& nw, NwMetrics& res );

void Trace1_Diag( const NwInput& nw, NwMetrics& res );

// update the score given the current score matrix and position
inline void UpdateScore( NwInput& nw, int i, int j ) noexcept
{
   int p1 = el(nw.score,nw.adjcols, i-1,j-1) + el(nw.subst,nw.substsz, nw.seqY[i], nw.seqX[j]);  // MOVE DOWN-RIGHT
   int p2 = el(nw.score,nw.adjcols, i-1,j  ) - nw.indelcost;   // MOVE DOWN
   int p3 = el(nw.score,nw.adjcols, i  ,j-1) - nw.indelcost;   // MOVE RIGHT
   el(nw.score,nw.adjcols, i,j) = max3( p1, p2, p3 );
}




#endif INCLUDE_COMMON_HPP
