#ifndef INCLUDE_COMMON_HPP
#define INCLUDE_COMMON_HPP

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <cuda_runtime.h>
using namespace std::string_literals;


// get the specified element from the given linearized matrix
#define el( mat, cols, i, j ) ( mat[(cols)*(i) + (j)] )


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


// Needleman-Wunsch status
enum class NwStat : int
{
   success               = 0,
   errorMemoryAllocation = 1,
   errorMemoryTransfer   = 2,
   errorKernelFailure    = 3,
   errorIoStream         = 4,
   errorInvalidFormat    = 5,
   errorInvalidValue     = 6,
};
// cuda status, used for getting the return status of cuda functions
extern thread_local cudaError_t cudaStatus;



// create an uninitialized array on the host
template< typename T >
class HostArray
{
public:
   HostArray():
      _arr { nullptr, []( T* ptr ) {} },
      _size {}
   { }

   void init( size_t size )
   {
      if( _size == size ) return;

      T* pAlloc = nullptr;
      if( size > 0 )
      {
         pAlloc = ( T* )malloc( size*sizeof( T ) );
         if( pAlloc == nullptr )
         {
            throw std::bad_alloc();
         }
      }

      pointer arr
      {
         pAlloc,
         []( T* ptr ) { if( ptr != nullptr ) free( ptr ); }
      };

      std::swap( _arr, arr );
      _size = size;
   }
   void clear()
   {
      init( 0 );
   }

   T& operator[] ( size_t pos ) { return data()[ pos ]; }
   const T& operator[] ( size_t pos ) const { return data()[ pos ]; }

   T* data() { return _arr.get(); }
   const T* data() const { return _arr.get(); }

   size_t size() const { return _size; }


private:
   using pointer = std::unique_ptr<T, void(*)(T*)>;
   pointer _arr;
   size_t _size;
};


// create an uninitialized array on the device
template< typename T >
class DeviceArray
{
public:
   DeviceArray():
      _arr { nullptr, []( T* ptr ) {} },
      _size {}
   { }

   void init( size_t size )
   {
      if( _size == size ) return;

      T* pAlloc = nullptr;
      if( size > 0 )
      {
         if( cudaSuccess != ( cudaStatus = cudaMalloc( &pAlloc, size*sizeof( T ) ) ) )
         {
            throw std::bad_alloc();
         }
      }

      pointer arr
      {
         pAlloc,
         []( T* ptr ) { if( ptr != nullptr ) cudaFree( ptr ); }
      };

      std::swap( _arr, arr );
      _size = size;
   }
   void clear()
   {
      init( 0 );
   }

   T* data() { return _arr.get(); }
   const T* data() const { return _arr.get(); }

   size_t size() const { return _size; }


private:
   using pointer = std::unique_ptr<T, void(*)(T*)>;
   pointer _arr;
   size_t _size;
};


// measure time between events
class Stopwatch
{
public:
   void start()
   {
      _start = Clock::now();
   }
   void lap( std::string lap_name )
   {
      auto curr = Clock::now();
      auto diff = std::chrono::duration_cast<Millis>( curr - _start ).count() / 1000.f;
      _start = curr;

      _laps.insert_or_assign( lap_name, diff );
   }
   void reset() noexcept
   {
      _start = {};
      _laps.clear();
   }

private:
   using Clock = std::chrono::steady_clock;
   using TimePoint = std::chrono::time_point<Clock>;
   using Millis = std::chrono::milliseconds;

   TimePoint _start;
   std::map< std::string, float > _laps;
};


// parameter that takes values from a vector
struct NwParam
{
public:
   NwParam() = default;
   NwParam( std::vector<int> values )
   {
      _values = values;
      _curr = 0;
   }

   int curr() { return _values[ _curr ]; }
   bool hasCurr() { return _curr < _values.size(); }
   void next() { _curr++; }
   void reset() { _curr = 0; }

   std::vector<int> _values;
   int _curr;
};

// parameters for the Needleman-Wunsch algorithm variant
struct NwParams
{
   NwParams()
   {
      _params = { };
      _isEnd = false;
   }
   NwParams( std::map< std::string, NwParam > params )
   {
      _params = params;
      // always allow the inital iteration, even if there are no params
      _isEnd = false;
   }

   NwParam& operator[] ( const std::string name ) { return _params.at( name ); }

   bool hasCurr() { return !_isEnd; }
   void next()   // updates starting from the last parameter and so on
   {
      for( auto iter = _params.rbegin();   iter != _params.rend();   iter++ )
      {
         auto& param = iter->second;
         param.next();
         
         if( param.hasCurr() ) return;
         param.reset();
      }
      _isEnd = true;
   }
   void reset()
   {
      for( auto iter = _params.rbegin();   iter != _params.rend();   iter++ )
      {
         auto& param = iter->second;
         param.reset();
      }
      _isEnd = false;
   }

   std::map< std::string, NwParam > _params;
   bool _isEnd;
};

// input for the Needleman-Wunsch algorithm variant
struct NwInput
{
   // IMPORTANT: dont't use .size() on vectors to get the number of elements, since it is not accurate
   // +   instead, use the alignment parameters below

   // host specific memory
   std::vector<int> subst;
   std::vector<int> seqX;
   std::vector<int> seqY;
   HostArray<int> score;
   
   // device specific memory
   DeviceArray<int> subst_gpu;
   DeviceArray<int> seqX_gpu;
   DeviceArray<int> seqY_gpu;
   DeviceArray<int> score_gpu;

   // alignment parameters
   int substsz;
   int adjrows;
   int adjcols;

   int indel;

   // device parameters
   int sm_count;
   int warpsz;

   // free all memory allocated by the Needleman-Wunsch algorithms
   void resetAllocs()
   {
      score.clear();

      seqX_gpu.clear();
      seqY_gpu.clear();
      score_gpu.clear();
   }
};

// results which the Needleman-Wunsch algorithm variant returns
struct NwResult
{
   Stopwatch sw;
   std::vector<int> trace;
   unsigned score_hash;
   unsigned trace_hash;
};

// update the score given the current score matrix and position
// NOTE: indel and most elements in the substitution matrix are negative, therefore find the maximum of them (instead of the minimum)
inline void UpdateScore( NwInput& nw, int i, int j ) noexcept
{
   int p1 = el(nw.score,nw.adjcols, i-1,j-1) + el(nw.subst,nw.substsz, nw.seqY[i], nw.seqX[j]);  // MOVE DOWN-RIGHT
   int p2 = el(nw.score,nw.adjcols, i-1,j  ) + nw.indel;   // MOVE DOWN
   int p3 = el(nw.score,nw.adjcols, i  ,j-1) + nw.indel;   // MOVE RIGHT
   el(nw.score,nw.adjcols, i,j) = max3( p1, p2, p3 );
}



// transfer data between the host and the device
template< typename T >
cudaError_t memTransfer(
   T* const dst,
   const T* const src,
   int elemcnt,
   cudaMemcpyKind kind
)
{
   cudaError_t status = cudaMemcpy(
      /*dst*/   dst,                     // Destination memory address
      /*src*/   src,                     // Source memory address
      /*count*/ elemcnt * sizeof( T ),   // Size in bytes to copy
      /*kind*/  kind                     // Type of transfer
   );

   return status;
}
template< typename T >
cudaError_t memTransfer(
   DeviceArray<T>& dst,
   const std::vector<T>& src,
   int elemcnt
)
{
   return memTransfer( dst.data(), src.data(), elemcnt, cudaMemcpyHostToDevice );
}
template< typename T >
cudaError_t memTransfer(
   HostArray<T>& dst,
   const DeviceArray<T>& src,
   int elemcnt
)
{
   return memTransfer( dst.data(), src.data(), elemcnt, cudaMemcpyDeviceToHost );
}

// transfer a pitched matrix to a contiguous matrix, between the host and the device
// + NOTE: dst and src cannot overlap
template< typename T >
cudaError_t memTransfer(
   T* const dst,
   const T* const src,
   int dst_rows,
   int dst_cols,
   int src_cols,
   cudaMemcpyKind kind
)
{
   cudaError_t status = cudaMemcpy2D(
      /*dst*/    dst,                      // Destination memory address
      /*dpitch*/ dst_cols * sizeof( T ),   // Pitch of destination memory (padded row size in bytes; in other words distance between the starting points of two rows)
      /*src*/    src,                      // Source memory address
      /*spitch*/ src_cols * sizeof( T ),   // Pitch of source memory (padded row size in bytes)
      
      /*width*/  dst_cols * sizeof( T ),   // Width of matrix transfer (non-padding row size in bytes)
      /*height*/ dst_rows,                 // Height of matrix transfer (#rows)
      /*kind*/   kind                      // Type of transfer
   );

   return status;
}
template< typename T >
cudaError_t memTransfer(
   HostArray<T>& dst,
   const DeviceArray<T>& src,
   int dst_rows,
   int dst_cols,
   int src_cols
)
{
   return memTransfer( dst.data(), src.data(), dst_rows, dst_cols, src_cols, cudaMemcpyDeviceToHost );
}


// iostream format flags guard
template< typename T >
class FormatFlagsGuard
{
public:
   FormatFlagsGuard( T& stream, int fwidth = 1, char ffill = ' ' )
      : _stream { stream }
   {
      // backup format flags and set the fill character and width
      _fflags = _stream.flags();
      _fwidth = _stream.width( fwidth );
      _ffill = _stream.fill( ffill );
   }

   ~FormatFlagsGuard()
   {
      // restore the format flags, fill character and width
      _stream.flags( _fflags );
      _stream.width( _fwidth );
      _stream.fill( _ffill );
   }

private:
   T& _stream;
   std::ios_base::fmtflags _fflags;
   std::streamsize _fwidth;
   char _ffill;
};









#endif // INCLUDE_COMMON_HPP
