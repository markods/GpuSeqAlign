#include <cstdio>
#include "common.hpp"

// for diagnostic purposes
void HashAndZeroOutMatrix(
   int* const mat,
   const int rows,
   const int cols,
   unsigned& _hash
) noexcept
{
   // variable used to calculate the hash function
   // http://www.cse.yorku.ca/~oz/hash.html
   // the starting value is a magic constant
   unsigned hash = 5381;

   for( int i = 0; i < rows; i++ )
   for( int j = 0; j < cols; j++ )
   {
      // add the current element to the hash
      int curr = el(mat,cols, i,j);
      hash = ( ( hash<<5 ) + hash ) ^ curr;
      
      // zero out the current element
      el(mat,cols, i,j) = 0;
   }

   // save the resulting hash
   _hash = hash;
}

// for diagnostic purposes
void PrintMatrix(
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

