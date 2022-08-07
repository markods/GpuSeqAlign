#pragma once
#include "common.hpp"

// print one of the optimal matching paths to a file
void Trace1_Diag( const NwInput& nw, NwMetrics& res )
{
   // variable used to calculate the hash function
   // http://www.cse.yorku.ca/~oz/hash.html
   // the starting value is a magic constant
   unsigned hash = 5381;

   // for all elements on one of the optimal paths
   bool loop = true;
   for( int i = nw.adjrows-1, j = nw.adjcols-1;  loop;  )
   {
      // add the current element to the trace and hash
      int curr = el(nw.score,nw.adjcols, i,j);
      res.trace.push_back( curr );
      hash = ( ( hash<<5 ) + hash ) ^ curr;

      int max = std::numeric_limits<int>::min();   // maximum value of the up, left and diagonal neighbouring elements
      int dir = '-';                               // the current movement direction is unknown

      if( i > 0 && j > 0 && max < el(nw.score,nw.adjcols, i-1,j-1) ) { max = el(nw.score,nw.adjcols, i-1,j-1); dir = 'i'; }   // diagonal movement if possible
      if( i > 0          && max < el(nw.score,nw.adjcols, i-1,j  ) ) { max = el(nw.score,nw.adjcols, i-1,j  ); dir = 'u'; }   // up       movement if possible
      if(          j > 0 && max < el(nw.score,nw.adjcols, i  ,j-1) ) { max = el(nw.score,nw.adjcols, i  ,j-1); dir = 'l'; }   // left     movement if possible

      // move to the neighbour with the maximum value
      switch( dir )
      {
      case 'i': i--; j--; break;
      case 'u': i--;      break;
      case 'l':      j--; break;
      default:  loop = false; break;
      }
   }

   // reverse the trace, so it starts from the top-left corner of the matrix
   std::reverse( res.trace.begin(), res.trace.end() );
   // save the hash value
   res.hash = hash;
}
