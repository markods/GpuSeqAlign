/*
./build -run 2048 10
./build -run 6144 10
./build -run 16384 10
./build -run 22528 10
*/

#include <string>
#include <time.h>
#include "Common.h"
// #define INT_MAX +2147483647
// #define INT_MIN -2147483648


// call in case of invalid command line arguments
void Usage( char* argv[] )
{
   fprintf(stderr, "nw dim [cost]\n", argv[0]);
   fprintf(stderr, "   dim   - square matrix dimensions\n");
   fprintf(stderr, "   cost  - insert and delete cost (positive integer)\n");
   fflush(stderr);
   exit(0);
}


// print one of the optimal matching paths to a file
void Traceback( const char* fname, int* score, int rows, int cols, int adjrows, int adjcols, unsigned* _hash )
{
   printf("   - printing traceback\n");
   
   // if the given file and matrix are null, or the matrix is the wrong size, return
   if( !fname || !score ) return;
   if( rows <= 0 || cols <= 0 ) return;
   // try to open the file with the given name, return if unsuccessful
   FILE *fout = fopen( fname, "w" );
   if( !fout ) return;
   
   // variable used to calculate the hash function
   // http://www.cse.yorku.ca/~oz/hash.html
   // the starting value is a magic constant
   unsigned hash = 5381;

   // for all elements on one of the optimal paths
   bool loop = true;
   for( int i = rows-1, j = cols-1;  loop;  )
   {
      // print the current element
      fprintf( fout, "%d\n", el(score,adjcols, i,j) );
      // add the current element to the hash
      hash = ( ( hash<<5 ) + hash ) ^ el(score,adjcols, i,j);

      int max = INT_MIN;   // maximum value of the up, left and diagonal neighbouring elements
      int dir = '-';       // the current movement direction is unknown

      if( i > 0 && j > 0 && max < el(score,adjcols, i-1,j-1) ) { max = el(score,adjcols, i-1,j-1); dir = 'i'; }   // diagonal movement if possible
      if( i > 0          && max < el(score,adjcols, i-1,j  ) ) { max = el(score,adjcols, i-1,j  ); dir = 'u'; }   // up       movement if possible
      if(          j > 0 && max < el(score,adjcols, i  ,j-1) ) { max = el(score,adjcols, i  ,j-1); dir = 'l'; }   // left     movement if possible

      // move to the neighbour with the maximum value
      switch( dir )
      {
      case 'i': i--; j--; break;
      case 'u': i--;      break;
      case 'l':      j--; break;
      default:  loop = false; break;
      }
   }

   // close the file handle
   fclose( fout );
   // save the hash value
   *_hash = hash;
}





// main program
int main( int argc, char** argv )
{
   fflush(stdout);
   if( argc != 3 ) Usage( argv );

   // number of rows, number of columns and insdelcost
   int rows = atoi( argv[1] );
   int cols = rows;
   int insdelcost = atoi( argv[2] );
   // add the padding (zeroth row and column) to the matrix
   rows++; cols++;
   // if the number of columns is less than the number of rows, swap them
   if( cols < rows ) { int temp = cols; cols = rows; rows = temp; }

   // adjusted matrix dimensions
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile B size (in order to be evenly divisible)
   int adjrows = 1 + tileBy*ceil( float( rows-1 )/tileBy );
   int adjcols = 1 + tileBx*ceil( float( cols-1 )/tileBx );

   // allocate memory for the sequences which will be compared and the score matrix
   int* const seqX  = ( int* ) malloc( adjcols * sizeof( int ) );
   int* const seqY  = ( int* ) malloc( adjrows * sizeof( int ) );
   int* const score = ( int* ) malloc( adjrows*adjcols * sizeof( int ) );

   // if memory hasn't been allocated
   if( !seqX || !seqY || !score )
   {
      fprintf(stderr, "Error: memory allocation failed\n");
      fflush(stderr);

      // free allocated memory
      free( seqX ); free( seqY ); free( score );
      exit(1);
   }

   // seed the random generator
   unsigned int seed = time( NULL );
// unsigned int seed = 1605868371;
   srand( seed );

   // initialize the sequences A and B to random values in the range [1-10]
   // +   also initialize the padding with zeroes
   seqX[0] = 0;
   seqY[0] = 0;
   for( int j = 1; j < adjcols; j++ ) seqX[j] = ( j < cols ) ? 1 + rand() % 10 : 0;
   for( int i = 1; i < adjrows; i++ ) seqY[i] = ( i < rows ) ? 1 + rand() % 10 : 0;

   // variables for measuring the algorithms' cpu execution time and kernel execution time
   float htime = 0, ktime = 0;
   // variables for storing the calculation hashes
   unsigned hash1 = 10, hash2 = 20, hash3 = 30, hash4 = 40;

   // use the Needleman-Wunsch algorithm to find the optimal matching between the input vectors
   // +   sequential cpu implementation
   printf("Sequential cpu implementation:\n" );
   CpuSequential( seqX, seqY, score, rows, cols, adjrows, adjcols, insdelcost, &htime );
   Traceback( "nw.out1.txt", score, rows, cols, adjrows, adjcols, &hash1 );
   printf("   hash=%10u\n", hash1 );
   printf("   time=%9.6fs\n", htime );
   fflush(stdout);

   // +   parallel cpu implementation
   printf("Parallel cpu implementation:\n" );
   CpuParallel( seqX, seqY, score, rows, cols, adjrows, adjcols, insdelcost, &htime );
   Traceback( "nw.out2.txt", score, rows, cols, adjrows, adjcols, &hash2 );
   printf("   hash=%10u\n", hash2 );
   printf("   time=%9.6fs\n", htime );
   fflush(stdout);

   // +   parallel gpu implementation
   printf("Parallel gpu implementation 1:\n" );
   GpuParallel1( seqX, seqY, score, rows, cols, adjrows, adjcols, insdelcost, &htime, &ktime );
   Traceback( "nw.out3.txt", score, rows, cols, adjrows, adjcols, &hash3 );
   printf("   hash=%10u\n", hash3 );
   printf("   time=%9.6fs ktime=%9.6fs\n", htime, ktime );
   fflush(stdout);

   // +   parallel gpu implementation
   printf("Parallel gpu implementation 2:\n" );
   GpuParallel2( seqX, seqY, score, rows, cols, adjrows, adjcols, insdelcost, &htime, &ktime );
   Traceback( "nw.out4.txt", score, rows, cols, adjrows, adjcols, &hash4 );
   printf("   hash=%10u\n", hash4 );
   printf("   time=%9.6fs ktime=%9.6fs\n", htime, ktime );
   fflush(stdout);

   // +   compare the implementations
   if( hash1 == hash2 && hash2 == hash3 && hash3 == hash4 ) printf( "TEST PASSED\n" );
   else                                                     printf( "TEST FAILED\n" );
   fflush(stdout);

   // free allocated memory
   free( seqX ); free( seqY ); free( score );
}


