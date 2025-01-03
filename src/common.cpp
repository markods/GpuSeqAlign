#include "common.hpp"
#include <cuda_runtime.h>

// cuda status, used for getting the return status of cuda functions
thread_local cudaError_t cudaStatus = cudaSuccess;

void NwPrintVect(
    std::ostream &os,
    const int *const vect,
    const int len)
{
    FormatFlagsGuard fg{os, 4};

    for (int i = 0; i < len; i++)
    {
        std::cout << std::setw(4) << vect[i] << ",";
    }
    std::cout << std::endl;
}

void NwPrintMat(
    std::ostream &os,
    const int *const mat,
    const int rows,
    const int cols)
{
    FormatFlagsGuard fg{os, 4};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            os << std::setw(4) << el(mat, cols, i, j) << ',';
        }
        os << std::endl;
    }
}

void NwPrintTiledMat(
    std::ostream &os,
    const int *const mat,
    const int rows,
    const int cols,
    const int tileWid /*without header column*/,
    const int tileHei /*without header row*/)
{
    FormatFlagsGuard fg{os, 4};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            os << std::setw(4) << el(mat, cols, i, j) << ',';
            if ((j - 1) % tileWid == 0)
            {
                os << "   ";
            }
        }

        os << std::endl;
        if ((i - 1) % tileHei == 0)
        {
            os << "   ";
        }
    }
}

void NwPrintHdrMat(
    std::ostream &os,
    const int *const tileHdrMat,
    const int rows,
    const int cols,
    const int hdrLen)
{
    FormatFlagsGuard fg{os, 4};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < hdrLen; k++)
            {
                int kHdrElem = (i * cols + j) * hdrLen + k;
                os << std::setw(4) << tileHdrMat[kHdrElem] << ',';
            }
            os << " // " << i << " " << j << std::endl;
        }
    }
}