#ifndef INCLUDE_PRINT_MAT_HPP
#define INCLUDE_PRINT_MAT_HPP

#include "fmt_guard.hpp"
#include "run_types.hpp"
#include <iostream>

template <typename T>
void NwPrintVect(
    std::ostream& os,
    const T* const vect,
    const size_t len)
{
    FormatFlagsGuard fg {os, 4};

    for (size_t i = 0; i < len; i++)
    {
        os << std::setw(4) << vect[i] << ' ';
    }
}

template <typename T>
void NwPrintMat(
    std::ostream& os,
    const T* const mat,
    const size_t rows,
    const size_t cols)
{
    FormatFlagsGuard fg {os, 4};

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            os << std::setw(4) << el(mat, cols, i, j) << ' ';
        }
        os << '\n';
    }
}

template <typename T>
void NwPrintTiledMat(
    std::ostream& os,
    const T* const mat,
    const size_t rows,
    const size_t cols,
    const size_t tileWid /*without header column*/,
    const size_t tileHei /*without header row*/)
{
    FormatFlagsGuard fg {os, 4};

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            os << std::setw(4) << el(mat, cols, i, j) << ' ';
            if ((j - 1) % tileWid == 0)
            {
                os << "   ";
            }
        }

        os << '\n';
        if ((i - 1) % tileHei == 0)
        {
            os << '\n';
        }
    }
}

template <typename T>
void NwPrintHdrMat(
    std::ostream& os,
    const T* const tileHdrMat,
    const size_t rows,
    const size_t cols,
    const size_t hdrLen)
{
    FormatFlagsGuard fg {os, 4};

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            for (size_t k = 0; k < hdrLen; k++)
            {
                int kHdrElem = (i * cols + j) * hdrLen + k;
                os << std::setw(4) << tileHdrMat[kHdrElem] << ' ';
            }
            os << " // " << i << ' ' << j << '\n';
        }
    }
}

#endif // INCLUDE_PRINT_MAT_HPP
