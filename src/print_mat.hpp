#ifndef INCLUDE_PRINT_MAT_HPP
#define INCLUDE_PRINT_MAT_HPP

#include "fmt_guard.hpp"
#include "run_types.hpp"
#include <iostream>

template <typename T>
void NwPrintVect(
    std::ostream& os,
    const T* const vect,
    const T len)
{
    FormatFlagsGuard fg {os, 4};

    for (int i = 0; i < len; i++)
    {
        os << std::setw(4) << vect[i] << ",";
    }
    os << "\n";
}

template <typename T>
void NwPrintMat(
    std::ostream& os,
    const T* const mat,
    const T rows,
    const T cols)
{
    FormatFlagsGuard fg {os, 4};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            os << std::setw(4) << el(mat, cols, i, j) << ',';
        }
        os << "\n";
    }
}

template <typename T>
void NwPrintTiledMat(
    std::ostream& os,
    const T* const mat,
    const T rows,
    const T cols,
    const T tileWid /*without header column*/,
    const T tileHei /*without header row*/)
{
    FormatFlagsGuard fg {os, 4};

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

        os << "\n";
        if ((i - 1) % tileHei == 0)
        {
            os << "   ";
        }
    }
}

template <typename T>
void NwPrintHdrMat(
    std::ostream& os,
    const T* const tileHdrMat,
    const T rows,
    const T cols,
    const T hdrLen)
{
    FormatFlagsGuard fg {os, 4};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < hdrLen; k++)
            {
                int kHdrElem = (i * cols + j) * hdrLen + k;
                os << std::setw(4) << tileHdrMat[kHdrElem] << ',';
            }
            os << " // " << i << " " << j << "\n";
        }
    }
}

#endif // INCLUDE_PRINT_MAT_HPP
