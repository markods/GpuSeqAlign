#ifndef INCLUDE_MATH_HPP
#define INCLUDE_MATH_HPP

// Get the specified element from the given linearized matrix.
#define el(mat, cols, i, j) (mat[(cols) * (i) + (j)])

template <typename T>
inline const T& min2(const T& a, const T& b) noexcept
{
    return (a <= b) ? a : b;
}
template <typename T>
inline const T& max2(const T& a, const T& b) noexcept
{
    return (a >= b) ? a : b;
}
template <typename T>
inline const T& min3(const T& a, const T& b, const T& c) noexcept
{
    return (a <= b) ? ((a <= c) ? a : c) : ((b <= c) ? b : c);
}
template <typename T>
inline const T& max3(const T& a, const T& b, const T& c) noexcept
{
    return (a >= b) ? ((a >= c) ? a : c) : ((b >= c) ? b : c);
}

#endif // INCLUDE_MATH_HPP
