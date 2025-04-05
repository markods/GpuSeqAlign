#ifndef INCLUDE_FORMAT_FLAGS_GUARD_HPP
#define INCLUDE_FORMAT_FLAGS_GUARD_HPP

#include <iomanip>
#include <iostream>

// iostream format flags guard
template <typename T>
class FormatFlagsGuard
{
public:
    FormatFlagsGuard(T& stream, int fwidth = 1, char ffill = ' ')
        : _stream {stream}
    {
        // backup format flags and set the fill character and width
        _fflags = _stream.flags();
        _fwidth = _stream.width(fwidth);
        _ffill = _stream.fill(ffill);
    }

    ~FormatFlagsGuard()
    {
        restore();
    }

    void restore()
    {
        // restore the format flags, fill character and width
        _stream.flags(_fflags);
        _stream.width(_fwidth);
        _stream.fill(_ffill);
    }

private:
    T& _stream;
    std::ios_base::fmtflags _fflags;
    std::streamsize _fwidth;
    char _ffill;
};

#endif // INCLUDE_FORMAT_FLAGS_GUARD_HPP
