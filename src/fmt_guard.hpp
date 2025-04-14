#ifndef INCLUDE_FORMAT_FLAGS_GUARD_HPP
#define INCLUDE_FORMAT_FLAGS_GUARD_HPP

#include <iomanip>
#include <iostream>

// Iostream format flags guard.
template <typename T>
class FormatFlagsGuard
{
public:
    FormatFlagsGuard(T& stream, int fwidth = 1, char ffill = ' ')
        : _stream {stream},
          _fflags {stream.flags()},
          _fwidth {stream.width(fwidth)},
          _ffill {stream.fill(ffill)},
          _fprecision {stream.precision()},
          _fexceptions {stream.exceptions()}
    { }
    ~FormatFlagsGuard()
    {
        restore();
    }

    void restore()
    {
        _stream.flags(_fflags);
        _stream.width(_fwidth);
        _stream.fill(_ffill);
        _stream.precision(_fprecision);
        _stream.exceptions(_fexceptions);
    }

private:
    T& _stream;
    std::ios_base::fmtflags _fflags;
    std::streamsize _fwidth;
    char _ffill;
    std::streamsize _fprecision;
    std::ios_base::iostate _fexceptions;
};

#endif // INCLUDE_FORMAT_FLAGS_GUARD_HPP
