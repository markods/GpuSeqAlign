#ifndef INCLUDE_CROSS_PLATFORM_HPP_
#define INCLUDE_CROSS_PLATFORM_HPP_

#if defined(_MSC_VER)
    #define BEGIN_DISABLE_WARNINGS \
        __pragma(warning(push, 0))

    #define END_DISABLE_WARNINGS \
        __pragma(warning(pop))

#elif defined(__clang__)
    #define BEGIN_DISABLE_WARNINGS                      \
        _Pragma("clang diagnostic push")                \
        _Pragma("clang diagnostic ignored \"-Wall\"")   \
        _Pragma("clang diagnostic ignored \"-Wextra\"") \
        _Pragma("clang diagnostic ignored \"-Wpedantic\"")

    #define END_DISABLE_WARNINGS \
        _Pragma("clang diagnostic pop")

#elif defined(__GNUC__)
    #define BEGIN_DISABLE_WARNINGS                      \
        _Pragma("GCC diagnostic push")                  \
        _Pragma("GCC diagnostic ignored \"-Wall\"")     \
        _Pragma("GCC diagnostic ignored \"-Wextra\"")   \
        _Pragma("GCC diagnostic ignored \"-Wpedantic\"")

    #define END_DISABLE_WARNINGS \
        _Pragma("GCC diagnostic pop")

#else
    #error "error: unknown compiler"

#endif

#endif // INCLUDE_CROSS_PLATFORM_HPP_
