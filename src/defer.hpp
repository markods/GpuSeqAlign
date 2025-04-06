#ifndef INCLUDE_DEFER_HPP
#define INCLUDE_DEFER_HPP

#include <utility>

#define ZIG_TRY(exp_res, expr)  \
    do                      \
    {                       \
        auto res = (expr);  \
        if (res != exp_res) \
        {                   \
            return res;     \
        }                   \
    } while (false)

// defer execution to scope exit
template <typename F>
class Defer
{
    // static_assert(std::is_nothrow_invocable_r_v<void, F>, "F must be a callable type with signature void() noexcept");

public:
    explicit Defer(F _func) noexcept
        : func {std::move(_func)}, active {true}
    { }

    Defer(const Defer&) = delete;
    Defer& operator=(const Defer&) = delete;

    Defer(Defer&& other) noexcept
        : func {std::move(other.func)}, active {other.active}
    {
        // The moved-from object must not run the function.
        other.active = false;
    }
    Defer& operator=(Defer&& other) noexcept
    {
        if (this != &other)
        {
            func = std::move(other.func);
            active = other.active;
            other.active = false;
        }
        return *this;
    }

    void operator()() noexcept
    {
        doDefer();
    }

    ~Defer() noexcept
    {
        doDefer();
    }

private:
    void doDefer() noexcept
    {
        if (active)
        {
            active = false;
            func();
        }
    }

private:
    F func;
    bool active;
};

template <typename F>
Defer<F> make_defer(F _func)
{
    return Defer<F>(std::move(_func));
}

#endif // INCLUDE_DEFER_HPP
