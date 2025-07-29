#ifndef ZVBDOT_COMMON_H
#define ZVBDOT_COMMON_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#ifdef __GNUC__
# define INLINE __attribute__((always_inline)) inline
# define NOINLINE __attribute__((noinline))
#else
# define INLINE inline
# define NOINLINE
#endif

class bf16 {
  uint16_t x;
  union u { float f; uint16_t x[2]; };

 public:
  bf16() : x(0) {}
  bf16(float f) : x((u {.f = f}).x[1]) {}
  operator float() const { return u {.x = {0, x}}.f; }
};

template<typename T, typename U>
T pun_to(U a)
{
  T res;
  memcpy(&res, &a, std::min(sizeof(T), sizeof(U)));
  return res;
}

template <typename T>
T register_barrier(T x)
{
#ifdef __GNUC__
  asm ("" : "+r" (x));
#endif
  return x;
}

static inline constexpr unsigned ilog2(uint64_t x) {
  unsigned res = 0;
  while (x > 1) {
    x >>= 1;
    res++;
  }
  return res;
}

template <auto Start, auto End, auto Inc, class F>
constexpr void INLINE constexpr_for(F&& f)
{
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

#endif
