#ifndef ZVBDOT_VECTOR_NATIVE_H
#define ZVBDOT_VECTOR_NATIVE_H

// Support routines for V and Zvbdot extensions.

#include "common.h"

struct vector_state {
  static const size_t mmax = 8;

public:
  void reset() {}

  template<typename T, int lmul>
  size_t vlmax()
  {
    size_t vlenb;
    asm ("csrr %0, vlenb" : "=r" (vlenb));
    return vlenb * lmul / sizeof(T);
  }

  template<typename in_t, int lmul>
  size_t vsetvl(size_t avl) {
    constexpr int imm = (ilog2(sizeof(in_t)) << 3) + ilog2(lmul);

    size_t vl;
    asm volatile ("vsetvli %0, %1, e%2, m%3, ta, mu" : "=r" (vl) : "r" (avl), "I" (8 * sizeof(in_t)), "I" (lmul));
    return vl;
  }

  template<typename T, int vreg>
  void load(const T* data) {
    asm volatile ("vle%0.v v%1, (%2)" :: "I" (sizeof(T) * 8), "I" (vreg), "r" (data) : "memory");
  }

  template<typename T, int vreg>
  void move(T value) {
    asm volatile ("vmv.s.x v%0, %1" :: "I" (vreg), "r" (value) : "memory");
  }

  template<typename T, int vreg>
  void store(T* data) {
    asm volatile ("vse%0.v v%1, (%2)" :: "I" (sizeof(T) * 8), "I" (vreg), "r" (data) : "memory");
  }

  template<typename T, int vreg, int max_n, int lmul=1>
  void INLINE load_matrix(const T* B, size_t ldb, size_t n)
  {
    constexpr_for<0, max_n, 1U>([&](auto i) {
      if (i < n) load<T, vreg + i*lmul>(&B[i*ldb]);
    });
  }

  template<typename T, int vreg, int max_n, int lmul=1>
  void INLINE store_matrix(T* B, size_t ldb, size_t n)
  {
    constexpr_for<0, max_n, 1U>([&](auto i) {
      if (i < n) store<T, vreg + i*lmul>(&B[i*ldb]);
    });
  }

  template<typename in_t, typename out_t, int a_reg, int b_reg, int c_reg, int c_off, bool masked>
  void matmul() {
    bool in_float = (in_t)0 != (in_t)0.1;
    bool in_bfloat = in_float && pun_to<uint16_t, in_t>(in_t(1)) == (pun_to<uint32_t, float>(float(1)) >> 16);
    bool in_unsigned_int = !in_float && (in_t)0 < (in_t)-1;
    bool in_signed_int = !in_float && !in_unsigned_int;
    bool pretty = false;

    if (in_float && sizeof(out_t) == sizeof(in_t)) {
      if (!pretty)
        asm volatile (".insn r 0x77, 0x1, 0x2b*2+%3, x%0, x%1, x%2" :: "I" (c_reg), "I" (a_reg), "I" (b_reg + c_off / 8), "I" (!masked));
      else if (masked)
        asm volatile ("vfbdot.vv v%0, v%1, v%2, %3, v0.t" :: "I" (c_reg), "I" (a_reg), "I" (b_reg), "I" (c_off));
      else
        asm volatile ("vfbdot.vv v%0, v%1, v%2, %3" :: "I" (c_reg), "I" (a_reg), "I" (b_reg), "I" (c_off));
    } else if (in_float && sizeof(out_t) == 2 * sizeof(in_t)) {
      if (!pretty)
        asm volatile (".insn r 0x77, 0x1, 0x2c*2+%3, x%0, x%1, x%2" :: "I" (c_reg), "I" (a_reg), "I" (b_reg + c_off / 8), "I" (!masked));
      else if (masked)
        asm volatile ("vfwbdot.vv v%0, v%1, v%2, %3, v0.t" :: "I" (c_reg), "I" (a_reg), "I" (b_reg), "I" (c_off));
      else
        asm volatile ("vfwbdot.vv v%0, v%1, v%2, %3" :: "I" (c_reg), "I" (a_reg), "I" (b_reg), "I" (c_off));
    } else if (in_unsigned_int && sizeof(out_t) == 4 * sizeof(in_t)) {
      if (!pretty)
        asm volatile (".insn r 0x77, 0x0, 0x2e*2+%3, x%0, x%1, x%2" :: "I" (c_reg), "I" (a_reg), "I" (b_reg + c_off / 8), "I" (!masked));
      else if (masked)
        asm volatile ("vqbdotu.vv v%0, v%1, v%2, %3, v0.t" :: "I" (c_reg), "I" (a_reg), "I" (b_reg), "I" (c_off));
      else
        asm volatile ("vqbdotu.vv v%0, v%1, v%2, %3" :: "I" (c_reg), "I" (a_reg), "I" (b_reg), "I" (c_off));
    } else if (in_signed_int && sizeof(out_t) == 4 * sizeof(in_t)) {
      if (!pretty)
        asm volatile (".insn r 0x77, 0x0, 0x2f*2+%3, x%0, x%1, x%2" :: "I" (c_reg), "I" (a_reg), "I" (b_reg + c_off / 8), "I" (!masked));
      else if (masked)
        asm volatile ("vqbdots.vv v%0, v%1, v%2, %3, v0.t" :: "I" (c_reg), "I" (a_reg), "I" (b_reg), "I" (c_off));
      else
        asm volatile ("vqbdots.vv v%0, v%1, v%2, %3" :: "I" (c_reg), "I" (a_reg), "I" (b_reg), "I" (c_off));
    } else {
      abort();
    }
  }
};

thread_local vector_state vstate;

static inline size_t rdinstret()
{
  size_t res;
  asm volatile ("rdinstret %0" : "=r" (res));
  return res;
}

#endif
