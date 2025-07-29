#ifndef ZVBDOT_VECTOR_MODEL_H
#define ZVBDOT_VECTOR_MODEL_H

// A model of the vector extension, so that target programs can be developed
// on a non-RISC-V machine.

#include "common.h"

struct vector_state {
  static const size_t mmax = 8;
  static const size_t vlen_min = 128;
  static const size_t vlen_max = 2048;

  size_t vlen;
  size_t sew;
  size_t vl;

  static const size_t nr = 32;
  uint64_t vregs[nr][vlen_max / 64];

  size_t loads;
  size_t stores;
  size_t moves;
  size_t matmuls;

public:
  void reset_stats()
  {
    loads = 0;
    stores = 0;
    moves = 0;
    matmuls = 0;
  }

  template<typename T, int lmul>
  size_t vlmax()
  {
    return vlen / (8 * sizeof(T)) * lmul;
  }

  template<typename in_t, int lmul>
  size_t vsetvl(size_t avl) {
    sew = 8 * sizeof(in_t);
    vl = std::min(avl, vlmax<in_t, lmul>());
    return vl;
  }

  uintptr_t mem_count(uintptr_t addr, uintptr_t len)
  {
    return (addr + len - 1) / (vlen / 8) - addr / (vlen / 8) + 1;
  }

  template<typename T>
  T& elt(int vreg, size_t n)
  {
    size_t vlen_in_elts = vlen / (8 * sizeof(T));
    T* vreg_within_group = (T*)vregs[vreg + n / vlen_in_elts];
    return vreg_within_group[n % vlen_in_elts];
  }

  template<typename T, int vreg>
  void load(const T* data) {
    assert(sizeof(T) * 8 == sew);
    for (size_t i = 0; i < vl; i++)
      elt<T>(vreg, i) = data[i];
    loads += mem_count((uintptr_t)data, vl * sizeof(T));
  }

  template<typename T, int vreg>
  void move(T value) {
    assert(sizeof(T) * 8 <= sew);
    elt<T>(vreg, 0) = value;
    moves++;
  }

  template<typename T, int vreg>
  void splat(T value) {
    assert(sizeof(T) * 8 == sew);
    for (size_t i = 0; i < vl; i++)
      elt<T>(vreg, i) = value;
    moves += mem_count(0, vl * sizeof(T));
  }

  template<typename T, int vreg, int vsrc>
  void move() {
    assert(sizeof(T) * 8 == sew);
    for (size_t i = 0; i < vl; i++)
      elt<T>(vreg, i) = elt<T>(vsrc, i);
    moves += mem_count(0, vl * sizeof(T));
  }

  template<typename T, int vreg>
  void vid() {
    assert(sizeof(T) * 8 == sew);
    for (size_t i = 0; i < vl; i++)
      elt<T>(vreg, i) = i;
    moves += mem_count(0, vl * sizeof(T));
  }

  template<typename T, int vreg>
  void store(T* data) {
    assert(sizeof(T) * 8 == sew);
    for (size_t i = 0; i < vl; i++)
      data[i] = elt<T>(vreg, i);
    stores += mem_count((uintptr_t)data, vl * sizeof(T));
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
    assert(sizeof(in_t) * 8 == sew);
    assert(b_reg % mmax == 0);
    assert(c_reg % std::max(size_t(1), 8 / vlmax<out_t, 1>()) == 0);
    assert(!masked || (a_reg != 0 && b_reg != 0 && c_reg != 0));

    for (size_t mi = 0; mi < mmax; mi++) {
      if (c_off && mi >= vlmax<out_t, 1>()) abort();
      size_t cr = c_reg + mi / vlmax<out_t, 1>();
      size_t ci = mi % vlmax<out_t, 1>();

      size_t mask_bit = mi + c_off;
      if (masked && !((elt<uint8_t>(0, mask_bit / 8) >> (mask_bit % 8)) & 1))
        continue;

      for (size_t ki = 0; ki < vl; ki++) {
        in_t a_elt = elt<in_t>(a_reg, ki);
        in_t b_elt = elt<in_t>(b_reg + mi, ki);
        elt<out_t>(cr, ci + c_off) += (out_t)a_elt * (out_t)b_elt;
      }
    }

    matmuls++;
  }
};

thread_local vector_state vstate;

#endif
