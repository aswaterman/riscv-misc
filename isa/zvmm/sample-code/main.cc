#ifdef __riscv
# include "vector-native.h"
#else
# include "vector-model.h"
#endif

#include "matrix-routines.h"

#include <cstdio>
#include <cstdlib>
#include <typeinfo>

template<typename in_t, bool trans>
void NOINLINE populate_matrix(size_t m, size_t n, int scale, in_t* A)
{
  for (size_t mi = 0; mi < m; mi++) {
    for (size_t ni = 0; ni < n; ni++) {
      if (trans)
        A[ni*m+mi] = ssize_t(mi + 1) + scale * ssize_t(ni + 1);
      else
        A[mi*n+ni] = ssize_t(mi + 1) + scale * ssize_t(ni + 1);
    }
  }
}

template<typename in_t, typename out_t>
void NOINLINE test(size_t m, size_t n, size_t k)
{
  if (int64_t(in_t(m + k)) != int64_t(m + k) || int64_t(1 - int64_t(in_t(k))) != 1 - int64_t(k) ||
      int64_t(in_t(k + n)) != int64_t(k + n) || int64_t(1 - int64_t(in_t(n))) != 1 - int64_t(n)) {
    printf("%zu %zu %zu will overflow for sizeof(in_t) = %zu\n", m, n, k, sizeof(in_t));
    abort();
  }

  in_t A[m*k];
  in_t B[k*n];
  out_t C[m*n];

  populate_matrix<in_t, false>(m, k, 1, A);

  populate_matrix<in_t, true>(k, n, -1, B);

  memset(C, 0, sizeof(C));

  matmul_abt<in_t, out_t>(m, n, k, A, B, C);
  matmul_abt<in_t, out_t>(m, n, k, A, B, C);

  for (size_t mi = 0; mi < m; mi++) {
    for (size_t ni = 0; ni < n; ni++) {
      // A(i,j) = i + j
      // B(i,j) = i - j
      // C(i,j) = -N*i*j + (i-j)*N*(N+1)/2 + N*(N+1)*(2N+1)/6
      int64_t expected = -int64_t(k)*(mi+1)*(ni+1) + int64_t((mi-ni)*k*(k+1))/2 + int64_t(k)*(k+1)*(2*k+1)/6;
      if (C[mi*n+ni] != 2 * expected) {
        printf("%zu %zu %zu C[%zu][%zu] = %ld, expected %ld\n", m, n, k, mi, ni, (long)C[mi*n+ni], (long)(2 * expected));
        abort();
      }
    }
  }
}

template<typename in_t, typename out_t>
void test()
{
  printf("test %s %s\n", typeid(in_t).name(), typeid(out_t).name());
  for (int i = 23; i <= 48; i++) {
    for (int j = 1; j <= 48; j++) {
      for (int k = 2; k <= 48; k++) {
        test<in_t, out_t>(i, j, k);
      }
    }
  }
}

template<typename in_t, typename out_t>
void benchmark()
{
  // Choose M and N to be the LCM of the various register-block sizes to
  // eliminate fringe-case overhead, and chose K to be very large to
  // minimize tile-transition cost.  This isn't necessary for correctness
  // but is meant to emphasize the inner-loop performance.
  size_t m = 15 * 23 * 11, n = 16, k = 4096;

  in_t* A = (in_t*)aligned_alloc(vstate.vlmax<char, 1>(), sizeof(in_t) * m * k);
  in_t* B = (in_t*)aligned_alloc(vstate.vlmax<char, 1>(), sizeof(in_t) * k * n);
  out_t* C = (out_t*)aligned_alloc(vstate.vlmax<char, 1>(), sizeof(out_t) * m * n);

  populate_matrix<in_t, false>(m, k, 0, A);
  populate_matrix<in_t, false>(k, n, 0, B);
  populate_matrix<out_t, false>(m, n, 0, C);

  printf("benchmark %s %s %zu %zu %zu\n", typeid(in_t).name(), typeid(out_t).name(), m, n, k);


#ifdef __riscv
  size_t insns = rdinstret();
#else
  vstate.reset_stats();
#endif

  matmul_abt<in_t, out_t>(m, n, k, A, B, C);

#ifdef __riscv
  insns = rdinstret() - insns;
  printf("%zu insns\n", insns);
#else
  printf("vlen-bit matmuls %zu %f\n", vstate.matmuls, vstate.matmuls / double(m * n * k));
  printf("vlen-bit loads %zu %f\n", vstate.loads, vstate.loads / double(vstate.matmuls));
  printf("vlen-bit stores %zu %f\n", vstate.stores, vstate.stores / double(vstate.matmuls));
  printf("vlen-bit moves %zu %f\n", vstate.moves, vstate.moves / double(vstate.matmuls));
#endif

  free(A);
  free(B);
  free(C);
}

int main(int argc, char** argv)
{
#ifndef __riscv
  if (argc != 2) {
    fprintf(stderr, "supply VLEN as first argument\n");
    abort();
  }

  vstate.vlen = strtoul(argv[1], nullptr, 10);

  if ((vstate.vlen & (vstate.vlen - 1))
      || vstate.vlen < vstate.vlen_min
      || vstate.vlen > vstate.vlen_max) {
    fprintf(stderr, "VLEN must be a power of 2 in the range [%zu, %zu]\n",
            vstate.vlen_min, vstate.vlen_max);
    abort();
  }
#endif

  benchmark<int8_t, int32_t>();
  benchmark<bf16, int32_t>();
  benchmark<double, double>();

  test<int8_t, int32_t>();
  test<bf16, float>();
  test<double, double>();

  return 0;
}
