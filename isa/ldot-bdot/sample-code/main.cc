#ifdef __riscv
# include "vector-native.h"
#else
# include "vector-model.h"
#endif

#include "matrix-routines.h"
#include "test-routines.h"

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
#ifndef __riscv
  benchmark<bf16, float>();
  benchmark<double, double>();
#endif

  quick_test<int8_t, int32_t>();
  quick_test<int16_t, int64_t>();
#ifndef __riscv
  quick_test<bf16, float>();
  quick_test<double, double>();
#endif

#ifndef __riscv
  exhaustive_test<int8_t, int32_t>();
  exhaustive_test<bf16, float>();
  exhaustive_test<double, double>();
#endif

  return 0;
}
