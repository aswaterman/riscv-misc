// Inner loop for MxKx8 register tiles
template<typename in_t, typename out_t, size_t m_unroll, int lmul, bool fringe>
void INLINE matmul_abt_8_body(size_t n_in, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const int a_reg = 31, b_reg = fringe ? 24 : 0, c_reg = fringe ? lmul : 8, mask = 0;
  const size_t ml = fringe ? n_in : vector_state::mmax;
  vstate.vsetvl<out_t, lmul>(ml);
  vstate.load_matrix<out_t, c_reg, m_unroll, lmul>(C, ldc, m_unroll);

  if (fringe) {
    vstate.move<uint8_t, mask>((1U << ml) - 1);
  }

  do {
    size_t vl = vstate.vsetvl<in_t, 1>(k);

    vstate.load_matrix<in_t, b_reg, vector_state::mmax - fringe>(B, ldb, ml);

    constexpr_for<0, m_unroll, 1>([&](auto i) {
      vstate.load<in_t, a_reg>(A + i*lda);
      vstate.matmul<in_t, out_t, a_reg, b_reg, c_reg+i*lmul, 0, fringe>();
    });

    A += vl;
    B += vl;
    k -= vl;
  } while (k);

  vstate.vsetvl<out_t, lmul>(ml);
  vstate.store_matrix<out_t, c_reg, m_unroll, lmul>(C, ldc, m_unroll);
}

// C += A * B^T using 1xKx8 register tiles
template<typename in_t, typename out_t>
void NOINLINE matmul_abt_1_8(size_t m, size_t n, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const size_t m_unroll = 1, n_unroll = 8;

  for (size_t mi = 0; mi < m; mi += m_unroll) {
    if (n % n_unroll)
      matmul_abt_8_body<in_t, out_t, m_unroll, 4, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_8_body<in_t, out_t, m_unroll, 4, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
    }
  }
}

// C += A * B^T using 5xKx8 register tiles
template<typename in_t, typename out_t>
void NOINLINE matmul_abt_5_8(size_t m, size_t n, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const size_t m_unroll = 5, n_unroll = 8;

  if (m % m_unroll)
    matmul_abt_1_8(m % m_unroll, n, k, A, lda, B, ldb, C, ldc);

  for (size_t mi = m % m_unroll; mi < m; mi += m_unroll) {
    if (n % n_unroll)
      matmul_abt_8_body<in_t, out_t, m_unroll, 4, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_8_body<in_t, out_t, m_unroll, 4, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
    }
  }
}

// C += A * B^T using 11xKx8 register tiles
template<typename in_t, typename out_t>
void NOINLINE matmul_abt_11_8(size_t m, size_t n, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const size_t m_unroll = 11, n_unroll = 8;

  if (m % m_unroll)
    matmul_abt_1_8(m % m_unroll, n, k, A, lda, B, ldb, C, ldc);

  for (size_t mi = m % m_unroll; mi < m; mi += m_unroll) {
    if (n % n_unroll)
      matmul_abt_8_body<in_t, out_t, m_unroll, 2, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_8_body<in_t, out_t, m_unroll, 2, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
    }
  }
}

// C += A * B^T using 23xKx8 register tiles
template<typename in_t, typename out_t>
void NOINLINE matmul_abt_23_8(size_t m, size_t n, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const size_t m_unroll = 23, n_unroll = 8;

  if (m % m_unroll)
    matmul_abt_1_8(m % m_unroll, n, k, A, lda, B, ldb, C, ldc);

  for (size_t mi = m % m_unroll; mi < m; mi += m_unroll) {
    if (n % n_unroll)
      matmul_abt_8_body<in_t, out_t, m_unroll, 1, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_8_body<in_t, out_t, m_unroll, 1, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
    }
  }
}

// Inner loop for 15xKx16 register tiles
template<typename in_t, typename out_t, bool fringe>
void INLINE matmul_abt_15_16_body(size_t n_in, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const int m_unroll = 15, n_unroll = 16;
  const size_t n = fringe ? n_in : n_unroll;
  const int b_reg = fringe ? 16 : 0, c_reg = fringe ? 1 : 16, a_reg = 31, mask = 0;

  vstate.vsetvl<out_t, 1>(n);
  vstate.load_matrix<out_t, c_reg, m_unroll>(C, ldc, m_unroll);

  if (fringe) {
    assert(sizeof(out_t) * 8 >= n_unroll);
    vstate.move<uint16_t, mask>((1U << n) - 1);
  }

  do {
    size_t vl = vstate.vsetvl<in_t, 1>(k);
    vstate.load_matrix<in_t, b_reg, n_unroll - fringe>(B, ldb, n);

    constexpr_for<0, m_unroll, 1>([&](auto i) {
      vstate.load<in_t, a_reg>(A + i*lda);
      vstate.matmul<in_t, out_t, a_reg, b_reg, c_reg+i, 0, fringe>();
      vstate.matmul<in_t, out_t, a_reg, b_reg+8, c_reg+i, 8, fringe>();
    });

    A += vl;
    B += vl;
    k -= vl;
  } while (k);

  vstate.vsetvl<out_t, 1>(n);
  vstate.store_matrix<out_t, c_reg, m_unroll>(C, ldc, m_unroll);
}

// C += A * B^T using 15xKx16 register tiles
template<typename in_t, typename out_t>
void NOINLINE matmul_abt_15_16(size_t m, size_t n, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const int m_unroll = 15, n_unroll = 16;

  if (m % m_unroll)
    matmul_abt_1_8(m % m_unroll, n, k, A, lda, B, ldb, C, ldc);

  for (size_t mi = m % m_unroll; mi < m; mi += m_unroll) {
    if (n % n_unroll)
      matmul_abt_15_16_body<in_t, out_t, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_15_16_body<in_t, out_t, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
    }
  }
}

// C += A * B^T
// Chooses register-tiling strategy based on VLEN and AEW
template<typename in_t, typename out_t>
void NOINLINE matmul_abt(size_t m, size_t n, size_t k, const in_t* A, const in_t* B, out_t* C)
{
  if (vstate.vlmax<out_t, 1>() >= 16)
    return matmul_abt_15_16(m, n, k, A, k, B, k, C, n);
  else if (vstate.vlmax<out_t, 1>() >= 8)
    return matmul_abt_23_8(m, n, k, A, k, B, k, C, n);
  else if (vstate.vlmax<out_t, 1>() >= 4)
    return matmul_abt_11_8(m, n, k, A, k, B, k, C, n);
  else if (vstate.vlmax<out_t, 1>() >= 2)
    return matmul_abt_5_8(m, n, k, A, k, B, k, C, n);
  else
    return matmul_abt_1_8(m, n, k, A, k, B, k, C, n);
}

template<typename in_t, typename out_t>
void NOINLINE mat_vec_mul_trans_simple(size_t m, size_t n, const in_t* A, const in_t* X, out_t* Y)
{
  const size_t n_unroll = 8;

  if (n % n_unroll)
    matmul_abt_8_body<in_t, out_t, 1, 4, true>(n % n_unroll, m, X, 0, A, m, Y, 0);

  for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
    matmul_abt_8_body<in_t, out_t, 1, 4, false>(n_unroll, m, X, 0, A + ni*m, m, Y + ni, 0);
  }
}

template<typename in_t, typename out_t>
void INLINE mat_vec_mul_trans_body(size_t m, const in_t* A, size_t lda, const in_t* X, out_t* Y)
{
  const int unroll = 4, y_emul  = 4;
  const int x_reg = 31, a_reg = 16, y_reg = 0;

  vstate.vsetvl<out_t, y_emul>(vector_state::mmax);
  vstate.load_matrix<out_t, y_reg, unroll, y_emul>(Y, vector_state::mmax, unroll);

  do {
    size_t vl = vstate.vsetvl<in_t, 1>(m);

    vstate.load<in_t, x_reg>(X);

    constexpr_for<0, 4, 1>([&](auto i) {
      vstate.load_matrix<in_t, a_reg, vector_state::mmax>(A + i * vector_state::mmax * lda, lda, vector_state::mmax);
      vstate.matmul<in_t, out_t, x_reg, a_reg, y_reg + i * y_emul, 0, false>();
    });

    X += vl;
    A += vl;
    m -= vl;
  } while (m);

  vstate.vsetvl<out_t, y_emul>(vector_state::mmax);
  vstate.store_matrix<out_t, y_reg, unroll, y_emul>(Y, vector_state::mmax, unroll);
}

// Y += A^T * X
template<typename in_t, typename out_t>
void NOINLINE mat_vec_mul_trans(size_t m, size_t n, const in_t* A, const in_t* X, out_t* Y)
{
  const size_t n_unroll = 32;

  if (n % n_unroll)
    mat_vec_mul_trans_simple(m, n % n_unroll, A, X, Y);

  for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
    mat_vec_mul_trans_body<in_t, out_t>(m, A + ni*m, m, X, Y + ni);
  }
}

template<typename in_t, typename out_t, size_t n_unroll, int lmul, bool fringe>
void INLINE mat_vec_mul_trans_rvv_body(size_t m, size_t n, const in_t* A, size_t lda, const in_t* X, out_t* Y)
{
  bool in_float = (in_t)0 != (in_t)0.1;
  bool in_unsigned_int = !in_float && (in_t)0 < (in_t)-1;
  bool in_signed_int = !in_float && !in_unsigned_int;

  const int out_emul = lmul * (sizeof(out_t) / sizeof(in_t));
  const int y_reg = 0, x_reg = y_reg + n_unroll * out_emul, a_reg = x_reg + lmul, tmp_reg = a_reg + lmul;

  vstate.vsetvl<out_t, out_emul>(-1);

  constexpr_for<0, n_unroll, 1>([&](auto i) {
    const int cur_y_reg = y_reg + i * out_emul;

    vstate.splat<out_t, cur_y_reg>(0);
  });

  do {
    size_t vl = vstate.vsetvl<in_t, lmul>(m);
    vstate.load<in_t, x_reg>(X);

    constexpr_for<0, n_unroll, 1>([&](auto i) {
      const int cur_y_reg = y_reg + i * out_emul;

      vstate.load<in_t, a_reg>(A + i*lda);

      if (sizeof(out_t) == sizeof(in_t) * 4) {
        #define emulate_fma(intermediate_t) \
          vstate.mul<in_t, intermediate_t, x_reg, a_reg, tmp_reg>(); \
          vstate.vsetvl<intermediate_t, lmul * 2>(vl); \
          vstate.acc<intermediate_t, out_t, tmp_reg, cur_y_reg>()

        if (in_signed_int && sizeof(in_t) == 1) {
          emulate_fma(int16_t);
        } else if (in_unsigned_int && sizeof(in_t) == 1) {
          emulate_fma(uint16_t);
        } else {
          abort();
        }

        if (i != n_unroll - 1)
          vstate.vsetvl<in_t, lmul>(vl);

        #undef emulate_fma
      } else {
        vstate.fma<in_t, out_t, x_reg, a_reg, cur_y_reg>();
      }
    });

    X += vl;
    A += vl;
    m -= vl;
  } while (m);

  constexpr_for<0, n_unroll, 1>([&](auto i) {
    const int cur_y_reg = y_reg + i * out_emul;

    vstate.vsetvl<out_t, 1>(1);
    vstate.load<out_t, x_reg>(Y + i);

    vstate.vsetvl<out_t, out_emul>(-1);
    vstate.redsum<out_t, cur_y_reg, x_reg>();

    vstate.vsetvl<out_t, 1>(1);
    vstate.store<out_t, x_reg>(Y + i);
  });
}

// Y += A^T * X
template<typename in_t, typename out_t>
void NOINLINE mat_vec_mul_trans_rvv(size_t m, size_t n, const in_t* A, const in_t* X, out_t* Y)
{
  const int ratio = sizeof(out_t) / sizeof(in_t);
  const int lmul = ratio <= 2 ? 2 : 1;
  const size_t n_unroll = 28 / (lmul * ratio);

  for (size_t ni = 0; ni < n % n_unroll; ni++)
    mat_vec_mul_trans_rvv_body<in_t, out_t, 1, lmul, true>(m, 1, A + ni*m, m, X, Y + ni);

  for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
    mat_vec_mul_trans_rvv_body<in_t, out_t, n_unroll, lmul, false>(m, n_unroll, A + ni*m, m, X, Y + ni);
  }
}
