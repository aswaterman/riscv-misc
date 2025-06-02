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
