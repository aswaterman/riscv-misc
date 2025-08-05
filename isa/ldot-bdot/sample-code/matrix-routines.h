// Inner loop for MxKx8 register tiles
template<typename in_t, typename out_t, size_t m_unroll, int lmul, int interleave, bool fringe>
void NOINLINE matmul_abt_8_body(size_t n_in, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  assert((lmul == 1 || interleave == 1) && m_unroll % interleave == 0);

  const bool schedule_a = lmul > 1 || (m_unroll / interleave) < 23;
  const int a1_reg = 31, b_reg = fringe ? 24 : 0, c_reg = fringe ? lmul : 8, mask = 0;
  const int a_reg = !schedule_a ? a1_reg : (fringe ? (lmul > 1 ? c_reg : b_reg) : a1_reg) - 1;
  const int ap_reg = (m_unroll / interleave) % 2 ? a1_reg : a_reg;
  const bool pipeline_a = schedule_a && !fringe && ap_reg == a_reg;
  const size_t ml = fringe ? n_in : vector_state::mmax;

  typedef uint32_t mask_t;
  assert(vector_state::mmax * interleave <= 8 * sizeof(mask_t));

  vstate.vsetvl<out_t, lmul>(ml);
  if (interleave == 1) {
    vstate.load_matrix<out_t, c_reg, m_unroll / interleave, lmul>(register_barrier(C), ldc * interleave, m_unroll / interleave);
  } else {
    for (int i = 0; i < interleave; i++) {
      size_t offset = i * vector_state::mmax;

      if (sizeof(mask_t) > sizeof(out_t))
        vstate.vsetvl<mask_t, 1>(1);
      vstate.move<mask_t, mask>(mask_t(-1) << offset);

      auto base = register_barrier(C - offset + ldc * i);
      vstate.vsetvl<out_t, 1>(ml + offset);
      vstate.load_matrix<out_t, c_reg, m_unroll / interleave, 1, true>(base, ldc * interleave, m_unroll / interleave);
    }
  }

  if (fringe) {
    if (sizeof(mask_t) > sizeof(out_t))
      vstate.vsetvl<mask_t, 1>(1);

    mask_t m = 0;
    for (int i = 0; i < interleave; i++)
      m |= ((mask_t(1) << ml) - 1) << (i * vector_state::mmax);
    vstate.move<mask_t, mask>(m);
  }

  if (pipeline_a) {
    vstate.vsetvl<in_t, 1>(k);
    vstate.load<in_t, ap_reg>(A);
  }

  do {
    size_t vl = vstate.vsetvl<in_t, 1, true>(k);
    k -= vl;

    if (pipeline_a && ap_reg != a_reg)
      vstate.move<in_t, a_reg, ap_reg>();

    vstate.load_matrix<in_t, b_reg, vector_state::mmax - fringe>(B, ldb, ml);

    constexpr_for<0, m_unroll, 1>([&](auto i) {
      if (!pipeline_a && (i == 0 || !schedule_a))
        vstate.load<in_t, a_reg>(A + i*lda);
      if (i != m_unroll-1 && schedule_a)
        vstate.load<in_t, i % 2 ? a_reg : a1_reg>(A + (i+1)*lda);
      if (i == m_unroll-1 && pipeline_a && k) {
        vstate.load<in_t, ap_reg>(A + vl);
      }

      vstate.matmul<in_t, out_t, i % 2 && schedule_a ? a1_reg : a_reg, b_reg, c_reg + (i / interleave) * lmul, vector_state::mmax * (i % interleave), fringe>();
    });

    A += vl;
    B += vl;
  } while (k);

  vstate.vsetvl<out_t, lmul>(ml);
  if (interleave == 1) {
    vstate.store_matrix<out_t, c_reg, m_unroll / interleave, lmul>(register_barrier(C), ldc * interleave, m_unroll / interleave);
  } else {
    for (int i = 0; i < interleave; i++) {
      size_t offset = i * vector_state::mmax;

      if (sizeof(mask_t) > sizeof(out_t))
        vstate.vsetvl<mask_t, 1>(1);
      vstate.move<mask_t, mask>(mask_t(-1) << offset);

      auto base = register_barrier(C - offset + ldc * i);
      vstate.vsetvl<out_t, lmul>(ml + offset);
      vstate.store_matrix<out_t, c_reg, m_unroll / interleave, lmul, true>(base, ldc * interleave, m_unroll / interleave);
    }
  }
}

// C += A * B^T using 1xKx8 register tiles
template<typename in_t, typename out_t>
void NOINLINE matmul_abt_1_8(size_t m, size_t n, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const size_t m_unroll = 1, n_unroll = 8;

  for (size_t mi = 0; mi < m; mi += m_unroll) {
    if (n % n_unroll)
      matmul_abt_8_body<in_t, out_t, m_unroll, 4, 1, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_8_body<in_t, out_t, m_unroll, 4, 1, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
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
      matmul_abt_8_body<in_t, out_t, m_unroll, 4, 1, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_8_body<in_t, out_t, m_unroll, 4, 1, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
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
      matmul_abt_8_body<in_t, out_t, m_unroll, 2, 1, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_8_body<in_t, out_t, m_unroll, 2, 1, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
    }
  }
}

// C += A * B^T using (23 * interleave)xKx8 register tiles
template<typename in_t, typename out_t, int interleave>
void NOINLINE matmul_abt_23_8(size_t m, size_t n, size_t k, const in_t* A, size_t lda, const in_t* B, size_t ldb, out_t* C, size_t ldc)
{
  const size_t m_unroll = 22 * interleave, n_unroll = 8;

  if (m % m_unroll)
    matmul_abt_1_8(m % m_unroll, n, k, A, lda, B, ldb, C, ldc);

  for (size_t mi = m % m_unroll; mi < m; mi += m_unroll) {
    if (n % n_unroll)
      matmul_abt_8_body<in_t, out_t, m_unroll, 1, interleave, true>(n % n_unroll, k, A + mi*lda, lda, B, ldb, C + mi*ldc, ldc);

    for (size_t ni = n % n_unroll; ni < n; ni += n_unroll) {
      matmul_abt_8_body<in_t, out_t, m_unroll, 1, interleave, false>(n_unroll, k, A + mi*lda, lda, B + ni*ldb, ldb, C + mi*ldc + ni, ldc);
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
    size_t vl = vstate.vsetvl<in_t, 1, true>(k);
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
    return matmul_abt_23_8<in_t, out_t, 2>(m, n, k, A, k, B, k, C, n);
  else if (vstate.vlmax<out_t, 1>() >= 8)
    return matmul_abt_23_8<in_t, out_t, 1>(m, n, k, A, k, B, k, C, n);
  else if (vstate.vlmax<out_t, 1>() >= 4)
    return matmul_abt_11_8(m, n, k, A, k, B, k, C, n);
  else if (vstate.vlmax<out_t, 1>() >= 2)
    return matmul_abt_5_8(m, n, k, A, k, B, k, C, n);
  else
    return matmul_abt_1_8(m, n, k, A, k, B, k, C, n);
}

template<typename T>
void NOINLINE vector_fill(T* p, T v, size_t n)
{
  vstate.vsetvl<T, 8>(n);
  vstate.splat<T, 0>(v);

  do {
    size_t vl = vstate.vsetvl<T, 8>(n);
    vstate.store<T, 0>(p);
    p += vl;
    n -= vl;
  } while (n);
}
