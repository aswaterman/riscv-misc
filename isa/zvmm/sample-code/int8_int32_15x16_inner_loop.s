.L172:
  # Stripmining in K dimension
  vsetvli s10, a4, e8, m1, ta, mu

  # Load a 16-row tile of B
  vle8.v v0, (a2)
  add a5,a2,s0
  vle8.v v1, (a5)
  add a5,a5,s0
  vle8.v v2, (a5)
  add a5,a5,s0
  vle8.v v3, (a5)
  add a5,a5,s0
  vle8.v v4, (a5)
  add a5,a5,s0
  vle8.v v5, (a5)
  add a5,a5,s0
  vle8.v v6, (a5)
  add a5,a5,s0
  vle8.v v7, (a5)
  add a5,a5,s0
  vle8.v v8, (a5)
  add a5,a5,s0
  vle8.v v9, (a5)
  add a5,a5,s0
  vle8.v v10, (a5)
  add a5,a5,s0
  vle8.v v11, (a5)
  add a5,a5,s0
  vle8.v v12, (a5)
  add a5,a5,s0
  vle8.v v13, (a5)
  add a5,a5,s0
  vle8.v v14, (a5)
  add a5,a5,s0
  vle8.v v15, (a5)

  # C[0, 15:0] += A[0, :] * B[:, 15:0]
  vle8.v v31, (a3)
  vqmmacc.s.vv v16, v31, v0, 0
  vqmmacc.s.vv v16, v31, v8, 8
  add a5,a3,s1

  # C[1, 15:0] += A[1, :] * B[:, 15:0]
  vle8.v v31, (a5)
  vqmmacc.s.vv v17, v31, v0, 0
  vqmmacc.s.vv v17, v31, v8, 8
  add a5,a5,s1

  # ...
  vle8.v v31, (a5)
  vqmmacc.s.vv v18, v31, v0, 0
  vqmmacc.s.vv v18, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v19, v31, v0, 0
  vqmmacc.s.vv v19, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v20, v31, v0, 0
  vqmmacc.s.vv v20, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v21, v31, v0, 0
  vqmmacc.s.vv v21, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v22, v31, v0, 0
  vqmmacc.s.vv v22, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v23, v31, v0, 0
  vqmmacc.s.vv v23, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v24, v31, v0, 0
  vqmmacc.s.vv v24, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v25, v31, v0, 0
  vqmmacc.s.vv v25, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v26, v31, v0, 0
  vqmmacc.s.vv v26, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v27, v31, v0, 0
  vqmmacc.s.vv v27, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v28, v31, v0, 0
  vqmmacc.s.vv v28, v31, v8, 8
  add a5,a5,s1
  vle8.v v31, (a5)
  vqmmacc.s.vv v29, v31, v0, 0
  vqmmacc.s.vv v29, v31, v8, 8
  add a5,a5,s1

  # C[14, 15:0] += A[14, :] * B[:, 15:0]
  vle8.v v31, (a5)
  vqmmacc.s.vv v30, v31, v0, 0
  vqmmacc.s.vv v30, v31, v8, 8

  # Loop until K dimension is exhausted
  sub a4,a4,s10
  add a3,a3,s10
  add a2,a2,s10
  bne a4,zero,.L172
