.L211:
  # Stripmining in K dimension
  vsetvli s3, a3, e8, m1, ta, mu

  # Load 8 rows of B
  vle8.v v0, (a1)
  add a4,a1,s11
  vle8.v v1, (a4)
  add a4,a4,s11
  vle8.v v2, (a4)
  add a4,a4,s11
  vle8.v v3, (a4)
  add a4,a4,s11
  vle8.v v4, (a4)
  add a4,a4,s11
  vle8.v v5, (a4)
  add a4,a4,s11
  vle8.v v6, (a4)
  add a4,a4,s11
  vle8.v v7, (a4)

  # C[0, 7:0] += A[0, :] * B[:, 7:0]
  vle8.v v31, (a2)
  vqbdots.vv v8, v31, v0, 0
  add a4,a2,s2

  # C[1, 7:0] += A[1, :] * B[:, 7:0]
  vle8.v v31, (a4)
  vqbdots.vv v9, v31, v0, 0
  add a4,a4,s2

  # ...
  vle8.v v31, (a4)
  vqbdots.vv v10, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v11, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v12, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v13, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v14, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v15, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v16, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v17, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v18, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v19, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v20, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v21, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v22, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v23, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v24, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v25, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v26, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v27, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v28, v31, v0, 0
  add a4,a4,s2
  vle8.v v31, (a4)
  vqbdots.vv v29, v31, v0, 0
  add a4,a4,s2

  # C[22, 7:0] += A[22, :] * B[:, 7:0]
  vle8.v v31, (a4)
  vqbdots.vv v30, v31, v0, 0

  # Loop until K dimension is exhausted
  sub a3,a3,s3
  add a2,a2,s3
  add a1,a1,s3
  bne a3,zero,.L211
