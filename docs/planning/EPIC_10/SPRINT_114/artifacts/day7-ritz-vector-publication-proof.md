# Day 7 Ritz Vector Lifting and Publication Boundary Proof

## Purpose

Day 7 implements the vector lifting and publication-boundary proof designed on
Day 6. The implementation adds public-surface tests for grow-m Lanczos,
shift-invert, thick-restart, and LOBPCG vector publication without moving
helpers or changing public result semantics.

## Implemented Proofs

| Proof | File | Test | Evidence |
|---|---|---|---|
| Grow-m vector lift on non-diagonal SPD | `tests/test_eigs.c` | `test_s114_growm_vector_lift_public_boundary` | Forces `SPARSE_EIGS_BACKEND_LANCZOS` on a Laplacian fixture and asserts `n_requested`, `n_converged`, backend identity, eigenvalue order, residuals, normalization, and orthogonality. |
| Shift-invert original-space vectors | `tests/test_eigs.c` | `test_s114_shift_invert_vector_publication_boundary` | Uses `NEAREST_SIGMA` with visible `sigma = 2.0`, verifies original-space residuals, and checks vector orthogonality. |
| Partial publication sentinel boundary | `tests/test_eigs.c` | `test_s114_growm_partial_vector_publication_sentinel_boundary` | Uses a deliberately tight tolerance and minimal valid grow-m budget to return `SPARSE_ERR_NOT_CONVERGED`; asserts consumed result shape and that caller-owned slots beyond `k` remain untouched. |
| Thick-restart vector lift boundary | `tests/test_eigs_thick_restart.c` | `test_s114_thick_restart_vector_publication_boundary` | Forces `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART` on a non-diagonal SPD fixture and checks public shape, residuals, and orthogonality. |
| LOBPCG publication with `block_size > k` | `tests/test_eigs_lobpcg.c` | `test_s114_lobpcg_block_size_gt_k_vector_publication_boundary` | Forces LOBPCG with `block_size = 5`, `k = 3`, and verifies only the requested columns are consumed with residual and orthogonality checks. |

## Helper-Movement Assessment

The tests prove public result-shape and vector-quality boundaries across the
current publication paths, but helper movement remains blocked for Day 10.
Reasons:

- grow-m and thick-restart share `s20_lift_ritz_vectors`, but their partial
  publication states still differ (`Y_long`/`m_actual` versus
  `Y_arrow`/restart state);
- LOBPCG publishes from `X[:, j]`, not from a Lanczos basis and reduced
  eigenvector matrix;
- partial-result documentation still needs Day 8's `m_cap` exhaustion proof
  before a shared partial-publication helper can hide control flow safely.

## Proof Boundaries

- No public API or install-header changes were made.
- No source movement, source-list edits, helper-target edits, Make edits,
  CMake edits, or reviewed CTest registration changes were made.
- No helper extraction was performed.
- Tests keep matrix sizes, `k`, `block_size`, `sigma`, tolerances,
  iteration budgets, backends, residual thresholds, orthogonality thresholds,
  and sentinels visible at call sites.

## Validation Plan

Day 7 modifies C tests, so the required quality gate is:

```sh
make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg
./build/test_eigs
./build/test_eigs_thick_restart
./build/test_eigs_lobpcg
make format && make lint && make test
```

## Completion Criteria

- Vector lifting and publication are directly tested through public results.
- Full and partial publication boundaries have explicit assertions.
- Helper movement remains blocked until Day 10 can consider the complete
  proof stack.
- No unsupported API, build, packaging, CTest, or source movement claim is
  introduced.
