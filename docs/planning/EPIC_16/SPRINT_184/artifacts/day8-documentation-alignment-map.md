# Sprint 184 Day 8: Documentation Alignment Map

## Purpose

Map every known downstream documentation and example surface that describes the
selected Sprint 184 family, `include/sparse_qr.h`, after the Day 4-Day 7 header
contract cleanup and organization pass. Day 8 is an audit and planning day; it
captures contradictions and edit priorities before changing tutorial-facing
documentation.

## Reviewed Surfaces

| Surface | QR content reviewed | Status |
| --- | --- | --- |
| `README.md` | QR API bullets, COLAMD mention, QR evidence block, public header table. | Needs targeted Day 9 alignment. |
| `docs/api_reference.md` | `sparse_qr.h` summary row. | Aligned; no edit needed. |
| `docs/tutorial.md` | QR factorization snippet and diagnostics handoff. | Needs targeted Day 9 alignment. |
| `docs/cookbook.md` | Solver routing table and QR evidence note. | Aligned enough; no immediate edit needed. |
| `docs/solver_selection.md` | QR routing row, COLAMD guidance, examples, QR evidence boundary. | Needs targeted Day 9 alignment. |
| `examples/README.md` | QR least-squares, minimum-norm, and COLAMD descriptions. | Mostly aligned; optional Day 9 polish. |
| `examples/example_least_squares.c` | Factor/rank/solve/free workflow. | Aligned; no required code edit. |
| `examples/example_minnorm.c` | Minimum-norm solve/refine workflow. | Aligned; optional options/cancellation note in docs. |
| `examples/example_colamd.c` | QR+COLAMD factor/solve/rank-info/free workflow. | Optional return-code handling polish if examples are edited. |

## Header Contracts Used for Comparison

The audit compared downstream docs against these cleaned QR header contracts:

- QR factorization requires an identity-permutation input matrix and borrows
  that matrix without modifying it.
- Successful factorization stores owned data inside caller-owned
  `sparse_qr_t`; callers release it with `sparse_qr_free()`.
- `sparse_qr_factor_opts()` accepts `NULL` options for defaults and supports
  progress cancellation through `SPARSE_ERR_CANCELLED`.
- `sparse_qr_solve()` returns standard QR least-squares/direct/basic solutions;
  underdetermined minimum-2-norm solutions use `sparse_qr_solve_minnorm()`.
- Rank tolerance is QR-local: positive `tol` is relative to `|R(0,0)|`, and
  nonpositive `tol` selects the default threshold.
- Nullspace, diagonal, rank-info, solution, and residual outputs are
  caller-owned.
- QR condition estimation is a quick R-diagonal diagnostic, not a full
  condition-number guarantee.
- Minimum-norm solve/refine build temporary QR factorizations internally, apply
  `opts` to those factorizations, and may propagate cancellation.

## Findings

| Priority | Surface | Finding | Day 9 action |
| ---: | --- | --- | --- |
| 1 | `README.md` | `sparse_qr_factor_opts(A, &opts, &qr)` is described as "with optional AMD column reordering"; the header now clearly recommends COLAMD for unsymmetric QR while AMD/RCM/ND form `A^T*A`. | Change the bullet to name COLAMD and keep AMD/RCM/ND as accepted symmetric-ordering options only if needed. |
| 2 | `docs/solver_selection.md` | The QR evidence boundary says the maintained QR proof supports only rank/nullity/nullspace residual behavior, while README/cookbook also describe selected minimum-norm and compatible least-squares comparison rows. | Update the boundary to include the selected minimum-norm/comparison evidence without widening into broad QR or external-library parity. |
| 3 | `docs/tutorial.md` | The QR snippet demonstrates the right lifecycle but omits factor/solve return-code handling and does not name caller-owned output buffers as clearly as the header. | Add concise error handling and ownership wording around `sparse_qr_factor()`, `sparse_qr_solve()`, and `sparse_qr_free()`. |
| 4 | `examples/README.md` | Minimum-norm docs point to the API but do not mention that `opts` apply to internal QR factorizations and cancellation can propagate. | Add a short note only if Day 9 updates the QR example text. |
| 5 | `examples/example_colamd.c` | The example checks factorization status but does not check `sparse_qr_solve()` or `sparse_qr_rank_info()` return codes before printing results. | Optional executable-example polish if Day 9 includes code edits. |

## Explicit Non-Issues

- `docs/api_reference.md` already describes `sparse_qr.h` at the right level:
  QR, least-squares, rank, nullspace, and minimum-norm contracts.
- `docs/cookbook.md` keeps QR evidence fixture-local and does not widen into
  broad QR or external-library parity.
- `examples/example_least_squares.c` already demonstrates the core
  factor/rank/solve/free lifecycle with solve error handling.
- `examples/example_minnorm.c` already uses the public minimum-norm solve and
  refinement APIs correctly for a teaching example.

## Day 9 Edit Boundary

Day 9 should keep edits narrow:

- update QR-facing README/tutorial/solver-selection/example text only;
- preserve all public QR declarations;
- avoid new performance, package, ABI, platform, or external-library claims;
- keep QR evidence fixture-local and selected-target specific;
- run the full C quality gate only if executable examples or headers change.

## Validation

Day 8 changed planning artifacts only. No new `.c` or `.h` edits were made for
this day, so the full C quality gate was not rerun.

- `git diff --check`: passed.
