# Day 4 Repeated and Clustered Ritz Selection Proof Design

## Purpose

Day 4 designs the repeated/clustered Ritz selection proof that must land before
any Ritz selection source movement. The proof has two separate lanes:

- exact repeated Ritz values are tested directly at the selector boundary;
- clustered-but-distinct spectra are tested through the public eigensolver
  surface.

This split avoids a false public Lanczos multiplicity claim. A scalar Lanczos
run can collapse an exact repeated eigenspace to one Krylov direction even when
the Ritz selector itself handles repeated values correctly.

## Ritz Selection Owner Inventory

| Owner | Current location | Day 5 proof role |
|---|---|---|
| `s20_select_indices` | `src/sparse_eigs.c` with internal declaration in `src/sparse_eigs_internal.h` | Direct repeated-value and tie-order proof for `LARGEST`, `SMALLEST`, and `NEAREST_SIGMA`. |
| Grow-m Ritz result publication | `src/sparse_eigs.c` `s46_run_growm_backend` | Public clustered-spectrum proof after `s20_select_indices` selects the target Ritz values. |
| Thick-restart selection consumers | `src/sparse_eigs_thick_restart.c` and `tests/test_eigs_thick_restart.c` | Existing parity tests remain evidence; Day 5 should not add thick-restart movement unless clustered grow-m evidence is stable. |
| Shift-invert nearest-sigma selector | `s20_select_indices` plus `SPARSE_EIGS_NEAREST_SIGMA` post-processing | Direct selector tie proof and existing public shift-invert tests provide enough design coverage for Day 5. |

## Existing Evidence to Preserve

| Existing evidence | File | Day 4 interpretation |
|---|---|---|
| LARGEST/SMALLEST diagonal smoke tests | `tests/test_eigs.c` | Keep as simple public ordering evidence. |
| Shift-invert diagonal tie around sigma | `tests/test_eigs.c` `test_shift_invert_diagonal_k3` | Keep as public nearest-sigma tie evidence. |
| Zero-spectrum no-divide-by-zero tests | `tests/test_eigs.c` | Keep as degenerate repeated-spectrum safety evidence, not a full multiplicity proof. |
| Cluster warning in refinement tests | `tests/test_eigs.c` | Preserve the rule that clustered eigenvectors can be arbitrary even when eigenvalue selection is valid. |
| Thick-restart KKT nearest-sigma parity | `tests/test_eigs_thick_restart.c` | Keep as clustered interior parity evidence. |
| Day 3 Lanczos behavior tests | `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c` | Provide the prerequisite recurrence/dispatch proof for Day 5 selector work. |

## Fixture Plan

| Fixture | Target | Values | Expected result |
|---|---|---|---|
| Repeated selector array | `tests/test_ldlt_backend_dispatch.c` or another existing internal eigensolver test file | `theta = {-4, -4, -1, 0.5, 0.5, 2, 2, 9}` | `LARGEST, k=4` selects indices `{7, 6, 5, 4}`; `SMALLEST, k=3` selects `{0, 1, 2}`. |
| Nearest-sigma selector tie array | Same internal test file | `theta = {-5, -3, -1, 1, 3, 5}` | `NEAREST_SIGMA, k=4` selects right side first on equal magnitude: `{5, 0, 4, 1}`. |
| Clustered public diagonal | `tests/test_eigs.c` | diagonal values with top cluster `{10.0, 9.99999, 9.99998}` and remaining values separated by at least `0.5` | `LARGEST, k=3` returns the three clustered top values in nonincreasing order within `1e-7`. |
| Clustered public lower spectrum | Optional Day 5 extension if the largest-cluster proof is stable | diagonal values with bottom cluster `{1.0, 1.00001, 1.00002}` and remaining values separated | `SMALLEST, k=3` returns the three clustered bottom values in nondecreasing order within `1e-7`. |

## Tolerance and Ordering Rules

- Direct selector tests use exact index assertions because the input `theta`
  array is already sorted and no floating-point iteration is involved.
- Public clustered solver tests assert eigenvalue ordering and values, not
  eigenvectors.
- Public clustered solver tests should use distinct gaps of about `1e-5`,
  which are close enough to exercise clustered selection but wide enough to
  avoid arbitrary equal-eigenspace behavior.
- Use `tol = 1e-12`, `reorthogonalize = 1`, and small `n` so the grow-m path
  can use a full basis without relying on a long restart sequence.
- For nearest-sigma tie behavior, assert the current selector contract:
  exact equal magnitudes choose the right endpoint first because the
  implementation uses `>` rather than `>=`.

## Day 5 Implementation Checklist

1. Add direct selector tests near existing internal eigensolver helper tests.
2. Assert repeated-value `LARGEST` and `SMALLEST` index choices directly.
3. Assert `NEAREST_SIGMA` equal-magnitude tie order directly.
4. Add one public clustered diagonal test in `tests/test_eigs.c`.
5. Keep expected values, tolerances, and cluster gaps visible at the test call
   site.
6. Avoid new helpers unless they stay local to the touched test file.
7. Do not move `s20_select_indices` or any Ritz selection source during Day 5.
8. Run `make format && make lint && make test` after Day 5 implementation.

## Movement Blockers

Ritz selection movement remains blocked until Day 5 proves:

- repeated values are selected deterministically at the selector boundary;
- exact equal-magnitude nearest-sigma ties preserve the current right-endpoint
  contract;
- clustered-but-distinct public spectra return the intended eigenvalue set;
- no public API, install-header, source-list, helper-target, Make, CMake, or
  reviewed CTest membership drift is needed.

## Validation Commands for Day 4

Day 4 changes documentation only. The required checks are:

```sh
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_114
```

No new C quality gate is required for Day 4 because no `.c` or `.h` files are
modified by the Day 4 design work. Day 3's code changes already passed
`make format && make lint && make test`.

## Completion Criteria

- Repeated and clustered spectrum cases have explicit Day 5 test targets.
- Exact repeated-value behavior is separated from public Lanczos multiplicity
  claims.
- Expected ordering, tolerances, and tie rules are documented.
- Ritz selection movement remains blocked until the Day 5 proof lands.
