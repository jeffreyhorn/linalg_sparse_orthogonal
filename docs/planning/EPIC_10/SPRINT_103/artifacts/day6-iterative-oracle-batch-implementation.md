# Sprint 103 Day 6 Iterative Oracle Batch Implementation

## Purpose

Day 6 implements the BiCGSTAB comparison batch selected on Day 5. The batch
adds focused iterative evidence without introducing a new shared helper,
external dense-reference script, public API, CMake target, or Makefile target.

## Implemented Batch

| test | fixture key | taxonomy class | reference behavior | expected result |
|---|---|---|---|---|
| `test_s103_bicgstab_nonsym_known_5_lu_reference` | `bicgstab_nonsym_known_5` | `nonsym-known-solution` | constructed `x_true`; linked-list LU cross-check; true residual | converges; residual `< 1e-10`; solution matches `x_true` and LU within `1e-8` |
| `test_s103_bicgstab_steam1_ilu_vs_gmres30_reference` | `bicgstab_steam1_ilu_vs_gmres30` | `nonsym-mm-medium` / `ill-conditioned-scale` | BiCGSTAB+ILU and GMRES(30)+ILU residual comparison on `steam1` | both converge; both true residuals `< 1e-4` |
| `test_s103_bicgstab_small_budget_expected_nonconvergence` | `bicgstab_small_budget_unsym_tridiag` | `nonsym-known-solution` | deliberately tiny iteration budget | returns non-converged status with finite residual |

## Touched Files

| file | change |
|---|---|
| `tests/test_bicgstab.c` | added Sprint 103 comparison section and registered three tests in the existing binary |
| `docs/planning/EPIC_10/SPRINT_103/WORKING_NOTES.md` | recorded Day 6 actions and validation |
| `docs/planning/EPIC_10/SPRINT_103/artifacts/day6-iterative-oracle-batch-implementation.md` | this implementation artifact |

No public headers, library sources, build files, Python helpers, or shared test
helpers were changed.

## Focused Validation Results

| command | result |
|---|---|
| `make build/test_bicgstab` | passed |
| `./build/test_bicgstab` | passed; 61 tests, 0 failures, 0 skips, 466 assertions |

New Sprint 103 evidence observed in the focused run:

| test | observed result |
|---|---|
| `test_s103_bicgstab_nonsym_known_5_lu_reference` | passed; 5 BiCGSTAB iterations, relative residual `1.136e-16` |
| `test_s103_bicgstab_steam1_ilu_vs_gmres30_reference` | passed; BiCGSTAB+ILU residual `6.950e-07`, GMRES(30)+ILU residual `1.369e-07` |
| `test_s103_bicgstab_small_budget_expected_nonconvergence` | passed; `SPARSE_ERR_NOT_CONVERGED`, finite residual `1.243e-01` |

## Full Validation Results

Because Day 6 changed a `.c` test file, the required full quality chain was
run:

| command | result |
|---|---|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed; `All tests passed.` |

The full test run also executed `test_bicgstab` with the new Sprint 103 tests:

| full-run test binary | result |
|---|---|
| `test_bicgstab` | passed; 61 tests, 0 failures, 0 skips, 466 assertions |
| `test_stagnation` | passed; 46 tests, 0 failures, 0 skips, 308 assertions |
| `test_iterative` | passed; 80 tests, 0 failures, 0 skips, 711 assertions |

## Claim Boundaries

Day 6 earns only bounded BiCGSTAB comparison evidence:

- deterministic nonsymmetric known-solution behavior agrees with LU and true
  residual checks for `bicgstab_nonsym_known_5`;
- `steam1` BiCGSTAB+ILU and GMRES(30)+ILU both satisfy the declared residual
  threshold on the named fixture;
- a deliberately under-budgeted nonsymmetric tridiagonal fixture reports
  expected non-convergence with a finite residual.

Day 6 does not claim:

- external package parity;
- external dense-helper evidence for BiCGSTAB;
- portable iteration-count or performance superiority;
- correctness on all nonsymmetric systems;
- GMRES is an independent external oracle;
- any public API or solver behavior changed.

## Day 7 Handoff

Day 7 should validate the implemented iterative evidence against the Day 2
ranking and rerank the remaining spectral/SVD work. Recommended checks:

- confirm no helper debt was introduced;
- confirm BiCGSTAB comparison gap is reduced but not closed broadly;
- decide whether LOBPCG or thick-restart remains the next highest-value
  spectral comparison lane;
- keep SVD deferred until the spectral/SVD overlap scope is frozen.

## Day 6 Conclusion

Sprint 103 now has a focused BiCGSTAB comparison batch implemented in the
existing test binary. The batch strengthens iterative evidence while preserving
the Day 4 helper boundary and Day 5 non-claims.
