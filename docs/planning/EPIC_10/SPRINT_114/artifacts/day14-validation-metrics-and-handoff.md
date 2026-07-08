# Sprint 114 Day 14: Validation, Metrics, and Non-Claim Handoff

## Purpose

Day 14 closes Sprint 114 by collecting final validation evidence, proof-owner
metrics, non-claim boundaries, and dependency-ordered residual debt. The
sprint added eigensolver proof coverage, cleaned bounded direct/iterative
exact-RHS setup, and cleaned bounded SVD proof-owner loops without claiming
public API, package, ABI, source-list, helper-target, or reviewed CTest
membership changes.

## Touched Surfaces

| Surface | Files |
|---|---|
| Eigensolver proof tests | `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c`, `tests/test_ldlt_backend_dispatch.c` |
| Direct/iterative exact-RHS cleanup | `tests/test_iterative.c`, `tests/test_bicgstab.c`, `tests/test_minres.c` |
| SVD proof-owner cleanup | `tests/test_svd.c` |
| Sprint planning evidence | `docs/planning/EPIC_10/SPRINT_114/PLAN.md`, `docs/planning/EPIC_10/SPRINT_114/WORKING_NOTES.md`, `docs/planning/EPIC_10/SPRINT_114/artifacts/*.md` |

No `src`, `include`, Make, CMake, CTest registration, package, CI, or install
metadata files changed.

## Proof-Owner Metrics

| Metric | Count / Status | Evidence |
|---|---:|---|
| Sprint artifacts produced | 14 | `day1` through `day14` artifacts in `docs/planning/EPIC_10/SPRINT_114/artifacts/` |
| Touched C test files | 8 | `git diff --name-only` over `.c` test files |
| New explicit `test_s114...` proof tests | 9 | `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c`, `tests/test_ldlt_backend_dispatch.c` |
| Eigensolver source movement | 0 | Day 10 continued no-move decision |
| Public API/header changes | 0 | no `include/` diffs |
| Source-list/build metadata changes | 0 | no `src`, Make, CMake, CTest, or CI metadata diffs |
| Helper-target changes | 0 | all new helpers remain file-local in test translation units |
| Reviewed CTest membership changes | 0 | no test registration metadata changed |

## Validation Evidence

Focused validation completed during the sprint:

- Day 9: `make build/test_eigs && ./build/test_eigs`
  - `test_eigs`: `43` tests, `0` failures, `956` assertions.
- Day 12:
  `make build/test_qr && ./build/test_qr && ./build/test_iterative && ./build/test_bicgstab && ./build/test_minres`
  - `test_qr`: `73` tests, `0` failures, `654` assertions.
  - `test_iterative`: `80` tests, `0` failures, `713` assertions.
  - `test_bicgstab`: `61` tests, `0` failures, `464` assertions.
  - `test_minres`: `43` tests, `0` failures, `702` assertions.
- Day 13: `make build/test_svd && ./build/test_svd`
  - `test_svd`: `98` tests, `0` failures, `1580` assertions.

Day 14 final validation passed because the branch modifies `.c` tests:

```text
make source-list-check
make format && make lint && make test
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_114 \
  tests/test_svd.c tests/test_iterative.c tests/test_bicgstab.c \
  tests/test_minres.c tests/test_eigs.c tests/test_eigs_thick_restart.c \
  tests/test_eigs_lobpcg.c tests/test_ldlt_backend_dispatch.c
```

Results:

- `make source-list-check`: `PASS (48 library sources)`.
- `make format && make lint && make test`: passed.
- `git diff --check`: passed.
- Trailing-whitespace scan across Sprint 114 docs and touched test files:
  no matches.

## Non-Claims

- No public API or install-header support changed.
- No package, ABI, Windows, CMake parity, or reviewed CTest membership claim is
  introduced.
- No source file split or eigensolver movement occurred.
- No broad direct/iterative exact-RHS oracle was introduced.
- No broad SVD proof abstraction was introduced.
- No helper target was added; cleanup helpers remain file-local.
- No Make, CMake, source-list, CI, or install metadata changed.

## Residual Deferred Debt

1. **Eigensolver Source Boundary Follow-Through**
   - Dependency: Sprint 114 proof stack and Day 10 no-move decision.
   - Remaining work: prove one private owner can move with source-list,
     compile-unit, dispatch, and fallback evidence. Candidate owners remain
     `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert setup, and
     `lanczos_iterate_op`.
   - Blocker: current ownership still crosses grow-m, thick-restart, LOBPCG,
     shift-invert, LDLT lifecycle, and build metadata boundaries.

2. **Direct/Iterative Cross-Solver Oracle Decision**
   - Dependency: Day 12 file-local exact-RHS cleanup.
   - Remaining work: decide whether QR, CG, GMRES, BiCGSTAB, and MINRES
     generated-RHS setup has enough common ownership for a shared helper.
   - Blocker: solver-specific proof values, preconditioner setup, restart
     behavior, accepted nonconvergence branches, and comparison assertions
     still belong at call sites.

3. **SVD Shared Proof Helper Decision**
   - Dependency: Day 13 storage-contract-specific cleanup.
   - Remaining work: decide whether reconstruction, orthogonality,
     Moore-Penrose, low-rank, and condition-number helpers can move to a
     shared test helper without hiding storage or dimension contracts.
   - Blocker: economy/full leading dimensions, product dimensions, sparse-vs-
     dense residual semantics, and condition-number interpretation remain
     proof-specific.

4. **Package, ABI, Platform, and Adoption Validation**
   - Dependency: future sprints that intentionally touch packaging,
     install-surface, or cross-platform support.
   - Remaining work: run install/package/ABI and Windows-reviewed validation
     only when those surfaces intentionally change.
   - Blocker: Sprint 114 intentionally did not touch those surfaces.

## Handoff

Sprint 114 closes with proof evidence in place and intentionally narrow cleanup
helpers. Downstream work should treat the eigensolver no-move decision,
direct/iterative file-local helper boundary, and SVD storage-contract helper
boundary as active constraints until a later sprint proves the next movement
step with build and validation evidence.
