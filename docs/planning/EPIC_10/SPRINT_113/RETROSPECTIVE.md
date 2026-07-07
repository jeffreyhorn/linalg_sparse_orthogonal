# Sprint 113 Retrospective

**Sprint:** 113 - Residual Behavior & Proof-Owner Closeout
**Duration:** 14 days (Days 1-14 landed on branch `sprint-113`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 113 started from Sprint 110's residual behavior-sensitive
      eigensolver and proof-owner debt.
- [x] Completed prior work was explicitly excluded from duplicate Sprint 113
      scope:
  - Matrix builder/input validation cleanup
  - Matrix I/O behavior ownership
  - CG exact-RHS helper work
  - SVD rank-deficient setup work
  - eigensolver handle/workspace validation work
- [x] Eigensolver grow-m sizing and retry behavior was selected as the bounded
      behavior owner.
- [x] Direct grow-m behavior proof landed in `tests/test_eigs.c`.
- [x] Eigensolver source movement was rejected for this sprint and replaced by
      an explicit no-move contract with future proof requirements.
- [x] LDLT CSC external dense-reference oracle cleanup landed in
      `tests/test_ldlt_csc.c`.
- [x] SVD partial-vector residual cleanup landed in
      `tests/test_svd_partial_helpers.h`.
- [x] Proof-owner metrics, membership drift status, broad-abstraction
      non-claims, and residual queues were documented.
- [x] Focused validation passed:
  - `make build/test_eigs && build/test_eigs`
  - `make build/test_ldlt_csc && build/test_ldlt_csc`
  - `make build/test_svd && build/test_svd`
- [x] Full required C/header quality gate passed:
  - `make format && make lint && make test`
- [x] Build/source/API drift checks found no Makefile, CMake, `cmake/`,
      `include/`, or `src/` drift.
- [x] Documentation hygiene passed:
  - `git diff --check`
  - trailing-whitespace scan
  - local Markdown link check
- [x] Sprint 113 closeout and handoff captured dependency-ordered residual
      deferred debt for final Epic 10 integration.

## What Went Well

1. **The sprint selected bounded owners before editing.**
   Day 1-3 kept the work from turning into broad source movement. The sprint
   selected grow-m behavior, LDLT CSC external oracle cleanup, and partial-SVD
   residual cleanup only after boundary artifacts identified proof values that
   had to remain visible.

2. **The eigensolver work added behavior proof without destabilizing ownership.**
   The Day 4 grow-m tests cover preparation/reuse, default and explicit
   capacity, too-small explicit iteration budgets, retry progress, and
   cancellation. Day 5-6 then correctly kept source movement deferred because
   adjacent Lanczos, Ritz, partial-publication, and shift-invert behavior still
   need direct proof.

3. **The direct-solver cleanup reduced local noise without hiding the oracle.**
   The LDLT CSC external dense-reference helper now owns allocation and cleanup
   through a local state object while keeping fixture identity, exact RHS,
   permutation, `ldlt_csc_solve`, dense-reference status, max-difference, and
   residual checks visible.

4. **The SVD cleanup stayed narrow.**
   The partial-SVD `A*v ~= sigma*u` helper centralizes only temporary vector
   allocation and residual accumulation. The tests still show shape, rank,
   options, singular-value expectations, diagnostic labels, and thresholds.

5. **Validation matched the touched surface.**
   Since `.c` and `.h` files changed, the sprint ran focused owner tests and
   the full `make format && make lint && make test` gate. It also verified no
   public API, install-header, helper-target, source-list, or reviewed CTest
   drift.

6. **The non-claims are useful closeout data.**
   Day 11 and Day 14 make it clear that Sprint 113 did not prove broad
   eigensolver source extraction, broad cross-solver proof abstraction, or broad
   SVD proof abstraction.

## What Didn't Go Well

1. **Eigensolver source movement remains blocked by proof gaps.**
   The grow-m owner is better tested, but shared Lanczos kernels, Ritz
   selection, Ritz vector lifting, partial-result publication, and shift-invert
   conversion still lack enough proof for safe movement.

2. **The remaining proof-owner queue is still large.**
   QR, CG, GMRES, BiCGSTAB, MINRES, SVD reconstruction, U/Vt orthogonality,
   Moore-Penrose products, low-rank loops, and condition-number logic all remain
   separate owners rather than solved by one abstraction.

3. **The sprint improved tests, not production architecture.**
   That was the right choice for this residual phase, but the codebase still
   carries maintainability pressure in large source and giant-test areas.

4. **Validation remains expensive.**
   The full quality gate is appropriate for `.c`/`.h` changes, but the
   nested-dissection and full test tail make late-sprint validation a long
   operation.

## Final Metrics

### Validation

| Metric | Sprint 113 close state |
|---|---:|
| focused eigensolver validation | `test_eigs`: 36 passed, 0 failed, 0 skipped |
| focused LDLT CSC validation | `test_ldlt_csc`: 100 passed, 0 failed, 0 skipped |
| focused SVD validation | `test_svd`: 98 passed, 0 failed, 0 skipped |
| full quality gate | `make format && make lint && make test` passed |
| Makefile/CMake/source-list drift | 0 files |
| public/install header drift | 0 files |
| runtime `src/` drift | 0 files |
| changed code/test files | 3 |
| reviewed CTest membership drift | 0 |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed |
| local Markdown link check | passed |

### Code and Test Changes

| File | Baseline lines | Close lines | Diff numstat | Purpose |
|---|---:|---:|---:|---|
| `tests/test_eigs.c` | 1560 | 1758 | +198 / -0 | grow-m behavior proof |
| `tests/test_ldlt_csc.c` | 3896 | 3915 | +79 / -60 | LDLT CSC external oracle cleanup |
| `tests/test_svd_partial_helpers.h` | 915 | 907 | +42 / -50 | partial-SVD residual helper cleanup |

### Sprint 113 Artifact Package

| Metric | Sprint 113 close state |
|---|---:|
| artifact files under `SPRINT_113/artifacts/` | 14 |
| planning and working-note files | 2 |
| retrospective files | 1 |
| artifact lines before retrospective | 2039 |
| working notes lines before retrospective | 594 |
| plan lines | 495 |

Notes:

- intake, selection, and eigensolver proof artifacts:
  - `day1-residual-intake-and-boundary.md`
  - `day2-eigensolver-behavior-owner-selection.md`
  - `day3-eigensolver-behavior-proof-design.md`
  - `day4-eigensolver-behavior-proof.md`
- movement/no-move and direct/iterative artifacts:
  - `day5-eigensolver-movement-decision.md`
  - `day6-eigensolver-no-move-contract.md`
  - `day7-direct-iterative-proof-owner-boundary.md`
  - `day8-direct-iterative-proof-owner-cleanup.md`
- SVD, metrics, validation, and closeout artifacts:
  - `day9-svd-proof-boundary-refresh.md`
  - `day10-svd-proof-owner-cleanup.md`
  - `day11-proof-owner-metrics-and-non-claims.md`
  - `day12-integrated-validation-plan.md`
  - `day13-integrated-validation-execution.md`
  - `day14-closeout-and-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- Add direct proof for `lanczos_iterate_op` behavior across basic,
  thick-restart, and LOBPCG-adjacent dispatch paths.
- Add repeated/clustered spectrum proof before moving Ritz selection.
- Add Ritz vector lifting proof before extracting shared vector-publication
  helpers.
- Add partial-result publication proof after `m_cap` exhaustion.
- Add shift-invert grow-m conversion proof.
- Revisit eigensolver source movement only after a single safe owner is proven.
- Clean up QR sequential RHS setup.
- Clean up CG preconditioner-specific exact-RHS setup.
- Clean up GMRES exact-RHS setup.
- Clean up BiCGSTAB exact-RHS setup.
- Clean up MINRES exact-RHS setup.
- Attempt broad direct/iterative oracle abstraction only after more
  solver-specific cleanup lanes prove common ownership.
- Split SVD reconstruction helper movement by storage contract.
- Split U/Vt orthogonality helper movement by economy/full leading-dimension
  convention.
- Extract Moore-Penrose product helpers only after preserving product dimension
  proof.
- Clean up dense low-rank proof loops.
- Clean up sparse low-rank proof loops.
- Clean up condition-number proof logic.

Still consciously constrained rather than silently solved:

- no broad eigensolver source split claim;
- no broad cross-solver proof abstraction claim;
- no broad SVD proof abstraction claim;
- no public API change;
- no install-header change;
- no helper-target change;
- no Make/CMake source-list change;
- no reviewed CTest membership change.

Not carried forward as unresolved Sprint 113 debt:

- Sprint 110 residual intake and boundary refresh;
- grow-m behavior proof selection and design;
- grow-m behavior test batch;
- eigensolver source movement/no-move decision;
- eigensolver no-move contract publication;
- LDLT CSC external dense-reference oracle cleanup;
- SVD partial-vector residual cleanup;
- proof-owner metrics and non-claims artifact;
- integrated validation plan;
- integrated validation execution;
- Sprint 113 closeout and handoff.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-residual-intake-and-boundary.md](./artifacts/day1-residual-intake-and-boundary.md)
- [day2-eigensolver-behavior-owner-selection.md](./artifacts/day2-eigensolver-behavior-owner-selection.md)
- [day3-eigensolver-behavior-proof-design.md](./artifacts/day3-eigensolver-behavior-proof-design.md)
- [day4-eigensolver-behavior-proof.md](./artifacts/day4-eigensolver-behavior-proof.md)
- [day5-eigensolver-movement-decision.md](./artifacts/day5-eigensolver-movement-decision.md)
- [day6-eigensolver-no-move-contract.md](./artifacts/day6-eigensolver-no-move-contract.md)
- [day7-direct-iterative-proof-owner-boundary.md](./artifacts/day7-direct-iterative-proof-owner-boundary.md)
- [day8-direct-iterative-proof-owner-cleanup.md](./artifacts/day8-direct-iterative-proof-owner-cleanup.md)
- [day9-svd-proof-boundary-refresh.md](./artifacts/day9-svd-proof-boundary-refresh.md)
- [day10-svd-proof-owner-cleanup.md](./artifacts/day10-svd-proof-owner-cleanup.md)
- [day11-proof-owner-metrics-and-non-claims.md](./artifacts/day11-proof-owner-metrics-and-non-claims.md)
- [day12-integrated-validation-plan.md](./artifacts/day12-integrated-validation-plan.md)
- [day13-integrated-validation-execution.md](./artifacts/day13-integrated-validation-execution.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)
