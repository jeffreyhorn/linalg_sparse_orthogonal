# Sprint 114 Retrospective

**Sprint:** 114 - Residual Eigensolver, Direct/Iterative & SVD Proof-Owner Follow-Through
**Duration:** 14 days (Days 1-14 landed on branch `sprint-114`)
**Status:** Complete

## Definition of Done Checklist

- [x] Sprint 114 started from Sprint 113's residual proof-owner debt and
      duplicate-fenced completed Sprint 113 work.
- [x] Completed prior work was explicitly excluded from duplicate Sprint 114
      scope:
  - grow-m behavior selection, design, and proof;
  - eigensolver no-move contract;
  - LDLT CSC external dense-reference cleanup;
  - partial-SVD residual helper cleanup;
  - Sprint 113 metrics, validation, closeout, and handoff artifacts.
- [x] Added direct eigensolver behavior proof across basic Lanczos,
      thick-restart, and LOBPCG-adjacent paths.
- [x] Added repeated/clustered Ritz selection proof before revisiting Ritz
      movement.
- [x] Added vector lifting and publication-boundary proof for grow-m,
      shift-invert, thick-restart, LOBPCG, and partial-publication sentinel
      behavior.
- [x] Added explicit grow-m `m_cap` exhaustion partial-result proof.
- [x] Added shift-invert grow-m conversion proof for nearest-sigma results.
- [x] Revisited eigensolver movement and published a continued no-move
      decision with future movement requirements.
- [x] Cleaned bounded direct/iterative exact-RHS setup while preserving
      solver-specific proof values.
- [x] Cleaned bounded SVD proof owners while preserving storage,
      leading-dimension, product-dimension, low-rank, and condition-number
      evidence.
- [x] Captured proof-owner metrics, touched-surface review, non-claims, and
      residual deferred debt.
- [x] Focused validation passed:
  - `make build/test_eigs && ./build/test_eigs`
  - `make build/test_qr && ./build/test_qr && ./build/test_iterative && ./build/test_bicgstab && ./build/test_minres`
  - `make build/test_svd && ./build/test_svd`
- [x] Source-list validation passed:
  - `make source-list-check`
- [x] Full required C quality gate passed:
  - `make format && make lint && make test`
- [x] Build/source/API drift checks found no `src/`, `include/`, Makefile,
      CMake, CTest registration, package, CI, or install metadata drift.
- [x] Documentation hygiene passed:
  - `git diff --check`
  - trailing-whitespace scan across Sprint 114 docs and touched tests.

## What Went Well

1. **The sprint converted residual debt into ordered proof batches.**
   Day 1 made Sprint 113's completed work evidence rather than unresolved
   debt, then ordered Sprint 114 around eigensolver proof prerequisites,
   direct/iterative cleanup, SVD cleanup, and final validation.

2. **The eigensolver work improved evidence without forcing a premature split.**
   Days 2-9 added behavior proof for Lanczos recurrence visibility,
   repeated/clustered Ritz selection, vector publication, bounded
   partial-results, and shift-invert conversion. Day 10 then correctly kept
   movement deferred because the candidates still cross source-list,
   build-metadata, backend, and cleanup-lifecycle boundaries.

3. **The Ritz and publication tests kept proof values visible.**
   The new tests expose dimensions, `k`, `block_size`, `sigma`, backend
   choices, tolerances, expected values, residual gates, orthogonality gates,
   iteration budgets, and sentinel values at the call sites.

4. **The direct/iterative cleanup reduced repeated setup without creating a
   false common oracle.**
   CG and GMRES reuse existing file-local generated-RHS helpers, while
   BiCGSTAB and MINRES gained local exact-RHS helpers. Preconditioners,
   restarts, accepted nonconvergence, comparison solvers, and residual
   thresholds remain in the tests that prove them.

5. **The SVD cleanup improved maintainability while respecting proof-owner
   boundaries.**
   Reconstruction, Vt orthogonality, Moore-Penrose, low-rank, sparse-vs-dense,
   and condition-number helpers now remove local repetition in
   `tests/test_svd.c` without moving ownership out of the file or claiming a
   broad SVD proof abstraction.

6. **Validation matched the actual touched surface.**
   Since the branch changed `.c` tests, Sprint 114 ran focused owner tests,
   source-list validation, the full `make format && make lint && make test`
   gate, and diff hygiene.

## What Didn't Go Well

1. **Eigensolver source movement is still not ready.**
   The sprint added useful proof, but every plausible movement candidate still
   crosses multiple owners: grow-m, thick-restart, LOBPCG, shift-invert, LDLT
   lifecycle, source-list, and build metadata.

2. **The proof-owner queue remains broad.**
   The branch improved file-local proof helpers, but cross-solver exact-RHS
   abstraction and shared SVD proof abstraction remain blocked until more
   solver-specific owners converge.

3. **The codebase still carries large-test pressure.**
   Sprint 114 intentionally avoided helper targets, source-list edits, and
   production source movement. That kept risk bounded, but the large test
   files still need future source-boundary and proof-owner follow-through.

4. **Full validation remains expensive.**
   The required quality gate is appropriate for `.c` changes, but late-sprint
   validation still takes meaningful time, especially through the full test
   tail.

## Final Metrics

### Validation

| Metric | Sprint 114 close state |
|---|---:|
| focused eigensolver validation | `test_eigs`: 43 passed, 0 failed |
| focused direct/iterative validation | `test_qr`: 73 passed; `test_iterative`: 80 passed; `test_bicgstab`: 61 passed; `test_minres`: 43 passed |
| focused SVD validation | `test_svd`: 98 passed, 0 failed |
| source-list validation | `make source-list-check`: `PASS (48 library sources)` |
| full quality gate | `make format && make lint && make test` passed |
| public/install header drift | 0 files |
| runtime `src/` drift | 0 files |
| Makefile/CMake/source-list drift | 0 files |
| reviewed CTest membership drift | 0 |
| helper-target drift | 0 |
| changed C test files | 8 |
| explicit `test_s114...` proof tests | 9 |
| eigensolver source movements | 0 |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed |

### Code and Test Changes

| File | Close lines | Diff numstat | Purpose |
|---|---:|---:|---|
| `tests/test_bicgstab.c` | 1826 | +59 / -25 | BiCGSTAB exact-RHS helper cleanup |
| `tests/test_eigs.c` | 2158 | +399 / -0 | grow-m, vector-publication, partial-result, clustered, and shift-invert proof |
| `tests/test_eigs_lobpcg.c` | 1417 | +100 / -0 | LOBPCG-adjacent and publication-boundary proof |
| `tests/test_eigs_thick_restart.c` | 1377 | +96 / -0 | thick-restart recurrence and vector-publication proof |
| `tests/test_iterative.c` | 2924 | +42 / -30 | CG/GMRES exact-RHS cleanup |
| `tests/test_ldlt_backend_dispatch.c` | 975 | +40 / -0 | repeated/nearest-sigma selector proof |
| `tests/test_minres.c` | 1649 | +96 / -35 | MINRES exact-RHS helper cleanup |
| `tests/test_svd.c` | 2809 | +176 / -260 | bounded SVD proof-owner cleanup |

### Sprint 114 Artifact Package

| Metric | Sprint 114 close state |
|---|---:|
| artifact files under `SPRINT_114/artifacts/` | 14 |
| artifact lines before retrospective | 1334 |
| working notes lines before retrospective | 546 |
| plan lines | 475 |
| retrospective files | 1 |

Notes:

- intake, design, and eigensolver proof artifacts:
  - `day1-residual-intake-and-boundary.md`
  - `day2-lanczos-iterate-behavior-design.md`
  - `day3-lanczos-iterate-behavior-proof.md`
  - `day4-ritz-selection-proof-design.md`
  - `day5-ritz-selection-proof.md`
  - `day6-ritz-vector-publication-design.md`
  - `day7-ritz-vector-publication-proof.md`
  - `day8-partial-result-publication-proof.md`
  - `day9-shift-invert-growm-conversion-proof.md`
- movement, direct/iterative, SVD, and closeout artifacts:
  - `day10-eigensolver-movement-decision.md`
  - `day11-direct-iterative-exact-rhs-cleanup-design.md`
  - `day12-direct-iterative-exact-rhs-cleanup.md`
  - `day13-svd-proof-owner-cleanup.md`
  - `day14-validation-metrics-and-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- Move one eigensolver private owner only after a future sprint provides exact
  old/new files, source-list and CMake updates, focused consumer proof,
  reviewed CTest count evidence where applicable, and rollback instructions.
- Revisit `s20_select_indices` movement after source-list/build metadata
  proof covers grow-m, thick-restart, and LOBPCG consumers.
- Revisit `s20_lift_ritz_vectors` movement only after grow-m and
  thick-restart partial-publication states have a shared proven owner.
- Revisit shift-invert setup/conversion movement after LDLT factor lifecycle,
  `used_csc_path_ldlt`, operator selection, public error propagation, and
  cleanup ownership are separated or directly proven.
- Revisit `lanczos_iterate_op` movement with explicit compile-unit proof for
  all current consumers.
- Decide whether QR, CG, GMRES, BiCGSTAB, and MINRES generated-RHS setup has
  enough common ownership for a shared direct/iterative oracle.
- Decide whether SVD reconstruction, U/Vt orthogonality, Moore-Penrose,
  low-rank, sparse-vs-dense, and condition-number helpers can share an owner
  without hiding storage, leading-dimension, product-dimension, fixture, or
  threshold proof.
- Validate package, ABI, Windows, CMake parity, install-header, and adoption
  surfaces when a future sprint changes those surfaces.

Still consciously constrained rather than silently solved:

- no eigensolver source split claim;
- no public API change;
- no install-header or ABI claim;
- no helper-target change;
- no Make/CMake source-list change;
- no reviewed CTest membership change;
- no package, platform, Windows, or CMake parity claim;
- no broad direct/iterative oracle claim;
- no broad SVD proof abstraction claim.

Not carried forward as unresolved Sprint 114 debt:

- Sprint 114 residual intake and duplicate fence;
- Lanczos behavior proof design and implementation;
- repeated/clustered Ritz selection proof design and implementation;
- Ritz vector lifting and publication-boundary proof design and
  implementation;
- grow-m `m_cap` exhaustion partial-result proof;
- shift-invert grow-m conversion proof;
- Day 10 eigensolver movement/no-move decision;
- direct/iterative exact-RHS cleanup design and bounded implementation;
- bounded SVD proof-owner cleanup;
- Sprint 114 validation, metrics, non-claims, and closeout handoff.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-residual-intake-and-boundary.md](./artifacts/day1-residual-intake-and-boundary.md)
- [day2-lanczos-iterate-behavior-design.md](./artifacts/day2-lanczos-iterate-behavior-design.md)
- [day3-lanczos-iterate-behavior-proof.md](./artifacts/day3-lanczos-iterate-behavior-proof.md)
- [day4-ritz-selection-proof-design.md](./artifacts/day4-ritz-selection-proof-design.md)
- [day5-ritz-selection-proof.md](./artifacts/day5-ritz-selection-proof.md)
- [day6-ritz-vector-publication-design.md](./artifacts/day6-ritz-vector-publication-design.md)
- [day7-ritz-vector-publication-proof.md](./artifacts/day7-ritz-vector-publication-proof.md)
- [day8-partial-result-publication-proof.md](./artifacts/day8-partial-result-publication-proof.md)
- [day9-shift-invert-growm-conversion-proof.md](./artifacts/day9-shift-invert-growm-conversion-proof.md)
- [day10-eigensolver-movement-decision.md](./artifacts/day10-eigensolver-movement-decision.md)
- [day11-direct-iterative-exact-rhs-cleanup-design.md](./artifacts/day11-direct-iterative-exact-rhs-cleanup-design.md)
- [day12-direct-iterative-exact-rhs-cleanup.md](./artifacts/day12-direct-iterative-exact-rhs-cleanup.md)
- [day13-svd-proof-owner-cleanup.md](./artifacts/day13-svd-proof-owner-cleanup.md)
- [day14-validation-metrics-and-handoff.md](./artifacts/day14-validation-metrics-and-handoff.md)

