# Sprint 113 Day 11: Proof-Owner Metrics and Non-Claims

## Purpose

Capture the concrete Sprint 113 proof-owner metrics through Day 10, document
membership drift status, and make the remaining proof-owner non-claims explicit
before the Day 12 integrated validation plan.

## Commands Used

Metrics and drift checks were captured with:

```sh
git diff --numstat
wc -l tests/test_eigs.c tests/test_ldlt_csc.c tests/test_svd_partial_helpers.h \
  docs/planning/EPIC_10/SPRINT_113/PLAN.md \
  docs/planning/EPIC_10/SPRINT_113/WORKING_NOTES.md \
  docs/planning/EPIC_10/SPRINT_113/artifacts/*.md
git show HEAD:tests/test_eigs.c | wc -l
git show HEAD:tests/test_ldlt_csc.c | wc -l
git show HEAD:tests/test_svd_partial_helpers.h | wc -l
git diff --name-only -- Makefile CMakeLists.txt cmake include src tests | sort
```

## Code and Test Metrics

| File | Baseline lines | Current lines | Diff numstat | Sprint 113 role |
|---|---:|---:|---:|---|
| `tests/test_eigs.c` | 1560 | 1758 | +198 / -0 | Day 4 grow-m sizing and retry behavior proof. |
| `tests/test_ldlt_csc.c` | 3896 | 3915 | +79 / -60 | Day 8 LDLT CSC external dense-reference oracle cleanup. |
| `tests/test_svd_partial_helpers.h` | 915 | 907 | +42 / -50 | Day 10 partial-SVD `A*v ~= sigma*u` residual helper cleanup. |

No `src/` file is changed in the current Sprint 113 worktree.

No install header under `include/` is changed in the current Sprint 113
worktree.

## Planning and Artifact Metrics

Current Sprint 113 documentation metrics:

| File | Current lines |
|---|---:|
| `docs/planning/EPIC_10/SPRINT_113/PLAN.md` | 495 |
| `docs/planning/EPIC_10/SPRINT_113/WORKING_NOTES.md` | 467 before this Day 11 update |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day1-residual-intake-and-boundary.md` | 106 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day2-eigensolver-behavior-owner-selection.md` | 150 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day3-eigensolver-behavior-proof-design.md` | 288 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day4-eigensolver-behavior-proof.md` | 105 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day5-eigensolver-movement-decision.md` | 151 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day6-eigensolver-no-move-contract.md` | 142 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day7-direct-iterative-proof-owner-boundary.md` | 148 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day8-direct-iterative-proof-owner-cleanup.md` | 112 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day9-svd-proof-boundary-refresh.md` | 122 |
| `docs/planning/EPIC_10/SPRINT_113/artifacts/day10-svd-proof-owner-cleanup.md` | 116 |

The Sprint 113 artifact set through Day 10 totals 1440 lines.

## Membership Drift Table

| Surface | Current diff status | Day 11 assessment |
|---|---|---|
| `Makefile` | unchanged | No helper-target, quality-target, or source-list drift. |
| `CMakeLists.txt` | unchanged | No CMake target, install, or CTest membership drift. |
| `cmake/` | unchanged | No package/config helper drift. |
| `include/` | unchanged | No public API or install-header drift. |
| `src/` | unchanged | No implementation source-list or runtime behavior drift. |
| `tests/test_eigs.c` | changed | Behavior proof additions only. Existing test binary membership unchanged. |
| `tests/test_ldlt_csc.c` | changed | Local test cleanup only. Existing test binary membership unchanged. |
| `tests/test_svd_partial_helpers.h` | changed | Included test helper cleanup only. Existing `test_svd` binary membership unchanged. |
| reviewed CTest surface | unchanged by file membership | No reviewed CTest registration count or test-name drift introduced by Sprint 113 changes. |

## Remaining Proof-Owner Residual Queue

### Eigensolver

Remaining eigensolver cleanup remains deferred because Day 5 and Day 6
explicitly rejected movement without broader proof:

- `lanczos_iterate_op` movement;
- Ritz selection on repeated or clustered spectra;
- Ritz vector lifting;
- partial-result publication after `m_cap` exhaustion;
- shift-invert grow-m conversion;
- shared helper visibility rules.

### Direct and Iterative Solvers

Remaining direct/iterative proof-owner cleanup candidates:

- QR sequential RHS setup;
- CG preconditioner-specific exact-RHS setup;
- GMRES exact-RHS setup;
- BiCGSTAB exact-RHS setup;
- MINRES exact-RHS setup;
- broad direct/iterative oracle abstraction.

### SVD

Remaining SVD proof-owner cleanup candidates:

- reconstruction helper movement;
- U/Vt orthogonality helper movement;
- Moore-Penrose product helper extraction;
- dense low-rank proof-loop cleanup;
- sparse low-rank proof-loop cleanup;
- condition-number proof cleanup.

## Broad-Abstraction Non-Claims

Sprint 113 does not claim that a broad cross-solver proof abstraction is safe.
The current evidence supports only the bounded owners already implemented:

- eigensolver grow-m behavior tests;
- LDLT CSC external dense-reference oracle cleanup;
- partial-SVD `A*v ~= sigma*u` residual helper cleanup.

Sprint 113 also does not claim that broad SVD reconstruction, orthogonality,
Moore-Penrose, low-rank, or condition-number helper extraction is safe. Those
proof owners use different matrix shapes, leading dimensions, tolerances, and
diagnostic expectations. They need their own boundary artifact before movement.

## Public and Reviewed Surface Assessment

Through Day 11:

- no public API changed;
- no install header changed;
- no runtime source file changed;
- no Makefile or CMake target changed;
- no helper target changed;
- no CTest registration changed;
- no reviewed Windows or Linux CTest membership changed;
- no package, ABI, or source-list surface changed.

## Day 12 Scope Input

Day 12 should plan validation around these touched surfaces:

- `tests/test_eigs.c`;
- `tests/test_ldlt_csc.c`;
- `tests/test_svd_partial_helpers.h`;
- `docs/planning/EPIC_10/SPRINT_113/PLAN.md`;
- `docs/planning/EPIC_10/SPRINT_113/WORKING_NOTES.md`;
- Sprint 113 artifacts through Day 11.

Required validation planning must include:

- focused eigensolver test rerun;
- focused LDLT CSC test rerun;
- focused SVD test rerun;
- full `make format && make lint && make test` gate because `.c` and `.h`
  files are changed;
- `git diff --check`;
- trailing whitespace and local Markdown link checks for Sprint 113 docs.
