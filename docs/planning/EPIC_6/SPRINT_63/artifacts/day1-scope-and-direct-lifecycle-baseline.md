# Sprint 63 Day 1: Scope and Direct-Lifecycle Baseline

Date: 2026-06-10
Branch: `sprint-63`

## Purpose

Freeze the Sprint 63 starting point before implementation work begins by
reconfirming the inherited Sprint 62 contract, the preserved reviewed
baseline, the strongest live LU and CSC lifecycle seams, and the most
important public/implementation/proof surfaces the sprint will touch next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_63/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_62/RETROSPECTIVE.md`
- `docs/planning/EPIC_6/SPRINT_62/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `ctest -N --test-dir build/quality-review-cmake`
  - `make -n quality-review-full`
- current live direct-lifecycle and CSC follow-through surfaces measured
  directly from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 63 starts from a frozen Sprint 62 close, not from renewed one-shot usability or configuration work

Sprint 62 already landed the first Epic 6 direct-usability package and closed
with the explicit repeated-run direct lifecycle intact. That means Sprint 63
is not reopening one-shot-wrapper debate or trying to widen repeated-run
support. It is the first bounded Epic 6 sprint centered on direct-lifecycle
uniformity, LU follow-through, CSC repeated-run coherence, and result-state
alignment.

### 2. The strongest local reviewed baseline remains the authoritative Sprint 63 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Sprint 63 should not invent a new validation story. It should inherit the
existing one and make later lifecycle and CSC edits obey it.

### 3. The highest-value Sprint 63 problem is concentrated in LU follow-through and CSC repeated-run seams

The live repo shows a clear concentration:

- strongest caller-facing lifecycle seams:
  - LU repeated-run interpretation
  - explicit `analysis` / `factors` lifecycle wording
  - CSC-backed direct repeated-run coherence expectations
- strongest implementation seams:
  - `src/sparse_analysis.c`
  - `src/sparse_lu.c`
  - `src/sparse_cholesky.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_ldlt_csc.c`
- strongest proof seams:
  - `tests/test_integration.c`
  - `tests/test_sparse_lu.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`

So the opening Epic 6 lifecycle batch should not pretend the entire repo is
equally affected. The highest-value work is LU lifecycle follow-through plus
CSC repeated-run uniformity.

### 4. Sprint 63 reduces cleanly to seven bounded workstreams

The project-plan scope collapses to:

1. internal path audit
2. LU lifecycle follow-through
3. CSC repeated-run uniformity
4. solve/refactor semantics alignment
5. benchmark/example proof refresh
6. regression expansion
7. validation and closeout

This is the right Day 1 shape because it turns a broad Epic 6 lifecycle goal
into a smaller implementation contract.

### 5. The strongest live Sprint 63 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- caller-facing docs and lifecycle story:
  - `README.md` = `983`
  - `docs/tutorial.md` = `464`
  - `docs/maintainer_guide.md` = `391`
- public direct and lifecycle headers:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_lu.h` = `359`
  - `include/sparse_cholesky.h` = `212`
  - `include/sparse_ldlt.h` = `334`
- strongest implementation/lifecycle seams:
  - `src/sparse_analysis.c` = `1020`
  - `src/sparse_lu.c` = `1034`
  - `src/sparse_cholesky.c` = `546`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_ldlt_csc.c` = `2127`
- strongest proof and workflow surfaces likely to matter in Sprint 63:
  - `tests/test_integration.c` = `2168`
  - `tests/test_sparse_lu.c` = `908`
  - `tests/test_chol_csc.c` = `4552`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_ldlt_csc.c` = `3680`
  - `benchmarks/bench_refactor.c` = `303`
  - `examples/example_analysis.c` = `210`

These are not all immediate edit targets, but they are the real Day 1 map for
where direct-lifecycle pressure now lives.

## Preserved Day 1 Non-Goal Fence

Sprint 63 Day 1 confirms the following non-goals before deeper work begins:

- no reopening the repeated-run workflow fence
- no broad configuration-surface rewrite in the same batch
- no packaging/platform widening disguised as lifecycle work
- no fake family uniformity that breaks explicit ownership or cancellation
  semantics
- no broad backend/AUTO-policy or benchmark-governance work unless it proves
  to be a real blocker to the lifecycle hardening path

## Day 1 Exit State

Sprint 63 now starts from one explicit direct-lifecycle implementation
baseline:

- the Sprint 62 direct-usability close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 direct-lifecycle claim has already narrowed to LU
  follow-through, CSC repeated-run seams, and solve/refactor semantics
- the next step is to rank those live lifecycle inconsistencies precisely
  before writing the bounded LU/CSC uniformity design
