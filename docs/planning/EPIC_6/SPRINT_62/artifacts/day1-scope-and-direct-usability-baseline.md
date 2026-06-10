# Sprint 62 Day 1: Scope and Direct-Usability Baseline

Date: 2026-06-10
Branch: `sprint-62`


## Purpose

Freeze the Sprint 62 starting point before implementation work begins by
reconfirming the inherited Sprint 61 contract, the preserved reviewed
baseline, the strongest live direct-solver usability seams, and the most
important public/implementation/proof surfaces the sprint will touch next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_62/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_61/RETROSPECTIVE.md`
- `docs/planning/EPIC_6/SPRINT_61/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `ctest -N --test-dir build/quality-review-cmake`
  - `make -n quality-review-full`
- current live direct-solver usability and lifecycle surfaces measured
  directly from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 62 starts from a frozen Sprint 61 close, not from renewed configuration or workflow-boundary work

Sprint 61 already landed the first Epic 6 typed configuration package and
closed with the repeated-run workflow fence intact. That means Sprint 62 is
not reopening configuration-first debate or trying to widen repeated-run
support. It is the first bounded Epic 6 sprint centered on one-shot direct
wrapper clarity, lifecycle coherence, and direct caller-risk reduction.

### 2. The strongest local reviewed baseline remains the authoritative Sprint 62 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Sprint 62 should not invent a new validation story. It should inherit the
existing one and make later direct-usability edits obey it.

### 3. The highest-value Sprint 62 problem is concentrated in one-shot direct wrappers and lifecycle-adjacent helper seams

The live repo shows a clear concentration:

- strongest caller-facing direct seams:
  - one-shot direct wrappers
  - explicit `analysis` / `factors` lifecycle wording
  - factor-state ownership and reuse expectations
- strongest implementation seams:
  - `src/sparse_analysis.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_lu.c`
- strongest proof seams:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_sparse_lu.c`

So the opening Epic 6 usability batch should not pretend the entire repo is
equally affected. The highest-value work is the direct one-shot and explicit
lifecycle story.

### 4. Sprint 62 reduces cleanly to seven bounded workstreams

The project-plan scope collapses to:

1. direct-usability audit
2. lifecycle/wrapper coherence design
3. one-shot hardening
4. explicit lifecycle convergence
5. example/docs adoption
6. regression expansion
7. validation and closeout

This is the right Day 1 shape because it turns a broad Epic 6 usability goal
into a smaller implementation contract.

### 5. The strongest live Sprint 62 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- caller-facing docs and lifecycle story:
  - `README.md` = `982`
  - `docs/tutorial.md` = `454`
  - `docs/maintainer_guide.md` = `367`
- public direct and lifecycle headers:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_lu.h` = `337`
- strongest implementation/lifecycle seams:
  - `src/sparse_analysis.c` = `1020`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_lu.c` = `1040`
- strongest proof surfaces likely to matter in Sprint 62:
  - `tests/test_integration.c` = `1976`
  - `tests/test_chol_csc.c` = `4552`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_sparse_lu.c` = `908`

These are not all immediate edit targets, but they are the real Day 1 map for
where direct-usability pressure now lives.

## Preserved Day 1 Non-Goal Fence

Sprint 62 Day 1 confirms the following non-goals before deeper work begins:

- no reopening the repeated-run workflow fence
- no broad configuration-surface rewrite in the same batch
- no packaging/platform widening disguised as usability work
- no fake convergence between one-shot and explicit lifecycle direct APIs that
  breaks explicit ownership or compatibility
- no broad backend/AUTO-policy work unless it proves to be a real blocker to
  the direct hardening path

## Day 1 Exit State

Sprint 62 now starts from one explicit direct-usability implementation
baseline:

- the Sprint 61 configuration close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 direct-usability claim has already narrowed to one-shot
  wrapper, lifecycle, and factor-state seams
- the next step is to rank those live direct caller risks precisely before
  writing the lifecycle/wrapper coherence design
