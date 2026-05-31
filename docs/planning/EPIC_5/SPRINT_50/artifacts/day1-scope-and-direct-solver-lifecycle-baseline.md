# Sprint 50 Day 1 Artifact: Scope and Direct-Solver Lifecycle Baseline

## Purpose

Capture the Sprint 50 starting baseline before direct-solver lifecycle design,
public-surface inventory, non-goal fencing, validation planning, and later
public API implementation begin.

## Starting Truth

Sprint 50 starts from a stable preserved Epic 4 and early Epic 5 baseline:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Epic 4 already left the major structural prerequisites Sprint 50 relies on:
  - bounded public repeated-run handles for iterative/eigensolver workloads
  - internal repeated-run groundwork from the Epic 4 reuse sprints
  - the maintainer-guide / README ownership boundary
  - an explicit residual follow-up journal in `EPIC_4_RETROSPECTIVE.md`
- Epic 5 already begins from a documented review and remediation queue:
  - `reviews/review-codex-2026-05-31.md`
  - `reviews/todo-codex-2026-05-31.md`

This means Sprint 50 is not opening with baseline recovery, documentation
redistribution, or generic repeated-run invention work. It is opening with the
bounded direct-solver lifecycle design work on top of an already-validated and
already-reviewed structural baseline.

## Day 1 Workstreams

Sprint 50 Day 1 confirms the sprint’s six bounded workstreams:

1. baseline recheck
2. direct-solver surface inventory
3. public lifecycle design
4. non-goal and compatibility fence
5. validation and landing plan
6. closeout and handoff

These come directly from the Sprint 50 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md` and stay consistent with the Epic 5
review/todo rule that the next direct-solver work should close lifecycle and
integration gaps, not reopen broad solver or documentation redesign.

## Highest-Value Authoritative Inputs

### Epic 5 planning and review inputs

- `docs/planning/EPIC_5/PROJECT_PLAN.md`
- `docs/planning/EPIC_5/SPRINT_50/PLAN.md`
- `docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
- `docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`

### Inherited execution-rule and residual-boundary inputs

- `docs/planning/EPIC_4/EPIC_4_RETROSPECTIVE.md`
- `README.md`
- `docs/maintainer_guide.md`
- `Makefile`
- `CMakeLists.txt`

### Highest-risk Day 1 direct-solver public lifecycle inputs

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `examples/example_analysis.c`

### Highest-risk Day 1 supporting implementation and factor-many inputs

- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt_csc.c`
- `tests/test_etree.c`

## Highest-Value Day 1 Conclusions

### 1. Sprint 50 is a direct-solver lifecycle design sprint, not a validation or reuse-baseline repair sprint

The preserved baseline remains explicit:

- `make quality-review-full` already exists as the strongest local reviewed
  baseline
- reviewed CMake parity remains exact and measurable at `53`
- Epic 4 already closed the internal repeated-run and documentation-ownership
  groundwork

Sprint 50 therefore starts from a preserved review-driven baseline rather than
from missing infrastructure or prior-sprint recovery work.

### 2. The repo already has one explicit public direct repeated-workflow precedent

The strongest Day 1 public lifecycle precedent is:

- `include/sparse_analysis.h`
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_refactor_numeric(...)`
  - `sparse_factor_solve(...)`

That means Sprint 50 does not need to invent the idea of an explicit direct
lifecycle from nothing. The main design job is to decide how this precedent
should relate to the still-dominant one-shot direct-solver caller model.

### 3. The main direct-solver tradeoff remains the compatibility-facing mutable `SparseMatrix` / in-place factor story

The public direct-solver story still splits across:

- one-shot copied/in-place factor paths:
  - LU
  - Cholesky
- separate-factor-container path:
  - LDL^T
- explicit analysis/factor/refactor path:
  - `sparse_analysis_t` / `sparse_factors_t`

This is the exact asymmetry Sprint 50 needs to make explicit. The problem is no
longer “there is no repeated direct workflow.” The problem is that the clearest
explicit repeated direct workflow is not yet the dominant public direct-solver
story.

### 4. The strongest Day 1 direct-solver support example already exists

`examples/example_analysis.c` is the clearest shipped demonstration of the
direct repeated-workflow path:

- analyze once
- factor numerically
- solve
- change values with the same sparsity pattern
- refactor numerically
- solve again

That makes it the strongest direct design precedent for Sprint 50. The sprint
does not need to hypothesize what a public direct lifecycle might look like; it
already has one real caller story to design around.

### 5. The Day 1 hotspot map is already explicit

The live direct lifecycle and later implementation/regression hotspots are
already concentrated:

- public direct lifecycle and one-shot headers:
  - `include/sparse_analysis.h` = `334`
  - `include/sparse_lu.h` = `327`
  - `include/sparse_cholesky.h` = `191`
  - `include/sparse_ldlt.h` = `310`
- later implementation hotspots:
  - `src/sparse_chol_csc.c` = `2194`
  - `src/sparse_ldlt_csc.c` = `2723`
- direct factor-many support surfaces:
  - `example_analysis.c` = `191`
  - `bench_refactor.c` = `159`
  - `bench_refactor_csc.c` = `388`
- later regression concentrations:
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_ldlt_csc.c` = `3637`
  - `tests/test_etree.c` = `2962`

That means Sprint 50 can stay a design sprint while still naming the exact
surfaces later implementation sprints must touch.

### 6. The key Day 1 architectural narrowing is now explicit

The main direct-solver lifecycle question is not whether to expose repeated-run
state at all. It is which bounded public model best closes the gap between:

- the current one-shot compatibility surfaces
- the explicit but under-centered analysis/factor/refactor precedent
- and the later factor-many / CSC / benchmark / example support surfaces

That gives Sprint 50 a clean Day 1 fence:

- preserve the current reviewed baseline and docs-policy ownership split
- inventory the public direct-solver lifecycle surfaces
- use the existing analysis/factor/refactor precedent as the main design anchor
- do not drift into Sprint 51 implementation or broad docs redistribution
