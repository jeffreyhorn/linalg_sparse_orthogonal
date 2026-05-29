# Sprint 48 Day 1 Artifact: Scope and Quality-Contract Baseline

## Purpose

Capture the Sprint 48 starting baseline before maintainer-guide design, README
reduction, quality-contract simplification, tutorial/header cross-reference
cleanup, and documentation sanity work begin.

## Starting Truth

Sprint 48 starts from a stable preserved Sprint 40/42/47 baseline:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code surfaces already exist:
  - `make deadcode`
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized because `deadcode*` still shares:
  - `build/deadcode-cmake`
  - `build/deadcode/`
- Sprint 42 already left behind the main lifecycle/cancellation guardrail:
  - public caveats still distinguish original/unfactored versus mutated/factored
    matrix use
- Sprint 47 already tightened the auxiliary benchmark/example/tooling surface:
  - modernized `bench_main`
  - bounded example safety cleanup
  - bounded dead-code tooling hardening

This means Sprint 48 is not opening with command-surface repair, solver
architecture work, or CI-baseline recovery. It is opening with bounded
documentation-ownership and quality-policy redistribution on top of a
preserved reviewed baseline and already-honest dead-code workflow contract.

## Day 1 Workstreams

Sprint 48 Day 1 confirms the sprint's seven bounded workstreams:

1. maintainer-policy home design
2. README reduction
3. maintainer-guide implementation
4. tutorial/header cross-reference reconciliation
5. quality-contract ownership simplification
6. docs sanity sweep
7. validation closeout

These come directly from the Sprint 48 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and stay consistent with the earlier
Epic 4 rule that maintainability cleanup should land through bounded ownership
improvements rather than broad command or CI redesign.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_48/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_47/artifacts/day14-closeout-and-handoff.md`

### Inherited execution-rule inputs

- `docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
- `docs/planning/EPIC_4/SPRINT_42/artifacts/day14-closeout-and-handoff.md`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### Highest-risk Day 1 documentation / quality-contract inputs

- `README.md`
- `docs/tutorial.md`
- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`
- `include/sparse_matrix.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_analysis.h`
- `benchmarks/README.md`
- `examples/README.md`

## Highest-Value Day 1 Conclusions

### 1. Sprint 48 is a documentation-ownership sprint, not a quality-command redesign sprint

The preserve-not-reopen boundary is explicit:

- preserve Sprint 40 validation-anchor truth
- preserve current reviewed/dead-code command semantics unless simplification
  requires clearer ownership wording
- preserve README as a strong user/operator entry point
- avoid broad CI redesign
- avoid dead-code workflow redesign

### 2. README is the main ownership-reduction hotspot

The live Day 1 size and seam evidence is explicit:

- `README.md` = `923`
- `Makefile` = `872`
- `docs/tutorial.md` = `413`
- `scripts/deadcode_report.py` = `550`
- `scripts/deadcode_workflow.sh` = `219`

The highest-density duplicated policy area is the current README block around:

- dead-code workflow
- reviewed local quality path
- cross-platform CI contract
- quality readiness checklist
- maintainer standards

That makes README the main Sprint 48 landing zone for:

- maintainer-policy extraction
- user-vs-maintainer scope reduction
- clearer ownership handoff to a new guide/policy home

### 3. The quality-contract is currently effective but over-distributed

The live repo now spreads the maintained quality contract across:

- `Makefile`
- `README.md`
- CI workflow files
- dead-code support scripts

The dry-run command surface confirms that `quality-review-full` still composes:

- reviewed Makefile path
- `deadcode-check`
- reviewed CMake parity path

That means Sprint 48 should not try to invent a new command topology first.
Its job is to simplify explanatory ownership so future changes require fewer
coordinated edits.

### 4. Lifecycle and behavior caveats are still repeated across multiple homes

The live Day 1 header/tutorial/README evidence shows duplicated caveat themes:

- original/unfactored matrix requirements
- factored-state restrictions
- lifecycle/cancellation caveats
- quality/dead-code interpretation rules

These currently appear across:

- README
- `docs/tutorial.md`
- public headers

That makes cross-reference reconciliation a real Sprint 48 seam rather than an
optional polish pass.

### 5. The main Day 1 documentation hotspot is role confusion, not missing information

The repo already contains the needed policy and quality information, but the
current home is often wrong:

- README carries both user-facing and maintainer-facing content
- command ownership is described in both Makefile output and README prose
- dead-code meaning exists in scripts, Makefile help, and README

The Day 1 interpretation is therefore:

- Sprint 48 is about reducing duplication and clarifying where policy lives
- it is not about inventing new policy from scratch

### 6. The front-half order of the sprint is fixed

The correct early sprint order is:

1. baseline and seam inventory
2. maintainer-guide design
3. README reduction
4. maintainer-guide implementation
5. tutorial/header cross-reference cleanup
6. quality-contract ownership simplification

That ordering preserves Sprint 40's core rule: maintainability cleanup should
be guided by measured seams and an explicit validation anchor before broader
redistribution lands.
