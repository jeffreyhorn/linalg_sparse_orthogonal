# Day 12 CI and Support-Surface Alignment

## Purpose

Reconcile CI workflows, public docs, benchmark docs, and maintainer guidance
against the widened Sprint 98 assurance model. The goal is to confirm that the
new bounded correctness and runtime/fill evidence is described consistently
without widening platform, coverage, benchmark, or public product claims.

## Surfaces Reviewed

Workflow surfaces:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

Local gate surfaces:

- `Makefile`
- `make quality-review-full`
- `make build/test_ldlt_csc && ./build/test_ldlt_csc`
- `make bench-reorder-sprint86`
- `make coverage`
- `make coverage-lcov`
- `make coverage-gcovr`

Documentation surfaces:

- `README.md`
- `INSTALL.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- Sprint 98 artifacts and working notes

## Alignment Findings

### Correctness Evidence

Sprint 98 added a bounded LDLT CSC external correctness lane:

- owner:
  - `tests/test_ldlt_csc.c`
  - `tests/ldlt_external_dense_reference.py`
- focused validation:
  - `make build/test_ldlt_csc && ./build/test_ldlt_csc`
- full post-code-change validation already run after the Day 5 C test change:
  - `make format && make lint && make test`

No workflow update is needed. The lane is exercised by normal Linux/macOS test
surfaces and by Windows CMake test registration/execution where supported.
Public docs do not need a new product claim for this proof lane.

### Runtime/Fill Evidence

Sprint 98 added a bounded reorder/fill artifact lane:

- owner:
  - `make bench-reorder-sprint86`
  - `bench_reorder --sprint86-slice --skip-factor`
- focused validation:
  - `make bench-reorder-sprint86`
- artifact owner:
  - `artifacts/day8-runtime-fill-comparison-batch1.md`
  - `artifacts/day9-runtime-fill-comparison-closeout.md`

No workflow update is needed. Linux `bench-fast` remains the supplemental PR
runtime signal, while the Sprint 98 artifact remains a local/planning
calibration artifact. It does not replace `make bench-canonical-report`.

### Coverage Evidence

Sprint 98 did not change coverage targets, thresholds, workflow behavior, or
coverage artifact expectations.

Current interpretation remains:

- coverage is supplemental
- coverage modes are tree-mutating
- return to normal direct/reviewed paths with `make clean`
- coverage percentages are assurance topology signals, not competitive product
  claims

No workflow or public-doc update is needed.

### Platform Workflow Claims

Existing workflow labels remain aligned:

- Linux remains the strongest reviewed source of truth, with supplemental
  runtime, benchmark, sanitizer, TSan, and coverage signals.
- macOS remains the enforced Apple Clang reviewed path plus supplemental
  Homebrew GCC and static-first install confidence.
- Windows remains the reviewed CMake-first consumer subset.

Sprint 98 does not add:

- reviewed Windows Makefile parity
- reviewed Windows install-validation parity
- macOS install/export parity
- cross-platform timing parity
- a new reviewed coverage lane
- a new reviewed benchmark lane

## Public Documentation Reconciliation

`README.md` and `INSTALL.md` already keep platform and benchmark claims
high-level:

- README points benchmark/reporting detail to `benchmarks/`.
- README points maintainer/quality policy to `docs/maintainer_guide.md`.
- INSTALL keeps platform setup and reviewed-platform interpretation bounded.
- Neither document claims broad LDLT external proof, portable benchmark timing,
  or widened coverage/workflow scope.

No public-doc edit is needed for Day 12.

## Maintainer Documentation Reconciliation

`docs/maintainer_guide.md` now contains:

- explicit LDLT CSC external proof ownership
- the Sprint 98 assurance topology snapshot
- benchmark-governance wording for the Sprint 98 reorder/fill artifact
- coverage and workflow topology marked audited but not widened

No additional maintainer-guide edit is needed on Day 12.

## Validation

Day 12 changed planning documentation only.

Hygiene checks:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_9/SPRINT_98
```

No `.c`, `.h`, Makefile, workflow, benchmark, script, or public-doc file was
modified for Day 12.

## Result

CI, local validation ownership, public docs, benchmark docs, and maintainer
guidance are aligned with the Sprint 98 assurance model. The widened evidence
remains bounded to:

- LDLT CSC external correctness on deterministic KKT fixtures
- reorder/fill calibration on the two-fixture Sprint 86 slice

Coverage and workflow surfaces were reviewed and remain unchanged.
