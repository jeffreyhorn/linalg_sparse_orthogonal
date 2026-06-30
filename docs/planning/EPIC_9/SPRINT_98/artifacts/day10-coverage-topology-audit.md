# Day 10 Coverage-Topology Audit

## Purpose

Inventory Sprint 98 proof-owner and comparison-owner topology after the
external correctness and runtime/fill lanes landed. The goal is to identify the
highest-value cleanup target for Day 11 without weakening proof ownership or
turning bounded evidence into broader coverage, workflow, or benchmark claims.

## Surfaces Audited

Correctness proof owners:

- `tests/test_chol_csc.c`
- `tests/chol_external_dense_reference.py`
- `tests/test_ldlt_csc.c`
- `tests/ldlt_external_dense_reference.py`
- `docs/maintainer_guide.md`

Runtime/fill comparison owners:

- `Makefile`
- `benchmarks/bench_reorder.c`
- `benchmarks/bench_amd_qg.c`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_9/SPRINT_98/artifacts/day8-runtime-fill-comparison-batch1.md`
- `docs/planning/EPIC_9/SPRINT_98/artifacts/day9-runtime-fill-comparison-closeout.md`

Coverage and workflow owners:

- `Makefile`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_9/SPRINT_98/artifacts/day1-assurance-baseline.md`
- `docs/planning/EPIC_9/SPRINT_98/artifacts/day3-proof-comparison-architecture-design.md`

## Current Topology

| Evidence class | Current owner | Sprint 98 role | Status |
|---|---|---|---|
| Cholesky CSC external correctness | `tests/test_chol_csc.c` plus `tests/chol_external_dense_reference.py` | prior bounded external lane used as design model | unchanged |
| LDLT CSC external correctness | `tests/test_ldlt_csc.c` plus `tests/ldlt_external_dense_reference.py` | new bounded external lane on `kkt5` and `kkt10` | landed and documented |
| Runtime/fill calibration | `make bench-reorder-sprint86` / `bench_reorder --sprint86-slice --skip-factor` | new Sprint 98 two-fixture artifact lane | landed and documented |
| Benchmark schema docs | `benchmarks/README.md` | local command/schema explanation | unchanged because schema did not change |
| Benchmark governance policy | `docs/maintainer_guide.md` | canonical/runtime/exploratory classification and claim fence | updated with Sprint 98 guardrail |
| Linux benchmark workflow | `.github/workflows/ci.yml` / `make bench-fast` | supplemental runtime signal | unchanged |
| Coverage workflow | `.github/workflows/ci.yml` / `make coverage` | supplemental, tree-mutating signal | unchanged |
| macOS and Windows workflow claims | platform workflow files plus maintainer guide | reviewed/staged/supplemental platform interpretation | unchanged |

## Fragmentation Findings

### Correctness Comparison Owners

The new LDLT CSC external lane is correctly family-local:

- the C harness owns solve/residual assertions
- the Python helper owns independent dense reference output
- the maintainer guide owns claim interpretation

There is no evidence that `tests/test_ldlt_csc.c` should be split or moved in
Day 11. The file is large, but moving the new harness now would widen risk and
weaken locality with the existing deterministic KKT helpers.

### Benchmark and Runtime/Fill Owners

The runtime/fill lane is structurally coherent:

- `bench_reorder` already emits the needed stable fields
- `make bench-reorder-sprint86` already names the bounded slice
- Day 8 owns the observed artifact
- Day 9 adds the maintainer guardrail

No benchmark C, Makefile, workflow, or schema cleanup is justified for Day 11.
The remaining fragmentation is discoverability: the Sprint 98 correctness and
runtime/fill evidence is described in separate maintainer-guide regions.

### Coverage-Related Docs and Scripts

No Sprint 98 work changed coverage targets, coverage thresholds, or coverage
workflow behavior.

Current coverage topology remains:

- `make coverage`, `make coverage-lcov`, and `make coverage-gcovr` are
  tree-mutating local modes
- Linux CI coverage is supplemental
- coverage percentages remain assurance topology signals, not competitive
  product claims
- returning to normal reviewed paths after coverage still requires
  `make clean`

No Day 11 coverage-target cleanup is justified.

### Workflow Labels and Validation Targets

Workflow labels remain consistent with the Sprint 98 evidence:

- Linux remains the strongest reviewed source of confidence
- Linux `bench-fast` and coverage remain supplemental signals
- macOS remains an enforced Apple Clang reviewed path plus supplemental GCC
  confidence
- Windows remains the reviewed CMake-first consumer subset

No Day 11 workflow edit is justified. Workflow changes would add platform
claim risk without improving Sprint 98 proof-owner clarity.

## Naming Cleanup vs Structural Cleanup

Naming cleanup candidates:

- add one compact Sprint 98 assurance-topology map to the maintainer guide
- cross-reference the new LDLT CSC correctness lane and reorder/fill artifact
  from the same map
- keep coverage and workflow unchanged but explicitly say they were audited and
  not widened

Structural cleanup candidates:

- move external dense-reference helper logic into shared code
- split `tests/test_ldlt_csc.c`
- add a generated reorder/fill report target
- add workflow artifact capture for `bench-reorder-sprint86`
- change coverage target structure

Day 10 rejects structural cleanup for Sprint 98 Day 11. Each candidate either
widens risk, changes execution behavior, or belongs to a later sprint after a
separate boundary.

## Selected Day 11 Cleanup Target

Add a compact Sprint 98 assurance-topology snapshot to
`docs/maintainer_guide.md`.

The snapshot should:

- live near existing proof-owner and benchmark-governance interpretation
  rather than in user-facing README content
- name the new LDLT CSC external correctness lane
- name the Sprint 98 reorder/fill artifact lane
- state that coverage and workflows were audited but not widened
- link evidence class to owner and validation command
- avoid changing test, benchmark, workflow, coverage, or Makefile behavior

This is the highest-value cleanup because Sprint 98 added two bounded evidence
lanes in different sections of the guide. A compact topology map improves
maintainability without moving proof code.

## Deferred Topology Cleanup Queue

Deferred:

- split `tests/test_ldlt_csc.c` only after a dedicated large-test extraction
  boundary
- share dense-reference Python helper code only if another solver family adds a
  third maintained external lane
- add a generated runtime/fill report target only if repeated artifacts become
  common enough to justify a Makefile surface
- capture `bench-reorder-sprint86` in CI only after deciding whether it is
  reviewed, supplemental, or artifact-only
- modify coverage targets only if a later sprint changes coverage ownership,
  threshold policy, or artifact expectations
- expand canonical reporting only after proving the wider report stays cheap
  and stable

## Validation

Day 10 changed planning documentation only.

Required hygiene:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_9/SPRINT_98
```

No `.c`, `.h`, Makefile, workflow, benchmark, script, or coverage target was
modified for this audit.
