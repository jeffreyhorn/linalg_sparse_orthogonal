# Sprint 98 Day 3: Proof/Comparison Architecture Design

## Purpose

Day 3 turns the Day 2 comparison ranking into a bounded proof/comparison
architecture. The goal is to define ownership, artifact locations, claim
boundaries, and validation expectations before widening any comparison lane.

No tests, benchmarks, scripts, workflows, source files, or headers are changed
on Day 3.

## Architecture Principles

1. Keep correctness proof separate from runtime/fill calibration.
2. Preserve external references as bounded oracles, not broad ecosystem
   comparison claims.
3. Prefer deterministic fixtures and explicit tolerances over larger corpora.
4. Keep benchmark reporting threshold-free unless a later sprint explicitly
   earns a threshold gate.
5. Keep public claims behind maintained evidence and maintainer-only detail.
6. Preserve Linux/macOS/Windows workflow asymmetry unless evidence changes.
7. Avoid proof-owner refactors unless they directly support the selected
   comparison lanes.

## Selected Implementation Lanes

| Lane | Selected target | Primary owner | Sprint 98 role |
|---|---|---|---|
| External correctness | LDLT CSC external correctness expansion | `tests/test_ldlt_csc.c` plus a new or adapted external helper | first widened maintained external-differential lane beyond Cholesky CSC |
| Runtime/fill comparison | bounded reorder/fill comparison lane | `benchmarks/bench_reorder.c`, `benchmarks/bench_amd_qg.c`, Make benchmark targets, and planning artifacts | bounded calibration evidence for fill/runtime on meaningful touched workloads |
| Support alignment | proof-owner and benchmark-governance docs | `docs/maintainer_guide.md`, `benchmarks/README.md`, Sprint 98 artifacts | keep public and maintainer language aligned after implementation |
| Workflow alignment | local reviewed commands and CI labels | `Makefile`, `.github/workflows/*.yml` only if needed | preserve proof ownership without widening platform claims |

## Correctness Architecture

### Selected Lane: LDLT CSC External Correctness

Day 3 selects an LDLT CSC external correctness lane, but keeps the exact Day 4
fixture and helper boundary open until the boundary freeze.

Preferred implementation shape:

| Piece | Proposed role |
|---|---|
| `tests/test_ldlt_csc.c` | owns the C harness, fixture invocation, tolerance assertions, and integration with existing LDLT CSC proof groups |
| new helper near `tests/chol_external_dense_reference.py` | owns bounded external dense/reference solve behavior if Day 4 confirms the helper is small and deterministic |
| deterministic fixtures | start with the smallest LDLT-appropriate fixtures that keep CI cost low |
| planning artifacts | record reference semantics, tolerance choices, and residual claim boundaries |
| maintainer docs | describe proof ownership only after the lane lands and validates |

### Reference Semantics

The first LDLT CSC external lane should prefer user-visible solve agreement:

- input:
  - one or two deterministic small fixtures
  - right-hand side derived from a known solution vector
- external reference output:
  - solution vector, or enough reference data to compare solution agreement
- maintained assertion:
  - max solution difference or residual strength under an explicit tolerance
- secondary assertion:
  - pivot/factor structure only if it is necessary to explain the solve
    comparison

The first lane should avoid making the external helper a clone of internal LDLT
factorization internals. A helper that must mirror Bunch-Kaufman pivot details,
CSC row storage, or implementation-specific permutation state is too coupled
for Day 5 implementation.

### Fixture Boundary

Day 4 should choose one of these boundaries:

1. SPD-as-LDLT fixture:
   - lowest implementation risk
   - easiest to compare against a dense solve
   - weaker as "indefinite LDLT" evidence
2. deterministic KKT fixture:
   - stronger LDLT-specific value
   - already present in `tests/test_ldlt_csc.c`
   - needs careful reference and tolerance design
3. small Matrix Market fixture:
   - closer to the current Cholesky external reference pattern
   - only acceptable if deterministic and fast

Day 3 preference:

- prefer deterministic KKT or another small LDLT-specific fixture if Day 4 can
  keep the external helper simple
- fall back to SPD-as-LDLT only if indefinite reference semantics would couple
  the helper too tightly to implementation internals

### Correctness Non-Goals

The LDLT CSC lane must not claim:

- broad LDLT external proof across all indefinite matrices
- parity with SuiteSparse, LAPACK, or another full solver stack
- validation of every 2x2 pivot path
- validation of every direct-family backend
- runtime or fill superiority

## Runtime/Fill Architecture

### Selected Lane: Reorder/Fill Comparison

Day 3 selects the reorder/fill lane because it already has bounded benchmark
and structural proof surfaces:

- `make bench-reorder-sprint86`
- `bench_reorder --sprint86-slice --skip-factor`
- `benchmarks/bench_reorder.c`
- `benchmarks/bench_amd_qg.c`
- reorder/fill assertions in existing tests

Preferred implementation shape:

| Piece | Proposed role |
|---|---|
| `bench_reorder` bounded slice | primary runtime/fill workload owner |
| planning artifact | captures selected metric interpretation and any Day 8/9 observed rows |
| `benchmarks/README.md` | benchmark-local schema or command documentation if implementation changes reporting |
| `docs/maintainer_guide.md` | authoritative claim fence if the maintained comparison model changes |
| CI workflow | no required Day 3 change; preserve as optional artifact capture only after explicit design |

### Allowed Metrics

The reorder/fill lane may report:

- fixture name
- ordering name
- `nnz_L`
- reorder path label
- fixture slice label
- ND base threshold
- reorder time
- factor time only when explicitly within the selected budget

The lane should prefer fill and structural comparison first, then runtime
context second. Runtime values should be interpreted as branch-local calibration
because they vary across machines.

### Runtime/Fill Non-Goals

The reorder/fill lane must not claim:

- universal reorder superiority
- portable timing parity
- cross-platform performance results
- package/platform product maturity
- full-corpus benchmark governance
- replacement of the canonical maintained benchmark report

## Artifact Ownership

| Artifact class | Location | Owner |
|---|---|---|
| external correctness helper | `tests/` beside existing reference helper | correctness proof owner |
| C harness assertions | selected `tests/test_*.c` proof owner | test owner |
| test fixtures | existing `tests/data/` or generated in-test deterministic fixtures | proof owner, not benchmark owner |
| benchmark workload output | benchmark stdout or generated local artifact | benchmark owner |
| sprint decision records | `docs/planning/EPIC_9/SPRINT_98/artifacts/` | sprint planning owner |
| benchmark schema docs | `benchmarks/README.md` | benchmark-local documentation owner |
| maintainer claim fence | `docs/maintainer_guide.md` | maintainer policy owner |
| public claims | `README.md` and `INSTALL.md` only if product-facing language changes | public documentation owner |
| workflow assertions | `.github/workflows/*.yml` only when a CI-owned proof changes | platform workflow owner |

## CI and Workflow Ownership

Day 3 does not require workflow changes.

If later implementation adds only test or benchmark code:

- local validation should prove the branch first
- Linux CI remains the strongest reviewed proof surface
- macOS and Windows language should remain unchanged unless their workflows
  actually run the widened lane

If later implementation adds a CI assertion:

- it must state whether the lane is reviewed, supplemental, or artifact-only
- it must avoid cross-platform parity language
- it must preserve Windows staged exclusions and macOS supplemental wording

## Claim Boundaries

### Correctness

Allowed:

- "Sprint 98 adds one bounded maintained LDLT CSC external correctness lane"
  only after the lane lands and validates.
- "The lane checks selected deterministic fixture solves against an external or
  independent reference under explicit tolerances."

Not allowed:

- "LDLT is externally validated across all indefinite matrices."
- "The project now has broad solver-family external proof."
- "The external helper proves runtime, fill, or package quality."

### Runtime/Fill

Allowed:

- "Sprint 98 records bounded reorder/fill calibration evidence on selected
  workloads."
- "The evidence is useful for before/after comparison and maintainers."

Not allowed:

- "The library is faster than competing libraries."
- "The benchmark result is portable across platforms."
- "The selected workload proves full benchmark superiority."

### Coverage

Allowed:

- "Coverage remains supplemental and useful for assurance topology."
- "Proof-owner topology is clarified where it supports the selected lanes."

Not allowed:

- "Coverage percentage is a competitive product claim."
- "Coverage workflow success proves cross-platform parity."

## Validation Plan

| Change type | Required validation |
|---|---|
| Day 3 docs-only design | `git diff --check` and trailing-whitespace scan on Sprint 98 planning files |
| LDLT external helper added or changed | focused LDLT CSC test command plus helper-level failure/skip checks |
| LDLT C harness changed | `make format && make lint && make test`; focused `test_ldlt_csc` rerun during development |
| benchmark C changed | `make format && make lint && make test`; focused affected benchmark command |
| benchmark script/report changed | focused script/report command, `git diff --check`, and full source validation only if C/header files changed |
| benchmark docs changed only | docs hygiene plus targeted claim scan |
| workflow changed | local equivalent command where possible; CI remains final syntax/platform proof |
| coverage target changed | focused coverage command or documented dry-run equivalent, then `make clean` before normal validation |

## Day 4 Boundary Freeze Inputs

Day 4 should freeze:

1. exact LDLT fixture class:
   - deterministic KKT, SPD-as-LDLT fallback, or small Matrix Market fixture
2. exact external helper shape:
   - new LDLT helper, shared dense solve helper, or no helper if a bounded
     independent reference is better
3. exact C harness insertion point inside `tests/test_ldlt_csc.c`
4. exact tolerances and skip/failure behavior
5. focused test command and full validation command for implementation days

Day 7 should freeze:

1. exact reorder/fill workload slice
2. exact metrics to preserve
3. whether any generated artifact is added or only planning output is captured
4. benchmark documentation changes, if any
5. focused benchmark command and validation command

## Day 3 Result

Sprint 98 now has a bounded proof/comparison architecture. The first
correctness expansion should be LDLT CSC external correctness, preferably on a
deterministic LDLT-specific fixture if Day 4 can keep the external reference
simple. The first runtime/fill expansion should be the bounded reorder/fill
lane, preserving threshold-free calibration language and avoiding broad
performance claims.
