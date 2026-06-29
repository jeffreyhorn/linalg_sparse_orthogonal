# Day 11 Coverage-Topology Cleanup

## Purpose

Implement the Day 10 cleanup target by adding a compact Sprint 98 assurance
topology map to the maintainer guide. This reduces discoverability
fragmentation without moving proof code, changing benchmark behavior, or
widening coverage/workflow claims.

## Cleanup Implemented

Updated:

- `docs/maintainer_guide.md`

Added:

- `Sprint 98 Assurance Topology Snapshot`

The snapshot maps each Sprint 98 evidence class to:

- owner
- validation command
- interpretation boundary

## Topology Map Contents

The maintainer-guide snapshot now names:

- LDLT CSC external correctness:
  - owner:
    - `tests/test_ldlt_csc.c`
    - `tests/ldlt_external_dense_reference.py`
  - validation:
    - `make build/test_ldlt_csc && ./build/test_ldlt_csc`
  - interpretation:
    - bounded deterministic KKT solve comparison on `kkt5` and `kkt10`
- reorder/fill calibration:
  - owner:
    - `make bench-reorder-sprint86`
    - `bench_reorder --sprint86-slice --skip-factor`
  - validation:
    - `make bench-reorder-sprint86`
  - interpretation:
    - bounded two-fixture artifact; `nnz_L` is the fill field and
      `reorder_ms` is local timing context
- coverage topology:
  - owner:
    - `make coverage`
    - `make coverage-lcov`
    - `make coverage-gcovr`
    - Linux supplemental coverage workflow
  - interpretation:
    - audited but not widened; coverage remains tree-mutating and supplemental
- workflow topology:
  - owner:
    - `.github/workflows/ci.yml`
    - `.github/workflows/macos-ci.yml`
    - `.github/workflows/windows-ci.yml`
  - interpretation:
    - audited but not widened; reviewed, supplemental, and staged platform
      claims stay unchanged

## Deliberately Unchanged

No changes were made to:

- test code
- benchmark code
- Makefile targets
- workflow files
- coverage targets
- benchmark schema documentation
- public README claims

## Validation

Day 11 changed documentation only.

Hygiene checks:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_9/SPRINT_98 docs/maintainer_guide.md
```

No focused test or benchmark rerun is required because no proof surface was
renamed, moved, or changed.

## Result

The selected proof-owner fragmentation is reduced by giving maintainers one
compact Sprint 98 map that ties correctness, runtime/fill, coverage, and
workflow topology together while preserving all bounded claim fences.
