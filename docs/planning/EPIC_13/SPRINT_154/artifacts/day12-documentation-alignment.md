# Day 12: Documentation Alignment

## Scope

Day 12 aligned maintainer, report-index, solver-selection, and README wording
with the Sprint 154 comparison harness and report-index integration.

The documentation now explains how to run and interpret the selected local QR
minimum-norm comparison without widening it into broad external-library,
platform, package, performance, ABI, hosted CI, or state-of-the-art proof.

## Documentation Updates

### README

Updated `README.md` to:

- list `make report-index-comparison-freshness` in the maintained Make command
  block;
- state that normalized comparison rows do not become release proof;
- describe the new narrow local comparison freshness gate in the QR corpus
  section;
- preserve the boundary that the comparison checks only
  `qr_underdetermined_minnorm_2x4` against the selected source-controlled
  dense reference helper.

### Maintainer Guide

Updated `docs/maintainer_guide.md` to:

- add a `Selected Comparison Freshness Gate` section;
- document `make report-index-comparison-freshness`;
- list the generated comparison artifacts under
  `build/comparison/qr_minnorm/`;
- list the six selected comparison rows required by the freshness gate;
- state that `skip` and `defer` rows are visible non-proof states;
- state that missing optional NumPy/SciPy dependencies cannot create pass
  evidence;
- add the comparison freshness command to common report-index checks;
- update the QR trust-boundary row to include
  `make report-index-comparison-freshness`.

### Report Documentation

Updated `benchmarks/README.md` to:

- include `make report-index-comparison-freshness` in the generated report
  handoff table;
- include `comparison` in the normalized report-index example;
- state that comparison rows remain fixture-local correctness evidence for the
  selected QR minimum-norm study only;
- state that local comparison rows do not become broad external-library parity.

### Solver Selection

Updated `docs/solver_selection.md` to:

- mention that `make report-index-comparison-freshness` adds one local QR
  minimum-norm comparison for `qr_underdetermined_minnorm_2x4`;
- preserve the non-claim that the lane is not broad QR or external-library
  parity.

## Stale Wording Search

Ran a focused stale/overbroad wording search across public docs, maintainer
docs, solver docs, algorithm docs, and Sprint 154 artifacts:

```sh
rg -n "state[- ]of[- ]the[- ]art|broad .*parity|ecosystem parity|NumPy parity|SciPy parity|package-manager|shared-library|performance superiority|hosted CI proof" \
  README.md docs/maintainer_guide.md benchmarks/README.md \
  docs/planning/EPIC_13/SPRINT_154 docs/solver_selection.md docs/algorithm.md
```

Result:

- matches in active public/maintainer docs are non-claims or scoped boundaries;
- no active wording was found claiming broad ecosystem parity;
- no active wording was found claiming state-of-the-art parity;
- no active wording was found converting local comparison output into hosted
  CI, package-manager, shared-library ABI, or portable performance proof.

## Interpretation Contract

The selected comparison documentation now has these boundaries:

- generated comparison output is local-only;
- the report family is `comparison/qr_minnorm`;
- required selected-row freshness covers six generated rows plus one contract
  row in normalized output;
- all six selected generated rows must be present and pass to support the
  fixture-local statement;
- `skip`, `defer`, `fail`, and `error` are not proof;
- optional NumPy/SciPy absence is not pass evidence;
- dirty worktree state remains explicit provenance, not release evidence.

## Non-Claims

The selected comparison lane does not claim:

- broad QR parity;
- NumPy parity;
- SciPy parity;
- LAPACK parity;
- SuiteSparse parity;
- Eigen parity;
- external-library ecosystem parity;
- hosted CI proof;
- release proof;
- platform portability proof;
- package-manager proof;
- shared-library or ABI proof;
- performance superiority;
- state-of-the-art status.

## Day 13 Handoff

Day 13 should run the selected comparison freshness gate, review generated
`study.tsv` and `summary.md`, and publish the first narrow study artifact or
study-summary documentation with these same local-only boundaries.
