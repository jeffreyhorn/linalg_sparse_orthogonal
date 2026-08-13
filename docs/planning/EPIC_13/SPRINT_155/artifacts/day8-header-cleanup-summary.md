# Sprint 155 Day 8 Header Cleanup Summary

## Purpose

Day 8 applied the first public-header cleanup tranche from the Day 7 contract.
The edit scope was limited to Doxygen/comment text in:

1. `include/sparse_ldlt.h`
2. `include/sparse_ic.h`

No public declarations, signatures, typedefs, enum values, struct fields,
macros, include guards, installed header names, or exported names were changed.

## LDLT Header Cleanup

`include/sparse_ldlt.h` was updated to make the call-site contract easier to
scan while preserving existing API behavior:

- shortened the top-level Bunch-Kaufman and repeated-run handoff prose;
- clarified that `sparse_ldlt_t` is an owned one-shot factor object separate
  from the `sparse_analysis.h` repeated-run lifecycle;
- condensed backend selector wording for AUTO, linked-list, CSC, and the
  empty-matrix linked-list no-op edge case;
- replaced long historical options-layout prose with a concise source-rebuild
  warning and preserved the existing source-compatibility statement;
- tightened `used_csc_path` telemetry wording without changing semantics;
- clarified callback cancellation behavior and the current linked-list-only
  progress-emission boundary;
- clarified reset-on-entry, free-after-success, and error-return behavior for
  factor, solve, inertia, refine, and condition-estimate calls.

## IC Header Cleanup

`include/sparse_ic.h` was updated to clarify preconditioner usage and ownership
without changing the IC(0) surface:

- shortened IC(0) introduction language and avoided broader preference claims;
- clarified that IC(0) is intended for SPD systems and is a natural CG
  preconditioner;
- clarified reuse of `sparse_ilu_t` storage and the `sparse_ic_precond()`
  callback handoff;
- documented reset-on-entry and free-after-success behavior for
  `sparse_ic_factor()`;
- made the identity row/column state requirement visible at the factor call
  site;
- clarified that `sparse_ic_solve()` overwrites `z` and may alias `r`;
- added callback NULL and dimension-mismatch return details;
- clarified that `sparse_ic_free()` accepts NULL and zeroed structs.

## Declaration Preservation Evidence

Baseline and after captures:

- `day8-header-declarations-before.txt`
- `day8-header-declarations-after.txt`

The raw capture includes line numbers, so comment-only line shifts are expected.
The normalized comparison strips file/line prefixes and sorts declaration text:

- `day8-header-declarations-normalized-diff.txt`

The normalized diff is empty, proving the declaration-like text captured by the
Day 7 command plan was preserved.

## Claim Scan

The unsupported-claim scan from the Day 7 contract returned no matches across:

- `include/sparse_ldlt.h`
- `include/sparse_ic.h`
- `include/sparse_eigs.h`
- `include/sparse_analysis.h`

No shared-library, dynamic ABI, package-manager, runtime-loader, broad Windows
parity, portable-performance, external-parity, or state-of-the-art claims were
introduced.

## Validation

Commands run:

```sh
git diff --check
make format && make lint && make test
```

Results:

- `git diff --check` passed before and after the full gate.
- `make format && make lint && make test` passed.
- The final test output ended with `All tests passed.`

## Day 9 Handoff

Day 9 should apply the second public-header cleanup tranche to:

1. `include/sparse_eigs.h`
2. `include/sparse_analysis.h`

Use the Day 7 contract again:

- capture Day 9 declaration-like baselines before editing;
- keep edits comment-only;
- preserve all declarations, typedefs, enum values, struct fields, macros,
  include guards, ownership rules, error returns, defaults, and preconditions;
- run the unsupported-claim scan;
- run `git diff --check`;
- run `make format && make lint && make test`.
