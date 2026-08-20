# Sprint 172 Day 11: Mechanical Guard Implementation

## Purpose

Implement the Day 10 lightweight guard for LU header section drift and tutorial
LU refinement signature drift.

## Implemented Guard

- `scripts/check_lu_header_docs_guard.sh`

The guard is a focused shell script with deterministic text-presence and
scoped text-absence checks. It does not parse C, compile code, inspect
generated docs, or infer ABI behavior from comments.

## Positive Checks

The guard requires `include/sparse_lu.h` to contain the Day 7 workflow section
headings:

- `/* Options */`
- `/* Factorization */`
- `/* Solves */`
- `/* Conditioning and transpose solves */`
- `/* Advanced solver phases */`
- `/* Refinement */`

The guard also requires the selected LU declaration names to remain present:

- `sparse_lu_factor_opts`
- `sparse_lu_factor`
- `sparse_lu_solve`
- `sparse_lu_solve_block`
- `sparse_lu_condest`
- `sparse_lu_solve_transpose`
- `sparse_apply_row_perm`
- `sparse_apply_inv_col_perm`
- `sparse_forward_sub`
- `sparse_backward_sub`
- `sparse_lu_refine`

For `docs/tutorial.md`, the guard requires the six-argument refinement snippet:

```c
sparse_lu_refine(A, LU, b, x, 3, 1e-15);
```

## Negative Checks

The guard rejects the stale five-argument tutorial snippet:

```c
sparse_lu_refine(A, LU, b, x, 3);
```

It also rejects unsupported claim wording in:

- the selected LU header;
- the scoped `docs/tutorial.md` LU section.

The scoped tutorial check avoids false failures on valid whole-file non-claim
boundary text elsewhere in the tutorial.

Rejected claim terms:

- `package-manager support`
- `shared-library support`
- `dynamic ABI`
- `runtime-loader`
- `broad Windows parity`
- `Windows Makefile parity`
- `Windows pkg-config parity`
- `external-library parity`
- `portable performance`
- `performance guarantee`
- `LU CSR parity`
- `state-of-the-art`

## Wiring Decision

The guard was implemented as a standalone script and was not wired into the
Makefile on Day 11.

Reason: Sprint 172 still has integrated validation and closeout days remaining.
Keeping the guard directly invocable first makes the new failure behavior easy
to review before deciding whether it belongs in a broader quality target.

## Validation

Ran:

```sh
bash scripts/check_lu_header_docs_guard.sh
git diff --check
```

Output from the focused guard:

```text
lu-header-docs-guard: header sections ok
lu-header-docs-guard: header declarations ok
lu-header-docs-guard: tutorial refinement signature ok
lu-header-docs-guard: unsupported claim absence ok
lu-header-docs-guard: passed
```

`git diff --check` passed.

Day 11 did not modify `.c` or `.h` files, so the full C quality gate was not
required for this day.

## Completion Status

Day 11 is complete. Selected-header drift and tutorial signature drift are now
guarded by a focused script with clear maintainer-facing output.

## Day 12 Handoff

Day 12 should include:

```sh
bash scripts/check_lu_header_docs_guard.sh
```

in the integrated header validation command set, along with the existing
header declaration checks, targeted claim scans, and any required C quality
gate for touched `.c` or `.h` files.
