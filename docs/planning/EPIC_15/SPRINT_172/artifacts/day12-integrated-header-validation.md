# Sprint 172 Day 12: Integrated Header Validation

## Purpose

Run the integrated Day 12 validation pass for the Sprint 172 LU public-header
cleanup, tutorial refinement signature fix, and focused drift guard.

## Scope

Validated artifacts and changed surfaces:

- `include/sparse_lu.h`
- `docs/tutorial.md`
- `scripts/check_lu_header_docs_guard.sh`
- Sprint 172 planning artifacts and working notes

The Sprint 172 code-facing changes include a public `.h` file, so the full C
quality gate is required for Day 12 validation.

## Full C Quality Gate

Ran:

```sh
make format && make lint && make test
```

Result: passed.

This confirms the LU header comment and heading cleanup remains compatible with
formatting, static-analysis, and the full local test suite.

## Focused Guard Validation

Ran:

```sh
bash scripts/check_lu_header_docs_guard.sh
```

Output:

```text
lu-header-docs-guard: header sections ok
lu-header-docs-guard: header declarations ok
lu-header-docs-guard: tutorial refinement signature ok
lu-header-docs-guard: unsupported claim absence ok
lu-header-docs-guard: passed
```

Result: passed.

The focused guard confirms the Day 7 LU section headings remain present, the
selected LU declaration names remain present, the Day 9 six-argument tutorial
refinement snippet remains present, the stale five-argument tutorial snippet is
absent, and unsupported claim wording is absent from the scoped LU header/docs
surface.

## Header Declaration Smoke Scan

Ran:

```sh
rg -n "^sparse_err_t |^typedef struct|^#ifndef|^#define" include/sparse_lu.h
```

Observed LU declaration surface:

```text
1:#ifndef SPARSE_LU_H
2:#define SPARSE_LU_H
58:typedef struct {
114:sparse_err_t sparse_lu_factor_opts(SparseMatrix *mat, const sparse_lu_opts_t *opts);
163:sparse_err_t sparse_lu_factor(SparseMatrix *mat, sparse_pivot_t pivot, double tol);
185:sparse_err_t sparse_lu_solve(const SparseMatrix *mat, const double *b, double *x);
209:sparse_err_t sparse_lu_solve_block(const SparseMatrix *mat, const double *B, idx_t nrhs, double *X);
236:sparse_err_t sparse_lu_condest(const SparseMatrix *mat_orig, const SparseMatrix *mat_lu,
262:sparse_err_t sparse_lu_solve_transpose(const SparseMatrix *mat, const double *b, double *x);
277:sparse_err_t sparse_apply_row_perm(const SparseMatrix *mat, const double *b, double *pb);
291:sparse_err_t sparse_apply_inv_col_perm(const SparseMatrix *mat, const double *z, double *x);
305:sparse_err_t sparse_forward_sub(const SparseMatrix *mat, const double *pb, double *y);
320:sparse_err_t sparse_backward_sub(const SparseMatrix *mat, const double *y, double *z);
343:sparse_err_t sparse_lu_refine(const SparseMatrix *mat_orig, const SparseMatrix *mat_lu,
```

Result: expected declaration names and include guard remain present.

The earlier Day 5 and Day 7 normalized declaration diffs remain the stronger
evidence that the header cleanup preserved the declaration-like surface across
the actual edit points.

## LU Refinement Signature Scan

Ran:

```sh
rg -n "sparse_lu_refine\([^\n]*\)" README.md docs/tutorial.md include/sparse_lu.h examples tests
```

Result: all direct documentation, example, and test references use the
six-argument public signature. The tutorial now uses:

```c
sparse_lu_refine(A, LU, b, x, 3, 1e-15);
```

## Claim Scan

Ran:

```sh
rg -n "state-of-the-art|external-library parity|portable performance|performance guarantee|package-manager support|shared-library support|dynamic ABI|runtime-loader|broad Windows parity|Windows Makefile parity|Windows pkg-config parity|LU CSR parity" include/sparse_lu.h docs/tutorial.md
```

Result:

- `include/sparse_lu.h`: no matches.
- `docs/tutorial.md`: matches are existing whole-file non-claim boundary
  statements that explicitly say what is not being claimed.
- The focused guard's scoped LU-section unsupported-claim check passed.

No package, ABI, runtime-loader, platform, performance, external-parity, LU
CSR parity, or state-of-the-art support claim was added by Sprint 172 Day 12.

## Deferral Guard Decision

The package-manager and static-package deferral guards were not required for
Day 12 because the Day 12 work did not change adoption, package, ABI,
runtime-loader, or platform-boundary wording. The Day 9 tutorial edit was a
signature-only correction, and the Day 11 script checks a scoped LU header/docs
surface rather than changing package or ABI policy.

## Whitespace Check

Ran:

```sh
git diff --check
```

Result: passed.

## Completion Status

Day 12 is complete. Required quality gates pass, the LU header cleanup remains
behavior-compatible, the focused guard passes, and the claim scan preserves the
Sprint 170/Sprint 171 package and ABI deferral boundaries.

## Day 13 Handoff

Use this integrated validation artifact as the evidence base for Day 13 claim
review. Day 13 should recheck that any closeout wording continues to describe
only header readability, tutorial signature alignment, and local guard coverage,
without promoting package-manager, shared-library, dynamic ABI, runtime-loader,
platform parity, performance, external-library parity, LU CSR parity, or
state-of-the-art claims.
