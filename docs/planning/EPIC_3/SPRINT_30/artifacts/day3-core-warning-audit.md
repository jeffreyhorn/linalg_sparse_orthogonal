# Sprint 30 Day 3 Core Warning Audit

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Audit Sprint 30’s four core warning files before editing them:

- `src/sparse_lu.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- `src/sparse_svd.c`

Day 3’s goal is to confirm whether the warning sites are compile-hygiene-only or signs of deeper numerical/API issues, and to pick a single cleanup idiom for Day 4-5.

## Audited Warning Sites

### `src/sparse_lu.c`

Site:

- `src/sparse_lu.c:478`

Context:

- `sparse_lu_condest()` computes `||A||_1` and returns `SPARSE_OK` with an infinite condition estimate when `norm_A == 0.0`.

Assessment:

- This is a compile-hygiene issue only.
- The behavior is semantically correct for the current API contract: a zero matrix is singular, so an infinite condition estimate is expected.
- No deeper inconsistency was found in the surrounding logic.

### `src/sparse_ldlt.c`

Site:

- `src/sparse_ldlt.c:1406`

Context:

- `sparse_ldlt_condest()` mirrors the LU condition-estimation logic and returns an infinite condition estimate when `norm_A == 0.0`.

Assessment:

- This is a compile-hygiene issue only.
- The behavior is consistent with the function’s documented purpose and with the LU path.
- No deeper numerical inconsistency was found in the surrounding logic.

### `src/sparse_qr.c`

Sites:

- `src/sparse_qr.c:1080`
- `src/sparse_qr.c:1237`
- `src/sparse_qr.c:1260`
- `src/sparse_qr.c:1520`

Contexts:

- `prev_rnorm = INFINITY` is used as an iterative-refinement sentinel.
- `info->condest = INFINITY` is used when rank diagnostics detect a singular/near-singular case.
- `return INFINITY` from `sparse_qr_condest()` is used when the rank-determined trailing diagonal is zero.

Assessment:

- All four sites are compile-hygiene issues only.
- The sentinel uses are ordinary “initialize with a value larger than any finite residual norm” patterns.
- The condition-estimate uses match the public header contract in `include/sparse_qr.h`.
- No deeper algorithmic or API inconsistency was found.

### `src/sparse_svd.c`

Sites:

- `src/sparse_svd.c:1687`
- `src/sparse_svd.c:1696`
- `src/sparse_svd.c:1704`
- `src/sparse_svd.c:1710`
- `src/sparse_svd.c:1721`

Context:

- `sparse_cond()` returns `INFINITY` for singular matrices and also returns `INFINITY` while reporting failure through `*err`.

Assessment:

- These sites are compile-hygiene issues only.
- The public header in `include/sparse_svd.h` explicitly documents this return contract.
- One behavior is mathematically debatable but not a Sprint 30 target: for `k == 0`, `sparse_cond()` currently returns `INFINITY`. That is established behavior and documented as part of the current contract, so Sprint 30 should not change it under the banner of warning cleanup.

## Cross-File Pattern Audit

A targeted search of `src/` and `include/` found that the warning-producing `INFINITY` uses are limited to the four audited source files. No parallel `NAN` or `INFINITY` pattern in the library proper requires a broader semantic refactor for Sprint 30.

Observed implementation pattern:

- Public documentation talks in terms of `INFINITY`.
- Source implementation uses the C `INFINITY` macro from `<math.h>`.
- On the current Apple Clang toolchain, `INFINITY` expands through `HUGE_VALF`, which triggers `-Wdouble-promotion` when assigned or returned as `double`.

## Day 3 Decision: Chosen Cleanup Idiom

For Day 4-5, use **`HUGE_VAL` for all implementation-side double infinity values in the four audited `src/` files**.

Why this idiom:

- `HUGE_VAL` is already the standard `<math.h>` double-valued positive infinity representation on the supported toolchains.
- It preserves the existing runtime semantics.
- It avoids the current Apple Clang `-Wdouble-promotion` warning caused by the float-typed `INFINITY` expansion.
- It is the smallest possible source edit, which is appropriate for Sprint 30’s compile-hygiene-only scope.

Examples of the intended transformation:

- `*condest = INFINITY;` -> `*condest = HUGE_VAL;`
- `double prev_rnorm = INFINITY;` -> `double prev_rnorm = HUGE_VAL;`
- `return INFINITY;` -> `return HUGE_VAL;`

## Rejected Alternatives

### 1. Cast every use: `(double)INFINITY`

Rejected because:

- It is mechanically correct, but noisier than necessary.
- It keeps the implementation coupled to the float-valued macro expansion that caused the warning in the first place.
- `HUGE_VAL` expresses the intended type more directly for `double` code.

### 2. Add a new project-wide infinity macro or helper in an internal header

Rejected for Sprint 30 because:

- The current library-proper warning set is only 11 lines across four files.
- A new shared macro/helper would add indirection without solving a broader problem that actually exists today.
- Sprint 30’s mandate is minimal-risk cleanup, not internal numeric-constant abstraction.

### 3. Change public documentation from “INFINITY” to “HUGE_VAL”

Rejected because:

- The public API contract is semantic, not implementation-typed.
- “Returns INFINITY” is clearer to readers than “returns HUGE_VAL”.
- The implementation can use `HUGE_VAL` while the documentation continues to describe the semantic result as infinity.

## Day 3 Conclusion

The Sprint 30 core warning cluster is purely compile-hygiene debt, not a hidden numerical bug. Day 4-5 should therefore:

1. make the smallest possible source edits,
2. preserve all existing public behavior and contracts,
3. and standardize on `HUGE_VAL` for implementation-side double infinity values in the four audited files.
