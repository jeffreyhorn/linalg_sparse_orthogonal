# Sprint 29 Day 10 — Item 7: API Accessor Error Reporting (Design)

## Decision

**Option (b) — document the silent-zero-on-absent / silent-zero-on-error
contract explicitly in `sparse_get` and `sparse_get_phys` docstrings.**

Do NOT add `sparse_get_err()` variants and do NOT add a global
`sparse_get_last_error()` accessor.

## Inventory of the accessor / scalar-return surface

| Function | Return type | Error signalling |
|---|---|---|
| `sparse_get(mat, row, col)` | `double` | Silent 0.0 on NULL / OOB / absent |
| `sparse_get_phys(mat, row, col)` | `double` | Silent 0.0 on NULL / OOB / absent |
| `sparse_rows(mat)` / `sparse_cols(mat)` / `sparse_nnz(mat)` | `idx_t` | Silent 0 on NULL |
| `sparse_qr_condest(qr)` | `double` | `-1.0` sentinel on NULL / unfactored / rank=0; `INFINITY` on singular |
| `sparse_cond(A, &err)` | `double` | `*err` out-parameter; `INFINITY` on singular |
| `sparse_norminf(mat, &norm)` | `sparse_err_t` | Return-error-code, value via out-parameter |

The codebase uses **four distinct error-signalling patterns** in the
scalar-return API surface.  The most aligned-with-error-channel one
(`sparse_norminf`) is in the numerical-property family (computes a
quantity that can fail mid-computation), not the structural-accessor
family (reads stored state).

## Why option (b) — rationale

**1. Existing aesthetic + idiomatic usage.**  Of the 90 in-repo
callers of `sparse_get` (10 in `src/`, 80 in `tests/`), the
overwhelming majority pre-validate indices via `sparse_rows` /
`sparse_cols` then read with `sparse_get`.  The silent-zero-on-
absent contract IS the legitimate dominant case (rows/cols are
known in-bounds; the only "error" return is "entry was never
stored", which equals zero in the sparse-matrix sense).

The few callers that need bounds checking already have the typed
shape information available — they're inside library code paths
that compute the indices from validated structural data.

**2. No new state to manage.**  Option (a)'s `sparse_get_err(mat,
row, col, &val)` variant adds an API surface line per accessor (×
~5 routines).  Option (b)'s `sparse_get_last_error()` would
require thread-local error state — adds runtime complexity (a
`_Thread_local sparse_err_t` similar to `sparse_errno`'s pattern)
and a query call that's easy to forget on the caller side.  Either
sub-variant of option (a) has a higher API-and-runtime cost than
documenting the contract.

**3. Existing precedent: `sparse_cond` + `sparse_norminf` are
numerical routines.**  Their error channels exist because the
underlying computation can fail at the *numerical* level (singular
matrix, NaN propagation).  Accessor errors are at the *structural*
level — null pointer or out-of-bounds index — which callers can
trivially pre-check.  Aligning the structural-accessor family with
the numerical-routine family would over-engineer.

**4. `sparse_get_phys` and `sparse_get` are hot-loop primitives.**
Pre-Sprint-18 SVD code's `sparse_get_phys` use is 24 call sites in
`src/`, all inside elimination + matvec inner loops.  Adding an
out-parameter check (whether `*val` or `*err`) imposes a branch and
a memory write per call — would degrade the hot path.  The silent-
return contract is the right shape for a hot accessor.

## Rejection rationale for option (a)

**Option (a)**: add `sparse_get_err(mat, row, col, double *val)`
returning `sparse_err_t`.

Rejected because:

- **API surface duplication**: every existing accessor returning
  a scalar would need a `_err` sibling (`sparse_get_err`,
  `sparse_get_phys_err`, etc.).  Net 5+ new public functions.
- **Hot-path overhead**: the per-call branch + memory write to
  `*val` is measurable in tight inner loops (Sprint 17/18 cache-
  walking traces showed `sparse_get` was already in the top-5 most-
  executed library functions).
- **Caller migration burden**: existing 90+ call sites would
  either need to migrate (touching most of the test suite) or
  coexist with both variants forever.  Net negative ergonomics.

## Rejection rationale for option (a)'s `sparse_get_last_error()` variant

Rejected because:

- **Thread-local state surface**: adds a `_Thread_local sparse_err_t`
  parallel to the existing `sparse_errno`'s pattern.  The error
  "leaks" across function boundaries if callers don't query before
  the next accessor call.  Common foot-gun.
- **Caller-side discipline required**: a caller doing
  ```c
  double v = sparse_get(A, i, j);
  if (sparse_get_last_error() != SPARSE_OK) { ... }
  ```
  is verbose AND error-prone.  Forgetting the query is silent
  success (latest error doesn't reset until the next accessor).
- **No precedent in the surrounding API**: nothing else in the
  library uses a global last-error pattern besides `sparse_errno`
  for I/O, which is itself a niche.

## Implementation surface

**Day 10 lands:**

1. Updated docstrings on `sparse_get` (`include/sparse_matrix.h:194-205`)
   and `sparse_get_phys` (`include/sparse_matrix.h:180-188`) that
   explicitly state:
   - The silent-zero contract for NULL pointer / out-of-bounds /
     absent-entry.
   - That callers needing to distinguish these cases should pre-
     check via `sparse_rows`, `sparse_cols`, and (if needed) walk
     the row/col header lists to test for entry presence.
   - That this is intentional API design, not an oversight, and
     points to this decision doc.

2. Updated docstrings on `sparse_rows`, `sparse_cols`, `sparse_nnz`:
   note silent-0-on-NULL similarly.

3. New tests in `tests/test_sparse_matrix.c`:
   - `test_sparse_get_silent_zero_on_null`: asserts `sparse_get(NULL,
     0, 0) == 0.0`.
   - `test_sparse_get_silent_zero_on_oob`: build a 4×4 matrix, assert
     `sparse_get(A, 100, 100) == 0.0` (out-of-bounds returns 0).
   - `test_sparse_get_zero_indistinguishable_from_absent`: insert
     `0.0` at one position, leave another absent; both read as `0.0`.
     This pins the contract that the silent-zero ambiguity is by
     design.

4. `docs/planning/EPIC_2/SPRINT_29/accessor_error_decision.md` (this
   doc).

## What does NOT ship

- No new `sparse_*_err()` accessor variants.
- No `sparse_get_last_error()` thread-local state.
- No source-code changes to `sparse_get` / `sparse_get_phys` /
  `sparse_rows` / `sparse_cols` / `sparse_nnz` implementations
  (silent-zero is preserved bit-identical).

## References

- `docs/planning/EPIC_2/SPRINT_29/PLAN.md` Day 10 section.
- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 Item 7.
- `include/sparse_matrix.h:180-244` — current accessor docstrings.
- `include/sparse_matrix.h:286` — `sparse_norminf` reference
  pattern (error-out-parameter for numerical routines).
- `include/sparse_svd.h:240` — `sparse_cond(&err)` reference
  pattern (err-out-parameter for numerical routines).
