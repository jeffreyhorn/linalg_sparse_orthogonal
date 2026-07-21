# Sprint 129 Day 12 - Bidiagonal/Golub-Kahan Helper Gate

## Purpose

Day 12 reviewed Bidiagonal and Golub-Kahan test helpers to decide whether
Sprint 129 should move any helper ownership on Day 13. The goal was to prevent
helper movement from blurring separate evidence lanes for implicit
Householder-based bidiagonal reconstruction, explicit Golub-Kahan `U`/`V`
reconstruction, wide-matrix transpose behavior, and SVD QR-iteration checks.

## Current Helper Inventory

| Area | Current owner | Helper or pattern | Current use | Ownership notes |
| --- | --- | --- | --- | --- |
| Bidiagonal reduction reconstruction | `tests/test_bidiag.c` | `bidiag_reconstruction_error` | Reconstructs checked-in and synthetic bidiagonal reductions from implicit Householder vectors. | Owns bidiagonal-reduction semantics, including transposed wide-matrix handling. |
| Golub-Kahan explicit reconstruction | `tests/test_svd.c` | `gk_reconstruction_error` | Reconstructs `A` from explicit `U`, `V`, `diag`, and `superdiag` arrays in Golub-Kahan extraction tests. | Owns explicit-vector GK semantics, not implicit bidiagonal Householder replay. |
| Full SVD reconstruction and orthogonality | `tests/test_svd_helpers.h` | `tf_svd_*` helpers | Reconstructs full SVD products and checks `U`/`Vt` orthogonality. | Already SVD-owned; not a bidiagonal or GK extraction movement candidate. |
| Partial SVD residual evidence | `tests/test_svd_partial_helpers.h` | Partial-SVD helper routines | Checks Lanczos/partial-SVD residual, orthogonality, and rank evidence. | Partial-SVD owner; not a Day 13 movement target. |
| Wide Golub-Kahan behavior | `tests/test_svd.c` | Owner-local checks in wide GK tests | Skips or scopes reconstruction when the bidiagonalization path is transposed. | Must remain distinct from bidiagonal reduction reconstruction, which recursively handles transpose. |

## Semantics That Must Stay Separate

- `bidiag_reconstruction_error` replays implicit left and right Householder
  vectors from a `sparse_bidiag_t` object.
- `gk_reconstruction_error` reconstructs through explicit `U`, `V`, `diag`,
  and `superdiag` arrays produced by Golub-Kahan extraction.
- Bidiagonal wide-matrix coverage handles `bd->transposed` by transposing the
  input matrix and recursively checking a non-transposed view.
- Golub-Kahan wide extraction has owner-local skip/scoping behavior for
  transposed cases and should not inherit the bidiagonal reduction helper's
  recursive transpose policy.
- QR-iteration and full SVD reconstruction helpers use different products,
  diagnostics, and tolerances from both Bidiagonal reduction and GK extraction.

## Extraction Candidate Table

| Candidate | Decision | Reason |
| --- | --- | --- |
| Move `bidiag_reconstruction_error` to a Bidiagonal-owned helper header | Tentatively allowed for Day 13 | This can create an explicit Bidiagonal helper owner if the helper remains Bidiagonal-only and preserves the current transpose recursion exactly. |
| Move `gk_reconstruction_error` to `tests/test_svd_helpers.h` or a GK helper header | Defer | It has one owner and no proven cross-file reuse; moving it now would add indirection without paying down repeated implementation debt. |
| Unify Bidiagonal and GK reconstruction helpers | Reject | The two helpers validate different products and transpose policies. A shared helper would hide the distinction between implicit Householder replay and explicit `U`/`V` reconstruction. |
| Move SVD orthogonality helpers | No action | They already live in the SVD helper owner and are outside the Day 12 Bidiagonal/GK boundary. |
| Move partial-SVD residual helpers | Defer | Partial-SVD evidence belongs to the partial-SVD owner and should be handled by a later owner-specific gate. |

## Build And Source Ownership Requirements

If Day 13 moves a helper, the movement should be limited to one bounded helper
owner:

- Prefer a Bidiagonal-owned header such as `tests/test_bidiag_helpers.h` if
  `bidiag_reconstruction_error` is extracted.
- Keep the helper static/header-local or otherwise test-only; do not expose a
  production API or public test framework API.
- Preserve the current `sparse_get_phys` comparison, implicit Householder
  application order, transposed recursion, cleanup paths, and diagnostics.
- Keep GK explicit `U`/`V` reconstruction in `tests/test_svd.c` unless a
  separate GK owner gate proves cross-file reuse.
- Confirm whether any new source file requires `Makefile` or `CMakeLists.txt`
  registration. A header-only test helper should not require source-list
  changes, but the include ownership still needs validation.
- Do not update maintainer-guide ownership text unless the helper movement
  creates a durable owner boundary that future tests should follow.

## Day 13 Implementation Or Deferral Checklist

Day 13 can proceed only if the implementation stays within this checklist:

- Move at most one helper.
- Name the helper with Bidiagonal ownership, for example
  `tf_bidiag_reconstruction_max_error`, rather than a generic SVD or
  minimum-norm name.
- Keep all call-site matrices, tolerances, and assertion diagnostics visible.
- Preserve the `bd->transposed` behavior with the same effective matrix shape
  and cleanup semantics.
- Leave `gk_reconstruction_error`, QR-iteration checks, SVD helpers, and
  partial-SVD helpers in their current owners.
- Run focused Bidiagonal validation after any helper movement:
  `make build/test_bidiag && ./build/test_bidiag`.
- If any SVD/GK file is touched, also run
  `make build/test_svd && ./build/test_svd`.
- Because any helper movement changes `.c` or `.h` files, finish with
  `make format && make lint && make test`.

If any checklist item requires a generic helper shape or broad source-list
movement, Day 13 should explicitly defer the extraction instead of forcing it.

## Day 12 Decision

Day 12 did not move code. The only tentatively promotable Day 13 action is a
Bidiagonal-owned extraction of `bidiag_reconstruction_error`, and only if it
remains a narrow test helper that preserves the existing semantics exactly.
Golub-Kahan reconstruction, SVD reconstruction helpers, QR-iteration checks,
and partial-SVD residual helpers remain in their current owners.
