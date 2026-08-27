# Sprint 184 Day 6: Organization Guardrail Design

## Purpose

Design the declaration organization proposal and guardrails for the selected
Sprint 184 family, `include/sparse_qr.h`, before any declaration order changes.
Day 6 is intentionally a design step: it records what Day 7 may change, what it
must preserve, and how reviewers can distinguish intentional organization from
accidental API drift.

## Current Declaration Order

| Order | Declaration surface | Current role |
| ---: | --- | --- |
| 1 | `sparse_qr_opts_t` | QR factorization options, sparse mode, progress callback cancellation. |
| 2 | `sparse_qr_t` | QR factor object and owned internal factor data. |
| 3 | `sparse_qr_factor()` | Default factorization entry point. |
| 4 | `sparse_qr_factor_opts()` | Option-aware factorization entry point. |
| 5 | `sparse_qr_apply_q()` | Implicit Q application. |
| 6 | `sparse_qr_form_q()` | Explicit dense Q formation. |
| 7 | `sparse_qr_solve()` | Standard least-squares/direct/basic solve. |
| 8 | `sparse_qr_refine()` | Iterative refinement for standard QR solve. |
| 9 | `sparse_qr_rank()` | Rank estimate from R diagonal threshold. |
| 10 | `sparse_qr_nullspace()` | Null-space basis extraction. |
| 11 | `sparse_qr_free()` | Factor object cleanup. |
| 12 | `sparse_qr_diag_r()` | R-diagonal extraction. |
| 13 | `sparse_qr_rank_info_t` | Rank diagnostics result structure. |
| 14 | `sparse_qr_rank_info()` | Rank diagnostics computation. |
| 15 | `sparse_qr_condest()` | R-diagonal condition diagnostic. |
| 16 | `sparse_qr_solve_minnorm()` | Minimum-norm solve. |
| 17 | `sparse_qr_refine_minnorm()` | Minimum-norm refinement. |

## Organization Findings

The current order is usable and already has a clear diagnostic section, but two
placements are less discoverable than they need to be:

- `sparse_qr_free()` appears after rank/nullspace declarations even though it
  is part of the factor object lifecycle.
- `sparse_qr_solve_minnorm()` and `sparse_qr_refine_minnorm()` appear after the
  diagnostics section even though users scanning solve operations naturally
  compare them with `sparse_qr_solve()` and `sparse_qr_refine()`.

No duplicate declarations, stale declaration groups, or unsupported section
claims were found. The issue is declaration discoverability, not API meaning.

## Proposed Section Model

Day 7 may add lightweight QR section headings and, if the guard evidence is
recorded, may reorganize declarations into this model:

| Section | Declaration surfaces |
| --- | --- |
| Options and factor object | `sparse_qr_opts_t`, `sparse_qr_t` |
| Factorization and lifecycle | `sparse_qr_factor()`, `sparse_qr_factor_opts()`, `sparse_qr_free()` |
| Q operations | `sparse_qr_apply_q()`, `sparse_qr_form_q()` |
| Solve operations | `sparse_qr_solve()`, `sparse_qr_refine()`, `sparse_qr_solve_minnorm()`, `sparse_qr_refine_minnorm()` |
| Rank, nullspace, and diagnostics | `sparse_qr_rank()`, `sparse_qr_nullspace()`, `sparse_qr_diag_r()`, `sparse_qr_rank_info_t`, `sparse_qr_rank_info()`, `sparse_qr_condest()` |

This proposal keeps related operations together without changing declarations,
signatures, ownership rules, or behavior.

## Allowed Day 7 Changes

Day 7 may:

- add plain section headings to improve generated and source-level scanability;
- move `sparse_qr_free()` next to factorization/lifecycle declarations;
- move minimum-norm solve/refine next to standard solve/refine declarations;
- update declaration-order evidence if and only if the reorder is intentional;
- leave declaration order unchanged and add headings only if the guard evidence
  suggests reorder noise is not worth the review cost.

Day 7 must not:

- add, remove, rename, or change public declarations;
- change struct field order;
- change function signatures;
- widen QR evidence, performance, package, ABI, platform, or external-library
  claims;
- reorganize SVD or LDLT declarations under Sprint 184.

## Guard Policy

Use two levels of evidence:

1. **Declaration-set preservation.** Strip comments and compare public QR
   declaration lines before and after the change. If only headings/comments are
   added, the diff must be empty and the hash must remain:
   `e1ec290dca650506021b144f03039a2ab528e91602cfc8f8d4c2821b9d6de6c0`.
2. **Intentional order evidence.** If Day 7 moves declarations, record a
   before/after declaration order table in the artifact and explicitly state
   that the declaration set is unchanged while order changed by design.

If QR section headings are added, a future mechanical guard can follow the LU
precedent in `scripts/check_lu_header_docs_guard.sh`:

- require expected QR section headings;
- require every QR public declaration token;
- scan the QR header and scoped QR docs for unsupported package, ABI, platform,
  broad parity, or performance claims.

That guard belongs to the later Sprint 184 mechanical-guard day unless Day 7
needs it immediately to make an organization change reviewable.

## Fallback

If a proposed Day 7 reorder makes the declaration diff noisy, exposes an
unexpected generated-doc side effect, or requires a guard broader than Sprint
184 needs, fall back to headings-only organization and keep declaration order
unchanged.

## Validation

Day 6 changed planning artifacts only. No new `.c` or `.h` edits were made for
this day, so the full C quality gate was not rerun.

Validation performed:

- `git diff --check`: passed.
- Focused comment-stripped QR declaration diff against `HEAD`: passed with no
  output.
