# Sprint 184 Day 7: Coherent Header Sections

## Purpose

Apply the Day 6 declaration organization design to the selected Sprint 184
public header family, `include/sparse_qr.h`, while preserving the public
declaration set and making the intentional order change reviewable.

## Changes

Day 7 updated `include/sparse_qr.h` with lightweight section headings and two
bounded declaration moves:

- added `Options and factor object`;
- added `Factorization and lifecycle`;
- moved `sparse_qr_free()` next to factorization declarations;
- added `Q operations`;
- added `Solve operations`;
- moved `sparse_qr_solve_minnorm()` and `sparse_qr_refine_minnorm()` next to
  standard solve/refine declarations;
- replaced the previous diagnostics banner with `Rank, nullspace, and
  diagnostics`.

No public declaration names, signatures, struct fields, or behavior claims were
changed by the Day 7 organization pass.

## Before/After Declaration Order

| Before Day 7 | After Day 7 |
| --- | --- |
| `sparse_qr_opts_t` | `sparse_qr_opts_t` |
| `sparse_qr_t` | `sparse_qr_t` |
| `sparse_qr_factor()` | `sparse_qr_factor()` |
| `sparse_qr_factor_opts()` | `sparse_qr_factor_opts()` |
| `sparse_qr_apply_q()` | `sparse_qr_free()` |
| `sparse_qr_form_q()` | `sparse_qr_apply_q()` |
| `sparse_qr_solve()` | `sparse_qr_form_q()` |
| `sparse_qr_refine()` | `sparse_qr_solve()` |
| `sparse_qr_rank()` | `sparse_qr_refine()` |
| `sparse_qr_nullspace()` | `sparse_qr_solve_minnorm()` |
| `sparse_qr_free()` | `sparse_qr_refine_minnorm()` |
| `sparse_qr_diag_r()` | `sparse_qr_rank()` |
| `sparse_qr_rank_info_t` | `sparse_qr_nullspace()` |
| `sparse_qr_rank_info()` | `sparse_qr_diag_r()` |
| `sparse_qr_condest()` | `sparse_qr_rank_info_t` |
| `sparse_qr_solve_minnorm()` | `sparse_qr_rank_info()` |
| `sparse_qr_refine_minnorm()` | `sparse_qr_condest()` |

The order change is intentional and matches the Day 6 allowed changes:
factor cleanup now lives with lifecycle declarations, and minimum-norm solve
operations now live with the other solve declarations.

## Guard Evidence

Focused declaration checks stripped comments from `include/sparse_qr.h` and
compared public QR declaration lines.

- Ordered comment-stripped QR declaration hash after organization:
  `5650cb782761cdbaa18c75b29b477f7957a1893f80d85e3114c2158cbf7b1734`
- Sorted comment-stripped QR declaration-set hash after organization:
  `d50272d2e12f03f0869c8514809359e2d76ab585bb35ec5a2a936cb348432ec3`
- Sorted comment-stripped QR declaration-set diff against `HEAD`: no output.

The ordered hash changed because declarations moved. The sorted declaration-set
diff stayed empty, which verifies the public declaration set was preserved.

## Validation

- `make format && make lint && make test`: passed.
- `make docs-check`: passed.
- `git diff --check`: passed.

## Day 8 Handoff

Day 8 should audit examples and documentation surfaces against the reorganized
QR header. The key surfaces remain README API overview, API reference,
tutorial, cookbook, solver selection, examples README, and the QR examples for
least-squares, minimum-norm, and COLAMD workflows.
