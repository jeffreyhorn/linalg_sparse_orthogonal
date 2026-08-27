# Sprint 184 Day 5: Advanced Contract Cleanup

## Purpose

Complete the second public-header contract cleanup pass for the selected
Sprint 184 family, `include/sparse_qr.h`, covering tolerance, workspace,
options/result, diagnostics, and cancellation wording without changing public
declarations.

## Scope

- Selected family: QR public header.
- Primary file changed: `include/sparse_qr.h`.
- Supporting records updated:
  - `docs/planning/EPIC_16/SPRINT_184/WORKING_NOTES.md`
  - `docs/planning/EPIC_16/SPRINT_184/artifacts/day5-advanced-contract-cleanup.md`
- Non-goals:
  - no public API declaration changes;
  - no implementation behavior changes;
  - no claims that QR-local rank diagnostics define a global rank policy;
  - no broad performance promises for sparse QR mode.

## Cleanup Summary

| Area | Day 5 cleanup |
| --- | --- |
| Tolerance | Normalized `sparse_qr_rank()` and `sparse_qr_rank_info()` wording so `tol > 0` and `tol <= 0` map to the implementation thresholds. |
| Workspace | Clarified that sparse QR mode uses O(m) working memory per active column while preserving the same public factorization contract. |
| Option behavior | Clarified that minimum-norm solve/refine use `opts` for their internal QR factorizations. |
| Result ownership | Clarified caller-owned outputs for null-space basis, R diagonal extraction, rank diagnostics, and minimum-norm solution/refinement outputs. |
| Diagnostics | Bounded rank and condition-estimate language to QR-local R-diagonal diagnostics. |
| Cancellation | Documented `SPARSE_ERR_CANCELLED` propagation for minimum-norm solve/refine when internal factorization progress callbacks cancel. |

## Implementation Cross-Check

The wording was checked against `src/sparse_qr.c` before recording the cleanup:

- `sparse_qr_factor_opts()` polls `opts->progress_cb` during Householder
  column elimination and returns `SPARSE_ERR_CANCELLED` after freeing
  intermediate state when the callback cancels.
- `sparse_qr_rank()` and `sparse_qr_rank_info()` use
  `tol * |R(0,0)|` for positive tolerances and
  `eps * max(m,n) * |R(0,0)|` for default tolerances.
- `sparse_qr_diag_r()` requires a caller-provided diagonal buffer.
- `sparse_qr_condest()` is based on QR R-diagonal data, so the header now
  avoids stronger condition-number guarantees.
- `sparse_qr_nullspace()` requires `null_dim`, accepts an optional `basis`
  output, and allocates temporary workspace internally.
- `sparse_qr_solve_minnorm()` and `sparse_qr_refine_minnorm()` build QR
  factorizations internally and therefore inherit option and cancellation
  behavior through those internal factorization calls.

## Declaration Preservation

Day 5 intentionally changed comments only. The focused declaration check strips
comments from `include/sparse_qr.h` and compares public QR declaration lines
against `HEAD`.

- Comment-stripped QR declaration hash before edit:
  `e1ec290dca650506021b144f03039a2ab528e91602cfc8f8d4c2821b9d6de6c0`
- Comment-stripped QR declaration hash after edit:
  `e1ec290dca650506021b144f03039a2ab528e91602cfc8f8d4c2821b9d6de6c0`
- Comment-stripped QR declaration diff: no output.

## Validation

- `make format && make lint && make test`: passed.
- `git diff --check`: passed.
- Focused comment-stripped QR declaration diff: passed with no output.

## Day 6 Handoff

Day 6 should design the declaration organization guardrail before any ordering
work. The default posture remains: preserve declarations and declaration order
unless an explicit organization-only exception is justified, guarded, and
recorded with before/after evidence.
