# Sprint 184 Day 10: Reference Documentation Alignment

## Purpose

Bring higher-level reference documentation into agreement with the cleaned and
reorganized QR public header, `include/sparse_qr.h`, after the Day 9
example/tutorial alignment pass.

## Files Updated

| File | Update |
| --- | --- |
| `docs/api_reference.md` | Expanded the `sparse_qr.h` source-of-truth row to name QR lifecycle, minimum-norm, R-diagonal diagnostics, and cancellation contracts. |
| `docs/cookbook.md` | Aligned the QR routing/evidence note with selected comparison freshness for QR minimum-norm and compatible least-squares rows. |
| `docs/solver_selection.md` | Added minimum-norm output and R-diagonal diagnostics to the QR diagnostics handoff row. |

## Alignment Notes

### API Reference

The API reference row now matches the QR header organization:

- factorization/lifecycle;
- least-squares and minimum-norm solve paths;
- rank/nullspace;
- R-diagonal diagnostics;
- cancellation contracts.

### Cookbook

The cookbook QR routing note now matches the solver-selection and README
evidence boundary. It still keeps QR proof scoped to fixture-local confidence
and avoids broad QR or external-library parity claims.

### Solver Selection

The diagnostics handoff row now directs users to inspect the QR-specific output
surfaces that the header exposes: rank, residual, nullity/nullspace,
minimum-norm output, and R-diagonal diagnostics.

## Unsupported-Claim Check

The Day 10 edits did not add claims for:

- raw QR basis parity;
- global rank-threshold policy;
- broad rank-deficient solve behavior;
- broad minimum-norm behavior;
- SuiteSparse, LAPACK, NumPy, or SciPy parity;
- broad platform or Windows report freshness;
- package/ABI support;
- portable performance;
- state-of-the-art status.

## Validation

- `make docs-check`: passed.
- `make api-docs-local-only`: passed.
- `git diff --check`: passed.
- Sorted comment-stripped QR declaration-set diff against `HEAD`: passed with
  no output.

Day 10 made no new `.c` or `.h` edits, so the full C quality gate was not
rerun for this day.
