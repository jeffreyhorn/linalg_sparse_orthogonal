# Sprint 184 Day 9: Example Contract Alignment

## Purpose

Align QR-facing example and tutorial documentation with the cleaned
`include/sparse_qr.h` contracts from Sprint 184 Days 4-7. Day 9 keeps the edit
surface narrow: Markdown documentation only, no executable example changes.

## Files Updated

| File | Update |
| --- | --- |
| `README.md` | Changed `sparse_qr_factor_opts()` QR API bullet to name COLAMD column reordering for unsymmetric/QR workflows. |
| `docs/tutorial.md` | Added QR factor/solve return-code handling, caller-owned output wording, cleanup wording, and minimum-norm internal-options note. |
| `docs/solver_selection.md` | Updated QR evidence boundary to include selected minimum-norm and compatible least-squares comparison rows without widening parity claims. |
| `examples/README.md` | Added a minimum-norm note that options apply to temporary internal QR factorizations, including progress cancellation. |

## Alignment Details

### README

The previous QR API bullet described `sparse_qr_factor_opts()` as using
"optional AMD column reordering." The cleaned header recommends COLAMD for
unsymmetric QR workflows and treats AMD/RCM/ND as accepted symmetric-ordering
options that form `A^T*A`. The README now names COLAMD for the QR-facing path.

### Tutorial

The QR tutorial snippet now mirrors the public header lifecycle:

- `sparse_qr_factor()` return status is checked before using the factor object;
- `sparse_qr_solve()` return status is checked before using solution output;
- solve failure releases factor data with `sparse_qr_free()`;
- `x` and `residual_norm` are called out as caller-owned outputs;
- `sparse_qr_free()` is described as releasing factor data stored inside the
  caller-owned QR object.

The tutorial also notes that options passed to minimum-norm solve/refine apply
to the temporary QR factorizations built internally by those routines.

### Solver Selection

The QR evidence boundary now matches the newer README/cookbook framing:

- selected QR corpus proof remains fixture-local;
- selected comparison freshness includes QR minimum-norm and compatible
  least-squares rows named in
  `tests/corpus/manifests/selected_report_targets.tsv`;
- the boundary still rejects raw QR basis parity, global rank-threshold policy,
  broad rank-deficient solve, broad minimum-norm behavior, external-library
  parity, platform/package/ABI, performance, and state-of-the-art claims.

### Examples README

The minimum-norm example description now mentions that options, when provided,
apply to temporary internal QR factorizations and can propagate progress
cancellation.

## Deferred

No executable example edits were required by the Day 8 audit. Optional
`examples/example_colamd.c` return-code polish remains available for a later
implementation day if code examples are touched for another reason.

## Validation

- `make docs-check`: passed.
- `git diff --check`: passed.
- Sorted comment-stripped QR declaration-set diff against `HEAD`: passed with
  no output.

Day 9 made no new `.c` or `.h` edits, so the full C quality gate was not rerun
for this day.
