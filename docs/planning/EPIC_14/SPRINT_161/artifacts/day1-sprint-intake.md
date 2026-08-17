# Day 1 Sprint Intake

## Summary

Day 1 established Sprint 161's scope, starting evidence, target-family
inventory, non-goals, and Day 2 handoff. The sprint should publish one bounded
partial-SVD comparison family using the Sprint 160 generated-comparison
pattern while preserving local-only support tier and explicit non-claims.

## Source Path Note

The prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but the active
Sprint 161 plan and project-plan section are under
`docs/planning/EPIC_14/PROJECT_PLAN.md`. This artifact follows the current
Epic 14 path and branch layout.

## Project-Plan Inputs

| Area | Sprint 161 Input |
| --- | --- |
| Goal | Publish the first bounded partial-SVD comparison family with subspace-safe metrics and generated freshness checks. |
| Prerequisites | Sprint 159 hosted generated evidence path, Sprint 151 partial-SVD corpus families, and Sprint 160 comparison lessons. |
| Core items | Target selection, metric contract, harness extension, focused tests, report integration, docs alignment, validation and closeout. |
| Deliverables | First bounded partial-SVD comparison family, subspace-safe comparison contract, generated freshness proof, and Sprint 162 Windows package handoff. |

## Sprint 160 Handoff Extract

Sprint 160 recommends `partial_svd_diag6_k2` as the first partial-SVD
comparison target because the source-controlled dense helper exists, the
diagonal singular values are deterministic, top-k values are stable, and raw
singular-vector identity can be avoided.

Required setup before implementation:

1. Select exactly one target family.
2. Define selected row IDs and tolerances.
3. Use the descriptor-backed comparison runner pattern.
4. Add `report_families.tsv` metadata before treating generated rows as
   selected evidence.
5. Add focused runner tests for dispatch, output files, row IDs, metadata, and
   optional dependency context.
6. Add normalizer tests for complete, missing, unexpected, duplicate, stale,
   fail, and defer row states.
7. Keep C proof-owner tests unchanged unless implementation or fixture-helper
   behavior changes.
8. Preserve `local_only` support tier unless stronger wording is explicitly
   earned later.

## Current Surface Inventory

| Surface | File(s) | Current State |
| --- | --- | --- |
| External comparison runner | `scripts/run_external_comparison.py` | Descriptor-backed QR-only targets: `qr-minnorm` and `qr-compatible-ls`. |
| Dense SVD reference helper | `tests/svd_external_dense_reference.py` | Source-controlled SVD and partial-SVD fixture matrices, including `partial_svd_diag6_k2`. |
| Partial-SVD corpus docs | `tests/corpus/README.md` | Sprint 140/Sprint 151 partial-SVD oracle/corpus rows are documented with `partial_svd_row_count=26`. |
| Report-family metadata | `tests/corpus/manifests/report_families.tsv` | QR comparison rows exist; no partial-SVD comparison row is selected yet. |
| Report normalizer | `scripts/normalize_report_index.py` | Selected QR comparison freshness exists; partial-SVD appears in selected oracle counts only. |
| Runner tests | `tests/test_run_external_comparison.py` | QR dispatch, artifact, row, and dependency behavior covered. |
| Normalizer tests | `tests/test_normalize_report_index.py` | Selected QR comparison row-state behavior covered; selected partial-SVD comparison row states not yet present. |
| SVD proof-owner tests | `tests/test_svd.c`, `tests/test_svd_partial_corpus.c`, `tests/test_svd_partial_helpers.h`, `tests/test_svd_partial_shared_helpers.h` | Solver and corpus behavior are already covered; no Day 1 code changes are needed. |
| Public and maintainer docs | `README.md`, `docs/maintainer_guide.md`, `docs/solver_selection.md`, `tests/corpus/README.md` | Partial-SVD corpus claims are bounded; later comparison wording must stay fixture-local and non-parity. |

## Candidate Target Families

| Candidate | Day 1 Disposition | Notes |
| --- | --- | --- |
| `partial_svd_diag6_k2` | Preferred first target | Deterministic diagonal fixture; top-2 singular values are stable; suitable for value/residual checks without raw vector identity. |
| `partial_svd_tall_diag_8x5_k3` | Defer | Good later rectangular target after first comparison publication path is proven. |
| `partial_svd_nonsym_rect10x8_k3` | Defer | Needs tighter metric review for nonsymmetric rectangular behavior. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | Defer | Repeated/clustered spectra require explicit subspace-safe handling. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | Defer | Projector/range behavior is valuable but too broad for first target selection. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | Defer | Sparse output/drop-tolerance semantics need separate non-claim handling. |
| `partial_svd_fail_closed_diag6_k2_v1` | Defer | Fail-closed and recovery semantics should follow a passing comparison family. |

## Explicit Non-Goals

Sprint 161 must not claim broad partial-SVD correctness, raw singular-vector
identity, vector sign/order identity, repeated-spectrum vector ordering,
convergence-rate superiority, partial-result guarantees, broad sparse-output
or drop-tolerance optimality, external-library parity, platform portability,
package-manager proof, shared-library ABI proof, performance superiority,
release proof, or state-of-the-art evidence.

## Assumptions

- The first comparison family remains `local_only`.
- The dense helper is a source-controlled fixture reference, not a NumPy,
  SciPy, LAPACK, or external-library parity baseline.
- Skip and defer rows are non-proof context.
- The runner should emit generated artifacts in the same shape as Sprint 160
  comparison outputs.
- `.c` and `.h` files should remain untouched unless implementation behavior
  changes.

## Stop Conditions

Stop before implementation if the selected family cannot provide exact row
IDs, metric names, tolerance semantics, support tier, artifact paths,
claim scope, and non-claims. Also stop if raw singular-vector identity or
external-library parity becomes necessary to make the target pass.

## Day 2 Handoff

Day 2 should select the first target family, run any needed helper probes,
document rejected candidates, and define initial row IDs, output path, support
tier, tolerance strategy, claim scope, and non-claims before code changes.

## Validation

Day 1 is documentation-only. Validation is limited to Markdown hygiene checks
for the Sprint 161 planning files.
