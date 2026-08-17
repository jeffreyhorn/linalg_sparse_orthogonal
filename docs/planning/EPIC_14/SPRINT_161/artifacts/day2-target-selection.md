# Day 2 Target Selection

## Summary

Day 2 selects `partial_svd_diag6_k2` as the first Sprint 161 partial-SVD
comparison publication target. The selection is intentionally narrow: a
diagonal 6x6 fixture with stable top-2 singular values and no need to claim raw
singular-vector identity, vector sign identity, repeated-spectrum ordering, or
external-library parity.

## Selected Family

| Field | Decision |
| --- | --- |
| Target name | `partial-svd-diag6-k2` |
| Fixture key | `partial_svd_diag6_k2` |
| Source helper | `tests/svd_external_dense_reference.py` |
| Matrix shape | 6x6 diagonal |
| Requested rank | 2 |
| Reference singular values | `9`, `6` |
| Output directory | `build/comparison/partial_svd_diag6_k2` |
| Generated study path | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| Report family | `comparison` |
| Subfamily | `partial_svd_diag6_k2` |
| Operation | `partial_svd` |
| Support tier | `local_only` |
| Claim scope | fixture-local partial-SVD diagonal top-k comparison only |

## Helper Probe

Command:

```sh
python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2
```

Observed output:

```text
OK 2
9
6
```

The helper confirms that the selected fixture has two stable top-k reference
singular values. Day 3 still needs to define the full comparison metric
contract before implementation, but the target does not require vector identity
or external-package parity to become useful generated evidence.

## Initial Expected Row Sketch

The exact row contract is a Day 3 deliverable, but Day 2 establishes the
initial selected row family so later work can review names before code changes.

| Initial row ID | Metric intent | Claim-bearing |
| --- | --- | --- |
| `comparison_partial_svd_diag6_k2_project_status_v1` | Project-side solver run completed for the selected fixture. | Yes |
| `comparison_partial_svd_diag6_k2_baseline_status_v1` | Source-controlled dense helper produced reference top-k singular values. | Yes |
| `comparison_partial_svd_diag6_k2_singular_value_0_v1` | Largest singular value equals the reference within tolerance. | Yes |
| `comparison_partial_svd_diag6_k2_singular_value_1_v1` | Second singular value equals the reference within tolerance. | Yes |
| `comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1` | Max absolute top-k singular-value delta stays within tolerance. | Yes |
| `comparison_partial_svd_diag6_k2_residual_norm_v1` | Solver residual for the selected top-k result stays within tolerance. | Yes |
| `comparison_partial_svd_diag6_k2_orthogonality_v1` | Solver-produced selected vectors satisfy bounded orthogonality diagnostics. | Diagnostic until Day 3 finalizes tolerance. |
| `comparison_partial_svd_diag6_k2_projector_diagnostic_v1` | Optional diagonal-safe projector or subspace diagnostic if implementation exposes enough data. | Diagnostic unless Day 3 promotes it. |

Day 3 may remove, rename, or split diagnostic rows. It must not add rows that
depend on raw singular-vector identity, sign identity, or ordering identity
beyond top-k singular-value ordering.

## Rejected And Deferred Candidates

| Candidate | Disposition | Reason |
| --- | --- | --- |
| `partial_svd_tall_diag_8x5_k3` | Defer | Adds tall rectangular shape and zero rows. It is a good later target after the generated publication path is proven. |
| `partial_svd_nonsym_rect10x8_k3` | Defer | Nonsymmetric rectangular behavior needs a stronger metric contract before publication. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | Defer | Clustered or repeated spectra increase the risk of accidental vector-order identity claims. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | Defer | Rank-deficient range-projector evidence is valuable but broader than the first comparison family. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | Defer | Sparse low-rank output and drop-tolerance behavior need separate claim boundaries. |
| `partial_svd_fail_closed_diag6_k2_v1` | Defer | Fail-closed and recovery semantics should follow an initially passing selected comparison family. |

## Fixture And Owner Map

| Area | Owner File(s) | Sprint 161 Use |
| --- | --- | --- |
| Reference fixture | `tests/svd_external_dense_reference.py` | Supplies selected top-k singular-value reference values. |
| Comparison runner | `scripts/run_external_comparison.py` | Later sprint days should add descriptor-backed partial-SVD target support. |
| Report metadata | `tests/corpus/manifests/report_families.tsv` | Later sprint days should add one selected partial-SVD comparison metadata row. |
| Freshness normalizer | `scripts/normalize_report_index.py` | Later sprint days should require the selected generated comparison rows. |
| Runner tests | `tests/test_run_external_comparison.py` | Later sprint days should add dispatch, artifact, row ID, metadata, and dependency-context checks. |
| Normalizer tests | `tests/test_normalize_report_index.py` | Later sprint days should add complete, missing, unexpected, duplicate, stale, fail, and defer row-state checks. |
| SVD proof-owner tests | `tests/test_svd.c`, `tests/test_svd_partial_corpus.c`, `tests/test_svd_partial_helpers.h`, `tests/test_svd_partial_shared_helpers.h` | Should remain unchanged unless implementation behavior changes. |
| Documentation | `README.md`, `docs/maintainer_guide.md`, `docs/solver_selection.md`, `tests/corpus/README.md` | Later sprint days should document the bounded comparison and non-claims. |

## Raw-Vector-Identity Non-Claims

The selected target permits singular-value and residual-oriented evidence
without claiming:

- raw left or right singular-vector identity;
- vector sign, orientation, or ordering identity;
- repeated-spectrum vector ordering;
- broad subspace correctness;
- NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library parity;
- performance, platform, package, ABI, release, or state-of-the-art proof.

## Completion Criteria Check

| Criterion | Status |
| --- | --- |
| Selected target is narrow enough to close in one sprint. | Met: diagonal top-2 fixture with existing helper. |
| Selection criteria are evidence-based and reproducible. | Met: source helper probe records stable `OK 2`, `9`, `6` output. |
| Rejected candidates have documented blockers or deferrals. | Met: all Day 1 candidates have explicit defer reasons. |

## Day 3 Handoff

Day 3 should finalize the metric contract. It should decide whether
orthogonality and projector/subspace rows are diagnostic or claim-bearing,
define exact tolerances, define stale/missing/defer/fail row-state semantics,
and lock the final selected row IDs before harness implementation begins.
