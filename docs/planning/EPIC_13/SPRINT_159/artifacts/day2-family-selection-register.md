# Day 2 Family Selection Register

## Scope

Day 2 selects the generated report families that may be promoted from
local-only freshness checks to reviewed hosted evidence. Selection is
intentionally narrower than the full report-index surface. A selected family
must tie to a concrete current claim, a bounded command, named artifacts, and
explicit non-claims.

## Source Inputs

| Input | Day 2 use |
| --- | --- |
| `tests/corpus/manifests/report_families.tsv` | Current row meanings, support tiers, freshness policies, artifact patterns, claim scopes, and non-claims. |
| `docs/maintainer_guide.md` | Current QR/SVD evidence table, selected oracle/comparison freshness gate docs, and report-index interpretation. |
| `docs/solver_selection.md` | Public QR and partial-SVD fixture-local evidence boundaries. |
| `README.md` | Public command inventory and QR/SVD evidence wording. |
| `docs/planning/EPIC_13/SPRINT_159/artifacts/day1-promotion-boundary.md` | Candidate command and report-family inventory. |

## Selection Result

| Family | Subfamily or slice | Decision | Rationale |
| --- | --- | --- | --- |
| `oracle` | `solver_backed` QR selected rows | Reviewed-hosted candidate | These rows map to current fixture-local QR rank, nullity, nullspace, residual, and minimum-norm wording. They are produced by one maintained gate and already have strict local freshness semantics. |
| `oracle` | `solver_backed` partial-SVD selected rows | Reviewed-hosted candidate | These rows map to current fixture-local partial-SVD clustered/repeated, rank-deficient projector, sparse low-rank, residual, orthogonality, fail-closed, and recovery wording. They are produced by the same maintained gate. |
| `comparison` | `qr_minnorm` | Reviewed-hosted candidate | This is a single fixture-local QR minimum-norm comparison against the selected source-controlled dense reference helper. It is narrow enough for hosted promotion review. |
| `oracle` | `generated_reference` | Supplemental-hosted candidate | These rows can help reviewers inspect generated reference context, but they should support selected solver-backed evidence rather than become the primary public claim surface. |
| `report_index` | `missing_generated` | Advisory-local | Missing-generated rows make absent local reports explicit. They do not manufacture pass evidence and should not become a hosted claim surface. |
| `corpus` | fixtures, generators, optional data, expected rows | Advisory-local | Source-controlled metadata defines row identity and expected values. It is not fresh observed solver evidence. |
| `benchmark`, `sentinel`, `guardrail` | all current rows | Deferred | These are performance or local measurement families. Sprint 159 is not a performance-publication sprint. |
| `coverage`, `deadcode` | all current rows | Deferred | These are quality/reporting families, not oracle or comparison correctness evidence. |
| `package`, `ci`, `documentation`, `runtime_backend` | all current rows | Advisory-local or deferred | These rows describe package, workflow, documentation, or governance surfaces. They do not prove selected oracle/comparison freshness. |

## Selected Hosted Candidate Register

| Candidate ID | Family | Command | Claim surface | Current support tier | Proposed Day 2 target tier |
| --- | --- | --- | --- | --- | --- |
| S159-H01 | QR oracle solver-backed rows | `make report-index-oracle-freshness` | Fixture-local QR rank/nullity/nullspace and minimum-norm evidence named in `docs/solver_selection.md`, `docs/maintainer_guide.md`, and `README.md`. | local-only generated | reviewed-hosted candidate |
| S159-H02 | Partial-SVD oracle solver-backed rows | `make report-index-oracle-freshness` | Fixture-local partial-SVD top-k, rank, projector, residual, orthogonality, sparse low-rank, fail-closed, and recovery evidence named in `docs/solver_selection.md` and `docs/maintainer_guide.md`. | local-only generated | reviewed-hosted candidate |
| S159-H03 | QR minimum-norm comparison rows | `make report-index-comparison-freshness` | One fixture-local QR minimum-norm generated comparison for `qr_underdetermined_minnorm_2x4`. | local-only generated | reviewed-hosted candidate |
| S159-S01 | Oracle generated-reference rows | `make report-index-oracle-freshness` | Generated-reference context for maintained expected rows. | local-only generated | supplemental-hosted candidate |

## Claim-To-Family Mapping

| Claim surface | Selected family | Boundaries |
| --- | --- | --- |
| QR corpus proof covers six fixture-local rows: `qr_rank_deficient_6x4_nullspace_v1`, `qr_rankdef_duplicate_5x4_v1`, `qr_rankdef_dependent_row_4x3_v1`, `qr_underdetermined_minnorm_2x4`, `qr_minnorm_3x6_exact_values`, and `qr_minnorm_5x10_exact_values`. | `oracle/solver_backed` QR selected rows | Does not claim raw QR basis parity, global rank-threshold policy, broad rank-deficient solve, broad minimum-norm behavior, SuiteSparse, LAPACK, NumPy, SciPy, hosted platform portability, performance, package/ABI, or state-of-the-art evidence. |
| QR minimum-norm comparison checks only `qr_underdetermined_minnorm_2x4` against the selected source-controlled dense reference helper. | `comparison/qr_minnorm` | Does not claim broad QR parity, external-library ecosystem parity, or release/platform proof. |
| Partial-SVD corpus proof covers `partial_svd_clustered_repeated_diag8x6_k3_v1`, `partial_svd_rankdef_diag6x4_k2_range_projector_v1`, `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`, and `partial_svd_fail_closed_diag6_k2_v1`. | `oracle/solver_backed` partial-SVD selected rows | Does not claim broad SVD correctness, raw vector identity, broad sparse-output optimality, convergence-rate, partial-result behavior, external-library parity, platform, performance, package, ABI, or state-of-the-art evidence. |
| Maintainer report-index freshness can diagnose stale, missing, failing, partial, missing-solver-family, or missing-fixture-key selected output. | `oracle/solver_backed`, `comparison/qr_minnorm`, normalizer semantics | The normalized index is not release proof by itself; it becomes reviewed hosted evidence only for selected rows after CI, artifact, and docs changes. |

## Advisory And Local-Only Register

| Family | Reason it stays out of reviewed hosted promotion on Day 2 |
| --- | --- |
| `report_index/missing_generated` | It records absence and diagnostics; absence is not pass evidence. |
| `corpus/fixtures`, `corpus/generators`, `corpus/expected`, `corpus/optional_data` | These rows define source-controlled metadata, expected values, and skip/defer semantics. They are supporting context only. |
| `benchmark/canonical`, `sentinel/*`, `guardrail/large_matrix` | These are performance, runtime, or guardrail surfaces requiring separate methodology and threshold policy. |
| `coverage/src`, `deadcode/report` | These are quality-reporting surfaces and remain advisory unless selected by a different sprint. |
| `package/static_install`, `ci/reviewed_lanes`, `documentation/report_guidance`, `runtime_backend/governance` | These describe support-tier and governance context. They are not oracle/comparison pass evidence. |
| Broad `python3 scripts/normalize_report_index.py --check-freshness` without selected family filters | Too broad for Sprint 159 hosted claim evidence; it may include families whose support tier remains advisory, supplemental, or local-only. |

## Minimum Hosted Output Requirements

### Oracle Candidates

Required hosted outputs for S159-H01 and S159-H02:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`
- deterministic summary with generated-reference row count, QR row count,
  partial-SVD row count, selected fixture-key coverage, freshness result, and
  command list
- normalizer diagnostics for stale, missing, failing, partial,
  missing-solver-family, and missing-fixture-key states

Expected current local row counts from maintainer guidance:

- `3` generated-reference rows
- `23` `solver_family=qr` rows
- `26` `solver_family=partial_svd` rows
- `52` generated oracle rows total

### Comparison Candidate

Required hosted outputs for S159-H03:

- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/summary.md`
- `build/comparison/qr_minnorm/manifest.tsv`
- deterministic summary with selected fixture key, dependency status, row
  count, freshness result, and command list
- normalizer diagnostics for missing, stale, skipped dependency, and failing
  comparison states

## Day 3 Runtime Inputs

Day 3 should measure or plan measurement for these commands:

```sh
make report-index-oracle-freshness
make report-index-comparison-freshness
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/run_external_comparison.py --target qr-minnorm
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
```

Day 3 should capture:

- wall-clock runtime;
- output size;
- generated artifact path stability;
- dependency/skip behavior;
- whether output is deterministic enough for hosted summary comparison;
- initial timeout and artifact-retention recommendation.

## Stop Conditions Before CI Work

- S159-H01, S159-H02, or S159-H03 cannot complete locally.
- Runtime is too long or unstable for hosted PR validation.
- Generated outputs are too large or too noisy for artifact retention.
- Dependency skips are ambiguous or treated as pass evidence.
- Docs or report metadata cannot distinguish reviewed-hosted selected rows
  from local-only/advisory families.
- Normalizer semantics allow stale, missing, or failing selected rows to pass.

## Completion Check

- Selected families are claim-bearing and narrowly scoped.
- Local-only and advisory families are explicitly documented.
- Minimum hosted outputs are known for runtime and artifact planning.
- Day 3 has an approved target list for runtime budget work.
