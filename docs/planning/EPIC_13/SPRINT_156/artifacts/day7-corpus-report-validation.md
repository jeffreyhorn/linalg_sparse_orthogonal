# Sprint 156 Day 7: Corpus And Report Validation

## Purpose

Validate the maintained QR and partial-SVD corpus/report evidence before the
Sprint 156 public claim audit. This artifact records the proof owners, current
row counts, freshness semantics, deferred corpus families, and non-claim
boundaries for generated local report evidence.

## Inputs Reviewed

- `docs/maintainer_guide.md`
- `README.md`
- `tests/corpus/README.md`
- `tests/corpus/manifests/fixtures.tsv`
- `tests/corpus/manifests/generators.tsv`
- `tests/corpus/manifests/optional_data.tsv`
- `tests/corpus/manifests/report_families.tsv`
- `tests/corpus/expected/*.tsv`
- `tests/corpus/schemas/report_index_fields.md`
- `scripts/validate_corpus_schema.py`
- `scripts/run_corpus_oracle.py`
- `scripts/normalize_report_index.py`
- `tests/test_qr_corpus.c`
- `tests/test_svd_partial_corpus.c`

## Commands Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Passed | Reported `tests/corpus ok`. |
| `make build/test_qr_corpus` | Passed | Built the focused QR corpus proof owner. |
| `./build/test_qr_corpus` | Passed | `14` tests, `0` failures, `0` skips, `258` assertions. |
| `make build/test_svd_partial_corpus` | Passed | Built the focused partial-SVD corpus proof owner. |
| `./build/test_svd_partial_corpus` | Passed | `10` tests, `0` failures, `0` skips, `247` assertions. |
| `make report-index-oracle-freshness` | Passed | Regenerated selected local oracle output and passed required oracle freshness with `54` normalized oracle rows. |
| `python3 scripts/normalize_report_index.py --family corpus --family oracle --check` | Passed | Normalized corpus/oracle index construction passed with `128` rows. |
| `python3 scripts/normalize_report_index.py --family corpus --family oracle --output build/report-index/day7-corpus-oracle-normalized.tsv` | Passed | Wrote ignored local inspection index with `128` rows. |
| `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | Passed | Required selected oracle freshness passed with `54` normalized oracle rows. |

Generated outputs under `build/` were left ignored and are not Sprint 156
source artifacts.

## QR Corpus Validation Summary

Maintained QR corpus evidence is fixture-local and local-only. It is backed by
the following selected fixture keys:

- `qr_rank_deficient_6x4_nullspace_v1`
- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`
- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

Current proof:

- schema validation passed;
- focused C proof owner passed with `14` tests and `258` assertions;
- selected combined oracle generation produced `23` QR solver-backed rows;
- all `23` QR solver-backed rows had `comparison_status=pass`;
- selected combined oracle freshness passed.

Claim scope:

- fixture-local rank, nullity, nullspace residual/subspace, minimum-norm
  status, residual, norm, and selected solution-value behavior;
- local-only generated evidence for the recorded command, commit, branch,
  platform, compiler, configuration, support tier, and fixture keys.

Non-claims:

- no broad QR correctness;
- no global rank-threshold policy;
- no broad rank-deficient least-squares or minimum-norm guarantee;
- no raw Q-basis, sign, orientation, or column-order parity;
- no SuiteSparse, LAPACK, NumPy, SciPy, Eigen, or external-library parity;
- no hosted CI, platform, package, ABI, performance, release, or
  state-of-the-art proof.

## Partial-SVD Corpus Validation Summary

Maintained partial-SVD corpus evidence is fixture-local and local-only. It is
backed by the following selected fixture keys:

- `partial_svd_clustered_repeated_diag8x6_k3_v1`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`
- `partial_svd_fail_closed_diag6_k2_v1`

Current proof:

- schema validation passed;
- focused C proof owner passed with `10` tests and `247` assertions;
- selected combined oracle generation produced `26` partial-SVD
  solver-backed rows;
- all `26` partial-SVD solver-backed rows had `comparison_status=pass`;
- selected combined oracle freshness passed.

Claim scope:

- fixture-local top-k value, rank, selected subspace-projector,
  triplet-residual, orthogonality, sparse-output shape/nnz/selected-value,
  Frobenius behavior, fail-closed status, no-partial-array diagnostic, and
  recovery behavior;
- local-only generated evidence for the recorded command, commit, branch,
  platform, compiler, configuration, support tier, and fixture keys.

Non-claims:

- no broad partial-SVD correctness;
- no broad vector/subspace, rectangular, nonsymmetric, repeated-spectrum,
  sparse-output, convergence-rate, partial-result, or pseudoinverse/minimum-
  norm claim;
- no raw singular-vector identity, sign/orientation/phase/basis-order parity;
- no LAPACK, NumPy, SciPy, Eigen, SuiteSparse, or external-library parity;
- no hosted CI, platform, package, ABI, performance, release, or
  state-of-the-art proof.

## Report Freshness And Index Notes

The selected Sprint 152 oracle freshness policy is current:

- selected local oracle command:
  `scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`;
- generated manifest commit:
  `00d62b13809bc2342555f1de8e01bcf072cae3f4`;
- generated manifest branch: `sprint-156`;
- generated manifest support tier: `local_only`;
- generated manifest row count: `52`;
- generated manifest solver families: `partial_svd`, `qr`, and `unknown`;
- QR solver row count: `23`;
- partial-SVD row count: `26`;
- generated-reference row count: `3`;
- selected fixture-key count: `10`;
- every generated oracle row reported `comparison_status=pass`.

The normalized report-index interpretation is scoped:

- `--family oracle --require-generated oracle --check-freshness` reported
  `54` rows because it includes `52` generated rows plus `2` source-controlled
  oracle contract rows;
- `--family corpus --family oracle --check` reported `128` rows because it
  includes `74` corpus rows plus `54` oracle rows for the selected combined
  local oracle output;
- the existing partial-SVD-only documentation count of `105` normalized
  corpus/oracle rows remains interpretable as the partial-SVD-only run shape:
  `74` corpus rows plus `31` oracle rows, not the selected combined QR plus
  partial-SVD freshness gate;
- freshness warnings with `generated_present_unchecked` are expected advisory
  diagnostics for generated rows; the required selected oracle freshness gate
  still passed.

No stale selected-row counts, missing selected fixture keys, or orphaned
selected oracle rows were found in the Day 7 run.

## Deferred Corpus-Family Queue

| Deferred family | Owner | Promotion criteria |
| --- | --- | --- |
| Broad QR rank-threshold policy | QR owner | Add tolerance-family design across scales and perturbations, expected rows, focused tests, generated oracle rows, and claim-boundary docs. |
| Broad QR rank-deficient least-squares and minimum-norm behavior | QR owner | Add solve-side fixtures beyond the selected bounded minimum-norm rows, expected residual/norm/value semantics, tests, and selected oracle freshness policy. |
| QR reordered/COLAMD corpus behavior | QR and reorder owners | Define ordering/fill semantics separately from nullspace and minimum-norm behavior; add tests and generated report rows with non-claim boundaries. |
| SuiteSparse or external QR corpus | Corpus and QR owners | Resolve optional-data provenance/licensing, add opt-in data handling, reviewed skip/defer semantics, and hosted or documented local evidence boundaries. |
| Broad partial-SVD repeated-spectrum and raw-vector behavior | SVD owner | Add subspace-safe or identity-safe oracle semantics, tolerances, tests, generated rows, and explicit non-claims. |
| Broad partial-SVD sparse-output/drop-tolerance optimality | SVD owner | Define expected optimality metric, sparse-output policy, focused fixtures, report rows, and promotion gate. |
| Partial-SVD convergence-rate and partial-result semantics | SVD owner | Define iteration-budget policy, failure/recovery guarantees, tests, generated rows, and public wording limits. |
| Hosted corpus/report evidence | CI and corpus owners | Promote selected local-only corpus/report commands to reviewed hosted lanes, then update support tiers and public wording. |

## Completion Criteria Check

- Maintained QR corpus claims are backed by current rows and focused test
  output.
- Maintained partial-SVD corpus claims are backed by current rows and focused
  test output.
- Report freshness wording matches generated local evidence and support tier.
- Generated rows remain ignored local artifacts, not committed proof.
- Deferred corpus work has owners and promotion criteria.
