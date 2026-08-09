# Sprint 147 Day 3 Corpus And Report Evidence Baseline

## Purpose

Day 3 captures the post-Epic-12 corpus and report evidence baseline for Epic
13 planning. The baseline separates source-controlled metadata from generated
local observations so later sprints can promote evidence only when the owning
command, freshness policy, and claim boundary support that promotion.

## Source-Controlled Corpus Baseline

### Fixture Rows

`tests/corpus/manifests/fixtures.tsv` currently has two maintained fixture
rows.

| Fixture | Owner | Scope | Validation Command | Non-Claim Boundary |
| --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1` | Sprint 138 | Fixture-local generated reference rank `3`, nullity `1`, and normalized null-vector residual metadata for a generated 6x4 rank-deficient matrix. | `python3 scripts/run_corpus_oracle.py` | No QR solver pass evidence from the manifest alone, no broad QR correctness, no raw-basis parity, no SuiteSparse parity, no broad corpus completeness, and no SVD correctness claim. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | Sprint 140 | Fixture-local partial-SVD clustered/repeated top-k subspace and budget behavior for a generated 8x6 diagonal matrix with `k=3`. | `python3 scripts/validate_corpus_schema.py` | No broad partial-SVD correctness, no raw singular-vector identity, no broad repeated-spectrum coverage, no external-library parity, and no performance claim. |

### Generator Rows

`tests/corpus/manifests/generators.tsv` currently has two deterministic
generator rows.

| Generator | Fixture | Algorithm | Floating Policy | Change Policy |
| --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_generator_v1` | `qr_rank_deficient_6x4_nullspace_v1` | `fixed_columns_c3_equals_c0_plus_c1` | Exact integer structure and values; exact rank/nullity; normalized null-vector residual tolerance `1e-10`. | Update generator version, fixture metadata, expected results, oracle rows, validation command, and docs together. |
| `partial_svd_clustered_repeated_diag8x6_generator_v1` | `partial_svd_clustered_repeated_diag8x6_k3_v1` | `fixed_diagonal_clustered_repeated_partial_svd` | Exact generated coordinates and values; top-k singular values, projector distance, residual, and orthogonality tolerances `1e-8`; status rows exact. | Update generator version, fixture metadata, expected results, oracle rows, validation command, and docs together. |

### Expected-Result Rows

`tests/corpus/expected/` currently has two expected-result files and eleven
ready-for-oracle rows.

| Expected File | Rows | Evidence Meaning |
| --- | ---: | --- |
| `qr_rank_deficient_6x4_nullspace_v1.tsv` | 3 | Expected rank, expected nullity, and expected normalized null-vector residual metadata for the QR fixture. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1.tsv` | 8 | Expected top-k singular values, left/right subspace projector distances, triplet residual, orthogonality residual, default-budget success, tight-budget failure, and no partial output arrays on tight-budget failure. |

Expected-result rows are source-controlled targets. They are not observed pass
evidence until a maintained proof command emits matching generated-local rows.

### Optional-Data Row

`tests/corpus/manifests/optional_data.tsv` currently has one optional-data
policy row: `suitesparse_rank_deficient_qr_subset_v1`.

The row documents that SuiteSparse rank-deficient QR optional data is disabled
by default, requires explicit license/redistribution review before use, and
does not prove SuiteSparse parity, external-library parity, or broad corpus
completeness when skipped.

## Report-Family Baseline

`tests/corpus/manifests/report_families.tsv` currently defines seventeen
family/subfamily rows:

| Family | Row Origin | Freshness Policy | Evidence Boundary |
| --- | --- | --- | --- |
| `corpus/fixtures` | source-controlled | `source_controlled` | Metadata identity and eligible fixture-local lanes only. |
| `corpus/generators` | source-controlled | `source_controlled` | Deterministic reproduction inputs and hashes only. |
| `corpus/optional_data` | source-controlled | `optional_data_skip` | Skip/defer policy only; no pass evidence from unavailable data. |
| `corpus/expected` | source-controlled | `source_controlled` | Expected targets only; no observed solver correctness. |
| `oracle/generated_reference` | generated local | `generated_compare_inputs` | Local fixture comparisons only for maintained expected rows. |
| `oracle/solver_backed` | generated local | `generated_compare_inputs` | Named fixture, command, commit, platform, compiler, configuration, and support tier only. |
| `benchmark/canonical` | generated local | `generated_local_advisory` | Local machine/configuration measurements only. |
| `sentinel/runtime` | generated local | `generated_compare_inputs` | Bounded local sentinel pass/fail rows only. |
| `sentinel/advisory` | generated local | `generated_local_advisory` | Local advisory measurements only. |
| `guardrail/large_matrix` | generated local | `generated_compare_inputs` | Maintained local guardrail rows only. |
| `deadcode/report` | generated local | `generated_local_advisory` | Maintainer classification aid only. |
| `coverage/src` | generated local | `generated_local_advisory` | Local coverage-tool summary only. |
| `package/static_install` | source-controlled | `source_controlled` | Static-first proof-owner command ownership only. |
| `ci/reviewed_lanes` | source-controlled | `hosted_ci_external` | Hosted reviewed lane definitions only; logs live outside source control. |
| `documentation/report_guidance` | documentation | `source_controlled` | Interpretation anchors only; no executable proof. |
| `report_index/missing_generated` | generated local | `generated_local_advisory` | Explicit absent-artifact rows only; no pass or freshness proof. |
| `runtime_backend/governance` | documentation | `source_controlled` | Runtime/backend policy boundary only. |

The baseline unit for reporting is the family/subfamily row, not only the
top-level family name.

## Proof-Owner Map

### QR Fixture-Local Closure

- Fixture key: `qr_rank_deficient_6x4_nullspace_v1`
- Source metadata: `tests/corpus/manifests/fixtures.tsv`,
  `tests/corpus/manifests/generators.tsv`, and
  `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv`
- C proof owner: `tests/test_qr_corpus.c`
- Generated-local proof command:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`
- Expected generated artifact:
  `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv`
- Claim boundary: rank `3`, nullity `1`, and solver-produced nullspace
  residual for this fixture only.

### Partial-SVD Fixture-Local Closure

- Fixture key: `partial_svd_clustered_repeated_diag8x6_k3_v1`
- Source metadata: `tests/corpus/manifests/fixtures.tsv`,
  `tests/corpus/manifests/generators.tsv`, and
  `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv`
- C proof owner: `tests/test_svd_partial_corpus.c`
- Generated-local proof command:
  `python3 scripts/run_corpus_oracle.py --include-partial-svd`
- Expected generated artifact:
  `build/corpus/oracle/partial_svd_clustered_repeated_diag8x6_k3_v1.oracle.tsv`
- Claim boundary: generated 8x6 clustered/repeated top-3 singular values,
  left/right top-k subspace projectors, triplet residuals, orthogonality,
  default-budget success, tight-budget fail-closed behavior, and no partial
  arrays on tight-budget failure for this fixture only.

## Generated-Local Evidence Classification

Source-controlled inputs:

- `tests/corpus/manifests/*.tsv`
- `tests/corpus/expected/*.tsv`
- `tests/corpus/schemas/*.md`
- `tests/test_qr_corpus.c`
- `tests/test_svd_partial_corpus.c`
- `scripts/validate_corpus_schema.py`
- `scripts/run_corpus_oracle.py`
- `scripts/normalize_report_index.py`
- report guidance in `tests/corpus/README.md`, `docs/maintainer_guide.md`, and
  `benchmarks/README.md`

Generated local/advisory outputs:

- `build/corpus/oracle/*.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`
- `build/report-index/normalized-index.tsv`
- `build/bench-reports/**`
- `build/deadcode/report.tsv`
- `coverage/coverage-src.info`

Generated rows can support only their exact family, command, commit, platform,
compiler, configuration, and support tier. Missing-generated rows and
optional-data skip rows are explicit absence or policy signals, not pass
evidence.

## Validation Command List

Core schema and index checks:

```sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --no-generated --check
python3 scripts/normalize_report_index.py --check
python3 scripts/normalize_report_index.py --check-freshness
```

Fixture-local corpus proof checks:

```sh
make build/test_qr_corpus && ./build/test_qr_corpus
make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
```

Focused report-family checks:

```sh
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --check-freshness
python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness
```

## Day 4 Handoff

Day 4 residual intake should treat the following as still unclosed unless a
later sprint provides direct evidence:

- broader QR corpus breadth beyond the named fixture;
- broader partial-SVD corpus breadth beyond the named fixture;
- external-library parity;
- generated report freshness as hosted/release evidence;
- broad performance or state-of-the-art claims;
- package, ABI, or platform claims outside the named proof owners.
