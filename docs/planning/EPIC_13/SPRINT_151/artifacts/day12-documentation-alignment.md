# Sprint 151 Day 12: Documentation Alignment

## Purpose

Align user-facing, algorithm, corpus, oracle-schema, and maintainer
documentation with the Sprint 151 partial-SVD maintained corpus expansion.

Day 12 updates documentation only. It does not add new solver behavior,
generated evidence, hosted CI proof, package/ABI support, performance evidence,
external-library parity, or state-of-the-art support.

## Updated Documentation Surfaces

| File | Alignment |
| --- | --- |
| `README.md` | Expanded the high-level SVD and public API summaries from the single Sprint 140 clustered/repeated fixture to the maintained Sprint 140/Sprint 151 partial-SVD fixture set. |
| `docs/solver_selection.md` | Updated SVD workflow guidance to name the four maintained partial-SVD fixtures, proof owners, oracle command, and non-claims. |
| `docs/cookbook.md` | Updated SVD/low-rank workflow guidance with fixture-local sparse low-rank, projector/residual, fail-closed, and recovery wording. |
| `docs/algorithm.md` | Replaced the clustered/repeated-only algorithm boundary with the maintained partial-SVD corpus boundary and comparison semantics. |
| `docs/maintainer_guide.md` | Updated the SVD trust-boundary table and added a dedicated partial-SVD corpus maintenance section with row counts, commands, stale-report rules, and future update rules. |
| `tests/corpus/README.md` | Updated partial-SVD corpus lane documentation with all four fixture keys, generator keys, expected row counts, manifest expectations, stale-report signals, and residual register. |
| `tests/corpus/schemas/oracle_fields.md` | Added Sprint 151 partial-SVD expected-row families and comparison-semantics guidance. |
| `tests/corpus/expected/README.md` | Added expected-result guidance that rejects raw singular-vector identity and prefers subspace/residual/status/sparse-output summaries. |

## Maintained Partial-SVD Fixture Set

The current maintained partial-SVD corpus lane is:

| Fixture | Generated Oracle Rows | Primary Evidence |
| --- | ---: | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | 8 | clustered/repeated top-3 values, projectors, residuals, orthogonality, default success, tight-budget fail-closed, no partial arrays |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | 7 | rank-deficient rectangular top-2 values, rank, left/right projectors, residuals, orthogonality, default success |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | 6 | sparse low-rank status, shape, nnz, selected values, dense Frobenius error, sparse-vs-dense Frobenius difference |
| `partial_svd_fail_closed_diag6_k2_v1` | 5 | non-repeated tight-budget non-convergence, no partial arrays, recovery status, default top-2 values, residuals |

The total maintained partial-SVD generated-local row count is `26`.

## Proof Owners And Commands

Source-controlled owners:

- fixture and generator metadata: `tests/corpus/manifests/fixtures.tsv` and
  `tests/corpus/manifests/generators.tsv`;
- expected rows: `tests/corpus/expected/partial_svd_*.tsv`;
- focused C proof owner: `tests/test_svd_partial_corpus.c`;
- generated local oracle owner: `scripts/run_corpus_oracle.py --include-partial-svd`;
- normalized report/freshness owner: `scripts/normalize_report_index.py` and
  `tests/test_normalize_report_index.py`.

Maintainer commands:

```sh
python3 scripts/validate_corpus_schema.py
make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
```

## Preserved Non-Claims

The updated docs keep these explicit non-claims:

- broad partial-SVD correctness;
- raw singular-vector identity;
- sign, orientation, phase, or arbitrary basis-order parity;
- broad rank-deficient behavior;
- broad sparse-output correctness or optimality;
- storage or drop-tolerance optimality;
- convergence rates or portable iteration counts;
- useful partial outputs after non-convergence;
- hosted CI proof;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art support.

## Validation

Commands run:

```sh
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
python3 tests/test_normalize_report_index.py
rg -n "one partial-SVD|one generated partial-SVD|partial_svd_row_count=8|eight generated|lacks eight|clustered/repeated fixture|clustered/repeated lane" README.md docs/algorithm.md docs/cookbook.md docs/maintainer_guide.md docs/solver_selection.md tests/corpus -g "*.md"
git diff --check
```

Results:

- corpus schema validation passed;
- partial-SVD oracle/report generation passed;
- normalized corpus/oracle report-index check passed with `105` rows;
- oracle freshness check passed with `31` rows and expected current
  strict-oracle warnings for generated-present rows whose full freshness
  comparison remains pending;
- normalized report-index tests passed;
- stale wording search found no active-doc Sprint 140-only wording that should
  have been updated; historical planning artifacts were intentionally not
  rewritten;
- whitespace check passed.

No `.c` or `.h` files changed on Day 12, so the C quality gate was not
required.
