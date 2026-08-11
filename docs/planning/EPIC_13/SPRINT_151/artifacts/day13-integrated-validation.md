# Sprint 151 Day 13 Integrated Validation

## Purpose

Day 13 validates the maintained partial-SVD corpus expansion as an integrated
surface: source-controlled corpus metadata, focused proof-owner tests,
generated-local oracle/report rows, active documentation wording, and the full
C quality gate required by the branch's C test changes.

## Validation Results

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Pass | Corpus manifests, schemas, and expected-result files validated successfully. |
| `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus` | Pass | Focused partial-SVD corpus proof owner passed `10` tests and `247` assertions. |
| `make build/test_svd && ./build/test_svd` | Pass | Affected broader SVD proof owner passed `114` tests and `2067` assertions. |
| `python3 tests/test_normalize_report_index.py` | Pass | Report-index unit coverage for Sprint 151 generated partial-SVD rows passed. |
| `python3 scripts/run_corpus_oracle.py --include-partial-svd` | Pass | Generated local oracle/report outputs were refreshed under `build/`. |
| `python3 scripts/normalize_report_index.py --family corpus --family oracle --check` | Pass | Normalized corpus/oracle report index passed with `105` rows. |
| `python3 scripts/normalize_report_index.py --family oracle --check-freshness` | Pass | Oracle freshness passed with `31` rows; expected generated-local strict-oracle warnings remained advisory. |
| Active-doc stale wording search | Pass | No active docs retained stale Sprint 140-only partial-SVD wording or stale row-count claims. |
| `make format && make lint && make test` | Pass | Full required C quality gate completed successfully. |

## Partial-SVD Corpus Evidence

The maintained partial-SVD corpus currently covers four deterministic fixtures:

| Fixture | Generated Oracle Rows | Focus |
| --- | ---: | --- |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | 8 | Clustered/repeated diagonal top-k values, subspace projectors, residuals, orthogonality, and tight-budget fail-closed behavior. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | 7 | Rank-deficient rectangular top-k values, rank, range projectors, residuals, and orthogonality. |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | 6 | Sparse low-rank output status, shape, retained entries, selected values, dense error, and sparse-vs-dense consistency. |
| `partial_svd_fail_closed_diag6_k2_v1` | 5 | Non-repeated tight-budget non-convergence, no partial arrays on failure, default recovery, values, and residuals. |

The generated-local partial-SVD oracle surface is `26` rows across those four
fixtures. The total generated oracle output is `29` rows when the QR oracle
rows are included by the same command.

## Documentation Claim Check

The active documentation now describes the selected fixture-family evidence
instead of a single Sprint 140 partial-SVD fixture. The stale wording search
covered `README.md`, `docs/algorithm.md`, `docs/cookbook.md`,
`docs/maintainer_guide.md`, `docs/solver_selection.md`, and
`tests/corpus/**/*.md` and found no matches for stale phrases such as
`one partial-SVD`, `partial_svd_row_count=8`, or `eight generated`.

## Non-Claims Preserved

Day 13 validation does not claim broad partial-SVD correctness, raw singular
vector identity, sign/orientation/phase parity, arbitrary basis ordering, broad
sparse-output optimality, convergence-rate guarantees, portable iteration
counts, external-library parity, hosted CI proof, package/ABI support,
performance, or state-of-the-art status.

## Day 14 Handoff

- Use this Day 13 validation artifact as the Sprint 151 closeout baseline.
- Preserve the four-fixture, `26`-row partial-SVD generated-local claim in the
  retrospective unless Day 14 validation changes the row set.
- Keep generated-local oracle rows scoped as local-only evidence until strict
  generated freshness comparison is promoted beyond advisory warnings.
