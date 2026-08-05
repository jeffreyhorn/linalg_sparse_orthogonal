# Maintained Numerical Corpus

This directory contains source-controlled metadata for maintained numerical
corpus fixtures. It records what each fixture is allowed to prove, how
generated fixtures are reproduced, how optional external data is skipped or
deferred, and where expected results live.

The corpus metadata is fixture-local evidence only. It does not claim broad
corpus completeness, SuiteSparse parity, external-library parity, broad QR or
SVD correctness, package/platform support, portable performance, coverage
completeness, or state-of-the-art status.

## Layout

| Path | Purpose |
| --- | --- |
| `manifests/fixtures.tsv` | Maintained fixture rows. |
| `manifests/generators.tsv` | Deterministic generated-matrix metadata rows. |
| `manifests/optional_data.tsv` | Optional external-data skip/defer policy rows. |
| `expected/` | Small committed expected-result rows for maintained fixtures. |
| `schemas/fixture_fields.md` | Fixture, generator, and optional-data field definitions. |
| `fixtures/` | Future promoted source-controlled matrix fixtures. |

Generated matrices, observed oracle rows, logs, report indexes, and local run
manifests belong under ignored `build/corpus/` or `build/corpus-reports/`, not
under this source-controlled directory.

## First Lane

Sprint 138 reserves the first durable fixture lane for:

- fixture key: `qr_rank_deficient_6x4_nullspace_v1`
- fixture family: `qr_rank_deficient`
- generator key: `qr_rank_deficient_6x4_nullspace_generator_v1`

The Day 5 rows are placeholders for layout validation. Later Sprint 138 days
must replace `TBD_*` values with generator hashes, maintained validation
commands, oracle rows, and expected-result semantics before treating the lane
as passing evidence.

## Optional Data

Optional external data is configured outside the repository with
`SPARSE_CORPUS_OPTIONAL_DATA_DIR`. Optional matrices, archives, extracted
datasets, and downloaded data must not be committed here.

Unavailable optional data is skip-policy evidence only. It is not solver pass
evidence.
