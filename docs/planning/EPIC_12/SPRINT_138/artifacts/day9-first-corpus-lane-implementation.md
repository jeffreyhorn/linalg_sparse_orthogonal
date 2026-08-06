# Sprint 138 Day 9 - First Corpus Lane Implementation

## Purpose

Day 9 lands the first deterministic corpus lane under maintained paths. It
turns the Day 8 QR fixture design into hash-backed generator metadata,
schema-valid expected-result rows, and mechanical validation coverage.

This implementation does not add observed oracle rows, a report command,
solver tests, generated outputs, optional external data, Matrix Market fixture
payloads, public documentation claims, or `.c`/`.h` changes.

## Implemented Lane

| Field | Value |
| --- | --- |
| Fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| Fixture family | `qr_rank_deficient` |
| Generator key | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| Matrix shape | `6x4` |
| Nonzeros | `14` |
| Expected rank | `3` |
| Nullity | `1` |
| Canonical format | `coo_zero_based_row_col_value_f64_text_v1` |
| Structure hash | `81496065f83410049f2c32556a3cb705375fe1e076112149a750489b4854f505` |
| Value hash | `2c6e0846a8a8bbe2c67786c25c029237acfccc891817ed3038b0b0e3646c36e2` |

## Generator Validation

`scripts/validate_corpus_schema.py` now includes a first-lane generator
registry entry for `qr_rank_deficient_6x4_nullspace_generator_v1`. The
validator regenerates canonical structure/value text for the fixed 6x4 matrix,
computes SHA-256 hashes, and checks the manifest row.

Validated generator properties:

| Property | Check |
| --- | --- |
| Algorithm | Must be `fixed_columns_c3_equals_c0_plus_c1`. |
| Parameters | Must be `rows=6;cols=4;expected_rank=3;nullity=1;dependency=c3-c0-c1`. |
| Dimensions | Fixture row must be `rows=6` and `cols=4`. |
| Nonzeros | Fixture row must be `nnz=14`. |
| Rank/nullity | Fixture row must be `expected_rank=3` and `nullity=1`. |
| Structure hash | Must match SHA-256 of canonical row/column text. |
| Value hash | Must match SHA-256 of canonical row/column/value text. |
| Canonical format | Must be `coo_zero_based_row_col_value_f64_text_v1`. |

## Expected-Result Rows

| Oracle row ID | Status | Meaning |
| --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_rank` | `ready_for_oracle` | Rank expected result is schema-valid and waits for Day 10 observed oracle output. |
| `qr_rank_deficient_6x4_nullspace_v1_nullity` | `ready_for_oracle` | Nullity expected result is schema-valid and waits for Day 10 observed oracle output. |
| `qr_rank_deficient_6x4_nullspace_v1_projector_residual` | `placeholder_pending_oracle_command` | Projector/subspace expected result has a fixture-local tolerance, but waits for the maintained oracle command. |

`ready_for_oracle` is not pass evidence. It means the source-controlled
expected row is ready for a future observed comparison.

## Oracle Row Scaffolding

Day 9 keeps observed oracle output out of source control. The row-generation
scaffolding is:

| Surface | Status |
| --- | --- |
| Expected-result row IDs | Present in `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv`. |
| Observed oracle schema | Present in `tests/corpus/schemas/oracle_fields.md`. |
| Generator/hash validation | Present in `scripts/validate_corpus_schema.py`. |
| Future observed output path | `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv`. |
| Future report output path | `build/corpus-reports/index.tsv`. |

## Validation Evidence

Day 9 validation used:

```sh
python3 scripts/validate_corpus_schema.py
git diff --check
```

Additional hygiene checked TSV row widths, trailing whitespace, focused
Markdown links, absence of generated corpus/report outputs, and absence of
`.c`/`.h` changes.

## Claim Boundaries

This lane may eventually support fixture-local QR rank, nullity, and
nullspace/subspace residual evidence after the Day 10 command emits observed
oracle rows.

It still does not claim:

- raw QR basis parity;
- broad QR correctness;
- global minimum-norm behavior;
- SuiteSparse or external-library parity;
- broad corpus completeness;
- SVD correctness;
- package, platform, performance, coverage, or state-of-the-art status.

## Day 9 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The first fixture lane exists under maintained paths. | Complete | Fixture, generator, expected-result, schema, and validator rows exist under `tests/corpus/` and `scripts/`. |
| Deterministic metadata can be validated. | Complete | Validator regenerates canonical first-lane metadata and checks hashes, dimensions, nnz, rank, nullity, parameters, and format. |
| The lane has expected-result and oracle-row ownership. | Complete | Expected row IDs are present, oracle schema exists, observed rows remain assigned to the future Day 10 command, and non-claims are preserved. |
