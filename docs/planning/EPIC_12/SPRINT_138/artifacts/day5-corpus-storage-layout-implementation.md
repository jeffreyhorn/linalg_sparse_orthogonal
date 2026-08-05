# Sprint 138 Day 5 - Corpus Storage Layout Implementation

## Purpose

Day 5 implements the maintained corpus storage skeleton designed on Day 4. It
adds source-controlled corpus directories, manifest skeleton files, first-lane
placeholders, expected-result placeholders, and layout documentation.

This implementation does not add a generator, oracle command, observed oracle
rows, report outputs, public documentation claims, optional external data, or
solver behavior changes.

## Added Source Layout

| Path | Status | Purpose |
| --- | --- | --- |
| `tests/corpus/README.md` | Added | Corpus ownership, row meaning, optional-data boundary, generated-output boundary, and first-lane placeholder notes. |
| `tests/corpus/manifests/fixtures.tsv` | Added | Fixture manifest header plus the first-lane placeholder row. |
| `tests/corpus/manifests/generators.tsv` | Added | Generator manifest header plus the first-lane generator placeholder row. |
| `tests/corpus/manifests/optional_data.tsv` | Added | Optional external-data manifest header. |
| `tests/corpus/expected/README.md` | Added | Expected-result ownership and non-claim policy. |
| `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` | Added | First-lane expected-result placeholder rows for rank, nullity, and future normalized null-vector residual comparison. |
| `tests/corpus/schemas/fixture_fields.md` | Added | Fixture, generator, and optional-data field definitions for the Day 5 skeleton. |
| `tests/corpus/fixtures/README.md` | Added | Placeholder documentation for future promoted stored matrix fixtures. |

## First-Lane Placeholder

| Field | Placeholder value |
| --- | --- |
| Fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| Fixture family | `qr_rank_deficient` |
| Generator key | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| Storage kind | `generated` |
| Rows / cols | `6` / `4` |
| Expected rank / nullity | `3` / `1` |
| Support tier | `local_only` |
| Validation command | `TBD_DAY10_CORPUS_ORACLE_COMMAND` |
| Generator hashes | `TBD_DAY8` |
| Introduced in | `TBD_DAY9` |

The placeholder rows are layout scaffolding only. They reserve stable row
shape and key names, but they are not pass evidence until later Sprint 138
days add deterministic generator metadata, expected hashes, oracle schema,
observed comparisons, and a maintained validation command.

## Generated and Optional Data Controls

| Control | Status |
| --- | --- |
| Generated outputs | No files under `build/corpus/` or `build/corpus-reports/` were created or committed. |
| Optional external data | No optional payloads, archives, downloads, or SuiteSparse reclassifications were committed. |
| Existing `.gitignore` | No change needed because no committed corpus `.mtx` fixture was added on Day 5. |
| Future committed `.mtx` fixtures | A later promotion must add explicit `.gitignore` exceptions for `tests/corpus/fixtures/**/*.mtx` if needed. |

## Initial Layout Validation Notes

| Check | Expected result |
| --- | --- |
| TSV header presence | `fixtures.tsv`, `generators.tsv`, `optional_data.tsv`, and first-lane expected-result TSV all have one header row. |
| First-lane references | `fixtures.tsv` references `qr_rank_deficient_6x4_nullspace_generator_v1`, which is present in `generators.tsv`. |
| Source-control boundary | No generated output path under `build/` is committed. |
| Optional-data boundary | Optional-data manifest exists, but no optional payload is committed under `tests/corpus/`. |
| Schema scope | Only fixture/generator/optional-data fields are documented; Day 6 still owns oracle row schema finalization. |

## Day 5 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The maintained storage layout exists in the repository. | Complete | `tests/corpus/` now contains manifest, expected-result, schema, and future fixture directories with committed files. |
| Skeleton rows match the Day 8 templates. | Complete | Manifest headers use Sprint 137 Day 8 fixture, generator, and optional-data field names; placeholder rows keep `TBD_*` values explicit. |
| No generated or optional external data is accidentally committed. | Complete | Only source-controlled metadata and README files were added; no `build/corpus/`, `build/corpus-reports/`, optional external payloads, or committed Matrix Market fixtures were added. |
