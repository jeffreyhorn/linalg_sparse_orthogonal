# Sprint 138 Day 12 - Focused Validation & Quality Gates

## Purpose

Day 12 runs the focused corpus/oracle validation suite and records quality-gate
evidence for the surfaces touched in Sprint 138 so far.

This is a validation artifact. It does not add solver behavior, generated
oracle/report outputs to source control, optional external data, public claims,
or `.c`/`.h` changes.

## Commands Run

| Command | Result | Meaning |
| --- | --- | --- |
| `python3 -m py_compile scripts/validate_corpus_schema.py scripts/run_corpus_oracle.py` | Pass | Touched Python scripts compile. |
| `python3 -B scripts/validate_corpus_schema.py` | Pass | Corpus fixture, generator, optional-data, expected-result, and first-lane deterministic hash metadata validate. |
| `env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py` | Pass | Default corpus/oracle command runs without optional external data. |
| Generated report split check | Pass | Report index has 3 oracle `pass` rows, 1 optional-data `skip` row, and 0 optional-data pass rows. |
| `git diff --check` | Pass | No whitespace errors in tracked diff. |
| Trailing-whitespace scan under Sprint 138 docs, `tests/corpus`, and touched scripts | Pass | No trailing whitespace found. |
| Focused Markdown link/path validation under `docs/planning/EPIC_12` | Pass | No missing local Markdown links found. |
| Corpus TSV column consistency check | Pass | Corpus TSV rows match their headers. |
| `.c`/`.h` touched-file check | Pass | No C or header files changed; full C quality gate was not required. |

## Corpus/Oracle Command Output

The maintained command wrote generated outputs under ignored build paths:

| Generated path | Status | Source-control policy |
| --- | --- | --- |
| `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv` | Generated successfully | Ignored; not committed. |
| `build/corpus-reports/index.tsv` | Generated successfully | Ignored; not committed. |
| `build/corpus-reports/skips.tsv` | Generated successfully | Ignored; not committed. |
| `build/corpus-reports/manifest.txt` | Generated successfully | Ignored; not committed. |

The generated `index.tsv` contained:

| Row type | Count | Interpretation |
| --- | --- | --- |
| Oracle `pass` rows | 3 | Fixture-local rank, nullity, and projector/subspace reference comparisons passed. |
| Optional-data `skip` rows | 1 | Optional SuiteSparse QR subset remains disabled by default and is policy evidence only. |
| Optional-data `pass` rows | 0 | Optional data was not counted as numerical pass evidence. |

## Quality Gate Selection

| Touched surface | Required gate | Day 12 result |
| --- | --- | --- |
| Python scripts | Syntax check plus focused command execution. | Passed for `scripts/validate_corpus_schema.py` and `scripts/run_corpus_oracle.py`. |
| Corpus TSV manifests and expected rows | Schema validation and TSV width checks. | Passed. |
| Sprint planning docs and corpus docs | Whitespace and focused Markdown link/path checks. | Passed. |
| Generated corpus/oracle reports | Verify generated under ignored `build/` paths and not committed. | Passed. |
| `.c` and `.h` files | `make format && make lint && make test` only if modified. | Not required; no `.c` or `.h` files changed. |

## Residuals

| Residual | Owner day or sprint | Reason |
| --- | --- | --- |
| Hosted or reviewed platform promotion | Later sprint or CI owner | Current corpus/oracle rows are `local_only`. |
| Solver-backed QR implementation closure | Sprint 139 | Day 12 validates the corpus/reference lane, not broad QR solver behavior. |
| Partial-SVD clustered/repeated-spectrum corpus lane | Sprint 140 | Reserved by taxonomy and oracle schema, not implemented in Sprint 138. |
| Report freshness normalization | Sprint 141 | Day 12 emits compatible rows; normalized stale-report checks are later work. |
| Optional external data availability | Later corpus owner | Default optional data remains disabled/skip evidence only. |

## Non-Claims

Day 12 validation does not claim:

- raw QR basis parity;
- broad QR correctness;
- global minimum-norm behavior;
- SuiteSparse or external-library parity;
- broad corpus completeness;
- SVD correctness;
- release readiness;
- package, platform, performance, coverage, or state-of-the-art status.

## Day 12 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| All required checks for touched surfaces pass. | Complete | Command table records passing script, corpus, oracle, docs, TSV, generated-output, and C-gate selection checks. |
| Skipped optional data is reported separately from pass evidence. | Complete | Report split check found 3 oracle pass rows, 1 optional-data skip row, and 0 optional-data pass rows. |
| Validation results are sufficient for Sprint 138 closeout. | Complete | Generated outputs, quality-gate selection, residuals, and non-claims are recorded for Day 14 closeout. |
