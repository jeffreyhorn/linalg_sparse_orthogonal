# Sprint 138 Day 14: Closeout & Validation Summary

## Purpose

Day 14 closes Sprint 138 by verifying deliverables, recording final validation
evidence, preserving residual ownership, and confirming the Sprint 139 QR
handoff.

Sprint 138 built the maintained numerical corpus architecture and first
durable oracle/report lane. It did not add broad corpus volume or widen public
claims beyond fixture-local evidence.

## Deliverable Checklist

| Sprint 138 deliverable | Status | Evidence |
| --- | --- | --- |
| Maintained corpus taxonomy | Complete | Day 2 and Day 3 artifacts define matrix-class axes, first-lane class selection, promotion gates, and claim boundaries. |
| Corpus storage and manifest layout | Complete | Day 4 and Day 5 artifacts define and implement `tests/corpus/` manifests, schemas, expected rows, and generated-output boundaries. |
| Oracle row schema | Complete | Day 6 and Day 7 artifacts define and implement observed oracle row semantics, comparison status, failure class, and first-lane row IDs. |
| First sustained oracle/report lane | Complete | Day 8 through Day 10 artifacts define the deterministic QR fixture, generator hashes, expected rows, and `scripts/run_corpus_oracle.py`. |
| Skip/defer semantics | Complete | Day 11 artifact and optional-data manifest row keep unavailable external data as skip/defer policy evidence only. |
| Focused validation | Complete | Day 12 artifact and Day 14 validation rerun record passing schema, oracle/report, docs, TSV, whitespace, and C-gate selection checks. |
| Corpus maintainer documentation | Complete | Day 13 updates to `tests/corpus/README.md` document ownership, row interpretation, stale-report assumptions, handoff requirements, and residuals. |
| Sprint 139 QR fixture handoff | Complete | Day 13 and this closeout artifact preserve fixture facts, oracle row IDs, tolerance, validation commands, and non-claims. |

## Final Validation Commands

| Command | Result | Reason |
| --- | --- | --- |
| `python3 -B scripts/validate_corpus_schema.py` | Pass | Validates maintained corpus TSV structure, required references, selected enums, first-lane generator hashes, expected-result rows, and optional-data guardrails. |
| `env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py` | Pass | Confirms the maintained oracle/report command runs with optional external data absent. |
| Generated report split check | Pass | Confirms 3 oracle pass rows, 1 optional-data skip row, and 0 optional-data pass rows. |
| `git diff --check` | Pass | Confirms no whitespace errors in tracked diff. |
| Trailing-whitespace scan for Sprint 138 docs, corpus docs, and touched scripts | Pass | Confirms no trailing whitespace in maintained documentation and script surfaces. |
| Focused Markdown local link/path validation under `docs/planning/EPIC_12` | Pass | Confirms planning Markdown local links resolve. |
| Corpus TSV width consistency check | Pass | Confirms source-controlled corpus TSV rows match their headers. |
| `.c`/`.h` touched-file check | Pass | No C or header files changed; `make format && make lint && make test` was not required. |

## Generated Output Summary

The final oracle run wrote generated output to ignored paths:

| Generated path | Source-control policy |
| --- | --- |
| `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv` | Generated local evidence; not committed. |
| `build/corpus-reports/index.tsv` | Generated report index; not committed. |
| `build/corpus-reports/skips.tsv` | Generated skip/defer policy rows; not committed. |
| `build/corpus-reports/manifest.txt` | Generated run manifest; not committed. |

The generated report index contained:

| Row class | Count | Interpretation |
| --- | ---: | --- |
| Oracle pass rows | 3 | Fixture-local reference rows for rank, nullity, and normalized null-vector residual. |
| Optional-data skip rows | 1 | Default disabled SuiteSparse QR subset policy evidence only. |
| Optional-data pass rows | 0 | Optional external data was not counted as solver pass evidence. |

## Sprint 139 QR Readiness Criteria

Sprint 139 can consume the first lane when it preserves these inputs:

- fixture key: `qr_rank_deficient_6x4_nullspace_v1`
- generator key: `qr_rank_deficient_6x4_nullspace_generator_v1`
- shape: 6 rows by 4 columns
- nonzeros: 14
- expected rank: 3
- expected nullity: 1
- null vector direction: `[-1, -1, 0, 1]`
- rank row ID: `qr_rank_deficient_6x4_nullspace_v1_rank`
- nullity row ID: `qr_rank_deficient_6x4_nullspace_v1_nullity`
- residual row ID: `qr_rank_deficient_6x4_nullspace_v1_projector_residual`
- initial normalized null-vector residual tolerance: `1e-10`
- validation commands:
  - `python3 scripts/validate_corpus_schema.py`
  - `python3 scripts/run_corpus_oracle.py`

QR closure for this first lane should use normalized null-vector residual
comparison rather than raw basis equality. Later solver-backed QR work may add
projector or two-way projection-distance rows. Any support-tier promotion
beyond local must come from reviewed generated evidence on the target lane.

## Residual Register

| Residual | Owner | Closeout status |
| --- | --- | --- |
| Solver-backed QR fixture closure | Sprint 139 | Ready with first-lane facts, oracle rows, and tolerance; not claimed complete in Sprint 138. |
| Reviewed hosted-platform corpus/oracle promotion | Sprint 139 or later | Not complete; current generated rows are local evidence only. |
| Partial-SVD clustered/repeated singular-value lanes | Sprint 140 | Not complete; taxonomy and oracle schema are ready for future rows. |
| Report freshness normalization and stale diagnostics | Sprint 141 | Not complete; Day 13 documents assumptions and expected fields. |
| Optional external-data availability and reviewed pass policy | Later Epic 12 work | Not complete; optional data remains disabled, skipped, or deferred by default. |
| Public adoption wording tied to corpus evidence | Later Epic 12 work | Not complete; public claims remain frozen to existing evidence. |

## Final Non-Claims

Sprint 138 does not claim broad corpus completeness, broad QR correctness, raw
QR basis parity, global least-squares or minimum-norm behavior, broad
partial-SVD correctness, SuiteSparse parity, external corpus parity, package
or ABI support, platform parity, portable performance, coverage completeness,
release readiness, or state-of-the-art status.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Maintained corpus taxonomy, layout, first lane, oracle/report command, and skip/defer semantics are present or explicitly deferred. | Complete | Day 2 through Day 13 artifacts and maintained `tests/corpus/` files cover each surface. |
| Validation matches touched surfaces and passes. | Complete | Final validation commands passed; no `.c` or `.h` files changed. |
| Sprint 139 has clear QR fixture, oracle, tolerance, and claim-boundary inputs. | Complete | QR readiness criteria and non-claims are recorded above and in Day 13 documentation. |
