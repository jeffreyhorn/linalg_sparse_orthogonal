# Day 1 Report Family Intake

## Purpose

Day 1 establishes Sprint 141 scope and records the inherited report evidence
before metadata-contract, generator, freshness-gate, or documentation
implementation begins. The sprint must normalize report metadata only where row
meaning can be preserved honestly.

This is an intake and planning artifact. It does not change source code,
report-generating commands, corpus schemas, public documentation, CI workflows,
or package/install behavior.

## Project-Plan Scope

Sprint 141 implements "Report Index Normalization & Freshness Gates" from
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

The sprint goal is to normalize maintained report metadata across:

- corpus and oracle rows;
- canonical benchmark reports;
- performance sentinel bundles;
- large-matrix guardrail reports;
- coverage summaries;
- dead-code report artifacts;
- package, install, CMake, and pkg-config proof lanes;
- CI summary lanes where row meaning can be preserved.

The sprint also adds stale-report diagnostics while preserving local
measurement, optional-data, support-tier, and non-claim boundaries.

## Handoff Inputs

| Input | Day 1 use |
| --- | --- |
| Sprint 138 corpus architecture | Provides schema-checked fixture, generator, expected-result, optional-data, generated-reference, support-tier, and skip/defer conventions. |
| Sprint 139 QR closure | Provides stale-report guidance and the generated-reference versus solver-backed report split. |
| Sprint 140 partial-SVD closure | Provides the immediate report-index handoff for generated-reference, solver-backed, skip, stale, unsupported, freshness, and support-tier semantics. |
| `tests/corpus/` | Current source-controlled report-row source for corpus families. |
| `scripts/run_corpus_oracle.py` | Current generated corpus/oracle report command and likely normalized-index input. |
| `Makefile` report targets | Current command authority for benchmark, sentinel, guardrail, dead-code, coverage, install, and package checks. |
| `.github/workflows/*.yml` | Current hosted platform and supplemental confidence lane definitions. |
| User-facing and maintainer docs | Current interpretation surface for report non-claims and regeneration commands. |

## Initial Report Family Inventory

| Family | Producer/source | Current evidence shape | Initial conclusion |
| --- | --- | --- | --- |
| Corpus manifest rows | `tests/corpus/manifests/*.tsv` | source-controlled TSV rows | Normalizable with low risk. |
| Corpus expected rows | `tests/corpus/expected/*.tsv` | source-controlled TSV rows | Normalizable with fixture-local row meaning. |
| Corpus oracle rows | `scripts/run_corpus_oracle.py` | generated rows under ignored `build/` paths | Normalizable if freshness metadata is explicit. |
| Solver-backed corpus rows | `run_corpus_oracle.py --include-solver-qr --include-partial-svd` | generated local proof rows | Normalizable only with support-tier and local/generated status fields. |
| Canonical benchmark reports | `make bench-canonical-report` | generated CSVs plus `index.tsv` and `manifest.txt` | Normalizable as local measurement snapshots, not performance claims. |
| Performance sentinels | `make performance-sentinels` | generated sentinel bundle | Needs separate hard-gate versus advisory-row semantics. |
| Large-matrix guardrails | `make large-matrix-guardrails` | generated guardrail index/manifest/report artifacts | Normalizable with optional-data and structural-report fields. |
| Dead-code reports | `make deadcode-report`, `make deadcode-check` | generated `report.md` and `report.tsv` | Normalizable as completeness/classification artifacts, not removal-ready proof. |
| Coverage reports | `make coverage` and backend-specific targets | generated lcov/gcovr HTML and summary files | High risk because backend/platform/tooling affect row meaning. |
| Package/install metadata | Make install, CMake install/export, `sparse.pc.in` | installed files, generated metadata, CI proof logs | Normalizable only with static-first and platform-scoped boundaries. |
| CI summary lanes | `.github/workflows/*.yml` | hosted job definitions and logs | Indexable as lane metadata; job logs are not source-controlled rows. |

## Day-Level Ownership Map

| Item | Day owner(s) |
| --- | --- |
| Item 1: Report Family Inventory | Days 1-2 |
| Item 2: Shared Metadata Contract | Days 3 and 5 |
| Item 3: Normalized Index Generator | Days 4, 6, 7, 8, 9 |
| Item 4: Stale-Report Gate | Days 10-11 |
| Item 5: Documentation Alignment | Day 12 |
| Item 6: Validation | Day 13, plus focused checks on implementation days |
| Item 7: Closeout | Day 14 |

## Initial Boundaries And Non-Claims

Sprint 141 does not start with any claim for:

- broad report completeness;
- benchmark or sentinel output as portable performance evidence;
- local generated reports as hosted release proof;
- broad corpus, QR, partial-SVD, package, install, ABI, shared-library, or
  platform parity;
- coverage or dead-code output as a universal quality claim;
- runtime/backend governance closure;
- external-library parity;
- state-of-the-art sparse linear algebra status.

## Initial Stop Conditions

- Stop if a report family cannot preserve row meaning but would need to be
  represented as a pass/fail proof.
- Stop if freshness checks require committing machine-local generated reports.
- Stop if runtime/backend policy decisions are required before Sprint 142.
- Stop if validation fails.
- Stop before public docs imply broader claims than source-controlled or
  generated fixture-local evidence supports.

## Day 1 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 141 project-plan item has a day-level owner. | Complete | Day-level ownership map above and in `WORKING_NOTES.md`. |
| Candidate report families are visible before schema design begins. | Complete | Initial report family inventory above and in `WORKING_NOTES.md`. |
| Report families that may not preserve row meaning are explicitly flagged. | Complete | Coverage, performance sentinels, package/install, and CI summary lanes are marked as higher-risk normalization surfaces. |
