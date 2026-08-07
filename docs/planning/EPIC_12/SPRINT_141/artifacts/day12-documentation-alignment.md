# Day 12 Documentation Alignment

## Purpose

Day 12 aligns maintainer, benchmark, corpus, package, cookbook, and README
documentation with the normalized report index and freshness gate added in
Sprint 141. The documentation explains how to regenerate and check normalized
rows while keeping source-controlled evidence, local measurements, generated
reports, and non-claims distinct.

## Updated Surfaces

| Surface | Update |
| --- | --- |
| `docs/maintainer_guide.md` | Added normalized report-index workflow, freshness diagnostic interpretation, dead-code indexing semantics, package proof-owner row semantics, and focused command examples. |
| `benchmarks/README.md` | Replaced deferred cross-report wording with current benchmark/sentinel/guardrail normalized-index commands and measurement non-claims. |
| `tests/corpus/README.md` | Added corpus/oracle normalized-index commands, freshness diagnostics, and required-generated interpretation. |
| `INSTALL.md` | Added normalized package proof-owner row guidance and static-first source-controlled scope. |
| `README.md` | Added compact maintainer commands for `--check` and `--check-freshness` plus a non-claim reminder. |
| `docs/cookbook.md` | Added compact cross-family normalized-index commands and interpretation boundaries. |
| `docs/planning/EPIC_12/SPRINT_141/WORKING_NOTES.md` | Recorded Day 12 work and validation. |

## Documented Commands

General maintainer workflow:

```sh
python3 scripts/normalize_report_index.py \
  --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py --check
python3 scripts/normalize_report_index.py --check-freshness
```

Corpus/oracle:

```sh
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

Benchmark/sentinel/guardrail:

```sh
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel --family guardrail \
  --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel --family guardrail \
  --check-freshness
```

Package:

```sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
```

## Aligned Semantics

| Topic | Documentation alignment |
| --- | --- |
| Source-controlled metadata | Package proof owners, corpus manifests, expected rows, CI lane definitions, and docs advisories are source-controlled context, not generated pass evidence. |
| Local measurements | Benchmark, sentinel advisory, coverage, and dead-code rows remain local/advisory unless a reviewed gate explicitly requires them. |
| Generated oracle rows | Freshness diagnostics are fixture-local and tied to command, commit, platform, compiler, configuration, support tier, artifact path, and native row ID. |
| Required generated rows | `--require-generated <family>` is documented as an explicit review-time requirement, not the default. |
| Package/install proof | Normalized package rows identify maintained static-first proof owners/templates; they do not prove an install command was just run. |
| Runtime/backend governance | Deferred rows remain a Sprint 142 handoff. |

## Non-Claims Preserved

The updated docs avoid implying:

- broad performance or portable timing guarantees from benchmark rows;
- coverage completeness or product quality from coverage rows;
- zero-dead-code status from dead-code rows;
- package-manager availability, shared-library ABI support, or runtime-loader
  guarantees from package proof-owner rows;
- broad solver correctness, external-library parity, or hosted platform proof
  from generated corpus/oracle rows;
- Sprint 141 closure for runtime/backend governance.

## Validation Evidence

Commands run for Day 12:

```sh
python3 -m py_compile scripts/validate_corpus_schema.py scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --no-generated --check
python3 scripts/normalize_report_index.py --check
git diff --check
```

The final validation pass is recorded in the Day 12 working notes and final
turn summary.

## Day 13 Handoff

Day 13 should run the validation pass across normalized index generation,
freshness checks, schema checks, focused script tests, doc hygiene, and any
required quality gates for touched surfaces. Since Day 12 changed docs and
Python scripts only, C quality gates remain unnecessary unless Day 13 adds or
changes C/header files.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Maintainers can regenerate and check report indexes from documented commands. | Complete | Maintainer guide, benchmark README, corpus README, install guide, README, and cookbook now list normalized-index commands. |
| Docs distinguish source-controlled evidence from local measurements. | Complete | Package proof-owner, benchmark/sentinel, coverage/dead-code, and corpus/oracle sections describe row scope and freshness boundaries. |
| User-facing docs avoid unsupported state-of-the-art, performance, or package claims. | Complete | README/cookbook additions frame the index as navigation/freshness diagnostics, not release proof. |
