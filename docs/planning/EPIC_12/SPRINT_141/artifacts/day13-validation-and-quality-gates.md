# Day 13 Validation And Quality Gates

## Purpose

Day 13 ran the Sprint 141 validation pass from the current normalized report
index implementation and documentation state. The pass focused on report index
generation, freshness gate behavior, corpus schema validation, script tests,
documentation hygiene, generated-output hygiene, and quality-gate scoping.

## Validation Summary

The Sprint 141 report-index and freshness validation passed.

No C or header files changed during Day 13. The required quality surface for
this day was therefore Python/script validation, corpus metadata validation,
report-index checks, freshness checks, and documentation hygiene rather than
`make format && make lint && make test`.

## Commands Run

| Command | Result | Evidence |
| --- | --- | --- |
| `python3 -m py_compile scripts/validate_corpus_schema.py scripts/normalize_report_index.py tests/test_normalize_report_index.py` | Pass | Python syntax/import compile check completed with exit `0`. |
| `python3 scripts/validate_corpus_schema.py` | Pass | `validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok` |
| `python3 tests/test_normalize_report_index.py` | Pass | `test-normalize-report-index: ok` |
| `python3 scripts/normalize_report_index.py --no-generated --output build/report-index/normalized-index.tsv` | Pass | Wrote ignored deterministic report index with `47` source-controlled rows. |
| `python3 scripts/normalize_report_index.py --no-generated --check` | Pass | `normalize-report-index: 47 rows ok` |
| `python3 scripts/normalize_report_index.py --check` | Pass | `normalize-report-index: 59 rows ok` |
| `python3 scripts/normalize_report_index.py --check-freshness` | Pass | `normalize-report-index: freshness ok (59 rows)` |
| `python3 scripts/normalize_report_index.py --family oracle --check-freshness` | Pass | Oracle generated rows reported stale warnings for old local generated artifacts and exited `0`. |
| `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness` | Pass | Runtime/backend governance emitted the expected Sprint 142 `defer` row and exited `0`. |
| `python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --check-freshness` | Pass | Coverage/dead-code missing generated rows stayed advisory; package proof-owner rows stayed source-controlled. |
| `python3 scripts/normalize_report_index.py --family coverage --require-generated coverage --check-freshness` | Expected failure | Missing generated coverage became `freshness: error` only under explicit `--require-generated coverage`; wrapper confirmed exit `1`. |
| `git diff --check` | Pass | No whitespace errors. |
| `rg -n "[ \t]+$" ...` | Pass | No trailing-whitespace matches in touched docs, scripts, tests, and corpus metadata. |
| `git check-ignore build/report-index/normalized-index.tsv` | Pass | Generated normalized index is ignored through `build/`. |

## Freshness Behavior Confirmed

| Case | Confirmed behavior |
| --- | --- |
| Current source-controlled rows | Report contract, package proof-owner, documentation, corpus fixture, generator, expected-result, and CI lane rows remain advisory/source-controlled rows governed by Git review and schema checks. |
| Missing advisory generated rows | Coverage, dead-code, benchmark, and sentinel advisory rows report missing generated artifacts without failing default freshness checks. |
| Missing required generated rows | `--require-generated coverage` promotes missing coverage output to `freshness: error` and returns nonzero. |
| Stale generated rows | Existing local oracle rows whose `source_commit` does not match current `HEAD` report `freshness: warning` by default. |
| Skipped rows | Optional corpus data rows report `freshness: skip` and do not fail default freshness checks. |
| Deferred rows | Runtime/backend governance reports `freshness: defer` and remains a Sprint 142 handoff. |

## Documentation Validation

The repository has a `make docs` target, but it runs Doxygen API generation:

```make
docs:
	@echo "Generating API documentation with Doxygen..."
	doxygen Doxyfile
```

Sprint 141 Day 13 did not run Doxygen because the touched documentation
surface was Markdown guidance and planning artifacts, not API comment
generation. Documentation validation for this day used path review,
`git diff --check`, and trailing-whitespace scans over the touched Markdown,
script, test, and corpus metadata surfaces.

## Generated Output Hygiene

Day 13 intentionally generated:

- `build/report-index/normalized-index.tsv`

That file remains ignored under `build/` and is not intended for source
control. No generated local measurement, coverage, dead-code, package, or
oracle report was promoted to committed release proof.

## Residual Risks And Owners

| Residual | Owner | Handling |
| --- | --- | --- |
| Existing local oracle rows are stale relative to current `HEAD`. | Corpus/oracle maintainer | Freshness gate surfaces warnings by default; use `--require-generated oracle` when a review requires current local oracle artifacts. |
| Runtime/backend governance rows are intentionally deferred. | Sprint 142 | Day 13 confirmed the row remains `defer`, not a Sprint 141 closure claim. |
| Coverage, dead-code, benchmark, and sentinel advisory generated reports may be absent locally. | Maintainers running those local flows | Default freshness checks keep them advisory; explicit `--require-generated <family>` promotes absence to an error for focused reviews. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| All required checks for touched surfaces pass. | Complete | Python compile, schema validation, generator tests, normalized-index checks, freshness checks, and doc hygiene passed. |
| Generated outputs are intentionally source-controlled or ignored. | Complete | `build/report-index/normalized-index.tsv` is ignored; no generated local proof artifacts were committed. |
| Residual risks are documented with owners or Sprint 142 handoff entries. | Complete | Oracle staleness, advisory generated reports, and runtime/backend governance are explicitly routed. |
