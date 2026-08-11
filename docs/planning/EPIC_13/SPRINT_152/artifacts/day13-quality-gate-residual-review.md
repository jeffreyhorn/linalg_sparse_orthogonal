# Sprint 152 Day 13 Quality Gate And Residual Review

## Purpose

Day 13 verifies the Sprint 152 changed surfaces and records the remaining
generated-report families that are intentionally advisory, deferred, or owned by
later sprint candidates.

## Changed-Surface Review

Commands:

```sh
git diff --name-only -- '*.c' '*.h'
git ls-files --others --exclude-standard -- '*.c' '*.h'
```

Result: no modified or untracked `.c` or `.h` files.

Because Sprint 152 has changed Makefile, Python report-index code, Python tests,
documentation, report-family metadata, and planning artifacts only, the Day 13
quality gate used the focused Python/report/documentation path. The full
`make format && make lint && make test` C gate was not required by the sprint
instructions.

## Focused Quality Gate

Selected oracle freshness:

```sh
make report-index-oracle-freshness
```

Result: passed. The target regenerated selected local oracle output and
reported `normalize-report-index: freshness ok (54 rows)`.

Corpus schema:

```sh
python3 scripts/validate_corpus_schema.py
```

Result: passed.

Python/report-index tests:

```sh
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
```

Result: passed with `test-normalize-report-index: ok`.

Normalized corpus/oracle index:

```sh
python3 scripts/normalize_report_index.py --family corpus --family oracle --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
```

Result: passed with `128` normalized rows.

Strict selected oracle freshness:

```sh
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --strict-generated --check-freshness
```

Result: exit `0`, no freshness errors, and `52` expected row-level
`generated_present_unchecked` warnings while the selected aggregate policy
passed.

Advisory and source-controlled families:

```sh
python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --family runtime_backend --check-freshness
```

Result: exit `0` with `11` advisory/source-controlled rows.

Whitespace and generated cache cleanup:

```sh
find scripts tests -type f -path '*/__pycache__/*' -delete
find scripts tests -type d -name __pycache__ -empty -delete
git diff --check
```

Result: passed.

## Stale-Reference Review

Command:

```sh
rg -n "run_corpus_oracle.py --include-solver-qr|run_corpus_oracle.py --include-partial-svd|require-generated oracle|report-index-oracle-freshness|QR-only|partial-SVD-only|105 rows" \
  README.md docs/maintainer_guide.md docs/solver_selection.md docs/algorithm.md \
  tests/corpus/schemas/report_index_fields.md tests/corpus/manifests/report_families.tsv
```

Result: active selected freshness references point at
`make report-index-oracle-freshness`. Remaining QR-only and partial-SVD-only
command references are intentionally documented as focused debugging variants
and do not satisfy the selected combined row-count policy by themselves. No
stale `105 rows` reference was found.

## Residual Generated-Family Register

| Family | Current Policy | Owner Candidate | Day 13 Disposition |
| --- | --- | --- | --- |
| `oracle/generated_reference` | selected generated compare input | Corpus maintainer | Closed for Sprint 152 local freshness gate. |
| `oracle/solver_backed` | selected generated compare input | Solver owner | Closed for Sprint 152 local freshness gate. |
| `benchmark/canonical` | generated local advisory | Benchmark maintainer | Residual; needs benchmark publication policy before claim-bearing use. |
| `sentinel/runtime` | generated compare input | Benchmark maintainer | Residual; hard-gate semantics remain performance/runtime owned. |
| `sentinel/advisory` | generated local advisory | Benchmark maintainer | Residual; advisory measurements remain non-claiming. |
| `guardrail/large_matrix` | generated compare input | Benchmark maintainer | Residual; large-matrix proof policy remains benchmark/guardrail owned. |
| `deadcode/report` | generated local advisory | Maintainer | Residual; advisory report output is not zero-dead-code proof. |
| `coverage/src` | generated local advisory | Maintainer | Residual; local coverage output is not coverage-completeness proof. |
| `report_index/missing_generated` | generated local advisory | Report maintainer | Supporting policy row remains explicit missing-output visibility. |
| `ci/reviewed_lanes` | hosted CI external | CI maintainer | Source-controlled lane metadata only; hosted logs remain external evidence. |
| `package/static_install` | source-controlled | Package maintainer | Source-controlled install proof-owner row; not generated freshness proof. |
| `runtime_backend/governance` | source-controlled | Report maintainer | Source-controlled governance row; generated sentinel data remains separate. |

## Day 14 Handoff

Day 14 should finalize Sprint 152 notes and artifacts, prepare the Sprint 153
ABI/package handoff, rerun the final lightweight report/schema/freshness checks,
and confirm no ignored generated output is staged or described as release proof.

## Non-Claims

Day 13 validation does not claim hosted CI oracle proof, package-manager
availability, shared-library ABI support, dynamic-loader support, broad
platform support, broad QR correctness, broad partial-SVD correctness,
external-library parity, portable performance, benchmark superiority, complete
coverage, zero dead code, or state-of-the-art sparse linear algebra status.
