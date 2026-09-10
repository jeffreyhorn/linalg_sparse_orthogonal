# Sprint 202 Day 11: Focused Freshness Validation

## Summary

Day 11 ran the focused validation set after the Sprint 202 macOS hosted
selected benchmark workflow, guard, manifest, and documentation updates. All
focused checks passed locally. No `.c` or `.h` files changed.

## Freshness Results

| Check | Result | Notes |
| --- | --- | --- |
| `make bench-canonical-report-freshness` | Passed | Regenerated the canonical report bundle and passed the local selected `bench_refactor_csc` freshness checker. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Covered selected metadata, manifest agreement, hosted-mode fixtures, malformed metadata, missing/duplicate rows, and path drift. |
| Hosted-mode local simulation | Passed | Generated the canonical bundle with macOS hosted metadata environment values and passed `check_bench_canonical_freshness.py --mode hosted`. |
| `python3 scripts/normalize_report_index.py --family benchmark --check-freshness` | Passed | Emitted advisory stale diagnostics for local benchmark rows and completed with `normalize-report-index: freshness ok (5 rows)`. |

## Workflow, Manifest, And Docs Results

| Check | Result | Notes |
| --- | --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Validated Linux, macOS, and Windows selected workflow guard coverage, including the Day 9 macOS selected-performance drift fixtures. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Validated selected target manifest structure and selected benchmark Linux/macOS metadata. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Validated README, INSTALL, benchmark, maintainer, corpus, and report-index schema claim markers. |
| `python3 -m py_compile ...` | Passed | Syntax-checked the changed Python guards and selected freshness scripts. |

## Claim Scan

Command:

```sh
rg -n "selected performance (proves|guarantees) portable performance|selected performance (proves|is) state-of-the-art|hosted selected performance (is|acts as) a timing gate|bench-canonical-report-freshness (proves|guarantees) speedup|Linux/macOS performance parity" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md
```

Result:

- Found only the intentional non-claim in `benchmarks/README.md`:
  `Neither hosted row creates Linux/macOS performance parity...`.
- Found no unsupported portable-performance, timing-gate, speedup, or
  state-of-the-art selected-performance claims.

## Hosted Residual

Local focused validation cannot prove that GitHub Actions has executed the
new macOS lane on the hosted runner. Day 12 must review hosted CI evidence for:

- job start on `macos-latest`;
- CPU metadata capture through `sysctl -n machdep.cpu.brand_string`;
- selected-only artifact upload;
- hosted-mode checker pass;
- workflow summary with one selected row and no broad performance wording.

## Quality-Gate Decision

No `.c` or `.h` files were modified. The full C quality gate is not required
for Day 11.
