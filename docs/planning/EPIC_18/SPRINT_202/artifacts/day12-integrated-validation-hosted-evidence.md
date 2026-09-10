# Sprint 202 Day 12: Integrated Validation And Hosted Evidence Review

## Summary

Day 12 ran the integrated local validation set for the Sprint 202 workflow,
manifest, docs, report-index, and selected benchmark freshness changes. All
local checks passed. Hosted CI evidence is a bounded residual because the
branch has no upstream configured and `gh run list --branch sprint-202 --limit
10` returned no runs.

## Changed Surface

Tracked changed files at validation time:

- `.github/workflows/macos-ci.yml`;
- `README.md`;
- `INSTALL.md`;
- `benchmarks/README.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/README.md`;
- `tests/corpus/manifests/selected_report_targets.tsv`;
- `tests/corpus/schemas/report_index_fields.md`;
- `tests/test_bench_canonical_freshness.py`;
- `tests/test_selected_comparison_workflow.py`;
- `tests/test_selected_performance_docs.py`.

No `.c` or `.h` files were modified.

## Integrated Local Validation

| Command | Result | Notes |
| --- | --- | --- |
| `make bench-canonical-report-freshness` | Passed | Regenerated the canonical bundle and passed local selected freshness for `bench_refactor_csc`. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed | Covered selected row, hosted metadata, manifest agreement, path drift, malformed metadata, and threshold-free policy. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Covered Linux/macOS/Windows selected workflow guard behavior, including the Sprint 202 macOS selected-performance lane. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Covered selected-performance documentation markers and overclaim rejection. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Covered selected target manifest structure and Linux/macOS selected benchmark metadata. |
| `python3 tests/test_normalize_report_index.py` | Passed | Covered normalized report-index construction and freshness behavior. |
| `python3 scripts/normalize_report_index.py --check` | Passed | Reported `normalize-report-index: 151 rows ok`. |
| `python3 scripts/normalize_report_index.py --family benchmark --check-freshness` | Passed | Reported advisory local benchmark rows and `normalize-report-index: freshness ok (5 rows)`. |
| `python3 -m py_compile ...` | Passed | Syntax-checked changed Python guards and selected freshness/normalizer scripts. |
| Hosted-mode local simulation | Passed | Generated canonical reports with Sprint 202 macOS hosted metadata and passed `check_bench_canonical_freshness.py --mode hosted`. |

## Hosted-Mode Local Simulation

Command:

```sh
BENCH_CANONICAL_REPORT_LABEL=sprint202-macos-hosted-performance \
SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-macos-latest \
SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
SPARSE_CANONICAL_BUILD_MODE=serial \
SPARSE_CANONICAL_CPU_MODEL=local-macos-simulation \
make bench-canonical-report &&
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

Result:

```text
bench-canonical-freshness: passed (mode=hosted; artifact=bench_refactor_csc; report_dir=build/bench-reports/canonical)
```

## Claim Scan

Command:

```sh
rg -n "selected performance (proves|guarantees) portable performance|selected performance (proves|is) state-of-the-art|hosted selected performance (is|acts as) a timing gate|bench-canonical-report-freshness (proves|guarantees) speedup|Linux/macOS performance parity" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md
```

Result:

- Found only the intended non-claim in `benchmarks/README.md`:
  `Neither hosted row creates Linux/macOS performance parity...`.
- Found no unsupported selected-performance portable-performance,
  state-of-the-art, timing-gate, or speedup claims.

## Hosted Evidence Review

Commands:

```sh
gh run list --branch sprint-202 --limit 10
git rev-parse --abbrev-ref --symbolic-full-name @{u}
```

Results:

- `gh run list --branch sprint-202 --limit 10`: returned no runs.
- `git rev-parse --abbrev-ref --symbolic-full-name @{u}`: failed with
  `fatal: no upstream configured for branch 'sprint-202'`.

Hosted CI evidence cannot be reviewed until the branch is pushed and GitHub
Actions runs.

## Hosted Residual And Rerun Checklist

After the branch is pushed or a PR is opened, review the macOS workflow run for:

- job `selected-performance-freshness` started on `macos-latest`;
- `sysctl -n machdep.cpu.brand_string` captured or explicitly fell back to
  `unknown`;
- `make bench-canonical-report` completed;
- `python3 scripts/check_bench_canonical_freshness.py --report-dir
  build/bench-reports/canonical --mode hosted` passed;
- artifact `sprint202-macos-selected-performance-freshness` uploaded exactly:
  - `build/bench-reports/canonical/bench_refactor_csc.csv`;
  - `build/bench-reports/canonical/index.tsv`;
  - `build/bench-reports/canonical/manifest.txt`;
- workflow summary reported one selected `bench_refactor_csc` row and no
  timing-threshold, portable-performance, broad-platform, package/ABI, release,
  or state-of-the-art claim.

## Quality-Gate Decision

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 12.
