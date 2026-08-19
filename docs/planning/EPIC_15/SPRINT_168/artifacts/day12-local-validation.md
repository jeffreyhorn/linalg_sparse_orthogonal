# Sprint 168 Day 12: Local Validation Sweep

## Purpose

Day 12 runs the local validation sweep for the selected Sprint 168 performance
publication lane. The sweep validates the selected report generator, selected
freshness checker, hosted-mode metadata path, workflow syntax, and claim-safe
documentation wording.

## Changed File Classes

Current Sprint 168 changes include:

- workflow YAML;
- `Makefile`;
- documentation;
- planning artifacts;
- shell report script;
- Python freshness checker.

No `.c` or `.h` files were modified, so the full C quality gate
(`make format && make lint && make test`) is not required for Day 12.

## Script And Workflow Checks

Ran:

```sh
bash -n scripts/bench_canonical_report.sh
```

Result: passed.

Ran:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/check_bench_canonical_freshness.py
```

Result: passed.

Ran:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 \
  scripts/check_bench_canonical_freshness.py --help
```

Result: passed and showed `--report-dir` plus `--mode {local,hosted}`.

Ran:

```sh
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/ci.yml"); puts "workflow yaml ok"'
```

Result: passed.

## Selected Local Report And Freshness Check

Ran:

```sh
PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness
```

Result: passed. The target regenerated:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/bench_chol_csc.csv`;
- `build/bench-reports/canonical/bench_iterative_reuse.csv`;
- `build/bench-reports/canonical/bench_eigs_reuse.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`.

The local freshness checker passed for the selected
`artifact=bench_refactor_csc` row.

## Hosted-Mode Local Equivalent

Ran:

```sh
env BENCH_CANONICAL_REPORT_LABEL=sprint-168-hosted-performance \
  SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
  SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
  SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-latest \
  SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
  SPARSE_CANONICAL_CPU_MODEL=unknown \
  SPARSE_CANONICAL_BUILD_MODE=serial \
  PYTHONDONTWRITEBYTECODE=1 \
  make bench-canonical-report

PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

Result: passed.

## Summary Logic Check

Ran the CI summary Python logic locally against the hosted-style generated
bundle. It printed:

```text
sprint168-performance-summary: artifact=bench_refactor_csc command=tests/data/suitesparse/nos4.mtx --repeat 1 fixture=nos4.mtx repeat=configured_repeat_1 support_tier=hosted_selected claim_boundary=hosted_selected_threshold_free
sprint168-performance-summary: report_label=sprint-168-hosted-performance runner_context=github-actions-ubuntu-latest build_flags=default_make_flags cpu_model=unknown build_mode=serial omp_num_threads=unset
sprint168-performance-summary: manifest_report_label=sprint-168-hosted-performance manifest_claim_boundary=hosted_selected_threshold_free non_claims=threshold_free_no_portable_performance_claim
```

## Claim Scan

Ran:

```sh
rg -n "state[- ]of[- ]the[- ]art|portable performance|performance guarantee|superiority|external-library parity|broad benchmark|timing threshold|regression threshold|hosted selected-performance|bench-canonical-report-freshness|hosted_selected_threshold_free" \
  README.md benchmarks/README.md docs/maintainer_guide.md docs/planning/EPIC_15/SPRINT_168
```

Result: passed by inspection. The selected-performance lane references are
present, and risky terms appear only in explicit non-claim or retained-boundary
wording.

## Full C Gate Decision

Skipped:

```sh
make format && make lint && make test
```

Reason: no `.c` or `.h` files were modified in the Sprint 168 Day 12 change
set. The current changed files are scripts, workflow YAML, Makefile,
documentation, and planning artifacts.

## Generated Output Policy

The selected report validation generated files below ignored `build/` paths.
Those generated outputs are validation artifacts and should not be staged.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected local checks pass. | Complete | `make bench-canonical-report-freshness` passed. |
| Skipped checks have explicit reasons. | Complete | Full C gate skipped because no `.c` or `.h` files changed. |
| Code/header changes, if any, pass the full C quality gate. | Not applicable | No `.c` or `.h` changes are present. |
| Hosted-mode local equivalent passes. | Complete | Hosted metadata generation plus `--mode hosted` checker passed. |
| Claim-safe docs remain bounded. | Complete | Targeted claim scan found risky terms only in non-claim/boundary contexts. |
