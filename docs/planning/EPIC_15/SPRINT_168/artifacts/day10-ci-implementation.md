# Sprint 168 Day 10: CI Lane Implementation

## Purpose

Day 10 wires the selected Sprint 168 performance report and freshness check
into hosted CI. The implementation follows the Day 9 design and keeps the lane
narrow: it regenerates the canonical report bundle, checks only the selected
`bench_refactor_csc` row in hosted mode, summarizes methodology metadata, and
uploads the generated report artifacts.

The lane remains threshold-free and does not create a timing regression gate,
portable performance claim, external-library comparison, broad benchmark
publication, package/ABI claim, or state-of-the-art sparse linear algebra
claim.

## Workflow Change

Updated `.github/workflows/ci.yml` to add:

```yaml
hosted-performance-freshness:
  name: Linux reviewed hosted selected performance freshness
  runs-on: ubuntu-latest
  timeout-minutes: 10
```

The job is separate from:

- Linux supplemental `bench-fast`;
- Linux reviewed oracle/comparison freshness;
- CMake/package/dead-code/coverage jobs.

This separation keeps selected performance publication failures attributable
and avoids broadening existing CI evidence lanes.

## Hosted Metadata

The job sets:

| Variable | Value |
| --- | --- |
| `BENCH_CANONICAL_REPORT_LABEL` | `sprint-168-hosted-performance` |
| `SPARSE_CANONICAL_SUPPORT_TIER` | `hosted_selected` |
| `SPARSE_CANONICAL_CLAIM_BOUNDARY` | `hosted_selected_threshold_free` |
| `SPARSE_CANONICAL_RUNNER_CONTEXT` | `github-actions-ubuntu-latest` |
| `SPARSE_CANONICAL_BUILD_FLAGS` | `default_make_flags` |
| `SPARSE_CANONICAL_BUILD_MODE` | `serial` |
| `SPARSE_CANONICAL_CPU_MODEL` | first `/proc/cpuinfo` model name, or `unknown` |

The hosted-mode checker rejects local/default metadata that would weaken the
hosted evidence contract.

## CI Steps

The new job:

1. checks out the repository;
2. collects hosted CPU model metadata into `$GITHUB_ENV`;
3. runs `make bench-canonical-report`;
4. runs:

   ```sh
   python3 scripts/check_bench_canonical_freshness.py \
     --report-dir build/bench-reports/canonical \
     --mode hosted
   ```

5. prints `sprint168-performance-summary` lines with selected row and manifest
   metadata;
6. uploads the canonical report bundle using
   `actions/upload-artifact@v4`.

## Uploaded Artifact

Artifact name:

```text
sprint168-selected-performance-freshness
```

Uploaded paths:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/bench_chol_csc.csv`;
- `build/bench-reports/canonical/bench_iterative_reuse.csv`;
- `build/bench-reports/canonical/bench_eigs_reuse.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`.

The full canonical bundle is uploaded because the generator produces four CSV
files. The reviewed hosted evidence classification remains limited to the
selected `bench_refactor_csc` row checked by
`check_bench_canonical_freshness.py`.

## Local Validation

Ran hosted-style local equivalent:

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

Ran workflow YAML parse:

```sh
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/ci.yml"); puts "workflow yaml ok"'
```

Result: passed.

Ran the CI summary Python logic locally against the hosted-style generated
bundle. It printed:

```text
sprint168-performance-summary: artifact=bench_refactor_csc command=tests/data/suitesparse/nos4.mtx --repeat 1 fixture=nos4.mtx repeat=configured_repeat_1 support_tier=hosted_selected claim_boundary=hosted_selected_threshold_free
sprint168-performance-summary: report_label=sprint-168-hosted-performance runner_context=github-actions-ubuntu-latest build_flags=default_make_flags cpu_model=unknown build_mode=serial omp_num_threads=unset
sprint168-performance-summary: manifest_report_label=sprint-168-hosted-performance manifest_claim_boundary=hosted_selected_threshold_free non_claims=threshold_free_no_portable_performance_claim
```

## Non-Claim Preservation

The CI lane does not:

- run full `make bench`;
- convert `bench-fast` into hosted performance evidence;
- compare raw timing or speedup values;
- define timing thresholds;
- claim portable speed;
- claim external-library parity;
- promote unselected canonical rows;
- imply package, shared-library, ABI, or broad platform support;
- support a state-of-the-art sparse linear algebra performance claim.

## Quality Gate

Day 10 changed workflow YAML and planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate (`make format && make lint && make test`)
is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| CI lane is wired for the selected report only. | Complete | New `hosted-performance-freshness` job runs the canonical report and hosted checker for the selected `bench_refactor_csc` row. |
| Local equivalents pass before relying on hosted CI. | Complete | Hosted-style local report generation and `--mode hosted` freshness check passed. |
| Workflow wording stays methodology-bound. | Complete | Job comments and summary text retain threshold-free, selected-row, non-portable performance boundaries. |
