# Sprint 168 Day 9: CI Lane Design

## Purpose

Day 9 designs the hosted CI lane for the selected Sprint 168 performance
publication path. The lane promotes one threshold-free canonical benchmark row
to reviewed hosted freshness evidence:

- benchmark: `bench_refactor_csc`;
- command: `tests/data/suitesparse/nos4.mtx --repeat 1`;
- artifact row: `build/bench-reports/canonical/index.tsv` with
  `artifact=bench_refactor_csc`;
- selected CSV: `build/bench-reports/canonical/bench_refactor_csc.csv`;
- freshness checker: `scripts/check_bench_canonical_freshness.py --mode hosted`.

The lane does not create a timing regression threshold, portable performance
claim, external-library comparison, package/ABI claim, or state-of-the-art
sparse linear algebra claim.

## Current CI Structure Reviewed

| Job | Current role | Sprint 168 decision |
| --- | --- | --- |
| `build-and-test` | Linux supplemental runtime, sanitizers, benchmark binary compile, and `bench-fast`. | Do not widen this job. Keep `bench-fast` supplemental and separate from selected hosted report freshness. |
| `cmake-build-and-test` | Linux enforced reviewed CMake parity. | No performance publication ownership. |
| `package-contract` | Linux reviewed static-first install/export proof. | No benchmark ownership. |
| `generated-report-freshness` | Linux reviewed hosted oracle/comparison freshness with artifact upload. | Reuse naming, timeout, summary, and upload conventions, but do not mix selected performance artifacts into oracle/comparison evidence. |
| `tsan` | Linux supplemental thread sanitizer coverage. | No performance publication ownership. |
| `lint` | Linux enforced Makefile compile-quality path. | No performance publication ownership. |
| `deadcode` | Linux enforced dead-code report/check path. | No performance publication ownership. |
| `coverage` | Linux supplemental coverage report. | No performance publication ownership. |

## Selected Job Placement

Add a separate job to `.github/workflows/ci.yml` after
`generated-report-freshness`:

```yaml
  hosted-performance-freshness:
    name: Linux reviewed hosted selected performance freshness
    runs-on: ubuntu-latest
    timeout-minutes: 10
```

Rationale:

- keeps performance evidence separate from oracle/comparison correctness
  evidence;
- makes CI failures attributable to the selected benchmark report lane;
- avoids widening `bench-fast` from supplemental smoke coverage into report
  publication;
- keeps the future artifact upload name independent and reviewable.

## Hosted Environment Contract

The hosted lane should set these variables for the report-generation step:

| Variable | Hosted value | Reason |
| --- | --- | --- |
| `BENCH_CANONICAL_REPORT_LABEL` | `sprint-168-hosted-performance` | Avoids `unlabeled` and names the lane scope. |
| `SPARSE_CANONICAL_SUPPORT_TIER` | `hosted_selected` | Marks only the selected hosted lane as reviewed evidence. |
| `SPARSE_CANONICAL_CLAIM_BOUNDARY` | `hosted_selected_threshold_free` | Preserves threshold-free interpretation. |
| `SPARSE_CANONICAL_RUNNER_CONTEXT` | `github-actions-ubuntu-latest` | Records hosted runner context for methodology. |
| `SPARSE_CANONICAL_BUILD_FLAGS` | `default_make_flags` | Avoids `not_recorded` in hosted mode while matching the default Make build. |
| `SPARSE_CANONICAL_BUILD_MODE` | `serial` | Makes the first hosted lane explicit and avoids OpenMP-speedup claims. |
| `SPARSE_CANONICAL_CPU_MODEL` | best effort from `/proc/cpuinfo`, fallback `unknown` | Records available hosted CPU context without requiring stable CPU assignment. |
| `OMP_NUM_THREADS` | unset | Keeps serial/default behavior explicit through `omp_num_threads=unset`. |

Suggested CPU-model collection:

```sh
cpu_model="$(awk -F': ' '/model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || true)"
if [ -z "$cpu_model" ]; then
  cpu_model="unknown"
fi
echo "SPARSE_CANONICAL_CPU_MODEL=$cpu_model" >> "$GITHUB_ENV"
```

## Hosted Steps

Recommended Day 10 workflow steps:

1. Check out the repository.
2. Set hosted performance metadata in `$GITHUB_ENV`, including CPU model.
3. Run the selected report generation command with hosted metadata:

   ```sh
   make bench-canonical-report
   ```

4. Run the strict hosted freshness check:

   ```sh
   python3 scripts/check_bench_canonical_freshness.py \
     --report-dir build/bench-reports/canonical \
     --mode hosted
   ```

5. Print a compact selected-performance summary that reads
   `index.tsv` and `manifest.txt`, reporting:
   - selected artifact;
   - command;
   - fixture;
   - repeat semantics;
   - support tier;
   - claim boundary;
   - report label;
   - runner context;
   - build flags;
   - CPU model;
   - build mode;
   - thread setting.
6. Upload the canonical performance report artifacts with `if: always()`.

## Artifact Upload

Use a distinct artifact name:

```yaml
name: sprint168-selected-performance-freshness
```

Upload only the canonical report bundle:

```yaml
path: |
  build/bench-reports/canonical/bench_refactor_csc.csv
  build/bench-reports/canonical/bench_chol_csc.csv
  build/bench-reports/canonical/bench_iterative_reuse.csv
  build/bench-reports/canonical/bench_eigs_reuse.csv
  build/bench-reports/canonical/index.tsv
  build/bench-reports/canonical/manifest.txt
```

Include the full canonical bundle because the generator currently emits four
CSV files, but the hosted evidence classification applies only to the selected
`bench_refactor_csc` row checked by `check_bench_canonical_freshness.py`.

Recommended upload policy:

- `if: always()`;
- `retention-days: 7`;
- `if-no-files-found: error`.

## Runtime Budget

Day 4 measured local canonical report runtime at approximately 3.21 seconds.
The hosted job should use a conservative `timeout-minutes: 10` budget.

This budget is intentionally separate from:

- `bench-fast`, which remains supplemental smoke coverage;
- full `make bench`, which remains a developer opt-in;
- oracle/comparison freshness, which retains the existing 15-minute job.

## Evidence Classification

Classify the lane as:

```text
Linux reviewed hosted selected performance freshness
```

Supported statement after hosted implementation:

```text
CI regenerates the selected `bench_refactor_csc` canonical report row for
`nos4.mtx --repeat 1` on the Linux GitHub Actions lane and checks that the
methodology metadata, artifact paths, selected row identity, and threshold-free
claim boundary are fresh and complete.
```

Retained non-claims:

- no portable performance guarantee;
- no timing regression threshold;
- no performance superiority claim;
- no external-library performance parity;
- no broad benchmark-family publication;
- no broad platform parity;
- no package, shared-library, or ABI evidence;
- no state-of-the-art sparse linear algebra claim.

## Failure Messages

The workflow should rely on the Day 8 checker for strict failure messages:

- `freshness: error: benchmark_selected_report_dir_missing`;
- `freshness: error: benchmark_selected_artifact_missing`;
- `freshness: error: benchmark_selected_schema`;
- `freshness: error: benchmark_selected_row_missing`;
- `freshness: error: benchmark_selected_row_duplicate`;
- `freshness: error: benchmark_selected_value`;
- `freshness: error: benchmark_selected_metadata_missing`;
- `freshness: error: benchmark_selected_claim_boundary`;
- `freshness: error: benchmark_selected_manifest_mismatch`.

The workflow step names should include "selected performance" so failures are
not confused with oracle/comparison freshness or `bench-fast`.

## Day 10 Implementation Checklist

Day 10 should:

1. Add the `hosted-performance-freshness` job to `.github/workflows/ci.yml`.
2. Set the hosted metadata env values before report generation.
3. Run `make bench-canonical-report`.
4. Run `check_bench_canonical_freshness.py --mode hosted`.
5. Add selected-performance summary output.
6. Upload the canonical report bundle as
   `sprint168-selected-performance-freshness`.
7. Validate locally with the Day 8 hosted-style command.
8. Run `git diff --check`.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Hosted lane design is bounded and reviewable. | Complete | Separate Linux job, selected command, selected row, hosted metadata, strict checker, and 10-minute timeout are specified. |
| Artifact ownership is explicit. | Complete | Upload list is limited to the canonical report bundle; only `bench_refactor_csc` is classified as selected hosted evidence. |
| Hosted evidence classification is not broader than the lane. | Complete | Evidence wording and retained non-claims reject portable speed, external parity, broad benchmark, platform, package, ABI, and state-of-the-art claims. |
