# Day 3: Methodology Lock and Collector Preparation

## Decision

Lock policy version `s225-v1` **before accepted baseline collection** for the single `SRT-BENCH-REFACTOR-CSC-NOS4` cell. The [Day 2 proposal](day2-timing-methodology-proposal.md) supplies the workload, row, metric, statistics, raw evidence, and invalid-trial rules; this document resolves its open choices. The numerical rules are engineering acceptance limits for one cell, not conclusions drawn from prior timing samples. No baseline is accepted and no CI timing gate is active yet.

| Decision | Locked `s225-v1` rule | Reason / remaining proof |
| --- | --- | --- |
| Runner scope | Use one macOS `macos-latest` timing job. A baseline's **resolved** image name/version, OS version, architecture, CPU model, `cc` path/full version, build recipe, flags, index width, and thread/backend environment define the exact class; every baseline job and candidate must match it. Unknown or blank identity invalidates the job. | Sprint 212's Linux label crossed CPU models, and macOS historical times still varied. The label alone is never a class. Hosted collection must demonstrate that repeated macOS jobs actually share one fingerprint; otherwise promotion stays blocked. |
| Build | Clean serial Make, `SPARSE_IDX_BITS=32`, `CC=cc`, default `-O2` Make flags, one thread, no sanitizer, coverage, OpenMP or external dense-backend override. Hash library, executable and fixture; record full source SHA and clean state. | Baseline/candidate binaries may differ because source commits differ; the recipe and measurement path must be comparable. A mere `default_make_flags` label is insufficient. Collector build-recipe verification is still pending. |
| Workload and clock | `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`; exactly one `chol_spd`/`nos4.mtx`/`n=100` row. Use the CSV's positive finite `refactor_csc_ms` from its monotonic CSC interval. | Whole-process time and the within-run speedup ratio measure other things. CSV precision is 0.001 ms; any zero or inadequate-resolution evidence invalidates the job and calls for a new policy and baseline. |
| Trial schedule | Per hosted job: 5 warmup processes, then 20 measured processes, serial. Baseline: three independent jobs on one baseline commit and fingerprint. Candidate: one job, repeated later for hosted stability audit. | A process is a trial; internal `--repeat 1` is one refactor per process. Five warmups and 20 measurements close the single-sample gap while keeping a bounded job. |
| Timeout and job cap | 30 seconds per process; dedicated timing job cap **20 minutes**, separate from existing 10-minute freshness jobs. Capture partial streams and invalidate the whole job on timeout. | Worst-case 25 timed-out processes consume 12.5 minutes before build/upload; 10 minutes cannot cover that schedule. Hosted runtime still must demonstrate adequate build/upload margin. |
| Baseline statistic | For each job, median of 20 measured decimal CSV values; MAD/median `<= 0.10`; nearest-rank p90/median `<= 1.50`. Across three baseline medians, `(max-min)/median_of_three <= 0.10`; reference is median of those three medians. | Bounds are pre-registered noise filters. If observed hosted noise exceeds them, reject the baseline. Do not relax limits using that same data and silently keep it. |
| Candidate decision | Validity and runner/recipe match first. Candidate median `<= 1.25 * reference`, MAD/median `<= 0.10`, p90/median `<= 1.50`; compare unrounded Decimal values and accept equality. Report dispersion failure separately from regression. | The 25% allowance is a scoped regression budget, not a portable speed claim. The existing S6 `500.0` ms local smoke ceiling is unrelated. |
| Invalid trials | Any failed, timed-out, missing, malformed, nonfinite, zero or negative warmup or measured trial invalidates the **whole job**; preserve all raw bytes and attempted ordinals. No selective retry, outlier deletion, or imputation. | Prevents favorable sample selection. A new full job may be attempted, with the failed job retained. |
| Baseline lifetime | Source-control the accepted baseline manifest and all baseline trial bytes after review. Candidate artifact retention target: 90 days, subject to repository policy. Baseline expires 30 days after collection and immediately on incompatible runner, fixture, compiler/build or metric change. | Seven-day freshness artifacts cannot serve as the sole baseline. Storage and retention feasibility still require hosted verification; no accepted baseline without them. |

The 25%, 10%, and 1.50 limits are intentionally versioned rather than tuned to Sprint 212's mixed-commit runs. Changing any limit, clock/metric, row format, runner class, process count, or warmup rule requires a new policy version and fresh baseline. The existing threshold-free freshness report retains `baseline=n/a`, `threshold=n/a`, `warmup=none_configured`, and `variance=not_computed_single_sample`; a timing decision belongs to a separate artifact and job.

## Collector candidate

Added `scripts/collect_refactor_csc_timing.py` for one `s225-v1` **baseline job bundle**. It checks the fixture digest and binary/library presence, records source and host/build context, invokes the exact command serially, writes raw stdout/stderr bytes for each trial, parses the selected metric, records per-trial status and hashes in `trials.tsv`, and writes `run.json`, a pending `decision.json`, and `hashes.sha256`. A POSIX timeout terminates the process group and retains partial streams. Official collection requires a clean hosted macOS checkout and nonblank image, CPU and compiler identity. `--dry-run` captures one warmup and one measured process with `dry_run_only` status; it cannot be accepted as a baseline.

The collector is a preparation step, not the final gate. Before hosted baseline intake, the workflow must enforce the claimed clean build recipe, verify the complete runner fingerprint, upload the bundle on failure as well as success, and bind the post-upload artifact ID/digest in a separate receipt. Baseline admission and candidate threshold logic remain Day 4-8 work. A stale or local binary, including the local dry-run binary, is never accepted merely because the collector labels a build recipe.

## Local dry-run proof

Commands executed:

```sh
make build/bench_refactor_csc SPARSE_IDX_BITS=32
python3 scripts/collect_refactor_csc_timing.py --output-dir /tmp/sprint225-day3-dryrun-v2 --dry-run
```

The collector reported `2 trials`, `valid=True`, `dry_run=True`. The two rows were warmup then measured; each retained stdout/stderr and its exact argv. All **seven** files named in `hashes.sha256` rehashed correctly. The selected `refactor_csc_ms` values in the final local tooling run were `0.058` and `0.067` ms. `run.json` recorded policy `s225-v1`, source `5de34925dc72ce896e2ba4f8e70d5f73741166d5`, `dry_run=true`, and the locked fixture digest. Those local values are **not** baseline samples, runner calibration, a variance estimate, or a timing pass. The bundle lives in `/tmp`, outside the repository.

## Remaining hosted blockers

1. Demonstrate three exact-fingerprint macOS hosted baseline jobs and independently inspect every retained raw file and hash. `macos-latest` may change the resolved image or CPU; a mismatch rejects the baseline group.
2. Confirm the full build, 25 trials and upload fit the dedicated 20-minute cap. If they do not, version a new policy *before* accepting any baseline.
3. Verify the actual compiler flags and index width used for both baseline and candidate; the Day 3 collector records a recipe string but does not independently enforce the build invocation.
4. Establish source-controlled baseline raw storage and the requested candidate-artifact retention or another durable review route.
5. Implement and test the independent baseline/candidate validator and CI timing job. Until then, the selected freshness jobs remain threshold-free and the timing gate remains **blocked**.

## Validation

`make build/bench_refactor_csc SPARSE_IDX_BITS=32` passed. The dry-run collector and independent seven-file SHA-256 check passed. Day 3 changed Python and planning documentation, no `.c` or `.h` source; the full C quality suite is not triggered by this day. Check Markdown links and whitespace before closeout.
