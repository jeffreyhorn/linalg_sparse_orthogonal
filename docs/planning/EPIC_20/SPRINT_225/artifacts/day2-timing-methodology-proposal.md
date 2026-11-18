# Day 2: Proposed Canonical Timing Methodology

**Status:** Draft for Day 3 review and lock. The numeric limits and runner choice below are proposed policy, not an accepted baseline or an active timing gate. No baseline or candidate timing was collected for this document.

## One-cell scope

| Dimension | Proposed contract |
| --- | --- |
| Selected target | `SRT-BENCH-REFACTOR-CSC-NOS4` only. |
| Executable and command | Repository-built `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`, from repository root. No `--indefinite-kkt` or extra positional fixture. |
| Fixture | Exact source-controlled `tests/data/suitesparse/nos4.mtx`; starting digest `6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80`. Digest changes require a new methodology and baseline. |
| Scenario and row | Exactly one CSV data row: `benchmark=bench_refactor_csc`, `category=proof`, `matrix=nos4.mtx`, `scenario=chol_spd`, `n=100`; reject an extra, missing or malformed row. |
| Metric | Parsed `refactor_csc_ms` from the benchmark CSV, in milliseconds, positive and finite. The program uses `clock_gettime(CLOCK_MONOTONIC)` around CSC conversion plus supernodal elimination. Do not substitute whole-process time, `refactor_public_ms`, or `speedup_refactor`. |
| Resolution | Current CSV prints three decimal places in milliseconds. Treat those printed decimal values as the authoritative samples and compare them without further rounding; if the resolution produces zero or unstable samples, block the gate and review a new benchmark/output protocol with a fresh baseline. |
| Backend | Default SPD Cholesky path, AMD analysis, CSC supernodal elimination with the existing argument `4`; no backend-selection environment override. |

`--repeat 1` means one internal refactor in **each** process. The proposed trial count below is a count of independent process invocations. Matrix load, analysis, initial factor priming, copying, perturbation, solves, and residual calculation are outside the selected `refactor_csc_ms` interval. This is a CSC refactor timing gate, not an end-to-end solve timing gate.

## Hosted runner and build proposal

Use one dedicated macOS hosted job derived from the existing `selected-performance-freshness` route, provisionally `macos-latest` with Apple Clang. The existing macOS route has shown one CPU-model label (`Apple M1 (Virtual)`) in Sprint 212; its timing still varied materially, so that history is **not** a baseline. The new job must record and require one exact resolved OS image/version, architecture, CPU model, compiler executable/version, CMake or Make build command, optimization flags, index width, library configuration, thread settings, and environment allowlist. The acceptance class is the full recorded fingerprint, not the `macos-latest` label. Day 3 must decide how to pin the image and verify the runtime fingerprint; if a stable class cannot be named and enforced, baseline collection and gate promotion remain blocked. Do not combine Linux and macOS samples.

Proposed build: clean serial Make build using the repository's default release-style flags, `CC=cc`, no sanitizer, coverage, OpenMP, or external dense-backend override. Record the compiler path and full `cc --version` output, build command, relevant `CFLAGS`/`CPPFLAGS`/`LDFLAGS`, library and executable SHA-256, source revision, clean/dirty state, OS/CPU identity, and fixture SHA-256. The baseline's executable digest is provenance, not an equality key: candidate source revisions can legitimately produce a different binary. Build recipe and runner fingerprint must match; an ambiguous `default_make_flags` label alone does not satisfy this.

Run with one benchmark process at a time, no parallel benchmark or build activity in the same job after compilation. Pin thread settings to one thread; capture any relevant `OMP_NUM_THREADS` and backend override values. A CI queue or shared-host contention cannot be proven absent, so variance checks are mandatory. Never label this a dedicated hardware machine unless the hosted environment actually supplies one.

## Trial schedule and invalid-trial treatment

1. A **baseline collection** consists of three independent hosted job runs on the same full runner/build fingerprint, each with 5 warmup processes followed by 20 measured processes. Run them against one exact baseline source commit and methodology version. The total planned baseline is 15 warmups and 60 measured samples.
2. A **candidate decision** consists of 5 warmup processes followed by 20 measured processes on the same full runner/build fingerprint and one exact candidate source commit. Candidate evaluation does not change the accepted baseline. Days 11-12 repeat candidate jobs to evaluate stability, but a single job's 20 samples produces one CI verdict.
3. Invoke the exact command in a fresh process for each trial, in serial order. Capture monotonic outer start/end timestamps and elapsed wall time for diagnostics only; the decision metric remains the benchmark's `refactor_csc_ms` field. Capture the raw stdout and stderr bytes, exit code, timeout flag, process start/end UTC, and ordinal for every invocation. Use a provisional 30-second per-process timeout and the CI job's 10-minute cap; verify on Day 3 that the full baseline/candidate schedule and upload fit those caps. Preserve a timed-out trial's partial streams.
4. A warmup is never used in statistics. Any missing, nonzero-exit, timed-out, malformed, nonfinite, zero, or negative warmup or measured sample invalidates that whole job. Do not retry only the bad trial or replace it with a favorable sample. A whole new job may be run with the failed job retained and linked. Report infrastructure failures distinctly from valid over-threshold failures.
5. No automatic outlier deletion, winsorization, or adaptive sample-count extension. Every sample and every excluded-by-design warmup is retained and counted.

The 5/20 counts allow a robust median and a visible upper-tail check while keeping work bounded for a 10-minute hosted job. Sprint 212's one process, no warmup, and no variance were insufficient; this protocol explicitly closes those gaps. The specific counts and timeout require Day 3 review before collection.

## Frozen-statistic and provisional decision proposal

For a sorted set of 20 positive finite measured values, the **median** is the arithmetic mean of positions 10 and 11 (one-based). The **MAD** is the median of the 20 absolute deviations from that median. Define relative dispersion as `MAD / median`. Define `p90` by nearest rank, position `ceil(0.90 * 20) = 18`, and upper-tail ratio `p90 / median`. Use decimal arithmetic on the printed CSV decimals; retain the unrounded intermediate results in the decision report. An exact boundary passes (`<=`), and no second rounding is allowed before comparison.

A baseline job is admissible only if relative dispersion `<= 0.10` and upper-tail ratio `<= 1.50`. Across the three job medians, require `(maximum - minimum) / median_of_three <= 0.10`. If all three jobs pass, the reference value is the median of their three medians. A candidate job passes only if its own relative dispersion `<= 0.10`, upper-tail ratio `<= 1.50`, and candidate median `<= 1.25 * reference value`. Candidate failure on dispersion is reported separately from a genuine timing regression. The 25% allowance is a provisional selected-workload regression budget, not a portable performance promise; the 10% and 1.50 bounds are provisional noise filters. Historical Sprint 212 macOS times ranged from `0.030` to `0.091` ms across commits and therefore do not justify any of these numeric limits. Day 3 must review the limits *before* collecting an accepted baseline. Any change after inspecting data increments the policy version, is documented, and requires fresh baseline collection and re-review.

Only a valid baseline with all 60 measured samples and complete raw evidence may be accepted. Missing, older, incompatible, or expired baseline means **no timing pass**. Do not fall back to the local S6 `500.0` ms smoke ceiling or selected freshness. If a candidate exceeds either threshold, fail the separate timing job but still upload its evidence. The selected freshness result remains `measurement` with `baseline=n/a` and `threshold=n/a`.

## Baseline version, update and retention proposal

Version a baseline by policy version, full source SHA, fixture digest, runner/build fingerprint, three hosted run/job IDs, 60 raw sample hashes, statistics, collection timestamps, and an explicit approval record. Propose a 30-day maximum baseline age at candidate run time. Recollect and review after age expiry, runner image/compiler change, fixture change, metric/output change, or benchmark measurement-path change. A normal candidate source change does not require identical executable bytes, but its build recipe and metric implementation must be checked against the baseline; incompatible implementation changes require a new baseline. No automatic baseline update on a candidate PR.

Retain the accepted baseline manifest and raw baseline trial files in a source-controlled versioned directory, subject to repository size/review approval. Retain all candidate trial files as a dedicated hosted artifact with an explicit retention setting long enough for review (proposed 90 days, subject to repository policy). If that setting or durable baseline storage is unavailable, do not promote the gate. Existing seven-day freshness artifacts are insufficient as the sole baseline record. The collector must write its report before a comparison failure, and the workflow must upload on success and failure.

## Raw-sample and provenance schema proposal

One immutable bundle per job, with byte-preserving files and deterministic names:

| File | Required contents |
| --- | --- |
| `run.json` | `schema_version`, `policy_version`, `role` (`baseline` or `candidate`), full source SHA and clean-state flag, workflow run ID/attempt and job identifier available at collection time, UTC start/end, runner image/version, OS/arch/CPU, compiler path/full version, build recipe and flags, thread/backend environment, executable/library/fixture hashes, exact argv, process timeout, expected warmup/measured counts, metric column and units. Required identity fields cannot be unknown or blank. |
| `trials/0001.stdout` and `trials/0001.stderr` through the final ordinal | Raw captured bytes, including empty streams. Each trial has two files even on timeout or failure. |
| `trials.tsv` | One row per attempted ordinal in execution order: phase (`warmup` or `measured`), argv identity, UTC timestamps, monotonic outer elapsed, exit status or timeout, metric value when parseable, stdout/stderr SHA-256, and validity reason. No dropped ordinal. |
| `hashes.sha256` | SHA-256 for every raw stream and metadata file except the hash manifest itself; reject missing, extra, or altered files. |
| `decision.json` | Policy version, baseline version/hash, exact raw sample set, sample counts, unrounded statistic/dispersion/threshold values, rule codes, final verdict, and explicit `freshness_result=separate`. The baseline bundle records admissibility rather than a candidate timing verdict. |

The hosted artifact ID, URL, upload digest, retention deadline, and final job conclusion become available **after** upload. Record them in a separate run-to-artifact receipt linked to the already hashed bundle; do not fabricate an artifact ID inside `run.json` or alter the uploaded files after hashing. If the CI environment does not expose a stable job identifier during collection, use the run ID, attempt and workflow job key in `run.json`, then bind the API job ID in that receipt.

Prefer an append-safe collector that records a trial immediately after it ends, then validates bundle completeness. If a process dies, retain partial output and mark it invalid. The validator must recompute statistics from `trials.tsv` and parsed raw stdout rather than trusting `decision.json` alone. Bind candidate evidence to the accepted baseline manifest hash. An artifact URL alone is insufficient; download and verify bytes before citing a hosted result.

## Review questions for Day 3

1. Can the macOS job be fixed to a named image and strict CPU/compiler/build fingerprint, and does the hosted environment expose all required identifiers without blank sentinels?
2. Does 5 warmups plus 20 measured processes fit within the 10-minute job cap, including serial build and upload? If not, revise the counts or job cap *before* baseline collection.
3. Is three-decimal-millisecond CSV precision sufficient for `nos4` at the selected runner? If not, version a higher-resolution metric and recollect every baseline sample.
4. Are the 25% allowance, 10% dispersion, 1.50 upper-tail ratio, and 30-day baseline age acceptable as a reviewed single-cell policy? No historical cross-commit timing should be treated as proof of those values.
5. Can baseline raw files be source-controlled and candidate artifacts retained for the proposed period? If not, establish another inspectable route before gate promotion.

## Day 2 outcome

This draft specifies a reproducible trial order, statistics, failure behavior, and raw-evidence format. It does **not** assert that a stable runner, reviewed threshold, acceptable baseline, or enforced timing job exists. The Day 3 lock must resolve the questions above before any measured run is accepted.

## Validation

Day 2 changes planning documentation only. Check relative links and whitespace; C source quality checks are not triggered.
