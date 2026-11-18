# Day 11: Negative Matrix Closure and First Hosted Candidate Attempt

## Negative-case matrix

Item 225.4's six required rejection classes have positive and negative fixtures. The suite uses synthetic accepted baseline data only. This matrix closes the planned 22 hours of negative-case implementation; it does not convert the real timing gate into a passing claim.

| Required class | Positive control | Negative fixture and observed result |
| --- | --- | --- |
| Changed workload | Canonical `nos4.mtx --repeat 1` bundle passes schema/identity in the synthetic fixture. | Changed fixture, command, repeat count, benchmark source, backend or build mode: `S225_CANDIDATE_IDENTITY`, exit 2. Checkout byte changes also fail. |
| Missing or old baseline | Synthetic reviewed pointer, three raw jobs and valid 30-day expiry reach the candidate decision. | Missing pointer/manifest/raw ZIP: `S225_BASELINE_MISSING`, exit 2. Expired, superseded, wrong-version or mismatched manifest: `S225_BASELINE_INVALID`, exit 2. |
| Single or incomplete sample | Five warmups and 20 measured samples pass count/shape checks. | One measured sample or omitted TSV row: `S225_CANDIDATE_INVALID`, exit 2. Missing warmup raw output: `S225_CANDIDATE_MISSING`, exit 2. |
| Unknown or different host | Matching resolved macOS image, CPU and compiler pass identity checks. | `unknown` image or changed CPU/compiler: `S225_CANDIDATE_IDENTITY`, exit 2. |
| Variance and threshold overclaim | Uniform `0.125` ms candidate passes the exact `1.25 × 0.100` ms synthetic ceiling. | Ten `0.100`/ten `0.200` ms samples: `S225_CANDIDATE_DISPERSION`, exit 1. Uniform `0.126` ms: `S225_TIMING_REGRESSION`, exit 1. |
| Claim or publication promotion | A matching raw bundle and report stay scoped to one named cell and runner. | Freshness-derived or portable/parity claim mutation, missing raw evidence, or changed report provenance: `S225_REPORT_INVALID`, exit 2. Workflow mutation removing raw upload or swallowing failure fails its guard. |

The same suite also covers timeout, malformed trial time, altered raw-stream hash, invalid metric precision, and a rejected Day 5 ZIP wrapped in forged accepted metadata. Failed decisions preserve available raw candidate bytes. `python3 tests/test_check_refactor_csc_timing.py` passed 17 test methods; the separate workflow mutation suite passed 2 methods and its six mutation cases. Python compilation, structured-workflow inventory and policy checks passed. No C or header files changed.

## Exact-head hosted attempt

Pushed source commit `38323f0e584e1a4d2aa8d2c92db5c18830366afc` and inspected [timing workflow run 44403764960](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44403764960), job `135153401892`, attempt 1. The run used `macos26` image version `20261102.0473.1`, `macOS-26.7.2-arm64-arm-64bit-Mach-O`, Apple M1 (Virtual), and `/usr/bin/cc` reporting `Apple clang version 21.0.0 (clang-2100.1.1.101)`. These fields came from the preflight host record, not a candidate collection. The host record identifies workflow job `canonical-refactor-csc-timing` and the exact source SHA.

The baseline preflight exited 2 with `S225_BASELINE_MISSING`. The clean build, raw candidate collection, and timing decision steps were skipped; the artifact upload passed. Artifact `sprint225-timing-44403764960-1`, ID `14473492579`, has GitHub SHA-256 `6ee6c5306ff2d55f9fd0748ba66ffa96ef0df7619183b356837b9a00f8725f2e`. I downloaded the ZIP and independently computed the same SHA-256. It contains exactly `sprint225-host-provenance.json` and `sprint225-timing-decision.json`. The report has schema 2, policy `s225-v1`, `status=blocked`, `primary_rule=S225_BASELINE_MISSING`, and null reference/candidate medians. The exact commit's tree has no `benchmarks/baselines/refactor-csc-nos4/active.json`, corroborating the failure.

## Disposition

There is **no hosted candidate bundle** to download or remeasure, so no candidate statistic, dispersion, or pass/fail threshold verdict exists. The upload proves the failure route, not timed performance. The first hosted candidate remains blocked by the rejected Day 5 baseline group; item 225.5's first 8 planned hours document this attempted run and its evidence, not a completed candidate measurement. No threshold, runner-class equality rule, or claim boundary was relaxed. A new, compatible, independently accepted three-job baseline is required before the candidate job may run its 5+20 trials.
