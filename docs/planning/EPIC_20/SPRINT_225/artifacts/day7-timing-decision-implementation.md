# Day 7: Separate Timing Decision Implementation

## Decision

Implemented the `s225-v1` one-cell timing validator in `scripts/check_refactor_csc_timing.py`. It has no route through canonical freshness and returns `S225_BASELINE_MISSING` with exit code 2 on this branch because `benchmarks/baselines/refactor-csc-nos4/active.json` does not exist. The rejected Day 5 bundles remain rejected; their diagnostic median is not a reference value. A synthetic accepted manifest is used only in tests.

## Inputs and evaluation

The validator reads an explicit active pointer and SHA-256-bound manifest, then checks three named raw baseline ZIPs before reading candidate timing values. It validates the accepted status, source commit, approval, hosted job identities, runner/compiler/build fingerprint, fixture and source digests, timestamps, 30-day expiry, 5+20 schedule, 53 member hashes, raw CSV row and metric, per-job dispersion, and between-job spread. It rejects the Day 5 schema and incompatible runner group even if a test rewrites a pointer and manifest around one of those ZIPs.

The candidate route accepts a raw ZIP or directory, verifies exact membership and all raw stream hashes, and independently parses all 25 trials, including warmups. Every trial needs the selected `nos4.mtx` row, exact argv, zero exit, no timeout, positive finite metric, and at most 30 seconds of recorded outer elapsed time. The candidate must match the accepted runner and measurement fingerprint; source and executable SHA values are recorded separately because candidate code can change. The CLI can require an exact candidate commit with `--expected-candidate-commit`.

For 20 measured values, the validator uses Decimal values from raw CSV, median, median absolute deviation divided by median, and nearest-rank p90 divided by median. A candidate with MAD/median above `0.10` or p90/median above `1.50` is unstable; otherwise a median above `1.25 ×` the three-job reference median is a regression. Equality passes. It writes `timing-decision.json` even for missing baseline evidence, with explicit rule code, status, hashes or hash-manifest reference, sample counts, runner identity, baseline job medians/spread, and exact decimal statistics. Exit 0 means pass, 1 means valid noisy or regressed data, and 2 means missing or invalid evidence.

The collector in `scripts/collect_refactor_csc_timing.py` now accepts `--role candidate` and writes schema version 2 plus benchmark-source and Makefile digests. Its receipt remains pending review; it cannot itself authorize a timing pass. The default role remains baseline for the existing intake workflow.

## Verification

`python3 tests/test_check_refactor_csc_timing.py` runs eight test methods covering absent accepted baseline; synthetic below-threshold pass; over-threshold regression; noisy candidate precedence; runner mismatch; corrupt raw stream; a rehashed warmup over 30 seconds; invalid metric precision or magnitude; missing baseline approval; and a forged pointer around a rejected Day 5 ZIP. Pass and regression fixture calculations use `0.100` ms reference, `0.110` ms passing candidate, `0.130` ms regressed candidate, and an exact `0.12500` ms ceiling. `python3 -m py_compile` passes for the collector, validator, and test. A local `--role candidate --dry-run` collected one warmup and one measured trial, both valid; it is tooling verification only. Running the CLI against the real branch writes a blocked report and exits 2 with `S225_BASELINE_MISSING`.

No `.c` or `.h` files changed. The requested C format/lint/test sequence is therefore not triggered. The separate hosted candidate job, verified build invocation and archive provenance, durable candidate upload, and an independently reviewed three-job accepted baseline remain for later work. Until those exist, this implementation is a fail-closed decision tool, not a passing CI timing gate.
