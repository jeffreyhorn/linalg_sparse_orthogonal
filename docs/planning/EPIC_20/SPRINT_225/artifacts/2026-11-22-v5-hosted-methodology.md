# Proposed s225-v5 hosted paired timing method

**Status: proposed before first v5 hosted collection.** This is a new
measurement version. The [v3 hosted run
`45073302636`](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/45073302636)
failed its locked dispersion rule and remains rejected. The separately
accepted local v4 cell for commit
`d637d4e0e832e49f7409f87fdf7bc9bc4b86582c` supports only that
exact candidate and its own dedicated-machine evidence. Neither result
is reclassified by this method.

## Locked hosted cell

The fixed baseline source is
`c42d6715fad7e895bae4941a006df3563f1afc38`; the candidate is the
exact pull-request or master-push commit. Both sides use clean source,
identical `bench_refactor_csc.c` and `nos4.mtx` bytes, the pinned serial
32-bit Make recipe, and `OMP_NUM_THREADS=1`. Measure only
`refactor_csc_ms` from
`build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 10000`
on the SPD `chol_spd` row. The longer repeat reduces process-level
timing noise relative to v3's `--repeat 1000`; it changes the workload
and therefore requires a new version. No v3 or v4 samples are pooled with
v5.

Three independent `macos-15-intel` hosted jobs each build both sides on
the same runner. Each job collects five warmup and 20 measured pairs,
alternating baseline/candidate order. Side subprocesses have a 30-second
timeout. The collector applies a 16-minute monotonic deadline within the
20-minute job, preserving time for cleanup and artifact upload. Failed or
incomplete jobs retain a blocked receipt and cannot pass.

## Decision and acceptance boundary

From each measured pair's verified raw CSV, compute
`candidate_ms / baseline_ms` using decimal arithmetic. Per job require
MAD/median `<= 0.10`, nearest-rank p90/median `<= 1.50`, and median
ratio `<= 1.25`. Across the three job medians require
`(max-min)/median <= 0.10`. These are the v3/v4 numerical limits,
unchanged. The validator requires the exact v5 workload, all 50 ordered
side trials, matching source/build/fixture and runner class, raw stream
hashes, valid timestamps and timeouts, and matching executable identities
across jobs. Missing, invalid, noisy, or regressed evidence fails closed.
No job or trial may be discarded to obtain a passing result.

A numerical v5 pass remains subject to independent review of the
exact-head original artifact ZIPs, digests, source and binary provenance,
raw trial rows, and recomputed statistics before acceptance. This is one
selected cell, not a portable speed, external ranking, or release-readiness
claim. The accepted local v4 record does not substitute for hosted v5
evidence.
