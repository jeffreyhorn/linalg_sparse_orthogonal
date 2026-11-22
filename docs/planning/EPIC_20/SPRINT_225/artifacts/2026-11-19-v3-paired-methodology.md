# Sprint 225 proposed `s225-v3` same-machine timing protocol

**Review status: proposed.** The earlier [pinned `s225-v1` group](2026-11-19-pinned-intel-baseline-decision.md)
and [approved-method `s225-v2` group](2026-11-19-v2-baseline-decision.md)
were rejected under their locked dispersion rules. They cannot become the
reference for this protocol. This document defines the new method before
collecting any `s225-v3` candidate evidence. Independent methodology and raw
evidence review are required before treating a numerical pass as accepted.

## Selected scope and source identity

The only measured cell is the `refactor_csc_ms` CSV field from
`build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1000`
on the SPD `chol_spd` row (`n=100`, `nnz=594`). Each subprocess averages
1,000 refactors; this is a new workload version, not a reinterpretation of
the previous `--repeat 1` or `--repeat 100` samples. Use a clean serial Make
build with `SPARSE_IDX_BITS=32`, default `cc`, `OMP_NUM_THREADS=1`, and no
backend, compiler, flag, or thread overrides. The immutable reference source
is master merge commit `c42d6715fad7e895bae4941a006df3563f1afc38`.
The candidate is the exact checked-out workflow commit. The chosen reference
commit has the same benchmark and Makefile bytes as the first `s225-v2`
candidate, but every pair rebuilds both sides on its own runner. A fixed
source commit is a versioned comparison target, not a historical timing
number or a claimed accepted baseline.

## Collection and decision

Three independent jobs run on `macos-15-intel`, each with a 20-minute cap.
The collector applies a 16-minute monotonic deadline across checkout,
builds, and trials, reserving the remaining job time for cleanup and the
`always()` raw-artifact upload. If that deadline is reached, the partial
evidence and a blocked receipt are uploaded; the pair cannot validate.
Raw collector output is written in the ignored `sprint225-pair-raw/`
directory at the repository root, outside the candidate `build/` tree
that `make clean` removes. Hosted run
[`45058540715`](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/45058540715)
failed before upload because its output was inside that tree; it produced no
valid pair evidence.
Feature branches use the pull-request trigger only; `master` uses the push
trigger. Running both for the same candidate at once would create competing
measurement groups on the hosted fleet. This scheduling rule does not waive
any trial or dispersion limit.
The complete resolved image/version, OS, architecture, CPU model, compiler
path/version, effective build settings, and both executable hashes must
match across the three jobs. Within each job, the baseline and candidate
are built from separate clean worktrees on the **same machine**. Each job
collects five warmup pairs and 20 measured pairs. Odd-numbered pairs run
baseline then candidate; even-numbered pairs run candidate then baseline.
Every side runs as a separate process, with a 30-second timeout and process
group termination. Trial order, timestamps, exit status, raw stdout/stderr,
their SHA-256 hashes, build receipts, source/fixture hashes, job identity,
and runner identity are retained. The three matrix artifacts and the
independent decision artifact have run-and-attempt-specific names and target
90-day retention. An absent or malformed artifact blocks the decision.
The decision binds each extracted artifact by the SHA-256 of its
`hashes.sha256` member; the independent reviewer also records and checks
each original artifact ZIP SHA-256 against GitHub's artifact digest.

For each measured pair, compute `candidate_ms / baseline_ms` from the
verified raw CSV values using decimal arithmetic. Warmups do not enter
statistics. Per job, compute the median of its 20 ratios, MAD/median, and
nearest-rank p90/median. Require each job's MAD/median `<= 0.10` and
p90/median `<= 1.50`. Require the spread of the three job median ratios,
`(max-min)/median`, to be `<= 0.10`. Require **each** job median ratio to
be `<= 1.25`; equality passes. No job or trial may be discarded to obtain
a favorable result. The decision script requires identical committed benchmark
source hashes on both sides so the timed metric has the same implementation.
It checks the 50 ordered side trials per job, raw hashes and CSVs, source
commits and committed source hashes,
the exact run/attempt identity, matched runner/build/executable class, and
all three recomputed statistics. Exit 0 is a bounded numerical pass, exit 1
is valid regression or unstable timing, and exit 2 is invalid or missing
evidence. The decision is uploaded even when collection fails.

## Review and claim boundary

The three paired jobs compare one source-pinned implementation against one
candidate on one macOS runner class. A numerical pass does not by itself
approve release readiness or establish portable speed, cross-platform
timing parity, external-library ranking, or state-of-the-art performance.
An independent reviewer must inspect this method, all original downloaded
artifact bytes and member hashes, source and binary provenance, sample
statistics, and the exact-candidate decision before the result is accepted.
Changes to the reference commit, command, schedule, class, limits, or
decision rule require another version and a fresh predeclared collection.
