# Sprint 225 Working Notes

**Branch:** `sprint-225`

**Base source commit:** `5de34925dc72ce896e2ba4f8e70d5f73741166d5` (merge of PR #248)

**Selected target:** `SRT-BENCH-REFACTOR-CSC-NOS4`
**Current timing-gate state:** **Blocked / not promoted**. Existing Linux/macOS selected benchmark freshness is threshold-free.

## Scope and decision ledger

| Day | Decision | Evidence | Remaining condition |
| --- | --- | --- | --- |
| 1 | Inventory one canonical `bench_refactor_csc`/`nos4.mtx --repeat 1` cell; keep freshness separate from timing. | [Day 1 inventory](artifacts/day1-canonical-timing-inventory.md); selected manifest; benchmark, report script, checker, Makefile, Linux/macOS workflows; Sprint 212 and Sprint 218 retrospectives. | Freeze methodology, collect an accepted hosted baseline, implement and verify a separate gate before any timing claim. |
| 2 | Draft a one-runner, one-cell timing method with 5 warmups, 20 measured processes per job, three baseline jobs, explicit dispersion and regression rules, and byte-preserving evidence. No numerical limit is locked yet. | [Day 2 methodology proposal](artifacts/day2-timing-methodology-proposal.md); Day 1 inventory; Sprint 212 threshold criteria. | Day 3 must review and lock the runner/build fingerprint, output precision, sample schedule, numerical limits, baseline age and retention before any accepted baseline collection. |
| 3 | Lock `s225-v1` method, specify a separate 20-minute timing-job cap, prepare a byte-preserving baseline collector, and verify a local dry run. The timing gate remains blocked. | [Day 3 lock and collector](artifacts/day3-methodology-lock-and-collector.md); `scripts/collect_refactor_csc_timing.py`; `/tmp/sprint225-day3-dryrun-v2` tooling bundle. | Hosted runner/build fingerprint, three accepted baseline jobs, durable storage, validator and CI wiring remain to be proven. |

## Evidence index

| Evidence | Status |
| --- | --- |
| Selected manifest row and one-row canonical CSV route | Inspected on Day 1. |
| Fixture SHA-256 | `6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80` on Day 1. |
| Current Linux/macOS hosted freshness jobs | Workflow definitions inspected on Day 1; no new hosted run or artifact verified for this sprint. |
| Stable named runner/compiler and binary provenance | `s225-v1` locks a single macOS resolved-image/CPU/compiler/build fingerprint, defined by accepted hosted baseline evidence; actual hosted stability and build-recipe enforcement remain unproved. |
| Warmups and independent process trials | `s225-v1` locks 5 warmups plus 20 measured processes per job and three baseline jobs. A local 1+1 dry run proved collection mechanics only. |
| Versioned baseline and dispersion/threshold policy | `s225-v1` locks 30-day baseline age, median/MAD/upper-tail rules and a 25% regression allowance. No baseline exists. Current selected freshness fields remain `n/a`/`not_computed_single_sample`. |
| Retained raw trial output and durable artifact route | Day 3 collector wrote byte-preserving local trial files and verified a seven-file SHA-256 manifest. Hosted upload, accepted source-controlled baseline, artifact receipt and 90-day candidate retention remain unverified. |
| Hosted timing decision | Blocked. No thresholded job or reviewed candidate evidence exists. |

## Planned effort ledger

The [plan](PLAN.md) allocates 10 engineering hours per day and 140 hours for the sprint; estimates are not actual tracked time. Days 1-2 address 20 hours of item 225.1. Day 3 allocates its final 4 hours to 225.1 (now complete as a policy lock) and 6 hours to 225.2 (collector preparation, baseline collection still open).

## Day 1 verification

- Inspected selected source, manifest, scripts, workflows, historical evidence and the Sprint 225 plan.
- Verified the fixture digest and base source revision.
- Documentation-only changes; no `.c` or `.h` edits. `git diff --check` is required before closeout.

## Day 2 verification

- Checked the proposed policy against the Sprint 212 threshold criteria and Day 1 benchmark timing semantics.
- Marked runner identity, numerical limits, output precision, and evidence retention as Day 3 review decisions, not accepted facts.
- Documentation-only changes; no `.c` or `.h` edits. Check links and whitespace before closeout.

## Day 3 verification

- Built `build/bench_refactor_csc` with 32-bit indices and ran the collector with `--dry-run` into `/tmp/sprint225-day3-dryrun-v2`.
- Independently checked 2 trial rows in warmup/measured order and all seven file hashes; this local tooling bundle is not baseline evidence.
- No `.c` or `.h` edits. Timing gate is still blocked pending hosted evidence and independent validation.
