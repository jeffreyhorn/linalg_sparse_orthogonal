# Sprint 225 Retrospective

**Sprint:** 225 - One Controlled Canonical Timing Gate

**Duration:** 14 planned days on branch `sprint-225`

**Status:** The one-cell timing method, raw collector, fail-closed validator, separate macOS CI job, and negative guards were delivered. **The timing gate remains blocked:** no compatible three-job baseline was accepted, and every hosted candidate attempt stopped before measurement with `S225_BASELINE_MISSING`. No timing pass or performance regression result is claimed.

## Source and Definition of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) and [sprint plan](PLAN.md) allocate 140 estimated hours across six items and fourteen 10-hour days. These are scope estimates, not measured labor or hosted runtime. [Working notes](WORKING_NOTES.md) and the [Day 14 evidence decision](artifacts/day14-final-timing-decision.md) reconcile each day and the product outcome. The intended gate was limited to `SRT-BENCH-REFACTOR-CSC-NOS4`: serial, 32-bit `bench_refactor_csc` on `nos4.mtx --repeat 1`, the `refactor_csc_ms` metric, and one resolved macOS runner/compiler class. Existing Linux/macOS selected benchmark freshness remains threshold-free and independent.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 225.1 Methodology Lock | 24 h | Completed the `s225-v1` workload, fingerprint, 5+20 trial, timeout, dispersion, baseline-age, threshold, raw-evidence and non-claim rules before baseline evaluation. |
| 225.2 Baseline Collection | 28 h | **Blocked product outcome.** Three retained hosted baseline ZIPs were audited, but image/CPU class drift, one excessive job MAD, and excessive between-job spread required rejection. No active accepted baseline exists. |
| 225.3 Timing Gate | 28 h | Implemented the byte-preserving collector, separate validator and report verifier, baseline preflight, clean-build candidate workflow, and always-uploaded failure receipt. It fails closed while baseline acceptance is missing. A real passing candidate has not exercised the build/collection route. |
| 225.4 Negative Cases | 22 h | Covered workload, source/build, baseline, host, sample, timeout, dispersion, threshold, artifact and claim mutations with explicit rule codes and synthetic positive controls. |
| 225.5 Hosted Repetition | 20 h | **Blocked product outcome.** Three hosted attempts on one exact source SHA uploaded verified preflight receipts; all returned `S225_BASELINE_MISSING` before candidate collection. No repeated timing statistic exists. |
| 225.6 Claim Calibration | 18 h | Updated benchmark and maintainer guidance, audited the evidence chain, kept the status blocked, and handed missing baseline/candidate proof to Sprint 226. |
| **Total** | **140 h** | **All planned scope accounted for; timing-gate completion criterion unmet.** |

## What Went Well

1. **The method was locked before viewing candidate timing.** The [Day 3 decision](artifacts/day3-methodology-lock-and-collector.md) fixed the exact command, `refactor_csc_ms`, resolved runner and compiler fingerprint, serial 32-bit build, five warmups plus 20 measured processes per job, and three baseline jobs. It set MAD/median `<= 0.10`, p90/median `<= 1.50`, between-job spread `<= 0.10`, candidate median `<= 1.25 ×` accepted reference, 30-day baseline lifetime, 30-second trial timeout and separate 20-minute job cap. No observed favorable run changed those limits.
2. **Raw baseline evidence survived rerun artifact loss.** The [Day 5 rejected set](artifacts/day5-baseline-variance-decision.md) retains original ZIPs for attempts 1, 3 and 4. Each has 54 unique members, 53 matching member hashes, and 5 warmup plus 20 measured trials. The independently recomputed medians were `0.029`, `0.0315` and `0.033` ms. They are diagnostic values from a rejected group, not an accepted reference.
3. **The separate gate fails visibly and preserves review receipts.** The [Day 8 workflow](artifacts/day8-hosted-timing-gate.md) checks baseline acceptance before candidate work, reports a distinct rule, and uploads available host context and decision even on failure. The [Day 12 ledger](artifacts/day12-hosted-repetition-blocked.md) records three exact-source hosted attempts, distinct artifact IDs and matching ZIP digests; all are `blocked` with null timing values. Their ZIP bytes remain in [the sprint artifact directory](artifacts/day12-blocked-hosted-attempts/).
4. **Negative cases protect the narrow claim.** [Day 9 drift checks](artifacts/day9-workload-baseline-drift.md), [Day 10 weak-evidence tests](artifacts/day10-weak-evidence-claim-boundary.md), and [Day 11 closure](artifacts/day11-negative-matrix-hosted-intake.md) reject changed fixture, command, source/build, runner, expired or missing baseline, insufficient or invalid trials, excessive dispersion, regression and report overclaim. Synthetic passes verify the arithmetic and schema, while the real branch remains blocked.
5. **Final wording follows the evidence.** The [benchmark guide](../../../../benchmarks/README.md#sprint-225-one-cell-timing-decision) and [maintainer guide](../../../../docs/maintainer_guide.md) name the locked one-cell policy, artifact route and rerun prerequisites. They keep threshold-free freshness separate from a blocked timing decision and make no portable speed, platform parity, external-library ranking or state-of-the-art claim.

## What Did Not Go Well

1. **`macos-latest` did not produce one stable comparison class.** Day 5 attempts crossed resolved image versions and Apple M1/M2 Pro virtual CPUs. Attempt 3's MAD/median was `0.111111…`, above `0.10`; the three medians' spread was `0.126984…`, also above `0.10`. The group was rejected rather than trimmed or retroactively accepted. The baseline archive hash also varied despite identical executable bytes, leaving build-provenance investigation for a later collection.
2. **A rerun removed an unretained artifact.** Day 5 attempt 2's hosted job succeeded, but its ZIP was not downloaded before a later rerun. It was excluded from statistics and acceptance. Subsequent attempts were downloaded and rehashed before rerunning, yet the incident shows that hosted retention settings alone do not secure every run attempt.
3. **The hosted candidate did not reach measurement.** On source `38323f0e584e1a4d2aa8d2c92db5c18830366afc`, [run `44403764960`](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44403764960) failed at baseline preflight on all three attempts. Each receipt contains only host context and blocked decision, with no raw candidate stream or median. Attempt 2 again used a different resolved image and CPU. Candidate build/upload margin within the 20-minute job, real-sample dispersion, and repeated threshold decisions remain unproven.
4. **The project's desired Sprint 225 timing-gate exit was not reached.** The separate job correctly turns missing evidence red. Its synthetic pass fixtures are implementation checks, not runner-specific hosted proof. No accepted baseline pointer, versioned reference, candidate raw bundle, timing pass or regression verdict can be reported as completed.

## Final Metrics and Claim

| Measure | Audited result |
| --- | --- |
| Planned effort | 140 estimated hours; all 14 days accounted for, actual hours not tracked |
| Target | `SRT-BENCH-REFACTOR-CSC-NOS4` / `refactor_csc_ms` on `nos4.mtx --repeat 1` only |
| Method | `s225-v1`, one matched resolved macOS class, three baseline jobs and one candidate job, each 5+20 trials |
| Baseline | Rejected Day 5 group; no accepted version or `active.json` |
| Baseline ZIP audit | Three retained ZIPs, 54 unique members and 53 matching member hashes each; Day 5 dispersion and class failures reproduced |
| Candidate host evidence | Run `44403764960`, attempts 1-3, source `38323f0e584e1a4d2aa8d2c92db5c18830366afc`; three digest-verified two-file blocked receipts |
| Candidate timing | No raw trials, median, MAD, p90 or threshold verdict; all attempts returned `S225_BASELINE_MISSING` |
| Local verification | 17 timing-validator tests, two workflow tests, selected performance docs, selected freshness fixtures, API-docs policy guard, relative links and whitespace passed on Day 14 |
| C quality trigger | No `.c` or `.h` changes in the final closeout; `make format && make lint && make test` not required for this documentation work |

The supported sentence is: **“The `s225-v1` one-cell macOS timing job is implemented and fails closed because no compatible accepted baseline exists; hosted candidate timing has not been measured.”** This does not imply portable speed, cross-platform parity, an external-library ranking, release readiness, or a passing regression gate. The retrospective commit will have a new source SHA; the older blocked receipts remain valid historical evidence of failure behavior, not an exact-head candidate result for that new commit.

## Daily Evidence

- [Day 1 canonical target inventory](artifacts/day1-canonical-timing-inventory.md)
- [Day 2 methodology proposal](artifacts/day2-timing-methodology-proposal.md)
- [Day 3 policy lock and collector](artifacts/day3-methodology-lock-and-collector.md)
- [Day 4 first hosted baseline intake](artifacts/day4-hosted-baseline-intake.md)
- [Day 5 baseline rejection and raw ZIPs](artifacts/day5-baseline-variance-decision.md)
- [Day 6 fail-closed gate design](artifacts/day6-timing-gate-design.md)
- [Day 7 decision implementation](artifacts/day7-timing-decision-implementation.md)
- [Day 8 hosted workflow and first blocked receipt](artifacts/day8-hosted-timing-gate.md)
- [Day 9 workload and baseline drift matrix](artifacts/day9-workload-baseline-drift.md)
- [Day 10 weak evidence and claim matrix](artifacts/day10-weak-evidence-claim-boundary.md)
- [Day 11 negative-case closure and hosted intake](artifacts/day11-negative-matrix-hosted-intake.md)
- [Day 12 repeated blocked attempts](artifacts/day12-hosted-repetition-blocked.md)
- [Day 13 hosted closeout and claim calibration](artifacts/day13-hosted-closeout-claim-calibration.md)
- [Day 14 final blocked decision and handoff](artifacts/day14-final-timing-decision.md)

## Residual Handoff

Sprint 226 must carry the missing accepted baseline and candidate proof as a **blocked gate**, per its candidate-readiness exit rule. Establish a repeatable resolved runner/compiler/build class, or review a new policy version before collecting fresh data. Retain and independently verify three compatible baseline jobs, source-control the accepted manifest/raw bundles and active pointer, then rerun the candidate job on the exact proposed commit. Download each artifact before another rerun; recompute the report from raw bytes and compare repeated decisions. Keep selected benchmark freshness threshold-free and separately reported. A missing hosted timing result or reviewer approval cannot be relabeled a release-ready performance claim.
