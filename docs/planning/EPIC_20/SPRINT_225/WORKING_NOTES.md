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
| 4 | Collect and audit the first exact-commit macOS baseline job; retain a pending review state because `s225-v1` requires three compatible jobs. | [Day 4 hosted intake](artifacts/day4-hosted-baseline-intake.md); [run 44395634285](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44395634285); artifact ID `14469044864`, ZIP SHA-256 `3093366692075f3fedabc0735bf6425b2b23ffb7880403919f338cf6cbc100fe`. | Two more exact-commit hosted runs, cross-run dispersion review, accepted raw baseline storage and validator remain open. |
| 5 | Reject the `s225-v1` baseline group. Retained attempts 1, 3, and 4 have complete raw trials but differ in runner class; attempt 3 also fails MAD and the group fails cross-run spread. Attempt 2 lost its artifact on rerun and is excluded. | [Day 5 variance and provenance decision](artifacts/day5-baseline-variance-decision.md); three locally retained, rehashed ZIPs in `artifacts/day5-rejected-baseline/`. | Review a pinned runner or new same-machine policy version; recollect three comparable jobs and retain each artifact before any rerun. No baseline accepted. |
| 6 | Close the `s225-v1` attempted baseline as rejected and specify a fail-closed, independent candidate timing-gate contract. | [Day 6 gate design](artifacts/day6-timing-gate-design.md); Day 5 worksheet and raw ZIPs; existing collector and freshness checker. | Implement the validator and candidate collection path; establish a compatible accepted baseline before any timing pass. |
| 7 | Implement a separate raw-evidence timing decision and candidate collector role; prove missing baseline blocks the actual branch. Synthetic accepted evidence exercises threshold and dispersion decisions only. | [Day 7 implementation and verification](artifacts/day7-timing-decision-implementation.md); `scripts/check_refactor_csc_timing.py`; `tests/test_check_refactor_csc_timing.py`. | Wire a separate hosted candidate job and artifact route; prove the build invocation and obtain three independently accepted, compatible baseline ZIPs before any gate passes. |
| 8 | Add a separate macOS candidate timing job with accepted-baseline preflight, clean serial build, raw candidate collection, independent decision, and always-uploaded failure receipt. Hosted run `44400684440` failed as expected with `S225_BASELINE_MISSING` and uploaded its report. | [Day 8 workflow, hosted result, and operator notes](artifacts/day8-hosted-timing-gate.md); `.github/workflows/sprint225-timing.yml`; artifact `14471524335` with verified ZIP digest. | Recollect and approve a compatible three-job baseline; verify actual build and archive provenance before a timing pass can be claimed. |

## Evidence index

| Evidence | Status |
| --- | --- |
| Selected manifest row and one-row canonical CSV route | Inspected on Day 1. |
| Fixture SHA-256 | `6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80` on Day 1. |
| Current Linux/macOS hosted freshness jobs | Workflow definitions inspected on Day 1; no new hosted run or artifact verified for this sprint. |
| Stable named runner/compiler and binary provenance | `s225-v1` locks a single macOS resolved-image/CPU/compiler/build fingerprint. Day 5 observed image/CPU drift and changing library archive hashes. A compatible accepted class and build-recipe proof remain unestablished. |
| Warmups and independent process trials | `s225-v1` locks 5 warmups plus 20 measured processes per job and three baseline jobs. A local 1+1 dry run proved collection mechanics only. |
| Versioned baseline and dispersion/threshold policy | `s225-v1` locks 30-day baseline age, median/MAD/upper-tail rules and a 25% regression allowance. Day 5's versioned attempted group is rejected; no active baseline exists. Current selected freshness fields remain `n/a`/`not_computed_single_sample`. |
| Retained raw trial output and durable artifact route | Three hosted raw ZIPs were rehashed and retained in the sprint artifact worktree. Earlier GitHub artifacts disappeared on rerun; accepted source-controlled baseline storage and candidate retention remain unproven. |
| Hosted timing decision | Separate workflow added on Day 8. Run `44400684440` failed at preflight and uploaded a digest-verified blocked decision. No candidate trials or passing evidence exist. |
| Day 7 timing validator | Implemented and tested against synthetic accepted evidence. Real branch exits 2 with `S225_BASELINE_MISSING`; no accepted reference exists. |
| First hosted baseline intake | Run `44395634285`, job `135124640857`, source `7e49ef3a96e00dcc815da3ba83b006934815c242`; 5 warmups + 20 measured, all valid, 54 ZIP files/53 verified member hashes. **Rejected as part of the incompatible Day 5 group.** |
| Repeated hosted baseline audit | Attempts 1, 3, and 4 each had 5 warmups + 20 valid measured trials and 53 valid member hashes. Their image/CPU fingerprints disagreed; attempt 3 MAD/median was `0.1111`, and cross-job spread was `0.12698` (both above `0.10`). **Rejected.** Attempt 2 has no retained ZIP and was excluded. |

## Planned effort ledger

The [plan](PLAN.md) allocates 10 engineering hours per day and 140 hours for the sprint; estimates are not actual tracked time. Days 1-2 address 20 hours of item 225.1. Day 3 allocates its final 4 hours to 225.1 (complete as a policy lock) and 6 hours to 225.2. Days 4-5 allocate 20 hours to 225.2; Day 6 allocates its final 2 hours to the documented rejection disposition, accounting for all 28 planned hours of 225.2. The **baseline deliverable remains blocked**, so the item is not marked achieved. Day 6 allocates 8 hours to item 225.3's gate design; Day 7 allocates 10 hours to the validator; Day 8 allocates its final 10 hours to separate CI wiring, accounting for all 28 planned hours. A hosted decision and accepted baseline are still required for a passing timing outcome.

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

## Day 4 verification

- The branch-only hosted intake workflow ran on pushed commit `7e49ef3a` and completed successfully. The prior plan/collector changes were committed and pushed to make the exact-source hosted run possible.
- Downloaded the artifact ZIP, matched its SHA-256 with GitHub's digest, verified all member hashes and selected trial identities, and confirmed no failed or missing trial.
- Day 4 notes are uncommitted; no second baseline run or accepted baseline is claimed. Existing selected freshness remains threshold-free.

## Day 5 verification

- Reran the same hosted commit and independently audited attempts 1, 3, and 4. Stored ZIP bytes in the sprint artifacts; their SHA-256 digests match the artifact API digests observed at download time.
- Attempt 2's job succeeded but its artifact was not downloaded before another rerun removed it. It is excluded from statistics and acceptance.
- Exact decimal checks rejected attempt 3 on MAD and the group on runner identity and across-run spread. No timing threshold or baseline was promoted.

## Day 6 verification

- Closed the attempted `s225-v1` baseline disposition as rejected; no active baseline pointer or reference value was created.
- Specified exact baseline/candidate input checks, ordered diagnostics, decision report fields, CI upload behavior and separation from threshold-free freshness.
- Documentation-only update; no `.c` or `.h` edits. Timing-gate implementation and hosted acceptance remain open.

## Day 7 verification

- Python compilation and eight validator tests passed, covering real missing baseline, synthetic pass/regression/noise, runner mismatch, corrupt raw output, timeout, invalid metric, absent approval, and rejection of a Day 5 bundle despite forged accepted metadata. A local candidate-role dry run collected two valid trials for tooling verification.
- The actual CLI wrote a machine-readable `S225_BASELINE_MISSING` report and returned exit 2. No accepted baseline, passing hosted candidate decision, or numerical performance claim was created.
- No `.c` or `.h` edits. Checked Markdown links and whitespace; existing selected freshness remains separate and threshold-free.

## Day 8 verification

- Added baseline-only preflight, a separate 20-minute macOS candidate timing workflow, a clean-build log, and an always-uploaded candidate/report artifact path with a run/attempt-specific name.
- The local preflight CLI emits a blocked `S225_BASELINE_MISSING` report; synthetic accepted preflight is `ready`, not timing `pass`.
- Python compilation and 11 validator tests passed. The structured workflow inventory and API-docs policy checks passed with 121 records and 11 uploads; no freshness report was changed.
- Hosted run `44400684440` failed with exit 2 and `S225_BASELINE_MISSING`; upload passed. Downloaded artifact `14471524335`, verified its SHA-256 against the GitHub digest, and inspected the single blocked decision JSON.
- No `.c` or `.h` edits. Actual candidate/build timing remains untested until an accepted baseline exists.
