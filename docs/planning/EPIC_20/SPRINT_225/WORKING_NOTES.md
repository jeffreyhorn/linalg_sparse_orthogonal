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
| 9 | Reject candidate workload, file, host/compiler, source/build, and baseline-version drift with intended rule codes; keep one synthetic canonical positive fixture. | [Day 9 expected/observed drift ledger](artifacts/day9-workload-baseline-drift.md); checkout-bound validator, hosted `--verify-checkout`, 14 timing test methods. | Finish weak-sample and claim-boundary negative cases on Day 10; a reviewed baseline is still required for any real timing pass. |
| 10 | Reject short, missing, timed-out, malformed, noisy and over-threshold candidate evidence; guard report claims and the always-uploaded artifact route with mutations. | [Day 10 expected/observed failure ledger](artifacts/day10-weak-evidence-claim-boundary.md); 17 timing tests, separate report verifier, workflow mutation tests. | Cross-check the six rejection classes on Day 11, then collect/inspect a hosted candidate only after an accepted baseline exists. |
| 11 | Close the six-class negative matrix and inspect the first exact-head hosted candidate attempt. The job captured runner/compiler context, then failed at `S225_BASELINE_MISSING` before candidate collection. | [Day 11 matrix and hosted intake](artifacts/day11-negative-matrix-hosted-intake.md); [run 44403764960](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44403764960); artifact `14473492579`, digest-verified two-file ZIP. | No raw candidate or timing statistic exists. Recollect and approve a compatible three-job baseline before repeating hosted candidate trials. |
| 12 | Repeat the exact-commit timing job twice, retaining each artifact. All three attempts fail closed at missing-baseline preflight; attempt 2 resolves to a different image version and CPU. | [Day 12 three-attempt ledger](artifacts/day12-hosted-repetition-blocked.md); [retained attempt ZIPs](artifacts/day12-blocked-hosted-attempts/); run `44403764960`, attempts 1–3. | No candidate trials or noise statistics exist. Establish a stable runner class and accepted baseline before repeating candidate measurements. |
| 13 | Close the 225.5 planned effort with a blocked product outcome; calibrate benchmark and maintainer guidance to the retained evidence. | [Day 13 closeout](artifacts/day13-hosted-closeout-claim-calibration.md); [benchmark guide](../../../../benchmarks/README.md#sprint-225-one-cell-timing-decision); [maintainer guide](../../../../docs/maintainer_guide.md). | Obtain three compatible reviewed baseline jobs, then collect and verify real candidate trials. Day 14 must keep the timing claim blocked unless that evidence exists. |
| 14 | Audit the complete one-cell chain and close Sprint 225 as blocked, with no timing pass or widened claim. | [Final decision and evidence index](artifacts/day14-final-timing-decision.md); Day 5 rejected baseline ZIPs; Day 12 blocked candidate receipts; exact-source hosted run `44403764960`. | Carry the missing accepted baseline and hosted candidate result into Sprint 226's candidate/readiness decision. |

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
| Day 9 drift guards | Candidate receipt now binds the fixture, benchmark source, Makefile, executable, library and width stamp to the checkout at decision time. Synthetic mutation matrix passes; hosted execution remains blocked at baseline preflight. |
| Day 10 weak-evidence and claim guards | Synthetic single-sample, missing-warmup, incomplete, timeout, malformed-time, hash, dispersion, regression, report-claim and workflow-upload mutations fail with distinct diagnostics. Raw files remain available; real timing gate is still blocked. |
| Day 11 hosted candidate attempt | Exact source `38323f0e584e1a4d2aa8d2c92db5c18830366afc`; run `44403764960`, job `135153401892`, `macos26` image `20261102.0473.1`, Apple M1 (Virtual), Apple clang 21.0.0. Preflight blocked before collection; artifact `14473492579` contains host context and blocked report, no raw candidate. |
| Day 12 repeated hosted attempts | Same source SHA and run ID; attempts 1–3 each upload a digest-verified two-file blocked receipt. Attempts 1/3 use Apple M1 (Virtual) image `20261102.0473.1`; attempt 2 uses Apple M2 Pro (Virtual) image `20261112.0504.1`. No timing values exist. |
| Day 13 hosted closeout | Item 225.5's 20 planned hours are accounted for, but no successful timing job, candidate raw bundle, measured statistic or threshold verdict exists. Benchmark and maintainer docs state the blocked scope and rerun prerequisites. |
| Day 14 final review | Rechecked the `s225-v1` method, six raw ZIP digests, baseline member hashes and decimal dispersion, exact source/job status, three blocked candidate receipts, and independent threshold-free freshness boundary. Sprint closeout remains blocked. |
| First hosted baseline intake | Run `44395634285`, job `135124640857`, source `7e49ef3a96e00dcc815da3ba83b006934815c242`; 5 warmups + 20 measured, all valid, 54 ZIP files/53 verified member hashes. **Rejected as part of the incompatible Day 5 group.** |
| Repeated hosted baseline audit | Attempts 1, 3, and 4 each had 5 warmups + 20 valid measured trials and 53 valid member hashes. Their image/CPU fingerprints disagreed; attempt 3 MAD/median was `0.1111`, and cross-job spread was `0.12698` (both above `0.10`). **Rejected.** Attempt 2 has no retained ZIP and was excluded. |

## Planned effort ledger

The [plan](PLAN.md) allocates 10 engineering hours per day and 140 hours for the sprint; estimates are not actual tracked time. Days 1-2 address 20 hours of item 225.1. Day 3 allocates its final 4 hours to 225.1 (complete as a policy lock) and 6 hours to 225.2. Days 4-5 allocate 20 hours to 225.2; Day 6 allocates its final 2 hours to the documented rejection disposition, accounting for all 28 planned hours of 225.2. The **baseline deliverable remains blocked**, so the item is not marked achieved. Day 6 allocates 8 hours to item 225.3's gate design; Day 7 allocates 10 hours to the validator; Day 8 allocates its final 10 hours to separate CI wiring, accounting for all 28 planned hours. Days 9-10 allocate 20 of item 225.4's 22 hours to drift, weak-sample and claim regressions; Day 11 allocates its final 2 hours to the reviewed six-class matrix, closing the planned negative-case work. Days 11-13 account for all 20 planned hours of item 225.5 (8+10+2) through hosted attempts and blocker audits, but no candidate measurement was obtained. Days 13-14 account for all 18 planned hours of item 225.6 (8+10) through documentation and final review. Thus all 14 days and 140 estimated hours are accounted for; this is not a claim of actual time worked. Items 225.2 and 225.5 remain product-blocked. A hosted candidate measurement and accepted baseline are still required for a passing timing outcome.

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

## Day 9 verification

- Rehashed synthetic candidate and baseline mutations reject changed fixture, command/repeat, backend, build mode, runner/compiler, source/build identity, and missing/expired/superseded/mismatched baseline states with the expected rule code.
- The positive synthetic candidate passes with a matching checkout; changing checkout fixture, executable, library archive, benchmark source, Makefile, candidate commit, or width stamp rejects it before numerical comparison.
- Python compilation and all 14 timing-validator tests passed. No `.c` or `.h` edits; timing remains blocked on the real branch until an accepted baseline exists.

## Day 10 verification

- Six weak-sample mutations fail with the intended `S225_CANDIDATE_MISSING` or `S225_CANDIDATE_INVALID` rule; the validator leaves available raw streams unchanged.
- Exact-ceiling, above-ceiling and noisy synthetic bundles yield pass, regression and dispersion rules respectively, without deleting the bundle.
- Recomputed report verification rejects freshness-derived or overbroad claims and absent candidate evidence. Workflow mutations reject missing or success-only artifact upload and decision suppression.
- Python compilation, 17 timing tests, two workflow mutation tests, structured-workflow inventory/policy checks, API-docs local-only guard, and whitespace checks passed. No `.c` or `.h` edits; the actual hosted timing status remains blocked without an accepted baseline.

## Day 11 verification

- Reviewed all six required negative-case classes and reran 17 timing-validator methods, two timing-workflow mutation methods, eight structured-workflow inventory methods and 50 structured-policy methods; all passed.
- Pushed exact commit `38323f0e584e1a4d2aa8d2c92db5c18830366afc`. Hosted run `44403764960` recorded macOS runner/compiler identity, then failed preflight as required because the active baseline pointer is absent. Upload passed.
- Downloaded artifact `14473492579`, matched its ZIP SHA-256 to the GitHub digest, inspected the host record and blocked decision, and confirmed there are no candidate raw trials to recompute.
- No `.c` or `.h` edits. The real timing gate remains blocked; no numerical threshold was relaxed or claim promoted.

## Day 12 verification

- Reran workflow run `44403764960` twice on exact source `38323f0e584e1a4d2aa8d2c92db5c18830366afc`; attempts 2 and 3 independently failed `S225_BASELINE_MISSING` and uploaded host context plus blocked decision.
- Downloaded and SHA-256-verified all three attempt ZIPs before later reruns could replace the artifact listing; retained their bytes in the Day 12 artifact directory.
- Checked exact two-file membership, source/run/attempt provenance, schema-2 blocked reports and null timing values. Attempt 2 changed resolved image version and CPU; no candidate noise statistic can be calculated.
- Documentation and retained-binary-artifact changes only. No C, header or timing-policy edits; the gate remains blocked with reproducible evidence.

## Day 13 verification

- Inspected the Day 12 ZIP audit, exact source/job identities and the Day 5 rejected baseline decision. All three hosted receipts are blocked preflight records, with no raw candidate statistics to recompute.
- Updated benchmark and maintainer guidance with the locked one-cell method, observed host/compiler context, artifact and rerun route, and explicit threshold-free versus blocked timing distinction.
- Selected performance docs, 17 timing-validator tests, two workflow tests, API-docs local-only guard (122 records), relative links and `git diff --check` passed. The three receipt ZIP SHA-256 values still match the Day 12 ledger. Real preflight returned the expected exit 2 with `S225_BASELINE_MISSING`. No `.c` or `.h` edits; no timing pass is claimed.

## Day 14 verification

- Rehashed all three rejected baseline ZIPs and three blocked candidate receipt ZIPs. Baseline member hashes, 20 measured values per retained job and the Day 5 medians/MAD/p90 values recomputed consistently; the group still violates class and spread limits.
- Confirmed local `HEAD` equals hosted source `38323f0e584e1a4d2aa8d2c92db5c18830366afc`. GitHub reports the named timing job completed with failure. Subsequent worktree edits are documentation and retained evidence only; no code or workflow change was promoted without exact-source candidate evidence.
- Real preflight returned exit 2, schema-2 `blocked`, `S225_BASELINE_MISSING`, and null reference/candidate medians. Selected freshness remains a separate threshold-free result.
- Selected docs, 17 timing-validator tests, two timing-workflow tests, all selected freshness cases, and the API-docs guard (122 structured records) passed. Relative links, trailing whitespace and `git diff --check` passed. No `.c` or `.h` edits; C quality targets are not required for Day 14.
