# Sprint 225 Plan: One Controlled Canonical Timing Gate

**Sprint Duration:** 14 days
**Goal:** Establish one reviewed, runner-specific timing regression gate for the canonical `bench_refactor_csc`/`nos4` workload, with retained raw samples and a decision independent of threshold-free freshness.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds 12 hours; Epic 20 permits at most 168 hours per sprint. CI queue and wall-clock time are separate from engineering estimates.

**Prerequisites:** Sprint 218's [measurement lessons](../SPRINT_218/RETROSPECTIVE.md), Sprint 212's [threshold-free selected-row decision](../../EPIC_19/SPRINT_212/RETROSPECTIVE.md), a stable named hosted runner/compiler class, and an artifact route that retains raw samples. The selected row is `SRT-BENCH-REFACTOR-CSC-NOS4`; the canonical executable is `bench_refactor_csc` with `tests/data/suitesparse/nos4.mtx --repeat 1`.

**Evidence rule:** Freeze the workload and acceptance policy before collecting baseline data. A local timing, a single hosted sample, a changed runner class, or a freshness pass cannot establish a timing threshold. Record every candidate trial, including invalid or failed trials, and bind the baseline and candidate evidence to source revision, compiler, runner class, workload identity, commands, and artifact hashes. If the runner or artifact route is unavailable, leave the timing gate unpromoted and record the blocker.

**Claim boundary:** A pass means only that this canonical workload met the reviewed threshold and dispersion policy on the named runner/compiler class. It does not establish portable speed, Linux/macOS or Windows timing parity, broad solver performance, superiority over external libraries, or state-of-the-art status. The existing selected freshness check remains threshold-free and reports a separate result.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 225.1 Methodology Lock | 24 h | Days 1-3: 10 + 10 + 4 |
| 225.2 Baseline Collection | 28 h | Days 3-6: 6 + 10 + 10 + 2 |
| 225.3 Timing Gate | 28 h | Days 6-8: 8 + 10 + 10 |
| 225.4 Negative Cases | 22 h | Days 9-11: 10 + 10 + 2 |
| 225.5 Hosted Repetition | 20 h | Days 11-13: 8 + 10 + 2 |
| 225.6 Claim Calibration | 18 h | Days 13-14: 8 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Inventory the Canonical Timing Cell

**Title:** Inventory the Canonical Timing Cell

**Theme:** Identify the precise benchmark, selected row, and current evidence boundary.

**Time estimate:** 10 hours (225.1: 10 h)

### Tasks

1. Read the Sprint 212 selection and Sprint 218 measurement results; map the current `bench_refactor_csc` command, `nos4` fixture, selected manifest row, report generator, CI jobs, and existing freshness checks.
2. Identify the benchmark's timing fields, clock source, internal repeat behavior, setup costs, and current raw-output format; distinguish process trials from `--repeat 1` inside one invocation.
3. Record the hosted runner, compiler, build mode, dependencies, and artifact retention options that can be held stable; note any unresolved ownership or availability issue.
4. Start `WORKING_NOTES.md` with the selected scope, source revision, evidence ledger, decisions, and explicit timing-gate blocker state.

### Deliverables

- Canonical-cell inventory and current-data-flow map.
- Initial runner/artifact feasibility and blocker ledger.

### Completion Criteria

- The selected workload and current threshold-free route are identified without treating existing freshness as timing evidence.
- Unverified runner or sample-retention assumptions are marked open.

---

## Day 2: Pre-Register the Timing Method

**Title:** Pre-Register the Timing Method

**Theme:** Specify a reviewable experiment before measuring it.

**Time estimate:** 10 hours (225.1: 10 h)

### Tasks

1. Freeze the fixture bytes, backend, command, build configuration, runner image/class, compiler family/version policy, clock, and process isolation rules.
2. Specify warmup and measured-trial counts, order, timeouts, invalid-trial handling, baseline age/version policy, and how raw stdout, stderr, exit status, and elapsed samples are retained.
3. Define the baseline and candidate statistics, dispersion measure, threshold comparison, rounding, and failure policy, including what happens when variance is too high.
4. Review the proposed policy against Sprint 212's rejected threshold assumptions and document why each chosen limit is appropriate for this one runner and workload.

### Deliverables

- Draft preregistered methodology and acceptance-policy document.
- Raw-sample and provenance schema proposal.

### Completion Criteria

- A reviewer can reproduce the proposed trial sequence and calculate the decision from the written policy.
- No threshold is chosen after seeing candidate results.

---

## Day 3: Lock the Method and Prepare Collection

**Title:** Lock the Method and Prepare Collection

**Theme:** Complete methodology review and make hosted baseline capture possible.

**Time estimate:** 10 hours (225.1: 4 h; 225.2: 6 h)

### Tasks

1. Resolve methodology review comments and version the frozen workload, statistic, dispersion, threshold, and invalid-trial rules; close item 225.1.
2. Prepare a baseline collector that records each warmup and measured invocation separately, including command, result, time, and raw output without silent trial deletion.
3. Bind the collector to exact fixture digest, source commit, binary/build identity, runner class, compiler, workflow run, and timestamp.
4. Verify the proposed artifact layout can be downloaded and independently checked; capture a dry run as tooling validation only.

### Deliverables

- Versioned methodology lock and collection schema.
- Baseline collector candidate and artifact-layout proof.

### Completion Criteria

- Item 225.1's 24 hours are accounted for before baseline acceptance.
- A dry run cannot be mistaken for an accepted baseline.

---

## Day 4: Collect Pinned Baseline Trials

**Title:** Collect Pinned Baseline Trials

**Theme:** Gather repeated samples on the named hosted class.

**Time estimate:** 10 hours (225.2: 10 h)

### Tasks

1. Run the frozen collector on the named hosted runner/compiler configuration and record run, job, commit, fixture, build, and artifact identifiers.
2. Download and inspect the retained baseline artifact; enumerate all warmups, valid measured trials, invalid trials, raw logs, and sample hashes.
3. Check that every accepted timing came from the frozen command and that no failed or timed-out invocation was silently omitted.
4. If collection is unavailable or contradicts the lock, retain the attempted-run evidence and correct the collector without promoting a baseline.

### Deliverables

- Pinned baseline trial bundle or precise collection failure record.
- Run/job/artifact and raw-sample inventory.

### Completion Criteria

- Accepted samples, if any, are traceable to one named runner class and the locked workload.
- Invalid trials remain visible in the artifact and decision record.

---

## Day 5: Analyze and Version the Baseline

**Title:** Analyze and Version the Baseline

**Theme:** Determine whether repeated baseline data are stable enough to gate.

**Time estimate:** 10 hours (225.2: 10 h)

### Tasks

1. Recompute the frozen statistic and dispersion measure from the downloaded raw samples, including warmup exclusion and invalid-trial accounting.
2. Compare repeated baseline runs for noise, runner drift, compiler drift, and workload drift using only the preregistered policy.
3. Version the accepted baseline with source, fixture, runner/compiler, sample hashes, statistic, and validity window; reject a noisy or incomplete baseline.
4. Document any proposed policy change as a new reviewed methodology version requiring fresh baseline collection.

### Deliverables

- Baseline calculation worksheet and versioned candidate baseline.
- Variance and provenance review, including rejected runs.

### Completion Criteria

- A second reader can reproduce each baseline value from retained raw samples.
- No unstable or nonmatching sample becomes a timing-gate baseline.

---

## Day 6: Close Baseline Acceptance and Design the Gate

**Title:** Close Baseline Acceptance and Design the Gate

**Theme:** Freeze the accepted baseline and its independent candidate decision.

**Time estimate:** 10 hours (225.2: 2 h; 225.3: 8 h)

### Tasks

1. Complete the baseline acceptance ledger and link its version, hashes, variance and provenance; close item 225.2 or explicitly block gate promotion.
2. Define the candidate input contract, exact baseline selection, runner/workload identity comparisons, sample-count requirements, and diagnostic rule names.
3. Design threshold and dispersion evaluation as a separate CI result from selected benchmark freshness, with no success fallback when inputs are missing.
4. Specify artifact retention for every candidate trial and a machine-readable decision report that cites baseline and candidate hashes.

### Deliverables

- Accepted baseline decision or explicit blocker.
- Gate input/output contract and CI wiring design.

### Completion Criteria

- Item 225.2's 28-hour evidence is reviewable.
- The gate design cannot turn a freshness result into a timing pass.

---

## Day 7: Implement the Timing Decision

**Title:** Implement the Timing Decision

**Theme:** Compare candidate trials with the locked baseline and policy.

**Time estimate:** 10 hours (225.3: 10 h)

### Tasks

1. Implement baseline schema/version and provenance validation before reading any timing statistic.
2. Parse all candidate trials, reject missing or malformed samples, and compute the preregistered statistic and dispersion without dropping outliers ad hoc.
3. Apply the reviewed threshold and dispersion rules; emit explicit pass/fail diagnostics and a report carrying sample counts, hashes, runner identity, and policy version.
4. Exercise valid, below-threshold and over-threshold fixtures with independently calculated expected decisions.

### Deliverables

- Timing decision implementation and initial positive/negative fixtures.
- Machine-readable decision report with raw-sample references.

### Completion Criteria

- The same baseline and candidate bundle always yield the same decision.
- Missing identity or invalid timing evidence fails closed.

---

## Day 8: Wire One Enforced CI Timing Gate

**Title:** Wire One Enforced CI Timing Gate

**Theme:** Run the canonical candidate and retain its complete evidence.

**Time estimate:** 10 hours (225.3: 10 h)

### Tasks

1. Add the named hosted CI job for the locked `bench_refactor_csc`/`nos4` cell, using the reviewed runner/compiler and baseline version.
2. Capture every warmup, measured, failed and timed-out candidate invocation, upload raw files and the decision report, and make decision failure fail the job.
3. Keep existing threshold-free freshness workflow and report fields separate; check that timing-job absence cannot be shown as a timing pass.
4. Run local fixture and workflow checks, then document exact commands, artifact names and rerun procedure.

### Deliverables

- One enforced candidate timing CI path and raw-artifact route.
- Workflow/decision separation proof and operator notes.

### Completion Criteria

- Item 225.3's 28 hours are accounted for.
- CI produces a separate timing status with retained candidate samples or a clear failure.

---

## Day 9: Reject Workload and Baseline Drift

**Title:** Reject Workload and Baseline Drift

**Theme:** Protect the gate from noncanonical inputs.

**Time estimate:** 10 hours (225.4: 10 h)

### Tasks

1. Add negative fixtures for changed fixture bytes, benchmark executable, backend, command, repeat count, build mode and runner/compiler identity.
2. Add missing, expired, superseded, malformed, or mismatched baseline-version cases.
3. Check that a candidate from an unknown host or different source/build identity is rejected with the intended diagnostic.
4. Run the negative suite and a canonical positive fixture; retain each observed verdict in the evidence ledger.

### Deliverables

- Workload, host and baseline drift regressions.
- Expected-versus-observed diagnostic table.

### Completion Criteria

- Each noncanonical input fails for its intended reason.
- The valid canonical bundle still passes schema and identity checks.

---

## Day 10: Reject Weak or Overclaimed Timing Evidence

**Title:** Reject Weak or Overclaimed Timing Evidence

**Theme:** Enforce sample adequacy, dispersion and claim boundaries.

**Time estimate:** 10 hours (225.4: 10 h)

### Tasks

1. Add single-sample, missing-warmup, incomplete-trial, timeout, malformed-time and altered-hash fixtures.
2. Add high-dispersion and threshold-overrun fixtures with independently computed expected results.
3. Add workflow/report mutation checks that reject a timing pass inferred from freshness, a missing hosted artifact, or a portable-speed/parity claim.
4. Verify all failures leave raw candidate samples available for inspection and report their distinct rule codes.

### Deliverables

- Sample, variance, threshold and claim-rejection regressions.
- Complete failure-fixture result ledger.

### Completion Criteria

- Sparse, noisy or over-threshold evidence cannot pass the timing gate.
- No failure path silently discards candidate data.

---

## Day 11: Finish Negative Cases and Start Hosted Runs

**Title:** Finish Negative Cases and Start Hosted Runs

**Theme:** Close adversarial coverage and collect the first hosted candidate.

**Time estimate:** 10 hours (225.4: 2 h; 225.5: 8 h)

### Tasks

1. Review the negative-case matrix for the six required rejection classes; close item 225.4 after the positive and negative suites pass.
2. Trigger or inspect the first candidate job on the exact branch commit and record run, job, source SHA, runner image, compiler and artifact IDs.
3. Download the raw candidate bundle and recompute its statistic, dispersion and pass/fail result independently.
4. Record any infrastructure or policy failure without weakening the threshold after seeing the candidate result.

### Deliverables

- Closed negative-case matrix and first hosted candidate ledger.
- Independent calculation or precise hosted blocker.

### Completion Criteria

- Item 225.4's 22 hours are complete.
- The first hosted result is tied to inspectable raw data and the exact tested commit.

---

## Day 12: Repeat Hosted Candidate Trials

**Title:** Repeat Hosted Candidate Trials

**Theme:** Test the decision under normal hosted-run variation.

**Time estimate:** 10 hours (225.5: 10 h)

### Tasks

1. Execute multiple independent hosted runs on the named runner class with the same candidate commit, policy version and workload.
2. Download and hash each artifact; compare every run's raw trials, calculated statistic, dispersion, decision and diagnostics.
3. Investigate noise, flaky verdicts, runner-image changes or clock anomalies; retain all runs rather than selecting favorable ones.
4. If a policy adjustment is needed, document a new review decision and recollect a compatible baseline before declaring the gate enforced.

### Deliverables

- Multi-run hosted sample and verdict ledger.
- Noise assessment with accepted or blocked gate decision.

### Completion Criteria

- Multiple complete hosted runs support the same reviewed rule, or the gate remains blocked with reproducible evidence.
- No threshold is revised using only a favorable candidate run.

---

## Day 13: Close Hosted Evidence and Calibrate Claims

**Title:** Close Hosted Evidence and Calibrate Claims

**Theme:** Tie the final gate to its run evidence and narrow public wording.

**Time estimate:** 10 hours (225.5: 2 h; 225.6: 8 h)

### Tasks

1. Verify the latest candidate commit has a successful named job, inspectable raw artifacts, matching hashes and consistent repeated decisions; close item 225.5 or record the blocker.
2. Update benchmark and maintainer docs with the exact runner, compiler, workload, baseline version, sample and dispersion policy, threshold, artifact route, and rerun instructions.
3. State separately the threshold-free freshness result and the timing-gate result; preserve non-claims for portable speed, platform parity and external-library rankings.
4. Run affected focused fixtures, docs checks, and `git diff --check`; if any `.c` or `.h` files changed, run `make format && make lint && make test`.

### Deliverables

- Hosted-evidence closeout and calibrated documentation candidate.
- Quality-check transcript and residual-risk ledger.

### Completion Criteria

- Item 225.5's 20 hours are accounted for.
- Documentation makes no claim wider than the retained one-runner evidence.

---

## Day 14: Review the One-Cell Timing Decision

**Title:** Review the One-Cell Timing Decision

**Theme:** Close the sprint with an auditable pass or an explicit blocked state.

**Time estimate:** 10 hours (225.6: 10 h)

### Tasks

1. Audit the methodology lock, baseline version, raw baseline and candidate samples, hashes, hosted job, threshold computation, dispersion and negative-case outcomes as one chain.
2. Recheck the proposed final source revision; any post-evidence source edit requires new exact-commit hosted candidate evidence.
3. Confirm the timing gate is enforced only for `bench_refactor_csc`/`nos4` on the named runner and that selected freshness remains an independent threshold-free status.
4. Finalize `WORKING_NOTES.md`, the 140-hour scope ledger, evidence links, pass/blocked decision, residual queue and Sprint 226 handoff.

### Deliverables

- Final one-cell timing-gate decision and reviewer-visible evidence index.
- Claim/non-claim review, quality-check results and next-sprint handoff.

### Completion Criteria

- The gate yields reproducible runner/workload-specific pass/fail evidence with retained raw data, or remains explicitly blocked without an unsupported claim.
- Item 225.6's 18 hours and all 14 days are accounted for; no portable speed or platform parity claim is inferred.
