# Sprint 214 Plan: ABI, Shared-Library And Release Readiness Decision

**Sprint Duration:** 14 days
**Goal:** Either implement one narrow shared-library/ABI proof path or close a
complete release/ABI design with acceptance gates and deferral guards.

**Time budget:** Each day is capped at 12 hours. The 14 days total `168`
hours, matching the Sprint 214 estimate in the Epic 19 project plan.

**Primary scope:** Decide the shared proof versus design/deferral path; define
symbol visibility, versioning, compatibility, loader, and platform metadata
policy; implement the selected build/install proof or stronger deferral guards;
specify and, where selected, run installed consumer evidence; and produce a
release readiness checklist with explicit evidence and stop gates.

**Non-goals:** Broad dynamic ABI compatibility, shared-library packages,
Windows/Linux/macOS shared parity, package-manager distribution, published
release artifacts, reproducible-release claims, and release readiness are not
assumed by this plan. Static-first install/export support remains the current
contract unless the selected proof explicitly earns a narrower additional
claim. A proof on one platform must not be described as support on others.

**Decision rule:** Day 2 records one of two paths. Select a narrow shared proof
only when its target platform, build/export surface, loader metadata, consumer
test, and CI evidence fit this sprint's acceptance gate. Otherwise complete
the ABI/release design and strengthen static-first deferral guards. If evidence
does not support either decision, stop for an explicit product decision rather
than silently enabling shared builds or claiming release readiness.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 214.1 ABI/Release Scope Decision | 24 h | Days 1-2: 12 + 12 |
| 214.2 ABI Policy Design | 30 h | Days 3-5: 12 + 12 + 6 |
| 214.3 Build/Install Proof Or Guard | 38 h | Days 5-8: 6 + 12 + 12 + 8 |
| 214.4 Downstream Consumer Evidence | 28 h | Days 8-10: 4 + 12 + 12 |
| 214.5 Release Checklist | 24 h | Days 11-12: 12 + 12 |
| 214.6 Validation And Closeout | 24 h | Days 13-14: 12 + 12 |
| **Total** | **168 h** | **14 days at 12 h each** |

---

## Day 1: ABI And Release Evidence Intake

**Title:** ABI And Release Evidence Intake
**Theme:** Establish the static-first baseline and the exact decision surface.
**Time estimate:** 12 hours

### Tasks

1. Re-read Sprint 214 items 214.1-214.6 and map each to a decision artifact,
   implementation surface, consumer evidence, and validation command.
2. Inventory the Sprint 170 shared-library/ABI decision, Sprint 207 package
   decision, current static install/export proofs, and active residual queue.
3. Inspect Make and CMake library targets, shared-build rejection, install
   metadata, public headers, package configuration, CI lanes, and existing
   static deferral guards without changing the maintained contract.
4. Create `WORKING_NOTES.md` with an evidence ledger, item checklist, risk
   register, changed-surface tracker, decision log, and validation matrix.
5. Write `artifacts/day1-abi-release-evidence-intake.md` with current claims
   and non-claims.

### Deliverables

- Baseline evidence and source-of-truth inventory.
- Sprint 214 working-notes scaffold and item traceability map.
- Day 1 intake artifact.

### Completion Criteria

- Every project-plan item has an evidence owner and planned validation path.
- Current static support is distinguished from deferred shared, ABI, loader,
  package, and release claims.
- Decision inputs and unknowns are recorded before implementation selection.

---

## Day 2: Scope Decision And Acceptance Gate

**Title:** Scope Decision And Acceptance Gate
**Theme:** Select a bounded proof or an explicit design/deferral closure.
**Time estimate:** 12 hours

### Tasks

1. Compare a single-platform narrow shared proof with design-only deferral
   against effort, CI availability, compatibility risk, and current static
   consumer behavior.
2. Define minimum proof gates: named platform/toolchain, explicit exported
   symbols, loader metadata, installed consumer execution, and negative checks
   against broader support claims.
3. Define deferral gates: exact future shared acceptance tests, current
   configure/install rejection, documentation non-claims, and regression
   coverage for unsupported selectors or metadata.
4. Record the selected path, rejected alternatives, stop conditions, and any
   product decision that needs user confirmation in
   `artifacts/day2-abi-release-scope-decision.md`.

### Deliverables

- ABI/release scope decision record for item 214.1.
- Selected-path acceptance matrix and non-claim list.
- Day 2 decision artifact.

### Completion Criteria

- Exactly one path is selected with evidence-based rationale, or unresolved
  product input is explicitly escalated before implementation.
- The selected path does not imply general shared-library or release support.
- The static-first package contract remains an explicit regression gate.

---

## Day 3: Symbol And Visibility Policy

**Title:** Symbol And Visibility Policy
**Theme:** Define what a dynamic consumer could link, not merely what builds.
**Time estimate:** 12 hours

### Tasks

1. Inventory public C API symbols, generated-version symbols, internal
   helpers, and any platform-specific export/import annotations.
2. Specify exported-symbol allowlist, hidden-by-default behavior, public
   header ownership, and how accidental exports would be detected.
3. Define whether the selected path proves symbols on one platform now or
   records exact future symbol-map/export-table acceptance tests.
4. Record failure cases for missing symbols, unintended exports, and static
   linkage changes in `artifacts/day3-symbol-visibility-policy.md`.

### Deliverables

- Symbol visibility and export/import design.
- Symbol inspection test plan or selected-platform proof requirements.
- Day 3 policy artifact.

### Completion Criteria

- Item 214.2 has an explicit public versus internal symbol boundary.
- Unsupported platforms and export mechanisms remain unclaimed.
- The policy is testable against installed artifacts rather than source names
  alone.

---

## Day 4: ABI Version And Compatibility Policy

**Title:** ABI Version And Compatibility Policy
**Theme:** Specify change classification and binary compatibility gates.
**Time estimate:** 12 hours

### Tasks

1. Define API version versus ABI version, compatibility promise level, and
   treatment of public structs, typedefs, calling conventions, and symbol
   removals or additions.
2. Specify version metadata and negative tests for mismatched or missing ABI
   identifiers without claiming compatibility across untested releases.
3. Define the minimum reference baseline and artifact retention needed for a
   future cross-version consumer or ABI-diff check.
4. Document breakage, rollback, and release-blocking criteria in
   `artifacts/day4-abi-compatibility-policy.md`.

### Deliverables

- ABI versioning and compatibility policy.
- Baseline and compatibility-test specification.
- Day 4 policy artifact.

### Completion Criteria

- The difference between a successful shared link and ABI compatibility is
  explicit.
- Future compatibility claims require a versioned baseline and consumer
  evidence, not only current-source builds.
- Public documentation cannot infer a broad ABI promise from the policy.

---

## Day 5: Loader Policy And Implementation Design

**Title:** Loader Policy And Implementation Design
**Theme:** Finish the ABI policy and design the selected build/install change.
**Time estimate:** 12 hours (214.2: 6 h; 214.3: 6 h)

### Tasks

1. Specify Linux SONAME and runtime search expectations, macOS install-name
   and `@rpath` expectations, and Windows DLL/import-library expectations as
   selected proof requirements or future gates, not automatic support claims.
2. Define how artifact names, dependency metadata, install locations, and
   consumer resolution will be inspected on the selected platform.
3. Design the narrow shared target and static coexistence checks if selected;
   otherwise design stronger configure, install, metadata, and docs guards.
4. Write `artifacts/day5-loader-and-build-design.md` with owned files,
   acceptance tests, rollback behavior, and platform exclusions.

### Deliverables

- Completed symbol/version/loader/platform policy for item 214.2.
- Implementation plan for the selected item 214.3 path.
- Day 5 design artifact.

### Completion Criteria

- All platform metadata expectations are explicit, including unproved ones.
- The build/install change has a bounded, testable acceptance gate.
- No shared selector or artifact is promoted merely because a linker can
  produce it.

---

## Day 6: Build Path Or Deferral Guard, First Pass

**Title:** Build Path Or Deferral Guard, First Pass
**Theme:** Implement the smallest selected build/configure behavior.
**Time estimate:** 12 hours

### Tasks

1. If shared proof was selected, add only the named-platform shared build path
   and explicit symbol/export control; keep the existing static path intact.
2. If deferral was selected, strengthen configure-time rejection of shared
   selectors and generated metadata that could imply a supported shared path.
3. Add positive and negative tests for the selected behavior, including a
   static build regression and unsupported-configuration failures.
4. Record commands, artifacts, and unresolved risks in
   `artifacts/day6-build-path-or-deferral-guard.md`.

### Deliverables

- Initial selected build or deferral-guard implementation.
- Focused build/configure regressions.
- Day 6 implementation artifact.

### Completion Criteria

- The selected path behaves as decided on Day 2.
- Unsupported shared builds cannot pass as maintained package support.
- Static-first build behavior has not regressed.

---

## Day 7: Install And Metadata Contract

**Title:** Install And Metadata Contract
**Theme:** Align installed files and package metadata with the selected path.
**Time estimate:** 12 hours

### Tasks

1. If shared proof was selected, install only the bounded shared artifact and
   validate its name, loader metadata, exported target, and coexistence with
   the existing static archive.
2. If deferral was selected, strengthen install/export checks against DLL,
   `.so`, `.dylib`, import-library, selector, or loader metadata claims.
3. Verify Make, CMake, `pkg-config`, and `find_package(Sparse)` descriptions
   remain truthful for the selected scope and do not drift from each other.
4. Write `artifacts/day7-install-and-metadata-contract.md` with artifact
   inventories and negative-test results.

### Deliverables

- Selected installed-artifact or deferral metadata checks.
- Static install/export regression evidence.
- Day 7 contract artifact.

### Completion Criteria

- Installed metadata matches the actual artifacts and supported consumer
  route.
- No broad platform or package-manager claim appears by implication.
- The maintained static consumer still installs and links.

---

## Day 8: Automation Gate And Consumer Design

**Title:** Automation Gate And Consumer Design
**Theme:** Complete implementation guards and define downstream proof.
**Time estimate:** 12 hours (214.3: 8 h; 214.4: 4 h)

### Tasks

1. Add a selected-platform CI lane and artifact inspection if shared proof was
   selected; otherwise add CI-enforced deferral-guard and docs-claim checks.
2. Test failure diagnostics for unsupported selectors, missing loader
   metadata, accidental shared artifacts, or incorrect install/export claims.
3. Define downstream consumer inputs, install prefix, runtime environment,
   link mode, and expected positive/negative outcomes for the selected path.
4. Write `artifacts/day8-automation-and-consumer-design.md` with a CI ownership
   table and consumer test specification.

### Deliverables

- Completed item 214.3 build/install proof or guard.
- CI and diagnostic regressions.
- Consumer acceptance design and Day 8 artifact.

### Completion Criteria

- The selected behavior is enforced in automation, not only a local command.
- Static and unsupported-path checks remain active.
- Item 214.4 has an exact, platform-scoped consumer test plan.

---

## Day 9: Installed Consumer Evidence

**Title:** Installed Consumer Evidence
**Theme:** Exercise the selected install surface from outside the source tree.
**Time estimate:** 12 hours

### Tasks

1. If shared proof was selected, build an out-of-tree consumer against the
   installed shared target on the named platform and execute it through the
   intended runtime-loader path.
2. If deferral was selected, specify executable future shared-consumer test
   fixtures and retain current installed static consumer execution evidence.
3. Include symbol-resolution, metadata, and failure diagnostics for a missing
   or wrongly named artifact, while avoiding untested cross-version claims.
4. Record commands, environment, and outputs in
   `artifacts/day9-installed-consumer-evidence.md`.

### Deliverables

- Installed consumer test or exact future-test specification.
- Static consumer preservation evidence.
- Day 9 consumer artifact.

### Completion Criteria

- Item 214.4 has an out-of-tree consumer evidence path.
- Any shared claim is limited to the executed platform and artifact.
- Deferral does not get described as installed shared proof.

---

## Day 10: Consumer Negative Matrix

**Title:** Consumer Negative Matrix
**Theme:** Prove that unsupported usage fails clearly and static usage remains.
**Time estimate:** 12 hours

### Tasks

1. Exercise version mismatch, wrong architecture/platform, absent runtime
   path, missing export, and unsupported shared selector as applicable to the
   selected path; record non-executable cases as future acceptance gates.
2. Confirm the installed static Make and CMake consumers still compile, link,
   and run with unchanged package metadata.
3. Validate CI test registration and failure messages for every implemented
   consumer check.
4. Write `artifacts/day10-consumer-negative-matrix.md` separating executed
   proof from specified future tests.

### Deliverables

- Consumer positive/negative evidence matrix for item 214.4.
- Static-install regression results.
- Day 10 matrix artifact.

### Completion Criteria

- Tested behavior and future acceptance tests are never conflated.
- No unsupported dynamic ABI, loader, or platform behavior is presented as
  supported.
- Consumer failures are detectable and actionable.

---

## Day 11: Release Checklist, Sources And Tags

**Title:** Release Checklist, Sources And Tags
**Theme:** Define release inputs and reproducibility evidence without shipping.
**Time estimate:** 12 hours

### Tasks

1. Draft a release checklist covering changelog entries, version metadata,
   tag policy, source archive contents, generated-file exclusions, and review
   signoff.
2. Define reproducible source archive requirements, comparison method,
   toolchain/environment capture, and failure criteria; do not label current
   archives reproducible without a passing comparison.
3. Specify how source, binary, and documentation artifacts would be named,
   retained, and checked, including paths deliberately not published.
4. Write `artifacts/day11-release-sources-and-tags.md` and update the
   checklist draft.

### Deliverables

- Changelog/tag/source-archive checklist for item 214.5.
- Reproducibility acceptance procedure.
- Day 11 release artifact.

### Completion Criteria

- Each release input has an owner, command or inspection, and stop condition.
- The checklist does not imply that a tag or artifact was published.
- Generated API HTML and other local-only output remain governed by their
  existing publication policy.

---

## Day 12: Provenance And CI Evidence Checklist

**Title:** Provenance And CI Evidence Checklist
**Theme:** Complete the release gate from artifact to consumer claim.
**Time estimate:** 12 hours

### Tasks

1. Add artifact checksums, provenance and source-to-binary traceability,
   package-origin statements, and platform ownership to the checklist.
2. Specify required CI evidence for static install, selected shared proof or
   deferral, consumer tests, docs checks, and full C/build quality gates.
3. Define release blockers, manual approvals, rollback actions, and the
   residual evidence needed before any release-readiness claim.
4. Reconcile README, INSTALL, maintainer guidance, and planning non-claims
   with the selected path; write
   `artifacts/day12-release-provenance-and-ci-gates.md`.

### Deliverables

- Complete release readiness checklist for item 214.5.
- Provenance and CI evidence matrix.
- Day 12 release-gates artifact.

### Completion Criteria

- The checklist covers changelog, tag, reproducible source archive, artifact,
  package provenance, and CI evidence explicitly.
- Release readiness remains a gated future claim unless all listed evidence
  exists and is reviewed.
- Public and maintainer wording matches the selected ABI/shared path.

---

## Day 13: Integrated Build And Consumer Validation

**Title:** Integrated Build And Consumer Validation
**Theme:** Run the selected path with static, documentation, and quality gates.
**Time estimate:** 12 hours

### Tasks

1. Run installed static Make and CMake consumer tests, the static/shared
   deferral guard, and selected shared consumer checks or exact deferral tests.
2. Run affected documentation and claim guards, plus targeted negative tests
   for metadata, configure selectors, and platform exclusions.
3. If any `*.c` or `*.h` files changed, run `make format && make lint && make
   test`; also run the reviewed build/CMake quality path appropriate to the
   changed build surface.
4. Record commands, outcomes, failures, reruns, and skipped checks in
   `artifacts/day13-integrated-validation.md`.

### Deliverables

- Integrated validation transcript for item 214.6.
- Failure-resolution and changed-surface record.
- Day 13 validation artifact.

### Completion Criteria

- All required checks for the selected path pass before closeout.
- Skipped checks have an explicit reason tied to the changed surface.
- Static-first regression and selected-path evidence agree with documentation.

---

## Day 14: Decision Closeout And Residual Queue

**Title:** Decision Closeout And Residual Queue
**Theme:** Confirm the exact earned claim and record what remains deferred.
**Time estimate:** 12 hours

### Tasks

1. Audit items 214.1-214.6 against the decision, code or guard changes,
   consumer evidence, release checklist, and validation transcript.
2. Recheck affected build/install, consumer, docs, and quality gates after
   final edits; resolve failures before claiming completion.
3. Update Epic 19 status and residual queue with exact supported scope,
   unproved platforms, future ABI tests, and release-readiness blockers.
4. Write `artifacts/day14-closeout-review.md` with final item status,
   decisions, evidence links, and ready-to-retrospective notes.

### Deliverables

- Final Sprint 214 item and claim-status table.
- Updated residual queue and closeout artifact.
- Evidence-backed handoff for the sprint retrospective.

### Completion Criteria

- A narrow shared proof is claimed only if its named-platform build, loader,
  install, and executed consumer gates pass; otherwise design/deferral closure
  is documented with enforceable guards.
- The release checklist is complete without asserting an unearned release or
  ABI compatibility guarantee.
- All required validation is green and the branch is ready for retrospective,
  commit, and PR creation.
