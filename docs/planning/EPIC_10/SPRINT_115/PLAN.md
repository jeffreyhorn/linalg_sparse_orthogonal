# Sprint 115 Plan: Residual Package Platform Parity & ABI Productization Decision

**Sprint Duration:** 14 days
**Goal:** Resolve Sprint 112's package/platform deferred debt in dependency
order by deciding which local install proofs should become reviewed lanes,
which Windows and macOS exclusions can move toward parity, and which broader
ABI/package-manager product claims remain future contracts.

**Starting Point:** Sprint 115 begins from:
- Sprint 112 static-first package support decision, local Make install proof,
  CMake install/export proof, platform-tier contract, and Windows/macOS
  staged-exclusion decisions
- Sprint 113 behavior/proof-owner closeout evidence
- Sprint 114 residual proof-owner follow-through and residual deferral
  decision
- existing package, CI, install, platform, and adoption documentation
  boundaries

The sprint must:
- consume only package/platform-facing residual debt from Sprint 114
- avoid inheriting eigensolver source movement, broad direct/iterative oracle,
  or broad SVD abstraction work
- decide whether local Linux install proofs should become reviewed CI lanes
- decide whether macOS CMake install/export parity can be reviewed or must
  remain deferred
- decide whether Windows install-validation can be reviewed or must remain
  deferred
- audit Windows thread/fuzz portability without overstating Windows parity
- review macOS backend/toolchain coverage, Homebrew GCC, and TSan claims
- decide shared-library/dynamic ABI and package-manager product contracts
- close with validation and package/platform handoff evidence

**End State:** Sprint 115 leaves behind:
- a duplicate-fenced package/platform residual intake artifact
- explicit Linux install CI promotion or no-promotion decision
- explicit macOS CMake install/export parity proof or deferral
- explicit Windows install-validation proof or deferral
- Windows thread/fuzz portability decision
- macOS backend/toolchain follow-through artifact
- shared-library/dynamic ABI product-contract decision
- package-manager support decision
- Sprint 114 package/platform residual intake and deferral-boundary artifact
- validation and package/platform handoff artifact

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 115 project-plan estimate.

---

## Day 1: Residual Package and Platform Intake

**Title:** Residual Intake
**Theme:** Establish Sprint 115 package/platform boundaries and duplicate fence
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 115 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 112 package/platform artifacts and retrospective residuals.
3. Re-read Sprint 114 residual deferral decision and package/platform-facing
   residuals.
4. Explicitly exclude non-package source-boundary residuals from Sprint 115.
5. Inventory install, CI, platform, package, ABI, and adoption documents that
   can be affected by Sprint 115.
6. Create Sprint 115 working notes and artifact directory.
7. Write the residual intake and duplicate-work exclusion artifact.

### Deliverables
- Sprint 115 working-notes baseline
- artifact directory
- residual package/platform intake artifact
- duplicate-work exclusion list
- dependency-ordered implementation map

### Completion Criteria
- completed Sprint 112 package work is not reintroduced as unresolved debt
- Sprint 114 non-package residuals are explicitly deferred out of Sprint 115
- all Sprint 115 project-plan items have a day-level owner

---

## Day 2: Linux Install Proof CI Promotion Design

**Title:** Linux Install Design
**Theme:** Decide what evidence is required before promoting local install proof
**Time estimate:** 12 hours

### Tasks
1. Inspect `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
   existing CI lanes.
2. Identify the exact local proof already provided by Make install and CMake
   install/export tests.
3. Define risks of adding reviewed Linux CI install lanes, including runtime,
   cache, artifact, and environment stability.
4. Define no-promotion criteria if the proof should remain local.
5. Draft the Linux install proof CI promotion decision criteria.
6. Write the Day 2 design artifact.

### Deliverables
- Linux install proof inventory
- reviewed-lane promotion criteria
- local-only no-promotion criteria
- CI surface risk list
- Day 2 design artifact

### Completion Criteria
- Day 3 can make an evidence-backed promotion/no-promotion decision
- no CI or package claim is changed before the decision
- install proof boundaries remain explicit

---

## Day 3: Linux Install Proof CI Promotion Decision

**Title:** Linux Install Decision
**Theme:** Promote a narrow Linux install lane or publish no-promotion contract
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 2 criteria to the current scripts and CI surface.
2. If safe, add the narrow reviewed Linux install lane with explicit ownership
   and expected evidence.
3. If not safe, publish a no-promotion contract preserving local-only proof.
4. Update any affected docs or CI comments so support wording stays accurate.
5. Run focused validation for touched scripts, docs, or workflow files.
6. Write the Day 3 decision artifact.

### Deliverables
- Linux install CI promotion or no-promotion artifact
- optional reviewed CI lane update
- support-wording drift notes
- focused validation evidence

### Completion Criteria
- Linux install proof status is unambiguous
- no unsupported Linux package/install claim is introduced
- validation matches touched files

---

## Day 4: macOS CMake Install and Export Parity Design

**Title:** macOS Install Design
**Theme:** Design reviewed macOS install/export parity proof or deferral
**Time estimate:** 12 hours

### Tasks
1. Inspect existing macOS CI and local CMake install/export proof.
2. Identify what a reviewed macOS CMake install/export lane would need to
   prove.
3. Check interaction with compiler, SDK, Homebrew, cache, and install prefix
   assumptions.
4. Define deferral criteria if reviewed proof is too broad for Sprint 115.
5. Define package/support claims that must remain fenced until proof lands.
6. Write the Day 4 design artifact.

### Deliverables
- macOS install/export proof design
- reviewed-lane requirements
- deferral criteria
- support-claim fence list
- Day 4 design artifact

### Completion Criteria
- Day 5 can implement or defer without rediscovering scope
- macOS parity claims remain evidence-bounded
- no Windows or Linux work is mixed into the macOS decision

---

## Day 5: macOS CMake Install and Export Follow-Through

**Title:** macOS Install Decision
**Theme:** Add reviewed macOS proof or publish explicit deferral
**Time estimate:** 12 hours

### Tasks
1. Apply Day 4 criteria to the current macOS workflow surface.
2. If safe, add or adjust the reviewed macOS install/export lane.
3. If not safe, publish a deferral artifact with exact missing proof.
4. Update package/platform support wording if needed.
5. Run focused validation for touched docs, scripts, or workflow files.
6. Write the Day 5 follow-through artifact.

### Deliverables
- macOS install/export proof or deferral artifact
- optional reviewed CI update
- support wording update notes
- focused validation evidence

### Completion Criteria
- macOS CMake install/export parity is either reviewed or explicitly deferred
- no full macOS install/export claim is made without reviewed proof
- validation evidence is captured

---

## Day 6: Windows Install-Validation Lane Design

**Title:** Windows Install Design
**Theme:** Design MSVC install-validation proof or deferral
**Time estimate:** 12 hours

### Tasks
1. Inspect existing Windows CMake reviewed consumer lanes.
2. Define the required Windows install-validation sequence:
   `cmake --install`, installed target lookup, downstream configure,
   compile, link, and run.
3. Identify reviewed CTest count and staged-exclusion interactions.
4. Identify MSVC, generator, install prefix, and artifact risks.
5. Define deferral criteria if the lane is too broad for Sprint 115.
6. Write the Day 6 design artifact.

### Deliverables
- Windows install-validation design
- MSVC downstream consumer proof requirements
- reviewed-count impact assessment
- deferral criteria
- Day 6 design artifact

### Completion Criteria
- Day 7 can add or defer Windows install validation
- Windows installed-package support remains unclaimed until proof exists
- reviewed-count obligations are explicit

---

## Day 7: Windows Install-Validation Follow-Through

**Title:** Windows Install Decision
**Theme:** Add or explicitly defer reviewed Windows install validation
**Time estimate:** 12 hours

### Tasks
1. Apply Day 6 criteria to the current Windows workflow surface.
2. If safe, add a reviewed Windows install-validation lane.
3. If not safe, publish a deferral artifact with exact missing proof and
   support wording.
4. Update documentation or workflow comments to prevent unsupported installed
   package claims.
5. Run focused validation for touched files.
6. Write the Day 7 follow-through artifact.

### Deliverables
- Windows install-validation proof or deferral artifact
- optional reviewed Windows CI update
- reviewed CTest/count notes
- focused validation evidence

### Completion Criteria
- Windows installed-package support status is explicit
- no unsupported Windows install claim is introduced
- touched-file validation is complete

---

## Day 8: Windows Thread and Fuzz Portability Audit

**Title:** Windows Portability Audit
**Theme:** Audit Windows staged exclusions for thread and fuzz/property coverage
**Time estimate:** 12 hours

### Tasks
1. Inspect `test_threads`, `test_sprint4_integration`, and `test_fuzz`
   Windows exclusion history and current implementation.
2. Identify platform-specific assumptions, APIs, timeouts, and filesystem
   behavior.
3. Choose at most one bounded Windows-native proof owner if low risk.
4. Define staged-exclusion contracts for remaining thread/fuzz/property gaps.
5. Document reviewed-count and support-wording impact.
6. Write the Day 8 audit artifact.

### Deliverables
- Windows thread/fuzz portability audit
- bounded proof-owner candidate or no-candidate decision
- staged-exclusion contracts
- reviewed-count impact notes

### Completion Criteria
- Windows thread/fuzz/property status is explicit
- no full Windows parity claim is made from partial proof
- Day 9 can implement or defer one bounded item cleanly

---

## Day 9: Windows Thread and Fuzz Portability Follow-Through

**Title:** Windows Portability Decision
**Theme:** Implement one bounded Windows proof or publish staged exclusions
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 8 audit decision.
2. If one bounded proof owner is safe, add the narrow Windows-native proof.
3. If not safe, publish staged-exclusion contracts for thread and fuzz/property
   lanes.
4. Update workflow comments, docs, or support wording as needed.
5. Run focused validation for touched code, workflows, scripts, or docs.
6. Write the Day 9 follow-through artifact.

### Deliverables
- Windows thread/fuzz proof or staged-exclusion artifact
- optional bounded test/workflow update
- support wording update notes
- focused validation evidence

### Completion Criteria
- thread/fuzz Windows status is documented and non-ambiguous
- any reviewed test count change is deliberate and documented
- validation matches touched surfaces

---

## Day 10: macOS Backend and Toolchain Follow-Through

**Title:** macOS Toolchain
**Theme:** Review macOS coverage backend, Homebrew GCC, and TSan feasibility
**Time estimate:** 12 hours

### Tasks
1. Inspect macOS workflow jobs and documented backend/toolchain assumptions.
2. Review coverage backend stability and whether any macOS coverage claim
   needs fencing.
3. Review Homebrew GCC version assumptions and package/toolchain wording.
4. Review macOS TSan feasibility and whether it should remain future work.
5. Promote only evidence-backed lanes; otherwise publish non-claims.
6. Write the Day 10 follow-through artifact.

### Deliverables
- macOS backend/toolchain follow-through artifact
- coverage backend decision
- Homebrew GCC assumption notes
- TSan feasibility decision
- validation notes for touched files

### Completion Criteria
- macOS backend/toolchain claims are evidence-bounded
- no unstable toolchain lane is promoted without proof
- package/platform handoff has current macOS truth

---

## Day 11: Shared-Library and Dynamic ABI Product Contract Decision

**Title:** ABI Contract
**Theme:** Decide whether dynamic ABI support belongs in Epic 10 or future work
**Time estimate:** 12 hours

### Tasks
1. Inspect current static-first build/package support and install/export
   behavior.
2. Define what shared-library/dynamic ABI support would require:
   build rules, package metadata, runtime-loader proof, symbol policy,
   versioning policy, and platform ownership.
3. Decide whether Sprint 115 should add proof or publish a future-work
   product contract.
4. Update support wording if any current docs imply unsupported dynamic ABI
   support.
5. Run documentation/build metadata validation for touched files.
6. Write the Day 11 ABI decision artifact.

### Deliverables
- shared-library/dynamic ABI product-contract decision
- required future proof checklist
- support wording drift notes
- validation evidence

### Completion Criteria
- dynamic ABI support status is explicit
- no shared-library product claim is introduced without proof
- future work has concrete acceptance criteria

---

## Day 12: Package-Manager Support Decision

**Title:** Package Managers
**Theme:** Decide future or bounded proof plan for package-manager support
**Time estimate:** 12 hours

### Tasks
1. Inventory any existing Homebrew, vcpkg, distro, Windows package-manager, or
   install-consumer references.
2. Define recipe and install/consumer proof requirements for each candidate.
3. Decide whether package-manager support remains future work or gets a
   bounded proof plan.
4. Fence documentation so package-manager support is not implied without
   recipes and reviewed proof.
5. Run documentation hygiene for touched files.
6. Write the Day 12 package-manager decision artifact.

### Deliverables
- package-manager support decision
- candidate recipe/proof checklist
- support wording non-claims
- validation evidence

### Completion Criteria
- package-manager support status is explicit
- no recipe/support claim is made without proof
- Sprint 116 adoption QA has accurate package truth

---

## Day 13: Sprint 114 Package/Platform Residual Intake

**Title:** Sprint 114 Residual
**Theme:** Consume only Sprint 114 package/platform-facing residual debt
**Time estimate:** 12 hours

### Tasks
1. Re-read Sprint 114 retrospective residual deferred debt and the Epic 10
   deferral decision.
2. Verify package, ABI, Windows, CMake parity, install-header, and adoption
   claims remain fenced unless Sprint 115 added reviewed evidence.
3. Confirm eigensolver source movement, broad direct/iterative oracle, and
   broad SVD abstraction work remain outside Sprint 115.
4. Update Sprint 115 working notes and handoff lists with the final residual
   routing.
5. Run documentation hygiene for touched docs.
6. Write the Sprint 114 package/platform residual intake artifact.

### Deliverables
- Sprint 114 package/platform residual intake artifact
- deferral-boundary notes
- claim-fence checklist
- adoption and Epic closeout handoff notes

### Completion Criteria
- Sprint 114 residual routing is documented and not forgotten
- Sprint 115 does not absorb non-package source-boundary debt
- Sprint 116 and Sprint 117 handoffs are explicit

---

## Day 14: Validation and Package/Platform Handoff

**Title:** Validation Handoff
**Theme:** Validate touched surfaces and publish final package/platform truth
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 115 artifacts, working notes, and touched files.
2. Run required checks for touched CI, build, scripts, docs, or code.
3. Verify unsupported package/platform claims were not introduced.
4. Capture final package/platform metrics and reviewed-lane decisions.
5. Publish final handoff to Sprint 116 adoption QA and Sprint 117 Epic
   closeout.
6. Write the Day 14 validation and package/platform handoff artifact.

### Deliverables
- final validation evidence
- package/platform decision matrix
- unsupported-claim checklist
- Sprint 116 adoption QA handoff
- Sprint 117 closeout handoff

### Completion Criteria
- required quality checks pass for all touched surfaces
- package/platform truth is explicit and adoption-ready
- Sprint 115 closes without unsupported Linux, macOS, Windows, ABI, or
  package-manager claims
