# Sprint 194 Plan: Adoption and API Coherence Simplification

**Sprint Duration:** 14 days
**Goal:** Make the project easier to adopt by simplifying user-facing workflow
guidance and consolidating support/readiness truth.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 194 estimate in the Epic 17
project plan.

**Primary scope:** Audit adoption friction across README, INSTALL, tutorial,
cookbook, solver-selection docs, API reference, examples, and maintainer docs;
add a compact support/readiness matrix; improve installed-consumer tutorials
for Make/pkg-config and CMake; normalize diagnostics wording; move narrative
workflow guidance out of selected public headers where possible without
changing declarations or Doxygen coverage; and run documentation, install,
example, and C quality gates appropriate to the changed surface.

**Non-goals:** Public API or ABI redesign, solver behavior changes, numerical
tolerance changes, broad package-manager support expansion, new platform
support claims, performance claims, release claims, state-of-the-art claims,
or partial cleanup across unrelated documentation without a consolidated user
truth.

---

## Day 1: Sprint Intake and Adoption Surface Inventory

**Title:** Adoption Intake
**Theme:** Establish Sprint 194 scope, owner files, user-facing adoption paths,
and support/readiness truth sources.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 194 section of the Epic 17 project plan and map items
   194.1 through 194.6 to likely owner docs, examples, headers, scripts, tests,
   and validation targets.
2. Inventory README, INSTALL, tutorial, cookbook, solver-selection, API
   reference, maintainer guide, examples, package proof docs, and selected
   evidence docs for adoption guidance and duplicated support claims.
3. Identify current support/readiness truth sources, including package-manager
   proof, Windows decision, selected comparison evidence, selected performance
   evidence, and residual queues.
4. Record adoption personas and workflows: source checkout, installed static
   archive consumer, `pkg-config` consumer, CMake consumer, solver chooser,
   and maintainer/release reviewer.
5. Create `WORKING_NOTES.md` with baseline inventory, risk register, user
   truth candidates, and Day 2 audit questions.

### Deliverables

- Sprint 194 working-notes scaffold.
- Adoption surface inventory.
- Support/readiness truth-source map.
- Initial duplication and friction register.

### Completion Criteria

- Sprint scope is traceable to items 194.1 through 194.6.
- User-facing and maintainer-facing truth sources are identified before edits.
- No documentation rewrite begins before the current adoption surface is
  recorded.

---

## Day 2: Adoption Friction Audit

**Title:** Adoption Audit
**Theme:** Rank documentation and example friction by user impact, duplication,
staleness risk, and evidence coupling.
**Time estimate:** 12 hours

### Tasks

1. Audit README, INSTALL, tutorial, cookbook, solver-selection docs, API
   reference docs, examples, and maintainer guide for duplicated setup,
   support, diagnostics, and evidence wording.
2. Compare each adoption path against current package, Windows, selected
   comparison, selected benchmark, and validation evidence.
3. Identify places where historical sprint evidence is presented as user truth
   or where user truth is buried in sprint-specific detail.
4. Rank cleanup candidates by reader impact, maintenance risk, validation
   coverage, and chance of accidental overclaim.
5. Create a Day 2 audit artifact with ranked issues and recommended cleanup
   boundaries.

### Deliverables

- Adoption-friction audit artifact.
- Ranked duplication and staleness table.
- Candidate cleanup boundaries.
- Initial overclaim and support-tier risk list.

### Completion Criteria

- Item 194.1 has a concrete, evidence-backed audit record.
- Cleanup candidates are ranked before implementation begins.
- Historical sprint evidence and active user-facing truth are separated in the
  audit.

---

## Day 3: Support Matrix Contract

**Title:** Support Matrix Design
**Theme:** Define a compact production-readiness/support matrix that becomes
the active user truth without erasing historical evidence.
**Time estimate:** 12 hours

### Tasks

1. Decide the support/readiness dimensions for the compact matrix, including
   build path, install path, platform, artifact type, validation status,
   evidence owner, and retained non-claims.
2. Map current truth from Sprint 188 package proof, Sprint 190 Windows
   decision, Sprint 191 comparison evidence, Sprint 192 benchmark evidence,
   and Sprint 193 review-surface work into matrix rows.
3. Define wording rules for `supported`, `validated`, `local-only`,
   `hosted-evidence`, `deferred`, `not claimed`, and `residual`.
4. Identify which existing docs should link to the matrix instead of
   duplicating support readiness wording.
5. Create the Day 3 support-matrix contract artifact.

### Deliverables

- Support/readiness matrix contract.
- Row and column definitions.
- Link-owner map for docs that should reference the matrix.
- Claim-boundary wording rules.

### Completion Criteria

- Item 194.2 has an agreed matrix schema before docs are edited.
- Matrix wording distinguishes user truth from historical sprint evidence.
- Non-claims are explicit for unsupported or deferred surfaces.

---

## Day 4: Support Matrix Implementation

**Title:** Support Matrix
**Theme:** Add the compact support/readiness matrix and wire top-level docs to
it as the active support truth.
**Time estimate:** 12 hours

### Tasks

1. Add or update the support/readiness matrix in the most appropriate
   user-facing documentation location.
2. Link README, INSTALL, tutorial, cookbook, and maintainer guide references to
   the matrix where they currently duplicate support/readiness details.
3. Preserve historical sprint evidence links for reviewers without making
   sprint artifacts the primary adoption path.
4. Include non-claim wording for package-manager breadth, Windows support
   breadth, shared-library support, portable performance, and state-of-the-art
   claims.
5. Create the Day 4 implementation artifact with changed files and retained
   claim boundaries.

### Deliverables

- Compact support/readiness matrix.
- Updated top-level references.
- Non-claim wording for unsupported or deferred surfaces.
- Day 4 implementation artifact.

### Completion Criteria

- Item 194.2 is implemented in user-facing docs.
- Support/readiness wording has one compact active truth source.
- Historical evidence remains available but does not dominate adoption docs.

---

## Day 5: Installed Consumer Tutorial Audit

**Title:** Consumer Tutorial Audit
**Theme:** Inspect current installed-consumer guidance and define minimal
Make/pkg-config and CMake tutorial flows.
**Time estimate:** 12 hours

### Tasks

1. Audit existing INSTALL, README, examples, package proof docs, and maintainer
   notes for installed static archive, `pkg-config`, and CMake consumer
   instructions.
2. Identify the smallest copy-pasteable external consumer examples that match
   current install evidence.
3. Separate source-tree developer workflows from installed-prefix consumer
   workflows.
4. Define expected commands, file names, include paths, link flags, artifact
   names, and failure wording for each tutorial path.
5. Create the Day 5 tutorial audit artifact.

### Deliverables

- Installed-consumer tutorial audit.
- Minimal Make/pkg-config tutorial contract.
- Minimal CMake tutorial contract.
- Source-tree versus installed-prefix boundary notes.

### Completion Criteria

- Item 194.3 has exact tutorial flows before docs are rewritten.
- Tutorial content matches current package/install evidence.
- Unsupported dynamic/shared-library or package-manager claims remain excluded.

---

## Day 6: Installed Consumer Tutorial Implementation

**Title:** Consumer Tutorials
**Theme:** Add or improve minimal installed-consumer instructions for
Make/pkg-config and CMake without broadening package claims.
**Time estimate:** 12 hours

### Tasks

1. Implement the Make/pkg-config installed-consumer tutorial with a minimal C
   source, compile command, expected include/link flags, and install-prefix
   assumptions.
2. Implement the CMake installed-consumer tutorial with a minimal
   `CMakeLists.txt`, target usage, build command, and expected static-link
   semantics.
3. Link tutorials from README/INSTALL without duplicating the full support
   matrix.
4. Add notes that local source-tree examples are separate from installed
   consumer validation.
5. Create the Day 6 implementation artifact with changed files and claim
   boundaries.

### Deliverables

- Improved installed-consumer tutorial.
- Make/pkg-config path guidance.
- CMake path guidance.
- Day 6 implementation artifact.

### Completion Criteria

- Item 194.3 is implemented with concise, copy-pasteable installed-consumer
  flows.
- Tutorial wording matches static archive and installed-prefix evidence.
- No new package-manager, shared-library, or dynamic-loader support claim is
  introduced.

---

## Day 7: Diagnostics Wording Contract

**Title:** Diagnostics Contract
**Theme:** Define consistent status/result/residual wording across direct,
iterative, QR/SVD, and eigensolver docs.
**Time estimate:** 12 hours

### Tasks

1. Audit current diagnostics wording in README, API reference docs, solver
   selection docs, cookbook, examples, public headers, and maintainer guide.
2. Group diagnostics language by direct solvers, iterative solvers, QR/SVD,
   eigensolvers, matrix I/O, and report/evidence tooling.
3. Define common terms for status codes, result structs, residuals,
   convergence, non-convergence, singularity, unsupported inputs, and expected
   failure modes.
4. Identify wording that could imply behavior changes, solver preference
   changes, or stronger correctness claims than tests support.
5. Create the Day 7 diagnostics wording contract artifact.

### Deliverables

- Diagnostics wording contract.
- Current wording inventory.
- Term normalization table.
- Claim-risk notes for diagnostics language.

### Completion Criteria

- Item 194.4 has a concrete wording standard before broad docs edits.
- Diagnostic terms are mapped to existing APIs and tests.
- No implementation or status-code behavior change is implied.

---

## Day 8: Direct and Iterative Diagnostics Cleanup

**Title:** Direct/Iterative Wording
**Theme:** Normalize diagnostics wording for direct and iterative solver docs
while preserving semantics.
**Time estimate:** 12 hours

### Tasks

1. Update direct-solver documentation for consistent factorization status,
   singularity, residual, refinement, and backend wording.
2. Update iterative-solver documentation for consistent convergence,
   non-convergence, iteration budget, residual, preconditioner, and result
   wording.
3. Keep solver-selection guidance practical without adding unsupported
   default-solver or superiority claims.
4. Cross-check examples and cookbook snippets for matching diagnostics terms.
5. Create the Day 8 cleanup artifact with before/after terminology notes.

### Deliverables

- Direct-solver diagnostics wording cleanup.
- Iterative-solver diagnostics wording cleanup.
- Example/cookbook wording alignment.
- Day 8 cleanup artifact.

### Completion Criteria

- Direct and iterative diagnostics docs use the Day 7 terminology.
- Wording changes are documentation-only unless a later audit justifies code
  changes.
- No solver behavior, status code, or tolerance semantics are changed.

---

## Day 9: QR/SVD/Eigensolver Diagnostics Cleanup

**Title:** QR/SVD/Eigs Wording
**Theme:** Normalize diagnostics wording for decomposition and eigensolver docs
without broadening numerical claims.
**Time estimate:** 12 hours

### Tasks

1. Update QR and SVD documentation for consistent rank, threshold, residual,
   reconstruction, orthogonality, and minimum-norm wording.
2. Update eigensolver documentation for consistent eigenvalue residual,
   backend, refinement, convergence, partial result, and selection wording.
3. Preserve selected evidence boundaries from recent QR, SVD, comparison, and
   benchmark sprints.
4. Cross-check public docs against maintainer claim-boundary notes.
5. Create the Day 9 cleanup artifact with retained non-claims.

### Deliverables

- QR/SVD diagnostics wording cleanup.
- Eigensolver diagnostics wording cleanup.
- Retained claim-boundary notes.
- Day 9 cleanup artifact.

### Completion Criteria

- QR/SVD/eigensolver diagnostics docs use the Day 7 terminology.
- Rank/threshold wording remains fixture- and API-accurate.
- No broad numerical superiority or state-of-the-art claim is added.

---

## Day 10: Header Narrative Audit

**Title:** Header Narrative Audit
**Theme:** Find public-header narrative workflow guidance that can move to docs
while preserving declarations and Doxygen coverage.
**Time estimate:** 12 hours

### Tasks

1. Audit public headers under `include/` for long-form workflow narrative,
   duplicated examples, support claims, and adoption guidance better owned by
   docs.
2. Separate required Doxygen API details from tutorial or support-readiness
   prose.
3. Identify candidate text that can move to README, INSTALL, API reference,
   tutorial, cookbook, or maintainer guide without losing generated API
   documentation.
4. Define preservation rules for declarations, macro names, struct fields,
   enum values, parameter docs, return-value docs, and Doxygen anchors.
5. Create the Day 10 header narrative audit artifact.

### Deliverables

- Public-header narrative audit.
- Candidate move list.
- Doxygen preservation checklist.
- Header non-goal and risk register.

### Completion Criteria

- Item 194.5 has a bounded candidate list before header edits.
- Required API documentation is distinguished from tutorial narrative.
- Header cleanup is explicitly constrained to documentation text unless a
  separate defect is found.

---

## Day 11: Header Narrative Cleanup

**Title:** Header Cleanup
**Theme:** Move selected narrative workflow guidance out of public headers
where safe while preserving exact declarations and API documentation coverage.
**Time estimate:** 12 hours

### Tasks

1. Apply selected public-header narrative cleanup from the Day 10 audit.
2. Move removed workflow/tutorial prose to the appropriate documentation owner
   when the content remains useful for adoption.
3. Preserve declarations, Doxygen command structure, parameter docs,
   return-value docs, examples required for API understanding, and generated
   API anchors.
4. Update docs links so users can find fuller workflow guidance outside public
   headers.
5. Create the Day 11 cleanup artifact with a declaration-preservation summary.

### Deliverables

- Selected public-header narrative cleanup.
- Relocated workflow guidance where appropriate.
- Declaration and Doxygen preservation summary.
- Day 11 cleanup artifact.

### Completion Criteria

- Item 194.5 is implemented for selected headers only.
- Public declarations and ABI-relevant text are unchanged except for narrative
  documentation cleanup.
- Any header changes are marked for full C quality-gate validation.

---

## Day 12: Documentation and Example Validation

**Title:** Docs and Examples Gate
**Theme:** Validate adoption docs, examples, install guidance, and Doxygen
coverage before final C gating.
**Time estimate:** 12 hours

### Tasks

1. Run available docs link checks, documentation guards, corpus/schema checks,
   Doxygen checks, and generated API documentation checks.
2. Build examples and tooling targets required by current Makefile validation.
3. Run install/package checks that exercise installed-prefix consumer paths
   where locally supported.
4. Inspect ignored generated outputs and confirm they are not accidentally
   staged.
5. Create the Day 12 validation artifact with commands, pass/fail results, and
   environment residuals.

### Deliverables

- Documentation validation artifact.
- Example build validation notes.
- Install/package validation notes.
- Generated artifact status notes.

### Completion Criteria

- Item 194.6 documentation/example/install checks are run or explicitly
  recorded as environment-unavailable.
- Generated or ignored artifacts are not staged accidentally.
- Any failing check is fixed before moving to the final quality gate.

---

## Day 13: Full Quality Gate and Claim Calibration

**Title:** Full Quality Gate
**Theme:** Run the final required quality gates and verify adoption claims are
bounded to evidence.
**Time estimate:** 12 hours

### Tasks

1. Run `make format`, `make lint`, and `make test` if any `.c` or `.h` files
   changed; otherwise run the focused documentation, install, examples, and
   guard validation set.
2. Re-run source-list, Doxygen, docs, package/install, and example checks
   affected by the sprint.
3. Audit final docs for unsupported support, portability, package-manager,
   dynamic-library, performance, release, or state-of-the-art claims.
4. Record final changed-file categories, validation evidence, residuals, and
   claim boundaries.
5. Create the Day 13 full-quality and claim-calibration artifact.

### Deliverables

- Final quality-gate artifact.
- Claim-calibration audit.
- Changed-surface summary.
- Residual and environment-unavailable notes.

### Completion Criteria

- Required quality gates pass before closeout.
- Public/user-facing claims match available evidence.
- Any header changes have full C gate evidence.

---

## Day 14: Closeout and Handoff

**Title:** Closeout
**Theme:** Complete final review, summarize adoption simplification, and prepare
the retrospective and PR handoff.
**Time estimate:** 10 hours

### Tasks

1. Re-check status, generated artifact hygiene, and final documentation
   consistency after validation.
2. Create the Day 14 closeout artifact with completed scope, changed files,
   final metrics, validation evidence, and residuals.
3. Prepare retrospective inputs covering what worked, what was constrained,
   final support/readiness matrix status, closed claims, and next-sprint
   handoff.
4. Confirm the sprint closes adoption/API coherence simplification rather than
   broad support, package, platform, performance, release, or state-of-the-art
   claims.
5. Ensure the branch is ready for retrospective, commit, push, and pull request
   creation.

### Deliverables

- Final Sprint 194 closeout artifact.
- Retrospective input notes.
- PR-ready implementation summary.
- Final residual handoff list.

### Completion Criteria

- Sprint 194 closes the selected adoption/API coherence simplification scope.
- Final validation evidence is recorded.
- Remaining gaps are documented as residuals or future candidates.
