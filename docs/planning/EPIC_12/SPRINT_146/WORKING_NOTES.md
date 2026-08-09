# Sprint 146 Working Notes

## Goal

Sprint 146 validates the final Epic 12 product state, recalibrates public claims, publishes closed gaps and residuals, and decides whether any state-of-the-art sparse linear algebra claim is earned by the evidence produced across Sprints 137-145.

## Starting Evidence

- Sprint 137 established the Epic 12 baseline, gap selection, evidence contract, residual register, and the rule that state-of-the-art claims require direct comparative evidence.
- Sprint 138 created the maintained numerical corpus architecture and separated source-controlled corpus metadata from generated local outputs.
- Sprint 139 closed the bounded QR rank-deficient residual with the `qr_rank_deficient_6x4_nullspace_v1` fixture and reviewed QR corpus checks.
- Sprint 140 closed the bounded partial-SVD clustered-spectrum residual with explicit projector, residual, convergence, and fail-closed checks.
- Sprint 141 normalized report indexes, freshness gates, and report-row semantics while preserving generated-output boundaries.
- Sprint 142 documented runtime backend governance, typed controls, sentinel scope, and performance non-claims.
- Sprint 143 completed the static-first package and ABI decision with CMake and pkg-config contract evidence.
- Sprint 144 promoted platform lanes with Linux source-of-truth coverage, macOS reviewed static-first install/export proof, and Windows CMake-first support boundaries.
- Sprint 145 simplified adoption surfaces and published the adoption claim map that ties README, install, examples, cookbook, solver-selection, and header guidance to bounded evidence.

## Evidence Family Inventory

| Family | Evidence Source | Day 1 Closeout Position |
| --- | --- | --- |
| Corpus | Sprint 138 corpus manifests, schema checks, generator metadata, and oracle fixtures | Treat source-controlled fixtures and metadata as prerequisite evidence; generated reports remain reproducible local outputs. |
| QR | Sprint 139 QR corpus fixture, QR tests, solver-selection notes, and residual documentation | Claim only bounded reviewed QR residual closure for the named rank-deficient nullspace fixture. |
| Partial-SVD | Sprint 140 partial-SVD corpus fixture, projector/residual checks, convergence semantics, and tests | Claim only bounded repeated/clustered-spectrum fixture coverage and fail-closed behavior. |
| Report | Sprint 141 normalized report indexes, report family TSV schema, freshness checks, and generated-output rules | Claim normalized index semantics and freshness gates, not generated report freshness from checked-in output. |
| Runtime/Backend | Sprint 142 typed controls, runtime governance docs, sentinel notes, and benchmark scope | Claim governed runtime/backend selection surfaces; do not claim portable performance or backend superiority. |
| Package | Sprint 143 static-first install, CMake export, pkg-config, and ABI contract evidence | Claim static archive package metadata and install support only within documented lanes. |
| Platform | Sprint 144 Linux, macOS, and Windows staged-lane evidence | Claim Linux source-of-truth, macOS reviewed static-first install/export, and Windows reviewed CMake subset only. |
| Adoption | Sprint 145 README, install docs, examples, cookbook, solver-selection, header guidance, and claim map | Claim simplified high-level workflow entry points only where linked to maintained examples and evidence. |
| Validation | Sprint 137-145 quality gates plus Sprint 146 final validation package | Final claims require local quality evidence and hosted CI reconciliation or an explicit hosted-only caveat. |

## Item-To-Day Owner Map

| Sprint 146 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Final Evidence Inventory | Days 1-3 | Day 1 establishes families and handoff; Days 2-3 complete artifact-backed inventory. |
| Item 2: Full Quality Baseline | Days 4-5 | Day 4 runs and records the baseline; Day 5 reconciles failures and local-vs-hosted limits. |
| Item 3: Cross-Platform/CI Reconciliation | Days 6-7 | Day 6 audits platform lanes; Day 7 publishes CI reconciliation and staged blocker status. |
| Item 4: Claim and Non-Claim Audit | Days 8-9 | Day 8 maps public claims to evidence; Day 9 recalibrates unsupported or widened claims. |
| Item 5: Residual Queue Publication | Days 10-11 | Day 10 consolidates residuals; Day 11 attaches blockers, promotion gates, and next-epic handoff. |
| Item 6: Epic 12 Retrospective | Days 12 and 14 | Day 12 drafts retrospective; Day 14 finalizes after project-plan reconciliation. |
| Item 7: Final Project Plan Reconciliation | Days 13-14 | Day 13 reconciles Sprint 137-146 plan status; Day 14 publishes final closeout package. |

## Final Closeout Criteria

1. Every public support, coverage, performance, package, platform, and numerical claim is tied to explicit Sprint 137-146 evidence or moved to the residual/non-claim register.
2. The final validation package includes local quality gates and hosted CI reconciliation, or explicitly marks hosted-only evidence that cannot be reproduced locally.
3. Residual work includes an owner surface, blocker, prerequisite evidence, and promotion gate.
4. State-of-the-art language appears only if direct comparative evidence supports it; otherwise the final closeout records a non-claim.
5. Generated reports remain reproducible artifacts, not source-controlled freshness claims.
6. Package, platform, ABI, runtime/backend, and performance language stays within documented support tiers.

## Final Non-Claim Guardrails

- No unqualified state-of-the-art sparse linear algebra claim.
- No broad external library parity claim.
- No broad QR, SVD, or partial-SVD numerical parity claim beyond reviewed corpus fixtures.
- No portable performance, benchmark-superiority, or backend-superiority claim.
- No shared-library ABI, dynamic loader, package-manager, or static/shared selector claim.
- No Windows Makefile, Windows pkg-config, reviewed Windows install-validation, or staged POSIX/pthread test-closure claim.
- No generated report freshness claim from source-controlled report rows.

## Initial Validation Requirements

| Surface | Required Validation |
| --- | --- |
| Markdown-only closeout docs | `git diff --check` and trailing-whitespace scan over touched Markdown. |
| Corpus and report metadata | `python3 scripts/validate_corpus_schema.py`, `python3 scripts/normalize_report_index.py --check`, and report freshness checks when the indexed rows change. |
| Examples and install/package surfaces | `make examples-build`, `bash tests/test_install.sh`, and `bash tests/test_cmake_install.sh` when those surfaces change. |
| C or public-header changes | `make format && make lint && make test`. |
| CI and workflow changes | Workflow syntax inspection plus hosted CI status/reconciliation when available. |

## Stop Conditions

- A validation failure cannot be explained or fixed within the sprint-day scope.
- A public claim cannot be tied to a named artifact, test, report row, or reviewed CI lane.
- Hosted CI evidence is required to promote a claim but the run, logs, or review context is unavailable.
- Package, platform, ABI, runtime/backend, or performance wording widens beyond documented support.
- A state-of-the-art decision lacks direct comparative evidence.

## Daily Log

### Day 1: Closeout Intake

- Re-read the Sprint 146 project-plan section and converted Items 1-7 into day-level owners.
- Reviewed Sprint 137-145 retrospective outputs as prerequisite evidence, not reopened work.
- Created the Sprint 146 artifact structure under `docs/planning/EPIC_12/SPRINT_146/artifacts/`.
- Established evidence families, final closeout criteria, non-claim guardrails, validation requirements, and stop conditions.
- Day 2 handoff: complete the detailed final evidence inventory for corpus, QR, partial-SVD, report, runtime/backend, package, platform, adoption, and validation surfaces.

### Day 2: Corpus Evidence

- Inventoried Sprint 137-140 corpus, QR, partial-SVD, and solver-correctness evidence.
- Mapped each fixture-local claim to a source-controlled owner, generated-local evidence path, validation command, and freshness rule.
- Recorded QR evidence boundaries for `qr_rank_deficient_6x4_nullspace_v1`, including rank `3`, nullity `1`, and normalized null-vector residual `<= 1e-10`.
- Recorded partial-SVD evidence boundaries for `partial_svd_clustered_repeated_diag8x6_k3_v1`, including top-3 values, subspace projector checks, residual/orthogonality checks, default success, tight-budget non-convergence, and no partial arrays on failure.
- Separated source-controlled manifests, expected rows, proof-owner tests, helper headers, and docs from generated local oracle/report rows under ignored `build/` paths.
- Published unresolved numerical gaps and promotion gates for broad QR, partial-SVD, external parity, convergence, hosted platform, performance, and state-of-the-art claims.
- Day 3 handoff: inventory report, runtime/backend, package, platform, adoption, and validation support evidence using the same source-controlled versus generated-local distinction.

### Day 3: Support Evidence

- Inventoried Sprint 141-145 report, runtime/backend, package, platform, adoption, and validation evidence.
- Mapped report-family row meanings, freshness policies, support tiers, owners, claim scopes, and non-claims to `tests/corpus/manifests/report_families.tsv`, report-index schema docs, normalization scripts, and report tests.
- Mapped runtime/backend governance to maintainer docs, benchmark docs, advisory sentinel rows, performance sentinel scripts, and normalized report-index coverage.
- Mapped static-first package evidence to Make install, CMake install/export, `pkg-config`, exact-version proof, static deferral guard, package docs, and workflow support-tier lanes.
- Mapped platform evidence to Linux source-of-truth, macOS reviewed static-first install/export proof, and Windows CMake-first reviewed subset plus supplemental install/downstream confidence.
- Mapped adoption evidence to README, INSTALL, examples, cookbook, solver-selection, selected public headers, and report-family coherence repairs.
- Published support-tier gaps for hosted CI reconciliation, generated report freshness, Windows parity, shared-library ABI, package-manager distribution, tutorial alignment, broader header cleanup, portable performance, and competitive positioning.
- Day 4 handoff: design the strongest feasible Sprint 146 final validation baseline using the Day 2 and Day 3 evidence inventories.

### Day 4: Validation Baseline Design

- Derived validation requirements from the Day 2 numerical evidence inventory and Day 3 support evidence inventory.
- Selected the Day 5 required local baseline across corpus schema, report normalization/freshness, static package deferral, Make install, CMake install, examples, focused QR proof, focused partial-SVD proof, local oracle generation, and Markdown hygiene.
- Defined conditional gates for `.c`/`.h`, report metadata, corpus metadata, package/install, examples, runtime/backend sentinel, and workflow changes.
- Listed optional generated-evidence refresh commands for sentinels, benchmarks, large-matrix guardrails, dead-code, coverage, and required-generated report-family probes.
- Separated hosted-CI-only evidence for Linux, macOS, Windows, Windows supplemental CMake install/downstream confidence, and Sprint 146 PR checks.
- Published the pass/fail capture template, environment constraint register, and stop-or-fix rules for Day 5.
- Day 5 handoff: run the required local validation baseline in order, record exact results, and stop on unclear required failures.

### Day 5: Final Local Quality Baseline

- Ran the required local validation baseline from Day 4.
- Passed corpus schema validation, report-index unit tests, source-controlled report normalization, generated-aware normalization, full freshness diagnostics, selected support-family normalization, and selected support-family freshness.
- Passed static package deferral, Make install/`pkg-config`, CMake install/export, and maintained examples build checks.
- Passed focused QR corpus proof for `qr_rank_deficient_6x4_nullspace_v1`: 4 tests, 0 failures, 83 assertions, solver residual `2.220e-16`.
- Passed focused partial-SVD corpus proof for `partial_svd_clustered_repeated_diag8x6_k3_v1`: 6 tests, 0 failures, 140 assertions.
- Refreshed local corpus/oracle generated evidence under ignored `build/` paths: 15-line oracle TSV, 16-line report index, 2-line skips TSV, and 16-line manifest.
- Skipped the full C quality gate because no `.c` or `.h` files changed in Sprint 146; focused compiled proof owners still ran.
- Day 6 handoff: reconcile hosted Linux, macOS, and Windows CI lanes against the passing local baseline without treating local generated rows as hosted proof.

### Day 6: CI Evidence Intake

- Inspected Linux, macOS, and Windows workflow definitions and classified reviewed, supplemental, staged, local-only, hosted-only, and deferred lanes.
- Queried GitHub Actions with `gh`; no hosted runs exist for `sprint-146` yet.
- Collected the latest available hosted baseline from `master` commit `daac9a85d516f72100c34b90b92ec78941a72200`, created after PR #161 merged on 2026-08-09.
- Recorded successful hosted master workflow runs: CI `31335415785`, macOS CI `31335415782`, and Windows CI `31335415791`.
- Recorded job-level hosted evidence for Linux reviewed Makefile, CMake, dead-code, static-first package, supplemental runtime/sanitizer/bench-fast, TSan, and coverage lanes.
- Recorded job-level hosted evidence for macOS Apple Clang reviewed path, Homebrew GCC supplemental leg, reviewed Make install/`pkg-config`, and reviewed CMake install/export.
- Recorded job-level hosted evidence for Windows reviewed CMake subset and supplemental CMake install/downstream confidence; Windows expected CTest count remains `56`.
- Published the platform blocker register for Windows pthread/POSIX staged tests, Windows Makefile and `pkg-config` parity, Windows install-validation parity, and shared-library ABI.
- Day 7 handoff: reconcile current support-tier wording against the green master baseline, green Day 5 local baseline, and missing branch-specific hosted Sprint 146 evidence.

### Day 7: Cross-Platform Reconciliation

- Reconciled Linux, macOS, and Windows support tiers against the Day 6 hosted CI intake and Day 5 local validation baseline.
- Confirmed Linux support wording matches the latest green `master` hosted baseline: reviewed Makefile compile-quality, reviewed CMake parity, dead-code, reviewed static-first package contract, and supplemental runtime/sanitizer/bench-fast, TSan, and coverage lanes.
- Confirmed macOS support wording matches the latest green `master` hosted baseline: Apple Clang reviewed path, Homebrew GCC supplemental leg, reviewed static-first Make install/`pkg-config`, and reviewed static-first CMake install/export.
- Confirmed Windows support wording matches the latest green `master` hosted baseline: reviewed MSVC CMake-first subset with `56` expected CTest registrations plus supplemental CMake install/downstream confidence.
- Confirmed Windows Makefile parity, Windows `pkg-config` parity, reviewed Windows install-validation parity, and staged pthread/POSIX test closure remain unpromoted.
- Found no Day 7 source wording mismatch requiring a README, INSTALL, maintainer-guide, report-row, or workflow change.
- Preserved the key limitation: `sprint-146` still has no hosted branch/PR CI evidence, so final closeout must not claim branch-specific hosted pass until CI runs.
- Day 8 handoff: audit public docs and selected headers against the reconciled support tiers and explicit non-claims.

### Day 8: Public Claim Audit

- Audited README, INSTALL, examples README, cookbook, tutorial, solver-selection docs, benchmark README, selected public headers, and maintainer guide scan matches for unsupported public claims.
- Scanned for state-of-the-art, external-library parity, package-manager, shared-library, dynamic ABI, Windows parity, portable performance, generated-report freshness, and related overclaim wording.
- Found no public wording issue requiring a Day 8 source fix.
- Mapped public QR and partial-SVD fixture-local claims to Day 2 evidence and Day 5 focused proof results.
- Mapped report, runtime/backend, package, platform, adoption, and header-cleanup claims to Day 3, Day 5, and Day 7 evidence.
- Preserved explicit non-claims for state-of-the-art status, broad external parity, broad QR/SVD/partial-SVD correctness, portable performance, generated report freshness, shared-library ABI, package-manager support, Windows parity, and branch-specific hosted Sprint 146 CI.
- Day 9 handoff: audit maintainer/support surfaces, report rows, CI comments, benchmark/report guidance, install validation docs, and sprint artifacts for the same claim boundaries.

### Day 9: Support And Maintainer Claim Audit

- Audited maintainer guide, benchmark README, report-index schema, report-family rows, CI comments, install validation scripts, static package deferral guard, and Sprint 146 artifacts through Day 8.
- Confirmed source-controlled report rows remain advisory/metadata and do not imply generated pass evidence.
- Confirmed generated oracle, benchmark, sentinel, guardrail, dead-code, and coverage rows remain local/generated evidence unless hosted logs or explicit generated refresh gates are present.
- Confirmed benchmark and sentinel wording remains local measurement context, not portable performance, benchmark superiority, or state-of-the-art proof.
- Confirmed package and ABI support remains static-first, with shared-library support, dynamic ABI compatibility, runtime-loader compatibility, package-manager distribution, and static/shared selectors preserved as non-claims.
- Confirmed Windows support remains reviewed CMake-first with supplemental CMake install/downstream confidence and staged pthread/POSIX, Makefile, `pkg-config`, and install-validation parity gaps.
- Found no support or maintainer wording issue requiring a Day 9 source fix.
- Day 10 handoff: consolidate residuals from Sprints 137-145 and Days 2-9 into a prioritized queue with owners, blockers, prerequisites, and promotion gates.

### Day 10: Residual Queue Design

- Consolidated residual debt from Sprint 137-145 retrospectives, Sprint 145 adoption handoff, and Sprint 146 Days 2-9 evidence, validation, CI, and claim-audit artifacts.
- Grouped residuals by numerical coverage, report/validation, runtime/backend, package/ABI, platform, adoption/docs, and competitive positioning.
- Prioritized branch-specific hosted CI reconciliation, Windows staged/parity closure, shared-library ABI productization, broad QR and partial-SVD expansion, generated report refreshes, tutorial alignment, broader header cleanup, runtime/backend promotions, package-manager distribution, and competitive/state-of-the-art decisions.
- Assigned owner roles, blockers, prerequisites, promotion gates, and future-work classifications for each residual.
- Defined reusable promotion gates for hosted CI, numerical proof, report freshness, package product work, ABI, platform promotion, documentation coherence, header cleanup, and competitive claims.
- Preserved non-claim linkage so residuals do not become roadmap promises without evidence.
- Day 11 handoff: publish the final residual queue with cross-links to Days 2-9 evidence and next-epic handoff candidates.

### Day 11: Residual Queue Publication

- Published the final residual queue with stable IDs `R1` through `R14`.
- Added next-epic handoff candidates for Windows platform closure, shared-library/ABI productization, numerical corpus expansion, report evidence refresh, adoption/documentation completion, runtime/backend follow-through, and competitive positioning.
- Mapped each residual to the explicit non-claim that remains in force until the promotion gate passes.
- Cross-linked the residual queue to Sprint 146 Day 1-10 evidence artifacts.
- Recorded residual validation notes from Day 5 local validation, Day 6 hosted evidence intake, Day 7 platform reconciliation, Day 8 public claim audit, Day 9 support claim audit, and Day 10 queue design.
- Day 12 handoff: draft the Epic 12 retrospective with earned claims, non-claims, closed gaps, validation evidence, and conservative state-of-the-art assessment.

### Day 12: Epic 12 Retrospective Draft

- Drafted the Epic 12 retrospective from the Sprint 146 evidence package.
- Summarized Epic 12 goals, closed gaps, changed surfaces, validation evidence, and residuals across Sprints 137-146.
- Identified bounded earned claims for maintained corpus/report architecture, fixture-local QR closure, fixture-local partial-SVD closure, normalized report semantics, runtime/backend governance, static-first package proof, platform tier clarity, and adoption simplification.
- Preserved non-claims for state-of-the-art status, broad external-library parity, broad QR/SVD/partial-SVD correctness, portable performance, generated report freshness, shared-library ABI, package-manager distribution, Windows parity gaps, and branch-specific hosted Sprint 146 CI.
- Recorded the conservative state-of-the-art assessment: Epic 12 improved maturity and evidence ownership, but did not produce direct comparative evidence sufficient for a state-of-the-art claim.
- Added next-epic recommendations tied to complete gap closure rather than partial progress.
- Day 13 handoff: reconcile Sprint 137-146 project-plan items against the draft retrospective and carry incomplete work into the residual queue without widening claims.

### Day 13: Final Project Plan Reconciliation

- Reconciled Sprints 137-146 against `docs/planning/EPIC_12/PROJECT_PLAN.md`.
- Marked Sprint 137, Sprint 138, Sprint 141, and the completed Sprint 146 documentation closeout items as complete.
- Marked QR, partial-SVD, runtime/backend, platform, and CI outcomes as complete only within their explicit fixture-local or support-tier boundaries.
- Recorded static-first shared-library deferral, package-manager distribution, Windows staged/parity gaps, broad numerical coverage, generated report refreshes, tutorial alignment, broader header cleanup, runtime/backend follow-through, and competitive positioning as residual or deferred work.
- Rejected unqualified state-of-the-art status, broad external-library parity, portable performance, dynamic ABI compatibility, and generated report freshness from source-controlled rows alone as current Epic 12 claims.
- Drafted the next-epic handoff candidates around complete gap closure: Windows platform closure, numerical corpus expansion, shared-library/ABI productization, report evidence refresh, and adoption/documentation completion.
- Day 14 handoff: finalize the Epic 12 retrospective and Sprint 146 closeout package using the Day 12 draft, Day 13 reconciliation, validation evidence, and residual queue.

### Day 14: Final Closeout Package

- Finalized the Epic 12 retrospective at `docs/planning/EPIC_12/EPIC_12_RETROSPECTIVE.md`.
- Published the Sprint 146 final closeout package with final deliverables, validation package, claim/non-claim audit status, residual queue, next-epic handoff, and Sprint 146 retrospective input notes.
- Confirmed all seven Sprint 146 project-plan items are complete or complete with explicit boundaries.
- Preserved branch-specific hosted Sprint 146 CI as residual `R1` because only the latest inspected hosted `master` baseline is available locally.
- Preserved state-of-the-art status, broad external-library parity, broad numerical parity, portable performance, shared-library ABI, package-manager distribution, Windows parity, and generated report freshness as non-claims.
- Confirmed the Sprint 146 closeout remains documentation-only; no `.c` or `.h` files changed, so the final C quality gate is not required.
- Sprint 146 is ready for the sprint retrospective and PR closeout once requested.
