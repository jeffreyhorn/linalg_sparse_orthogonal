# Day 1: Support Intake

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Theme:** Establish Sprint 205 scope, inherited support decisions, and the
documentation surfaces that need consolidation.  
**Time estimate:** 12 hours  
**Branch:** `sprint-205`  
**Base commit:** `9a2b4ff6`

## Scope

Day 1 establishes the Sprint 205 support/readiness and adoption-routing
surface. It does not change user-facing support wording yet. The current
repository posture is:

- `INSTALL.md#support-readiness-matrix` is the public support/readiness
  authority.
- README, tutorial, cookbook, solver-selection, API reference, examples, and
  benchmark docs route users to task-specific workflows.
- `docs/maintainer_guide.md` owns detailed maintainer interpretation for
  claim boundaries, selected evidence, residuals, generated API policy, and
  report-index behavior.
- Sprints 198 through 204 closed selected gaps or recorded explicit
  re-deferrals; none of those closures promoted broad package-manager,
  platform, ABI, performance, release, or state-of-the-art support.
- Sprint 205 should reduce repeated caveats by routing users to authoritative
  support truth, not by deleting claim boundaries.

## Item Traceability

| Item | Day 1 mapping | Planned evidence |
| --- | --- | --- |
| 205.1 Public Doc Audit | Identified public surfaces and current support/readiness owner. | Day 2 public documentation audit and Day 3 maintainer/report audit. |
| 205.2 Quick Reference Design | Identified `docs/cookbook.md`, `docs/solver_selection.md`, README, tutorial, and INSTALL as candidate quick-reference route surfaces. | Day 4 quick-reference design artifact. |
| 205.3 Support Truth Consolidation | Identified `INSTALL.md#support-readiness-matrix` as initial authority and flagged repeated caveats for audit before consolidation. | Day 5 support-truth architecture and Day 7 consolidation batch. |
| 205.4 Diagnostics Vocabulary | Identified direct, iterative, QR/SVD, eigensolver, benchmark, and report-index wording as the vocabulary scope. | Day 9 diagnostics vocabulary design and Day 10 implementation evidence. |
| 205.5 Claim Guard Updates | Identified existing API, install, benchmark, Windows, and claim-boundary guards as likely reuse surfaces. | Day 11 guard design and Day 12 implementation evidence. |
| 205.6 Validation | Identified docs/API/install/claim guards and C-gate escalation rule. | Day 13 integrated validation and Day 14 closeout. |

## Public And Maintainer Surface Map

| Surface | Current Day 1 role | Sprint 205 handling |
| --- | --- | --- |
| `README.md` | First contact for build, docs, benchmarks, and support wording. | Audit for duplicated caveats and route to quick reference/support truth. |
| `INSTALL.md` | Support/readiness matrix and install/package boundary authority. | Preserve as support truth unless Day 5 deliberately selects another owner. |
| `docs/tutorial.md` | Introductory learning path with links to install/support, benchmarks, reports, headers, and maintainer interpretation. | Keep tutorial learning-focused; replace support detail with authoritative links where safe. |
| `docs/cookbook.md` | Task-oriented workflow recipes and problem-shape hints. | Strong candidate for quick-reference placement or cross-link. |
| `docs/solver_selection.md` | Detailed solver choice and selected evidence boundary guide. | Use as deep-dive target; avoid duplicating every detailed table in the quick reference. |
| `docs/api_reference.md` | Source-controlled API route and generated API local-only policy wording. | Preserve Sprint 204 local-only generated API boundaries. |
| `examples/README.md` | Example discovery and user routing. | Route examples to quick reference/support matrix without creating support claims. |
| `benchmarks/README.md` | Benchmark commands, report-index interpretation, and local/selected evidence boundaries. | Keep as measurement/report authority, not adoption support proof. |
| `docs/maintainer_guide.md` | Detailed support, claim, residual, and guard interpretation. | Day 3 audit owner for guard and vocabulary alignment. |
| Existing docs guards and tests | Executable claim-boundary protection. | Reuse or extend rather than inventing a separate support-claim system. |

## Inherited Decision Inventory

| Area | Current inherited decision | Claim boundary retained for Sprint 205 |
| --- | --- | --- |
| Package/Homebrew | Sprint 198 completed only a bounded developer-mode local Homebrew static source formula proof. | No Homebrew support, Homebrew/core readiness, bottles, Linuxbrew, public tap, package-manager distribution, shared-library package support, dynamic ABI, or runtime-loader claim. |
| Windows Cholesky | Sprint 199 recorded selected Windows Cholesky evidence as re-deferred for promotion. | No promoted selected Windows freshness or broad Windows report freshness from that evidence. |
| Allocation failure | Sprint 200 closed the selected `sparse_symbolic_lu()` owner proof. | No broad allocation-failure or state-of-the-art reliability claim. |
| Review-surface reduction | Sprint 201 closed selected SVD helper extraction. | No broad SVD behavior, repository-wide maintainability, package, platform, API, or ABI claim. |
| Hosted benchmark freshness | Sprint 202 closed one additional hosted selected-performance lane for macOS alongside Linux for the selected benchmark row. | No portable performance, timing threshold, benchmark-family publication, package/ABI, release, broad platform, or state-of-the-art performance claim. |
| Windows QR incompatible | Sprint 203 re-deferred Windows QR incompatible selected freshness after local proof and guard work. | No Windows QR selected freshness, broad QR Windows proof, package/ABI, performance, release, or state-of-the-art claim. |
| Generated API | Sprint 204 selected stronger local-only generated API policy. | No hosted API docs, retained generated-doc artifact, committed generated HTML, ABI completeness, package support, or release evidence. |

## Initial Consolidation Hypotheses

These are hypotheses for later days, not Day 1 decisions:

1. Keep `INSTALL.md#support-readiness-matrix` as the support/readiness truth
   and convert repeated caveats elsewhere into links.
2. Add a compact adoption quick reference that routes by problem shape and
   environment question rather than by implementation history.
3. Keep deep solver evidence in `docs/solver_selection.md` and benchmark/report
   evidence in `benchmarks/README.md`.
4. Normalize diagnostics vocabulary around status/result, residual,
   convergence, freshness, support tier, and claim boundary terms.
5. Reuse existing guard families for claim protection: install docs checks,
   API routing/local-only checks, selected performance docs checks, Windows
   PowerShell/claim validators, and manifest/report-index tests.

## Day 1 Non-Goal Ledger

Day 1 intentionally does not:

- edit README, INSTALL, tutorial, cookbook, solver-selection, API reference,
  examples, benchmarks, or maintainer docs;
- add or change support claims;
- add quick-reference content before the audit and design days;
- change `.c`, `.h`, Makefile, CMake, workflow, or generated output behavior;
- run generated API or full C validation, because no user-facing docs, source,
  public header, or guard files changed on Day 1.

## Validation And Hygiene

| Check | Day 1 result |
| --- | --- |
| `git diff --check` | Planned after Day 1 artifact creation. |
| C/header quality gate | Not required for Day 1; no `.c` or `.h` edits. |
| Generated-output status | No generated output intentionally created. |
| User-facing claim drift | No user-facing docs edited on Day 1. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Every Sprint 205 item has an initial evidence path or artifact category. | Met. Item traceability is recorded in `WORKING_NOTES.md` and this artifact. |
| All public and maintainer support surfaces are identified before edits. | Met. Public and maintainer surface maps are recorded before any user-facing documentation changes. |
| Unsupported package, ABI, platform, performance, release, and state-of-the-art claims remain explicitly out of scope. | Met. The inherited decision inventory and non-goal ledger retain those claim boundaries. |

## Day 1 Disposition

Day 1 is complete. Day 2 should perform the public documentation audit and
produce a duplication/friction inventory before any consolidation or quick
reference edits.
