# Sprint 216 Day 11: Quality Gate And Retrospective Outline

## Scope And Platform

The [Day 9 validation matrix](day9-residual-decision-validation-matrix.md)
assigns Day 11 the installed-static and package-deferral gates. The
[Day 10 record](day10-focused-validation.md) contains the claim, API, corpus,
comparison, benchmark, pilot-synthetic and documentation outcomes. Day 11 ran
on `sprint-216`, Darwin `24.6.0` x86_64, Apple Clang `17.0.0`, Python
`3.14.5`, CMake and local Homebrew, from a dirty checkout at
`c5f30490fc02d34ed27be363e1586c3472089faf`. No tracked or untracked
`.c` or `.h` file changed. The mandatory source chain
`make format && make lint && make test` was therefore **not triggered**, not
silently counted as a pass.

## Day 11 Phase Results

Commands ran serially from the repository root. Every executed phase exited
`0`; no failing gate was waived.

| Phase | Exact command | Exit | Result and limit |
| --- | --- | ---: | --- |
| Static package validator regressions | `python3 tests/test_static_install_contract.py` | 0 | 21 tests passed; validates the static installed-package contract and mutation rejections. |
| Static/shared deferral | `bash scripts/static_package_deferral_check.sh` | 0 | Shared selectors rejected; default CMake/Make package remains static-only; no shared export or ABI claim. |
| Provider guard regressions | `python3 tests/test_package_manager_deferral_guard.py` | 0 | Provider wording/negative fixtures passed; no public provider status inferred. |
| Selected provider boundary | `bash scripts/package_manager_deferral_check.sh` | 0 | Deferral record, provider absence, public non-claims, and developer-mode local Homebrew static source formula proof boundary passed. This is not Homebrew/core, a public tap, bottles or Linuxbrew support. |
| Make installed consumers | `bash tests/test_install.sh` | 0 | 26 passed, 0 failed: static archive and 19 headers installed, `pkg-config` exact version/flags, basic and maintained-example compile/run, missing-archive validator and explicit-path link rejection, uninstall. |
| CMake installed consumers | `bash tests/test_cmake_install.sh` | 0 | 30 passed, 0 failed, 0 skipped: static imported target/config metadata, 19 headers, out-of-tree `find_package(Sparse)` build/run, exact/mismatched version tests, missing-archive rejection. |
| Source gate | `make format && make lint && make test` | Not run | Conditional gate; no C/header diff or untracked C/header file. Any later C/header edit activates all three phases in order. |

The package-manager guard invoked the selected local Homebrew proof in a
temporary developer-mode tap. Its `0` exit establishes only the guarded
local proof path. The Make install script invoked `make clean`; the ignored
Day 10 `build/comparison/` files were absent afterwards. Their observed
results and SHA-256 snapshots remain in the Day 10 record, but the snapshots
are **not** a retained release artifact. Rebuild and rerun freshness before
any later comparison/report claim. The CMake install script used its own
temporary build/prefix, and the serial order avoided build-tree conflicts.
After the Day 11 planning edits, `python3 -m unittest
tests.test_external_baseline_claim_docs` (5 tests), `make support-docs-guard`,
and `make docs-check` (18 reference/source pages for 18 public headers) each
exited `0` on the same local host. Local links, trailing-whitespace inspection,
and `git diff --check` are the final Day 11 document hygiene checks.

## Reconciliation And Repeat Gates

- Day 10 focused checks and Day 11 package/install checks all passed in their
  local scopes. They cover the changed docs, Python claim guard, generated API
  routing, corpus/report schema, selected comparison and benchmark freshness,
  synthetic blocked-pilot behavior, static install and provider non-claims.
- Day 2 discrepancies [D2-2 through D2-6](day2-outcome-validation-reconciliation.md#discrepancy-and-gap-register)
  are only partially narrowed: current-tree local reruns replace stale
  historical check counts, but neither hosted currency nor external v1
  schema/solver evidence was supplied. The 13 current pilot tests supersede
  the historical nine-test count as a *current local test count*, not as an
  external result.
- Day 14 must rerun relevant text/claim guards and `git diff --check` after
  retrospective and project-plan edits, verify all relative links and
  untracked Markdown whitespace, and repeat affected API/corpus/report or
  package gates if those contracts change. Report/benchmark freshness needs
  rebuilding after the Day 11 `make clean`. If any C/header file changes,
  run the full source chain and do not close on a failing phase.
- Hosted Windows Cholesky/QR selected freshness, candidate-specific release
  CI, shared loader/ABI behavior, public package/API publication and the
  external CHOLMOD/timing matrix were **not run locally**. These remain
  [queue](../RESIDUAL_QUEUE.md) gates, not Day 11 passes. The
  [Day 9 decision](day9-residual-decision-validation-matrix.md#state-of-the-art-evidence-decision)
  still says no state-of-the-art claim is earned.

## Epic 19 Retrospective Outline

The Day 12 draft should create `docs/planning/EPIC_19/EPIC_19_RETROSPECTIVE.md`
with the following evidence-led structure; the outline is not the final
retrospective or item 216.5 completion.

1. **Objective, scope and chronology.** Start with the [Epic 19 project
   plan](../../PROJECT_PLAN.md)
   and [Day 2 reconciliation](day2-outcome-validation-reconciliation.md).
   Distinguish Sprint 207-215 merged outcomes from Sprint 216 current-tree
   closeout work and retain D2-1's corrected 215/216 status.
2. **Selected sprint outcomes by capability.** Use the [Day 6](day6-project-plan-status-207-211.md)
   and [Day 7](day7-project-plan-status-212-215.md) item ledgers for all 54
   selected deliverables, then cite primary sprint evidence: [provider
   decision](../../SPRINT_207/artifacts/day5-provider-decision.md),
   [Windows Cholesky decision](../../SPRINT_208/artifacts/day5-promotion-decision.md),
   [Windows QR decision](../../SPRINT_209/artifacts/day8-manifest-decision.md),
   [allocation sweep](../../SPRINT_210/artifacts/day6-failure-sweep.md),
   [review extraction](../../SPRINT_211/artifacts/day10-behavior-regression.md),
   [benchmark policy](../../SPRINT_212/artifacts/day5-product-policy-decision.md),
   [API policy](../../SPRINT_213/artifacts/day5-product-policy-decision.md),
   [ABI/release checklist](../../SPRINT_214/RELEASE_READINESS_CHECKLIST.md),
   and [blocked external pilot](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md).
   A complete decision or design is not a promoted capability.
3. **Claim recalibration and validation.** Link the [Day 3 claim inventory](day3-claim-inventory.md),
   [Day 4](day4-user-maintainer-claims.md) and [Day 5](day5-benchmark-api-corpus-claims.md)
   edits, [Day 10 local checks](day10-focused-validation.md) and this Day 11
   installed-static gate. Report exit codes, platform, dirty-tree provenance,
   generated-output deletion by `make clean`, and the conditional C gate
   honestly; add Day 13/14 final validation deltas before calling the epic
   closed.
4. **Unsupported claims, negative evidence and limits.** State explicitly
   that public provider distribution, Windows selected freshness, broad
   allocation/review completion, timing thresholds/portable speed, published
   generated HTML, shared ABI/release, accepted external parity and
   state-of-the-art status were not earned. Cite the [Day 9 decision](day9-residual-decision-validation-matrix.md)
   and D2-4/D2-5 blockers; synthetic tests, source-controlled dense helpers,
   a blocked `attempt.json`, workflow registration and local proof cannot
   replace accepted hosted/external evidence.
5. **Residual handoff and final decision.** Link the [21-item priority
   queue](../RESIDUAL_QUEUE.md) with P0 prerequisite order, P1 selected
   closures and P2 scope decisions. Day 13 should assess the minimum
   state-of-the-art evidence bar; Day 14 should assign 216.1-216.6 statuses
   only after final gates and consistency review.

The [Epic 18 retrospective](../../../EPIC_18/EPIC_18_RETROSPECTIVE.md) is a
format reference, not evidence that Epic 19 achieved the same capabilities.
