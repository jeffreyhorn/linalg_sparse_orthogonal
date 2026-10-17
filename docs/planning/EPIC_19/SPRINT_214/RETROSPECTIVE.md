# Sprint 214 Retrospective

**Sprint:** 214 - ABI, Shared-Library And Release Readiness Decision

**Duration:** 14 days (Days 1-14 on branch `sprint-214`)

**Status:** Closed as ABI/release design with stronger static-first deferral

## Source Artifact Note

Sprint 214 follows the Epic 19 project-plan section for items 214.1-214.6.
The plan, working notes, release checklist, residual queue, and 14 daily
artifacts are under `docs/planning/EPIC_19/SPRINT_214/`. The Day 2 decision
selected complete ABI/release design and stronger deferral guards, not a
narrow shared-library proof. This preserves the Sprint 170 static-first and
Sprint 207 public package-provider decisions.

## Definition Of Done Checklist

- [x] Inventoried static build/install/consumer evidence, public header and
      symbol risks, platform CI ownership, and release evidence gaps.
- [x] Compared a named Linux shared proof with design/deferral and selected
      the latter before implementation.
- [x] Specified provisional dynamic export scope, visibility, ownership,
      layout, index/scalar, ABI epoch, cross-version, and ELF/Mach-O/PE loader
      acceptance gates without approving a dynamic ABI.
- [x] Strengthened unsupported shared/link-mode configure rejection and the
      installed static artifact/package-metadata contract.
- [x] Added 14 positive/mutation contract fixtures and registered selected
      Linux/macOS/Windows package checks; hosted branch results are not
      inferred from workflow registration.
- [x] Revalidated installed Make/`pkg-config` and CMake `find_package(Sparse)`
      consumers, missing-archive diagnostics, and package-version rejection.
- [x] Completed a release-readiness checklist for changelog, candidate
      identity, immutable tag, two-build archive reproducibility, provenance,
      candidate CI, approval, and rollback, without claiming a release.
- [x] Updated README, INSTALL, maintainer guidance, Epic 19 status, and a
      prioritized Sprint 214 residual handoff for the exact static-first
      support boundary.
- [x] Ran Day 13 integrated local quality/consumer checks and Day 14 affected
      install, guard, docs, format/source-list, and whitespace rechecks.

## What Went Well

1. **The scope decision preceded the guard work.** Days 1-2 made the choice
   against a partial shared target using explicit export, loader, consumer,
   platform, and compatibility criteria. Days 3-5 then recorded future gates
   without turning a provisional symbol subset into a product promise.

2. **Static package proof became more exact.** The configure guard rejects
   unsupported shared/link-mode selectors; installed-prefix validation checks
   static artifacts and package metadata together. The 14 fixtures exercise
   both valid static layouts and drift cases.

3. **Consumer evidence includes failures that matter.** Make and CMake tests
   execute out-of-tree installed consumers and reject absent static archives
   or incompatible package versions. A successful source-tree compile is not
   substituted for an installed-package result.

4. **The release checklist names stop conditions.** It ties a future release
   to one clean candidate, source archive, build provenance, installed files,
   hosted jobs, approvals, and rollback. Local static and Homebrew proofs are
   explicitly excluded as release or public-provider evidence.

5. **Closeout kept the evidence tiers separate.** Day 13 passed reviewed local
   compile and CMake quality checks, and Day 14 reran affected checks. The
   Epic status and residual queue still mark hosted candidate, shared ABI,
   and release readiness as unearned.

## What Didn't Go Well

1. **There is no shared artifact or dynamic consumer.** The selected path is
   a sound decision/deferral closure, but it does not give downstream users a
   shared library, an export allowlist, or a supported ABI epoch.

2. **Platform proof is asymmetric.** The current Linux/macOS routes include
   Make/`pkg-config` and CMake static checks; Windows is a CMake/MSVC static
   subset. This sprint records no branch-specific hosted job results and no
   Windows Make or `pkg-config` execution parity.

3. **Release gates are designed, not satisfied.** There is no approved
   changelog, frozen candidate, tag/signing policy, independently matching
   archives, source-to-binary record, hosted candidate matrix, or signoff.
   The checklist cannot be treated as release authorization.

4. **Full local validation is costly.** The Day 13 provider guard included a
   Homebrew CMake source bootstrap, and the generated API local-only fixture
   suite was long-running. The evidence record distinguishes those completed
   checks from Day 14's scoped reruns rather than silently omitting them.

5. **Static version metadata is easy to overread.** `VERSION=2.2.0` and CMake
   `ExactVersion` constrain package selection, not binary compatibility.
   Future shared work needs retained old binaries and executable cross-version
   tests before making any stability promise.

## Final Metrics

| Metric | Sprint 214 close state |
| --- | --- |
| Project-plan items reconciled | 6 of 6, on the selected design/deferral path |
| Daily artifacts | 14 |
| Shared libraries or dynamic ABI epochs promoted | 0 |
| Public provider or release artifacts promoted | 0 |
| Make installed-consumer test | 25/25 passed on Days 13 and 14 |
| CMake installed-consumer test | 30/30 passed, 0 skipped, on Days 13 and 14 |
| Static install contract fixtures | 14/14 passed on Days 13 and 14 |
| Reviewed CMake/CTest quality | 60/60 passed locally on Day 13 |
| Reviewed compile quality | passed locally on Day 13 |
| Static, API-docs, support-docs, and provider guards | passed locally on Day 13; affected guards rechecked on Day 14 |
| C/header files changed | 0 |

The Day 13 reviewed CMake build emitted a non-fatal warning in unchanged
`tests/test_svd_partial_corpus.c:83`; the target exited zero and 60/60 tests
passed. Day 14 passed `make docs-check`, `make api-docs-local-only`,
`make support-docs-guard`, package-provider guard fixtures, `make format-check`,
`make source-list-check`, and final whitespace checks. The full Day 13
`make quality-review-compile`, `make quality-review-cmake`, and
`make package-manager-deferral-guard` were not rerun for Day 14's
documentation-only edits. The separate `make format && make lint && make test`
trigger did not apply because no `*.c` or `*.h` changed.

## Closed Claim

Sprint 214 closes the ABI/shared-library/release *decision* by retaining the
static-first package, specifying future ABI and release acceptance gates, and
strengthening rejection of unsupported shared selectors, artifacts, metadata,
and claims. Local Make/CMake static installs and downstream consumers passed.

It does **not** claim shared-library build/install, dynamic ABI or
cross-version compatibility, Linux SONAME, macOS install-name/RPATH, Windows
DLL/import-library/CRT behavior, Windows Make or `pkg-config` parity, public
package-manager distribution, binary packages, reproducible source archives,
hosted candidate CI, generated API publication, release readiness, or
state-of-the-art status.

The evidence trail is [the plan](./PLAN.md),
[working notes](./WORKING_NOTES.md),
[release checklist](./RELEASE_READINESS_CHECKLIST.md),
[residual queue](./RESIDUAL_QUEUE.md), and these daily artifacts:

- [Day 1 evidence intake](./artifacts/day1-abi-release-evidence-intake.md)
- [Day 2 scope decision](./artifacts/day2-abi-release-scope-decision.md)
- [Day 3 symbol policy](./artifacts/day3-symbol-visibility-policy.md)
- [Day 4 compatibility policy](./artifacts/day4-abi-compatibility-policy.md)
- [Day 5 loader design](./artifacts/day5-loader-and-build-design.md)
- [Day 6 configure guard](./artifacts/day6-build-path-or-deferral-guard.md)
- [Day 7 install contract](./artifacts/day7-install-and-metadata-contract.md)
- [Day 8 automation design](./artifacts/day8-automation-and-consumer-design.md)
- [Day 9 consumer evidence](./artifacts/day9-installed-consumer-evidence.md)
- [Day 10 negative matrix](./artifacts/day10-consumer-negative-matrix.md)
- [Day 11 release sources](./artifacts/day11-release-sources-and-tags.md)
- [Day 12 provenance gates](./artifacts/day12-release-provenance-and-ci-gates.md)
- [Day 13 validation](./artifacts/day13-integrated-validation.md)
- [Day 14 closeout](./artifacts/day14-closeout-review.md)

## Residuals And Next-Sprint Readiness

The prioritized [Sprint 214 residual queue](./RESIDUAL_QUEUE.md) carries:

| Residual | Closure condition |
| --- | --- |
| Candidate-specific static source release | Clean reviewed commit, changelog, tag policy, manifest, two-build matching archives, source-to-binary provenance, hosted candidate CI, approvals, and rollback. |
| First shared-library product path | Explicit platform/toolchain/subset decision, reviewed exports and install metadata, runtime loader inspection, and executed installed dynamic consumer plus failure cases. |
| Dynamic ABI compatibility | ABI epoch and break policy, approved symbol/layout/ownership contract, retained old executable baseline, and cross-version tests for each claimed combination. |
| Platform expansion | Separate ELF, Mach-O, and PE loader/consumer proof; Windows Make and `pkg-config` parity only if separately selected and executed. |
| Hosted and distribution proof | Branch/candidate run URLs and digests; separate decisions for public provider, binary release, and generated API publication. |

Sprint 215 can use the static-first boundary as a planning input without
assuming a shared ABI or release. Sprint 216 should reconcile this queue with
the wider Epic 19 residuals and promote only claims backed by exact evidence.

## Final Assessment

The sprint delivered a complete, testable design/deferral outcome rather than
a partial shared-library demonstration. Static package regressions are more
specific and the future ABI/release bar is explicit. The product remains
static-first; a release or dynamic ABI requires a new decision and the
unexecuted gates in the checklist and residual queue.
