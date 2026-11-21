# Sprint 226 Day 1: Candidate Preconditions and Ownership Inventory

## Decision

The candidate is **not frozen**. The current `master` merge commit, `c42d6715fad7e895bae4941a006df3563f1afc38`, is an inventory baseline only. This sprint's plan and notes are untracked, so the working tree is not a clean release input. No tag, archive, candidate-specific hosted receipt, release approval, or timing pass is asserted. Day 2 must choose a clean, reviewed, committed candidate and record its full identity.

## Prerequisite reconciliation

| Source | Day 1 reading | Implication for Sprint 226 |
| --- | --- | --- |
| [Sprint 214 release checklist](../../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md) and [residual queue](../../../EPIC_19/SPRINT_214/RESIDUAL_QUEUE.md) | Static-first path selected; candidate identity, changelog, license/source manifest, reproducible archive, installed-file mapping, hosted platform proof, approvals, and rollback remain release gates. Shared library, dynamic ABI, and provider release are deferred. | Use the checklist as gates; do not treat its design or historical local passes as candidate proof. |
| [Sprint 217](../../SPRINT_217/RETROSPECTIVE.md) | Norm and ILUT diagnostics work was locally closed at its retrospective; review/merge was pending then. | Carry numerical validation into exact-candidate quality proof; historical local results are not current release evidence. |
| [Sprint 218](../../SPRINT_218/RETROSPECTIVE.md) | Matmul construction work was locally closed at its retrospective; review/merge was pending then. | Include its tests in candidate quality gates. |
| [Sprint 219](../../SPRINT_219/RETROSPECTIVE.md) | 32/64 width-aware build, installed metadata, and consumers were locally closed; hosted 64-bit and final acceptance remained open at its retrospective. | Verify index width and clean-prefix consumers on the candidate and claimed platforms. |
| [Sprint 220](../../SPRINT_220/RETROSPECTIVE.md) | Structured local-only API workflow guard was locally delivered; hosted validation was pending at its retrospective. | Preserve the generated API publication boundary and rerun applicable candidate guards. |
| [Sprint 221](../../SPRINT_221/RETROSPECTIVE.md) | External evidence validator was locally delivered; first real external cell and hosted CI were pending. | Do not infer external performance or state-of-the-art evidence from validator existence. |
| [Sprint 222](../../SPRINT_222/RETROSPECTIVE.md) | Two local CHOLMOD numerical pairs passed, but a complete observed v1 cell and publication remained blocked by provenance and rights review. | Do not promote an external comparison or rights claim as part of candidate readiness. |
| [Sprint 223](../../SPRINT_223/RETROSPECTIVE.md) | One bounded Windows Cholesky selected row had exact-head hosted freshness at its Day 14 head; subsequent commits require a new audit. | Retain the historical claim only at its recorded SHA; obtain fresh candidate evidence if used. |
| [Sprint 224](../../SPRINT_224/RETROSPECTIVE.md) | One bounded Windows QR incompatible row had exact-head hosted freshness at its Day 14 head; subsequent commits require a new audit. | Do not generalize selected rows to broad Windows parity. |
| [Sprint 225](../../SPRINT_225/RETROSPECTIVE.md) | Timing method and fail-closed job exist, but no compatible baseline was accepted and hosted attempts returned `S225_BASELINE_MISSING` before candidate measurement. | Timing gate is **blocked**. A selected-row freshness pass cannot substitute for a timing pass. |

These are the status statements in sprint retrospectives, not an independent re-audit of every earlier hosted artifact against a future candidate SHA. The current merge commit includes PR #249, but no Sprint 226 release proof is attached to it.

## Source and package inventory

| Surface | Current repository evidence | Open release gate |
| --- | --- | --- |
| Version/changelog | `VERSION` contains `2.2.0`; Make and CMake consume it. No root `CHANGELOG.md` was found. | Review version coherence across generated header, `sparse.pc`, CMake version file and installed consumers; add and approve candidate changelog. |
| Legal/source | `LICENSE` exists. Git tracks 49 `src/*.c`, 18 `include/*.h`, and one `include/*.h.in` template. The checklist specifies the minimum source/license/build/test/docs manifest. | Review the full tracked file list and archive contents, including source license and generated-file exclusions. |
| Make install | `Makefile` installs the static archive, public headers, generated version header, and `sparse.pc` into a prefix. `SPARSE_IDX_BITS` accepts 32 or 64 with a build-tree stamp. | Build from candidate archive in clean 32/64 trees; verify metadata and downstream use. |
| CMake install | `CMakeLists.txt` installs the static target, public and generated headers, exported `Sparse::` target, exact-version config, and `sparse.pc`; index width is a 32/64 cache selector. | Verify candidate install manifest and CMake consumers; compare installed outputs with archive inputs. |
| First use | `README.md`, `INSTALL.md`, and `examples/README.md` provide the existing static source/install route. | Run the documented route from extracted candidate source and a clean prefix; check links and header coverage. |
| Archive | No Sprint 226 candidate archive or independent hash pair exists. | Build twice from clean independent checkouts, compare bytes/manifests, extract and build one copy. |

## Claimed platform and gate inventory

The maintained user-facing scope is a Unix static Make/`pkg-config` install route, a CMake static installed consumer route, and a Windows MSVC CMake install/downstream subset. The exact set of platforms to claim for this candidate is a **Day 2 owner decision**, not silently inherited from workflow registrations. The [install support matrix](../../../../../INSTALL.md) states the current boundaries.

| Platform route | Registered gate | Candidate-specific status |
| --- | --- | --- |
| Linux | `.github/workflows/ci.yml`: reviewed Make/CMake quality and `package-contract`, including installed Make/`pkg-config` and CMake consumers. | No Sprint 226 candidate SHA/run URL/log/artifact digest recorded. |
| macOS | `.github/workflows/macos-ci.yml`: Apple Clang reviewed path, `install-and-pkgconfig`, and `cmake-install-export`; supplemental GCC is separate. | No Sprint 226 candidate SHA/run URL/log/artifact digest recorded. |
| Windows | `.github/workflows/windows-ci.yml`: MSVC reviewed CMake subset and `install-and-downstream`; no Make or `pkg-config` execution parity. | No Sprint 226 candidate SHA/run URL/log/artifact digest recorded. |
| Canonical timing | `.github/workflows/sprint225-timing.yml`: separate macOS `s225-v1` decision job. | Blocked upstream by absent accepted baseline; no candidate measurement. |

The [Sprint 214 checklist](../../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md) also requires documentation/provider guards, C/build quality, static/shared deferral checks, exact candidate commit identity, and retained hosted provenance. A green workflow definition or previous sprint run is not a candidate result.

## Ownership and blockers

| Role or gate | Day 1 state | Required action |
| --- | --- | --- |
| Candidate selection and release maintainer | Unconfirmed | Name an authorized maintainer and have them approve the exact candidate, version, changelog, tag/signing policy, and go/no-go record. |
| Build/release and independent archive verifier | Unconfirmed | Assign distinct review responsibility for source manifest, two-build archive comparison, and source-to-install mapping. |
| CI/platform owner | Unconfirmed | Confirm named platform tiers and review exact-SHA hosted results, logs, runner identities, and hashes. |
| API/ABI and documentation owners | Unconfirmed | Review static-only wording, installed first use, API links, and absence of unsupported ABI/provider/performance claims. |
| License/legal and product owners | Unconfirmed | Review license, release scope, external-claim boundary, and any inherited blocked gate. |
| Rollback owner | Unconfirmed | Define cancellation/withdrawal, consumer communication, and corrective-version procedure for an actual destination. |
| Changelog and clean candidate | Missing | Produce approved changelog and a committed, clean candidate before freeze. |
| Sprint 225 timing | Blocked | Obtain an accepted baseline and exact-candidate decision, or record an explicit owner-approved scope decision retaining the timing blocker. |

No owner identity, approval, or release authorization is inferred from repository authorship or a prior PR merge.

## Claim boundary and Day 2 handoff

The currently supported language is **static source build and installed first use within the documented routes**. No reproducible source release, release-ready candidate, tag, public distribution, shared ABI, provider package, portable speed, external winner, or broad state-of-the-art claim follows from Day 1. Day 2 must freeze the candidate definition and platform/gate matrix, settle ownership assignments, and carry all unresolved rows forward as blockers.
