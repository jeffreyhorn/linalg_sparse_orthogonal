# Sprint 226 Day 2: Static Candidate Definition and Freeze Decision

## Decision

**Candidate freeze blocked.** The reviewable base snapshot is commit `c42d6715fad7e895bae4941a006df3563f1afc38` (tree `934a16b09ada33e12f726ec944fc675bd7d244f4`) with `VERSION=2.2.0`. It is a **proposed source basis**, not an approved release candidate. Its tree has no `CHANGELOG.md`; the current branch has uncommitted Sprint 226 documentation; a release maintainer, rollback owner, and required reviewers have not been named or approved. The Sprint 214 checklist makes the changelog, clean candidate, and owner review gates mandatory. No archive or hosted run from this base may be called candidate proof until a new committed, reviewed SHA is selected.

This is the [Day 2 plan](../PLAN.md)'s explicit blocked-candidate outcome. Day 3's first candidate archive is dependent on clearing it. An exploratory archive of the base commit, if made, must be labeled exploratory and cannot close the release gate.

## Source, version, and license inventory

| Field | Recorded value | Review action |
| --- | --- | --- |
| Base commit | `c42d6715fad7e895bae4941a006df3563f1afc38` | Inspect full diff and choose final candidate after all release inputs are committed. |
| Base Git tree | `934a16b09ada33e12f726ec944fc675bd7d244f4` | Use this tree identity to detect any changed source input. |
| Tracked-file manifest | 4,557 paths; `git ls-tree -r --name-only HEAD \| shasum -a 256` = `9a7df5b62c46f532efead0068939ab5c4d107a0294a98b5bb0e8e222599a39b1` | Recompute against the final candidate; review the **full** listing, not just the digest. |
| Top-level categories | 4,174 `docs/`, 183 `tests/`, 69 `src/`, 44 `scripts/`, 20 `benchmarks/`, 19 `include/`, 19 `examples/`, 12 root files; remaining paths in `.github/`, `cmake/`, and other tracked directories. | Inspect for generated/local-only material and unexpected release content. |
| Required source and build inputs | `LICENSE`, `VERSION`, `README.md`, `INSTALL.md`, `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `Doxyfile`, `.clang-format`, `.clang-tidy`, and the tracked `cmake/`, `include/`, `src/`, `examples/`, `tests/`, `scripts/`, and `benchmarks/` paths are present in the base tree. | Validate the final archive against the [Sprint 214 minimum and forbidden-path rules](../../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md). |
| License | Tracked root `LICENSE`; SHA-256 `ddc5e5f268b30436950b0bd1b16a510eabafde89b94808f5ee762b7977e90315`. | License reviewer must approve its inclusion and any third-party inventory. |
| Version | Root `VERSION` is `2.2.0`; SHA-256 `c4a2383a03bdb6739d16a0e24058e4b9c7da4e63d203e0be2f448868cc03c530`. The last commit touching `VERSION` is `9ebb24047b62efde32a8afa213e0e86c73b55549`. | Release maintainer must decide whether `2.2.0` is an appropriate candidate version and check generated header, `sparse.pc`, and CMake exact-version output. |
| Changelog | No root `CHANGELOG.md` in the base tree; no `v2.2.0` tag found locally. | Draft a version-specific entry from an owner-approved comparison base; review API/package changes, migrations, known issues, and static-only non-claims. Do not guess a prior release boundary. |
| Worktree | `sprint-226` has untracked plan and Day 1/2 artifacts. | Commit and review the intended source inputs, then confirm a clean checkout for the selected SHA. |

The full tracked-file manifest can be regenerated with `git ls-tree -r --name-only c42d6715fad7e895bae4941a006df3563f1afc38`. The Git tree object pins content and paths; the separate path-list digest is a quick inventory check. Neither is a built archive hash, license approval, or source-to-installed-file proof.

### Changelog review outline

A maintainer should first choose the previous release or comparison commit. A proposed entry should then enumerate exact numerical and matmul changes from Sprints 217–218, 32/64 index-width package changes from Sprint 219, the local-only workflow guard from Sprint 220, evidence tooling from Sprints 221–225, and any user-visible API changes. It must distinguish delivered code from blocked external evidence, hosted freshness, and timing results. The current `VERSION` value predates these sprints, so simply labeling all changes `2.2.0` without an approved version decision would be misleading. This outline is a review input, **not** an approved changelog entry.

## Proposed candidate scope and quality matrix

The rows below are the narrowest proposed static source scope consistent with the [current install support matrix](../../../../../INSTALL.md). They become *claimed candidate platforms* only after the release owner adopts them and exact-SHA evidence passes. The installed package remains a static archive plus headers and metadata; source extraction and clean-prefix consumers are required.

| Platform/toolchain | Width and installed first-use route | Required candidate gate | State |
| --- | --- | --- | --- |
| Linux Ubuntu / reviewed GCC | 32-bit default Make static install plus `pkg-config`; CMake `find_package(Sparse)` static consumer. Selected 64-bit functional and package-width checks are separate. | `.github/workflows/ci.yml`: `lint`, `cmake-build-and-test`, `package-contract`, `selected-index-width-64`; applicable static, docs, provider, and source-manifest guards. | Registered, **no candidate run**. |
| macOS / Apple Clang reviewed lane | 32-bit default Make/`pkg-config` and CMake static installs; selected 64-bit functional lane. Homebrew GCC is supplemental. | `.github/workflows/macos-ci.yml`: Apple Clang `build-and-test`, `install-and-pkgconfig`, `cmake-install-export`, `selected-index-width-64`. | Registered, **no candidate run**. |
| Windows / MSVC 2022 CMake subset | 32-bit default static CMake install and downstream consumer only. | `.github/workflows/windows-ci.yml`: `build-and-test`, `install-and-downstream`, and applicable PowerShell validation. | Registered, **no candidate run**. |
| Canonical timing / named macOS runner | Separate `s225-v1` `bench_refactor_csc`/`nos4` cell; no cross-platform timing inference. | Accepted three-job baseline plus exact-candidate `sprint225-timing.yml` decision and raw artifact review, or a documented product/release-owner scope decision retaining the blocker. | **Blocked:** [Sprint 225](../../SPRINT_225/RETROSPECTIVE.md) accepted no baseline or candidate timing. |

Required local candidate work includes `make format && make lint && make test` when C/header files change, `make quality-review-compile`, `make quality-review-cmake`, static install/consumer suites, source-archive reproduction, first-use walkthrough, and documentation/claim guards. The final [release checklist](../../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md) determines which checks are mandatory for the adopted platform scope. A historical workflow pass, selected-row freshness, or one local test does not close a candidate gate.

**Exclusions:** No shared-library artifact, dynamic ABI or cross-version compatibility, Windows Make/`pkg-config` parity, Windows 64-bit installed-width claim, public package provider, binary distribution, generated API HTML publication, broad performance superiority, or state-of-the-art claim. No tag, release upload, or public readiness statement is authorized by this Day 2 definition.

## Ownership, review, and rollback checkpoints

| Checkpoint | Required role | Current disposition |
| --- | --- | --- |
| Candidate SHA, version, changelog, tag/signing and final go/no-go | Release maintainer | **Unassigned / approval pending.** Repository ownership or PR authorship is not an approval. |
| Archive manifest, independent byte comparison, source-to-install map | Build/release owner and independent verifier | **Unassigned / evidence absent.** |
| Platform matrix and hosted job/artifact review | CI/platform owners for Linux, macOS, Windows | **Unassigned / exact-candidate jobs absent.** |
| Public header/ABI and installed first-use wording | API/ABI and documentation reviewers | **Unassigned / review pending.** |
| License, external claim and product scope | License/legal and product owners | **Unassigned / review pending.** |
| Cancellation, withdrawal and corrective-version procedure | Rollback owner and release maintainer | **Unassigned / destination and communication route undecided.** Preserve immutable tag and original asset digest if a later release is corrected. |
| Sprint 225 timing blocker disposition | Product and release owners | **Pending.** No silent waiver or timing-pass claim. |

Approvals must cite the selected exact SHA and the reviewed evidence packet. Until then, all checkpoints remain pending.

## Refreeze and evidence invalidation rule

1. Assign a candidate ID only to a reviewed commit whose intended source inputs, including the changelog, are committed. Record full commit SHA, Git tree, version, clean-checkout status, and approved platform scope.
2. If any tracked source, build, test, packaged documentation, version, metadata, or release manifest changes, mint a new candidate ID and repeat the two-archive comparison, source-to-install map, clean-prefix consumer runs, relevant local quality checks, hosted jobs, and first-use audit on that SHA. Previous run URLs remain historical evidence only.
3. A documentation-only change outside the archive still changes the commit; compare the tree and archive contents, record the difference, and obtain an explicit reviewer decision on which checks require rerun. Never reuse an exact-head hosted attestation for a different SHA without that decision.
4. Preserve failed and superseded logs, hashes, and decisions in the ledger. Do not overwrite an archive or move an approved immutable tag to conceal a failed candidate.

## Day 3 dependency

Before calling an archive the **first candidate archive**, resolve the missing changelog/version decision, identify the release and rollback owners, commit intended inputs, and select a clean reviewed SHA. If those conditions remain open, Day 3 may prepare archive tooling or run an exploratory build, but its candidate-release deliverable must remain blocked. This record is complete as a Day 2 **blocked freeze decision**, not as a release signoff.
