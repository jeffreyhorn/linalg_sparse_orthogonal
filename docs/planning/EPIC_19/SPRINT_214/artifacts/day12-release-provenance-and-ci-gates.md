# Sprint 214 Day 12: Release Provenance And CI Gates

## Decision And Evidence Boundary

Item 214.5 now has a complete **checklist design**, not a completed release.
Day 11 defined the candidate source/tag/archive contract; Day 12 adds the
source-to-binary chain, platform CI ownership, manual approvals, blockers,
and rollback. The selected product path remains static-first. Sprint 170
shared-library/dynamic-ABI and Sprint 207 package-provider deferrals are not
superseded. No release commit, tag, archive, binary asset, public provider,
or generated API HTML asset has been approved or published.

The authoritative gate is `../RELEASE_READINESS_CHECKLIST.md`. It requires
the tag/commit, two independently produced matching source archives and
SHA-256 hashes, approved manifest, extracted static build, installed-file
hashes, toolchain/dependency/flag record, and installed consumer result to
refer to the same candidate. A source checksum by itself cannot establish
source-to-binary traceability. The local Homebrew source formula proof and
CI build artifacts are neither release origins nor release packages.

## Required Candidate CI Evidence

| Platform/owner | Named job and required proof | Current evidence boundary |
| --- | --- | --- |
| Linux CI | `ci.yml` `package-contract`: Make install/`pkg-config` consumer, CMake install/export consumer, static package fixture tests, static/ABI docs guard, and provider guard. `ci.yml` `lint` and `cmake-build-and-test`: reviewed compile/CMake quality. | Jobs are registered. Day 12 has no candidate-specific hosted run URL or result. |
| macOS CI | `macos-ci.yml` `install-and-pkgconfig` and `cmake-install-export`: separate static install/consumer routes, fixture tests, and static guard. `build-and-test` owns reviewed Apple Clang compile/CMake quality. | Day 9-10 local installed-consumer results are scoped macOS evidence, not a tagged archive or hosted candidate proof. |
| Windows CI | `windows-ci.yml` `install-and-downstream`: MSVC CMake static `.lib` install, metadata validator, and out-of-tree consumer; `build-and-test` owns reviewed CMake/CTest subset. | Workflow registration is not hosted candidate evidence. Windows Make and `pkg-config` execution parity remain unclaimed. |
| Documentation/claims | Static package guard, local-only generated API guard, applicable docs check, and provider guard where package/claim surfaces change. Inspect source/release upload manifests for `docs/api/html/`. | Existing local-only policy remains; generated HTML is not a release artifact. |
| Shared/ABI decision | Confirm unsupported shared selectors, artifacts, loader metadata, and claims remain rejected. A future positive proof must separately exercise exports, ABI compatibility, loader, and installed dynamic consumers per platform. | Selected deferral path only; no shared proof is claimed or required for a static-source-only candidate. |

Every candidate job requires a run URL, commit SHA, runner/toolchain identity,
command log, exit status, and artifact digest if one is produced. A failed,
missing, or skipped required job is a stop, not a presumed pass. If any `*.c`
or `*.h` changes, `make format && make lint && make test` is additionally
required. Day 13 will run integrated local validation; it cannot substitute
for a future candidate's hosted evidence.

## Public Wording Reconciliation

`README.md` keeps the shortest static-first install summary and now says the
source instructions do not claim release readiness or a reproducible archive.
`INSTALL.md` explicitly excludes tagged releases, reproducible source
archives, binary release packages, and release readiness from current install
support, and links the maintainer checklist. `docs/maintainer_guide.md` points
reviewers to the same gates while retaining Sprint 170/207 non-claims. No
document offers a new download, release tag, provider install path, shared
selector, dynamic ABI, or hosted generated API HTML. The Sprint 213 local-only
generated documentation policy remains unchanged.

## Approval, Stop And Correction

Release, build, platform CI, documentation, API/ABI, legal/license, and product
owners must sign off on a specific candidate and public wording. A static
source-only release could be considered without claiming shared ABI or public
provider support, but that still requires all source, provenance, CI, and
approval gates. Binary packages, shared ABI, and provider distribution each
require a separate explicit product decision and named-platform proof.

Today the missing changelog, clean candidate, tag/signing decision,
reproducibility comparison, source-to-binary record, hosted candidate CI,
and manual approvals are blockers. If a published asset later proves wrong,
halt uploads, disable links, notify consumers, preserve the original digest,
and correct with a new version/tag rather than moving a tag or replacing
bytes. The release maintainer must approve the actual destination-specific
correction and communication procedure before publication. No rollback was
exercised on Day 12.

## Day 13 Handoff

Run the integrated installed static consumers, fixture negatives, deferral
and documentation guards, and applicable build quality checks. Record exact
commands, outcomes, and skipped gates. Keep this checklist in its
evidence-incomplete state even if local checks pass; no candidate release or
hosted CI run is established by Day 12 documentation work.

## Day 12 Local Validation

- `bash scripts/static_package_deferral_check.sh`: passed, including selector,
  static package metadata, and public non-claim checks.
- `bash scripts/check_api_docs_local_only.sh`: passed; no generated HTML was
  staged, tracked, or found in workflow publication paths.
- `make docs-check`: passed Doxygen generation and API coverage for the 18
  checked-in public headers. Its generated `docs/api/html/` output remains
  ignored local material, not a release asset.
- `make support-docs-guard`: passed.
- `python3 tests/test_package_manager_deferral_guard.py`: passed its fixture
  regressions. The full Homebrew-backed `make package-manager-deferral-guard`
  was not run on Day 12; it invokes the separate local formula proof and is
  not being cited as a passed provider check.
- `make format && make lint && make test`: not required; Day 12 changed no C
  or header file. Whitespace checks and final status are in `WORKING_NOTES.md`.
