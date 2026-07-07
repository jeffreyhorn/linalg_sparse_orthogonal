# Day 4 ABI Support Decision

## Purpose

Day 4 makes the Sprint 112 support-tier decision from the Day 2 package audit
and Day 3 ABI options. The decision freezes what the rest of Sprint 112 should
prove, and fences unsupported package or ABI claims before install/export
validation and documentation alignment begin.

## Decision Summary

| field | value |
|---|---|
| decision owner | Sprint 112 |
| decision sprint | Sprint 112 Day 4 |
| package shape before decision | static-first |
| package shape after decision | static-first |
| ABI promise | exact-version package metadata only; no dynamic ABI compatibility promise |
| affected platforms | Linux, macOS, Windows |
| claim state before work | static-first earned baseline; shared-library and ABI support unclaimed |
| claim state after work | static-first selected for validation; shared-library and ABI support remain non-claims |

## Decision

Decision:

> Preserve the maintained static-first package surface as the explicit Epic 10
> package support tier. Sprint 112 will refresh and strengthen static install,
> export, package metadata, and downstream consumer proof. It will not add or
> claim shared-library packaging or dynamic ABI stability.

Reason:

> The current implementation, build files, install scripts, CMake package
> metadata, pkg-config metadata, CI comments, README, INSTALL guide, and
> maintainer guide all describe a static-first package story. Day 3 found no
> maintained shared artifacts, no runtime-loader proof, no symbol visibility or
> export policy, no SONAME/SOVERSION policy, and no reviewed shared-library
> platform lanes. The reliable sprint path is to validate the already earned
> static-first support tier rather than create a broad ABI claim without the
> required proof.

## Current Contract After Decision

| field | current value |
|---|---|
| library artifact | static archive `libsparse_lu_ortho.a` |
| package version behavior | single-sourced from `VERSION`; CMake package compatibility is `ExactVersion`; pkg-config reports the same version |
| shared-library output | not maintained and not claimed |
| runtime-loader proof | none required for the selected static-first tier |
| symbol/export policy | no ABI symbol baseline or export policy claimed |
| cross-platform install proof | Unix-side Make/CMake install scripts plus platform-tier interpretation; Windows remains CMake-first reviewed subset |

## Selected Static-First Proof Requirements

| proof area | required evidence | owner |
|---|---|---|
| static archive install | staged Make install verifies `libsparse_lu_ortho.a`. | Day 6 |
| no unexpected shared artifacts | install proof checks no `.so`, `.dylib`, `.dll`, or similar shared artifacts. | Day 6 / Day 7 |
| public installed headers | install proof checks public headers plus generated `sparse_version.h`. | Day 6 / Day 7 |
| pkg-config metadata | `pkg-config --cflags`, `--libs`, and `--modversion` checks against staged install. | Day 6 |
| pkg-config consumers | generated consumer and maintained example compile, link, and run through installed flags. | Day 6 / Day 8 |
| CMake install/export | staged CMake install verifies config/version/targets files and static target export. | Day 7 |
| CMake consumer | `examples/cmake_example/` configures, builds, links, and runs through `find_package(Sparse)`. | Day 7 / Day 8 |
| exact-version behavior | exact installed version succeeds and mismatched lower version is rejected when applicable. | Day 7 |
| documentation non-claims | public and maintainer docs preserve shared-library, ABI, Windows, and macOS boundaries. | Day 12 |

## Rejected Support Paths

| rejected path | reason |
|---|---|
| shared-library package support in Sprint 112 | Requires new build rules, package metadata, runtime-loader validation, symbol policy, platform ownership, and documentation work outside current evidence. |
| dynamic ABI compatibility claim | No binary compatibility policy, symbol baseline, SONAME/SOVERSION policy, or reviewed runtime lanes exist. |
| ABI stability inferred from Sprint 110 public-header drift | Header stability is useful source/package evidence but not proof of binary compatibility. |
| Windows install-validation parity | Windows reviewed scope remains CMake-first consumer subset only. |
| macOS reviewed install/export parity | macOS static install/pkg-config proof remains supplemental confidence, not full reviewed parity. |

## Documentation Queue

These surfaces should be revisited only after Days 5-11 refresh the selected
proof:

| doc surface | required wording check |
|---|---|
| `INSTALL.md` | Static-first contract, package verification commands, platform tier caveats, and non-claims. |
| `README.md` | Compact package summary and link to `INSTALL.md`; avoid maintainer-proof overload. |
| `docs/maintainer_guide.md` | Detailed proof ownership, reviewed versus supplemental platform lanes, and ABI non-claims. |
| `CMakeLists.txt` comments | Static-first and exact-version comments remain accurate after validation. |
| `sparse.pc.in` | Metadata remains static package metadata and does not imply ABI compatibility. |
| CI workflow comments | Platform scope remains accurate if Days 9-11 change reviewed or supplemental lanes. |

## Non-Claims After Decision

After Day 4, the project still does not claim:

- shared-library package support;
- dynamic ABI stability;
- SONAME/SOVERSION compatibility;
- symbol export stability;
- runtime-loader support for installed shared artifacts;
- ABI stability inferred from public-header stability;
- Windows Makefile parity;
- Windows separate reviewed install-validation lane;
- macOS full reviewed install/export parity;
- package support beyond the exact static-first install/export surfaces that
  Sprint 112 validates.

## Follow-Up Work

| follow-up | owner | reason |
|---|---|---|
| Install/consumer proof design | Day 5 | Convert this decision into concrete Make, CMake, pkg-config, and consumer commands. |
| Make install proof | Day 6 | Refresh Unix-side static install, pkg-config, and uninstall evidence. |
| CMake install/export proof | Day 7 | Refresh installed CMake package and exact-version evidence. |
| Downstream consumer proof | Day 8 | Confirm installed public-header consumer behavior stays coherent. |
| Platform-tier contract | Day 9 | Align Linux, macOS, and Windows wording with selected package tier. |
| Documentation alignment | Day 12 | Update public and maintainer docs from refreshed evidence. |

## Completion Criteria Status

- The support-tier decision is evidence-backed and reviewable.
- Unsupported ABI and package claims are fenced before implementation.
- Install and consumer proof work is scoped to the selected static-first tier.
