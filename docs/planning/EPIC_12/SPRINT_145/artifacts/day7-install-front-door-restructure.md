# Sprint 145 Day 7 INSTALL Front-Door Restructure

## Purpose

Simplify `INSTALL.md` around the first-use static install and downstream
consumer paths while preserving the exact package, ABI, and platform support
boundaries inherited from Sprint 144.

## Changed Surface

| Surface | Change | Owner |
| --- | --- | --- |
| `INSTALL.md` `Start Here` | Replaced question bullets with a five-step install/setup ladder. | Install front door |
| `INSTALL.md` `Support Split` | Clarified which surfaces own build-tree adoption, Unix `pkg-config`, installed CMake, and reviewed-platform interpretation. | Install support routing |
| `INSTALL.md` `Quick Start (Makefile)` | Added the downstream `pkg-config` compile command near the Unix static install path. | Unix static package consumer |
| `INSTALL.md` `Maintained Install Contract` | Reframed the section as the package-shape claim boundary, separate from command-only first use. | Static-first package contract |
| `INSTALL.md` `CMake Build` | Added a concise installed CMake consumer lead-in and explicit Windows CMake-first note. | CMake install/export consumer |
| `INSTALL.md` `Platform Notes` | Clarified that platform notes do not widen reviewed platform tiers. | Platform support interpretation |
| `INSTALL.md` `Verifying the Installation` | Clarified when install validation scripts should be run. | Local install proof |

No source files, public headers, build rules, package metadata, or CI workflow
files were changed.

## First-Use Install Path

| Step | INSTALL route | Boundary |
| --- | --- | --- |
| Prove local build first | `README.md#start-here`, `examples/README.md#start-here` | Build-tree adoption before install. |
| Install static package | `Quick Start (Makefile)` or `CMake Build` | Static archive package surface only. |
| Consume downstream | `pkg-config` or `find_package(Sparse)` | Both describe the installed static archive. |
| Validate install surface | `tests/test_install.sh`, `tests/test_cmake_install.sh` | Local Unix-side proof for install/export behavior. |
| Interpret platform coverage | `Supported platforms` | Linux/macOS/Windows support tiers remain differentiated. |

## Claim Boundary Review

| Area | Day 7 result |
| --- | --- |
| Static-first package | Preserved as the maintained install/export contract. |
| CMake install/export | Preserved as a static archive export through `Sparse::sparse_lu_ortho`. |
| `pkg-config` | Preserved as a Unix-side downstream consumer route and install validation surface. |
| Shared libraries and ABI | Explicitly remain deferred; no shared-library, dynamic ABI, runtime-loader, or static/shared selector support was added. |
| Package managers | Explicitly remain out of scope; no Homebrew, apt, dnf, vcpkg, Conan, or distribution claim was added. |
| Platform tiers | Linux remains strongest reviewed source of truth; macOS keeps reviewed static-first install/export proof; Windows remains CMake-first with no Makefile or `pkg-config` parity claim. |
| Reports and benchmarks | No benchmark/report wording changed; install proof remains distinct from runtime performance evidence. |

## Validation

| Check | Result |
| --- | --- |
| INSTALL anchor scan | Passed |
| install-doc unsupported-claim scan | Passed: matches are explicit non-claims or support boundaries |
| `git diff --check` | Passed |
| untracked artifact whitespace scan | Passed |
| `.c` / `.h` changed-file scan | Passed: no paths |

`make format && make lint && make test` was not required because Day 7 changed
only documentation.

## Day 7 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| INSTALL is easier to follow for first-use static install and downstream consumption. | Complete | `Start Here` now gives a five-step path and the Make/CMake sections include immediate consumer commands. |
| Platform support-tier wording remains consistent with Sprint 144. | Complete | Linux, macOS, and Windows tiers remain explicitly differentiated. |
| Install-doc validation and claim scans pass. | Complete | Anchor, unsupported-claim, whitespace, and changed-file scans passed. |

## Day 8 Handoff

Day 8 should align solver-selection and diagnostics front-door wording with the
README, examples, cookbook, and INSTALL ladders. Keep solver-family choices
short at the front, then route QR, partial-SVD, runtime/backend, benchmark, and
report evidence into their deeper owners.
