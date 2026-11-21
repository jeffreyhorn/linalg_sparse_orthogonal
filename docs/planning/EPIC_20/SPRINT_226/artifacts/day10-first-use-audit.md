# Sprint 226 Day 10: Completed First-Use Audit

## Result and scope

The maintained static **source-to-installed-consumer instructions work locally** on the Day 3 exploratory base archive. The [Day 9 command transcript](day9-clean-prefix-first-use.md) records a fresh extraction, clean Make and CMake prefixes, default 32-bit and selected 64-bit consumers, 63/63 CTests for the default CMake build, three static package contract passes, runtime output, and retained hashes. This is local macOS procedure proof. It is **not** a release-candidate result: the archive is from base commit `c42d6715fad7e895bae4941a006df3563f1afc38`, while the Day 9–10 documentation corrections are still uncommitted and absent from that archive.

## Public entry to installed first solve

| Step | Current route and observed check | Disposition |
| --- | --- | --- |
| Public entry | [README Start Here](../../../../../README.md#start-here) routes to the [examples Start Here](../../../../../examples/README.md#start-here), then to the staged [INSTALL Start Here](../../../../../INSTALL.md#start-here). README's older `/usr/local` CMake snippet was replaced by links to the maintained staged route. | **Usable locally**; current Markdown links resolve. |
| First build-tree solve | `make`, `make examples`, and `build/example_basic_solve` from the fresh extraction yielded solution `[1,1,1,1,1]` and residual `0`. | **Pass** on local exploratory source. |
| Make installed consumer | [INSTALL Unix tutorial](../../../../../INSTALL.md#unix-makepkg-config-consumer) installs into a writable prefix. The copied tutorial `main.c` compiled with `pkg-config --cflags/--libs sparse`, resolved the exact Make prefix, and reported width 32, version 2.2.0, `nnz: 1`, `OK`. | **Pass** on local exploratory source. |
| CMake installed consumer | [INSTALL CMake build](../../../../../INSTALL.md#cmake-build) and [consumer](../../../../../INSTALL.md#cmake-consumer) use a separate build tree and prefix. The maintained `examples/cmake_example` solved `[1,1,1]` at both width 32 and width 64 from separate installed prefixes. The exact tutorial `CMakeLists.txt` plus copied `main.c` also built and ran against the width-32 CMake prefix. | **Pass** on local exploratory source. |
| Public API and installed headers | [API reference](../../../../../docs/api_reference.md#source-of-truth) lists all 18 checked-in public `.h` files and explicitly documents the generated `sparse_version.h`. All 19 headers are present in each Day 9 Make, CMake-32 and CMake-64 prefix. | **No missing or undocumented installed header** in this scope. |

The Make install has `sparse.pc` and **no** CMake export. Both CMake installs have `sparse.pc` and four CMake export/config files. [README Installation](../../../../../README.md#installation) and [INSTALL's contract and installed-file table](../../../../../INSTALL.md#installed-files) now state that distinction, preventing a user from trying `find_package(Sparse)` against a Make-only prefix. The README's CMake section now links to the staged-prefix route instead of presenting a second `/usr/local` command sequence. The full downstream compile command remains in one tutorial location.

## Version, width, package and claim checks

| Check | Observed result |
| --- | --- |
| Version coherence | Root `VERSION` is `2.2.0`. All three installed `sparse_version.h` and `sparse.pc` files report `2.2.0`; both CMake prefixes' `SparseConfigVersion.cmake` files report `2.2.0`. Make correctly has no CMake version file. An exact `find_package(Sparse 2.2.0 EXACT REQUIRED)` configured successfully; `2.2.1 EXACT` failed with the expected not-accepted version diagnostic. Ignored logs: `build/sprint226/day10/version-accept.log` SHA-256 `9f7db015b5c21915e617a241f64732c30114f2f8b17ae88979cd9cf66e82f362`; reject log SHA-256 `921f610e36d2766611117b4681bdd3a946bb0102c07699b2c53816c01c7b4163`. |
| Index width | Make and default CMake headers/`sparse.pc` report 32; selected CMake headers/`sparse.pc` report 64. Each corresponding consumer printed the selected runtime width and used its own prefix. Width mismatch rejection was previously exercised in the [Day 6 consumer proof](day6-local-installed-consumers.md), on the same exploratory base source. |
| Static-only package | Each Day 9 prefix has one static archive; no `.so`, `.dylib`, or `.dll` was present. Make package has no CMake export; CMake package has four config/export files. `check_static_install_contract.py` passed for all three prefixes. No shared ABI or provider support follows. |
| External and timing claims | README and INSTALL now distinguish Sprint 222's two-sided **local** numerical execution from an **accepted/published** CHOLMOD result. The latter remains blocked by provenance/rights review. Sprint 225's thresholded timing gate remains blocked by a missing accepted baseline; selected freshness is not a speed verdict. |
| Platform scope | Local walkthrough was on macOS 15.7.9 x86_64 with Apple Clang 17. The [Day 8 hosted matrix](day8-platform-provenance-and-first-use-inventory.md) contains historical base runs only. No Linux, macOS, or Windows release-candidate platform result exists. Windows remains a CMake/MSVC subset, without Make or `pkg-config` execution parity. |

## Documentation defects and verification

The route-affecting defects inventoried on Day 8 are resolved in the working tree: stale external-baseline status, duplicate/system-prefix install shortcuts, mismatched CMake consumer prefix, and ambiguous Make-versus-CMake export wording. No C, header or build rule was changed. The corrected text needs to enter a reviewed candidate archive before its package-level documentation provenance can pass.

The scoped local Markdown audit checked README, INSTALL, examples README and API reference: respectively **94, 19, 27, 12** relative file links and **48, 17, 12, 1** anchor links. No missing target or unmatched anchor was found. External URLs and generated HTML were outside this local-link pass. After the Day 10 edits, `make support-docs-guard`, `bash scripts/static_package_deferral_check.sh`, `python3 tests/test_api_docs_routing.py`, `python3 tests/test_package_manager_deferral_guard.py`, and `API_DOCS_PYTHON=build/sprint226/day7/api-venv/bin/python bash scripts/check_api_docs_local_only.sh` passed. The API local-only guard checked 123 workflow job/step records. Full optional Homebrew provider proof remains the **incomplete** Day 7 result; its focused policy test passing here does not close that aggregate gate. `git diff --check` and Markdown whitespace checks passed.

## Candidate handoff

The first-use route is locally usable, but the plan's candidate-specific completion criterion is **blocked**. The release maintainer must approve version/changelog and a clean committed candidate containing these docs; build/release owner must rebuild both candidate archives and install map; platform owners must rerun exact-SHA hosted consumers and quality; documentation owner must recheck links and packaged docs from that candidate. No Day 3–9 exploratory hash is promoted after these text edits.
