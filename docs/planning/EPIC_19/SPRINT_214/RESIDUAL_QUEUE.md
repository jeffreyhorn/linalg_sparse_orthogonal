# Sprint 214 Residual Handoff

Sprint 214 closes the ABI/release *decision* with a complete design and
stronger static-first deferral guards. This queue is the input to the Epic 19
final residual review in Sprint 216, not an approved shared-library or
release roadmap. Reopen a product scope only with an explicit decision.

| Priority | Residual | Closure evidence required | Current boundary |
| --- | --- | --- | --- |
| 1 | Candidate-specific static source release | Changelog and reviewed clean commit; approved immutable tag/signing policy; full tracked-source manifest; two independent matching archive byte hashes and extraction/build; source-to-installed-file provenance; candidate-specific Linux/macOS/Windows jobs for every claimed platform; owner approvals and rollback plan. Use `RELEASE_READINESS_CHECKLIST.md`. | Checklist design complete. No candidate, tag, archive, reproducibility comparison, hosted candidate run, or release approval. |
| 2 | Named-platform shared-library product decision | Select an ABI subset and first platform/toolchain; approve separate static/shared target and package selector semantics; retain existing static consumers. A proposed Linux proof must pass export, loader, install, and executed out-of-tree dynamic consumer gates before any Linux claim. | No shared target or artifact. `BUILD_SHARED_LIBS=ON` and unsupported selectors remain rejected. |
| 3 | Dynamic ABI and compatibility policy proof | Review the provisional eight-function opaque-matrix/metadata subset in the Day 3 artifact; curate an exact export allowlist with hidden-by-default symbols; specify object ownership, callback and struct layout, index width and scalar variants, ABI epoch and break policy; retain an old executable/binary baseline and run cross-version consumers for any compatibility promise. | Policy design only. Package `VERSION=2.2.0` and CMake `ExactVersion` are not an ABI epoch or compatibility proof. |
| 4 | Per-platform runtime-loader and installed consumer proof | Linux: inspect ELF SONAME, dependencies and search path. macOS: inspect Mach-O install-name/RPATH and relocation. Windows: inspect PE DLL exports, import library, DLL search and CRT. On each claimed platform, build/install, link and execute a consumer, then prove missing/wrong library and metadata diagnostics in hosted CI. | No shared loader/consumer test executed. Existing Windows proof is CMake/MSVC static only; Windows Make and `pkg-config` execution parity remain unclaimed. |
| 5 | Hosted branch and release evidence | Capture Linux/macOS/Windows run URL, commit, runner/toolchain, logs, result and artifact digest for the exact branch or candidate. Reconcile actual hosted failures before status promotion. | Workflow registrations and Day 13 local passes are not hosted results. |
| 6 | Separate distribution/publication choices | Reopen Sprint 207 public package-provider and Sprint 213 generated API publication decisions only with their own destination, retention, provenance, install or routing, rollback, and claim guards. Binary packages need a separate product decision and per-platform provenance. | Developer-mode local Homebrew source proof and ignored `docs/api/html/` do not authorize public distribution or generated HTML publication. |

The current earned scope is a static archive source install. Unix Make
`pkg-config` and CMake `find_package(Sparse)` installed consumers passed
locally on Day 13. Reviewed Linux/macOS static workflow lanes and a Windows
CMake static lane exist, but this sprint has not recorded branch-specific
hosted results. No shared ABI, cross-version, broad platform, binary package,
provider, reproducible release, or release-readiness claim follows.
