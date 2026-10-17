# Sprint 214 Day 2: ABI And Release Scope Decision

## Decision

**Selected path: complete ABI/release design with stronger static-first
deferral guards.** Sprint 214 will not add a shared-library target, install a
shared artifact, or promote dynamic ABI or release readiness. Days 3-5 will
define the policy and exact future acceptance tests; Days 6-8 will strengthen
the existing configure/install/metadata/docs guard boundary; Days 9-10 will
retain installed static consumer evidence and specify future shared-consumer
fixtures. Days 11-14 will complete the release checklist and validation.

This is one of the two outcomes authorized by the Sprint 214 project plan. It
preserves, rather than supersedes, the Sprint 170 static-first product decision
and the Sprint 207 public package-provider deferral. No additional product
approval is required to *document and guard* this existing boundary. A later
request to ship a shared library or make a release-readiness claim must be an
explicit new product decision backed by the gates below.

## Evidence Used

| Input | Decision consequence |
| --- | --- |
| Sprint 170 Day 2/4/8/9 artifacts | Many public structs expose layout, library-owned allocations and callbacks cross public boundaries, `SPARSE_IDX_BITS` changes signatures/layouts, and the old archive inventory counted 359 global symbols. That historical count is not a current export allowlist. |
| Current `CMakeLists.txt` | `BUILD_SHARED_LIBS=ON` fails configure, `sparse_lu_ortho` is an explicit `STATIC` target, installed metadata exports a static imported target, and package-version matching is `ExactVersion`, not a dynamic ABI promise. |
| Current `Makefile` and `sparse.pc.in` | Make builds and installs `libsparse_lu_ortho.a`; `sparse.pc` describes the static archive. No maintained shared Make target or static/shared package selector exists. |
| Install and guard tests | `tests/test_install.sh`, `tests/test_cmake_install.sh`, and `scripts/static_package_deferral_check.sh` enforce static artifacts/consumers and reject shared artifacts, metadata, or unsupported claims. The static guard passed locally on Day 1. |
| Reviewed CI ownership | Linux and macOS have reviewed static package-contract lanes; Windows has CMake static install/downstream proof only. No reviewed shared-loader/consumer lane exists on any platform. |
| Sprint 207 decision and Epic 18 `E18-RQ-009` | Public provider distribution and release/shared/ABI readiness are residual product work, not implications of a working local source install. |

## Option Comparison

| Factor | Narrow shared proof now (rejected) | ABI/release design with strengthened deferral (selected) |
| --- | --- | --- |
| Candidate scope | The narrowest plausible candidate is Linux/GCC/CMake, separate from existing static targets and without macOS/Windows claims. | Keep the existing static Make/CMake package and platform evidence tiers unchanged. |
| Build/export effort | Requires PIC/shared target, deliberate export list and hidden-by-default behavior, install/export coexistence, and package metadata policy. Current broad public layout and uncurated symbols make even a small dynamic subset nontrivial. | Preserve explicit `STATIC` target and shared configure rejection; add targeted negative fixtures for accidental selectors, artifacts, metadata, and claim drift. |
| Loader/consumer effort | Requires Linux SONAME/RPATH policy and inspection, installed out-of-tree dynamic consumer execution, missing-library failure tests, and hosted CI lane. A successful link alone would not establish compatibility. | Specify exact future loader/consumer tests while continuing to execute the installed static consumers. |
| Compatibility risk | Source/package version `2.2.0` and CMake `ExactVersion` do not define ABI epoch or cross-version behavior; concrete structs and callbacks remain exposed. | Days 3-5 can define scope, version and break policy, ownership, and per-platform metadata before implementation. |
| Sprint fit | The 38-hour implementation and 28-hour consumer allocations would need to absorb export curation, runtime behavior, test isolation, and CI without an established ABI scope. That is not a credible promotion gate today. | The same allocations can produce enforceable deferral and executable future acceptance specifications without a partial shared product. |
| Claim risk | A single Linux proof could be mistaken for broad shared packaging, ABI stability, other-platform support, or release readiness. | Public docs continue to state exactly what static install paths are supported and what remains deferred. |

The narrow option is rejected for this sprint because the required proof stack
is not merely a build-system toggle. A local shared file would be weak evidence
until the ABI subset, exports, loader, installed consumer, and hosted checks are
defined and pass together. A shared proof can be reopened after the policy
design and named-platform evidence are reviewed.

## Selected Acceptance Gates

| Gate | Required evidence and owner | Failure/stop condition |
| --- | --- | --- |
| Decision trace | This record and `WORKING_NOTES.md` select only design/deferral; Sprint 170 and 207 decisions remain cited. | Any text treats this decision as shared or release promotion. |
| ABI policy | Days 3-5 document public dynamic subset versus excluded declarations, symbol visibility/allowlist, allocation/callback/layout/index-width rules, ABI version and break policy, and platform loader expectations. | Policy leaves an apparent broad ABI guarantee or an unowned platform. |
| Configure boundary | `CMakeLists.txt` keeps `BUILD_SHARED_LIBS=ON` as an actionable configure failure; `scripts/static_package_deferral_check.sh` validates it and flags new unsupported shared selectors/targets. | A shared request configures successfully or is silently ignored as a supported choice. |
| Build/install boundary | Make/CMake keep the static archive and static imported target; install tests reject `.so`, `.so.*`, `.dylib`, `.dll`, shared import metadata, and unselected loader fields. | A shared artifact or target becomes installed without a new decision and complete proof. |
| Package metadata | `sparse.pc.in`, CMake package files, and generated installed metadata remain static-scoped; no static/shared component or selector is advertised. | Metadata describes a link mode or runtime loader behavior not actually supported. |
| Installed consumers | `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh` continue to build/run out-of-tree static consumers where supported. Day 9/10 specify exact future dynamic consumer positive/negative fixtures. | Static consumer regresses, or specified future tests are described as executed shared proof. |
| Platform/CI boundary | Linux/macOS reviewed static package lanes and Windows CMake static lane remain the stated scope; guard wiring is reviewed for selected changes. | Workflow comments or jobs imply shared-loader or platform parity proof that was not executed. |
| Docs/non-claims | README, INSTALL support matrix, maintainer guide, and relevant guards continue to reject shared, dynamic ABI, loader, release, and broad package claims. | A positive claim appears before matching artifact and CI evidence. |
| Release checklist | Days 11-12 name changelog, tag, reproducible source archive, artifact checksums/provenance, CI evidence, ownership, and blockers as future gates. | A checklist is described as a completed release or reproducibility proof. |
| Validation | Days 13-14 run affected guards, install/export tests, docs checks, and relevant build quality paths; `make format && make lint && make test` is mandatory if any `*.c` or `*.h` changes. | Required check fails or is skipped without an explicit reason. |

The implementation owner may strengthen an existing check or add a focused
regression; it must not weaken the static contract just to make a proposed
shared test pass. The Day 1 package-provider guard was interrupted during a
Homebrew CMake source build, so it is not counted as a passing baseline. If
provider wording or metadata changes, rerun it in a prepared environment or
record the blocker before closing validation.

## Exact Future Shared-Proof Acceptance Tests

These are **specifications, not tests already implemented or passed**. Days
3-5 may refine their fixtures and expected outputs, but may not remove a gate
without an explicit replacement decision.

1. **ABI subset and exports:** choose a named public header/function subset;
   build with hidden visibility by default; inspect the shared artifact's
   dynamic symbol table and compare it with a checked-in allowlist. Fail on
   missing required exports or extra internal symbols.
2. **Layout and ownership:** compile a consumer against installed headers and
   verify index-width, scalar type, callback calling convention, and
   allocation/free contract for every selected ABI object. Excluded concrete
   structs must not acquire an accidental compatibility promise.
3. **Version/compatibility:** define an ABI epoch distinct from `VERSION` and
   the CMake exact package-version rule. Use retained old/new artifacts and a
   cross-version installed consumer to test each compatibility claim; reject
   mismatched ABI or absent metadata. Until then, no cross-version claim.
4. **Linux candidate:** on a named GCC/Ubuntu lane, inspect SONAME and needed
   libraries, install into a temporary prefix, link and execute an out-of-tree
   dynamic consumer with only the installed target, and demonstrate expected
   failure when the shared object or runtime path is removed.
5. **macOS expansion:** separately inspect install-name and RPATH on a named
   Apple Clang lane; link/run an installed consumer and test relocation or
   missing-library behavior before claiming macOS shared support.
6. **Windows expansion:** separately inspect DLL/export table and import
   library on a named MSVC lane; link/run an installed consumer under the
   intended DLL search/CRT policy and test missing-DLL behavior before
   claiming Windows shared support.
7. **Static coexistence and metadata:** retain Make/CMake static consumers,
   define target/component/`pkg-config` selection semantics, and verify
   generated package files name only installed artifacts. No package-manager
   distribution follows automatically.
8. **Hosted/claim gate:** wire selected platform tests into reviewed CI and
   inspect hosted outputs before changing support matrix or README wording.
   A Linux-only proof stays Linux-only and does not establish general ABI
   compatibility or release readiness.

## Non-Claims And Stop Conditions

Sprint 214's selected path does not claim shared-library build/install,
dynamic ABI compatibility, stable exported symbols, runtime-loader support,
Linux SONAME, macOS install-name/RPATH, Windows DLL/import-library behavior,
cross-version compatibility, broad platform parity, public package-manager
distribution, reproducible release artifacts, or release readiness. Source
compatibility, static package exact-version behavior, and a completed release
checklist must not be substituted for those claims.

Stop and seek a new explicit product decision if implementation needs to add a
shared target/artifact, widen the public support tier, remove a static
regression, change the Sprint 170 or 207 product posture, or publish a release
artifact. Stop validation and report a blocker if a required guard, installed
consumer, docs check, or changed-C quality gate fails; do not record a pass by
skipping it.

## Day 2 Outcome

Item 214.1 is complete as a scope decision. Days 3-5 may design ABI/release
policy, and Days 6-10 may strengthen deferral and consumer acceptance
specifications under this selected path. No code or public support claim is
changed by this Day 2 planning artifact.

## Day 2 Validation

`git diff --check` and new-file whitespace checks passed for the Sprint 214
Markdown files. Day 2 changed planning documentation only; no `*.c` or `*.h`
files changed, so the full C quality gate was not required. The Day 1 static
package deferral guard remains the current locally passing baseline; it was
not represented as a new hosted or dynamic-consumer result.
