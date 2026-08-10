# Sprint 147 Day 10 ABI And Package Evidence Gate

## Purpose

Day 10 defines the evidence gate for the Sprint 153 shared-library ABI product
decision. Sprint 153 may either implement the first supported shared-library
surface or publish a stronger tested static-first deferral. In both cases,
shared-library support, dynamic ABI compatibility, runtime-loader behavior,
package-manager distribution, and static/shared selector support must remain
non-claims unless executable proof and documentation support them.

## Current Static-First Baseline

| Surface | Current Evidence | Current Boundary |
| --- | --- | --- |
| CMake configure | `CMakeLists.txt` rejects `BUILD_SHARED_LIBS=ON`. | Shared-library packaging and dynamic ABI support are deferred. |
| Deferral guard | `scripts/static_package_deferral_check.sh` checks rejection wording, static package metadata, no shared ABI metadata, no selector wording, and docs boundaries. | Prevents accidental shared-library or ABI implication. |
| Make install | `tests/test_install.sh` validates static archive install, headers, `pkg-config`, downstream compile/link/run, uninstall, and no shared artifacts. | Unix-side static archive package proof. |
| CMake install/export | `tests/test_cmake_install.sh` validates static archive install/export, `find_package(Sparse)`, exact/mismatch version behavior, downstream compile/link/run, static metadata, and no shared imported metadata. | Local static CMake package proof. |
| Linux CI | `.github/workflows/ci.yml` carries reviewed static-first package-contract proof. | Strongest reviewed source-of-truth package lane. |
| macOS CI | `.github/workflows/macos-ci.yml` carries reviewed static-first Make install/`pkg-config` and CMake install/export proof. | Reviewed macOS static archive package proof. |
| Windows CI | `.github/workflows/windows-ci.yml` carries supplemental CMake install/downstream confidence. | CMake-first supplemental confidence, not separate reviewed install-validation parity until Sprint 149 decides it. |

## Product Decision Options

| Option | Meaning | Minimum Evidence |
| --- | --- | --- |
| Implement shared-library support | Add supported shared build/install/export/downstream/loader behavior on selected platforms. | Public symbol/visibility policy, build rules, package metadata, downstream consumers, loader checks, docs, CI proof, and rollback plan. |
| Strengthen static-first deferral | Keep shared support unsupported, but improve rejection diagnostics, guards, docs, and downstream static proof. | Configure rejection, static metadata checks, no shared artifact checks, package docs, local install/export proof, and relevant hosted proof. |
| Reject package-manager distribution for now | Preserve package-manager support as a residual behind ABI/release mechanics. | Explicit residual with blockers: ABI/product posture, recipe ownership, versioning, update/uninstall proof, and CI install proof. |

Sprint 153 must make the decision explicitly. It must not leave
`BUILD_SHARED_LIBS=ON` behavior ambiguous.

## Shared-Library Implementation Gate

Shared-library support may be claimed only if all selected platform requirements
are satisfied.

| Evidence Area | Required Proof |
| --- | --- |
| Public symbol inventory | List exported public functions, version symbols, internal-only symbols, and symbols intentionally hidden. |
| Header/API audit | Confirm public headers, structs, enums, typedefs, callbacks, allocator hooks, and macros are compatible with the selected ABI policy. |
| Visibility/export policy | Define Linux/macOS visibility attributes and Windows `__declspec(dllexport/dllimport)` or equivalent handling without breaking static consumers. |
| Version and ABI policy | Define package version, ABI epoch or compatibility rule, soname/install-name/DLL naming if supported, and compatibility test scope. |
| Build rules | CMake and, if selected, Makefile rules build static and shared artifacts intentionally with no accidental selector semantics. |
| Install/export metadata | Installed CMake package and `pkg-config` metadata describe the selected static/shared surface without unsupported claims. |
| Loader proof | Downstream executable loads the installed shared artifact from the installed prefix on each supported platform. |
| Downstream consumers | CMake `find_package(Sparse)`, exact-version, mismatch-version, and simple compile/link/run consumers pass for selected shared/static modes. |
| Platform CI | Hosted Linux/macOS/Windows proof is recorded for every platform claimed. |
| Documentation | README, INSTALL, maintainer guide, package docs, and workflow comments describe support tier and non-claims consistently. |

If support is implemented for only one or two platforms, wording must name only
those platforms and keep the remaining platforms deferred.

## Stronger Static-First Deferral Gate

If Sprint 153 chooses deferral, it should strengthen the current static-first
contract rather than merely restate it.

Required proof:

- `BUILD_SHARED_LIBS=ON` fails at configure time with wording that names the
  static package contract, shared-library deferral, dynamic ABI deferral, and
  missing loader/downstream requirements.
- Install tests continue to reject `.so`, `.dylib`, `.dll`, import-library, and
  shared imported-target metadata.
- CMake package metadata continues to export the static archive target only.
- `sparse.pc` continues to describe static archive package metadata and carries
  no `Libs.private`, shared, soname, ABI, package-manager, or selector wording
  unless a later product decision supports it.
- `scripts/static_package_deferral_check.sh` protects the deferral wording and
  unsupported metadata boundaries.
- README, INSTALL, and maintainer guide keep shared-library support, dynamic
  ABI compatibility, runtime-loader behavior, package-manager support, and
  static/shared selector support as explicit non-claims.

## Package Metadata Checklist

| Metadata Surface | Static-First Requirement | Shared-Support Requirement If Implemented |
| --- | --- | --- |
| CMake target | Static archive target only. | Separate static/shared targets or selector policy with explicit support semantics. |
| `SparseConfig.cmake` | No shared imported metadata, ABI promise, or selector claim. | Correct imported target type, location, include dirs, transitive dependencies, and version policy. |
| `SparseConfigVersion.cmake` | Exact/mismatch version behavior remains tested. | Same, plus ABI/version compatibility policy if supported. |
| `sparse.pc` | Static archive description, `Libs` self-contained for current surface, no `Libs.private`. | Shared/static dependency split only if dependency visibility and selector behavior are supported and tested. |
| Installed artifacts | Static archive, public headers, CMake package files, and `sparse.pc`; no shared artifacts. | Named shared artifacts, import libraries if applicable, static artifacts if retained, and no unsupported extras. |
| Uninstall/update behavior | Static files removed by maintained uninstall proof where available. | Shared/static files, metadata, and loader-visible artifacts handled intentionally. |

## Downstream Consumer Validation

Static-first validation requirements:

```sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
bash scripts/static_package_deferral_check.sh
```

Shared-library validation requirements if implemented:

- configure/build/install selected shared mode;
- compile and run a downstream CMake consumer against installed shared target;
- compile and run a downstream `pkg-config` consumer where supported;
- validate exact-version and mismatched-version behavior;
- validate loader/runtime behavior from the installed prefix;
- validate no unsupported artifacts or metadata appear;
- validate static mode still works if static support is retained;
- run hosted CI for every platform claimed.

Any `.c` or `.h` changes require:

```sh
make format && make lint && make test
```

## Platform Validation Requirements

| Platform | Static-First Baseline | Shared Support Gate |
| --- | --- | --- |
| Linux | Reviewed package-contract lane plus local install/export scripts. | Hosted shared build/install/export/downstream/loader proof before Linux shared claim. |
| macOS | Reviewed static-first Make install/`pkg-config` and CMake install/export proof. | Hosted `.dylib` install-name/loader/downstream proof before macOS shared claim. |
| Windows | Reviewed CMake subset plus supplemental CMake install/downstream confidence; Sprint 149 owns install-validation parity decision. | Hosted `.dll`/import-lib/export macro/downstream/loader proof before Windows shared claim. |

Do not infer cross-platform shared support from a single-platform shared proof.

## Documentation And Non-Claim Updates

Both decision paths must update or audit:

- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `benchmarks/README.md` if package/report interpretation changes;
- `.github/workflows/*.yml` comments and job names;
- `tests/corpus/manifests/report_families.tsv` if package or CI row scopes
  change;
- Sprint 153 artifacts and final residual queue.

Required non-claims unless explicitly earned:

- no shared-library support;
- no dynamic ABI compatibility;
- no runtime-loader compatibility;
- no package-manager distribution;
- no static/shared package selector support;
- no Windows Makefile parity;
- no Windows `pkg-config` parity;
- no broad platform parity.

## Sprint 153 Decision Handoff

Sprint 153 should begin with:

1. public symbol and header inventory;
2. static/global state and allocator/callback audit;
3. Linux/macOS/Windows loader requirement audit;
4. package metadata audit for CMake and `pkg-config`;
5. product decision: implement shared support or strengthen deferral;
6. implementation or deferral proof;
7. downstream consumer validation;
8. documentation and report-row alignment;
9. local quality and package checks;
10. hosted proof or explicit residuals for platforms not promoted.

## Stop Conditions

- `BUILD_SHARED_LIBS=ON` configures successfully without full shared-library
  support being intentionally implemented and tested.
- Shared artifacts are installed while docs still say static-first only.
- CMake or `pkg-config` metadata implies shared support, ABI compatibility, or
  selector support without downstream proof.
- A single-platform shared proof is described as cross-platform support.
- Runtime-loader behavior is claimed without installed-prefix loader tests.
- Windows shared support is claimed without export/import and hosted loader
  evidence.
- Package-manager support is claimed without recipe, versioning,
  install/update/uninstall, and CI proof.
- Static-first install/export proof regresses while working on shared support.
- Required C quality gates fail after `.c` or `.h` changes.

## Day 11 Handoff

Day 11 should define the external comparison evidence gate. It should reuse the
Day 10 rule that package/ABI support claims require executable proof and apply
the same discipline to comparison claims: no broad external-library parity or
state-of-the-art wording without direct, named, versioned, caveated comparison
evidence.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 153 can make a product-level decision. | Complete | Product decision options, implementation gate, deferral gate, and Sprint 153 handoff are defined. |
| Shared-library support cannot be accidentally implied. | Complete | Stop conditions and metadata checklist reject unsupported shared, ABI, loader, selector, and package-manager wording. |
| Static-first support remains guarded if shared support is deferred. | Complete | Static-first deferral gate preserves configure rejection, install/export checks, deferral guard, docs, and hosted package proof boundaries. |
