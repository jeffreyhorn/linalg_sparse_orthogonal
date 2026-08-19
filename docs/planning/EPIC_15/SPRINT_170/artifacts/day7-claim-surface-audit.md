# Sprint 170 Day 7: Package And ABI Claim Surface Audit

## Purpose

Audit public documentation, package metadata templates, workflows, tests, and
guards for package, ABI, shared-library, runtime-loader, package-manager, and
platform wording before Sprint 170 selects a product posture.

## Documentation Claim-Surface Inventory

| Surface | Current wording posture | Day 7 finding |
| --- | --- | --- |
| `README.md` | Keeps a short static-first install summary and points detailed package behavior to `INSTALL.md`. Explicitly says Windows remains CMake-first and does not claim Makefile parity, `pkg-config` execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. | Consistent with static-first evidence. No immediate correction required. |
| `INSTALL.md` | Owns the detailed install/package contract. States the maintained install contract is static-first and that shared-library packaging, dynamic ABI compatibility, runtime-loader behavior, package-manager distribution, static/shared selectors, Windows Makefile parity, and Windows `pkg-config` execution parity are out of scope. | Strongest user-facing package contract. No immediate correction required. |
| `docs/maintainer_guide.md` | Owns maintainer interpretation for package and ABI boundaries, proof owners, CI lanes, and non-claims. States any future shared-library or wider ABI claim is a separate product contract. | Consistent and detailed. Candidate for Day 10 alignment once Day 8 decides the posture. |
| `docs/api_reference.md` | States the generated API reference does not imply dynamic ABI compatibility, shared-library support, package-manager distribution, broad platform parity, external-library parity, or state-of-the-art coverage. | Correctly prevents API docs from being mistaken for ABI docs. |
| `docs/tutorial.md` | Points installed package readers to `INSTALL.md` and identifies static-first package support as bounded evidence. | Consistent with package boundary. |
| `docs/cookbook.md` | Uses bounded solver examples and avoids package/ABI widening. | No package/ABI correction found. |
| `docs/solver_selection.md` | Separates solver-selection evidence from package, ABI, external-library, and state-of-the-art parity claims. | Consistent with Sprint 169 performance/evidence separation. |
| `docs/algorithm.md` and `docs/algorithm_history.md` | Explicitly avoid being install/support/package/ABI references. | Correct ownership boundary. |

## Package Metadata Wording Inventory

| Surface | Current static-first metadata | Unsupported wording guard |
| --- | --- | --- |
| `sparse.pc.in` | `Description: Static archive package metadata for sparse linear algebra`; `Cflags: -I${includedir}`; `Libs: -L${libdir} -lsparse_lu_ortho -lm @SPARSE_PC_LIBS_EXTRA@`. | No `Libs.private`, shared/static selector, ABI variable, package-manager metadata, or loader wording. |
| `cmake/SparseConfig.cmake.in` | Minimal `@PACKAGE_INIT@`, target include, and required-component check. | No components, shared/static selector, ABI variable, loader metadata, or package-manager wording. |
| Generated CMake package version | Uses `ExactVersion`. | Exact package version must not be described as broad dynamic ABI compatibility. |
| CMake target export | Installs `Sparse::sparse_lu_ortho` as a static imported target. | Install tests reject `SHARED IMPORTED`, `MODULE IMPORTED`, and imported `.so`/`.dylib`/`.dll` metadata. |

The metadata templates are aligned with the current static archive support
claim. They intentionally do not advertise dynamic ABI support.

## Guard And Test Ownership Notes

| Owner | Guarded behavior |
| --- | --- |
| `tests/test_install.sh` | Unix-side Make install/uninstall plus `pkg-config`; checks static archive installation, no shared-library artifacts, installed header count, exact version resolution, `sparse.pc` static description, no `Libs.private`, no unsupported package/ABI wording, downstream compile/link/run, and uninstall cleanup. |
| `tests/test_cmake_install.sh` | Unix-side CMake install/export; checks static imported target metadata, install-prefix include/archive locations, no source/build path leaks, no shared imported metadata, no unsupported loader/shared-selector metadata, `sparse.pc` static description, exact-version behavior, mismatch rejection, and downstream `find_package` consumers. |
| `scripts/static_package_deferral_check.sh` | Local package-contract guard; checks `BUILD_SHARED_LIBS=ON` rejection, explicit static target, archive-only install metadata, no public export/import or ABI selector macros, no shared ABI metadata, no package selectors, and preserved support wording. |
| `.github/workflows/ci.yml` | Linux reviewed package-contract lane runs Make install/pkg-config proof, CMake install/export proof, and static deferral proof. |
| `.github/workflows/macos-ci.yml` | macOS reviewed package lanes run Make install/pkg-config proof and CMake install/export plus static deferral proof. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake install/downstream validation checks installed `.lib`, package metadata, maintained downstream CMake consumers, exact/mismatch version behavior, no DLL/shared imported metadata, and metadata-only `sparse.pc` inspection. |
| `tests/corpus/manifests/report_families.tsv` | Report-family claim boundaries say package rows do not imply package-manager availability, shared-library ABI support, or dynamic linking claims. |
| `tests/corpus/schemas/report_index_fields.md` | Report index schema keeps package-manager, shared-library ABI, and dynamic-linking claims out of report-row pass interpretation. |

Guard ownership is clear and consistent. The package-contract tests are
evidence owners, while documentation explains how far that evidence can be
cited.

## Inconsistency And Staleness Review

No Day 7 blocking inconsistency was found in the audited package/ABI surfaces.
The current wording repeatedly preserves these non-claims:

- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager distribution;
- static/shared selectors;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- state-of-the-art or external-library parity from package evidence.

The wording is intentionally repetitive, but the repetition is serving a guard
purpose: different entry points reach different audiences. The Day 8 product
decision can choose to retain the repetition, consolidate it, or add a single
decision record as the authoritative citation.

## Candidate Documentation And Guard Updates

These are candidates for later decision/implementation days, not Day 7 edits:

| Candidate | Rationale | Suggested owner day |
| --- | --- | --- |
| Add a Sprint 170 product decision record as the canonical ABI/package posture. | Current non-claims are correct but distributed across README, INSTALL, maintainer guide, scripts, workflows, and report manifests. | Day 8 or Day 9 |
| Add a short README link to the new decision record if Day 8 selects static-first continuation. | README already points to INSTALL, but a decision record would make the shared-library deferral easier to cite in PR reviews. | Day 10 |
| Teach `static_package_deferral_check.sh` to require the decision record after it exists. | Prevent future docs from drifting away from the selected product posture. | Day 11 |
| Add a generated or manual package/ABI claim index. | Helps reviewers locate all package and ABI claim surfaces without broad text searches. | Day 12 |
| Preserve exact CMake version wording as package-version behavior, not ABI behavior. | Already true, but likely worth reiterating in the product decision record. | Day 8 or Day 10 |

## Day 8 Handoff

Day 8 can synthesize the prior evidence with a low ambiguity level:

- Headers expose enough concrete layout and lifecycle risk that dynamic ABI is
  not ready by default.
- The archive symbol table is not curated for a shared library.
- Make and CMake package behavior is intentionally static-first.
- Public docs, workflows, tests, and metadata guards consistently describe
  static-first support and shared-library/dynamic-ABI deferral.

The natural decision candidate is to retain static-first support for Sprint
170 and create a future work queue for shared-library exploration instead of
promoting shared-library support inside this sprint.

## Day 7 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Documentation claim-surface inventory | Complete | README, INSTALL, API reference, maintainer guide, tutorial, cookbook, solver selection, and algorithm docs were audited. |
| Package metadata wording inventory | Complete | `sparse.pc.in`, `SparseConfig.cmake.in`, generated package version posture, and CMake target export posture were reviewed. |
| Guard/test ownership notes | Complete | Make install, CMake install, static deferral, CI lanes, and report-manifest boundaries were mapped. |
| Candidate doc and guard updates | Complete | Candidates are deferred to Day 8/Day 10+ after product posture selection. |
| Day 7 claim-surface audit artifact | Complete | This file. |

## Validation

Day 7 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Package and ABI claim surfaces are known. | Complete | Public docs, metadata, tests, scripts, workflows, and report manifests are inventoried. |
| Inconsistencies are ready for Day 8/Day 10 decisions. | Complete | No blocking inconsistency found; candidate consolidation/guard updates are listed. |
| No support claim is broadened during audit. | Complete | The artifact preserves shared-library and dynamic ABI non-claims. |
