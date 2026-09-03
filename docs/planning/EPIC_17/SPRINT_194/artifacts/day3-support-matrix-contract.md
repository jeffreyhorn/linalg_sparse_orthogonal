# Day 3 Support Matrix Contract

## Objective

Define a compact production-readiness/support matrix contract that can become
the active user-facing support truth without erasing detailed proof ownership.

## Owner Decision

The support/readiness matrix should live in `INSTALL.md`.

Rationale:

- `INSTALL.md` already owns operational setup, installed package shape,
  downstream consumer workflows, platform support, and install validation.
- It is user-facing, unlike `docs/maintainer_guide.md`.
- It is closer to support/readiness questions than README, tutorial, cookbook,
  examples, API reference, benchmark docs, or corpus docs.
- It can link to maintainer, benchmark, corpus, and planning surfaces for
  proof detail without making those surfaces the first user path.

`docs/maintainer_guide.md` remains the proof interpretation owner.
`tests/corpus/manifests/selected_report_targets.tsv` remains selected report
target row authority. Benchmark and corpus docs remain detailed evidence
semantics owners.

## Required Matrix Columns

| Column | Purpose |
| --- | --- |
| Surface | The support/readiness area a user might ask about. |
| Current user status | Approved vocabulary term from this contract. |
| Primary user path | The first command or doc a user should use. |
| Evidence owner | Script, test, workflow, manifest, or maintainer doc that owns proof detail. |
| Retained non-claims | Explicit boundaries that the row must not imply. |

## Approved Status Vocabulary

| Status | Definition |
| --- | --- |
| `supported` | Normal documented user path with maintained validation for its named scope. |
| `validated` | Explicit local or hosted validation exists, but support is narrower than a broad platform/product claim. |
| `local-only` | Evidence exists only in the local checkout or ignored build artifacts unless another row names hosted evidence. |
| `hosted-evidence` | A hosted CI lane runs a selected proof or publishes selected metadata/artifacts for the named scope. |
| `deferred` | The project intentionally does not claim the surface until named prerequisites exist. |
| `not claimed` | No current user support statement exists for the surface. |
| `residual` | Known limitation or environment-dependent blocker; not pass evidence. |

## Initial Matrix Row Contract

| Surface | Current user status | Primary user path | Evidence owner | Retained non-claims |
| --- | --- | --- | --- | --- |
| Local source build and first solve | `supported` | `README.md`, `examples/README.md`, `make`, `make examples` | Makefile, examples, `make examples-build`, `make test` | No install, package-manager, performance, or broad platform-parity claim. |
| Unix Make static install | `supported` | `INSTALL.md#quick-start-makefile` | `tests/test_install.sh`, `sparse.pc.in`, Linux/macOS install lanes | No shared-library, dynamic ABI, runtime-loader, or package-manager support. |
| Unix `pkg-config` consumer | `validated` | `INSTALL.md#using-via-pkg-config` | `tests/test_install.sh`, `sparse.pc.in` | No Windows `pkg-config` execution parity. |
| Installed CMake consumer | `supported` | `INSTALL.md#using-from-a-cmake-project`, `examples/cmake_example/` | `tests/test_cmake_install.sh`, `cmake/SparseConfig.cmake.in` | No shared-library or dynamic ABI promise. |
| Windows MSVC CMake install/downstream | `validated` | `INSTALL.md#windows-msvc` | `.github/workflows/windows-ci.yml`, Windows install/downstream lane, PowerShell validator | No Windows Makefile parity, Windows `pkg-config` execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |
| Windows selected Cholesky comparison freshness | `hosted-evidence` | Windows selected comparison workflow and selected target manifest | Windows selected Cholesky lane, `normalize_report_index.py --selected-target cholesky-spd-tridiag-5`, PowerShell guard | No broad Windows report freshness, Windows selected oracle freshness, Windows selected benchmark freshness, or unselected Windows comparison families. |
| Linux/macOS selected comparison freshness | `hosted-evidence` | `make report-index-comparison-freshness` | `selected_report_targets.tsv`, Linux/macOS hosted selected comparison lanes, comparison tests | No broad external-library parity, broad report freshness, package/ABI support, performance superiority, or state-of-the-art claim. |
| Linux selected performance freshness | `hosted-evidence` | `make bench-canonical-report-freshness` | `selected_report_targets.tsv`, `benchmarks/README.md`, `check_bench_canonical_freshness.py`, Linux hosted lane | No portable performance, timing threshold, release benchmark, platform parity, package/ABI, or state-of-the-art claim. |
| Local benchmark/sentinel reports | `local-only` | `make bench-canonical-report`, `make performance-sentinels` | `benchmarks/README.md`, benchmark scripts/tests | No hosted proof or portable speedup claim unless a selected hosted lane says so. |
| Local generated API HTML | `local-only` | `docs/api_reference.md`, `make api-docs-freshness` | `Doxyfile`, API docs coverage/local-only scripts | No hosted API publication or completeness beyond checked-in public headers selected by Doxyfile. |
| Package-manager distribution | `not claimed` | Use source install via Make or CMake | Homebrew proof material, package-manager deferral guard, Sprint 188 artifacts | No Homebrew/core, bottles, Linuxbrew, tap, vcpkg, Conan, pkgsrc, distro/system package, or broad provider support. |
| Shared-library and dynamic ABI support | `deferred` | Static install only | Static package deferral guard, Sprint 170 package/ABI decision | No `.so`, `.dylib`, `.dll`, import-library, loader, SONAME, install-name/RPATH, selector, or dynamic ABI support. |
| Broad ecosystem or state-of-the-art parity | `not claimed` | Use selected evidence docs only for scoped proof | Epic 17 review/todo, selected target manifest, benchmark/corpus docs | No broad SuiteSparse/PETSc/Trilinos/Eigen/SciPy parity, portable superiority, or unqualified state-of-the-art status. |

## Wording Rules

- Use "supported" only for named, documented paths with maintained validation.
- Use "validated" for narrow proof surfaces where broad support would be too
  strong.
- Use "hosted-evidence" only when a hosted workflow is named and its scope is
  clear.
- Use "local-only" for generated local artifacts, local benchmark snapshots,
  and generated API HTML.
- Use "deferred" when a future prerequisite is already documented.
- Use "not claimed" when there is no current support path.
- Use "residual" for unavailable local tooling, missing optional dependencies,
  stale/missing local generated artifacts, or other blocker context that must
  not be read as pass evidence.

## Link Policy

| Source | Link behavior after matrix exists |
| --- | --- |
| `README.md` | Keep a short adoption/support summary and link to the matrix. |
| `INSTALL.md` | Own the matrix and keep install commands/proof detail nearby. |
| `docs/tutorial.md` | Link to the matrix only for installed consumer and support status handoff. |
| `docs/cookbook.md` | Link to the matrix for package/platform/performance boundaries. |
| `docs/solver_selection.md` | Keep solver and diagnostics guidance local; link to the matrix for package/platform/report support boundaries. |
| `docs/api_reference.md` | Keep declaration and local Doxygen truth local; link to the matrix for support/readiness status. |
| `examples/README.md` | Keep runnable example guidance local; link to the matrix for installed-consumer/support interpretation. |
| `benchmarks/README.md` | Keep benchmark/report semantics local; link to matrix only for support summary if needed. |
| `tests/corpus/README.md` | Keep selected target semantics local; do not duplicate the support matrix. |
| `docs/maintainer_guide.md` | Retain proof semantics; link to the matrix as the user-facing support summary. |

## Day 4 Implementation Boundary

Day 4 should add the matrix and only the minimal routing links needed to make
it the active support/readiness truth. It should not change:

- selected target manifest rows;
- CI workflow behavior;
- package templates;
- install rules;
- generated report semantics;
- public headers;
- support levels.

## Validation

Day 3 changed planning documentation only.

```sh
git diff --check
```

No `.c` or `.h` files were modified, so `make format && make lint && make
test` is not required for this day.
