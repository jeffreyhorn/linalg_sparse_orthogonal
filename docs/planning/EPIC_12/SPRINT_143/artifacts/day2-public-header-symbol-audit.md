# Day 2 Public Header And Symbol Audit

## Purpose

Day 2 audits the installed public header surface and the current static archive
symbol surface before Sprint 143 makes a package/ABI product decision. The
goal is to separate ordinary public C API declarations from ABI-sensitive
surfaces and to define what would be required before shared-library support
could be reviewed honestly.

## Installed Header Inventory

The current install surface includes 19 public headers:

| Header | ABI-relevant role |
| --- | --- |
| `include/sparse_analysis.h` | Symbolic/numeric analysis structs, reorder typed options, factor lifecycle, refactor path. |
| `include/sparse_bidiag.h` | Bidiagonalization factor struct and lifecycle. |
| `include/sparse_cholesky.h` | Cholesky backend enum/options and one-shot factor/solve functions. |
| `include/sparse_csr.h` | Public CSR/CSC structs, conversion functions, and free functions. |
| `include/sparse_dense.h` | Dense helper struct and dense/Givens declarations installed as public headers. |
| `include/sparse_eigs.h` | Eigensolver enums, options, result, reusable handle, callbacks, and solve entries. |
| `include/sparse_ic.h` | IC(0) factor/solve/preconditioner functions reusing ILU storage. |
| `include/sparse_ilu.h` | ILU/ILUT factor structs, options, solve/free/preconditioner functions. |
| `include/sparse_iterative.h` | Iterative solver options/results, callbacks, reusable handles, matrix-free APIs. |
| `include/sparse_ldlt.h` | LDLT factor struct, backend enum/options, solve/refine/inertia/condest APIs. |
| `include/sparse_lu.h` | LU options and factor/solve/refine/permutation helper APIs. |
| `include/sparse_lu_csr.h` | LU CSR working-format structs/functions installed as public declarations. |
| `include/sparse_matrix.h` | Opaque `SparseMatrix`, matrix construction, mutation, query, arithmetic, I/O, and permutation APIs. |
| `include/sparse_qr.h` | QR factor/result/rank info structs and QR solve/refine/rank/minimum-norm APIs. |
| `include/sparse_reorder.h` | Reorder and permutation APIs. |
| `include/sparse_svd.h` | SVD options/results and full, partial, rank, pseudoinverse, low-rank APIs. |
| `include/sparse_types.h` | Core scalar/index typedefs, error/reorder/pivot enums, progress callback, version include. |
| `include/sparse_vector.h` | Publicly installed vector helper header with no exported declarations in the simple scan. |
| `include/sparse_version.h.in` | Generated installed `sparse_version.h` template and version macros. |

## Public Declaration Shape

| Category | Observed count | Notes |
| --- | ---: | --- |
| Function declarations in simple public scan | 130 | Includes `sparse_*` declarations plus installed non-`sparse_` helper declarations such as `dense_*`, `givens_*`, and `lu_csr_*`. |
| Public structs | 31 | Includes result/options/factor structs and installed working/helper structs. |
| Public enums | 15 | Includes error, pivot, reorder, backend, solver-selection, and typed-option enums. |
| Callback typedefs | 4 | Progress, iterative progress, preconditioner, and matrix-free callbacks. |
| Installed generated version macros | 5 | Major, minor, patch, encoded integer, and version string macros. |

The public surface is broad enough that ABI support cannot be inferred from
successful static linking. Struct layout, enum values, callback signatures,
type-width macros, and ownership/lifecycle rules would need an explicit
compatibility policy before any dynamic ABI claim.

## ABI-Sensitive Surfaces

| Surface | ABI sensitivity | Current status |
| --- | --- | --- |
| `idx_t` width | `SPARSE_IDX_BITS` selects `int32_t` or `int64_t`; callers and the library must be rebuilt with the same width. | Build-time width contract, not a stable binary ABI across widths. |
| `sparse_scalar_t` | Currently aliases `double`; callback/result buffers depend on this layout. | Real-only scalar lane; no complex or multi-precision ABI claim. |
| Public structs | Layout and field order matter for options, results, factors, handles, CSR/CSC, QR/SVD/LDLT/ILU/analysis objects. | Source/header contract only; no long-term binary layout policy. |
| Public enums | Numeric values are ABI-visible where used in options/results and error handling. | Values are source-visible; no ABI versioning policy exists. |
| Opaque `SparseMatrix` | Forward declaration hides matrix internals from public headers. | Stronger ABI posture than exposed matrix struct, but allocation/free semantics still matter. |
| Callback typedefs | Function pointer signatures bind caller/plugin binary interfaces. | Source-stable only; no shared-library callback ABI policy. |
| Ownership contracts | Many APIs require caller allocation/free, zeroed structs, or specific free helpers. | Documented in headers but not ABI-tested through dynamic boundaries. |
| Build option macros | `SPARSE_OPENMP`, `SPARSE_MUTEX`, `SPARSE_IDX_BITS`, thresholds, and optional dense backend probes can affect ABI/link behavior. | Build/report context unless selected package path promotes specific guarantees. |
| Installed non-`sparse_` helpers | `dense_*`, `givens_*`, `lu_csr_*`, and similar names are installed and globally visible in the static archive. | Must be classified before shared-library export support. |

## Current Static Archive Symbol Evidence

Local symbol listing command:

```sh
nm -g --defined-only build/libsparse_lu_ortho.a
```

Observed summary:

| Metric | Count |
| --- | ---: |
| Defined global symbols in current static archive | 359 |
| `sparse_`-prefixed defined global symbols | 222 |
| Non-`sparse_` or internal-looking defined global symbols | 137 |

Representative non-`sparse_` global symbols currently visible from the static
archive include:

- `dense_create`, `dense_free`, `dense_gemm`, `dense_gemv`;
- `givens_compute`, `givens_apply_left`, `givens_apply_right`;
- `lu_csr_from_sparse`, `lu_csr_eliminate`, `lu_csr_solve`,
  `lu_csr_free`;
- `chol_csc_*` internal CSC Cholesky helpers;
- `ldlt_csc_*` internal CSC LDLT helpers;
- `colamd_*`, `graph_*`, `lanczos_*`, and other implementation helpers;
- test-only or override-looking dense-kernel symbols such as
  `chol_csc_supernodal_set_dense_kernels_override_for_test`.

This is acceptable for the current static archive contract, where object files
are linked directly and no exported dynamic symbol promise exists. It is not
acceptable as-is for a reviewed shared-library ABI claim because internal
implementation helpers would become part of the dynamic export surface unless
visibility/export policy is added.

## Export And Visibility Audit

Searches for export/import and dynamic-ABI metadata across `include/`,
`CMakeLists.txt`, `cmake/`, and `sparse.pc.in` found no current use of:

- `SPARSE_API`;
- `SPARSE_EXPORT`;
- `SPARSE_IMPORT`;
- `__declspec`;
- visibility attributes;
- `SOVERSION`;
- `WINDOWS_EXPORT_ALL_SYMBOLS`;
- `C_VISIBILITY_PRESET`;
- `VISIBILITY_INLINES_HIDDEN`.

This confirms the current package contract remains static-first. It also means
there is no existing mechanism to distinguish public exported symbols from
implementation symbols for shared-library builds.

## Shared-Library Proof Requirements

If Sprint 143 selects shared-library ABI support, the minimum proof set should
include:

| Requirement | Why it is needed |
| --- | --- |
| Explicit export macro policy | Public headers need a deliberate `SPARSE_API`-style mechanism, including Windows import/export behavior. |
| Hidden-by-default visibility for implementation symbols | Prevent `chol_csc_*`, `ldlt_csc_*`, `graph_*`, `lanczos_*`, and other helpers from becoming public ABI. |
| Reviewed exported symbol allowlist | The dynamic library should export only intended public API names and required runtime symbols. |
| ABI-sensitive struct and enum policy | Options/results/factors/handles need documented compatibility expectations before binary consumers can rely on layout. |
| Versioning policy | CMake package exact-version behavior is not enough for shared ABI. Need library version, compatibility semantics, and possibly `SOVERSION`/install-name rules. |
| Platform-specific loader proof | Linux, macOS, and Windows each need loader/import-library/RPATH/install-name proof or explicit platform limitation. |
| Downstream dynamic consumer proof | Installed consumer must compile, link, run, and load against the shared artifact. |
| Static/shared coexistence decision | Package metadata must decide whether both variants can coexist and how `pkg-config`/CMake select them. |
| Negative unsupported-artifact checks | If shared support is not selected for a platform or package path, proof must show unsupported artifacts are absent or rejected. |

## Risk And Unknowns

| Risk | Impact | Day 5 decision input |
| --- | --- | --- |
| Broad internal global symbol surface | Shared builds would over-export implementation details without visibility controls. | Strongly increases shared-library implementation cost. |
| Installed helper headers expose non-`sparse_` names | Dynamic export policy must decide whether `dense_*`, `givens_*`, and `lu_csr_*` are public ABI or internal leakage. | May favor static-first strengthening unless a public API boundary is narrowed. |
| Public structs are layout-visible | Any shared ABI promise would need layout compatibility rules or a major-version policy. | Requires ABI policy before support claim. |
| `SPARSE_IDX_BITS` changes binary layout | Index width is a build-mode dimension and cannot be treated as a universal ABI. | Shared support would need width-specific artifacts or explicit limitation. |
| No existing export/import macro | Windows shared support would require public header changes and downstream consumer proof. | Adds implementation and validation burden. |
| No current loader proof | Runtime loader behavior is untested for installed shared artifacts. | Day 4 platform/loader audit must quantify this. |

## Day 3 Inputs

Day 3 should continue with install/export metadata rather than changing code:

- Make install/uninstall static archive behavior;
- CMake `install(TARGETS)` and `install(EXPORT)` metadata;
- `SparseConfig.cmake.in` package shape;
- exact-version `SparseConfigVersion.cmake` semantics;
- `sparse.pc.in` link flags and selector absence;
- `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh` proof coverage.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Installed public headers are accounted for. | Complete | Header inventory lists all 19 installed headers. |
| ABI-sensitive surfaces are separated from ordinary source implementation. | Complete | ABI-sensitive surface table separates type widths, struct layout, enum values, callbacks, ownership, and internal global symbols. |
| Shared-library proof requirements are concrete enough for Day 5 decision. | Complete | Proof requirements table names export, visibility, allowlist, versioning, loader, downstream consumer, variant selection, and negative checks. |
