# Sprint 133 Day 2 - Public Header and Symbol Exposure Audit

## Purpose

Day 2 audits installed public headers, consumer-visible version macros,
function declarations, public structs/enums, callback types, and symbol
exposure risk before Sprint 133 decides whether to implement shared-library
ABI support or continue explicit static-first support.

This is a documentation-only audit. It does not change headers, exported
symbols, build rules, install behavior, package metadata, tests, or public
support wording.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| `include/*.h` | Source header set installed by Make and CMake package flows. |
| `include/sparse_version.h.in` | Generated installed version macro template. |
| `tests/test_install.sh` | Downstream `pkg-config` consumer includes and version macro proof. |
| `examples/cmake_example/main.c` | Maintained installed CMake consumer include set. |
| `CMakeLists.txt` | CMake install/export owner and generated header owner. |
| `Makefile` | Make install owner for source headers plus generated `sparse_version.h`. |
| `INSTALL.md` and `README.md` | Current installed-header and static-first support wording. |

## Installed Header Inventory

Current install tests expect all source headers under `include/` plus generated
`sparse_version.h`. The source header inventory is:

| Header | Public surface | ABI risk if shared support is selected | Notes |
| --- | --- | --- | --- |
| `sparse_analysis.h` | Analysis options/results and factor lifecycle declarations. | High | Exposes multiple public structs/enums and factor handle fields. |
| `sparse_bidiag.h` | Bidiagonalization result struct and routines. | High | Public result struct contains caller-visible arrays and dimensions. |
| `sparse_cholesky.h` | Cholesky options and factor/solve declarations. | Medium | Main matrix type is opaque, but options enum/struct shape is public. |
| `sparse_csr.h` | CSR/CSC structs, conversions, and free routines. | High | `SparseCsr` and `SparseCsc` expose raw pointer layout. |
| `sparse_dense.h` | Dense matrix struct. | High | Public struct exposes data pointer and leading-dimension shape. |
| `sparse_eigs.h` | Eigensolver options, results, handle, and routines. | High | Public options/results/handle structs and callback-adjacent fields are layout-sensitive. |
| `sparse_ic.h` | Incomplete Cholesky wrappers over ILU storage. | Medium | Uses `sparse_ilu_t` public layout from `sparse_ilu.h`. |
| `sparse_ilu.h` | ILU/ILUT options, factors, and preconditioners. | High | Public factor structs expose sparse storage arrays. |
| `sparse_iterative.h` | Iterative solver options, handles, callbacks, and routines. | High | Public handle structs and callback typedefs would be ABI-sensitive. |
| `sparse_ldlt.h` | LDLT options, factors, backend metadata, and routines. | High | Public factor/options/backend structs expose layout and enum values. |
| `sparse_lu.h` | LU options, factor/solve/refinement declarations. | Medium | `SparseMatrix` stays opaque, but options structs and enum values are public. |
| `sparse_lu_csr.h` | CSR LU options/factors and solve declarations. | High | Public factor structs expose CSR LU arrays and pivot data. |
| `sparse_matrix.h` | Opaque matrix handle and core matrix routines. | Medium | `SparseMatrix` is opaque, but constants and function signatures are public. |
| `sparse_qr.h` | QR result/rank structs and routines. | High | Public QR structs expose arrays, rank, and factor dimensions. |
| `sparse_reorder.h` | Reordering and permutation routines. | Medium | Mostly function signatures over opaque matrix and caller-owned arrays. |
| `sparse_svd.h` | SVD options/results and routines. | High | Public result structs expose singular vectors/values and dimensions. |
| `sparse_types.h` | Core typedefs, error enum, scalar/index macros, progress callback. | High | `idx_t`, `sparse_scalar_t`, `sparse_err_t`, and callback payload shape affect every public signature. |
| `sparse_vector.h` | Header guard only in current scan. | Low | Installed but no visible public API declarations today. |
| generated `sparse_version.h` | Version macros. | Medium | Compile-time version metadata; not an ABI compatibility policy by itself. |

## Installed Consumer Include Set

| Consumer | Includes observed | Interpretation |
| --- | --- | --- |
| `tests/test_install.sh` generated consumer | `<sparse/sparse_types.h>`, `<sparse/sparse_matrix.h>` | Minimal `pkg-config` proof covers version macros, matrix create/insert/nnz/free, and static link/run behavior. |
| `examples/cmake_example/main.c` | `<sparse/sparse_types.h>`, `<sparse/sparse_matrix.h>`, `<sparse/sparse_lu.h>`, `<sparse/sparse_lu_csr.h>` | Maintained CMake consumer proof covers version macros, matrix construction, LU CSR solve route, and installed target linkage. |
| `INSTALL.md` include example | `<sparse/sparse_types.h>` | Documentation confirms installed include prefix shape. |

The installed package exposes more headers than the current downstream smoke
consumers include. Day 4 and Days 11-12 should decide whether consumer proof
needs broader header inclusion coverage for the selected package contract.

## ABI-Sensitive Declaration Map

| Declaration family | Current source-facing behavior | ABI sensitivity if shared support is selected |
| --- | --- | --- |
| Opaque matrix handle | `typedef struct SparseMatrix SparseMatrix` in `sparse_matrix.h`. | Lower risk than public layout, but allocation/free ownership and pointer identity remain ABI-sensitive. |
| Public storage structs | `SparseCsr`, `SparseCsc`, dense matrix, LU CSR, ILU, LDLT, QR, bidiag, SVD, eigensolver, and iterative handles expose fields. | High; field order, size, alignment, ownership, and enum member values become compatibility constraints. |
| Core typedefs | `idx_t` depends on `SPARSE_IDX_BITS`; `sparse_scalar_t` is currently `double`. | High; type width and scalar type affect every compiled caller and cannot vary across a binary ABI without explicit policy. |
| Error and option enums | `sparse_err_t`, pivot/order/backend/eigensolver/iterative enums, and option enums are visible. | Medium to high; numeric values and additions must be governed if shared ABI is promised. |
| Callback typedefs | Progress, iterative callback, preconditioner, matrix-free matvec, and cancellation-adjacent callback types are public. | High; callback calling convention, payload layout, and lifetime rules become ABI contract. |
| Function declarations | Public `sparse_*` and `lu_csr_*` routines across solver, matrix, reorder, conversion, and utility headers. | High; symbol names, parameter types, return types, ownership, and error behavior need symbol/version proof under shared support. |
| Macros and constants | `SPARSE_IDX_BITS`, `SPARSE_SCALAR_BITS`, `SPARSE_NODES_PER_SLAB`, `SPARSE_DROP_TOL`, eigensolver thresholds, version macros. | Medium; compile-time behavior and inlined decisions are source contract today, and need explicit ABI treatment if shared support is selected. |
| Static inline helpers | No broad installed inline-helper owner was identified in Day 2 scan. | Low current risk, but Day 9 should confirm before symbol/ABI proof design. |

## Version and Feature Macro Inventory

| Macro or template | Header | Current meaning | ABI interpretation |
| --- | --- | --- | --- |
| `SPARSE_VERSION_MAJOR` | generated `sparse_version.h` | Major component from repo `VERSION`. | Package metadata, not ABI policy alone. |
| `SPARSE_VERSION_MINOR` | generated `sparse_version.h` | Minor component from repo `VERSION`. | Package metadata, not ABI policy alone. |
| `SPARSE_VERSION_PATCH` | generated `sparse_version.h` | Patch component from repo `VERSION`. | Package metadata, not ABI policy alone. |
| `SPARSE_VERSION_ENCODE(maj, min, pat)` | generated `sparse_version.h` | Integer version encoding helper. | Source macro; no shared ABI promise by itself. |
| `SPARSE_VERSION` | generated `sparse_version.h` | Encoded current version. | Consumer compile-time metadata. |
| `SPARSE_VERSION_STRING` | generated `sparse_version.h` | String form of current version. | Consumer compile-time metadata. |
| `SPARSE_IDX_BITS` | `sparse_types.h` | Compile-time selected index width, default 32. | ABI-critical if shared support is selected. |
| `SPARSE_SCALAR_BITS` | `sparse_types.h` | Bit width of current `sparse_scalar_t`. | ABI-critical if scalar widening is ever supported. |
| `SPARSE_PRIDX` / `SPARSE_SCNIDX` | `sparse_types.h` | Format fragments matching `idx_t`. | Source compatibility helpers tied to index ABI. |
| `SPARSE_NODES_PER_SLAB` | `sparse_matrix.h` | Public allocation/block-size constant. | Source-visible tuning constant; compatibility policy unclear. |
| `SPARSE_DROP_TOL` | `sparse_matrix.h` | Public default drop tolerance constant. | Source-visible numeric policy. |
| `SPARSE_CSC_THRESHOLD` | `sparse_matrix.h` | Public conversion threshold constant. | Source-visible behavior hint. |
| `SPARSE_EIGS_THICK_RESTART_THRESHOLD` | `sparse_eigs.h` | Eigensolver threshold constant. | Source-visible behavior hint. |
| `SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD` | `sparse_eigs.h` | Eigensolver auto-selection threshold constant. | Source-visible behavior hint. |

No installed macro currently declares shared-library availability, symbol
visibility, ABI version, soname, or package-manager support.

## Symbol Visibility and Shared-Library Risk

Current static-first packaging does not define a public export macro such as
`SPARSE_API`, does not define an ABI version macro, and does not install a
shared-library artifact. If shared support is selected later in Sprint 133,
the design must decide at least:

- whether public functions receive an explicit export/import annotation;
- which symbols are exported and which remain hidden;
- whether public struct layout is frozen, versioned, or moved behind opaque
  handles;
- whether `idx_t` width and `sparse_scalar_t` are one ABI per build or part of
  the binary package identity;
- whether enum values and callback payload layouts are compatibility promises;
- how generated version metadata relates to ABI versioning;
- how downstream CMake and `pkg-config` consumers select static versus shared
  link behavior.

## Install-Facing But Not Dynamic ABI Contracts

| Header or surface | Current disposition |
| --- | --- |
| All installed `include/*.h` headers | Source-facing public headers under the static-first package contract. They are not dynamic ABI contracts today. |
| `sparse_vector.h` | Installed but currently header-guard-only in Day 2 scan; keep as install-facing source surface until Day 3 verifies install intent. |
| Public option/result structs | Source-facing caller contracts today; shared ABI support would need explicit layout policy or wrapper strategy. |
| Generated `sparse_version.h` | Installed version metadata; not a soname, ABI epoch, or compatibility promise today. |

## Day 3 Handoff

Day 3 should audit install shape and package metadata with these specific
header/symbol questions in mind:

- confirm the exact installed header count and whether `sparse_vector.h`
  should remain installed;
- compare Make install and CMake install header layout and generated
  `sparse_version.h` behavior;
- inspect whether CMake export or `sparse.pc` can express `SPARSE_IDX_BITS`,
  scalar width, static link flags, or library type if needed;
- verify that package metadata still describes a static archive surface and
  does not imply shared ABI support;
- record whether any installed artifact names or generated config files would
  need to change if shared support is selected.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every installed header has intended support status or explicit unknown status. | Complete | Installed header inventory classifies all source headers plus generated `sparse_version.h`. |
| ABI-sensitive declarations are separated from source-compatibility-only declarations. | Complete | ABI-sensitive declaration map separates opaque handles, public layouts, core typedefs, callbacks, function symbols, macros, and current source-only install surfaces. |
| Shared-library risk is recorded before the design decision. | Complete | Symbol visibility and shared-library risk section records export, layout, ABI width, callback, enum, version, and consumer-link decisions required before support can be selected. |
