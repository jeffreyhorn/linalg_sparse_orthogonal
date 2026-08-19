# Sprint 170 Day 4: Symbol And Visibility Feasibility

## Purpose

Inspect the static archive object composition, global symbol surface, and
build-system visibility controls before the Sprint 170 shared-library ABI
product decision.

## Current Build And Install Posture

The maintained build posture is explicitly static-first:

- CMake rejects `BUILD_SHARED_LIBS=ON` at configure time.
- CMake declares `add_library(sparse_lu_ortho STATIC ...)`.
- CMake install metadata installs only an `ARCHIVE` destination for the
  library target.
- Make builds and installs `build/libsparse_lu_ortho.a`.
- `scripts/static_package_deferral_check.sh` guards the static target,
  absence of shared install destinations, exact shared-library deferral
  wording, absence of public export/import macros, and absence of unsupported
  shared ABI metadata/selectors.

This means the current package surface does not require a curated dynamic
export list today. A shared-library product would.

## Static Archive Object Composition

Local archive inspected:

```sh
build/libsparse_lu_ortho.a
```

The archive currently contains 47 implementation objects:

| Object family | Representative objects |
| --- | --- |
| Core matrix/types/allocation | `sparse_matrix.o`, `sparse_matrix_build_internal.o`, `sparse_matrix_io.o`, `sparse_types.o`, `sparse_alloc_internal.o`, `sparse_factor_state_internal.o` |
| Direct solvers | `sparse_lu.o`, `sparse_cholesky.o`, `sparse_qr.o`, `sparse_qr_householder.o`, `sparse_bidiag.o`, `sparse_ldlt.o` |
| Compressed backends | `sparse_csr.o`, `sparse_lu_csr.o`, `sparse_lu_csr_struct.o`, `sparse_chol_csc.o`, `sparse_chol_csc_supernodal.o`, `sparse_ldlt_csc.o`, `sparse_ldlt_csc_rowadj.o`, `sparse_ldlt_csc_supernodal.o`, `sparse_ldlt_dense.o` |
| Iterative and eigensolvers | `sparse_iterative.o`, `sparse_iterative_block.o`, `sparse_iterative_minres.o`, `sparse_iterative_workspace_internal.o`, `sparse_eigs.o`, `sparse_eigs_workspace_internal.o`, `sparse_eigs_dense_internal.o`, `sparse_eigs_selection_internal.o`, `sparse_eigs_lobpcg.o`, `sparse_eigs_thick_restart.o` |
| Reordering and graph support | `sparse_reorder.o`, `sparse_reorder_nd.o`, `sparse_reorder_amd_qg.o`, `sparse_colamd.o`, `sparse_etree.o`, `sparse_analysis.o`, `sparse_graph*.o` |
| SVD and dense helpers | `sparse_svd.o`, `sparse_svd_partial.o`, `sparse_dense.o`, `sparse_vector.o` |

The object inventory confirms that the static archive intentionally packages
many backend and helper modules together. That is normal for static linking,
but it is not a reviewed dynamic export boundary.

## Global Symbol Inventory

Command used:

```sh
nm -g --defined-only build/libsparse_lu_ortho.a
```

Local macOS `nm` reports C symbols with a leading underscore. The scan found:

| Measurement | Count |
| --- | ---: |
| Global defined symbols in the static archive | 359 |
| Global `_sparse_*` symbols | 222 |
| Global symbols outside the public-looking `_sparse_*`, `_dense_*`, and `_lu_csr_*` prefixes | 124 |

Representative public-looking symbols include:

- `_sparse_create`, `_sparse_free`, `_sparse_copy`
- `_sparse_lu_factor`, `_sparse_lu_solve`
- `_sparse_cholesky_factor`, `_sparse_cholesky_solve`
- `_sparse_qr_factor`, `_sparse_qr_solve`
- `_sparse_ldlt_factor`, `_sparse_ldlt_solve`
- `_sparse_svd`, `_sparse_svd_partial`
- `_sparse_eigs_sym`, `_sparse_eigs_sym_with_handle`
- `_sparse_iter_handle_prepare_cg`, `_sparse_solve_cg_with_handle`
- `_sparse_strerror`, `_sparse_errno`, `_sparse_idx_bits`,
  `_sparse_scalar_bits`

Representative internal-looking globals include:

- `_chol_csc_*` compressed Cholesky backend helpers
- `_ldlt_csc_*` compressed LDLT backend helpers
- `_lu_dense_factor`, `_lu_csr_eliminate`, `_lu_detect_dense_blocks`
- `_lanczos_*`, `_s20_*`, `_s21_*`, `_s29_*`, `_s49_*`, `_s85_*`
  eigensolver and solver-internal helpers
- `_graph_*` and `_fm_*` graph partitioning helpers
- `_pool_alloc`, `_pool_release`, `_pool_free_all`
- test/runtime override hooks such as
  `_chol_csc_supernodal_set_dense_kernels_override_for_test`,
  `_ldlt_csc_set_kernel_override`, and graph override begin/end helpers

These globals are not necessarily a problem for the static archive. They are a
major blocker for any unqualified shared-library claim because a naive shared
build could export backend internals, workspace helpers, test override hooks,
and compatibility controls as accidental ABI.

## Visibility-Control Inventory

Current visibility and export controls:

| Control | Status |
| --- | --- |
| Explicit static target | Present in CMake. |
| `BUILD_SHARED_LIBS=ON` rejection | Present and guarded. |
| Shared install destinations | Absent and guarded. |
| Public export/import macro such as `SPARSE_API` | Absent and guarded as an unsupported-claim tripwire. |
| Compiler visibility preset such as `C_VISIBILITY_PRESET hidden` | Not present. |
| Linker export map or version script | Not present. |
| Windows `.def` file or `__declspec(dllexport/dllimport)` policy | Not present. |
| Linux SONAME policy | Not present. |
| macOS install-name/RPATH policy | Not present. |
| Dynamic ABI version/epoch metadata | Not present. |
| Installed shared consumer/runtime-loader proof | Not present. |

The current absence of export controls is consistent with the static-first
contract because shared-library configuration is rejected. It is not sufficient
for a shared-library product.

## Shared-Build Leakage Risks

| Risk | Severity | Notes |
| --- | --- | --- |
| Internal helper leakage | High | Many non-public helper families are global in the archive and could become exported symbols in a naive dynamic build. |
| Test override hook leakage | High | Dense-kernel, LDLT, graph, and ND override hooks are useful for tests and compatibility gates but should not become dynamic ABI accidentally. |
| Public-looking internal `_sparse_*_internal` symbols | High | Prefix-based export would still leak workspace and factor-state internals. |
| No hidden-by-default build policy | High | Without hidden visibility plus explicit exports, new helper functions can silently widen the dynamic surface. |
| No platform-specific export mechanism | High | Windows needs an import/export macro or `.def` policy; Linux/macOS need export lists and loader metadata. |
| No ABI symbol versioning | Medium | Linux symbol versioning is not always required, but no policy exists to distinguish source version from ABI compatibility. |
| No internal symbol naming rule | Medium | Some helpers use subsystem prefixes rather than `static` or private namespaces, making accidental exports harder to filter. |

## Symbol-Governance Requirements For A Future Shared Library

Before shared-library support can be claimed, the project should add all of the
following:

1. A product decision naming the exact supported dynamic surface.
2. A public export macro or export-list approach that is hidden-by-default.
3. A generated or maintained symbol allowlist that maps only supported public
   API symbols, not backend helpers or test hooks.
4. CI checks that compare built shared-library exports against the allowlist on
   each supported platform.
5. Windows import-library/DLL policy and downstream installed shared consumer
   proof.
6. Linux SONAME and optional version-script policy.
7. macOS install-name/RPATH policy.
8. Documentation separating source compatibility, static package support, and
   dynamic ABI compatibility.
9. Tests proving that static package metadata still does not imply shared
   support unless the shared path is intentionally enabled.

## Static-First Implications

The Day 4 evidence supports continuing the current static-first package
contract unless Sprint 170 explicitly funds a larger shared-library effort.
The static archive can contain internal global helpers without promising a
dynamic export boundary; a shared library cannot.

The current `BUILD_SHARED_LIBS=ON` rejection is therefore not a limitation to
work around casually. It is the guard preventing consumers from inferring ABI
stability from a symbol table that has not been curated.

## Day 4 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Public/internal symbol boundary notes | Complete | Recorded public-looking API symbols and internal-looking helper families from the archive. |
| Visibility-control inventory | Complete | Confirmed static target and shared rejection exist; hidden/export/version policies do not. |
| Shared-build leakage risk list | Complete | Internal helper, test hook, workspace, and platform export risks are documented. |
| Symbol-governance requirements | Complete | Listed prerequisites for any future shared-library product claim. |
| Day 4 symbol-visibility artifact | Complete | This file. |

## Validation

Day 4 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Symbol exposure risk is explicit. | Complete | The artifact records the 359-symbol global archive surface and internal-looking helper leakage risk. |
| Future shared-library requirements are separated from current static support. | Complete | Export macro, allowlist, SONAME/install-name/DLL, and runtime-loader proof are future requirements, not current claims. |
| Static-first behavior remains accurately described. | Complete | The artifact preserves the current CMake rejection, static target, static install destination, and guard posture. |
