# Sprint 214 Day 3: Symbol And Visibility Policy

## Status And Scope

This is a **future acceptance design**, not an enabled dynamic ABI. Day 2
selected ABI/release design with stronger static-first deferral. The maintained
package still builds and installs only a static archive; `BUILD_SHARED_LIBS=ON`
remains rejected. No public header annotation, shared target, export map, DLL
definition, or support claim is added on Day 3.

Three sets must stay distinct:

1. **Installed C declarations:** source-level API in checked-in `include/*.h`;
   installation alone does not grant dynamic ABI status.
2. **Archive-defined globals:** linker-visible static definitions, including
   helpers and test hooks; these are not an export policy.
3. **Provisional dynamic proof allowlist:** eight functions named below, to be
   approved only after a future product decision, signature/ownership review,
   platform export control, and installed consumer tests. The current effective
   shared export set is empty.

## Current Inventory

The local macOS archive `build/libsparse_lu_ortho.a` was current according to
`make -q build/libsparse_lu_ortho.a` (exit 0). `make source-list-check` passed
with 49 library sources. The archive has 49 implementation objects plus
Darwin's `__.SYMDEF SORTED` index. `nm -g --defined-only` reports 361 unique
defined globals: 360 text symbols and one data symbol. Leading Mach-O symbol
underscores were removed before comparison with C names. Sprint 170's older
47-object/359-global snapshot is not a current allowlist.

Each checked-in public header was parsed separately with the local Clang C11
AST (`-Ibuild/include -Iinclude`, default `SPARSE_IDX_BITS=32`), selecting
direct `FunctionDecl` nodes, not declarations inherited from other headers or
the SDK. This found 165 distinct direct function declarations in 18 headers.
All 165 names have definitions in the current archive; 196 defined globals
are not direct declarations in those installed headers. These are local
inventory counts, not ABI guarantees across configurations.

The local inventory can be reproduced with these commands; the `jq` filter
selects declarations originating in each root header, and Darwin `nm` names
are normalized by removing one leading underscore before comparison:

```sh
make source-list-check
make -q build/libsparse_lu_ortho.a
ar -t build/libsparse_lu_ortho.a
nm -g --defined-only build/libsparse_lu_ortho.a
for header in include/*.h; do
    clang -std=c11 -Ibuild/include -Iinclude -x c -fsyntax-only \
        -Xclang -ast-dump=json "$header" |
        jq -r '.. | objects | select(.kind? == "FunctionDecl" and .loc != null and (.loc.includedFrom? == null)) | .name'
done | sort -u
```

Retain only defined `nm` records and normalize the platform linker prefix.
Repeat the comparison for each proposed index-width/toolchain configuration
before freezing an ABI manifest.

| Installed source header | Direct C function declarations |
| --- | ---: |
| `sparse_analysis.h` | 6 |
| `sparse_bidiag.h` | 2 |
| `sparse_cholesky.h` | 3 |
| `sparse_csr.h` | 8 |
| `sparse_dense.h` | 10 |
| `sparse_eigs.h` | 5 |
| `sparse_ic.h` | 4 |
| `sparse_ilu.h` | 6 |
| `sparse_iterative.h` | 19 |
| `sparse_ldlt.h` | 7 |
| `sparse_lu.h` | 11 |
| `sparse_lu_csr.h` | 13 |
| `sparse_matrix.h` | 32 |
| `sparse_qr.h` | 14 |
| `sparse_reorder.h` | 6 |
| `sparse_svd.h` | 9 |
| `sparse_types.h` | 4 |
| `sparse_vector.h` | 6 |
| **Total** | **165** |

The generated installed `sparse_version.h` defines version macros, including
`SPARSE_VERSION_STRING`; it does **not** define linker symbols. The four
functions declared directly in `sparse_types.h` are runtime functions, not
generated-version symbols: `sparse_strerror`, `sparse_errno`,
`sparse_idx_bits`, and `sparse_scalar_bits`. `VERSION=2.2.0` and generated
macros identify source/package version, not a dynamic ABI epoch.

The 196 archive-only globals include `chol_csc_*`, `ldlt_csc_*`, `graph_*`,
`fm_*`, `lanczos_*`, `pool_*`, and test hooks such as
`chol_csc_supernodal_set_dense_kernels_override_for_test`. Conversely,
`eigen2x2`, `givens_*`, `lu_dense_*`, and `vec_*` have declarations in installed
headers. Neither a `sparse_` prefix nor an internal-looking name is a reliable
export rule. Explicit declaration ownership and a reviewed allowlist are
decisive.

The single global data symbol is `sparse_reorder_nd_base_threshold`. Its
declaration is in `src/sparse_reorder_nd_internal.h`, not an installed public
header, and tests mutate it. A dynamic export comparison must reject
unapproved data exports as well as functions.

## Provisional Dynamic Proof Allowlist

The smallest plausible future proof is an opaque `SparseMatrix` lifecycle and
single-element round trip using `include/sparse_matrix.h` and
`include/sparse_types.h`. The proposed **exact C-name list** is:

| C symbol | Declaration owner | ABI concern before approval |
| --- | --- | --- |
| `sparse_create` | `sparse_matrix.h` | `idx_t` width and returned-handle ownership. |
| `sparse_free` | `sparse_matrix.h` | Library-side deallocation of opaque handle. |
| `sparse_set` | `sparse_matrix.h` | Index/scalar/error representation. |
| `sparse_get` | `sparse_matrix.h` | Index/scalar return and invalid-access semantics. |
| `sparse_strerror` | `sparse_types.h` | Error enum values and returned string lifetime. |
| `sparse_errno` | `sparse_types.h` | Error reporting across the boundary. |
| `sparse_idx_bits` | `sparse_types.h` | Must agree with consumer `SPARSE_IDX_BITS`. |
| `sparse_scalar_bits` | `sparse_types.h` | Must agree with consumer scalar ABI. |

All eight exist in the static archive. This is a **candidate**, not an
approved public dynamic ABI. Day 4 must decide compatibility, error/ownership
semantics, and whether this subset is viable. A future decision may narrow or
replace it but must record an exact reviewed list before enabling shared
builds. The other 157 installed-header functions and 196 archive-only globals
are excluded from this proposed dynamic proof. They remain available through
the existing static package where supported; exclusion does not remove
declarations or alter static linking.

## Export And Header Ownership Policy

- Hidden-by-default visibility must apply to **shared-target objects** on
  supported ELF/Mach-O toolchains. Static archive behavior must remain intact.
- Only approved declarations would carry explicit export/import annotations
  or appear in an equivalent platform export file. Do not use
  `WINDOWS_EXPORT_ALL_SYMBOLS`, `sparse_*` wildcards, or automatic export of
  all non-static functions.
- A future source-controlled manifest must map each C name to its installed
  declaration owner and reviewed signature. Generated version macros are not
  exports.
- Linux needs an explicit export map/version script with non-allowlisted
  project symbols local; SONAME policy follows on Days 4-5. macOS must account
  for leading underscores in Mach-O export-list syntax and inspect the
  installed dylib. Windows needs a reviewed `.def` or `__declspec` policy and
  inspection of DLL exports and import-library bindings.
- Target names, install layout, and static/shared coexistence require a later
  product decision. Current `SparseConfig.cmake` and `sparse.pc` must not
  advertise the provisional list.

## Future Artifact-Level Acceptance Tests

These tests are specified, not implemented or passed. They must inspect a
built **and installed** shared artifact, not only source or static `nm` output.

| Test | Positive result | Required negative failure |
| --- | --- | --- |
| Declaration/manifest agreement | Every approved name has one reviewed installed-header declaration and expected signature/configuration. | Missing declaration, ambiguous ownership, or silent signature drift fails. |
| Actual dynamic exports | Normalize target-platform output and compare installed binary's defined project exports with the exact manifest. | Missing approved symbol or extra helper/test-hook export fails. |
| Import/link binding | Out-of-tree consumer links only against installed shared package metadata and calls approved functions. | Consumer requiring excluded symbol cannot resolve it through shared package. |
| Runtime round trip | Consumer checks index/scalar width, creates opaque matrix, sets/gets value, and frees it through library. | Width mismatch or absent runtime library fails clearly; link-only success is insufficient. |
| Static regression | Existing `tests/test_install.sh` and `tests/test_cmake_install.sh` still install, link, and run static consumers. | Export annotations or metadata must not break static linking or silently switch consumers to shared. |
| Platform expansion | Each claimed platform has export-table inspection and installed runtime consumer in reviewed CI. | Linux proof cannot count as macOS or Windows export/loader support. |

For the selected deferral path, the applicable executable checks remain
`BUILD_SHARED_LIBS=ON` rejection, the static package guard, static install
tests, and public non-claim checks. A later shared proof needs a checked-in
export manifest and automated installed-binary comparison on every claimed
platform. Source-only regexes, archive `nm` output, or a local link alone are
insufficient.

## Day 3 Completion And Handoff

The boundary is explicit for planning: 165 installed-header declarations are
the current static source-level surface; 196 additional archive globals are
not direct installed-header declarations; only the eight named opaque-matrix
functions are proposed for a future narrow dynamic proof. No shared export is
approved or installed now. Day 4 must resolve ABI version/compatibility,
layout, ownership, and width assumptions. Day 5 must record Linux, macOS, and
Windows loader/export expectations without promoting unsupported platforms.

## Day 3 Validation

`make source-list-check`, `bash scripts/static_package_deferral_check.sh`,
`git diff --check`, and new-file whitespace checks passed locally. Day 3
changed planning Markdown only; no `*.c` or `*.h` files changed, so the full
C quality gate was not required. The archive/AST comparison is local evidence,
not a hosted shared-library or dynamic ABI result.
