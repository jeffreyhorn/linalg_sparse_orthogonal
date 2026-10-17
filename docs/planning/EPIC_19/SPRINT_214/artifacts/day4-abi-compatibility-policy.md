# Sprint 214 Day 4: ABI Version And Compatibility Policy

## Status And Evidence

Day 2 selected ABI/release design with stronger static-first deferral. This
document defines **future dynamic ABI acceptance gates**. It does not assign
an ABI epoch, enable a shared target, change installed metadata, or promise
compatibility. Day 3's eight-function opaque-matrix allowlist remains
provisional.

The current `VERSION` is `2.2.0`. Make and CMake generate
`sparse_version.h` macros from it; `sparse.pc` reports a package version;
CMake writes `SparseConfigVersion.cmake` with `ExactVersion` matching. A local
`bash tests/test_cmake_install.sh` run passed 27/27 checks, including
exact-version installed consumer execution and mismatched package-version
rejection. These are **package resolution tests**, not old-binary/new-library
tests. No dynamic ABI identifier or shared artifact exists today.

## Version Vocabulary And Promise Level

| Term | Meaning | Current status |
| --- | --- | --- |
| Source/API version | `VERSION` and generated `SPARSE_VERSION_*` macros identify the source package. | `2.2.0`; no binary compatibility implication. |
| Package version | `sparse.pc` and CMake `ExactVersion` select an installed static package. | Tested, but not an ABI epoch. |
| Dynamic ABI identity | Future approved epoch plus index/scalar configuration and platform ABI constraints bound to an installed binary. | Undefined; no shared package installed. |
| Same-build shared proof | Consumer compiled with matching installed headers links and runs against the matching shared artifact on one named platform. | Not implemented; would not prove cross-version compatibility. |
| Backward compatibility | An old retained consumer binary runs unchanged with a newer library in the same supported ABI family. | Unclaimed until old-binary/new-library tests pass. |

A first shared proof, if selected by a later product decision, may claim only
**same-build, named-platform execution** until a separate compatibility gate
passes. `VERSION` patch/minor/major changes do not automatically imply binary
compatibility. An ABI epoch denotes an intended compatibility family, but its
number alone is not evidence. New-consumer/old-library compatibility is a
separate direction, not implied by old-consumer/new-library evidence.

## Future ABI Identity Contract

Before a shared target is installable, choose a positive ABI epoch and a
source-controlled manifest for the approved export subset. Minimum fields:

| Field | Required gate |
| --- | --- |
| `abi_epoch` | Positive integer selected with the first approved shared product decision; undefined in today's static-only package. A break needs a new epoch or a tested compatibility shim. |
| `idx_bits` | Compiled `SPARSE_IDX_BITS` (32 or 64); `idx_t` width changes signatures and must not share a binary identity. |
| `scalar_abi` | Scalar kind and width, currently real `double`/64-bit; width alone does not identify every future representation. |
| Export manifest | Exact approved names, declaration owners, signatures, and aliases; review every diff before binary promotion. |
| Platform ABI | OS, architecture, calling convention, object format, and Windows compiler/CRT/allocator model for each claimed lane. |
| Loader/package binding | Installed artifact name, SONAME/install-name/DLL identity, and package selection rules that resolve to that artifact; Day 5 owns final platform encoding. |

A future generated installed header and package metadata must expose the
chosen epoch/configuration separately from `SPARSE_VERSION`. The installed
binary, metadata, and consumer compile settings must agree. Loader identity
must prevent accidental substitution of a different epoch or width; if a
filename cannot encode every field, configure and runtime checks must reject
mismatch before selected API use. No `SPARSE_ABI_*` macro or metadata is
added to the current static package by this policy.

For Day 3's proposed subset, `SparseMatrix` is opaque and
`sparse_create`/`sparse_free` must keep allocation and release inside the
library. `sparse_set`/`sparse_get` use `idx_t` and `sparse_scalar_t`, so both
consumer and library must agree on their representations. `sparse_get`
currently returns zero for a NULL matrix, invalid index, or absent entry;
changing that is a behavioral contract change even with an unchanged C
signature. `sparse_err_t` numeric values and representation,
`sparse_strerror()` string lifetime, and `sparse_errno()` semantics need
review. The current implementation uses `_Thread_local` errno state, not yet
a promoted dynamic ABI guarantee.

## Change Classification

| Change | Future ABI treatment |
| --- | --- |
| Internal implementation change with identical approved exports, signatures, metadata, and behavior | Potentially compatible, subject to retained old-consumer execution. Archive-only helpers are not automatically ABI. |
| Add approved export | Old consumers may remain compatible if existing contracts and tests pass; a new consumer using the added symbol cannot run on an old library lacking it. Update manifest and minimum package version. |
| Remove/rename approved export or change its signature/return type | Breaking unless a compatible old symbol/shim remains and old binaries pass; otherwise new epoch/loader identity and no same-epoch release. |
| Change `idx_t`, scalar ABI, selected error enum value/representation, calling convention, or selected callback signature | Breaking; use distinct identity and reject mismatch, not merely source recompilation. |
| Change selected concrete struct size/alignment/field order or enum width | Binary break, including append-only fields. The provisional eight-function subset excludes concrete factor/result/option structs and callbacks. |
| Change opaque `SparseMatrix` internal layout | Not itself a public layout break if only approved accessors touch it; lifecycle and behavior still need old-consumer proof. |
| Change allocation/free pairing, borrowed lifetime, error semantics, or `sparse_get` silent-zero behavior | Potential behavioral break even with unchanged signatures; block until reviewed and tested. |
| Change loader identity or import library | Breaking if old binaries cannot locate/link the new artifact; inspect installed metadata and run old consumer. |

The other 157 installed-header functions and 196 archive-only globals do not
become dynamic ABI merely because they have declarations or external linkage.
Widening dynamic scope requires a reviewed manifest update and separate
product decision.

## Future Positive And Negative Tests

These are specifications, not implemented tests. Cross-version tests must
use artifacts from **distinct builds**, not just rebuild current sources.

| Test | Required success | Required failure fixture |
| --- | --- | --- |
| Current static package | Make/CMake install tests retain static archive, metadata, exact-version consumer, and mismatch rejection. | Shared artifact/selector or ABI claim without selection fails the static guard. |
| ABI metadata completeness | Installed header/package metadata, manifest, and binary identify the same epoch, index/scalar ABI, and platform. | Remove an identifier or alter one surface; validation must fail rather than infer ABI from `VERSION`. |
| Width/scalar mismatch | Consumer for approved tuple runs with matching artifact. | 64-bit-index consumer against 32-bit library, or altered scalar identity, is rejected before selected API use. |
| Epoch mismatch | Matching epoch resolves package and loader identity. | Old-epoch consumer against new-epoch-only install fails predictably unless an intentionally shipped shim passes tests. |
| Export/signature drift | Installed export table and declaration manifest agree exactly. | Missing/renamed approved export, changed signature, or extra helper export fails. |
| Backward compatibility | Retained old executable built against baseline N runs unchanged with library N+1 and checks values, errors, and ownership on every claimed platform. | Deliberately incompatible library or missing loader fails; a freshly rebuilt consumer cannot substitute for the old binary. |
| New consumer against old library | Tested only if separately claimed and version-gated. | Added symbol is unavailable in old library and must not be described as compatible. |

`sparse_idx_bits()` and `sparse_scalar_bits()` help runtime probes but cannot
encode epoch, calling convention, enum representation, or loader identity. A
successful link or same-source rebuild never proves binary compatibility.

## Reference Baseline And Retention

There is **no prior supported shared release** from which to manufacture a
cross-version baseline. The first approved shared release must retain:

1. Immutable source/tag and generated installed headers, with source/package
   version and ABI identity manifest.
2. Installed shared binary and import library where applicable, export table,
   loader metadata, CMake/`pkg-config` files, checksums, and build/platform
   provenance.
3. Old consumer source **and compiled executable**, compile/link flags,
   width/scalar selection, runtime environment, expected output/errors, and
   CI result.
4. A retention location, duration, and owner surviving every supported
   compatibility window; Days 11-12 own the release checklist details.

For a same-epoch update, compare old baseline with the new installed binary:
export and metadata diffs, an ABI-diff tool where available, and executable
old-consumer/new-library tests on each claimed platform. Test static consumers
separately. An ABI-diff tool cannot replace behavior or loader execution.
Do not delete a baseline while a compatibility claim depends on it.

## Breakage, Rollback, And Release Blockers

Before public shared promotion, a failed symbol, metadata, or consumer check
means retain static-first packaging, disable any experimental shared path,
and keep shared claims deferred. Do not weaken the static guard to accept an
incomplete build. After a shared release, silently deleting its artifact is
not a safe rollback: preserve the compatible line, ship a reviewed shim, or
issue a new epoch with an explicit migration notice.

Block shared or release-readiness claims for any of these conditions:

- missing or contradictory ABI identity, export manifest, loader name, or
  installed metadata;
- unreviewed selected layout/calling-convention/ownership break, width/scalar
  mismatch, or old-consumer failure inside a claimed compatibility family;
- missing retained baseline, provenance, or CI result for a claimed
  cross-version/platform tier;
- failing static install/export regression, deferral guard, or required
  C/build/docs quality check;
- unfulfilled release archive, tag, checksum, or provenance gates from
  Days 11-12.

## Day 4 Handoff And Validation

Day 5 must define how Linux SONAME, macOS install-name/RPATH, and Windows
DLL/import-library metadata encode or enforce the future ABI identity, while
keeping all three unclaimed today. The Day 3 eight-function list stays
provisional until that policy and a new proof decision exist.

`bash tests/test_cmake_install.sh` passed locally (27 passed, 0 failed, 0
skipped), confirming the existing static package and exact-version behavior.
Day 4 changed planning Markdown only, so no C/header quality gate was
required. Markdown whitespace validation is recorded in `WORKING_NOTES.md`.
