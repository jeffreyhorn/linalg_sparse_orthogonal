# Sprint 165 Day 5 ABI Non-Claim Audit

## Purpose

Day 5 audits public headers, version/package docs, package metadata, and
generated-reference owners for ABI wording. The goal is to separate source API
compatibility, package version metadata, and binary ABI support before Day 6
wording cleanup.

## Surfaces Inspected

| Surface | Files | ABI-Relevant Finding |
| --- | --- | --- |
| Public package front door | `README.md` | Correctly treats package checks as package proof, not package-manager distribution or dynamic-loader evidence. Shared-library packaging remains deferred. |
| Operational package docs | `INSTALL.md` | Correctly states static-first package contract, exact version metadata, shared-library deferral, dynamic ABI non-claim, runtime-loader non-claim, and Windows CMake-first support. |
| Maintainer package policy | `docs/maintainer_guide.md` | Correctly states package-version metadata should not be described as a broad dynamic ABI guarantee. |
| API reference policy | `docs/api_reference.md` | Correctly states API reference does not imply dynamic ABI compatibility, shared-library support, package-manager distribution, or broad platform parity. |
| Tutorial/cookbook/solver selection | `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md` | Mentions package/ABI only as non-claim boundaries around local evidence and solver-selection guidance. |
| CMake package metadata | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in` | Correctly uses exact package version compatibility while noting it is not a dynamic ABI guarantee. |
| pkg-config metadata | `sparse.pc.in` | No ABI wording; static archive metadata only. |
| Public headers | `include/*.h` | Mostly uses source rebuild/layout wording. `include/sparse_cholesky.h` still uses an older "ABI break" phrase. |

## Source API, Package Metadata, And Binary ABI Distinction

| Category | Supported Today | Non-Claim Boundary |
| --- | --- | --- |
| Source API compatibility | Checked-in public declarations, documented option/result structs, zero-initialized option structs where documented, and public enum/macro names are source API contracts. | Source compatibility does not imply binary compatibility across previously compiled objects. |
| Package version metadata | `VERSION` is propagated into generated `sparse_version.h`, `SparseConfigVersion.cmake`, and `sparse.pc`; CMake package compatibility is exact-version only. | Exact package metadata is not a dynamic ABI promise and does not define soname/install-name/DLL policy. |
| Static archive package behavior | Make/CMake install static archive, public headers, CMake package metadata, and `sparse.pc`; downstream consumers build from installed headers and static archive. | Static archive install/export proof does not imply shared-library support or runtime-loader behavior. |
| Binary ABI support | No supported dynamic ABI contract exists today. | Stable binary struct layout, downstream object compatibility across versions, shared-library ABI, symbol visibility, loader metadata, and package-manager ABI policies are deferred. |

## ABI-Adjacent Wording Findings

| Location | Current Wording Pattern | Assessment | Replacement Candidate |
| --- | --- | --- | --- |
| `include/sparse_cholesky.h` | `@warning **ABI break in v2.0.0.** Adding backend and used_csc_path changed this struct's size...` | Ambiguous. It accurately warns that older binaries must be recompiled, but the phrase "ABI break" can be read as implying the project otherwise maintains an ABI policy. | `@warning **Source rebuild required for v2.0.0 options layout.** The backend and used_csc_path fields were added after the v1.x fields. Source initializers that zero-initialize sparse_cholesky_opts_t remain valid, but downstream objects compiled against the older struct layout must be rebuilt.` |
| `include/sparse_ldlt.h` | `@warning **Source rebuild required for v2.1.0 options layout.** ... downstream objects compiled against the older struct layout must be rebuilt.` | Good pattern. It communicates rebuild requirement without implying a broader ABI policy. | Keep. |
| `include/sparse_eigs.h` | `@warning **Source rebuild required for v2.2.0 options/result layout.** ... downstream objects compiled against the older struct layout must be rebuilt.` | Good pattern. It distinguishes source initializer behavior from compiled-object layout. | Keep. |
| `CMakeLists.txt` | Comment says exact package-version compatibility does not claim a broad dynamic-ABI guarantee. | Good package metadata boundary. | Keep. |
| `INSTALL.md` and `docs/maintainer_guide.md` | Exact package version metadata is described separately from dynamic ABI compatibility. | Good public/maintainer boundary. | Keep. |

## Replacement Language Candidates

Use this wording when cleaning ABI-adjacent public docs or headers:

```text
Source rebuild required for <version> <options/result> layout.
```

```text
Source initializers that zero-initialize the struct remain valid, but
downstream objects compiled against the older struct layout must be rebuilt.
```

```text
Exact package version metadata is used for installed package resolution. It is
not a dynamic ABI compatibility guarantee.
```

```text
The maintained package surface is the installed static archive plus headers and
metadata. Shared-library packaging, loader behavior, and dynamic ABI support
remain deferred product decisions.
```

Avoid this wording unless a future product decision creates a real ABI policy:

- `ABI compatible`
- `ABI stable`
- `stable ABI`
- `binary compatible`
- `soname policy`
- `shared ABI`
- `ABI guarantee`
- `compatible binaries`

## Deferred ABI Product Decision Register

| Deferred Decision | Required Future Evidence |
| --- | --- |
| Dynamic ABI compatibility policy | Public symbol policy, struct layout/versioning policy, compatibility matrix, downstream binary tests, and release rules. |
| Shared-library support | Export/import macros, symbol visibility policy, generated shared target metadata, installed shared consumers, platform loader validation, and packaging docs. |
| Linux SONAME support | `SOVERSION`/SONAME policy, install metadata, runtime loader tests, and downstream shared consumer proof. |
| macOS install-name/RPATH support | install-name policy, RPATH behavior, loader validation, and downstream shared consumer proof. |
| Windows DLL/import-library support | export/import decoration, import library validation, DLL placement policy, runtime loader tests, and downstream shared consumer proof. |
| Package-manager distribution ABI policy | Provider-specific packaging files, package-manager install tests, version/upgrade behavior, and explicit support-tier docs. |

## Day 6 Cleanup Handoff

Day 6 should apply a narrow wording cleanup:

1. Update `include/sparse_cholesky.h` to replace the older "ABI break" phrase
   with the source-rebuild/options-layout wording used by `sparse_ldlt.h` and
   `sparse_eigs.h`.
2. If the public header changes, preserve declarations and run the required
   `.h` quality gate: `make format && make lint && make test`.
3. Re-scan touched files for `ABI break`, `ABI compatible`, `ABI stable`,
   `binary compatible`, `soname policy`, and `ABI guarantee`.
4. Keep exact package version wording in `CMakeLists.txt`, `INSTALL.md`, and
   `docs/maintainer_guide.md` unchanged unless a concrete drift is found.

## Validation Notes

Day 5 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 5.

## Completion Check

- Accidental ABI-promising wording was identified.
- Version metadata remains separated from dynamic ABI support.
- Replacement wording is ready for scoped Day 6 cleanup.
