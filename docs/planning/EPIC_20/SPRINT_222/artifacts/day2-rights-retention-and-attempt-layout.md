# Sprint 222 Day 2: Rights, Retention, and Attempt Layout

## Decision

**Publication review: pending.** Continue the correctness pilot only as an ignored, local, append-only attempt. This ledger identifies source terms and likely obligations, but does not approve redistribution of the CHOLMOD binary, its dependencies, raw logs, or an external comparison claim. The maintainer/publication reviewer must sign an exact-bundle decision after runtime module composition, transitive licenses, privacy content, and numerical evidence are known. No solver was run for this review.

## Fixture origin and rights ledger

| Item | Evidence | Retention and publication disposition |
| --- | --- | --- |
| Exact selected fixture | `tests/data/symmetric_4.mtx`, 114 bytes, SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`; unchanged since commit `aa1c04eab3536031b9a92ee92587260520ce1913` on 2026-03-21. `git show <commit>:tests/data/symmetric_4.mtx` hashes identically. | Already public in the repository. Retain the original bytes and hash; do not substitute a downloaded matrix or a rewritten file. |
| Attribution | The introducing commit names Jeffrey Horn and describes eight created reference matrices. The file has no separate upstream attribution or license header. The repository `LICENSE` is MIT, copyright 2026 Jeffrey Horn. These facts support repository provenance, but the commit message alone does not establish independent authorship of each matrix entry. | Record “repository fixture, introduced by Jeffrey Horn” and repository commit in metadata. Keep v1 `attribution_review=pending` until the maintainer confirms original creation or any external source/terms, and confirms the intended public attribution wording. A public copy of the fixture must carry the applicable MIT notice if it is a copy or substantial part of the licensed work. |

The `HB/nos4` SuiteSparse Collection matrix is outside this cell. Its collection attribution and terms must not be silently applied to `symmetric_4.mtx`.

## External package and dependency ledger

| Component | Evidence checked on 2026-11-10 | Local retention / redistribution gate |
| --- | --- | --- |
| SuiteSparse source and installed package | Selected tag `v7.12.2` / source commit `42151688813c45846a597edcb601435a0e38f3dd`; local `pkg-config --modversion cholmod` = `5.3.4`, prefix `/usr/local/Cellar/suite-sparse/7.12.2`. Installed `LICENSE.txt` SHA-256 `c07f7bb69aa870c774bd01f040dabd179b7cc8c60021c2c56db9d63d37deabcb`. See [upstream tagged license bundle](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.12.2/LICENSE.txt) and [tagged build options](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.12.2/README.md). | Package-wide “SuiteSparse license” is inadequate: modules have separate terms. Keep the installed license text and source URL as local provenance. Public binary redistribution needs an exact component/notice inventory and obligations review. |
| Candidate CHOLMOD dylib | `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib`, SHA-256 `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`. This matches Sprint 215 discovery, **not** a runtime-loaded identity proof. | The local pilot may inspect and hash the installed file and keep an ignored local candidate copy for byte-backed validation. Do not commit or publish that binary before rights and loaded-path review. |
| CHOLMOD modules | Installed `LICENSE.txt` lists Check, Cholesky, Utility and Partition under LGPL 2.1 or later; MatrixOps, Modify and Supernodal under GPL 2 or later; Include `cholmod.h` and related headers under Apache 2.0. The tagged [CHOLMOD module license index](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.12.2/CHOLMOD/Doc/License.txt) and README specify that `CHOLMOD_GPL=OFF` omits GPL modules. The pilot requests simplicial factorization, but that request does not prove the binary omitted Supernodal or other GPL code. | Determine the **actual installed binary's** compiled modules and build flags, including optional METIS/GPU/Partition, before binary distribution. Do not infer LGPL-only terms from the selected simplicial method. Preserve full relevant license notices and meet any source/relinking obligations identified by the reviewer. |
| AMD, COLAMD, CAMD, CCOLAMD | `otool -L` lists `libamd.3`, `libcolamd.3`, `libcamd.3`, `libccolamd.3`. The installed license bundle identifies their respective entries as BSD 3-Clause. | Linkage is a candidate dependency list, not runtime-loaded path proof. For redistribution, identify exact loaded files and reproduce notices, disclaimers, and no-endorsement terms. |
| SuiteSparse_config | `otool -L` lists `libsuitesparseconfig.7`; exact loaded path and component license were not established by the installed bundle inspection. | Keep role and hash as local metadata after runtime capture; obtain component-specific terms before distributing its binary. |
| BLAS/LAPACK and OpenMP | Candidate `otool -L` lists macOS Accelerate, `/usr/local/opt/libomp/lib/libomp.dylib`, and system `libSystem.B.dylib`. Homebrew metadata for installed libomp reports MIT, but its exact loaded version and `LICENSE.TXT` still need capture. | Do not bundle Apple system libraries. If bundling libomp or another runtime, inspect its exact license/notice and binary; record system-library use separately. Validate loaded dependency graph, not just linker declarations. |

The candidate library's `otool -L` output is evidence of declared linkage only. It cannot establish actual load paths, absence of embedded optional code, or the full transitive dependency graph. The upstream tag and installed package license are terms sources; neither establishes compliance for an as-yet-undefined public bundle. The initial Sprint 215 survey's broad claim about CHOLMOD headers being LGPL is too coarse for the installed `LICENSE.txt`, which labels `cholmod.h` Apache 2.0. Use component/file-specific terms.

## Local bundle boundary and redaction

Proposed unique path: `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/<UTC-YYYYMMDDThhmmssZ>-<unique-run-id>/`. Create it once with exclusive directory creation; never reuse or overwrite a failed attempt. Keep `attempt.json`, raw per-command stdout/stderr, command/exit manifest, matrix and control hashes, solver outputs, binary/archive copies needed for byte-backed validation, generated v1 record, and validator diagnostics in that local directory. Copy exact binary bytes only into this ignored local bundle after recording origin and hash. A follow-up run receives a new directory and links back to the prior attempt by ID; it does not replace it.

Local review can retain the repository fixture, ordinary commands, numeric outputs and hashes in this ignored workspace while the rights decision is pending. The rights ledger does not authorize uploading the local bundle, attaching it to a PR, or distributing bundled binaries. Public metadata may eventually use hashes, versions and license/attribution references without bundling third-party bytes, but even that claim needs an exact-record publication review. Before any disclosure, inspect raw streams and manifests for user names, home paths, host identifiers, environment secrets, package credentials, serials and other private data. Preserve original raw bytes and hashes locally; create a separately hashed redacted derivative if approved. Never silently redact the evidence being validated, because its hashes would no longer match.

Proposed directory roles: `inputs/` for frozen source/control manifest, `build/` for retained executable/archive and build logs, `baseline/` for candidate/loaded-library identity and license inventory, `raw/` for each solver's stdout/stderr and status, `validation/` for v1 record and validator diagnostics, `review/` for a later exact-bundle rights decision. These are a layout proposal; the current pilot writes flat files and `attempt.json`, so implementation must adapt it deliberately without changing the frozen cell. The v1 `artifacts` list must use real normalized bundle-relative files, hashes, media types and roles; a markdown plan does not count as a generated attempt.

## Frozen source and control manifest

| Input or control | SHA-256 / identity |
| --- | --- |
| Fixture | `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335` |
| Day 3 identity and conversion contract | `5bd0708e36816135ed99a95a8b28fd23b7b43876a8956ce64c830f3139eb9943` |
| Day 4 correctness contract | `7ca1249219cb2842a9a4827ecdeb57024a5a03b2b0762400988051505c91483e` |
| Day 5 optional timing methodology (timing remains not run) | `bcee23be60b8612f91951f6afa94ab7d48f7c85085b43c97319ff8d614c49235` |
| V1 schema | `7a0c61ef6ea47a68da3caf34a378e73f5305736c518ee76731fbed2d8865bfe4` |
| Pilot C / CSC guard / runner | `28e736f57087782459ca4afe9c8f00232b279764cdf85c4c6660062ede1fa50a` / `50cd5925b74af82c3049bb76ebd3b59802b9224f0b07dad34301de8308dbe466` / `fe151b9265de2ef4f1d5509fc11410a30e1ebafd00afb0bb59c8e56add0277c7` |
| Candidate CHOLMOD dylib / installed license bundle | `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c` / `c07f7bb69aa870c774bd01f040dabd179b7cc8c60021c2c56db9d63d37deabcb` |

These are Day 2 snapshot identities, not substitutes for Day 3-5 runtime capture. Recheck each source/control byte before execution; changed inputs require a new decision or record rather than a silently updated hash.

## Reviewer and unresolved questions

**Owner:** repository maintainer for fixture attribution and proposed public wording; a rights reviewer designated by the maintainer for the exact external-binary/license bundle. No approval is recorded.

1. Confirm whether Jeffrey Horn created `symmetric_4.mtx` independently, whether another matrix inspired its values, and the exact attribution/notice required for a copied fixture.
2. Establish compiled CHOLMOD module composition, including GPL modules and METIS, from package build metadata or a reproducible source build. Confirm all license/notice and source availability duties for any binary distribution.
3. Identify actual loaded transitive libraries and their terms, especially SuiteSparse_config and libomp. Determine whether any non-system library bytes are necessary in a public artifact; prefer metadata-only disclosure if sufficient.
4. Review the exact proposed evidence bundle and wording after numerical validation, including private data in raw logs. Record a separate hashed `publication_review` artifact only for that exact bundle. Until then `publication_review=pending`, `attribution_review=pending`, and no public correctness claim is earned.

**Item 222.1 status:** Day 1 fixture/evidence inventory plus this Day 2 rights ledger account for the planned 20-hour work scope. The fixture and controls are frozen; rights and publication remain explicitly pending. Local execution may proceed inside the recorded boundary; publication cannot.

**Day 4 runtime addendum:** The loader inventory subsequently found `libgfortran.5.dylib`, `libgcc_s.1.1.dylib` and `libquadmath.0.dylib` from Homebrew GCC 15.2.0_1, in addition to the Day 2 candidates. Exact paths and hashes are in [Day 4's runtime ledger](day4-cholmod-runtime-identity.md). Their distribution terms must be reviewed for any binary bundle. Exported CHOLMOD Supernodal, MatrixOps and Modify symbols were present in the loaded dylib; requested simplicial execution does not remove those compiled symbols or settle GPL obligations. Publication stays pending.
