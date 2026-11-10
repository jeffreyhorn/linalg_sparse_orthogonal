# Sprint 222 Day 4: CHOLMOD Runtime Identity

## Verdict

**Candidate library reconciled; factor/runtime provenance still incomplete.** The Day 3 helper's `dladdr(cholmod_start)` and dyld loader trace both resolve `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib`, whose SHA-256 is the pinned candidate digest `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`. A separate initialization-only ABI probe resolves the same library and reports CHOLMOD `5.3.4`, `CHOLMOD_INT` backed by a four-byte `int`, and a successful lower CSC allocation. No factor or solver was run. Requested natural/simplicial/no-postorder controls are visible in `cholmod_common`; **effective factor ordering and mode remain unobserved**.

The raw, ignored local bundle is:

`build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day4-baseline-runtime-20261110T204702Z-t0u_3ynb/`

`runtime-inventory-manifest.json` SHA-256: `a9e349cd9310f58ce121f9b6185033872440106f6443a5848d93730d7ee1bcb9`. It binds the probe outputs, full loader traces, ten non-system loaded images, selected system image UUIDs, package receipt, symbol inventory, component versions and `otool` dependency declarations. Keep these raw files local under the Day 2 rights boundary.

## Package and source identity

| Item | Captured evidence |
| --- | --- |
| Upstream selection | [SuiteSparse v7.12.2 release](https://github.com/DrTimothyAldenDavis/SuiteSparse/releases/tag/v7.12.2), tag ref `42151688813c45846a597edcb601435a0e38f3dd` confirmed by `git ls-remote`; release lists CHOLMOD 5.3.4, AMD 3.3.4, COLAMD/CAMD/CCOLAMD 3.3.5, SuiteSparse_config 7.12.2. |
| Installed acquisition | Homebrew prefix `/usr/local/Cellar/suite-sparse/7.12.2`; `INSTALL_RECEIPT.json` says `built_as_bottle=true`, `poured_from_bottle=true`, compiler `clang`; receipt SHA-256 `9a1580758bc23f829a5df7bb2d609724d37d00eded84e1e3fdd840a23d54e98f`. This records package acquisition, not a reproducible compile-command ledger for the bottle. |
| Installed header and metadata | `cholmod.h` SHA-256 `f06e51fee9677ec48f650140aad588c565f793b6d60d45d2f9583cea8fa30bf1`; `pkg-config --modversion cholmod` and installed CMake config both say 5.3.4. The probe's runtime `cholmod_version()` returned `5.3.4`. |
| Candidate versus loaded | Pinned candidate path, helper `dladdr` path, probe `dladdr` path, and dyld loaded path are identical. The file hash above matches Sprint 215's candidate; no replacement package identity was observed in these probes. Recheck before a later solver attempt. |

The helper loader probe ran `cholmod_pilot project __s222_intentionally_missing_fixture__` with `DYLD_PRINT_LIBRARIES=1`; it intentionally exited `2` after printing the loaded CHOLMOD path and `load_status=10`. This is an expected input rejection, **not a solver failure**: `main` reaches no factorization on this path. A separate `cholmod_abi_probe.c` in the ignored bundle called `cholmod_version`, `cholmod_start`, and `cholmod_allocate_sparse`, then freed its allocation. It did not call `cholmod_analyze`, `cholmod_factorize`, or `cholmod_solve`. Both probes retained raw stdout/stderr and exit-code files. The ABI probe compiled with `-Werror` and exited `0`.

## Loaded non-system dependency inventory

The paths below were extracted from the dyld trace of the initialization probe and were also present in the helper loader trace. The hashes are of the **actual files at the resolved loaded paths**. `required-loaded-dependencies.json` SHA-256 is `92a42aef507f3fb7168bd97628380fa025f5b5d35a0db2d2b86988c21555eaf4`; raw dyld traces remain in the bundle. Each image's UUID and separate `otool -L` output are also retained.

| Role | Loaded file / version | SHA-256 |
| --- | --- | --- |
| CHOLMOD | `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib` | `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c` |
| AMD | `/usr/local/Cellar/suite-sparse/7.12.2/lib/libamd.3.3.4.dylib` | `8e8f4ced0170297beaf78b9bae9ee3c78c10d4db7579d615ed0a38d23e2ec3fa` |
| COLAMD | `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcolamd.3.3.5.dylib` | `c00ad32dd58b206f3483c4fc9dca24807c6820a6b5a2e591fbb98cdc1bf403ed` |
| CAMD | `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcamd.3.3.5.dylib` | `d400e6864c72159ba3a6f3301ceaef0145c478e49539870453c6ab9b728d4ed8` |
| CCOLAMD | `/usr/local/Cellar/suite-sparse/7.12.2/lib/libccolamd.3.3.5.dylib` | `6b07baccffa46dbb5bea424355d9e77b3751ba5f02a2a677fa6d264e12bd337a` |
| SuiteSparse_config | `/usr/local/Cellar/suite-sparse/7.12.2/lib/libsuitesparseconfig.7.12.2.dylib` | `44dd9b69bcf8ab4b12417864deb04c7b5ef23316225774c9a4171b5435f93b1d` |
| OpenMP | `/usr/local/Cellar/libomp/22.1.1/lib/libomp.dylib` | `8599c7fa645d686ec48bdec796059dce6227a06c6e39162d6d54bac3f8a854fb` |
| Fortran runtime | `/usr/local/Cellar/gcc/15.2.0_1/lib/gcc/current/libgfortran.5.dylib` | `427716b5b9b314eea15ac87655562d665537a68104985faeec37740f2ee199fe` |
| GCC runtime | `/usr/local/Cellar/gcc/15.2.0_1/lib/gcc/current/libgcc_s.1.1.dylib` | `330ab8338fb8c8a8e7474eff607f6272d98743ca797e6f7ba2501918f2919349` |
| Quadmath runtime | `/usr/local/Cellar/gcc/15.2.0_1/lib/gcc/current/libquadmath.0.dylib` | `7b025284681e72ecea2a1fc9a802972637c05303d17edffbd37024835228d9a0` |

The trace also lists macOS 15.7.9 build `24G830` system images. In particular, loaded Accelerate has UUID `19FE3E25-5E58-33D8-8887-C696D3D7083B`, its `libBLAS.dylib` UUID is `F6B5187B-FD0A-36FB-AF99-D96DE8E76E73`, and `libLAPACK.dylib` UUID is `82F14ACC-7F8F-3D2C-A13A-C485B549C67A`. These paths are in Apple's dyld shared cache and are not ordinary readable files at their listed paths, so no honest per-image SHA-256 can be reported. `selected-system-images.json` retains path, UUID, category and `sha256=null`; the raw trace and `sw_vers` output retain their context. The complete loader trace includes other Apple system images, not all of which are relevant dependency roles. Do not substitute the framework path, a symlink or an unrelated SDK file hash for a loaded-image digest.

## ABI, module, and factor controls

| Control | Requested or observed evidence | Status |
| --- | --- | --- |
| CHOLMOD integer and value storage | Runtime `sizeof(int)=sizeof(int32_t)=4`; `CHOLMOD_INT=0`; allocated sparse `itype=0`, `dtype=0` (`CHOLMOD_DOUBLE`), `xtype=1` (`CHOLMOD_REAL`), `stype=-1`, `sorted=1`, `packed=1`, `common.status=0`. | **Observed in initialization-only probe**; matching the selected int32/real-double/lower CSC ABI. |
| Ordering and factor mode | Probe set `nmethods=1`, `method[0].ordering=CHOLMOD_NATURAL=0`, `postorder=0`, `supernodal=CHOLMOD_SIMPLICIAL=0`. Day 3 helper source sets the same controls. | **Requested only**. No factor exists yet; later factor `ordering`, `is_super`, `minor`, `is_ll`, and statuses must be captured and checked. |
| Compiled CHOLMOD modules | `nm -gU` on the loaded dylib finds `cholmod_super_numeric`, `cholmod_super_symbolic`, `cholmod_add`, `cholmod_ssmult`, `cholmod_rowadd`, `cholmod_rowdel`, and `cholmod_metis`, plus selected solve APIs. | Presence of exported symbols indicates these paths are in the binary; it does not prove which paths will execute, full build flags, or METIS availability. The Day 2 rights ledger must treat GPL module terms as potentially applicable to binary redistribution. |
| Build options | Bottle receipt identifies clang and acquisition; installed CMake config gives version and target imports but no complete `CHOLMOD_GPL`, OpenMP/CUDA, optimization or module CMake cache. The dylib links libomp and exports module symbols. | **Unverified exact flags/module configuration**. Do not claim an LGPL-only or fully reproducible baseline build from link flags or requested simplicial mode. |

## Open gates and next owner

1. **Day 4 provenance blocker:** exact bottle build flags and complete module configuration are unavailable from inspected installed metadata. Obtain a versioned Homebrew bottle recipe/build attestation or reproduce a pinned source build with a retained CMake cache and binary hash. If identity differs, create a new cell/attempt, never relabel this one.
2. **System-library hash limitation:** loaded BLAS/LAPACK are dyld shared-cache images, identified by path/UUID and OS build but not by per-file SHA-256. The Sprint 215 method asks for loaded dependency hashes. Decide and document an evidence-equivalent Apple shared-cache fingerprint, or use a hashable pinned BLAS/LAPACK runtime before asserting complete v1 provenance. A missing required dependency digest blocks promotion under the strict reading of the frozen method.
3. **Factor-control gate:** only a later, fresh two-sided solve may establish effective ordering, simplicial mode, factor minor, statuses, residuals and oracle checks. Day 5 owns host/thread freeze before that attempt. A later run must recheck all loaded paths and hashes because this is a Day 4 snapshot.
4. **Rights gate:** loaded libgfortran/libgcc_s/libquadmath add transitive terms for any binary bundle. Update the Day 2 rights review before publication; keep all generated evidence local in the meantime.

This Day 4 artifact provides the named ABI and factor-control blocker required by the plan. It makes no observed correctness, timing, redistribution or publication claim.
