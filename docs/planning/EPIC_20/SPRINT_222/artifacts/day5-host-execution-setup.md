# Sprint 222 Day 5: Host and Independent Execution Setup

## Outcome and evidence boundary

**Preparation complete; the v1 observed cell remains blocked.** The validator self-test passed before Day 5's local probe and staging work. The frozen fixture, method, source inputs, retained project binaries, and ten non-system loaded dependency files matched their recorded hashes. Two byte-identical fixture copies and separate project/baseline working and raw-output directories are ready. **No solver was executed, no timing was measured, and no v1 cell ID was assigned.**

The ignored, local-only evidence bundle is `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day5-host-setup-20261110T205302Z-r58kyq4a/`. Its `day5-setup-manifest.json` SHA-256 is `56515f31f7afcaa804118f9eeefac6fac1fbd0846694442de07af27ce754a56b`. The manifest covers the raw command outputs, preflight, host record, source snapshot, staged artifacts, partial key, both input copies, and run recipe. Keep the bundle under the Day 2 rights and local-retention boundary.

## Host and execution manifest

`host-environment.json` (SHA-256 `aa356d6bb429ced60646483c8d6097c1c89efefecda771ac1175a5187d8ed564`) was captured at `2026-11-10T20:54:21Z`. `host-commands.json` retains the exact argv, exit status, stdout and stderr paths for the host queries; the manifest hashes those streams.

| Field | Captured value |
| --- | --- |
| OS and architecture | macOS 15.7.9, build `24G830`, `x86_64`; MacBookPro15,3. |
| CPU and RAM | Intel Core i9-9980HK @ 2.40 GHz; 8 physical / 16 logical cores; 34,359,738,368 bytes (32 GiB) RAM. |
| Compiler and runtime | Apple clang 17.0.0 (`clang-1700.6.4.2`); Apple `libSystem.B.dylib` dyld UUID `1317B5A4-98B3-31DD-8261-3C2AF7AEC2C2`, as seen in Day 4. |
| Working directory | `/Users/jeff/experiments/linalg_sparse_orthogonal`. |
| Host fingerprint | SHA-256 `3f4db376401dc48cf182656aa1f31f2bf41234de468a7d65a2030777b40e7e70` of hostname, OS build, architecture and CPU model joined in that order; raw hostname omitted for privacy. |
| Clock, power and locale | `mach_absolute_time()` 1 ns resolution; AC power at capture; `C.UTF-8` locale. Raw `uptime`, `pmset`, `sw_vers`, `uname` and `sysctl` outputs retained. |
| Thread environment | All four relevant variables were unset at capture. Planned commands set `OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. |
| Effective OpenMP | A separate local OpenMP probe under those four overrides reported `omp_max_threads=1` and `omp_parallel_team_size=1`. This is a probe of the configured runtime, not a solver observation. |
| Effective BLAS | **Unverified.** Accelerate exposed no verified effective-thread query in this setup; the environment request is not proof of the actual BLAS thread count. |

The first OpenMP probe compile, using the Homebrew header via `-I` with `-Werror -Wpedantic`, failed on external `omp.h` enum constants. Its argv, stderr and nonzero exit are retained as `omp-probe-compile-attempt1.*`. Recompiling with that external header via `-isystem` passed under the same warnings policy; `omp-probe-compile-attempt2.*` and `omp-probe-run.*` record the successful compile and one-thread execution. The probe source and executable are in the ignored bundle, not repository C files.

## Preflight and provenance checks

Day 5 invoked the pilot runner's `checked_probe` with `python3 scripts/validate_external_baseline_evidence.py --schema tests/corpus/schemas/external_baseline_evidence_v1.json --self-test` from the repository root. Exit `0`, empty stderr, and `external-baseline-evidence-v1: self-test PASS` are in `preflight.json` (SHA-256 `e4cee3bf16d4600e055bb3ec31c60aa2e084401135d66791421107a449f5281c`) and raw streams. This is validator preflight, not a solver pass. CHOLMOD availability and int32 allocation had already been probed without a solve on Day 4; its loaded-library identity was rechecked here by file hash. A new preflight and runtime identity check are required immediately before each later solver process.

| Recheck | Day 5 result |
| --- | --- |
| Selected fixture | `tests/data/symmetric_4.mtx` and both staged copies: SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`, 114 bytes. RHS `[13,14,17,16]`, oracle `[1,1,1,1]`. |
| Method and schema | The three frozen Sprint 215 method documents, v1 schema, pilot source, CSC helper, runner and validator match their Day 1 hashes: nine frozen files in all. |
| Project build | HEAD `dab7943c218e997cda6898c9cdd8d3572a6029c9`; 93 tracked build inputs match Day 3. The retained `cholmod_pilot` SHA-256 is `ed02d301d781e61c9fdb8077abcf8d427517c387ccdb61c40306287e8492474f`; static archive SHA-256 is `f522144e469f6bee5ceac6318666e2869a63ec0d0b33431fcab1e51adde666d3`. |
| Baseline runtime | Loaded CHOLMOD file `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib` SHA-256 `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`; all ten non-system loaded dependency files match the Day 4 inventory. Shared-cache Apple BLAS/LAPACK still have UUIDs and OS build, but no per-image file SHA-256. |
| Source state at preparation | `preparation-git-status.z` records the dirty worktree with untracked planning documents; tracked patch is empty. `preparation-untracked.patch` SHA-256 `ae8b3ad5753a51a9b0eb837713024d2a461f6a2b08347b670e2d38559a665e18` is reconstructible against HEAD. This snapshot predates this Day 5 document and must be replaced with a fresh execution-time status and patch. |

`retained-artifacts.json` lists the copied executable, archive, generated header, Day 3 build logs and flags, Day 4 runtime manifest/traces, ten non-system dylibs and frozen source files with sizes and SHA-256. The manifest hashes all 35 retained files. The local copies support later audit; no redistribution or rights approval is implied.

## Two-process run recipe

`run-recipe.json` (SHA-256 `cec476f6837f889ee11e357b9be8865d4ad09c8a00acbfc990c6daeb1f414e96`) specifies full absolute argv, cwd, fixture and raw-output paths for each side. Its status is `prepared_not_run`. Both sides execute the same retained helper binary with their respective `project` or `baseline` mode and separate, byte-identical matrix copies:

| Side | Dedicated cwd and input | Dedicated outputs |
| --- | --- | --- |
| Project | `prepared/project/`, `prepared/project/input/symmetric_4.mtx` | `prepared/project/raw/{stdout.bin,stderr.bin,exit-code.txt,command-manifest.json}` |
| Baseline | `prepared/baseline/`, `prepared/baseline/input/symmetric_4.mtx` | `prepared/baseline/raw/{stdout.bin,stderr.bin,exit-code.txt,command-manifest.json}` |

Each side requires a **new process** and fresh factor/solver state. The recipe unsets `DYLD_LIBRARY_PATH`, `DYLD_INSERT_LIBRARIES` and `LD_LIBRARY_PATH`; sets the four thread variables above; allows at most 120 seconds per side and 900 seconds overall; and requires raw stdout/stderr bytes, exit code, UTC timestamps and command metadata under append-only attempt paths. The prepared `raw/` directories are empty. Before executing either command, re-run the validator self-test from the repository root, compare the current Git status and full dirty patch with a fresh capture, re-hash both inputs and binaries, reconcile actually loaded libraries, and record effective threads and factor controls. These direct helper commands use isolated fixture copies; they do not themselves invoke the Python runner's higher-level fixed-fixture preflight. The caller must explicitly perform that gate and retain its result.

## Cell-key and claim status

`partial-cell-key.json` (SHA-256 `7caaab7cc40b97328cffff16ded77883d35aec9678dfc923a3e79c653a09b0af`) contains the frozen problem, project/backend/int32 identity, requested CHOLMOD controls, host fingerprint and observed OpenMP probe result. It deliberately sets `cell_id=null`, `evidence_status=blocked`, `block_gate=execution`, and `publication_review=pending`. It is a **preparation record**, not a schema-valid observed v1 record.

The blockers are exact baseline bottle build options and complete module configuration; a loaded Apple BLAS/LAPACK digest or approved shared-cache equivalent; observed BLAS thread count; effective factor controls; a fresh source patch and status at execution; and two solver outputs with independent correctness checks. Day 4 also leaves rights and attribution review open. A Day 6 project run may be captured as a local attempt if its preflight passes, but no two-sided observed cell, timing comparison, ranking or publication claim follows from Day 5. If the strict provenance requirements remain unsatisfied, keep the cell blocked and retain the raw attempt without promotion. This explicitly records item 222.2's unresolved provenance scope and prepares item 222.3's independent execution layout.
