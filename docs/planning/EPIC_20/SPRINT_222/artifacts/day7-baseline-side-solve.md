# Sprint 222 Day 7: Baseline-Side CHOLMOD Solve

## Disposition and evidence boundary

**Local CHOLMOD correctness passed; the v1 observed cell remains blocked by provenance.** A new baseline process solved the frozen 4 × 4 SPD matrix from its own copied input and fresh factor state. The raw process exit, load, analysis, factorization and solve statuses passed, and a separate calculation from the untouched matrix and RHS passed both fixed numerical gates. This is a single baseline-side correctness attempt. Day 8 owns two-sided numerical reconciliation; no performance, timing, publication or state-of-the-art claim follows from this run.

The ignored local attempt is `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day7-baseline-rin7rs32/`. `manifest.json` SHA-256: `a140595d6b5c7150f686749af0133d6e9fd3950cd3ef04305926c6f494fe6dee`. Its 29 indexed files include the raw streams, exact command records, source snapshot, checker sources, CSC reconstruction, independent metrics, post-run host check and disposition. No Day 6 project factor, solution or output file was used as baseline input.

## Preflight and execution identity

The attempt ran the Day 5 recipe's v1 validator self-test first, from the repository root. Exit `0` and `external-baseline-evidence-v1: self-test PASS` are retained as raw `validator-preflight` streams and command metadata. Before the solver, the selected source fixture, separate Day 5 project and baseline copies, and fresh Day 7 baseline copy all matched SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`. The retained helper SHA-256 was `ed02d301d781e61c9fdb8077abcf8d427517c387ccdb61c40306287e8492474f` and linked project archive SHA-256 was `f522144e469f6bee5ceac6318666e2869a63ec0d0b33431fcab1e51adde666d3`; 93 tracked build inputs and the retained frozen method/source/schema files matched their prior hashes. `pkg-config --modversion cholmod` returned `5.3.4`, corroborating but not replacing runtime identity evidence.

The fresh execution-time source snapshot recorded HEAD `dab7943c218e997cda6898c9cdd8d3572a6029c9`, an empty tracked patch and eight untracked Sprint 222 Markdown files. `source-snapshot/git-status.z` SHA-256: `35ce9e6dff90e930e305aa9c3158641f48a20c40294560620008fe9157a77a7a`. `source-snapshot/untracked.patch` SHA-256: `c590bacdf33d8c0fa18185af07ab785c1ba338cea406d6b4d67165cbb75bc36c`; it passed `git apply --check` in an empty temporary directory, and the exact untracked bytes are copied beside it. This Day 7 artifact and new working notes were written after the snapshot, so future attempts need their own status and patch.

A post-run host identity recheck of OS build, architecture and CPU reproduced the Day 5 privacy-safe fingerprint `3f4db376401dc48cf182656aa1f31f2bf41234de468a7d65a2030777b40e7e70`; the raw hostname was omitted. `host-identity-recheck.json` SHA-256 is `73e2b7dfac16e8428c0772dc91bed33779eb70bc15a03ded40a7dde6def2d227`. This is a same-host check after execution, not a measurement of effective BLAS threads.

The helper reported the actually loaded path `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib`, the same file as Day 4, with SHA-256 `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`. Immediately before execution all ten named non-system dependency files matched the Day 4 hashes, and each path appeared in this process's `DYLD_PRINT_LIBRARIES=1` raw stderr trace. Apple shared-cache BLAS/LAPACK paths also appeared, but no per-image file SHA-256 is available. The trace contained no warning or error line.

`raw/baseline-solve.command.json` SHA-256: `86d99aacc67a7ec9ae2f516ed3a9497447fa4f0c48d4d7ba91ac52d85e4c6934`. It records the full absolute helper argv with `baseline` mode, fresh input path, dedicated Day 7 cwd, UTC start/end, process exit `0`, timeout result and raw-stream digests. The process was separate from Day 6 and used the same four one-thread environment **requests** (`OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` all `1`). The 120-second side and 900-second overall caps were respected. Effective Accelerate BLAS thread count remains unverified; wall duration is command metadata, **not timing evidence**.

## Raw phase and factor diagnostics

`raw/baseline-solve.stdout.bin` SHA-256: `ed545ce746977365bd59e064fb9336642a4d30af6455498227570552bf093346` (331 bytes). `raw/baseline-solve.stderr.bin` SHA-256: `3a1971045bad41938018114c3df9d8e07ad828c8eca3783a4dd727a76b120567` (84,700 bytes, primarily dyld tracing). Both were retained as bytes before parsing. The complete stdout fields were:

```text
loaded_cholmod_path=/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib
side=baseline
load_status=0
matrix_nnz=10
csc_nnz=7
csc_stype=-1
csc_sorted=1
csc_packed=1
csc_itype=0
analyze_status=0
factor_status=1
cholmod_status=0
factor_minor=4
ordering=0
is_super=0
is_ll=0
solve_status=1
cholmod_solve_status=0
x=1,1,1,1
```

The helper checked that its lower CSC view reconstructs the loaded full matrix **before** analysis. A separate byte-based reconstruction in `independent-csc.json` (SHA-256 `f257c1bb726eeae973b771bfaf3d4e4cc79e375f4cfff76edc125b90048d0359`) confirmed the frozen lower arrays `p=[0,3,5,6,7]`, `i=[0,1,2,1,2,2,3]`, `x=[10,1,2,12,1,14,16]` and the original expanded matrix. That independent reconstruction verifies the expected input; the actual in-process arrays are covered by the helper's source-level check and successful path, not by a separate runtime dump. Its checker source SHA-256 is `438c18367ffc531fadcc350bd87899305f3c487e8d5e1089d67a9f6c99e62059`.

`analyze_status=0` records `CHOLMOD_OK` after analysis; `factor_status=1`, `cholmod_status=0` and `factor_minor=4` show successful full-rank factorization. The **effective** factor diagnostics are `ordering=0` (natural), `is_super=0` (simplicial), and `is_ll=0` (LDL storage). The latter was observed, not assumed to match the project's storage. `solve_status=1` and `cholmod_solve_status=0` confirm successful solve. No allocation failure, non-OK/warning status or timeout was observed. The helper did not report factor nonzeros or memory use; those values are unavailable rather than zero.

## Independent numerical gate

The retained `capture-and-check.py` SHA-256 is `61d4e800d91fdf12ddc942acea62a02cf2c00db024e259c76bddb8a673214221`. It reparsed the frozen Matrix Market bytes, expanded the symmetric matrix, verified the 4 × 4 shape, seven stored entries and original RHS `[13,14,17,16]`, then recomputed metrics from raw `x=[1,1,1,1]` and untouched `A` using float64 `math.fsum`. `independent-metrics.json` SHA-256: `980ff9988ef6699689bbb6c5d58b7ce054c60cd73f4efa68158e8e75bb1292ba`.

| Quantity | Independent value | Fixed gate |
| --- | ---: | --- |
| Residual vector `A·x-b` | `[0,0,0,0]` | All finite. |
| Residual infinity norm | `0` | `≤ 1.4495071809506044e-12`: **pass**. |
| Backward error | `0` | Denominator `34`; recorded. |
| Oracle infinity error | `0` | `≤ 1.380483029476766e-13`: **pass**. |

All 63 recorded preflight, input, binary/dependency, runtime-path, CSC/phase and numerical checks passed. The exact solution on this small fixture is an observed output, not evidence of broad solver superiority or a performance result. The raw status checks and independent numerical checks together support only this baseline-side local correctness result.

## Remaining gates

The CHOLMOD bottle's exact build options and complete module configuration are still unverified. Apple shared-cache BLAS/LAPACK needs a digest or an explicitly accepted equivalent; effective BLAS threading is still unobserved. Day 8 must reconcile both independently captured solutions under the frozen cross-solver gate and examine asymmetric failures. A later observed v1 record additionally requires a complete cell key and its own validation; Day 2 rights and attribution review still gates publication. Keep this attempt local and provenance-blocked until those requirements are met.
