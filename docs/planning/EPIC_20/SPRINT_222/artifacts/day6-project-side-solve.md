# Sprint 222 Day 6: Project-Side Solve

## Disposition

**Local project correctness passed; the two-sided v1 cell remains blocked.** One fresh project process solved the frozen 4 × 4 SPD fixture with no-reorder linked-list Cholesky. The process exited `0`, its load, factor and solve status fields were all `0`, `used_csc_path=0`, and independent recomputation of the residual and oracle error passed the predeclared Sprint 215 limits. This is **one project-side attempt**, not an external comparison, timing result, schema-valid observed cell or publication decision.

Raw evidence is retained only under ignored `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day6-project-rmbv856q/`. Its `manifest.json` SHA-256 is `6e4c6f28601c9acb696192add7f700723b22ad469f3001b2f7ed752940da7781`; all 25 indexed files passed a post-run digest and size check. The manifest, `attempt.json`, `pre-solve-checks.json`, source snapshot, independent metrics, raw byte streams, exact command metadata and `capture-and-check.py` are available there. The captured checker source SHA-256 is `1d3a11e5e5c691dcdf5ad2ccaef24a18fd9019fa30489ed92b562094eb812339`. No failed attempt was overwritten or substituted.

## Preflight, source and identity

The attempt began with the v1 validator self-test, using the exact Day 5 `run-recipe.json` preflight command from the repository root. It exited `0` with `external-baseline-evidence-v1: self-test PASS` before the solver was launched. Raw preflight stdout/stderr and command metadata are in `raw/validator-preflight.*`. The selected fixture and fresh copied input both had SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`; the separate Day 5 baseline copy still matched. The project helper and archive matched their Day 3/Day 5 hashes: `ed02d301d781e61c9fdb8077abcf8d427517c387ccdb61c40306287e8492474f` and `f522144e469f6bee5ceac6318666e2869a63ec0d0b33431fcab1e51adde666d3` respectively. All 93 tracked build inputs and the retained frozen source/method/schema files matched their recorded hashes. `pkg-config --modversion cholmod` returned `5.3.4`.

Immediately before the solve, the source snapshot recorded HEAD `dab7943c218e997cda6898c9cdd8d3572a6029c9` and seven untracked Sprint 222 planning files, with no tracked diff. `source-snapshot/git-status.z` SHA-256 is `f2469d0a5a1814964b53485765e9302fcac8c7c71debc68979fb82d5da7fe15b`; the full untracked byte patch SHA-256 is `80113ecd42d4fa172c78a988baa91ac9204e0e04828c002c1d77a41a3bf0f3a2`. Its `git apply --check` succeeded in an empty temporary directory. The snapshot has exact copies and individual hashes in `source-snapshot/untracked/` and `untracked.json`. This Day 6 artifact and updated working notes were written **after** the snapshot; a later attempt must capture its own source state.

The actual loaded CHOLMOD path reported by the process was `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib`, matching the Day 4 path and SHA-256 `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`. The ten named non-system dependency files matched their Day 4 hashes immediately before execution, and all ten paths appeared in this process's `DYLD_PRINT_LIBRARIES=1` raw stderr trace. Apple BLAS/LAPACK loaded paths also appeared in that trace, but their shared-cache images still have no per-image SHA-256. The trace contained no warning or error line. This path inventory and pre-run file check support the local identity finding; they do not resolve the Day 4 baseline build/module provenance blocker.

## Isolated command and raw output

The exact absolute argv, cwd, environment overrides, UTC start/end, exit code, timeout disposition and byte hashes are in `raw/project-solve.command.json` (SHA-256 `3056f8881a313f5de06bd88af1edaf02d47f963ea7edefa433fda77b26ebc308`). The command used the retained `cholmod_pilot` executable in `project` mode, with the new `day6-project-rmbv856q/input/symmetric_4.mtx` copy and that attempt directory as cwd. It ran as a new process with its own factor state, under the Day 5 four one-thread environment requests. The dynamic loader trace was enabled for path verification. The 120-second per-side and 900-second overall caps were respected; the captured wall duration is execution metadata only, **not solver timing evidence**. Effective BLAS thread count was not observed.

`raw/project-solve.stdout.bin` SHA-256: `c1aa1fc97c5e7f60dc12d8b0f11fbfd66ab889d7476dd85d7bed17ac8c02d703` (218 bytes). `raw/project-solve.stderr.bin` SHA-256: `cb67d98e5fa277c10423a1c867468a0696a4a5f7e1248df5b0e21f5a6f232988` (84,700 bytes, primarily dyld tracing). Both raw byte files were written before parsing and were never normalized or sanitized. The exact stdout fields were:

```text
loaded_cholmod_path=/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib
side=project
load_status=0
matrix_nnz=10
factor_status=0
used_csc_path=0
solve_status=0
x=0.99999999999999989,1,1.0000000000000002,1
```

The helper source at the frozen HEAD requests `SPARSE_REORDER_NONE` and `SPARSE_CHOL_BACKEND_LINKED_LIST`. `used_csc_path=0` is the helper's effective path diagnostic. The helper does not expose project factor nonzeros or pivot diagnostics, so those values were not inferred. The output vector is finite; no allocation failure, non-OK status or warning was observed.

## Independent numerical gate

The attempt's separate checker reparsed the untouched Matrix Market bytes, verified the symmetric 4 × 4, seven-stored-entry input, expanded it to the full matrix, and checked the fixed RHS `[13,14,17,16]` against oracle `[1,1,1,1]`. It then used the raw project solution, Python float64 arithmetic with `math.fsum` for row accumulation, and the predeclared `T=32·4·2^-52`, `K=17/7` formulas. `independent-metrics.json` SHA-256 is `5737de18408a23607f225b9a47afd5a0c286af53df9ded477aa59199cac9c32a`.

| Quantity | Independent result | Limit / verdict |
| --- | ---: | --- |
| `x` | `[0.9999999999999999, 1, 1.0000000000000002, 1]` | Four finite components. |
| Residual vector `A·x-b` | `[-1.7763568394002505e-15, 0, 3.552713678800501e-15, 0]` | Finite. |
| Residual infinity norm | `3.552713678800501e-15` | `≤ 1.4495071809506044e-12`: **pass**. |
| Backward error `R/(||A||∞·||x||∞+||b||∞)` | `1.0449157878825003e-16` | Denominator `34`; recorded, not an independent fitted cutoff. |
| Oracle infinity error | `2.220446049250313e-16` | `≤ 1.380483029476766e-13`: **pass**. |

All 52 recorded preflight, identity, status, runtime-path and numerical checks passed. The project's local correctness disposition is supported by raw status **and** the independent numerical gate. The still-missing baseline solve prevents the cross-solver consistency check.

## Remaining gates

Exact Homebrew bottle build options and complete CHOLMOD module configuration remain unknown; Apple shared-cache BLAS/LAPACK lacks a loaded-image digest or accepted equivalent; effective BLAS threads are unobserved; effective CHOLMOD factor controls require the Day 7 baseline process; and the source snapshot must be refreshed for every later attempt. The Day 2 rights and attribution review is separate and pending. Retain this project attempt as a local, blocked-provenance record. Do not assign a v1 observed cell ID, publish a comparison or treat process wall time as a performance sample.
