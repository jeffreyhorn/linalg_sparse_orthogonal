# Sprint 222 Day 10: Frozen-Input Repeat and Identity Check

## Outcome

A new append-only project/CHOLMOD pair passed the independent numerical gates and reproduced the first pair's solutions and metrics. It does **not** establish a second observation of the same v1 cell: the dirty source-patch digest changed as sprint documents accumulated, and the required baseline build, BLAS identity and effective thread evidence is still incomplete. The repeat's canonical `cell_id` remains `null`. Its independently validated v1 record has `evidence_status=blocked`; no `correctness_observed`, timing or publication claim follows from the local solves.

The local, ignored repeat is `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-repeat-n4cnx0bs/`. Its 45-file `manifest.json` SHA-256 is `bfc37b2c6419e7c69b1b964d6327d75c07e5e924a387ca44dc3b0ecde33dcb7e`; `repeat-result.json` SHA-256 is `b26ac00a6f0684a1f06e759f891c2bea17eeca6fdf103a7a18750ecb7fe690ce`. A separate validation bundle is `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day10-v1-validation-yqlyl1z8/`. Its 29-file `manifest.json` SHA-256 is `cfdd54715c4f8484406fb5ef30ace6d7b6f989a0b9686015b13828904c59838b`. Both manifests were independently checked against every indexed file's size and SHA-256 after capture. Their retained `repeat-runner.py` and `validate-repeat.py` give the exact local checking procedure.

## Fresh execution and identity controls

The repeat created separate project and baseline directories, each with its own 114-byte fixture copy, cwd, command log, raw stdout/stderr and process/factor state. The v1 validator self-test passed before either solve; an additional preflight passed immediately before each side. In total, 219 recorded checks passed with none failed. Both commands exited `0`, without timeout, under the 120-second side and 900-second overall caps. Wall durations are command metadata only and were not collected as performance evidence.

The frozen fixture and both copies had SHA-256 `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`. The executable (`ed02d301d781e61c9fdb8077abcf8d427517c387ccdb61c40306287e8492474f`), project archive (`f522144e469f6bee5ceac6318666e2869a63ec0d0b33431fcab1e51adde666d3`), 93 tracked build inputs, frozen source/method files, loaded CHOLMOD 5.3.4 dylib (`0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`) and ten file-backed non-system dependencies matched the retained identities. Both fresh dyld traces named the expected loaded paths. The baseline reported the same lower CSC (`nnz=7`, `stype=-1`, sorted/packed, `CHOLMOD_INT`), `factor_minor=4`, `ordering=0`, `is_super=0`, `is_ll=0` and successful phase statuses. The project reported its expected statuses and `used_csc_path=0`.

The HEAD was `dab7943c218e997cda6898c9cdd8d3572a6029c9`; the repeat source status was the same before both sides, with no tracked patch. Its reconstructible untracked patch passed `git apply --check`; SHA-256 `2d0f7a4e19c78b474734d3f1d6fe4b8a318825c68d35f2239afb66fb91d03993`. That differs from the first project's `80113ecd42d4fa172c78a988baa91ac9204e0e04828c002c1d77a41a3bf0f3a2`, because later planning documents were present. The four requested thread variables (`OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) were each `1`; a separate OpenMP probe observed team size `1`. The privacy-safe host fingerprint before each side and after the run matched Day 5 and Day 7: `3f4db376401dc48cf182656aa1f31f2bf41234de468a7d65a2030777b40e7e70`. Actual Accelerate BLAS thread count inside the solver remains unobserved. Day 6 had no execution-time host fingerprint, so full host identity across first and repeat pairs cannot be proven.

## First-versus-repeat result

`first-versus-repeat.json` in the validation bundle (SHA-256 `08e0f7be98b7a3337631032b48f777bfe611a7b3da56e63ebe475341887e791b`) preserves the field-by-field comparison. Recomputed metrics came from the frozen matrix and the repeat's raw solution bytes. The fixed residual and oracle limits were `1.4495071809506044e-12` and `1.380483029476766e-13` for each side; the paired oracle limit was `2.760966058953532e-13`.

| Check | First pair | Repeat pair | Disposition |
| --- | --- | --- | --- |
| Project solution | `[0.99999999999999989,1,1.0000000000000002,1]` | Identical | Numerical match. |
| Project residual / backward / oracle error | `3.552713678800501e-15` / `1.0449157878825003e-16` / `2.220446049250313e-16` | Identical | All finite and within fixed gates. |
| Baseline solution | `[1,1,1,1]` | Identical | Numerical match. |
| Baseline residual / backward / oracle error | `0` / `0` / `0` | Identical | All finite and within fixed gates. |
| Paired infinity difference | `2.220446049250313e-16` | Identical | Within combined oracle gate. |
| Project raw stdout SHA-256 | `c1aa1fc97c5e7f60dc12d8b0f11fbfd66ab889d7476dd85d7bed17ac8c02d703` | Same | Exact result stream match. |
| Baseline raw stdout SHA-256 | `ed545ce746977365bd59e064fb9336642a4d30af6455498227570552bf093346` | Same | Exact result stream match. |
| Project raw stderr SHA-256 | `cb67d98e5fa277c10423a1c867468a0696a4a5f7e1248df5b0e21f5a6f232988` | `aa7f22549743f2594406b6b170ddd2f73b168f45bbc303a491da42f7f30183b7` | Changed process/loader trace bytes retained. |
| Baseline raw stderr SHA-256 | `3a1971045bad41938018114c3df9d8e07ad828c8eca3783a4dd727a76b120567` | `78c69083df65c47d99ae5ef910ae62b6fc9326a025690254bf9676ebf51ffd48` | Changed process/loader trace bytes retained. |

## Independent v1 validation and open gate

`validator-transcripts.json` retains full argv, exit code and raw stream hashes for three separate CLI invocations of `scripts/validate_external_baseline_evidence.py` with the checked-in v1 schema and the appropriate `--bundle-root`:

| Input | CLI verdict |
| --- | --- |
| First Day 9 `blocked-record.json` | Exit `0`: `valid record_id=s222-d9-provenance-blocked evidence_status=blocked`. |
| Repeat `repeat-candidate.rejected.json` | Exit `1`: `missing_provenance /environment record_id=s222-d10-repeat-candidate: observed run needs captured environment`. This candidate contains actual repeat observations, but is preserved only as a rejected candidate. |
| Repeat `repeat-blocked.json` | Exit `0`: `valid record_id=s222-d10-repeat-blocked evidence_status=blocked`. |

The accepted repeat record has `cell_id=null` and v1 sides marked `blocked`. Those labels describe claim eligibility, while the separate raw solver evidence records successful local numerical work. No attempt was overwritten or reclassified to hide a failure. The remaining promotion blockers are the exact CHOLMOD bottle build/module configuration, approved identity for Apple shared-cache BLAS/LAPACK, observed effective BLAS threading, a complete one-cell source/environment key and publication/attribution review. The changed dirty-patch digest is specifically a **different potential cell identity** under v1, despite identical executable and numerical output. No canonical ID was computed by filling unknown values with assumptions.
