# Sprint 222 Day 9: V1 Record and Validation Gate

## Outcome

**A byte-backed v1 `blocked` run record validated; an observed-pass record was rejected.** Days 6–8 produced two independent local numerical passes and a paired consistency pass, but the selected v1 contract requires complete build, dependency, host and thread provenance before either side may appear as an observed `pass` in a valid record. Those facts are still missing. Day 9 therefore did **not** assign a canonical cell ID or label the pair `correctness_observed`. The solver outputs remain available as raw local artifacts, and the blocked record provides a schema-checked handoff without promoting them.

The final ignored bundle is `build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/day9-v1-record-_tcqsm4j/`. `day9-manifest.json` SHA-256 is `234bd44e3421c876ba7345de3876f5e2dfa6cea2418fb4616d39bd5df961677b`; its 32 indexed files passed byte-size and SHA-256 checks. The source index `source-artifact-index.json` SHA-256 is `8e8bea7feefecc89ce4c52eaf962fcdaaf9be08ad41f55ac323914af949b738b` and enumerates **246 retained files** under the Day 3–8 local bundles, including both original solver attempts and the earlier read-only Day 8 reconciliation drafts. Day 9 rechecked the source manifests before copying evidence. The exact record generator is retained as `build-record.py` in the Day 9 bundle.

## Input and artifact ledger

The Day 9 bundle copied the two raw stdout/stderr streams, exact command manifests, independent metric files, project and baseline bundle manifests, the Day 8 reconciliation, Day 3 build record, Day 4 runtime record, Day 5 host/recipe records, the retained helper binary and CHOLMOD dylib, and the Day 6 source patch/status. Each copied file has its own listed SHA-256 in both v1 records and in the Day 9 manifest. The original Day 6 and Day 7 bundles remain untouched, with manifest SHA-256 values `6e4c6f28601c9acb696192add7f700723b22ad469f3001b2f7ed752940da7781` and `a140595d6b5c7150f686749af0133d6e9fd3950cd3ef04305926c6f494fe6dee`. The final Day 8 reconciliation manifest SHA-256 is `7e20fb0e8fc4c91cd29834569624ce141456455d96d147246b8cf3165598df88`. No solver output was rewritten after observation.

| Item | SHA-256 / outcome |
| --- | --- |
| Frozen fixture | `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335`; identical source and per-side attempt copies. |
| Three Sprint 215 method files | Their current byte hashes are populated in both v1 records; the validator independently read and checked the repository files. |
| Project helper / archive | `ed02d301d781e61c9fdb8077abcf8d427517c387ccdb61c40306287e8492474f` / `f522144e469f6bee5ceac6318666e2869a63ec0d0b33431fcab1e51adde666d3`. The helper itself is copied into the Day 9 bundle. |
| Loaded CHOLMOD dylib | `/usr/local/Cellar/suite-sparse/7.12.2/lib/libcholmod.5.3.4.dylib`, SHA-256 `0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c`; copied into the Day 9 bundle. |
| Project raw stdout / stderr | `c1aa1fc97c5e7f60dc12d8b0f11fbfd66ab889d7476dd85d7bed17ac8c02d703` / `cb67d98e5fa277c10423a1c867468a0696a4a5f7e1248df5b0e21f5a6f232988`. |
| Baseline raw stdout / stderr | `ed545ce746977365bd59e064fb9336642a4d30af6455498227570552bf093346` / `3a1971045bad41938018114c3df9d8e07ad828c8eca3783a4dd727a76b120567`. |
| Paired numerical reconciliation | `d814185fbcc753101ebc76076fb2bec9c2d11d0861c0b1ca39682759c8b852af`; 148 final checks passed, including residual, oracle and cross-solver gates. |
| Day 6 source patch / status | `80113ecd42d4fa172c78a988baa91ac9204e0e04828c002c1d77a41a3bf0f3a2` / `f2469d0a5a1814964b53485765e9302fcac8c7c71debc68979fb82d5da7fe15b`. The Day 7 patch differs because new untracked documentation was added between runs; no tracked build input or executable byte changed. |

`project-observation.json` and `baseline-observation.json` in the bundle are structured derivatives of their respective raw stdout fields and the Day 8 independent metrics. The original byte streams are retained alongside them. The candidate observed record references these derivatives using the schema's `project_raw` and `baseline_raw` roles, so its failed validation cannot be attributed to absent solution/status bytes. The valid blocked record lists the same copied artifacts, but does not claim a schema-valid numerical result.

## Actual CLI validation

Both records were sent to `scripts/validate_external_baseline_evidence.py` with the checked-in `external_baseline_evidence_v1.json` via `--schema`, their own `--input`, and the Day 9 directory as `--bundle-root`. `validator-verdicts.json` retains full argv, exit codes and raw stdout/stderr hashes.

| Input | CLI result | Meaning |
| --- | --- | --- |
| `observed-candidate.rejected.json` (SHA-256 `b53cbababe32787432c823c1c939a3f3bc3be7b787fc667b3b49e2b6d1327479`) | Exit `1`; `missing_provenance /environment record_id=s222-d9-observed-candidate: observed run needs captured environment`. Raw stderr SHA-256 `11abb480faa70627eccdef9c3d1ec6d791f79f010c74c54f54542991db0f16a3`. | Deliberately submitted both local `pass` sides with the **actual missing** complete environment. The validator rejected the promotion. This is the first reported gate, not an exhaustive list of gaps. |
| `blocked-record.json` (SHA-256 `515e4264d0d542a2a5d33337bbd5599a799ef64fe4d15410fb590c94152f0236`) | Exit `0`; `external-baseline-evidence-v1: valid record_id=s222-d9-provenance-blocked evidence_status=blocked`. Raw stdout SHA-256 `2fb0214045c8e8761e32cc7647402eef4f4602ee0bcb200db8cf9acf3f177b79`. | Valid **blocked execution-evidence record**, with `cell_id=null`, no v1 side pass labels, no timing measurements and no external-result claim. The raw attempts remain separately linked as byte-backed artifacts. |

The schema rejects an execution-blocked record that carries solver `pass` fields. It also requires complete provenance and a canonical cell ID whenever both sides are `pass`, even when publication review is still pending. Therefore the validated blocked record uses `correctness.project.state=blocked` and `correctness.baseline.state=blocked`; that is a **v1 claim status**, not a claim that the Day 6 and Day 7 processes never executed. Their local numerical results remain in the raw bundles and Day 8 ledger. The candidate was preserved exactly as rejected; no threshold, schema rule or output was changed to make it pass.

## Explicit unresolved fields and Day 10 handoff

1. **Execution-time host and threads:** Day 5 and Day 7 host fingerprints match, but Day 6 lacks an execution-time host fingerprint. The four one-thread environment values were requested on both sides, while effective Accelerate BLAS threads were never observed. V1 requires a complete `environment` and positive observed thread counts; no count was invented.
2. **Baseline binary build and dependencies:** The loaded CHOLMOD path and ten non-system dependency hashes are known, but the Homebrew bottle's exact build options and complete module composition remain unverified. Apple BLAS/LAPACK are loaded from the dyld shared cache with UUIDs and OS build, but no per-image file SHA-256 or approved identity substitute. `cell.baseline.build_options` and `module_composition` remain `null` with a named reason; the dependency list contains only the known file-backed entries.
3. **Canonical cell identity:** The Day 6 and Day 7 dirty worktree patches differ only in added documentation, yet v1's cell key includes a single project patch digest. The generated record cannot honestly assign one fully verified cell ID across both execution snapshots while the other identity gaps remain. It leaves `cell_id=null` with a reason. A fresh paired rerun with a frozen source snapshot or an approved identity rule is needed before promotion.
4. **Publication:** Fixture attribution and SuiteSparse/module/transitive license review remain pending. These are separate from a passing numerical gate and do not justify moving files into a selected report.

Item 222.3's planned 34-hour execution allocation is accounted for across Days 5–9: isolated setup, two separate solves, numerical reconciliation and this record attempt. That describes planned scope, **not elapsed work time or an accepted observed cell**. Day 9's complete-observed-record deliverable is blocked by the named evidence gaps. Day 10 can perform an append-only repeat from the frozen input, but must capture execution-time host/thread facts and retain a stable source identity; a repeat alone cannot reconstruct unavailable bottle build options or shared-cache file hashes.

Two earlier Day 9 preparation directories remain ignored and local. Their script failed before invoking the validator because a local helper named `copy` shadowed Python's `copy` module and one call still used the old name. `preparation-failures.json` SHA-256 `2a636a2fd537c4237dbc93bd97da5899a0cc4deebeb7fc15f21dace1f4a31c32` indexes all 31 partial files and their error reasons. Neither was a solver run or a validator verdict. The final generator corrected these local script errors and preserved all prior directories.

`git check-ignore` confirms the Day 9 bundle is excluded by `.gitignore`'s `build/` rule. No generated evidence was added to a selected report manifest, canonical benchmark row or `docs/api/html/`; the only repository changes for Day 9 are this artifact and the sprint working-note entry.
