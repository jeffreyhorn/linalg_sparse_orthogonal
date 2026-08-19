# Sprint 168 Day 4: Runtime Suitability And Local Dry Run

## Purpose

Day 4 runs the selected Day 3 performance lane locally, measures runtime and
generated output size, inspects CSV/index/manifest stability, and decides
whether the lane remains suitable for hosted CI promotion.

## Commands Run

| Command | Result | Runtime |
| --- | --- | --- |
| `BENCH_CANONICAL_REPORT_LABEL=sprint-168-day4-dry-run make bench-canonical-report` | Passed | `real 3.21`, `user 1.17`, `sys 0.72` |
| `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` | Passed | `real 0.01`, `user 0.00`, `sys 0.00` |
| `git status --short --ignored build/bench-reports/canonical build/bench_refactor_csc` | Passed | Confirmed generated `build/` output is ignored. |

The full canonical report ran quickly on the local machine. The focused
selected command was effectively instantaneous locally. Hosted runtime still
needs PR CI confirmation after workflow wiring, but the local dry run does not
show a runtime blocker.

## Generated Output Size

| Artifact | Size | Line count | Notes |
| --- | ---: | ---: | --- |
| `build/bench-reports/canonical/bench_refactor_csc.csv` | 351 bytes | 2 | One header row and one selected measurement row. |
| `build/bench-reports/canonical/index.tsv` | 2506 bytes | 5 | Header plus four canonical rows. |
| `build/bench-reports/canonical/manifest.txt` | 1882 bytes | 44 | Bundle-level metadata and non-claim notes. |
| `build/bench-reports/canonical/` | 24K | n/a | Entire generated canonical bundle. |

The selected report artifacts are small enough for CI logs and artifact upload.

## Selected CSV Inspection

The selected CSV emitted:

```text
benchmark,category,matrix,scenario,n,nnz,ldlt_dense_backend_request,ldlt_dense_backend_selected,ldlt_dense_backend_fallback,analyze_ms,refactor_public_ms,refactor_csc_ms,solve_public_ms,solve_csc_ms,speedup_refactor,res_public,res_csc
bench_refactor_csc,proof,nos4.mtx,chol_spd,100,594,n/a,n/a,n/a,1.057,0.140,0.069,0.005,0.003,2.03,8.24e-16,7.06e-16
```

The focused direct command emitted the same stable row identity with different
timing values:

```text
bench_refactor_csc,proof,nos4.mtx,chol_spd,100,594,n/a,n/a,n/a,0.211,0.117,0.065,0.005,0.003,1.80,8.24e-16,7.06e-16
```

Stable row identity fields:

- `benchmark=bench_refactor_csc`
- `category=proof`
- `matrix=nos4.mtx`
- `scenario=chol_spd`
- `n=100`
- `nnz=594`
- backend fields are `n/a` for this selected SPD/Cholesky mode
- residual fields are present and small

Expected variable fields:

- timing columns;
- derived `speedup_refactor`;
- generated timestamp in metadata;
- branch/commit/platform/compiler context.

## Index And Manifest Inspection

The selected `index.tsv` row records:

| Field | Observed Day 4 value | Stability assessment |
| --- | --- | --- |
| `surface` | `canonical` | Stable |
| `category` | `measurement` | Stable |
| `report_label` | `sprint-168-day4-dry-run` | User/CI supplied |
| `git_commit` | `33be1dc8` | Commit-specific |
| `git_branch` | `sprint-168` | Branch-specific |
| `platform` | Local Darwin host string | Platform-specific; hosted lane must record runner platform separately. |
| `compiler` | Local Apple Clang string | Toolchain-specific; hosted lane must record CI compiler. |
| `build_mode` | `serial` | Stable if CI sets or detects consistently. |
| `omp_num_threads` | `unset` | Stable if CI leaves unset or sets explicitly. |
| `artifact` | `bench_refactor_csc` | Stable |
| `relative_path` | `bench_refactor_csc.csv` | Stable |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` | Stable |
| `support_tier` | `local_only` | Must be updated or supplemented for hosted selected evidence. |
| `claim_boundary` | `local_threshold_free` | Must be updated or supplemented for hosted selected evidence. |
| `fixture_or_workload` | `nos4.mtx` | Stable |
| `repeat_semantics` | `configured_repeat_1` | Stable |
| `warmup` | `not_recorded` | Missing for hosted methodology; Day 5 must decide policy. |
| `variance` | `not_recorded` | Missing for hosted methodology; Day 5 must decide policy. |
| `baseline` / `threshold` | `n/a` / `n/a` | Correct for threshold-free publication. |
| `methodology_notes` | `threshold_free_local_measurement;not_portable_performance_claim` | Must be refined for hosted selected evidence without superiority wording. |

The manifest repeats the same context and includes explicit non-claim notes.
It is a useful human-readable hosted artifact candidate.

## Runtime Risks

| Risk | Day 4 assessment | Handling |
| --- | --- | --- |
| Full canonical report runtime | Low locally: `3.21s`. | CI can likely run selected lane; Day 9 should still keep a bounded runtime budget. |
| Focused selected command runtime | Low locally: `0.01s`. | Candidate is suitable for hosted CI timing/report publication. |
| Hosted runner variance | Unknown until CI. | Keep timing threshold-free and record runner/toolchain metadata. |
| Full canonical bundle scope | Medium. It emits four rows, while Sprint 168 selected one. | Day 5/Day 7 should decide whether freshness requires only `bench_refactor_csc` or keeps the full bundle as context. |
| Timing values in freshness | High if compared directly. | Freshness must compare row identity/metadata presence, not raw timing equality. |
| `support_tier=local_only` | Known mismatch for hosted publication. | Day 5 should design hosted selected support-tier semantics. |
| `warmup=not_recorded` and `variance=not_recorded` | Methodology gap. | Day 5 should either record explicit policy or keep these as deliberate non-claims. |

## Missing Methodology Data For Day 5

Day 5 should decide how hosted selected performance metadata records:

- CI runner label or operating-system image;
- compiler executable and version;
- build flags;
- build mode override, likely via `SPARSE_CANONICAL_BUILD_MODE`;
- `OMP_NUM_THREADS`, either explicit or deliberately `unset`;
- selected lane support tier distinct from `local_only`;
- hosted selected claim boundary distinct from `local_threshold_free`;
- warmup policy;
- repeat count and variance policy;
- raw timing non-freshness policy;
- artifact upload path and selected-row identity.

## CI Suitability Decision

The selected `bench_refactor_csc` lane remains suitable for Sprint 168 hosted
publication.

Decision:

- keep `bench_refactor_csc` as the primary lane;
- keep `tests/data/suitesparse/nos4.mtx --repeat 1` as the selected fixture
  and repeat scope;
- use generated `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt` as
  the selected artifact set;
- design freshness around row identity, metadata presence, selected artifact
  existence, and claim-boundary fields;
- do not compare timing values for freshness;
- do not promote all canonical rows unless Day 7 explicitly chooses full-bundle
  freshness as context.

## Day 5 Handoff

Day 5 should design methodology metadata for hosted selected performance
publication. The main design question is whether to extend the existing
canonical report script with hosted selected-lane fields or add a thin
selected-performance wrapper that filters and validates the `bench_refactor_csc`
row.

## Validation Notes

Day 4 changed only Sprint 168 planning artifacts and generated ignored
`build/` report output. No `.c` or `.h` files were modified, so the full C
quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected lane has a bounded runtime plan. | Complete | Full canonical report completed in `3.21s`; focused selected command completed in `0.01s` locally. |
| Unstable output fields are identified before CI wiring. | Complete | Timing, speedup, timestamp, branch/commit, platform, and compiler are variable; raw timing freshness is rejected. |
| Lane remains suitable or is narrowed explicitly. | Complete | `bench_refactor_csc` remains selected with one fixture, one repeat scope, and selected artifact set. |
