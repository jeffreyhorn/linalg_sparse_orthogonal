# Sprint 132 Day 8 - Sentinel Implementation Batch

## Purpose

Record the Day 8 implementation of the selected Sprint 132 metadata batch:
structured `performance-sentinels` row metadata, canonical report runtime
context, and documentation alignment.

No benchmark C files, public APIs, backend dispatch paths, OpenMP scheduling,
or hard timing thresholds were changed.

## Implemented Changes

| Area | File | Change |
| --- | --- | --- |
| Sentinel TSV schema | `scripts/performance_sentinels.sh` | Added structured columns for `report_family`, `support_tier`, `claim_boundary`, `artifact`, `backend_request`, `backend_selected`, `backend_fallback`, `dense_kernel`, and `panel_solver`. |
| Sentinel S5 rows | `scripts/performance_sentinels.sh` | Kept S5 as `reviewed_thresholded` with `local_wall_gate` and backend fields set to `n/a`. |
| Sentinel S2 rows | `scripts/performance_sentinels.sh` | Kept S2 as `reviewed_threshold_free` with `local_threshold_free`; parsed Cholesky dense-kernel and panel-solver descriptors from `bench_chol_csc`. |
| Sentinel skip rows | `scripts/performance_sentinels.sh` | Preserved explicit skip behavior while carrying support tier, claim boundary, artifact, and unknown/n/a backend metadata. |
| Canonical report index | `scripts/bench_canonical_report.sh` | Added `platform`, `compiler`, `build_mode`, and `omp_num_threads` to each `index.tsv` row. |
| Canonical manifest | `scripts/bench_canonical_report.sh` | Added platform, compiler, build mode, and `OMP_NUM_THREADS` context. |
| Benchmark docs | `benchmarks/README.md` | Updated `make performance-sentinels` artifact descriptions and narrow S5/S2 metadata interpretation. |
| Maintainer docs | `docs/maintainer_guide.md` | Updated canonical and sentinel metadata policy without changing threshold or portability claims. |

## Generated Metadata Evidence

`make performance-sentinels` generated:

```text
report_family	sentinel_id	status	support_tier	claim_boundary	command	build_mode	omp_num_threads	matrix_or_fixture	metric	value	baseline	threshold	artifact	backend_request	backend_selected	backend_fallback	dense_kernel	panel_solver	notes
sentinel	S5	pass	reviewed_thresholded	local_wall_gate	make wall-check	serial	unset	bcsstk14	qg_amd_reorder_ms	134.2	130	2x	wall_check.txt	n/a	n/a	n/a	n/a	n/a	existing_threshold_gate_passed
sentinel	S2	report	reviewed_threshold_free	local_threshold_free	build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1	serial	unset	nos4.mtx	factor_ll_ms	0.770	n/a	n/a	bench_chol_csc_nos4.csv	unset	builtin	n/a	builtin	batched_panel	threshold_free;chol_env=unset;ldlt_env=unset
```

`make bench-canonical-report` generated an `index.tsv` header with:

```text
surface	category	report_label	generated_at_utc	git_commit	git_branch	platform	compiler	build_mode	omp_num_threads	artifact	relative_path	command
```

and manifest entries for:

```text
platform=Darwin yog-sothoth 24.6.0 Darwin Kernel Version 24.6.0: Wed Nov  5 21:30:23 PST 2025; root:xnu-11417.140.69.705.2~1/RELEASE_X86_64 x86_64
compiler=Apple clang version 11.0.0 (clang-1100.0.33.17)
build_mode=serial
omp_num_threads=unset
```

The numeric timing values are local run evidence only. They are recorded here
to prove the generated schema shape, not to create new performance claims.

## Focused Validation Output

| Command | Result |
| --- | --- |
| `bash -n scripts/performance_sentinels.sh` | Passed. |
| `bash -n scripts/bench_canonical_report.sh` | Passed. |
| `make performance-sentinels` | Passed; generated `sentinels.tsv`, `manifest.txt`, `wall_check.txt`, and `bench_chol_csc_nos4.csv`. |
| `make bench-canonical-report` | Passed; generated canonical CSV artifacts, `index.tsv`, and `manifest.txt`. |
| `sed -n '1,6p' build/bench-reports/sentinels/sentinels.tsv` | Confirmed new sentinel metadata columns and representative S5/S2 rows. |
| `sed -n '1,6p' build/bench-reports/canonical/index.tsv` | Confirmed platform/compiler/build/thread context in canonical index rows. |

## Touched Files

- `scripts/performance_sentinels.sh`
- `scripts/bench_canonical_report.sh`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_11/SPRINT_132/WORKING_NOTES.md`
- `docs/planning/EPIC_11/SPRINT_132/artifacts/day8-implementation-batch.md`

No `.c` or `.h` files were changed.

## Unchanged Claims Statement

This implementation changes generated metadata shape and documentation only.
It does not:

- add a new benchmark
- add a new sentinel lane
- add a new hard timing threshold
- change S5 wall-check pass/fail behavior
- convert S2 Cholesky CSC rows into a timing gate
- claim backend parity or optional backend availability
- claim OpenMP speedup or library-owned thread-count control
- claim portable performance, scalability, or memory behavior

## Day 9 Handoff

Day 9 should use the final generated field names to clean up benchmark
documentation and report-index handoff language. It should focus on making
the updated metadata easy to interpret without broadening benchmark claims.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implemented changes match the Day 7 plan. | Complete | Script/docs batch matches the selected S6/C1/documentation scope. |
| No benchmark, backend, coverage, or public performance claim changes silently. | Complete | No C/header files changed; unchanged-claims statement records the scope boundary. |
| Focused validation passes or the sprint stops with a blocker. | Complete | Script syntax checks and both focused report-generation targets passed. |
