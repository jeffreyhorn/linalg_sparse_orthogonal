# Sprint 192 Day 2: Candidate Benchmark Lane Audit

## Selection Summary

Day 2 selects the existing `bench_refactor_csc` canonical benchmark row on
`tests/data/suitesparse/nos4.mtx --repeat 1` as the Sprint 192
methodology-bound performance evidence lane.

| Field | Decision |
| --- | --- |
| Selected target ID | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| Benchmark artifact | `bench_refactor_csc` |
| Selected workload | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Fixture | `nos4.mtx` |
| Matrix metadata | `matrix_size=n=100` |
| Report family | `benchmark` |
| Subfamily | `canonical` |
| Platform scope | Hosted Linux selected lane plus local freshness shape checks |
| Hosted workflow | `.github/workflows/ci.yml` job `hosted-performance-freshness` |
| Hosted artifact | `sprint168-selected-performance-freshness` |
| Runtime budget | Existing hosted job timeout is `10` minutes; local Day 2 generation completed in about 14 seconds on this machine. |
| Repeat policy | Keep `configured_repeat_1` for the selected workload until Day 3/Day 9 explicitly change methodology or threshold policy. |
| Warmup policy | Keep `warmup=none_configured` and treat it as a limitation, not hidden methodology. |
| Variance policy | Keep `variance=not_computed_single_sample` pending the Day 9 regression policy decision. |
| Threshold policy | Provisional threshold-free policy: `baseline=n/a`, `threshold=n/a`, and `claim_boundary=hosted_selected_threshold_free`. |
| Acceptance meaning | Fresh methodology metadata and selected artifact publication for one hosted benchmark row; not a timing threshold or speed claim. |

This selection follows the Sprint 187 default candidate. Day 2 did not find a
stronger reason to switch to another benchmark family.

## Candidate Ranking

Scores use `1` for weak/high-risk and `5` for strong/low-risk.

| Rank | Candidate | Evidence value | Determinism | Hosted runtime fit | Metadata readiness | Claim safety | Implementation fit | Total | Day 2 disposition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `bench_refactor_csc` on `nos4.mtx --repeat 1` | 5 | 5 | 5 | 5 | 4 | 5 | 29 | Selected. |
| 2 | S6 from `performance-sentinels` | 4 | 5 | 4 | 4 | 3 | 3 | 23 | Keep as Day 9 regression-policy input, not the selected hosted lane. |
| 3 | `bench_chol_csc` canonical row | 3 | 5 | 5 | 4 | 3 | 3 | 23 | Defer; useful context but currently unselected and adjacent to Cholesky comparison work. |
| 4 | `bench_iterative_reuse` canonical row | 4 | 4 | 4 | 3 | 3 | 3 | 21 | Defer; convergence and reuse interpretation broaden the claim surface. |
| 5 | `bench_eigs_reuse` canonical row | 4 | 4 | 4 | 3 | 2 | 3 | 20 | Defer; eigensolver timing and convergence semantics add avoidable risk. |
| 6 | `wall-check` | 3 | 3 | 3 | 2 | 2 | 2 | 15 | Reject for Sprint 192 hosted evidence; existing local threshold semantics do not transfer cleanly to hosted methodology. |

## Candidate Audit Findings

| Candidate | Finding | Rationale |
| --- | --- | --- |
| `bench_refactor_csc` on `nos4.mtx --repeat 1` | Selected. | It is already the only selected benchmark manifest row, has source-controlled fixture input, generated index/manifest metadata, a dedicated freshness checker, hosted Linux workflow wiring, and focused tests. |
| S6 from `performance-sentinels` | Retain as policy input. | It uses the same selected fixture/command as a local smoke ceiling, but promoting its threshold to hosted evidence needs separate baseline and variance policy. |
| `bench_chol_csc` canonical row | Deferred. | It is generated in the canonical bundle, but selecting it would add a second hosted performance row and blur Sprint 191/190 Cholesky evidence with performance evidence. |
| `bench_iterative_reuse` canonical row | Deferred. | It has reuse value, but selected methodology would need additional convergence, repeat, and workload interpretation before promotion. |
| `bench_eigs_reuse` canonical row | Deferred. | It has useful eigensolver coverage, but repeated eigensolver timing and convergence behavior create a larger claim and variance surface. |
| `wall-check` | Rejected for this sprint. | It is a narrow local hard timing gate with machine-class assumptions, not a methodology-bound hosted publication lane. |

## Current Selected Lane Evidence

Day 2 generated the canonical report bundle with:

```sh
make bench-canonical-report
```

The command wrote:

- `build/bench-reports/canonical/bench_refactor_csc.csv`
- `build/bench-reports/canonical/bench_chol_csc.csv`
- `build/bench-reports/canonical/bench_iterative_reuse.csv`
- `build/bench-reports/canonical/bench_eigs_reuse.csv`
- `build/bench-reports/canonical/index.tsv`
- `build/bench-reports/canonical/manifest.txt`

The selected `bench_refactor_csc` row recorded:

| Field | Day 2 local value |
| --- | --- |
| `surface` | `canonical` |
| `category` | `measurement` |
| `artifact` | `bench_refactor_csc` |
| `relative_path` | `bench_refactor_csc.csv` |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| `report_family` | `benchmark` |
| `status` | `measurement` |
| `support_tier` | `local_only` |
| `claim_boundary` | `local_threshold_free` |
| `fixture_or_workload` | `nos4.mtx` |
| `matrix_size` | `n=100` |
| `repeat_semantics` | `configured_repeat_1` |
| `warmup` | `none_configured` |
| `variance` | `not_computed_single_sample` |
| `baseline` | `n/a` |
| `threshold` | `n/a` |
| `backend_context` | `n/a` |
| `methodology_notes` | `threshold_free_local_measurement;not_portable_performance_claim` |

Day 2 local generation completed in about 14 seconds on this machine. The
hosted workflow already declares `timeout-minutes: 10` for
`hosted-performance-freshness`, which remains the provisional runtime budget
until hosted evidence or Day 7 design work justifies a change.

## Selected Lane Boundaries

Sprint 192 implementation should stay within these boundaries:

- one selected target ID: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- one selected artifact: `bench_refactor_csc`;
- one selected workload: `tests/data/suitesparse/nos4.mtx --repeat 1`;
- one hosted workflow job: `hosted-performance-freshness` on Linux;
- no Windows or macOS selected benchmark freshness promotion;
- no promotion of `bench_chol_csc`, `bench_iterative_reuse`, or
  `bench_eigs_reuse` from contextual canonical rows to selected hosted rows;
- no portable performance, speedup, architecture-independent, external-library,
  package, ABI, release, or state-of-the-art claim.

## Implementation Direction

Day 2 sets this implementation direction for Days 3-14:

1. Keep `bench_refactor_csc` on `nos4.mtx --repeat 1` as the selected lane.
2. Use Day 3 to define the methodology contract around the existing field set,
   with explicit treatment of single-sample and no-warmup limitations.
3. Use Day 6 to verify or harden `normalize_report_index.py --family benchmark
   --check-freshness` against the selected benchmark contract.
4. Use Day 7/Day 8 to review whether hosted artifact upload scope should stay
   as the full canonical bundle or narrow to manifest-required selected files.
5. Use Day 9 to decide whether threshold-free remains the right policy or
   whether a conservative sentinel can be justified without overclaiming.

## Retained Non-Claims

Day 2 retains these non-claims:

- no portable performance claim;
- no performance superiority claim;
- no architecture-independent speedup claim;
- no state-of-the-art performance claim;
- no broad benchmark-family publication claim;
- no package-manager proof;
- no shared-library or ABI proof;
- no external-library parity claim;
- no Windows selected benchmark freshness;
- no macOS selected benchmark freshness.

## Day 2 Validation

Commands run:

```sh
make bench-canonical-report
python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode local
BENCH_CANONICAL_REPORT_LABEL=sprint-192-day2-hosted-shape SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-latest SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags SPARSE_CANONICAL_CPU_MODEL=unknown SPARSE_CANONICAL_BUILD_MODE=serial make bench-canonical-report
python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 tests/test_bench_canonical_freshness.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- canonical report generation passed and wrote the six-file bundle;
- local selected benchmark freshness passed;
- hosted-shape selected benchmark freshness passed with emulated hosted
  metadata;
- normalized benchmark freshness passed with five rows;
- benchmark freshness regression tests passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed on Day 2, so `make format && make lint &&
  make test` is not required for this day.

Generated benchmark artifacts remain ignored under `build/`.
