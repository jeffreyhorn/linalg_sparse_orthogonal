# Benchmarks

Permanent benchmark drivers for the sparse linear algebra library.
Built via `make bench`; invoked individually via `make bench-<name>`
or by running the binary in `build/` directly.

Use this file for benchmark command groups, CSV fields, report artifacts, and
measurement caveats. Use the [README](../README.md) for the top-level project
route, [docs/cookbook.md](../docs/cookbook.md) for compressed-first workflow
handoffs, [examples/README](../examples/README.md) for API adoption examples,
and the [Maintainer Guide](../docs/maintainer_guide.md) for reviewed-baseline
and proof-owner interpretation.

## Quick Navigation

| Need | Section |
|---|---|
| Read benchmark rows without overclaiming performance | [Reading Benchmark Results](#reading-benchmark-results) |
| Compile benchmarks and examples without running long workloads | [Compile-only gate](#compile-only-gate) |
| Find reorder benchmark coverage and COLAMD routing | [Reorder coverage](#reorder-coverage) |
| Pick the benchmark family for a workflow | [Workflow groups](#workflow-groups) |
| Understand canonical, runtime, and exploratory lanes | [Current maintained category split](#current-maintained-category-split) |
| Capture threshold-free canonical reports | [Current maintained category split](#current-maintained-category-split) |
| Capture bounded local sentinel reports | [Current maintained category split](#current-maintained-category-split) |
| Capture large-matrix guardrail reports | [Current maintained category split](#current-maintained-category-split) |
| Find generated report indexes and freshness metadata | [Report index handoff](#report-index-handoff) |
| Use the main solver-harness CLI | [bench_main](#bench_main) |
| Use the symmetric eigensolver CLI and CSV schemas | [bench_eigs](#bench_eigs) |

## Reading Benchmark Results

Benchmarks are local measurement tools. They help answer questions like:

- is this branch faster or slower than another branch on the same machine?
- does a repeated-run workflow reduce setup cost for this fixture?
- did a backend, reorder, or preconditioner choice change local runtime,
  iteration count, fill, residual, or selected path?

They do not prove portable performance across machines, compilers, operating
systems, BLAS or dense-kernel backends, OpenMP runtimes, thread counts, matrix
corpora, or build options. Treat timing columns as comparable only when the
environment and command line are recorded or intentionally held fixed.

When reading CSV rows:

- identify the workload first: matrix, dimensions, fixture family, solver,
  backend, reorder, preconditioner, and scenario fields define the comparison
  cell
- use residual, status, convergence, iteration, fill, and path columns before
  interpreting timing columns
- compare timing rows across branches only when build mode, compiler, backend
  request/selection, `OMP_NUM_THREADS`, matrix corpus, and repeat count are
  aligned
- treat generated `manifest.txt` and `index.tsv` files as the source of
  command, branch, commit, compiler/platform, artifact, and label context
- keep examples as the learning path and tests as correctness owners; benchmark
  CSVs are measurement artifacts

Recommended handoff from API adoption:

| Need | Start here | Read as |
|---|---|---|
| Learn the API workflow | `examples/README.md` | runnable usage, not timing evidence |
| Choose a solver family | `docs/solver_selection.md` | problem-shape guidance |
| Capture the maintained local benchmark surface | `make bench-canonical-report` | threshold-free branch-local CSV bundle |
| Check selected canonical report freshness | `make bench-canonical-report-freshness` | selected `bench_refactor_csc` metadata/artifact freshness, not timing proof |
| Check bounded local sentinel behavior | `make performance-sentinels` | local sentinel report plus the existing `wall-check` gate |
| Investigate a specific backend or algorithm | individual `bench_*` binary | focused local measurement |
| Validate correctness or public contract | `make test` and focused tests | regression/oracle/property evidence |

The practical workflow is:

1. Pick the API route from the README, solver-selection guide, or examples.
2. Run the smallest relevant benchmark or report target.
3. Save the emitted CSV plus manifest/index context.
4. Compare against a second run only after checking that workload and
   environment fields still line up.
5. Treat differences as local evidence that may justify follow-up profiling,
   not as a universal performance ranking.

## Compile-only gate

Routine local quality checks should now catch benchmark/example compile
drift without executing the long-running benchmark workloads:

- `make tooling-build`
  - builds all benchmark and example binaries
  - does not run them
- `make lint`
  - now includes the same compile-only tooling gate before the existing
    source-level lint passes
- `make quality-review-compile`
  - reviewed local compile-quality wrapper
  - routes through `format-check` + `lint`, so benchmark/example compile drift
    is covered there as well

Focused subsets remain available:

- `make bench-build`
- `make examples-build`

This compile-only gate is a support check for benchmark drift; it is not the
owner of repository-wide reviewed-baseline, dead-code, or maintainer-policy
claims.

## Reorder coverage

The benchmark entry points intentionally expose different reorder surfaces
depending on what the underlying factorization path actually supports:

- `bench_main --reorder none|rcm|amd|nd`
  - solver-harness entry point for LU and `--cholesky`
  - intentionally does not accept `colamd`
  - LU / Cholesky factorization options use symmetric reorderings only
  - `--help` and invalid `--reorder` errors now explicitly point users to
    `bench_reorder` / `bench_colamd` for COLAMD comparisons
- `bench_reorder`
  - cross-ordering comparison harness for `none`, `rcm`, `amd`, `colamd`,
    and `nd`
  - supports both direct reorder calls and `--reorder-via-analyze`
  - supports `--sprint86-slice` for the bounded ND rerun corpus
    (`bcsstk14`, `Pres_Poisson`); the flag name is historical, but the current
    use is a small, named fixture slice
  - emits stable CSV rows with:
    - `matrix`
    - `n`
    - `reorder`
    - `nnz_L`
    - `reorder_ms`
    - `factor_ms`
    - `reorder_path`
    - `fixture_slice`
    - `nd_base_threshold`
  - interpret the added evidence fields narrowly:
    - `reorder_path` = `direct` or `analyze`
    - `fixture_slice` = `sprint86` when `--sprint86-slice` is active,
      otherwise `all`
    - `nd_base_threshold` = current ND base-threshold setting for the run; only `reorder=nd` rows use it
- `bench_colamd` and `example_colamd`
  - QR-focused comparison tools for `none`, `amd`, and `colamd`
  - use the same lowercase mode labels as the benchmark CLI
- `bench_chol_csc` and `bench_ldlt_csc`
  - backend-comparison tools, not general reorder sweeps
  - keep their fixed reorder choices in the code path being compared so
    backend timings stay like-for-like

| Binary                 | Topic                                                   | Smoke target              |
|------------------------|---------------------------------------------------------|---------------------------|
| `bench_main`           | One-shot LU / Cholesky / SpMV / iterative harness       | `make bench-suitesparse`  |
| `bench_scaling`        | LU scaling sweep                                        | (in `make bench`)         |
| `bench_fillin`         | Fill-in vs reordering quality                           | (in `make bench`)         |
| `bench_convergence`    | Iterative-solver convergence rates                      | (in `make bench`)         |
| `bench_svd`            | Sparse SVD (bidiagonalisation + QR)                     | (in `make bench`)         |
| `bench_refactor`       | Direct repeated-run lifecycle: Cholesky analyze once / refactor many | (in `make bench`) |
| `bench_refactor_csc`   | Direct repeated-run lifecycle measurement: SPD Cholesky by default, plus optional indefinite LDL^T KKT mode | (in `make bench`) |
| `bench_iterative_reuse`| Public repeated-run iterative handle measurement: CG, GMRES, MINRES | (in `make bench`)    |
| `bench_eigs_reuse`     | Public repeated-run eigensolver handle measurement: grow-m, thick-restart, explicit LOBPCG | (in `make bench`) |
| `bench_colamd`         | QR/COLAMD ordering quality                              | (in `make bench`)         |
| `bench_bicgstab`       | BiCGStab convergence                                    | (in `make bench`)         |
| `bench_chol_csc`       | CSC Cholesky backend comparison                         | (in `make bench`)         |
| `bench_ldlt_csc`       | LDL^T linked-list vs CSC + dispatch                     | (in `make bench`)         |
| `bench_eigs`           | Symmetric eigensolver backend sweep                     | `make bench-eigs`         |

## Workflow groups

The shipped benchmark surfaces are easiest to read in four bounded groups:

- one-shot compatibility/comparison:
  - `bench_main`
  - `bench_scaling`
  - `bench_fillin`
  - `bench_convergence`
  - `bench_svd`
  - `bench_colamd`
  - `bench_bicgstab`
  - `bench_chol_csc`
  - `bench_ldlt_csc`
  - `bench_eigs`
- direct repeated-run lifecycle:
  - `bench_refactor`
  - `bench_refactor_csc`
- iterative public-handle reuse:
  - `bench_iterative_reuse`
- eigensolver public-handle reuse:
  - `bench_eigs_reuse`

## Current maintained category split

The current maintained performance-governance split is:

- canonical maintained measurement surface:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- regression-sensitive runtime lane:
  - `bench_scaling`
  - `bench_fillin`
  - `bench_colamd`
  - `bench_reorder --skip-factor`
  - `bench_amd_qg` remains adjacent to this lane but should stay bounded
- exploratory or broader comparison lane:
  - `bench_main`
  - `bench_convergence`
  - `bench_svd`
  - `bench_bicgstab`
  - `bench_eigs`
  - broader `bench_reorder`

Interpretation:

- the canonical maintained surface is where efficiency follow-through should
  stay centered
- the runtime lane is useful for current-branch checks but should not be
  marketed as a portable performance guarantee
- the exploratory lane stays valuable without defining the compact maintained
  benchmark face of the repo
- examples remain the API-adoption teaching surface; benchmarks remain the
  workflow/performance measurement surface

For threshold-free local or CI-friendly reporting on the maintained canonical
surface, use:

- `make bench-canonical-report`
- optionally set `BENCH_CANONICAL_REPORT_LABEL=<label>` on that command to
  attach a bounded comparison label to the bundle metadata

That target writes one CSV per canonical maintained benchmark under:

- `build/bench-reports/canonical/`

plus a bounded bundle-level metadata surface:

- `manifest.txt`
  - exact command mapping
  - explicit artifact inventory
  - generated timestamp
  - bounded report label from `BENCH_CANONICAL_REPORT_LABEL`
  - git commit / branch when locally available
  - platform and compiler string
  - runner context, build flags, and CPU model when supplied by local or
    hosted report generation
  - build mode and `OMP_NUM_THREADS`
- `index.tsv`
  - one structured row per emitted canonical artifact
  - includes generated time, git branch/commit, platform, compiler, build
    mode, runner context, build flags, CPU model, and `OMP_NUM_THREADS`
  - keeps the same bounded canonical surface identity and command mapping in a
    machine-readable comparison form
  - appends methodology fields for publication review:
    - `report_family`
    - `status`
    - `support_tier`
    - `claim_boundary`
    - `fixture_or_workload`
    - `matrix_size`
    - `repeat_semantics`
    - `warmup`
    - `variance`
    - `baseline`
    - `threshold`
    - `backend_context`
    - `methodology_notes`

Use `make bench-canonical-report-freshness` when you need the selected
canonical performance report to be current and interpretable. That target
regenerates the canonical bundle and checks only:

- `artifact=bench_refactor_csc`
- `relative_path=bench_refactor_csc.csv`
- `command=tests/data/suitesparse/nos4.mtx --repeat 1`
- `fixture_or_workload=nos4.mtx`
- `repeat_semantics=configured_repeat_1`
- `baseline=n/a`
- `threshold=n/a`

The reviewed Linux hosted selected-performance lane runs the same selected-row
check in hosted mode with `support_tier=hosted_selected` and
`claim_boundary=hosted_selected_threshold_free`. Hosted mode also requires a
non-local runner context, recorded build flags, and a non-`unlabeled` report
label. Those hosted-selected support and claim fields apply only to the
selected `bench_refactor_csc` row; unselected canonical rows remain
`local_only` / `local_threshold_free`. `cpu_model=unknown` remains acceptable
because GitHub-hosted runner CPU assignment can vary.

This is intentionally not a pass/fail timing gate:

- compare the emitted CSV rows across branches or runs
- treat it as artifact-friendly reporting, not a portable performance guarantee
- use the bundle metadata to make before/after or cross-branch snapshots
  easier to line up without widening the benchmark claim surface
- read platform, compiler, build mode, and `OMP_NUM_THREADS` as local
  comparison context, not as a portability or OpenMP speedup claim
- read `status=measurement`, `support_tier=local_only`, and
  `claim_boundary=local_threshold_free` as the unselected canonical row
  boundary
- read `baseline=n/a` and `threshold=n/a` as proof that canonical rows are not
  hard timing gates
- read `warmup=not_recorded` and `variance=not_recorded` literally; do not
  infer warmup, samples, statistical variance, or confidence intervals
- read the hosted selected-performance lane as freshness and methodology
  evidence for the selected `bench_refactor_csc` row only, not as portable
  speed evidence or broad benchmark publication
- keep `bench-fast` as the bounded runtime lane and `wall-check` as the narrow
  thresholded regression gate that already has a justified machine-class
  baseline
- for the bounded ND rerun slice, use:
  - `make bench-reorder-sprint86`
  - this expands to `bench_reorder --sprint86-slice --skip-factor`
  - the target name is historical; treat the current output as branch-local
    evidence for the bounded ND lane, not as a canonical benchmark claim

For bounded local regression sentinels, use:

- `make performance-sentinels`

That target writes a compact bundle under:

- `build/bench-reports/sentinels/`

with these artifacts:

- `sentinels.tsv`
  - structured rows for each sentinel metric
  - includes report family, sentinel id, status, support tier, claim boundary,
    command, build mode, `OMP_NUM_THREADS`, fixture, metric, value, baseline,
    threshold, artifact, backend request/selection/fallback, dense-kernel
    descriptor, panel-solver descriptor, and notes
  - appends methodology fields:
    - `baseline_provenance`
    - `repeat_semantics`
    - `warmup`
    - `variance`
    - `methodology_notes`
- `manifest.txt`
  - git commit and branch when available
  - platform and compiler string
  - `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND`
  - exact sentinel commands
  - S5 baseline provenance, S5/S2/S3 repeat semantics, warmup state, variance
    state, and non-superiority caveats
- `wall_check.txt`
  - raw output from the existing thresholded `wall-check` lane when that lane
    runs
- `bench_chol_csc_nos4.csv`
  - raw threshold-free Cholesky CSC row when the S2 lane runs
- `bench_refactor_csc_kkt.csv`
  - raw threshold-free LDLT KKT row when the S3 lane runs

Interpret the bundle narrowly:

- S5 wraps the existing `make wall-check` threshold gate and may fail the
  target.
- S2 captures `bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` as
  threshold-free report context only.
- S3 captures `bench_refactor_csc --indefinite-kkt --repeat 1` as
  threshold-free LDLT backend report context only.
- S5 rows report backend-specific fields as `n/a`; S2 rows preserve the
  Cholesky dense-kernel and panel-solver descriptors parsed from
  `bench_chol_csc`.
- S3 rows preserve the LDLT dense-backend request, selected backend, and
  fallback fields parsed from `bench_refactor_csc`.
- Missing binaries, fixtures, or baselines are reported as explicit skip rows
  where practical; skips are not passes.
- The bundle is local regression evidence, not a portable timing guarantee.
- Timing rows are meaningful only with the recorded backend request/fallback
  context and OpenMP runtime settings.
- S5 `pass` or `fail` status is meaningful only with the recorded baseline,
  threshold, fixture, command, baseline provenance, and local machine context.
- S2 and S3 `status=report` rows are not pass/fail rows and do not prove
  backend superiority.
- Rows with `warmup=not_recorded` or `variance=not_recorded` must not be
  described as warmup-controlled or statistical summaries.
- Generated sentinel artifacts under `build/bench-reports/sentinels/` should
  be regenerated from `make performance-sentinels`; do not hand-edit them.

Report-index handoff:

- generated canonical and sentinel artifacts carry enough branch, commit,
  command, artifact, platform, compiler, build mode, and thread context to be
  indexed as local evidence
- `support_tier` and `claim_boundary` fields in sentinel rows should be
  preserved by any report index instead of collapsed into pass/fail timing
  status
- canonical `support_tier`, `claim_boundary`, `repeat_semantics`, `warmup`,
  `variance`, `baseline`, `threshold`, and `methodology_notes` fields should
  remain visible in downstream summaries
- sentinel `baseline_provenance`, `repeat_semantics`, `warmup`, `variance`,
  and `methodology_notes` fields should remain visible in downstream summaries
- backend fields that report `n/a`, `unknown`, or fallback context must remain
  visible in any downstream index because they limit comparison scope
- canonical rows remain threshold-free even when their index includes
  platform/compiler/build/thread context

For bounded large-matrix reorder and graph guardrails, use:

- `make large-matrix-guardrails`

That target writes a compact bundle under:

- `build/bench-reports/large-matrix-guardrails/`

with these default artifacts:

- `index.tsv`
  - one row per guardrail lane
  - reviewed lanes `G1` through `G4` should report `pass`
  - supplemental lanes `S1` and `S2` report `skip` unless explicitly enabled
- `manifest.txt`
  - branch, commit, platform, compiler, timestamp, and supplemental-mode flag
- `test_reorder_amd_qg.txt`
  - reviewed qg-AMD wrapper and `banded-n10000-bw5` structural guardrail
- `test_reorder_nd.txt`
  - reviewed ND generated-family, named-matrix, policy, and residual coverage
- `test_graph.txt`
  - reviewed graph partition, separator, generated-family, and determinism
    coverage
- `bench_reorder_sprint86.csv`
  - bounded two-fixture reorder/fill slice for `bcsstk14` and `Pres_Poisson`
  - the `sprint86` label is historical; read it as the current named
    two-fixture slice

## Report index handoff

Use generated report indexes to find artifacts and freshness context after a
report target runs. They are navigation and interpretation aids, not broader
performance, scalability, coverage, or platform guarantees.

| Report target | Report directory | Index artifact | Freshness/context artifact | Read as |
|---|---|---|---|---|
| `make bench-canonical-report` | `build/bench-reports/canonical/` | `index.tsv` | `manifest.txt` | Threshold-free local snapshot of the maintained benchmark surface. |
| `make bench-canonical-report-freshness` | `build/bench-reports/canonical/` | `index.tsv` | `manifest.txt` | Selected `bench_refactor_csc` artifact and methodology freshness for `nos4.mtx --repeat 1`; not a timing gate. |
| `make performance-sentinels` | `build/bench-reports/sentinels/` | `sentinels.tsv` | `manifest.txt` | Local sentinel bundle; only the existing wall-check lane is thresholded. |
| `make large-matrix-guardrails` | `build/bench-reports/large-matrix-guardrails/` | `index.tsv` | `manifest.txt` | Reviewed/supplemental guardrail lanes with explicit pass, fail, or skip rows. |
| `make report-index-comparison-freshness` | `build/comparison/qr_minnorm/` | `study.tsv` | `manifest.tsv` | Local fixture-level QR minimum-norm comparison against the selected source-controlled dense reference helper. |

When reading any generated report index:

- start with the generation command and report directory
- check `manifest.txt` for branch, commit, timestamp, platform, compiler,
  build mode, thread settings, and supplemental-mode context when present
- use row identity fields such as lane id, sentinel id, command, artifact, and
  category before interpreting status
- treat `skip`, `n/a`, fallback, and supplemental rows as scope information,
  not as passing evidence
- keep CSV timing rows tied to the recorded environment and command line
- regenerate reports instead of editing generated indexes by hand

Use the normalized report index when you need a cross-report view:

```sh
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel --family guardrail --family comparison \
  --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel --family guardrail --family comparison \
  --check-freshness
```

The normalized index preserves benchmark rows as local/advisory measurements,
sentinel S5 rows as the existing wall-check hard gate, sentinel S2 and S3 rows
as threshold-free context, and large-matrix guardrail reviewed/supplemental
lanes as separate row meanings. Comparison rows remain fixture-local
correctness evidence for the selected QR minimum-norm study only. Freshness
diagnostics do not convert local timing rows into portable performance claims
or local comparison rows into broad external-library parity.

The selected performance freshness check is deliberately outside broad
normalized-index promotion: it validates the `bench_refactor_csc` canonical
row, generated artifact paths, methodology metadata, manifest agreement, and
threshold-free claim boundary. It does not make `bench_chol_csc`,
`bench_iterative_reuse`, or `bench_eigs_reuse` reviewed hosted performance
evidence.

Generated benchmark, sentinel, guardrail, and normalized-index outputs belong
under ignored `build/` paths such as `build/bench-reports/sentinels/` and
`build/report-index/`. The generated comparison artifacts under
`build/comparison/qr_minnorm/` follow the same local-output rule. Regenerate
those outputs from the maintained commands; do not edit or commit generated
timing/report/comparison rows by hand.

Supplemental report mode is opt-in:

```sh
SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails
```

Supplemental mode adds threshold-free local reports for:

- `bench_reorder --skip-factor`
- `bench_amd_qg --skip-bitset`

Interpret this bundle narrowly:

- `G1` through `G3` are structural test lanes; they own pass/fail invariants
  through their test binaries, not through benchmark timing.
- `G4` validates the bounded `bench_reorder` CSV shape and structural fill
  rows; `nnz_L` is the primary fill field, while `reorder_ms` remains local
  timing context.
- `S1` and `S2` are useful maintainer reports, but they are not reviewed
  quality gates unless a future sprint defines a separate baseline contract.
- No max-RSS row in this bundle should be compared across platforms as a
  pass/fail memory claim.

The two refactor benchmarks remain the strongest benchmark-side adoption
surfaces for the public repeated-run direct lifecycle:

- `bench_refactor`
  - compares one-shot Cholesky factorization against the analyze-once /
    factor-many direct path on same-pattern value changes
  - reports one-shot average, one-time analysis cost, initial numeric factor,
    average later refactor cost, repeated-run average cost, speedup, and final
    residual
- `bench_refactor_csc`
  - default mode keeps the SPD / Cholesky repeated-run workflow and compares
    the public repeated-run path against the direct CSC/supernodal completion
    path
  - `--indefinite-kkt` switches to a synthetic above-threshold KKT saddle-point
    workload and compares the public repeated-run LDL^T path against the direct
    resolved-analysis CSC completion path
  - reads as the main throughput/measurement surface for the large-`n` CSC-backed
    repeated-run direct lane, not as the error-path contract surface; failed
    refactor preservation stays owned by `tests/test_integration.c`
  - the family-local large-`n` analysis-backed CSC helper parity stays owned
    by `tests/test_chol_csc.c`, not by this benchmark surface
  - the bounded seeded generative large-`n` lifecycle follow-through stays
    owned by `tests/test_fuzz.c`, not by this benchmark surface
  - reports CSV rows with:
    - `benchmark`
    - `category`
    - `matrix`
    - `scenario`
    - `ldlt_dense_backend_request`
    - `ldlt_dense_backend_selected`
    - `ldlt_dense_backend_fallback`
    - `analyze_ms`
    - `refactor_public_ms`
    - `refactor_csc_ms`
    - `solve_public_ms`
    - `solve_csc_ms`
    - `speedup_refactor`
    - `res_public`
    - `res_csc`
  - interpret the LDL^T backend fields narrowly:
    - SPD / Cholesky rows report `n/a` in those three columns
    - `--indefinite-kkt` reports the normalized backend request, the current
      env/probe-derived backend name, and whether the widened LDL^T selector
      reported fallback to builtin
    - this is bounded observability for the retained repeated-run LDL^T lane,
      not a broad portability or performance claim

Read that support split narrowly:

- examples such as `example_analysis` stay the adoption entry points
- `bench_refactor` / `bench_refactor_csc` stay the retained
  workflow/performance measurement surfaces after adoption
- tests still own regression/oracle/property guarantees for the large-`n`
  CSC-backed lifecycle lane

`bench_chol_csc` remains the maintained benchmark-side measurement surface for
the backend-aware Cholesky CSC lane:

- it still compares linked-list, CSC scalar, and CSC supernodal timings on
  one fixed AMD-reordered workload so fallback and accelerated paths stay
  comparable
- each CSV row now also reports:
  - `benchmark`
  - `category`
  - `scenario`
  - `csc_scalar_path`
  - `csc_supernodal_path`
  - `csc_supernodal_dense_kernel`
  - `csc_supernodal_panel_solver`
- the path columns stay intentionally stable at:
  - `scalar`
  - `supernodal`
- `csc_supernodal_dense_kernel` identifies the active dense-kernel descriptor
  behind the supernodal lane; on the current default build it reports
  `builtin`
- `csc_supernodal_panel_solver` identifies whether the supernodal lane has
  the batched panel-solve callback required by the supernodal kernel
  landing; on the current default build it reports `batched_panel`
- this keeps the benchmark refresh bounded to path measurability and
  truthfulness, not broad benchmark-governance churn
- the public callback/runtime semantics remain test-owned in
  `tests/test_integration.c`; this benchmark stays a measurement surface, not
  the owner of progress/cancel truth
- it is not the owner of the staged public-path oracle/parity lane
  or the bounded seeded lifecycle property lane; those remain test-owned in
  `tests/test_integration.c` and `tests/test_fuzz.c`

So the benchmark-side reading stays:

- benchmarks = measurement surfaces
- examples = adoption surfaces
- tests = regression/oracle/property owners

The two reuse benchmarks stay intentionally narrow and should be read as public
handle-path measurement surfaces, not broad solver bake-offs:

- `bench_iterative_reuse`
  - compares one-shot and explicit public-handle repeated-run paths for:
    - `CG`
    - `GMRES`
    - `MINRES`
  - now reports stable CSV rows with:
    - `benchmark`
    - `category`
    - `matrix`
    - `scenario`
    - `solver`
    - `one_shot_total_ms`
    - `reuse_total_ms`
    - `speedup`
    - last-run iteration / residual / convergence and status fields
  - intentionally does not claim public repeated-run-handle support for:
    - `BiCGSTAB`
    - block iterative workflows
- `bench_eigs_reuse`
  - compares one-shot and explicit public-handle repeated-run paths for:
    - grow-m Lanczos
    - thick-restart Lanczos
    - explicit LOBPCG
  - now reports stable CSV rows with:
    - `benchmark`
    - `category`
    - `matrix`
    - `scenario`
    - `backend`
    - `one_shot_median_ms`
    - `reuse_median_ms`
    - `speedup`
    - last-run iteration / convergence / residual / basis-size fields
    - retained agreement fields:
      - `lambda_max_diff`
      - `residual_diff`
      - `backend_used`
  - intentionally stays narrower than `bench_eigs`, which remains the broader
    backend/preconditioner sweep harness

## bench_main

Main solver-harness benchmark for:

- LU solve timing
- Cholesky solve timing via `--cholesky`
- SpMV-only timing via `--spmv`
- iterative-solver timing via `--iterative`

### CLI notes

```
bench_main [matrix.mtx]
bench_main --dir PATH
bench_main --size N
bench_main --help
```

Important CLI behavior:

- `--help` / `-h` now prints the live usage block
- malformed numeric or enum-like arguments fail with explicit flag-local
  diagnostics
- missing option values fail explicitly instead of silently falling through
- conflicting modes such as `--spmv --iterative` are rejected
- `--reorder` accepts only:
  - `none`
  - `rcm`
  - `amd`
  - `nd`
- unsupported `colamd` requests are intentionally redirected to:
  - `bench_reorder`
  - `bench_colamd`

## bench_eigs

Drives the three symmetric eigensolver backends — grow-m Lanczos
(`SPARSE_EIGS_BACKEND_LANCZOS`), Wu/Simon thick-restart Lanczos
(`_LANCZOS_THICK_RESTART`), and Knyazev LOBPCG (`_LOBPCG`) — across
the standard SuiteSparse + KKT corpus, with optional preconditioner
sweeps (NONE / IC0 / LDLT) for the LOBPCG branch.

### CLI summary

```
bench_eigs --sweep default [--csv] [--repeats N]   # full corpus sweep
bench_eigs --compare       [--csv] [--repeats N]   # 3-backend × 3-precond pivot
bench_eigs --matrix <path> --k N --which {LARGEST|SMALLEST|NEAREST}
                             [--sigma F] [--backend B] [--precond P]
                             [--block-size N] [--tol F] [--max-iters N]
                             [--csv] [--repeats N]
bench_eigs --help                                   # full help
```

When no mode flag is given, `--sweep default` runs. `--repeats`
defaults to 3 for the smoke target; bump to 5 when collecting more
stable local timing numbers.

### CSV schema

For `--sweep` and `--matrix` (one row per (matrix, k, which, backend,
precond) combination):

```
matrix, n, k, which, sigma, backend, precond,
iterations, peak_basis, wall_ms, residual, status
```

For `--compare` (one row per (matrix, k, which, precond), three
backend triples per row):

```
matrix, n, k, which, sigma, precond,
growing_m_iters, growing_m_wall_ms, growing_m_residual, growing_m_status,
thick_iters,     thick_wall_ms,     thick_residual,     thick_status,
lobpcg_iters,    lobpcg_wall_ms,    lobpcg_residual,    lobpcg_status
```

The `backend` / `precond` columns echo the configuration; `wall_ms`
is the median of `--repeats` runs; `peak_basis` is the doubles-times-
n peak Lanczos basis as exposed via `sparse_eigs_t.peak_basis_size`;
`status` is `OK` for converged or `NOT_CONVERGED` / etc. for
diagnosed failures.

### Default sweep

Runs the following grid:

- (nos4 n=100, bcsstk04 n=132) × {LARGEST, SMALLEST} × {k=3, k=5} × {3 backends}
- bcsstk14 n=1806 × LARGEST × {k=3, k=5} × {3 backends} (SMALLEST excluded —
  bottom of spectrum is too clustered for un-preconditioned convergence
  in a smoke-target runtime budget)
- KKT-150 (synthetic indefinite saddle-point) × NEAREST_SIGMA at σ=0
  × {3 backends}

About 33 rows total; ~15 seconds at `--repeats 2` on a 2025
M2-class development machine, ~25 seconds at `--repeats 3`.

### Compare mode

Smaller focused corpus (3 entries) × 3 preconditioners (NONE / IC0 /
LDLT), pivoted so each row shows the three backends side-by-side.
Useful for comparing how the preconditioner choice changes LOBPCG
iteration count and convergence status on the same matrix/workload.
