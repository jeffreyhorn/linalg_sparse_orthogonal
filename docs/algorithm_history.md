# Algorithm History and Measurement Appendix

This appendix preserves historical algorithm measurements, sprint-era
implementation decisions, benchmark interpretation context, and links to
planning artifacts. It is not the first-use adoption guide, current API
contract, install/support contract, package or ABI reference, or a portable
performance guarantee.

Use these current surfaces first:

- [`README.md`](../README.md) for the front-door project overview.
- [`docs/solver_selection.md`](solver_selection.md) for solver-choice
  guidance.
- [`examples/README.md`](../examples/README.md) for maintained runnable
  examples.
- [`benchmarks/README.md`](../benchmarks/README.md) for benchmark commands,
  generated reports, and local-measurement interpretation.
- [`docs/algorithm.md`](algorithm.md) for the concise current algorithm
  reference.
- [`docs/maintainer_guide.md`](maintainer_guide.md) for support-tier,
  validation, package, ABI, and maintainer-policy ownership.

## Scope and Claim Boundary

Historical measurements in this appendix are branch-local, fixture-specific,
configuration-sensitive evidence. They do not create portable performance
claims, external-library parity claims, package or ABI support claims, platform
support claims, or broad correctness guarantees.

Generated report and index references keep the Sprint 131 boundary: report
indexes describe traceability and freshness, not broad correctness, coverage
completeness, release status, or performance guarantees.

## Direct Solver and Factorization History

### Cholesky Fill and CSC Backend Measurements

The current algorithm reference keeps only the current Cholesky and CSC
backend behavior. Historical SPD fill and timing context belongs here:

- early SuiteSparse no-reorder comparisons showed Cholesky factor storage
  below LU factor storage on representative SPD fixtures such as `nos4` and
  `bcsstk04`;
- later CSC Cholesky measurements compared the linked-list path,
  scalar-CSC path, and supernodal CSC path on `nos4`, `bcsstk04`,
  `bcsstk14`, `s3rmt3m3`, `Kuu`, and `Pres_Poisson`;
- scalar-CSC speedups generally grew with matrix size, while `Kuu` exposed a
  localized scalar-CSC regression from repeated column shifting during
  drop-tolerance pruning;
- the supernodal path avoided that `Kuu` scalar regression and became the
  maintained large-SPD direction;
- analyze-once / factor-many measurements showed a larger CSC advantage after
  the AMD cost amortized across repeated numeric refactors.

Evidence links:

- [`docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md`](planning/EPIC_2/SPRINT_17/PERF_NOTES.md)
- [`docs/planning/EPIC_2/SPRINT_18/bench_day12.txt`](planning/EPIC_2/SPRINT_18/bench_day12.txt)
- [`docs/planning/EPIC_2/SPRINT_19/bench_day2_refactor.txt`](planning/EPIC_2/SPRINT_19/bench_day2_refactor.txt)

These rows remain local fixture evidence, not a portable performance claim.

### Supernodal Cholesky Proof Trail

The supernodal Cholesky path was validated by comparing scalar and batched
results on small SPD fixtures, AMD-permuted fixtures, dense synthetic blocks,
block-diagonal matrices, and boundary cases with both singleton and large
supernode dispatch paths. The current reference keeps the algorithmic
conditions and helper responsibilities; the day-by-day validation chronology
belongs here as historical proof context.

### CSC LDLT Scaffolding and Supernodal History

The CSC LDLT path first used a scaffold that expanded the lower triangle into
a full symmetric `SparseMatrix`, called the linked-list
`sparse_ldlt_factor`, and unpacked the result back into CSC layout. The native
CSC Bunch-Kaufman kernel later replaced that wrapper as the production path.

Supernodal LDLT mirrored the Cholesky batched path but added LDLT-specific
constraints:

- 2x2 Bunch-Kaufman pivot pairs must not straddle a supernode boundary;
- the batched path depends on cached `pivot_size` from a previous native CSC
  factorization;
- dense block factorization must reproduce the cached pivot pattern or fall
  back to scalar handling.

Historical Sprint 19 captures recorded batched LDLT wins on SPD fixtures such
as `bcsstk14` and `bcsstk04`, while indefinite KKT-style matrices remained
blocked on scalar handling until full symbolic-pattern preallocation could
cover the batched cmod fill.

Evidence link:

- [`docs/planning/EPIC_2/SPRINT_19/bench_day14.txt`](planning/EPIC_2/SPRINT_19/bench_day14.txt)

### Row-Adjacency Index History

The scalar CSC LDLT kernel added a per-row adjacency index so cmod phases can
iterate prior stored columns directly instead of scanning `[0, step_k)` with a
binary search per candidate column. Historical benchmark captures showed this
restored linked-list-like sparse-row scaling for larger matrices.

## Reordering and Fill History

### AMD Quotient-Graph Chronology

The quotient-graph AMD implementation replaced the earlier bitset route to
avoid quadratic memory. Its current production form is variable-only:
affected adjacency lists are rebuilt by sorted merge, exact minimum degree is
recomputed on rebuilt lists, and workspace stays bounded by
`5 * nnz + 6 * n + 1` integer entries.

Historical context:

- Sprint 22 introduced the simplified quotient-graph baseline and measured
  large memory reductions versus the bitset implementation.
- Sprint 23 attempted canonical Davis mechanisms including element
  absorption, supervariable detection, approximate-degree formula support, and
  dense-row skip.
- Sprint 24 reverted those additions after the closing benchmark/profile work
  showed large wall-time regressions on irregular SuiteSparse SPD fixtures.
- The revert restored the variable-only baseline and preserved fill behavior.

Evidence links:

- [`docs/planning/EPIC_2/SPRINT_22/bench_day13_amd_qg.txt`](planning/EPIC_2/SPRINT_22/bench_day13_amd_qg.txt)
- [`docs/planning/EPIC_2/SPRINT_24/fix_decision_day1.md`](planning/EPIC_2/SPRINT_24/fix_decision_day1.md)
- [`docs/planning/EPIC_2/SPRINT_24/bench_summary_day9.md`](planning/EPIC_2/SPRINT_24/bench_summary_day9.md)

### Nested Dissection Sprint 22-28 Chronology

The current reference describes ND as a multilevel vertex-separator pipeline.
The historical chronology is longer:

- Sprint 22 introduced the initial ND route.
- Sprint 23 added leaf-AMD splicing and FM bucket improvements.
- Sprints 24-26 explored coarsening floor, HCC, spectral bisection,
  intermediate/final FM strategies, separator lift strategies, and root
  bisection variants.
- Sprint 27 made the production default more stable with Kuu-safe HCC routing
  and an `ND_BASE_THRESHOLD` default of 128.
- Sprint 28 added non-pipeline-level postorder infrastructure and retired the
  literal `0.85x` Pres_Poisson target after repeated empirical attempts failed
  to meet it without unacceptable regressions elsewhere.

The retained high-level lesson is that ND has workload-class-sensitive
behavior: regular PDE meshes and irregular fixtures respond differently to
the same tuning knobs. Advisory environment-variable combinations should be
read as fixture-class local evidence, not universal fill superiority.

Evidence links:

- [`docs/planning/EPIC_2/SPRINT_24/nd_coarsen_floor_decision.md`](planning/EPIC_2/SPRINT_24/nd_coarsen_floor_decision.md)
- [`docs/planning/EPIC_2/SPRINT_25/coarsening_decision.md`](planning/EPIC_2/SPRINT_25/coarsening_decision.md)
- [`docs/planning/EPIC_2/SPRINT_25/spectral_bisection_decision.md`](planning/EPIC_2/SPRINT_25/spectral_bisection_decision.md)
- [`docs/planning/EPIC_2/SPRINT_26/hcc_sep_zero_diagnosis.md`](planning/EPIC_2/SPRINT_26/hcc_sep_zero_diagnosis.md)
- [`docs/planning/EPIC_2/SPRINT_27/hcc_kuu_diagnosis.md`](planning/EPIC_2/SPRINT_27/hcc_kuu_diagnosis.md)
- [`docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_decision.md`](planning/EPIC_2/SPRINT_27/nd_base_threshold_decision.md)
- [`docs/planning/EPIC_2/SPRINT_28/headline_summary.md`](planning/EPIC_2/SPRINT_28/headline_summary.md)
- [`docs/planning/EPIC_2/SPRINT_28/non_pipeline_decision.md`](planning/EPIC_2/SPRINT_28/non_pipeline_decision.md)

### Retired Pres_Poisson Target

The historical Pres_Poisson default-path target of `<= 0.85x` of AMD fill was
retired after repeated Sprint 24-28 evidence showed the in-house multilevel
pipeline could not reach it safely. The current default landed near `0.923x`
on the historical capture, and future attempts require fundamentally
different machinery such as production METIS interop, coordinate-aware
ordering, or another new pivot rather than more stackable advisory knobs.

This retired target remains historical context only. It is not a current
support promise or a current benchmark threshold.

## Benchmark and Report Governance History

### Reorder/Fill Report Interpretation

Reorder and graph reports separate structural, runtime, and guardrail
evidence:

- `nnz_L`, `nnz_R`, `nnz_LU`, `fill_ratio`, `bandwidth`, and
  `separator_size` are structural context for a named fixture and ordering.
- `reorder_ms`, `factor_ms`, and command wall time are local timing context
  that depends on host, compiler, build mode, backend, and thread settings.
- `peak_rss_mb` is a platform-local memory proxy for before/after
  investigation, not cross-platform pass/fail evidence.
- large-matrix guardrail reports are bounded structural guardrail artifacts,
  not broad scalability or portable performance proof.

Current benchmark commands, report directories, CSV fields, and generated
index interpretation live in [`benchmarks/README.md`](../benchmarks/README.md).

### Wall-Check and Sentinel History

The `make wall-check` timing gate was introduced after a qg-AMD wall-time
regression accumulated across several day-by-day commits without an
intermediate signal. The gate intentionally uses a small fixture set so it can
catch large single-day regressions cheaply:

- `bench_amd_qg --only bcsstk14` for qg-AMD timing;
- `bench_reorder --only Pres_Poisson --skip-factor` for AMD and ND reorder
  timing.

The baseline file uses `KEY=VALUE_MS` rows with comments that explain when
each baseline landed and what previous values were. Later work added
`pres_poisson_nd_ms` with a wider threshold because the ND partition phase
showed higher local variance on sustained macOS arm64 runs.

`make performance-sentinels` later wrapped `wall-check` and added
threshold-free Cholesky CSC backend-aware rows under
`build/bench-reports/sentinels/`. Those additional rows remain local
measurement context under recorded backend and OpenMP runtime settings.

Evidence links:

- [`docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt`](planning/EPIC_2/SPRINT_24/wall_check_baseline.txt)
- [`docs/planning/EPIC_2/SPRINT_25/profile_day11_pres_poisson_nd.txt`](planning/EPIC_2/SPRINT_25/profile_day11_pres_poisson_nd.txt)
- [`docs/planning/EPIC_2/SPRINT_25/nd_wall_time_decision.md`](planning/EPIC_2/SPRINT_25/nd_wall_time_decision.md)

### Report-Index Boundary

Sprint 131 keeps generated report indexes as traceability and freshness
evidence. Index rows do not create broad correctness, coverage-completeness,
release, or performance guarantees. Any future report-index adoption language
should preserve that generated-versus-curated boundary.

## Eigensolver Implementation History

### Backend Rollout

The symmetric eigensolver surface grew from a grow-m Lanczos route into three
publicly selectable backends:

- grow-m Lanczos for small and straightforward symmetric problems;
- thick-restart Lanczos for bounded-memory Krylov solves that preserve locked
  Ritz-pair progress across phases;
- LOBPCG for preconditioned block eigenvalue workflows.

The current reference keeps the mathematical behavior and API fields. This
appendix preserves the rollout and measurement context.

### OpenMP Reorthogonalization History

MGS reorthogonalization parallelizes only the inner vector axis under
`-DSPARSE_OPENMP`; the outer prior-vector loop remains serial because each
projection updates the vector used by the next projection. A compile-time
threshold avoids paying fork/join overhead on small `n`.

Evidence link:

- [`docs/planning/EPIC_2/SPRINT_21/bench_day6_omp_scaling.txt`](planning/EPIC_2/SPRINT_21/bench_day6_omp_scaling.txt)

### Thick-Restart and Shift-Invert History

Thick-restart Lanczos reduced peak basis storage by preserving locked Ritz
pairs in an arrowhead state and extending the Krylov subspace across phases.
Shift-invert mode used LDLT factorization of `A - sigma I` so interior
eigenvalues become exterior eigenvalues of the inverse operator.

Historical measurement captures compared grow-m Lanczos, thick-restart, and
shift-invert behavior across SuiteSparse fixtures. Those captures remain
local fixture evidence, not broad backend or portable performance claims.

Evidence links:

- [`docs/planning/EPIC_2/SPRINT_20/bench_day13_lanczos.txt`](planning/EPIC_2/SPRINT_20/bench_day13_lanczos.txt)
- [`docs/planning/EPIC_2/SPRINT_21/bench_day14.txt`](planning/EPIC_2/SPRINT_21/bench_day14.txt)

### LOBPCG History

LOBPCG was added for workloads where a preconditioned block method can beat
sequential Lanczos convergence, especially on ill-conditioned SPD problems or
clustered requested eigenvalues. Historical comparison rows include
preconditioned and unpreconditioned mode pivots; they should be read as local
measurement context.

Evidence link:

- [`docs/planning/EPIC_2/SPRINT_21/bench_day14_compare.txt`](planning/EPIC_2/SPRINT_21/bench_day14_compare.txt)

## Planning Artifact Links

Reserved for curated links to planning artifacts whose evidence remains
historical context rather than current adoption guidance.
