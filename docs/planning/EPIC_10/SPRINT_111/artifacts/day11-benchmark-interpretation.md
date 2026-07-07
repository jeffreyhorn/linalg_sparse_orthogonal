# Day 11 Benchmark Interpretation Documentation

## Purpose

Day 11 makes benchmark outputs easier to read without turning local timing
artifacts into broad portability or competitive claims. The work ties benchmark
documentation back to solver-selection guidance and keeps examples, tests, and
benchmarks in their separate roles.

## Touched Files

- `README.md`
- `benchmarks/README.md`
- `docs/solver_selection.md`
- `docs/planning/EPIC_10/SPRINT_111/WORKING_NOTES.md`
- `docs/planning/EPIC_10/SPRINT_111/artifacts/day11-benchmark-interpretation.md`

## Inventory Inputs

Benchmark and report surfaces reviewed:

- `benchmarks/README.md`
- `Makefile` benchmark targets:
  - `bench-build`
  - `tooling-build`
  - `bench-fast`
  - `bench-reorder-sprint86`
  - `bench-canonical-report`
  - `performance-sentinels`
  - `large-matrix-guardrails`
  - `wall-check`
- benchmark report scripts:
  - `scripts/bench_canonical_report.sh`
  - `scripts/performance_sentinels.sh`
  - `scripts/large_matrix_guardrails.sh`
  - `scripts/wall_check.sh`
- user-facing benchmark references in:
  - `README.md`
  - `docs/solver_selection.md`
  - `docs/tutorial.md`

## Documentation Changes

| Surface | Change |
|---|---|
| `benchmarks/README.md` | Added `Reading Benchmark Results` guidance covering local measurement scope, CSV interpretation, environment sensitivity, report manifests, and the example/test/benchmark role split. |
| `README.md` | Replaced "benchmarks prove" wording with branch-local measurement wording tied to machine, compiler, dependency, fixture, and configuration. |
| `docs/solver_selection.md` | Added comparison caveats for machine, compiler, backend selection, matrix corpus, build options, and thread settings. |

## Interpretation Contract

Benchmark rows should be read in this order:

1. Workload identity:
   - matrix;
   - dimensions;
   - fixture family;
   - solver;
   - backend;
   - reorder;
   - preconditioner;
   - scenario.
2. Correctness and status context:
   - residual;
   - status;
   - convergence;
   - iteration count;
   - fill;
   - selected path or backend.
3. Timing context:
   - wall time;
   - median or average definition when documented;
   - repeat count;
   - build mode;
   - compiler and platform;
   - dense backend and OpenMP settings.
4. Artifact context:
   - `manifest.txt`;
   - `index.tsv`;
   - command line;
   - branch;
   - commit;
   - bounded report label.

## Evidence Boundaries

- Examples teach API workflows.
- Tests own regression, oracle, and property evidence.
- Benchmarks measure local workflow/performance behavior.
- `bench-canonical-report` is a threshold-free local snapshot, not a pass/fail
  timing gate.
- `performance-sentinels` includes the existing thresholded `wall-check` lane,
  but its additional rows remain local context.
- `large-matrix-guardrails` combines structural test lanes with bounded
  benchmark/report rows; timing rows are not portable memory or speed claims.
- Individual benchmark binaries are focused measurement tools, not broad
  competitive bake-offs.

## Remaining Audience-Boundary Debt

- `README.md` still contains a compact "high-performance path" phrase for CSR
  LU. It is acceptable as a capability label, but future README passes should
  keep it tied to measured local benchmark evidence rather than universal
  speed claims.
- `benchmarks/README.md` is intentionally detailed and still carries
  maintainer-oriented lane names. Day 12 should decide whether any of that
  belongs in a maintainer-only subsection or should remain benchmark-local.
- Historical target labels such as `bench-reorder-sprint86` remain in public
  command docs because they are live target names; the docs now frame them as
  historical labels, not current sprint claims.

## Validation

Day 11 changed documentation only. Validation:

- `git diff --check`
- trailing-whitespace scan over touched Day 11 docs

## Completion Criteria Status

- Users can now read benchmark CSVs and report bundles with the needed
  workload, environment, and artifact context.
- Comparison claims are framed as local and evidence-bounded.
- Docs do not imply universal performance results.
- The solver guide links benchmark handoff back to local measurement instead
  of treating benchmark output as solver-selection truth.
