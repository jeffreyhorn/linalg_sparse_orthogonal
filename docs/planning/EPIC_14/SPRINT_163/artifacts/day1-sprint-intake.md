# Sprint 163 Day 1 Sprint Intake

## Scope

Sprint 163 selects a narrow, methodology-bound performance publication surface.
The sprint should publish selected canonical benchmark and sentinel evidence
with clear local-machine, compiler, build-mode, thread-count, command, artifact,
and claim-boundary context. It must not convert local benchmark rows into
portable performance guarantees.

## Source Artifact

The active Sprint 163 planning source is
`docs/planning/EPIC_14/PROJECT_PLAN.md`, section "Sprint 163:
Methodology-Bound Performance Publication". The user prompt referenced
`docs/planning/EPIC_12/PROJECT_PLAN.md`, but the current branch plan was
created from the Epic 14 section.

## Starting Point

- Branch: `sprint-163`
- Baseline commit: `5f0b0027`
- Prior sprint handoff: Sprint 162 closed Windows package parity as a retained
  non-claim. Sprint 163 must keep performance publication independent from
  package, install, ABI, and Windows package proof.

## Performance Surface Inventory

| Surface | Entry Point | Output / Evidence | Current Boundary |
| --- | --- | --- | --- |
| Benchmark compilation | `make bench-build`, `make tooling-build` | Benchmark and example binaries | Compile drift confidence only; not timing evidence. |
| Full benchmark run | `make bench` | Broad benchmark execution | Too broad for a narrow publication claim without later selection. |
| Fast runtime lane | `make bench-fast` | Short runtime benchmark subset | Maintainer confidence; not portable performance proof. |
| Canonical report | `make bench-canonical-report` | `build/bench-reports/canonical/*.csv`, `index.tsv`, `manifest.txt` | Threshold-free local snapshot of maintained benchmark rows. |
| Canonical script | `scripts/bench_canonical_report.sh` | Rows for `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`, and `bench_eigs_reuse` | Already records report label, UTC timestamp, commit, branch, platform, compiler, build mode, `OMP_NUM_THREADS`, artifact, relative path, and command. |
| Sentinel report | `make performance-sentinels` | `build/bench-reports/sentinels/sentinels.tsv`, `manifest.txt`, raw wall-check and CSV artifacts | Local sentinel bundle; only S5 wraps the thresholded `wall-check` gate. |
| Sentinel script | `scripts/performance_sentinels.sh` | S5 threshold rows plus S2 Cholesky CSC and S3 LDLT KKT threshold-free rows | Preserves `support_tier`, `claim_boundary`, backend request/selection/fallback, dense-kernel, panel-solver, and notes fields. |
| Wall-check gate | `make wall-check` | Local pass/fail regression result against `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt` | Narrow local timing gate, not a broad performance or superiority claim. |
| Large-matrix guardrails | `make large-matrix-guardrails` | Guardrail index and manifest artifacts | Structural guardrail evidence; adjacent unless later selected. |
| Normalized report index | `scripts/normalize_report_index.py` | Cross-family normalized index and freshness diagnostics | Navigation/freshness aid; not release proof. |
| Public benchmark docs | `README.md`, `benchmarks/README.md` | User-facing benchmark interpretation and commands | Explicitly warns benchmark rows are local/advisory and not portable guarantees. |
| CI surfaces | `.github/workflows/ci.yml`, platform workflows | Supplemental benchmark/report freshness and platform confidence jobs | CI environment evidence only; does not prove broad platform performance. |
| Package proof handoff | Sprint 162 artifacts and static-first checks | Windows package non-claim guard path | Must remain separate from performance publication. |

## Candidate Day 2 Selection Set

The best initial selection candidates are:

- Canonical maintained report rows from `make bench-canonical-report` because
  the existing script already emits methodology metadata and a stable artifact
  identity.
- Sentinel S5 rows from `make performance-sentinels` because they wrap the
  existing thresholded `wall-check` gate.
- Sentinel S2/S3 rows from `make performance-sentinels` as threshold-free
  backend-context rows, provided they stay separate from the S5 gate.

Deferred or adjacent surfaces:

- `make bench` is too broad for this sprint's first publication surface.
- `make bench-fast` is useful runtime confidence but weaker as a publication
  artifact because it is not currently the canonical report owner.
- `make large-matrix-guardrails` is structural guardrail evidence rather than
  performance publication unless explicitly selected later.
- Package and install validation remain out of scope for performance evidence.

## Non-Goals

Sprint 163 explicitly excludes:

- portable performance superiority;
- state-of-the-art performance claims;
- broad platform performance claims;
- package-manager, shared-library, dynamic ABI, runtime-loader, and Windows
  package parity claims;
- external-library parity;
- runtime-backend superiority;
- treating OpenMP or CI timing as portable speedup evidence;
- treating generated report indexes as release proof;
- reusing Sprint 162 package evidence as performance evidence.

## Assumptions

- A methodology-bound publication can be useful if it records exact generation
  commands, build/runtime context, row identity, and claim boundaries.
- Any source-controlled document produced by this sprint should summarize or
  index generated outputs, not hand-edit generated timing rows.
- Threshold-free rows should remain descriptive, while thresholded rows should
  name their baseline and threshold.
- Public documentation should preserve the existing distinction between
  benchmarks as measurement surfaces and tests as correctness owners.

## Completion Check

- Scope is tied to `docs/planning/EPIC_14/PROJECT_PLAN.md`.
- Current benchmark/report proof owners are identified.
- Sprint 163 performance work is separated from Sprint 162 package proof.
- Non-goals and assumptions are recorded before report selection work begins.
