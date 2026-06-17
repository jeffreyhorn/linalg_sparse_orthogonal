# Sprint 76 Day 3 Artifact: Benchmark Governance Re-audit

Date: 2026-06-17
Branch: sprint-76

## Purpose

Re-rank the live benchmark-governance surface by actual reporting value,
proof leverage, and maintenance clarity so Sprint 76 starts from the strongest
bounded governance seam rather than from a generic benchmarking wishlist.

## Inputs Reviewed

- `README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- `Makefile`
- `scripts/bench_canonical_report.sh`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/bench_reorder.c`
- `benchmarks/bench_amd_qg.c`

## Main Result

Sprint 76's benchmark-governance pressure is no longer a generic
"more reports, more benchmarks, more thresholds" problem. It is now one
ranked contradiction map:

- strongest first target:
  - canonical reporting workflow and longitudinal-comparison schema
- strongest second target:
  - benchmark-local role and interpretation surface
- strongest third target:
  - authoritative threshold and category policy surface
- strongest support-surface contradiction:
  - compact front-door benchmark summary
- strongest adjacent but not first-batch lane:
  - regression-sensitive runtime surfaces around `bench-fast`,
    `wall-check`, `bench_reorder`, and `bench_amd_qg`

## Ranked Findings

### 1. Canonical reporting workflow and schema are the strongest first landing

The strongest current governance seam is concentrated in:

- `scripts/bench_canonical_report.sh`
- `Makefile`
- the four canonical maintained emitters the script drives

This lane ranks first because the current repository already has the right
bounded benchmark face:

- one explicit canonical maintained surface
- one threshold-free local or CI-friendly report command
- one stable four-CSV bundle
- one manifest with exact command mapping

But the strongest remaining gap is also clear:

- longitudinal comparison still depends on a very small manifest plus manual
  interpretation of the emitted CSV set
- the reporting workflow is truthful, but not yet the strongest governed
  comparison surface it could be
- the highest-value Sprint 76 leverage is therefore artifact schema,
  metadata, and workflow governance rather than new numeric workloads

### 2. `benchmarks/README.md` is the strongest second contradiction center

The benchmark-local README is the densest current interpretation surface for:

- canonical maintained proof
- regression-sensitive runtime lanes
- exploratory comparison lanes
- `bench-fast`
- `wall-check`
- `bench-canonical-report`

It ranks second because it already contains most of the right category split,
but it is also where the user-facing role map is easiest to blur:

- runtime surfaces can start to read more canonical than they really are
- exploratory breadth can start to look claim-bearing if the wording drifts
- the canonical report workflow can be misread as a stronger comparison or
  threshold surface than intended

So it is the strongest interpretation surface after the reporting workflow
itself.

### 3. `docs/maintainer_guide.md` is the strongest third lane

The maintainer guide already owns the authoritative benchmark-governance
policy:

- canonical vs runtime vs exploratory category split
- stable canonical output ownership
- threshold-free `bench-canonical-report` reading
- explicit warnings against pseudo-governance or portability overclaim

It ranks third rather than first because:

- it is already policy-coherent
- it is support-first rather than the best first landing center
- the stronger immediate Sprint 76 gap is making the report surface and the
  benchmark-local interpretation surface easier to compare and harder to
  misread

### 4. `README.md` is support-only, not the first design center

The top-level README still matters because it owns the compact benchmark story:

- Linux/macOS/Windows reviewed-baseline reading
- `bench-fast` as bounded PR-time runtime signal
- `bench-canonical-report` as threshold-free reporting
- compact canonical benchmark summary

But it is not the strongest first landing because it is already intentionally
compact and deliberately avoids owning the full benchmark-governance contract.

### 5. The runtime lane is important, but not the first batch center

The current regression-sensitive runtime lane remains real:

- `bench-fast`
- `wall-check`
- `bench_reorder --skip-factor`
- `bench_amd_qg`

This lane matters because it is where narrow timing thresholds are already
justified and where local performance regressions can be caught.

It does not rank first because the strongest current Sprint 76 problem is not
"invent a stronger threshold policy." It is:

- keep the canonical surface small
- make longitudinal reports easier to compare
- keep the runtime lane bounded
- avoid letting the exploratory lane blur into the maintained claim-bearing
  surface

## Day 4 Implication

The next boundary pass should treat Sprint 76's first landing as:

- required first center:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
- strongest second center:
  - `benchmarks/README.md`
- strongest third/support center:
  - `docs/maintainer_guide.md`
- support-only front-door follow-through:
  - `README.md`

## Exit State

Sprint 76 now has one explicit Day 3 governance rerank:

- start from the canonical reporting workflow and schema
- treat benchmark-local interpretation as the strongest second seam
- keep maintainer policy as authoritative support rather than the first edit
  center
- keep runtime-threshold surfaces bounded rather than widening them into the
  canonical proof contract
