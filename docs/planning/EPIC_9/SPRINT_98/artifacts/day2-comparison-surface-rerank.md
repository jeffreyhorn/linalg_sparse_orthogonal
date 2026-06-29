# Sprint 98 Day 2: Comparison-Surface Rerank

## Purpose

Day 2 ranks the highest-value next external correctness, runtime, and fill
comparison lanes from the live Sprint 98 baseline. The goal is to choose
bounded Sprint 98 candidates that can strengthen maintained evidence without
turning advisory benchmark or ecosystem signals into product claims.

## Ranking Criteria

Correctness candidates were ranked by:

- user-visible algorithm value
- availability of a trusted external or independent reference
- deterministic local reproducibility
- maintenance cost and proof-owner clarity
- CI suitability
- risk of widening public claims beyond maintained evidence

Runtime/fill candidates were ranked by:

- workload relevance
- existing benchmark or test ownership
- reporting clarity
- repeatability and artifact cost
- risk of misleading timing or superiority claims
- fit with the Sprint 90 comparison contract

## Current Claim-Bearing Baseline

The current maintained external correctness lane remains:

- `tests/test_chol_csc.c`
- `tests/chol_external_dense_reference.py`
- SuiteSparse SPD fixtures:
  - `tests/data/suitesparse/nos4.mtx`
  - `tests/data/suitesparse/bcsstk04.mtx`

That lane checks Cholesky CSC SPD solves against an external-process dense
reference helper. It is the current model for a maintained differential proof:
bounded fixtures, deterministic reference behavior, explicit tolerances, and
family-local interpretation.

The current maintained runtime/fill evidence remains bounded to:

- canonical maintained performance surfaces:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- threshold-free canonical report generation:
  - `make bench-canonical-report`
- bounded reorder/runtime slice:
  - `make bench-reorder-sprint86`
- PR-time supplemental runtime signal:
  - `make bench-fast`

These are calibration surfaces, not broad timing-governance or superiority
proof.

## External Correctness Ranking

| Rank | Candidate | Value | Reference availability | Cost/risk | Day 2 decision |
|---:|---|---|---|---|---|
| 1 | LDLT CSC external correctness expansion | High: adjacent direct-family solver and large existing proof owner | Medium-high: existing LDLT CSC tests already compare scalar/native/supernodal paths and solve residuals; external dense LDLT-style oracle still needs design | Medium: large file, indefinite tolerance semantics, pivot/permutation complexity | Select for Sprint 98 architecture design |
| 2 | Iterative solver external correctness comparison | High: user-visible CG/GMRES behavior | Medium: true residuals and Cholesky/LU cross-checks exist; external oracle would likely be residual/reference-solve based rather than a separate solver stack | Medium-high: convergence variability and preconditioner semantics can blur proof meaning | Keep as second Sprint 98 candidate if LDLT design proves too costly |
| 3 | Eigensolver/LOBPCG external correctness comparison | High: capability-signaling solver family | Medium: Ritz residuals and dense tridiagonal references exist; robust external eigenvalue oracle would need tighter design | High: tolerance, clusters, iteration caps, and runtime make CI fit harder | Defer from first implementation batch |
| 4 | QR external correctness comparison | Medium-high: important solver family, less directly adjacent to current maintained external lane | Medium-low: reconstruction/internal checks exist, but external reference path is not already factored out | High: numeric tolerance and fixture selection need new proof architecture | Residual queue |
| 5 | SVD external correctness comparison | Medium-high: public capability value | Medium-low: dense/off-path comparisons exist, but a maintained external oracle would be numerically sensitive | High: runtime, rank/tolerance semantics, and dense-reference cost | Residual queue |
| 6 | Ordering/fill correctness comparison | Medium: important structural behavior | Low as external correctness; better fit as runtime/fill and structural comparison | Medium: external orderings could imply ecosystem parity not currently owned | Use for runtime/fill lane, not correctness lane |

## Correctness Fix-Now Candidate

### Selected Candidate: LDLT CSC External Correctness Expansion

Rationale:

- It is adjacent to the current maintained Cholesky CSC external proof lane.
- `tests/test_ldlt_csc.c` already contains rich internal reference structure:
  scalar vs analysis-aware CSC paths, factor-state comparisons, solve residuals,
  KKT fixtures, random indefinite fixtures, and row-adjacency checks.
- The lane would strengthen direct-family assurance without jumping to a much
  broader solver family.
- It gives Day 3 a concrete architecture question:
  whether to adapt the external dense-reference helper pattern, create a new
  LDLT-specific external solve helper, or define a narrower differential oracle
  around deterministic indefinite fixtures.

Required Day 3 design questions:

- Which matrix class is acceptable for the first maintained LDLT external lane:
  SPD-as-LDLT, bounded KKT, or another deterministic indefinite fixture?
- What is the reference output:
  solution vector, residual strength, factor signature, pivot signature, or a
  combination?
- How should permutation and 2x2 pivot behavior be represented without making
  the external helper overly coupled to implementation internals?
- Which fixtures are small and deterministic enough for local and CI proof?
- Should the first lane live beside `tests/chol_external_dense_reference.py` or
  become a family-specific helper?

## Correctness Residual Queue

| Candidate | Reason deferred |
|---|---|
| Iterative solver external comparison | Strong value, but convergence and preconditioner semantics make it a better fallback or second correctness batch after LDLT architecture is explicit |
| Eigensolver/LOBPCG comparison | Valuable, but clustered spectra, iteration caps, and runtime make it too risky for the first Sprint 98 external expansion |
| QR comparison | Needs new reference architecture and fixture selection before implementation |
| SVD comparison | Needs tighter rank, tolerance, and runtime boundaries before it can be maintained cheaply |
| Broader SuiteSparse correctness corpus | Current fixtures are useful, but widening the corpus without ownership design would create maintenance cost before claim value |

## Runtime/Fill Ranking

| Rank | Candidate | Value | Existing owner | Cost/risk | Day 2 decision |
|---:|---|---|---|---|---|
| 1 | Reorder/fill comparison lane | High: directly tied to fill quality and touched workload calibration | `bench_reorder`, `bench_amd_qg`, ordering tests, `make bench-reorder-sprint86` | Medium: must stay bounded and avoid superiority language | Select for Sprint 98 architecture design |
| 2 | Canonical report metadata extension | Medium-high: improves branch-local comparison artifacts | `scripts/bench_canonical_report.sh`, `make bench-canonical-report` | Medium: easy to overexpand canonical surface | Keep as possible support batch, not first metric expansion |
| 3 | `bench_fillin` modernization | Medium: clear fill-centered benchmark | `benchmarks/bench_fillin.c`, `make bench-fast` | Medium: synthetic patterns may not support competitive claims | Pair with reorder/fill only if Day 3 keeps claim fence tight |
| 4 | Direct-family canonical runtime comparison | Medium: already canonical through `bench_refactor_csc` and `bench_chol_csc` | canonical report surface | Low-medium: already owned, but less aligned with new fill evidence | Preserve, do not widen first |
| 5 | Iterative/eigensolver runtime comparison | Medium-high capability value | `bench_iterative_reuse`, `bench_eigs_reuse` | Medium-high: easy to read as solver superiority | Preserve canonical output, defer widening |

## Runtime/Fill Fix-Now Candidate

### Selected Candidate: Reorder/Fill Comparison Lane

Rationale:

- The repo already has a bounded reorder runtime slice through
  `make bench-reorder-sprint86`.
- `bench_reorder` reports fill and runtime context fields that are already
  interpreted as branch-local evidence.
- `bench_amd_qg` is adjacent to the same structural/fill comparison story and
  already reports implementation and fill-quality comparison context.
- Ordering and fill tests already exercise structural behavior across
  SuiteSparse fixtures, giving Sprint 98 a proof-adjacent runtime/fill lane
  without inventing a broad benchmark program.
- The lane can stay explicitly threshold-free and calibration-oriented.

Required Day 3 design questions:

- Which exact workload slice should own Sprint 98 runtime/fill evidence:
  current `bench-reorder-sprint86`, a smaller named fill slice, or a report
  artifact that captures existing rows without widening execution cost?
- Which metrics are allowed:
  fill count, fill ratio, reorder time, factor-skipped runtime, or all of them?
- Where should the artifact live:
  benchmark stdout, generated report, planning artifact, or CI artifact?
- How should docs phrase the result so it remains calibration evidence rather
  than superiority proof?
- Should `bench_fillin` remain synthetic support evidence or be included in the
  selected lane?

## Runtime/Fill Residual Queue

| Candidate | Reason deferred |
|---|---|
| Canonical report surface expansion | Useful only if it does not make the canonical surface less stable or more expensive |
| `bench_fillin` standalone elevation | Synthetic and useful, but weaker than the existing reorder/fill workload for maintained comparison |
| Iterative/eigensolver runtime widening | Existing canonical surfaces are valuable; widening them first risks stronger solver performance claims than evidence supports |
| Full `make bench` comparison | Too expensive and too broad for a maintained Sprint 98 lane |
| Cross-platform timing comparison | Explicitly outside current platform proof strength |

## Fix-Now vs Residual Queue

Fix now through Day 3 architecture:

1. LDLT CSC external correctness expansion.
2. Reorder/fill runtime comparison lane.

Keep ready as fallback if Day 3 finds a blocker:

1. Iterative solver external correctness comparison.
2. Canonical report metadata/support alignment.

Residual for later sprints or later Sprint 98 only after explicit design:

1. Eigensolver/LOBPCG external comparison.
2. QR and SVD external comparison.
3. broader SuiteSparse correctness corpus.
4. broad benchmark-surface expansion.
5. cross-platform timing or coverage claim changes.

## Claim Fence

Sprint 98 may strengthen:

- bounded maintained external correctness evidence for one additional lane
- bounded runtime/fill calibration artifacts for a selected workload
- proof-owner and benchmark-owner clarity
- maintainer documentation that explains the widened assurance model

Sprint 98 must not claim:

- broad solver-family external proof
- universal speed leadership
- cross-platform timing parity
- package/platform parity beyond Sprint 97's static-first contract
- coverage percentage as a competitive product claim
- exploratory benchmark output as maintained product truth

## Day 2 Result

Sprint 98 now has one authoritative comparison-surface ranking. Day 3 should
design a bounded proof/comparison architecture around LDLT CSC external
correctness and reorder/fill runtime evidence, while preserving the Sprint 90
claim fence and keeping higher-risk solver-family lanes in the residual queue.
