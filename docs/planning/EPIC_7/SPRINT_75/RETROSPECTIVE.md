# Sprint 75 Retrospective

**Sprint:** 75 — Performance Backend Architecture Phase 2  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 75 scope, backend hotspot map, and validation baseline were fixed
      before implementation work began
- [x] the strongest live backend/performance seams were re-ranked from the
      repo instead of treated as one generic performance bucket
- [x] the first landing stayed bounded to the CSC supernodal Cholesky
      dense-kernel/runtime lane and did not widen into a broad backend
      abstraction rewrite
- [x] the dense-kernel descriptor now owns one clearer shipped batched
      `solve_panel` seam
- [x] the CSC supernodal panel path now consumes that batched panel-solve seam
      directly on the touched backend-aware route
- [x] the public CSC runtime contract now exposes one bounded truthful wrapper
      phase through `cholesky_factor_csc` with `4` orchestration checkpoints
- [x] the maintained benchmark surface now makes the new panel-solver seam
      directly measurable through `csc_supernodal_panel_solver`
- [x] focused proof remained in the right owners:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_chol_csc.c`
- [x] maintained public/policy wording now states the narrower Sprint 75
      runtime-vs-benchmark ownership split directly
- [x] proof-owner alignment was closed without redundant regression work
- [x] the full Sprint 75 branch passed the standard code-day gate, the
      strongest reviewed baseline, and the focused Cholesky/runtime/example/
      benchmark/install follow-ons
- [x] Sprint 75 closed with one explicit second backend-aware package and a
      ranked Sprint 76 carry-forward queue

## What Went Well

1. **Sprint 75 landed real backend-aware implementation work instead of only reclassifying hotspots.**
   The branch made substantive changes in:
   - `src/sparse_dense.c`
   - `src/sparse_chol_csc_supernodal.c`
   - `src/sparse_chol_csc_internal.h`
   - `src/sparse_cholesky.c`
   - `include/sparse_cholesky.h`
   and tied that work to focused proof in:
   - `tests/test_chol_csc.c`
   - `tests/test_integration.c`
   - `benchmarks/bench_chol_csc.c`

2. **The backend lane stayed properly bounded.**
   Sprint 75 did not collapse into:
   - a broad backend abstraction-layer rewrite
   - a fake optional external-backend maturity story
   - shared-library or plugin-style backend claims
   - benchmark-threshold pass/fail governance
   - widened reviewed-platform or install-validation claims
   That kept the work aligned with the Sprint 70 and Sprint 64 truthfulness
   fence.

3. **The dense-kernel seam is materially clearer now.**
   Day 7 made the shipped dense-kernel descriptor the clearer owner of the
   batched `solve_panel` seam, and the touched supernodal panel path now
   consumes that seam directly rather than row-by-row single-RHS looping.

4. **The public CSC runtime contract is more truthful without overpromising parity.**
   Day 10 did the useful narrow thing:
   - established one wrapper-owned `cholesky_factor_csc` public phase
   - fixed that phase to `4` bounded orchestration checkpoints
   - proved cancel-before-writeback preservation of the original caller matrix
     shell
   - avoided fake per-column CSC callback parity

5. **The benchmark surface now measures the new seam directly.**
   Day 11 added the stable `csc_supernodal_panel_solver` field, which made the
   Day 7 kernel landing reviewable from `bench_chol_csc` without turning
   benchmarks into the owner of public runtime truth.

6. **The validated close state is strong.**
   Sprint 75 ended with:
   - `make format` passed
   - `make lint` passed
   - `make test` passed
   - `make quality-review-full` passed
   - reviewed CMake parity still exact at `53`
   - Makefile/CMake parity still `53 vs 53`
   - reviewed CMake `ctest` still `53 / 53`
   - focused Cholesky/runtime proof owners revalidated explicitly
   - representative examples, benchmarks, and install/package regressions
     still clean

## What Didn't Go Well

1. **Sprint 75 deepens one backend lane; it does not finish backend maturity.**
   The CSC supernodal Cholesky lane is clearer now, but the branch does not
   yet deliver:
   - broader backend-family parity across eigs, QR, or SVD
   - a generalized backend framework
   - a stronger product claim around optional or external backends

2. **Runtime parity remains intentionally narrow.**
   That is the correct Sprint 75 outcome, but it means Epic 7 still carries
   real deferred work around backend/runtime observability in:
   - eigensolver backend dispatch and runtime policy
   - later QR and SVD backend-aware follow-through

3. **The benchmark proof is clearer than the full governance story.**
   Sprint 75 made the panel-solver seam measurable, but it did not solve:
   - longitudinal benchmark reporting
   - stronger canonical comparison policy
   - broader benchmark-governance cleanup

4. **Runtime asymmetry in the reviewed suite remains visible.**
   The full reviewed path passed, but `test_reorder_nd` still dominated the
   reviewed CMake time even though Sprint 75 itself was not a reorder sprint.
   That remains operational friction for later proof-heavy work.

5. **The branch depended on disciplined non-moves.**
   Sprint 75’s success required not widening backend claims, not treating the
   benchmark field as a runtime contract, and not reopening broader packaging
   or capability scope. That discipline held, but later backend work still
   needs to preserve it.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 75 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `346.76 sec` |
| reviewed `test_reorder_nd` time | `234.86 sec` |
| install regression | `11 / 11` |
| CMake install regression | `13 / 13` |

### Sprint 75 artifact package

| Metric | Sprint 75 close state |
|---|---:|
| total artifact files under `SPRINT_75/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/landing artifacts | `6` |
| review/closeout artifacts | `3` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-performance-backend-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-rerun-recheck.md`
  - `day3-performance-hotspot-reaudit.md`
  - `day4-first-backend-boundary.md`
  - `day8-post-landing-audit-and-rerank.md`
- design/landing artifacts:
  - `day5-backend-policy-design.md`
  - `day6-design-freeze-and-proof-map.md`
  - `day7-kernel-integration-batch2.md`
  - `day9-callback-runtime-policy-design.md`
  - `day10-callback-runtime-follow-through-batch.md`
  - `day11-benchmark-proof-refresh.md`
- review/closeout artifacts:
  - `day12-regression-and-fallback-proof-alignment.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed backend/performance package

| Metric | Sprint 75 close state |
|---|---:|
| public headers touched in landed package | `1` |
| implementation `.c` files touched in landed package | `3` |
| internal helper headers touched | `1` |
| focused proof-owner tests touched | `2` |
| maintained benchmark sources touched | `1` |
| maintained public/policy docs touched | `3` |

Notes:

- public headers touched:
  - `include/sparse_cholesky.h`
- implementation `.c` files touched:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_cholesky.c`
- internal helper headers touched:
  - `src/sparse_chol_csc_internal.h`
- focused proof-owner tests touched:
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
- maintained benchmark sources touched:
  - `benchmarks/bench_chol_csc.c`
- maintained public/policy docs touched:
  - `README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`

## Residual Deferred Debt

Sprint 75 deliberately stopped after the second bounded backend phase. The
main open work it intentionally hands forward is:

- benchmark governance and longitudinal reporting from the stronger measurable
  backend surface
- eigensolver backend/runtime parity as the strongest remaining backend-aware
  second lane
- QR and SVD backend-aware follow-through only where a bounded proof-backed
  seam justifies the cost
- later packaging, ABI, or platform convergence only where maintained
  evidence supports a stronger claim

Still consciously constrained rather than silently “solved”:

- no broad backend abstraction layer
- no optional external-backend maturity claim
- no shared-library or plugin-style backend story
- no benchmark-threshold pass/fail portability story
- no widened reviewed-platform or install-validation claim

Not carried forward as unresolved Sprint 75 debt:

- the backend hotspot rerank
- the Day 7 kernel integration batch
- the Day 10 runtime follow-through batch
- the Day 11 benchmark proof refresh
- the proof-owner alignment pass
- the full Day 13 validation sweep
- the Day 14 closeout and ranked Sprint 76 handoff queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-and-performance-backend-baseline.md](./artifacts/day1-scope-and-performance-backend-baseline.md)
- [day1-authoritative-inputs.txt](./artifacts/day1-authoritative-inputs.txt)
- [day2-validation-baseline-and-rerun-recheck.md](./artifacts/day2-validation-baseline-and-rerun-recheck.md)
- [day3-performance-hotspot-reaudit.md](./artifacts/day3-performance-hotspot-reaudit.md)
- [day4-first-backend-boundary.md](./artifacts/day4-first-backend-boundary.md)
- [day5-backend-policy-design.md](./artifacts/day5-backend-policy-design.md)
- [day6-design-freeze-and-proof-map.md](./artifacts/day6-design-freeze-and-proof-map.md)
- [day7-kernel-integration-batch2.md](./artifacts/day7-kernel-integration-batch2.md)
- [day8-post-landing-audit-and-rerank.md](./artifacts/day8-post-landing-audit-and-rerank.md)
- [day9-callback-runtime-policy-design.md](./artifacts/day9-callback-runtime-policy-design.md)
- [day10-callback-runtime-follow-through-batch.md](./artifacts/day10-callback-runtime-follow-through-batch.md)
- [day11-benchmark-proof-refresh.md](./artifacts/day11-benchmark-proof-refresh.md)
- [day12-regression-and-fallback-proof-alignment.md](./artifacts/day12-regression-and-fallback-proof-alignment.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 75 accomplished the bounded second backend-aware landing it was supposed
to accomplish.

It did not pretend to solve “performance” in the abstract. It made one real
CSC supernodal Cholesky backend seam clearer, made the public CSC runtime
story more truthful, made the new panel-solver seam benchmark-visible, proved
the right safety boundaries in the right owners, and closed from a fully
validated reviewed baseline.

That leaves Sprint 76 in a stronger position: it can start from a real
measurable backend-aware landing rather than from a speculative architecture
goal or a generic performance backlog.
