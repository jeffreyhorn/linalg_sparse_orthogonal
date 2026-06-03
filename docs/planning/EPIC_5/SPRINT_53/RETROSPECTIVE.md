# Sprint 53 Retrospective

**Sprint:** 53 — CSC Direct-Solver Completion & Dispatch Follow-Through  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 53 baseline and scope captured from the Sprint 52 validated Phase 2 package
- [x] reviewed validation/truthfulness baseline rechecked before CSC follow-through work
- [x] shared analysis-aware indefinite LDL^T path audit completed against the live repo
- [x] first bounded analysis-aware LDL^T CSC integration batch landed
- [x] second bounded analysis-aware LDL^T CSC integration batch landed
- [x] first LDL^T dispatch tightening batch landed
- [x] second LDL^T dispatch tightening batch landed
- [x] indefinite factor-many benchmark proof landed from the live repeated-run path
- [x] Cholesky / LDL^T dispatch reconciliation audit completed
- [x] high-signal README dispatch reconciliation batch completed
- [x] focused indefinite repeated-run regression expansion completed
- [x] post-landing compatibility audit completed
- [x] full validation sweep completed from the landed CSC follow-through state
- [x] Sprint 53 closeout and Sprint 54 handoff completed from the validated baseline

## What Went Well

1. **Sprint 53 stayed inside the Sprint 50-52 compatibility fence while still landing meaningful CSC depth.**
   The sprint did not reopen the direct-solver public model:
   - one-shot LU / Cholesky / LDL^T remain first-class
   - repeated direct runs remain analysis/factors-centric
   - no generic public direct handle
   - no raw CSC/native storage exposure
   That kept the work bounded to real CSC completion and dispatch follow-through.

2. **The shared indefinite repeated-run CSC path is materially more coherent now.**
   Sprint 53 pulled the highest-value LDL^T CSC orchestration seams into shared helpers for:
   - resolved-analysis preparation
   - CSC completion
   - supernodal-attempt / scalar-fallback ownership
   That reduced duplication between the one-shot CSC dispatch path and the shared repeated-run path without pretending the whole LDL^T family is now trivial.

3. **The LDL^T dispatch story is both tighter and more truthful.**
   The sprint improved backend selection and reporting by:
   - centralizing backend selection
   - making forced CSC telemetry report the actual selected numeric path
   - rejecting invalid shared-helper configuration directly instead of masking it behind unrelated fallback
   That is a real maintainability and correctness improvement in one of the repo’s more subtle CSC paths.

4. **Sprint 53 added real indefinite factor-many benchmark evidence.**
   Before this sprint, CSC repeated-run proof was stronger on SPD paths than on indefinite ones. Day 8 fixed that by extending `bench_refactor_csc` with:
   - a bounded LDL^T KKT repeated-run mode
   - measured public repeated-run vs direct CSC completion comparison
   That gave the sprint a real indefinite performance/proof surface instead of just routing tests.

5. **A real repeated-run LDL^T bug surfaced and was fixed during benchmark work.**
   The first indefinite benchmark run exposed a reordered-LDLT permutation bug in the shared solve path. Catching and fixing that during Sprint 53 is a strong sign that the sprint’s benchmark/proof work was exercising meaningful behavior rather than just shallow happy paths.

6. **The CSC docs now better match the implementation.**
   Sprint 53 clarified the top-level story:
   - Cholesky CSC dispatch is the simpler family-local case
   - LDL^T CSC dispatch is the layered scalar-prepass plus CSC-pipeline case
   - `bench_refactor_csc --indefinite-kkt` is the bounded indefinite proof surface
   That removed stale pre-Sprint-53 wording without forcing fake symmetry between the two families.

7. **The sprint closed from a real validated baseline.**
   Day 13 passed:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   and preserved the truthfulness anchors:
   - reviewed CMake parity `53`
   - Makefile/CMake parity `53 vs 53`
   - reviewed CMake `ctest` `53 / 53`
   - reviewed CMake total time `124.22 sec`

## What Didn't Go Well

1. **LDL^T CSC is still the most layered direct-solver dispatch story.**
   Sprint 53 improved it substantially, but it is still inherently more complex than Cholesky because the scalar BK pre-pass remains authoritative and can still force resolved-analysis changes.

2. **The sprint proved bounded CSC repeated-run behavior, not universal simplification.**
   The LDL^T CSC path is more coherent, but it is not suddenly a one-shape-fits-all repeated-run path. That remains an honest boundary rather than a sprint failure, but it is still a visible source of complexity.

3. **The structure-compatibility boundary is still intentionally cheap.**
   Sprint 53 benefited from Sprint 52’s `nnz`-drift guard and preserved that fence. It did not become a full structural-pattern verifier, so same-pattern repeated-run claims still need to stay carefully bounded.

4. **The transient reviewed-CMake false alarm consumed validation time.**
   Day 13 briefly hit a direct-`ctest` missing-executable report for `test_reorder_amd_qg`. It did not reproduce and the validated state was green, but it still imposed extra investigation overhead during the final sweep.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 53 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `124.22 sec` |

### Sprint 53 artifact package

| Metric | Sprint 53 close state |
|---|---:|
| total artifact files under `SPRINT_53/artifacts/` | `15` |
| integration/dispatch/benchmark artifacts (Days 4-8) | `5` |
| reconciliation/audit/validation/closeout artifacts (Days 9-14) | `6` |

### CSC follow-through outputs

| Metric | Sprint 53 close state |
|---|---:|
| touched `*.c` / `*.h` files in the landed CSC package | `9` |
| touched docs/benchmark surfaces in the landed CSC package | `2` |
| focused CSC/regression proof homes touched | `4` |
| targeted Sprint 53 follow-on commands rerun in Day 13 | `9` |

Notes:

- touched `*.c` / `*.h` files in the landed CSC package:
  - `benchmarks/bench_refactor_csc.c`
  - `include/sparse_ldlt.h`
  - `src/sparse_analysis.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_sprint20_integration.c`
- touched docs/benchmark surfaces in the landed CSC package:
  - `README.md`
  - `benchmarks/README.md`
- focused CSC/regression proof homes touched:
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_sprint20_integration.c`
- targeted Sprint 53 follow-on commands rerun in Day 13:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/example_analysis`
  - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_refactor_csc --indefinite-kkt --repeat 1`

## Residual Deferred Debt

Sprint 53 was explicitly about bounded CSC completion/follow-through on top of
the existing public direct lifecycle. The main open work it intentionally hands
forward is:

- any later CSC/dispatch depth beyond the bounded Sprint 53 completion seams
- any later family-local cleanup where LU or other solver-family differences should remain special-case
- broader benchmark or caller-surface evolution beyond the bounded Sprint 53 proof surfaces
- any future stronger structural-pattern validation beyond the current cheap guard

Not carried forward as unresolved Sprint 53 debt:

- missing shared analysis-aware LDL^T CSC completion follow-through
- missing tighter LDL^T dispatch ownership
- missing indefinite factor-many benchmark proof
- missing top-level CSC dispatch reconciliation wording
- missing focused indefinite repeated-run regression expansion
- missing post-landing compatibility audit
- missing full validated closeout baseline

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day4-analysis-aware-ldlt-integration-batch1.md](./artifacts/day4-analysis-aware-ldlt-integration-batch1.md)
- [day5-analysis-aware-ldlt-integration-batch2.md](./artifacts/day5-analysis-aware-ldlt-integration-batch2.md)
- [day6-ldlt-dispatch-batch1.md](./artifacts/day6-ldlt-dispatch-batch1.md)
- [day7-ldlt-dispatch-batch2.md](./artifacts/day7-ldlt-dispatch-batch2.md)
- [day8-indefinite-factor-many-benchmark-proof.md](./artifacts/day8-indefinite-factor-many-benchmark-proof.md)
- [day10-dispatch-reconciliation-batch.md](./artifacts/day10-dispatch-reconciliation-batch.md)
- [day11-regression-expansion-batch.md](./artifacts/day11-regression-expansion-batch.md)
- [day12-post-landing-compatibility-audit.md](./artifacts/day12-post-landing-compatibility-audit.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 53 achieved its goal:

- the repo now has a more coherent analysis-aware LDL^T CSC repeated-run path
- LDL^T CSC dispatch ownership and telemetry are tighter and more truthful
- the sprint added real indefinite CSC factor-many proof rather than only SPD repeated-run evidence
- the top-level CSC contract wording now better matches the implementation
- the focused indefinite repeated-run regression floor is stronger
- the sprint closed from a fully validated reviewed baseline with exact preserved truthfulness anchors

Sprint 54 can now build on a validated CSC follow-through package rather than
needing to re-prove whether the indefinite CSC repeated-run path is real,
whether LDL^T dispatch ownership is coherent, or whether the README/benchmark
story matches the measured implementation state.
