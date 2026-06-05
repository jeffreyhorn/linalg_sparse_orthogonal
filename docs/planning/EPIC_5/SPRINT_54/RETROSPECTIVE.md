# Sprint 54 Retrospective

**Sprint:** 54 — Public Repeated-Run Solver Lifecycle Completion  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 54 baseline and scope captured from the Sprint 53 validated post-CSC-follow-through package
- [x] reviewed validation/truthfulness baseline rechecked before solver-lifecycle completion work
- [x] public repeated-run solver lifecycle audit completed against the live repo
- [x] steady-state repeated-run solver support boundary decided explicitly
- [x] bounded iterative public-handle expansion landed for MINRES
- [x] supported iterative-handle contract tightened across the final iterative support set
- [x] eigensolver lifecycle/proof tightening landed across the final supported backend set
- [x] public reuse benchmark audit completed
- [x] iterative/eigensolver reuse benchmarks aligned to the final public support set
- [x] high-signal README/example/tutorial adoption and proof batches completed
- [x] post-landing compatibility audit completed
- [x] full validation sweep completed from the landed repeated-run solver state
- [x] Sprint 54 closeout and Sprint 55 handoff completed from the validated baseline

## What Went Well

1. **Sprint 54 finished the repeated-run solver support boundary instead of drifting into handle sprawl.**
   The sprint made the real steady-state surface explicit:
   - iterative public handles:
     - `CG`
     - `GMRES`
     - `MINRES`
   - eigensolver public handles:
     - grow-m Lanczos
     - thick-restart Lanczos
     - explicit `LOBPCG`
   - explicit exclusions:
     - `BiCGSTAB`
     - block iterative workflows
   That kept the sprint architectural outcome clear instead of “handles everywhere eventually.”

2. **MINRES was added in a bounded, contract-consistent way.**
   Sprint 54 did not bolt MINRES onto a special-case public path. It routed MINRES onto the same public `sparse_iter_handle_t` lifecycle story already used for `CG` and `GMRES`, with direct proof for:
   - explicit prepare + reuse
   - zero-init/on-demand growth
   - same-handle growth across later larger solves

3. **The eigensolver handle story is now fully explicit and directly proved.**
   Before Sprint 54, the public handle surface was real, but some backend support was still more implicit than explicit. The sprint fixed that by:
   - tightening the public contract wording
   - adding direct proof for grow-m Lanczos
   - adding direct proof for thick-restart Lanczos
   - adding direct proof for explicit `LOBPCG`
   That leaves the eigensolver repeated-run surface much easier to trust and maintain.

4. **The reuse benchmarks now match the real supported public surface.**
   Sprint 54 aligned the benchmark drivers to the actual supported handle set:
   - `bench_iterative_reuse` now covers `CG`, `GMRES`, and `MINRES`
   - `bench_eigs_reuse` now covers grow-m, thick-restart, and explicit `LOBPCG`
   That removed a real proof/documentation drift where the public support set had outgrown the benchmark evidence.

5. **The caller-facing docs are materially more honest now.**
   Sprint 54 aligned the top-level README, examples README, tutorial, and benchmark README so they now say the same thing as the code:
   - supported repeated-run families are named explicitly
   - excluded families are named explicitly
   - shipped examples remain intentionally one-shot-first
   This is a real usability and maintainability improvement because it reduces false expectations.

6. **The sprint preserved the existing compatibility fences.**
   Sprint 54 did not reopen broad solver redesign:
   - one-shot solver APIs remain first-class
   - repeated-run handles remain bounded opt-in paths
   - no public workspace-layout exposure was introduced
   - no broad iterative/eigensolver family expansion happened outside the decided fence
   That kept the work coherent with the earlier Epic 4 and Epic 5 lifecycle decisions.

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
   - reviewed CMake total time `144.25 sec`

## What Didn't Go Well

1. **Not every measured reuse path shows a strong speedup.**
   Sprint 54 correctly framed the reuse benchmarks as local evidence rather than blanket promises. That honesty matters because some supported paths stayed near parity or slightly below:
   - `gmres-unsym-220` = `0.85x`
   - `thick-bcsstk14-k5` = `0.99x`
   The sprint improved support clarity, not universal performance dominance.

2. **The examples remain intentionally one-shot-first.**
   That was the right scope choice, but it also means the public repeated-run story still depends more on headers, README wording, tests, and benchmarks than on multiple shipped end-to-end handle demos.

3. **Excluded families are now clearer, but still remain a visible capability boundary.**
   Sprint 54 made the exclusion of `BiCGSTAB` and block iterative workflows explicit and honest. That is better than ambiguity, but it still means the public repeated-run support surface is intentionally not universal.

4. **The sprint was more about completion and alignment than large new functionality.**
   That was appropriate, but it also means a meaningful share of the value came from boundary-setting, proof tightening, and benchmark/doc alignment rather than from many large new algorithmic landings.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 54 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `144.25 sec` |

### Sprint 54 artifact package

| Metric | Sprint 54 close state |
|---|---:|
| total artifact files under `SPRINT_54/artifacts/` | `15` |
| decision/implementation/proof artifacts (Days 4-9) | `6` |
| adoption/audit/validation/closeout artifacts (Days 10-14) | `5` |

### Repeated-run solver outputs

| Metric | Sprint 54 close state |
|---|---:|
| touched `*.c` / `*.h` files in the landed repeated-run solver package | `7` |
| caller-facing docs surfaces updated | `4` |
| focused direct proof homes touched | `2` |
| targeted Sprint 54 follow-on commands rerun in Day 13 | `9` |

Notes:

- touched `*.c` / `*.h` files in the landed repeated-run solver package:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `src/sparse_iterative.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- caller-facing docs surfaces updated:
  - `README.md`
  - `examples/README.md`
  - `docs/tutorial.md`
  - `benchmarks/README.md`
- focused direct proof homes touched:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
- targeted Sprint 54 follow-on commands rerun in Day 13:
  - `./build/test_iterative`
  - `./build/test_minres`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

## Residual Deferred Debt

Sprint 54 was explicitly about bounded public repeated-run solver lifecycle
completion. The main open work it intentionally hands forward is:

- larger tutorial/example modernization if a later sprint wants explicit
  repeated-run teaching code beyond the current bounded docs updates
- any later public-handle expansion beyond the bounded Sprint 54 support set
- broader benchmark or caller-surface evolution built on the now-explicit
  repeated-run solver fence

Not carried forward as unresolved Sprint 54 debt:

- missing steady-state repeated-run solver support-boundary decision
- missing public MINRES handle support
- missing explicit direct proof for the final supported eigensolver backend set
- missing aligned public reuse benchmarks for the final supported solver set
- missing high-signal README/example/tutorial support-boundary reconciliation
- missing post-landing compatibility audit
- missing full validated closeout baseline

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day4-solver-surface-decision-batch.md](./artifacts/day4-solver-surface-decision-batch.md)
- [day5-iterative-handle-expansion-batch1.md](./artifacts/day5-iterative-handle-expansion-batch1.md)
- [day6-iterative-contract-tightening-batch.md](./artifacts/day6-iterative-contract-tightening-batch.md)
- [day7-eigensolver-lifecycle-tightening-batch.md](./artifacts/day7-eigensolver-lifecycle-tightening-batch.md)
- [day8-public-reuse-benchmark-alignment-audit.md](./artifacts/day8-public-reuse-benchmark-alignment-audit.md)
- [day9-public-reuse-benchmark-alignment-batch.md](./artifacts/day9-public-reuse-benchmark-alignment-batch.md)
- [day10-regression-and-example-adoption-batch1.md](./artifacts/day10-regression-and-example-adoption-batch1.md)
- [day11-regression-and-example-adoption-batch2.md](./artifacts/day11-regression-and-example-adoption-batch2.md)
- [day12-post-landing-compatibility-audit.md](./artifacts/day12-post-landing-compatibility-audit.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 54 achieved its goal:

- the repo now has an explicit steady-state public repeated-run solver support boundary
- `MINRES` is part of the real public iterative handle surface alongside `CG` and `GMRES`
- the supported eigensolver handle backend set is now fully explicit and directly proved
- the public reuse benchmarks now match the real supported solver families
- the highest-signal docs now say the same thing as the code
- the sprint preserved one-shot-first compatibility and honest bounded exclusions
- the sprint closed from a fully validated reviewed baseline with exact preserved truthfulness anchors

Sprint 55 can now build on a validated, explicit repeated-run solver support
surface rather than needing to re-decide what is supported, whether the public
benchmarks match the code, or whether the caller-facing docs are overstating
solver lifecycle symmetry.
