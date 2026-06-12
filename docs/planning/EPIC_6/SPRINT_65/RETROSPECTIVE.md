# Sprint 65 Retrospective

**Sprint:** 65 — Performance Governance, Benchmark Consolidation & Solver Efficiency Follow-Through  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 65 baseline and scope captured from the Sprint 64 validated Epic 6 state
- [x] reviewed validation/truthfulness baseline rechecked before benchmark-governance or efficiency code landed
- [x] benchmark binaries reduced to a concrete role map instead of one generic performance-governance backlog
- [x] canonical maintained benchmark surface fixed before output normalization landed
- [x] normalized output and taxonomy contract designed before benchmark binary edits began
- [x] exact canonical-surface implementation fence fixed before benchmark normalization landed
- [x] first canonical direct benchmark normalization batch landed on `bench_refactor_csc` and `bench_chol_csc`
- [x] remaining canonical normalization landed on `bench_iterative_reuse` and `bench_eigs_reuse`
- [x] bounded direct repeated-run CSC/Cholesky solver-efficiency follow-through landed on the strongest measured seam
- [x] threshold-free local/CI-friendly canonical report surface landed without widening into timing-threshold CI gates
- [x] docs, examples, benchmarks, and maintainer guidance updated to match the landed Sprint 65 governance story
- [x] full validation sweep completed from the final landed Sprint 65 tree
- [x] Sprint 65 closeout and handoff completed from the validated baseline

## What Went Well

1. **Sprint 65 turned “performance governance” into a real maintained product surface instead of another benchmark-discussion sprint.**
   The sprint kept the implementation center of gravity on:
   - benchmark-role audit
   - canonical-surface selection
   - output normalization
   - bounded reporting
   - one efficiency follow-through seam
   It did not widen into:
   - broad timing-threshold CI policy
   - backend-architecture churn
   - packaging/platform work
   - generic benchmark rewrites across every binary
   That scope discipline is why Sprint 65 closed as one coherent
   performance-governance sprint instead of as a loose mix of benchmark notes
   and one-off measurements.

2. **The repo now has one explicit canonical maintained performance surface.**
   Before Sprint 65, the benchmark story was still split across:
   - direct CSV-ish proof surfaces
   - iterative human-readable summaries
   - eigensolver human-readable summaries
   - broader exploratory and historical harnesses
   Sprint 65 fixed that by narrowing the canonical maintained surface to:
   - `bench_refactor_csc`
   - `bench_chol_csc`
   - `bench_iterative_reuse`
   - `bench_eigs_reuse`
   That is a meaningful productization outcome because the repo now has a
   small benchmark set it can honestly maintain and point to.

3. **The benchmark output story is much more machine-readable and reviewable than it was at sprint start.**
   Sprint 65 normalized the canonical outputs around stable identity and
   category fields:
   - `benchmark`
   - `category`
   - `matrix`
   - `scenario`
   and then kept the path/backend/timing/residual fields honest per benchmark.
   The iterative and eigensolver reuse surfaces are no longer prose-only
   summaries. They now emit stable retained rows that line up with the direct
   CSC proof lane.

4. **`make bench-canonical-report` is the right kind of governance landing.**
   Sprint 65 did not create a fake timing-threshold gate. It added a bounded,
   threshold-free reporting surface:
   - `make bench-canonical-report`
   which writes stable canonical CSV snapshots and a manifest under:
   - `build/bench-reports/canonical/`
   That is a much better fit for the repo’s current truthfulness contract than
   pretending it is ready for strict cross-run timing thresholds everywhere.

5. **The docs/example/benchmark ownership split is clearer at sprint close than it was at sprint start.**
   Sprint 65 finished with one coherent public story:
   - examples teach workflow and ownership
   - benchmarks prove retained workflow/performance behavior
   - `make bench-canonical-report` captures threshold-free canonical snapshots
   That wording now lines up across:
   - `README.md`
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
   - `docs/maintainer_guide.md`
   This is one of the strongest productization outcomes in the sprint because
   the user-facing adoption path and the maintained proof path are no longer
   intermixed.

6. **The bounded efficiency landing hit the right seam.**
   Sprint 65 did not diffuse effort across iterative, eigensolver, LDL^T, and
   direct code all at once. It kept the first efficiency follow-through on:
   - the direct repeated-run CSC/Cholesky lane
   - specifically `src/sparse_chol_csc_supernodal.c`
   The landed monotonic `row_map` seek cleanup is narrow, defensible, and tied
   to the strongest measured direct repeated-run hotspot from the canonical
   evidence lane.

7. **The sprint preserved the stronger reviewed baseline while landing governance and benchmark changes.**
   Day 13 passed:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   and preserved the maintained reviewed anchors:
   - reviewed CMake parity `53`
   - Makefile/CMake parity `53 vs 53`
   - full reviewed CMake `ctest` `53 / 53`
   - full reviewed CMake total real time `784.97 sec`
   That keeps Sprint 65 credible as real maintained-governance work rather
   than docs-only benchmark interpretation.

8. **Sprint 65 ended with a smaller and more explicit next queue.**
   The sprint did not just normalize benchmark outputs. It also clarified what
   comes next:
   - packaging, ABI, and platform-quality convergence
   - platform residual recheck against the reviewed truthfulness fence
   - bounded packaging/productization improvements
   - dead-code and platform follow-through only where justified
   - CI and contract reconciliation around the resulting packaging/platform
     surface
   That is a cleaner handoff than a generic “more performance work later”
   backlog.

## What Didn't Go Well

1. **The sprint still carried a large design/policy component relative to the size of the implementation delta.**
   That was the correct tradeoff for benchmark governance, but it means Sprint
   65 still spent substantial effort on category design, output policy, and
   interpretation alignment relative to the amount of solver code changed.

2. **The canonical benchmark surface is stronger, not comprehensive.**
   Sprint 65 intentionally left a lot outside the canonical lane:
   - `bench_refactor`
   - `bench_ldlt_csc`
   - regression-sensitive runtime harnesses
   - broader exploratory drivers
   That is the right outcome, but it also means the repo still carries a
   broader benchmark inventory whose roles remain intentionally unequal.

3. **The direct repeated-run efficiency batch was deliberately narrow.**
   Sprint 65 landed one real solver-efficiency improvement, but it did not try
   to solve all follow-through candidates in one sprint:
   - no iterative-handle efficiency churn
   - no eigensolver-handle efficiency churn
   - no LDL^T symmetry-for-its-own-sake batch
   - no broader backend layering
   That boundedness is good engineering, but it means the efficiency story is
   still incremental rather than “done.”

4. **The reviewed validation path is now very expensive.**
   Sprint 65 still closed cleanly, but the reviewed CMake path took:
   - `784.97 sec`
   with:
   - `test_reorder_nd` alone taking `574.47 sec`
   That does not block correctness, but it does make performance-governance
   sprints more expensive to validate than the landed code size might suggest.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 65 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `784.97 sec` |

### Sprint 65 artifact package

| Metric | Sprint 65 close state |
|---|---:|
| total artifact files under `SPRINT_65/artifacts/` | `15` |
| main audit/design/integration artifacts | `8` |
| reporting/docs/validation/closeout artifacts | `4` |

Notes:

- main audit/design/integration artifacts:
  - `day3-benchmark-role-audit.md`
  - `day4-benchmark-role-rerank-and-canonical-surface-candidates.md`
  - `day5-output-and-taxonomy-normalization-design.md`
  - `day6-canonical-performance-surface-and-implementation-fence.md`
  - `day7-solver-efficiency-target-selection-and-landing-design.md`
  - `day8-benchmark-taxonomy-and-output-batch1.md`
  - `day9-canonical-baseline-consolidation-batch.md`
  - `day10-direct-csc-cholesky-efficiency-batch.md`
- reporting/docs/validation/closeout artifacts:
  - `day11-local-ci-friendly-regression-checks.md`
  - `day12-docs-and-example-alignment.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Sprint 65 landed performance-governance package

| Metric | Sprint 65 close state |
|---|---:|
| canonical maintained benchmark binaries normalized | `4` |
| new reporting surfaces materially added | `1` |
| solver-efficiency hot paths materially tightened | `1` |
| targeted Day 13 follow-on commands rerun | `18` |
| explicitly deferred later benchmark/productization lanes | `5` |

Notes:

- canonical maintained benchmark binaries normalized:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- new reporting surfaces materially added:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - stable canonical CSV and manifest output under
    `build/bench-reports/canonical/`
- solver-efficiency hot paths materially tightened:
  - direct repeated-run CSC/Cholesky supernodal `row_map` walk in
    `src/sparse_chol_csc_supernodal.c`
- targeted Day 13 follow-on commands rerun:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_sparse_lu`
  - `./build/test_qr`
  - `./build/test_svd`
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_svd_lowrank`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- explicitly deferred later benchmark/productization lanes:
  - packaging, ABI, and platform-quality convergence
  - platform residual recheck against the preserved reviewed truthfulness fence
  - bounded packaging/productization improvements on the highest-value
    release/install seams
  - dead-code and platform follow-through only where the audited
    productization story justifies it
  - CI and contract reconciliation around the resulting packaging/platform
    surface

## Residual Deferred Debt

Sprint 65 was explicitly about turning the benchmark surface into a clearer
maintained performance-governance surface and then applying that information to
one highest-value solver-efficiency seam. The main open work it intentionally
hands forward is:

- packaging, ABI, and platform-quality convergence
- platform residual recheck against the reviewed Linux/macOS/Windows truth
  fence
- bounded packaging and install/release improvements where the current product
  story is still thin
- dead-code and platform follow-through only where the productization audit
  justifies it
- CI and contract reconciliation around the resulting packaging/platform
  surface

Still consciously constrained rather than silently “solved”:

- no broad timing-threshold CI policy
- no claim that every benchmark binary is canonical
- no reopening of the backend-architecture-first queue
- no fake platform closure beyond reviewed evidence
- no broad solver-efficiency rewrite outside the bounded Cholesky CSC seam

Not carried forward as unresolved Sprint 65 debt:

- missing benchmark-role taxonomy
- missing canonical maintained performance surface
- missing normalized machine-readable outputs on the canonical lane
- missing threshold-free canonical report surface
- missing workflow teaching versus proof ownership split
- missing validated Sprint 65 closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-benchmark-role-audit.md](./artifacts/day3-benchmark-role-audit.md)
- [day4-benchmark-role-rerank-and-canonical-surface-candidates.md](./artifacts/day4-benchmark-role-rerank-and-canonical-surface-candidates.md)
- [day5-output-and-taxonomy-normalization-design.md](./artifacts/day5-output-and-taxonomy-normalization-design.md)
- [day6-canonical-performance-surface-and-implementation-fence.md](./artifacts/day6-canonical-performance-surface-and-implementation-fence.md)
- [day7-solver-efficiency-target-selection-and-landing-design.md](./artifacts/day7-solver-efficiency-target-selection-and-landing-design.md)
- [day8-benchmark-taxonomy-and-output-batch1.md](./artifacts/day8-benchmark-taxonomy-and-output-batch1.md)
- [day9-canonical-baseline-consolidation-batch.md](./artifacts/day9-canonical-baseline-consolidation-batch.md)
- [day10-direct-csc-cholesky-efficiency-batch.md](./artifacts/day10-direct-csc-cholesky-efficiency-batch.md)
- [day11-local-ci-friendly-regression-checks.md](./artifacts/day11-local-ci-friendly-regression-checks.md)
- [day12-docs-and-example-alignment.md](./artifacts/day12-docs-and-example-alignment.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 65 achieved its goal:

- the repo now has one explicit canonical maintained performance surface
- the canonical lane now emits stable normalized machine-readable outputs
- the repo now has a threshold-free canonical report surface that fits its
  current truthfulness contract
- the example/benchmark ownership split is clearer and more product-like than
  it was at sprint start
- one bounded direct repeated-run CSC/Cholesky efficiency seam was tightened
  without reopening broader backend or platform work
- the sprint closed from a fully reviewed validated baseline and handed
  forward a smaller, cleaner, and more explicit Sprint 66 queue
