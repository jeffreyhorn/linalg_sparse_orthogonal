# Sprint 67 Retrospective

**Sprint:** 67 — Large-Source Maintainability Phase 3  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 67 maintainability scope and validation baseline captured before implementation work landed
- [x] residual hotspot audit reduced the broad maintainability claim to a ranked live seam map
- [x] the first graph/reorder landing boundary was fixed before code edits began
- [x] the strongest graph/reorder ownership extraction batch landed without widening into the broader graph subsystem
- [x] the duplicated ND compatibility/default-policy baseline converged to one shared internal owner
- [x] the highest-value CSC residual seam landed as a bounded large-`n` Cholesky analysis-to-CSC handoff convergence batch
- [x] maintained docs/regression ownership wording was realigned to the landed maintainability boundaries
- [x] full validation sweep completed from the landed Sprint 67 tree
- [x] Sprint 67 closeout and handoff completed from the validated baseline

## What Went Well

1. **Sprint 67 avoided fake “cleanup” and stayed on the strongest ownership seams.**
   The sprint did not turn into a generic refactor wave. It reduced the broad
   project-plan wording to a concrete seam map, then stayed centered on:
   - graph/reorder orchestration ownership
   - shared ND policy duplication
   - one bounded large-`n` Cholesky CSC residual seam

2. **The first graph/reorder batch removed real mixed-ownership pressure.**
   Sprint 67 made `src/sparse_graph.c` and `src/sparse_reorder_nd.c` clearer by
   extracting:
   - `graph_uncoarsen_options_t`
   - `graph_uncoarsen_options_from_env(...)`
   - `nd_emit_leaf_amd(...)`
   - `nd_partition_current_graph(...)`
   - `nd_recurse_side(...)`
   That is a real maintainability win because the recursive/orchestration shell
   is now easier to read without pretending the rest of the graph family needed
   the same treatment immediately.

3. **The ND compatibility/default-policy story is now cleaner and less duplicated.**
   `sparse_reorder_nd_default_policy()` now owns the internal ND
   compatibility/default-policy baseline. `src/sparse_analysis.c` no longer
   carries its own separate duplicated parser/default shell for that same lane.
   The sprint preserved the important compatibility rule that typed analysis
   values still override compatibility env vars exactly as shipped.

4. **The second implementation landing stayed small but high value.**
   Sprint 67 did not chase a broad CSC/iterative batch after the graph work.
   It identified that the highest-value remaining residual seam was the large-`n`
   Cholesky analysis-backed CSC handoff and closed that lane specifically:
   - `src/sparse_analysis.c`
   - `src/sparse_chol_csc.c`
   - `src/sparse_chol_csc_internal.h`
   - `tests/test_chol_csc.c`
   - `tests/test_integration.c`

5. **The proof-surface ownership split is clearer than it was at sprint start.**
   At close:
   - `tests/test_reorder_nd.c` owns the shared ND compatibility/default-policy convergence lane
   - `tests/test_chol_csc.c` owns the family-local large-`n` analysis-backed Cholesky CSC handoff lane
   - `tests/test_integration.c` owns the public one-shot vs explicit repeated-run Cholesky parity/failure-preservation lane
   - benchmark surfaces remain benchmark-side proof, not substitutes for those regression owners

6. **The sprint preserved the strongest reviewed baseline across real implementation changes.**
   Day 13 passed:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   with maintained reviewed anchors still exact at:
   - reviewed CMake parity `53`
   - Makefile/CMake parity `53 vs 53`
   - full reviewed CMake `ctest` `53 / 53`
   - full reviewed CMake total real time `418.98 sec`

7. **The carry-forward queue is smaller and more honest.**
   Sprint 67 closed the strongest maintainability contradictions it actually
   touched and left behind a bounded residual queue instead of pretending the
   entire CSC/iterative backlog was solved.

## What Didn't Go Well

1. **Sprint 67’s “CSC/iterative residual decomposition” headline landed more narrowly than the epic-level wording suggests.**
   The sprint did land the highest-value CSC residual seam, but it did not
   become a broad second decomposition batch across:
   - `src/sparse_chol_csc.c`
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_iterative.c`
   - `src/sparse_eigs.c`
   That was the right tradeoff, but it means the sprint delivered a sharper
   maintainability slice than the epic title alone might imply.

2. **Comment/chronology cleanup stayed bounded rather than becoming its own visible batch.**
   Sprint 67 improved ownership clarity in touched files, but it did not run a
   separate broad chronology scrub across permanent implementation surfaces.
   That avoided fake productivity, but it also means some stale sprint-history
   wording outside the touched seams remains for later work.

3. **The proof and validation path is still dominated by `test_reorder_nd`.**
   Sprint 67 closed cleanly, but the reviewed CMake path still spent:
   - `291.93 sec`
   in `test_reorder_nd` out of the:
   - `418.98 sec`
   total. That cost is inherited rather than created here, but it remains the
   main practical weight on future maintainability validation.

4. **Build/regression alignment was more about proof ownership than build-surface movement.**
   The sprint did align the maintained regression story, but it did not need a
   larger build-surface restructure. That is fine technically, but it means the
   “build/regression alignment” item closed more through proof-ownership and
   docs truthfulness than through visible build-system edits.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 67 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `418.98 sec` |

### Sprint 67 artifact package

| Metric | Sprint 67 close state |
|---|---:|
| total artifact files under `SPRINT_67/artifacts/` | `15` |
| baseline/audit/design artifacts | `9` |
| implementation artifacts | `3` |
| alignment/validation/closeout artifacts | `3` |

Notes:

- baseline/audit/design artifacts:
  - `day1-scope-and-maintainability-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-touched-surface-recheck.md`
  - `day3-residual-hotspot-audit.md`
  - `day4-first-landing-boundary.md`
  - `day5-graph-reorder-decomposition-design.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day8-shared-nd-policy-convergence-design.md`
  - `day10-post-landing-audit-and-rerank.md`
- implementation artifacts:
  - `day6-graph-reorder-ownership-extraction-batch1.md`
  - `day9-shared-nd-policy-convergence-batch.md`
  - `day11-large-n-cholesky-analysis-csc-handoff-batch.md`
- alignment/validation/closeout artifacts:
  - `day12-build-and-regression-alignment.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Sprint 67 landed maintainability package

| Metric | Sprint 67 close state |
|---|---:|
| materially touched implementation/internal ownership surfaces | `6` |
| materially strengthened proof surfaces | `3` |
| maintained docs/regression truth surfaces aligned | `3` |
| targeted Day 13 follow-on commands rerun | `10` |

Notes:

- materially touched implementation/internal ownership surfaces:
  - `src/sparse_graph.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_nd_internal.h`
  - `src/sparse_analysis.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_chol_csc_internal.h`
- materially strengthened proof surfaces:
  - `tests/test_reorder_nd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`
- maintained docs/regression truth surfaces aligned:
  - `README.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
- targeted Day 13 follow-on commands rerun:
  - `./build/test_integration`
  - `./build/test_graph`
  - `./build/test_reorder_nd`
  - `./build/test_chol_csc`
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

## Residual Deferred Debt

Sprint 67 was explicitly about the strongest remaining ownership contradictions
after the productization close, not about solving every large file equally. The
main open work it intentionally hands forward is:

- bounded CSC/analysis residual decomposition beyond the landed Cholesky large-`n` handoff seam
- iterative/eigensolver residual decomposition only where the remaining ownership blur still justifies the proof cost
- stale sprint-history/comment chronology cleanup on later touched permanent implementation or header files
- further build/regression alignment only when future decomposition work actually moves ownership again

Still consciously constrained rather than silently “solved”:

- no broad CSC family redesign
- no broad iterative/eigensolver decomposition wave
- no packaging/platform/build churn reopening
- no fake abstraction layer that blurs family ownership
- no repo-wide chronology scrub disconnected from touched ownership seams

Not carried forward as unresolved Sprint 67 debt:

- the strongest graph/reorder orchestration ownership contradiction
- duplicated ND compatibility/default-policy baseline ownership
- the large-`n` Cholesky analysis-backed CSC handoff duplication
- unclear proof ownership between `test_reorder_nd`, `test_chol_csc`, and `test_integration`
- missing validated Sprint 67 closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-and-maintainability-baseline.md](./artifacts/day1-scope-and-maintainability-baseline.md)
- [day1-authoritative-inputs.txt](./artifacts/day1-authoritative-inputs.txt)
- [day2-validation-baseline-and-touched-surface-recheck.md](./artifacts/day2-validation-baseline-and-touched-surface-recheck.md)
- [day3-residual-hotspot-audit.md](./artifacts/day3-residual-hotspot-audit.md)
- [day4-first-landing-boundary.md](./artifacts/day4-first-landing-boundary.md)
- [day5-graph-reorder-decomposition-design.md](./artifacts/day5-graph-reorder-decomposition-design.md)
- [day6-graph-reorder-ownership-extraction-batch1.md](./artifacts/day6-graph-reorder-ownership-extraction-batch1.md)
- [day7-post-landing-audit-and-rerank.md](./artifacts/day7-post-landing-audit-and-rerank.md)
- [day8-shared-nd-policy-convergence-design.md](./artifacts/day8-shared-nd-policy-convergence-design.md)
- [day9-shared-nd-policy-convergence-batch.md](./artifacts/day9-shared-nd-policy-convergence-batch.md)
- [day10-post-landing-audit-and-rerank.md](./artifacts/day10-post-landing-audit-and-rerank.md)
- [day11-large-n-cholesky-analysis-csc-handoff-batch.md](./artifacts/day11-large-n-cholesky-analysis-csc-handoff-batch.md)
- [day12-build-and-regression-alignment.md](./artifacts/day12-build-and-regression-alignment.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 67 achieved its goal:

- the repo’s strongest remaining maintainability pain points are smaller and more clearly owned than they were at sprint start
- graph/reorder orchestration is cleaner without widening into a fake whole-subsystem rewrite
- ND compatibility/default-policy ownership is no longer duplicated across major surfaces
- the large-`n` Cholesky analysis-to-CSC handoff is cleaner and more explicitly proven
- the maintained proof and documentation story now matches the landed ownership boundaries
- the sprint closed from a fully reviewed validated baseline and hands forward a smaller, clearer residual maintainability queue
