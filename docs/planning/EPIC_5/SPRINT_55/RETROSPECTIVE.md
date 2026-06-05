# Sprint 55 Retrospective

**Sprint:** 55 — Large-Source Decomposition Phase 1  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 55 baseline and scope captured from the Sprint 54 validated repeated-run solver package
- [x] reviewed validation/truthfulness baseline rechecked before decomposition work
- [x] `sparse_eigs.c` seam audit completed against the live repo
- [x] first eigensolver extraction boundary designed explicitly before code movement
- [x] bounded LOBPCG extraction landed
- [x] second eigensolver extraction boundary re-audited and designed from the Day 5 landed state
- [x] bounded thick-restart extraction landed
- [x] `sparse_iterative.c` seam audit completed against the live repo
- [x] first iterative extraction boundary designed explicitly before code movement
- [x] bounded MINRES extraction landed
- [x] stale sprint-history narrative removed from touched permanent implementation files
- [x] post-landing compatibility audit completed
- [x] full validation sweep completed from the landed decomposition state
- [x] Sprint 55 closeout and next-phase handoff completed from the validated baseline

## What Went Well

1. **Sprint 55 delivered real ownership seams instead of a cosmetic file shuffle.**
   The sprint moved durable backend-owned implementation clusters into permanent
   files:
   - eigensolver:
     - `src/sparse_eigs_lobpcg.c`
     - `src/sparse_eigs_thick_restart.c`
   - iterative:
     - `src/sparse_iterative_minres.c`
   That means the sprint improved maintainability through real ownership
   boundaries rather than through comment-only or declaration-only churn.

2. **The largest eigensolver hotspot was reduced materially.**
   The main retained eigensolver orchestration file dropped from:
   - `src/sparse_eigs.c`: `3233` -> `1534`
   That is a meaningful ownership improvement, not a marginal trim. The
   retained file is now much more front-door/orchestration-focused, while
   backend-specific code lives where a maintainer would expect it.

3. **The iterative decomposition stayed bounded and pragmatic.**
   Sprint 55 did not try to split all iterative families at once. It chose the
   cleanest first seam:
   - `MINRES`
   and landed it without reopening:
   - the public iterative handle contract
   - one-shot/default solver behavior
   - the explicit exclusion of `BiCGSTAB` from the public repeated-run handle
     set
   That restraint kept the sprint coherent.

4. **The build-system ownership surfaces stayed aligned.**
   The sprint carried the new files through both supported local build paths:
   - `Makefile`
   - `CMakeLists.txt`
   So the decomposition is not true only for one tool path or one local build
   mode. That matters for maintainability because ownership boundaries that
   exist only in one build surface are not stable.

5. **The comment cleanup improved permanent code quality without damaging useful explanation.**
   Sprint 55 removed stale `Sprint ... Day ...` narrative from touched
   implementation files while preserving durable algorithm and ownership
   commentary. That is the right trade:
   - less historical noise in production files
   - no loss of maintainer-facing explanation where it still matters

6. **The sprint preserved the public solver/lifecycle fence.**
   Sprint 55 stayed decomposition-first instead of quietly reopening behavior:
   - no public API redesign
   - no support-boundary drift
   - no behavior-visible lifecycle change
   - no solver-family scope expansion
   That makes the decomposition easier to trust because it did not mix major
   architecture changes into ownership work.

7. **The sprint closed from a real validated reviewed baseline.**
   Day 13 passed:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   and preserved the truthfulness anchors:
   - reviewed CMake parity `53`
   - Makefile/CMake parity `53 vs 53`
   - reviewed CMake `ctest` `53 / 53`
   - reviewed CMake total time `253.48 sec`

## What Didn't Go Well

1. **Sprint 55 intentionally stopped at Phase 1 seams, not full large-file completion.**
   That was the correct scope choice, but it means the residual queue is still
   visible:
   - iterative:
     - `GMRES`
     - shared block-wrapper scaffolding
   - eigensolver:
     - more trimming inside retained `src/sparse_eigs.c`
     - possible future private-header taxonomy cleanup

2. **The retained orchestration files are smaller, but not yet small.**
   Sprint 55 materially improved both hotspots, but it did not finish the
   entire large-source problem:
   - `src/sparse_eigs.c` remains a meaningful orchestration file at `1534`
     lines
   - `src/sparse_iterative.c` remains a meaningful orchestration file at `1985`
     lines
   So the sprint solved Phase 1, not the whole maintainability agenda.

3. **The sprint produced maintainability value more than user-visible feature value.**
   That is appropriate for this phase, but it also means much of the sprint’s
   benefit is structural:
   - better ownership
   - cleaner permanent implementation commentary
   - easier future extraction paths
   rather than large new user-facing functionality.

4. **Some solver-family seams remain intentionally deferred.**
   Sprint 55 kept `BiCGSTAB` out of the decomposition and public-handle
   discussion for good reasons, but that also means the iterative ownership map
   is still uneven across solver families.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 55 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `253.48 sec` |

### Sprint 55 artifact package

| Metric | Sprint 55 close state |
|---|---:|
| total artifact files under `SPRINT_55/artifacts/` | `15` |
| baseline/audit/design artifacts (Days 1-4, 6, 8-9, 12) | `9` |
| landed implementation/cleanup/validation/closeout artifacts (Days 5, 7, 10-11, 13-14) | `6` |

### Decomposition outputs

| Metric | Sprint 55 close state |
|---|---:|
| extracted permanent implementation files | `3` |
| main large-source files materially reduced | `2` |
| touched build-system surfaces aligned to the new ownership split | `2` |
| targeted Sprint 55 follow-on commands rerun in Day 13 | `8` |

Notes:

- extracted permanent implementation files:
  - `src/sparse_eigs_lobpcg.c`
  - `src/sparse_eigs_thick_restart.c`
  - `src/sparse_iterative_minres.c`
- main large-source files materially reduced:
  - `src/sparse_eigs.c`: `3233 -> 1534`
  - `src/sparse_iterative.c`: `2377 -> 1985`
- touched build-system surfaces aligned to the new ownership split:
  - `Makefile`
  - `CMakeLists.txt`
- targeted Sprint 55 follow-on commands rerun in Day 13:
  - `./build/test_iterative`
  - `./build/test_minres`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
  - `./build/example_iterative`
  - `./build/example_eigs`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

## Residual Deferred Debt

Sprint 55 was explicitly about bounded large-source decomposition Phase 1. The
main open work it intentionally hands forward is:

- later iterative decomposition:
  - `GMRES`
  - shared block-wrapper scaffolding
- later eigensolver cleanup/decomposition:
  - more trimming of retained `src/sparse_eigs.c`
  - possible future private-header taxonomy cleanup if it clearly improves
    maintainability
- still intentionally deferred:
  - broad public API redesign
  - reopening the public repeated-run support boundary
  - turning `BiCGSTAB` into a Sprint 55-style public-handle topic
  - large documentation/tutorial rewrites unrelated to source ownership

Not carried forward as unresolved Sprint 55 debt:

- missing eigensolver Phase 1 extraction
- missing iterative Phase 1 extraction
- missing build-system alignment for extracted ownership seams
- missing historical-comment cleanup in touched permanent implementation files
- missing post-landing compatibility audit
- missing full validated closeout baseline

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-sparse-eigs-seam-audit.md](./artifacts/day3-sparse-eigs-seam-audit.md)
- [day4-eigensolver-decomposition-batch1-design.md](./artifacts/day4-eigensolver-decomposition-batch1-design.md)
- [day5-eigensolver-decomposition-batch1.md](./artifacts/day5-eigensolver-decomposition-batch1.md)
- [day6-eigensolver-decomposition-batch2-design.md](./artifacts/day6-eigensolver-decomposition-batch2-design.md)
- [day7-eigensolver-decomposition-batch2.md](./artifacts/day7-eigensolver-decomposition-batch2.md)
- [day8-sparse-iterative-seam-audit.md](./artifacts/day8-sparse-iterative-seam-audit.md)
- [day9-iterative-decomposition-batch1-design.md](./artifacts/day9-iterative-decomposition-batch1-design.md)
- [day10-iterative-decomposition-batch1.md](./artifacts/day10-iterative-decomposition-batch1.md)
- [day11-historical-comment-reduction-sweep.md](./artifacts/day11-historical-comment-reduction-sweep.md)
- [day12-post-landing-compatibility-audit.md](./artifacts/day12-post-landing-compatibility-audit.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 55 achieved its goal:

- the repo now has real permanent ownership seams in both of its largest
  remaining solver implementation hotspots
- `src/sparse_eigs.c` and `src/sparse_iterative.c` are materially smaller and
  more orchestration-focused than at sprint start
- the touched permanent implementation files are cleaner maintainability
  surfaces because stale sprint-history narrative was removed
- the public solver/lifecycle fence stayed intact throughout the decomposition
  work
- the sprint closed from a fully validated reviewed baseline with exact
  preserved truthfulness anchors

Sprint 56 can now start from a cleaner, validated decomposition baseline rather
than needing to re-establish whether Sprint 55’s ownership splits were real,
whether the build surfaces agreed, or whether the reviewed local quality
contract drifted during the large-source work.
