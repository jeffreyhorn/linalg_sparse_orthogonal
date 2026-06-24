# Sprint 88 Retrospective

**Sprint:** 88 — Front-Door Usability & Workflow Simplification  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 88 fixed the front-door, support-surface, and implementation-day
      validation contract before landing usability work
- [x] the strongest live usability contradiction map was reranked from the
      current tree rather than inherited generically from Sprint 87
- [x] Sprint 88 fixed one explicit first implementation fence centered on:
  - `README.md`
- [x] Sprint 88 landed one bounded README/front-door batch:
  - `README.md` now behaves more like an adoption-first surface
  - it routes the smallest real workflow choices before deeper support and
    policy references
- [x] Sprint 88 landed one bounded examples/workflow batch:
  - `examples/README.md` now behaves as a compact post-README adoption map
  - direct, repeated-run direct, iterative, eigensolver, and
    installed-consumer lanes read as distinct next-step choices
- [x] Sprint 88 landed one bounded support-surface consolidation batch:
  - `INSTALL.md` now clearly owns operational setup, staged install,
    installed-consumer workflows, and install-surface validation
  - benchmark and maintainer surfaces remained retained owners instead of
    being widened through install/support cleanup
- [x] Sprint 88 re-audited the planned header/API narrative lane and closed it
      explicitly as unnecessary for this sprint
- [x] Sprint 88 used bounded follow-through correctly:
  - `README.md`, `examples/README.md`, and `INSTALL.md` moved as the main
    user-facing owners
  - `benchmarks/README.md` remained the benchmark-local command/proof owner
  - `docs/maintainer_guide.md` remained the maintainer-only policy owner
  - `include/sparse_iterative.h`, `include/sparse_eigs.h`,
    `include/sparse_matrix.h`, and `include/sparse_types.h` remained
    API-local contract owners rather than adoption/support centers
- [x] Sprint 88 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
  - `make bench-canonical-report`
- [x] Sprint 88 closed with one explicit Sprint 89 handoff queue instead of
      another generic Epic 8 usability summary

## What Went Well

1. **Sprint 88 chose the right first usability lane.**
   The sprint did not start by rewriting examples, benchmarks, or headers
   first. It correctly tightened the main adoption surface at `README.md`
   before widening into example and support follow-through.

2. **The Day 6 README landing made the front door materially clearer.**
   `README.md` now routes users through a smaller first-choice map rather than
   making them parse benchmark, support, and maintainer density before they
   can decide how to start.

3. **The Day 9 examples landing improved the real post-README path.**
   `examples/README.md` now reads like the actual next-step map after the
   README rather than a flatter inventory of shipped examples.

4. **The Day 11 support batch improved ownership without widening claims.**
   `INSTALL.md` now behaves like the real operational setup and
   installed-consumer owner without dragging benchmark-policy or
   maintainer-policy wording into the same batch.

5. **Scope discipline held all the way through Day 12.**
   Sprint 88 did not reopen packaging or platform semantics from Sprint 87,
   did not smear benchmark policy into front-door docs, and did not force a
   speculative header rewrite when the retained headers were already acting as
   API-local contract owners.

6. **Sprint 89 now starts from a cleaner public-surface package.**
   The final Epic 8 sprint inherits a clearer adoption/support/package
   narrative and does not need to spend its first days cleaning up the basic
   audience split.

## What Didn't Go Well

1. **Sprint 88 did not need the planned header/API narrative implementation lane.**
   That was the correct evidence-based outcome, but it also means one planned
   line item closed as explicitly unnecessary rather than as a landed code or
   header batch.

2. **The strongest usability gains stayed in docs/support surfaces.**
   That was the right choice for this sprint, but it means Sprint 88 improved
   adoption clarity more than it changed any underlying algorithm or runtime
   behavior.

3. **The reviewed runtime long pole still dominates the close baseline.**
   Sprint 88 closed cleanly from its Day 13 baseline, but the reviewed path
   still carries:
   - reviewed CMake total = `408.39 sec`
   - reviewed `test_reorder_nd` = `222.30 sec`

4. **Canonical benchmark reporting stayed unchanged by design.**
   That was the correct bounded choice, but it also means Sprint 88 did not
   alter the benchmark/reporting ownership model while simplifying the
   front-door surfaces.

5. **Workflow/platform wording stayed intentionally narrower than a broad user story.**
   Sprint 88 preserved the support split instead of flattening it, so some
   users still need to move from README to examples to INSTALL to get the full
   picture. That is more truthful, but not maximally terse.

## Final Metrics

### Validation and usability-close anchors

| Metric | Sprint 88 close state |
|---|---:|
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `408.39 sec` |
| reviewed `test_reorder_nd` time | `222.30 sec` |
| example follow-on: `example_analysis` | passed |
| example follow-on: `example_basic_solve` | passed |
| `example_analysis` residual | `4.44e-16` |
| `example_basic_solve` residual | `0.00e+00` |
| Make/pkg-config install proof | `bash tests/test_install.sh` passed |
| Make/pkg-config install proof totals | `13` passed, `0` failed |
| CMake install/export proof | `bash tests/test_cmake_install.sh` passed |
| CMake install/export proof totals | `15` passed, `0` failed, `0` skipped |
| canonical reporting follow-on | `make bench-canonical-report` passed |

### Public-surface headline

| Metric | Sprint 88 close state |
|---|---:|
| primary front-door owner | `README.md` |
| post-front-door workflow owner | `examples/README.md` |
| operational setup / installed-consumer owner | `INSTALL.md` |
| benchmark-local command/proof owner | `benchmarks/README.md` |
| maintainer-only policy owner | `docs/maintainer_guide.md` |
| public-header narrative batch landed | `0` |
| public-header narrative batch explicitly unnecessary | `1` |
| package/platform contract reopened | `0` |

### Sprint 88 artifact package

| Metric | Sprint 88 close state |
|---|---:|
| total artifact files under `SPRINT_88/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/follow-through artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-front-door-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-maintained-support-surface-recheck.md`
  - `day3-user-journey-audit.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day12-narrative-freeze-and-validation-queue.md`
- design/follow-through artifacts:
  - `day4-first-usability-boundary.md`
  - `day5-workflow-simplification-design.md`
  - `day6-readme-tutorial-batch.md`
  - `day8-examples-workflow-simplification-design.md`
  - `day9-examples-workflow-simplification-batch.md`
  - `day10-support-surface-consolidation-design.md`
  - `day11-support-surface-consolidation-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed implementation package

| Metric | Sprint 88 close state |
|---|---:|
| user-facing docs/support files touched | `3` |
| install/export proof scripts touched | `0` |
| workflow files touched | `0` |
| public header files touched | `0` |
| C/C++ implementation files touched | `0` |
| benchmark-local docs touched | `0` |

Notes:

- user-facing docs/support files touched:
  - `README.md`
  - `examples/README.md`
  - `INSTALL.md`
- retained owners intentionally left untouched after recheck:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`

## Residual Deferred Debt

Sprint 88 deliberately stopped after the highest-value front-door, example, and
support-surface package. The main open work it hands forward is:

- final Epic 8 end-state re-audit
- external comparison sweep
- final cross-surface fix batch from refreshed evidence
- final validation/reporting sweep
- Epic 8 closeout

Still consciously constrained rather than silently "solved":

- no package/platform contract reopening
- no benchmark/reporting ownership widening
- no shared adoption/support/maintainer megadoc
- no public-header narrative rewrite for its own sake
- no runtime-tail work on the retained `test_reorder_nd` long pole

Not carried forward as unresolved Sprint 88 debt:

- the front-door contradiction rerank
- the bounded workflow-simplification architecture contract
- the Day 6 README/front-door landing
- the Day 9 examples/workflow landing
- the Day 11 support-surface consolidation
- the Day 12 support/proof-owner freeze
- the Day 13 full validation sweep
- the Day 14 explicit Sprint 89 handoff queue

## Key Deliverables

1. **One bounded front-door simplification landed at the strongest adoption owner.**
   `README.md` now routes users toward the smallest real workflow choice
   before deeper support, benchmark, or maintainer references.

2. **One bounded examples/workflow simplification landed at the post-front-door owner.**
   `examples/README.md` now gives users a clearer next-step map across direct,
   repeated-run direct, iterative, eigensolver, and installed-consumer paths.

3. **One bounded support-surface consolidation landed at the operational owner.**
   `INSTALL.md` now clearly owns setup, staged install, installed-consumer
   guidance, and local package-proof interpretation.

4. **One bounded ownership freeze prevented unnecessary churn.**
   `benchmarks/README.md`, `docs/maintainer_guide.md`, and the highest-signal
   public headers were re-audited and correctly retained as their existing
   owners instead of being widened through Sprint 88 docs churn.

5. **The sprint now closes from one explicit usability validation baseline.**
   The reviewed path, maintained examples, install/export proof scripts, and
   canonical benchmark-report bundle are all explicit in writing at sprint
   close.

## Bottom Line

Sprint 88 achieved its purpose. The project now has a clearer front door, a
cleaner post-README example path, a more explicit operational install/support
owner, and a sharper audience split across adoption, support, benchmark, and
maintainer surfaces, all without widening package, platform, benchmark, or
header claims beyond what the repo can realistically maintain. Sprint 89 can
now focus on final Epic 8 end-state audit, external comparison, and closeout
instead of first untangling the basic public-surface story.
