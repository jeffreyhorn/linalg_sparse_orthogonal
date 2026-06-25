# Sprint 89 Day 4: Final Integration Boundary

## Purpose

Fix the first bounded Sprint 89 final-integration implementation fence so the
next design pass can define one real external-comparison and final-evidence
contract instead of another broad final-cleanup rewrite.

## Main Result

Sprint 89 now has one explicit first implementation fence:

- required first landing:
  - bounded external comparison and end-state evidence package
- directly forced support surfaces only if the first landing truly needs them:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `scripts/bench_canonical_report.sh`
  - `benchmarks/README.md`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_fillin.c`
- support-only proof, workflow, and closeout surfaces that stay later unless
  the first landing truly forces movement:
  - `make quality-review-full`
  - reviewed representative binaries under `build/quality-review-cmake/`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - Sprint 89 retrospective, Epic 8 closeout, and final project-summary
    surfaces
- explicitly deferred from the first landing:
  - final cross-surface fix batch as a first-batch center
  - full validation/reporting sweep as a first-batch center
  - residual-queue finalization as a first-batch center
  - Epic 8 summary writing as a first-batch center
  - broad reopening of product, capability, packaging, or usability lanes

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- the best first Sprint 89 move is one bounded external-comparison and
  end-state-evidence lane
- the first landing should decide how the final correctness, package-shape,
  and bounded performance evidence will be gathered and interpreted before any
  last-mile implementation widening moves
- install/export proof, benchmark/reporting surfaces, and support docs remain
  directly allowed only if the comparison contract truly forces them to move
- reviewed runtime reduction, residual-queue calibration, and all final
  closeout writing stay later unless the evidence lane proves they must move

## Preserved First-Batch Fence

The preserved first-batch non-goal fence is explicit now:

- no blind final-fix batch before external evidence exists
- no broad reopening of earlier sprint scope
- no speculative runtime or capability widening without the comparison lane
  justifying it
- no summary/retrospective writing before the final validated baseline exists
- no support-surface churn detached from a real evidence-package seam

## Exit State

- Sprint 89 now has one bounded first final-integration landing center.
- Day 5 can design one explicit external-comparison and final-evidence
  contract inside that fence.
- Later fix, validation, residual-calibration, and closeout-writing work is
  held back until the evidence lane is defined.
