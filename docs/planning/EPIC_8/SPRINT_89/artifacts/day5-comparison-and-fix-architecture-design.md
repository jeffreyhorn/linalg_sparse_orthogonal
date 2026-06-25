# Sprint 89 Day 5: Comparison and Fix Architecture Design

## Purpose

Define the bounded external-comparison and final-fix contract that Sprint 89
will actually support before any end-state evidence or last-mile
implementation work lands.

## Main Result

Sprint 89 now has one explicit first implementation contract:

- required implementation center:
  - bounded external comparison and end-state evidence package
- directly forced support surfaces only if the first batch truly needs them:
  - `tests/test_chol_csc.c`
  - `tests/chol_external_dense_reference.py`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `benchmarks/bench_reorder.c`
  - `Makefile`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- retained later owners unless the first batch truly changes their
  obligations:
  - `scripts/bench_canonical_report.sh`
  - `make quality-review-full`
  - Sprint 89 retrospective
  - Epic 8 closeout notes
  - final project-summary surfaces

## Ownership Split

The Day 5 ownership split is now fixed:

- maintained correctness comparison owner:
  - `tests/test_chol_csc.c`
- retained external dense reference helper owner:
  - `tests/chol_external_dense_reference.py`
- maintained package-shape truth owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- bounded performance-reference support owner:
  - `benchmarks/bench_reorder.c`
- bounded runtime rerun contract owner if the comparison lane truly needs a
  dedicated local driver:
  - `Makefile` through `make bench-reorder-sprint86`
- retained canonical reporting owner after the comparison lane:
  - `scripts/bench_canonical_report.sh`
- support-surface wording owners only if the evidence package truly changes
  how the contract should be read:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`

## Comparison Contract

The strongest comparison contract is now explicit:

- maintained correctness comparison lane:
  - bounded CHOLMOD-class SPD direct-solver comparison through the retained
    external dense reference lane already owned by `tests/test_chol_csc.c`
    and `tests/chol_external_dense_reference.py`
- maintained package-shape comparison lane:
  - installed consumer and export-surface truth through
    `tests/test_install.sh` and `tests/test_cmake_install.sh`
- bounded performance-reference support lane:
  - touched reorder/ND runtime evidence through
    `bench_reorder --sprint86-slice --skip-factor`
  - this is support for the final runtime reading, not a broad product
    correctness or benchmark-superiority claim
- explicitly advisory but not first-contract lanes:
  - METIS-class graph/reordering comparison remains useful advisory context
    only
  - broader sparse-solver ecosystem comparison remains outside the maintained
    Sprint 89 contract

## Outcome Interpretation

The strongest outcome interpretation contract is now fixed:

- immediate final fix candidate:
  - maintained correctness comparison disagreement on the bounded SPD lane
  - package/install/export contract mismatch on the maintained local proof
    surfaces
  - clear touched-lane runtime contradiction on the retained reorder/ND
    evidence surface that stays attributable and bounded
- calibrated non-claim:
  - comparison confirms the repo remains intentionally bounded rather than
    broad or best-in-class on a lane
  - package/platform asymmetry remains truthful and explicitly supported only
    on the maintained surfaces
- future residual item:
  - advisory ecosystem gaps that remain real but fall outside the maintained
    Sprint 89 comparison contract

## Final-Fix Entry Contract

The strongest final-fix entry contract is now explicit:

- a final fix batch should land only if the comparison package exposes:
  - a correctness mismatch on the maintained SPD comparison lane
  - a local install/export or consumer-shape contradiction
  - a bounded reorder/ND runtime or proof-surface contradiction still large
    enough to justify one last touched implementation pass
  - or a support-surface wording contradiction made unavoidable by the
    evidence package
- a final fix batch should not land just because:
  - advisory ecosystem comparisons look broader elsewhere
  - the repo remains intentionally bounded on capability or platform shape
  - a performance result is merely less impressive than an external system
    without contradicting the maintained contract

## Strongest Clarification

The useful Day 5 clarification is explicit now:

- Day 6 should not try to compare "everything"
- it should preserve the Sprint 80 oracle fence and the Sprint 84 direct
  differential lane
- it should add package-shape truth and bounded runtime-reference support to
  that same final evidence package
- it should keep canonical reporting and all closeout writing as later lanes
  rather than collapsing them into the first evidence batch

## Preserved First-Batch Fence

The preserved first-batch fence is explicit:

- no mandatory heavyweight external stack for normal builds
- no broad correctness claim inflation from performance-reference lanes
- no advisory METIS-class or wider sparse-solver comparison promoted into the
  maintained first contract
- no blind final-fix batch before evidence exists
- no support-surface churn detached from the landed evidence seam
- no closeout-writing drift into the first comparison batch

## Exit State

- Sprint 89 now has one bounded external-comparison and final-fix
  architecture contract.
- Ownership between correctness comparison, package-shape proof,
  performance-reference support, retained validation/reporting, and later
  closeout writing is fixed before Day 6 begins.
- Any final implementation batch is now gated by objective evidence-entry
  criteria rather than by generic endgame pressure.
