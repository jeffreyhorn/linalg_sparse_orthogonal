# Sprint 90 Day 5: Competitive Target and Product Contract Design

## Purpose

Define what Epic 9 is actually trying to make the library become so later
implementation, comparison, and closeout work can aim at one stable bounded
target instead of a vague "state of the art" ambition.

## Main Result

Sprint 90 now has one explicit Epic 9 target-state contract:

- target-state reading:
  - bounded state-of-the-art sparse linear algebra library
- explicitly not the target:
  - generic research playground with unconstrained surface sprawl
  - fully productized broad-platform/broad-backend industrial sparse stack
  - benchmark-supremacy claim machine

The strongest success-marker order is now fixed:

- first:
  - compressed-first product-model convergence on the highest-value public
    direct and interop workflows
- second:
  - portable dense/backend maturity beyond the current scalar builtin core and
    bounded Darwin-only acceleration seams
- third:
  - one real capability-breadth widening beyond the current bounded real-only
    scalar and family-local limits
- fourth:
  - cleaner runtime/threading, proof, and reviewed-surface maturity where it
    materially strengthens the first three lanes
- fifth:
  - lower chronology/duplication/maintenance drag across the permanent product
    and maintainer surfaces

## Contract Reading

The Day 5 target reading is now explicit:

- Epic 9 should aim to make this repo a serious modern sparse numerical
  library that is:
  - unusually well validated
  - structurally closer to compressed-first compute reality
  - backend-aware in a portable way
  - broader in capability on at least one high-value lane
  - cleaner in permanent product narrative and build topology
- Epic 9 should *not* try to claim:
  - full compressed-first replacement of every linked-list-centered surface
  - broad complex/mixed-precision maturity across the whole product
  - full platform symmetry
  - full shared-library and packaging maturity across all consumer lanes
  - universal runtime or ordering superiority

## Ownership Split

The Day 5 ownership split is now fixed:

- target-state and public-product contract owners:
  - `docs/planning/EPIC_9/PROJECT_PLAN.md`
  - Sprint 90 Day 3 and Day 4 audit artifacts
- first implementation-center owners implied by the target:
  - `README.md`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- second implementation-center owners implied by the target:
  - `src/sparse_dense.c`
  - `src/sparse_ldlt_csc.c`
  - later touched direct-family consumers
- third implementation-center owners implied by the target:
  - `include/sparse_types.h`
  - later touched iterative/eigs/direct public headers and implementation
    seams
- later coherence/convergence owners, but not first-center owners:
  - `docs/maintainer_guide.md`
  - `README.md`
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `Makefile`
  - `CMakeLists.txt`
  - workflow files under `.github/workflows/`

## Success Markers

The most important success markers are now explicit:

- product-model convergence:
  - high-value direct and interop workflows no longer read as
    linked-list-first by default
- backend maturity:
  - the library has at least one serious portable accelerated dense/backend
    lane beyond the builtin scalar baseline
- capability breadth:
  - the repo ships one real additional high-value capability lane rather than
    only clearer non-claims
- runtime and proof maturity:
  - reviewed runtime concentration and runtime-control complexity are reduced
    where they materially affect product credibility
- package/platform truthfulness:
  - the package/build story remains honest and gets cleaner without fake
    symmetry or fake ABI promises
- maintainability/coherence:
  - the largest mixed-role source and proof owners, chronology residue, and
    duplicated topology surfaces are smaller and easier to navigate

## First-Pass Claim Fence

The first-pass claim fence is now fixed:

- valid Epic 9 end-state claims may include:
  - more compressed-first
  - more competitive backend ceiling
  - broader than the Epic 8 capability surface
  - better calibrated and better compared
  - cleaner and easier to maintain
- invalid Epic 9 end-state claims remain:
  - fully generic sparse numerical platform
  - broad complex and mixed-precision maturity
  - symmetric cross-platform product parity
  - broad shared-library product maturity
  - universal best-in-class runtime or reorder quality

## Strongest Clarification

The useful Day 5 clarification is now explicit:

- Epic 9 should aim higher than "research-grade but careful"
- it should not aim so high that it starts lying about product maturity
- the right target is a bounded state-of-the-art sparse linear algebra library
  with explicit non-claims
- this means:
  - real structural product convergence first
  - real backend and capability movement next
  - calibration, coherence, and maintainability cleanup in support
  - no fake broadening detached from proof and implementation ownership

## Exit State

- Sprint 90 now has one bounded Epic 9 target-state contract.
- "State of the art" now has a concrete local meaning for this repo.
- Later comparison, risk-fence, review, todo, and project-plan work can
  reference one stable success-marker and claim-fence package.
