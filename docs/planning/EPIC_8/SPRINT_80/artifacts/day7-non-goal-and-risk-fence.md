# Sprint 80 Day 7: Epic 8 Non-goal and Risk Fence

## Purpose

Freeze the strongest Epic 8 execution risks and non-goals before later sprints
start spending implementation budget. This artifact turns the Day 1-6 baseline,
gap ranking, external-oracle contract, and benchmark contract into one explicit
claim fence.

## Inputs Reconciled

- `docs/planning/EPIC_8/reviews/review-codex-2026-06-18.md`
- `docs/planning/EPIC_8/reviews/todo-codex-2026-06-18.md`
- `docs/planning/EPIC_8/PROJECT_PLAN.md`
- `docs/planning/EPIC_8/SPRINT_80/PLAN.md`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- `CMakeLists.txt`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

## Approved Epic 8 Target Claims

Epic 8 may truthfully try to earn the following claims if later sprints land
the required implementation and proof:

- the library can evolve from a linked-list-first public product toward a more
  compressed-first workflow without erasing the bounded mutable shell
- the dense/backend ceiling can be raised while keeping a self-contained
  builtin fallback
- selected capability ceilings can be narrowed through bounded, proof-backed
  surface expansion
- maintained external differential proof can grow in one bounded direct-solver
  lane first
- benchmark measurability can improve without turning canonical reporting into
  a timing gate
- maintainability concentration can be reduced through bounded source and
  giant-test architecture work
- package/platform confidence can improve without pretending current reviewed
  parity is broader than it is

## Explicitly Deferred Claims

The following may become valid later, but they are not part of the first Epic 8
contract:

- broad external sparse-solver ecosystem comparison
- unsymmetric direct external-oracle work
- broad graph/reordering comparison beyond advisory context
- broad scalar-family genericity or complex-support claims
- broad platform/install parity claims across Linux, macOS, and Windows
- shared-library or dynamic-ABI maturity claims
- portable benchmark-threshold or timing-verdict claims
- whole-repo storage-model replacement in one batch

## Explicitly Prohibited Interpretations

Epic 8 must not be described as:

- a generic rewrite of the whole library
- automatic state-of-the-art parity with mature sparse packages
- proof that the project now has broad backend maturity before real backend
  acceleration lands
- proof of broad cross-platform install/export parity before maintained
  reviewed evidence exists
- proof of shared-library or dynamic-ABI maturity while the package surface
  remains static-first
- proof of broad scalar/index/capability genericity before those surfaces are
  actually widened and tested
- proof that canonical benchmark reports are pass/fail timing gates
- proof that advisory external-comparison lanes are required maintained
  dependencies

## Risk Register

| Rank | Risk | Why it is dangerous | Primary mitigation |
|---|---|---|---|
| 1 | Over-broad architecture churn | Could turn Epic 8 into an unfocused rewrite that breaks proof ownership and slows delivery. | Keep each sprint centered on one bounded ceiling with explicit support-only surfaces. |
| 2 | Fake state-of-the-art claim inflation | Could make docs and planning less truthful than the shipped code and proof surface. | Keep every claim tied to reviewed proof, maintained scripts, or bounded benchmark interpretation. |
| 3 | External dependency sprawl | Could outrun CI, packaging, and cross-platform evidence while raising maintenance cost. | Start with one bounded CHOLMOD-class correctness lane and one BLAS/LAPACK-class performance-reference lane only. |
| 4 | Platform-claim inflation | Could imply reviewed Windows/macOS parity that the workflows explicitly do not prove. | Preserve Linux as strongest reviewed truth; keep macOS supplemental and Windows CMake-subset wording explicit. |
| 5 | Benchmark-governance drift | Could turn artifact-friendly reporting into noisy pass/fail timing theater. | Preserve canonical threshold-free reporting, bounded runtime signals, and the narrow existing threshold gate split. |

## Mitigation Ordering

The mitigation order Epic 8 should preserve is:

1. Freeze the claim fence before implementation sprawl begins.
2. Attack the product/storage ceiling first, because it is still the strongest
   structural competitiveness gap.
3. Raise the dense/backend ceiling second, because performance maturity is the
   next strongest state-of-the-art blocker.
4. Expand capability and assurance only through bounded, proof-backed lanes.
5. Improve maintainability, runtime concentration, and package/platform
   maturity without reopening disallowed claims.

## Day 7 Exit State

Sprint 80 now has one explicit Epic 8 non-goal and risk fence:

- approved claims are separated from deferred claims
- prohibited interpretations are fixed directly in writing
- the highest-value execution risks are ranked with ordered mitigations

Later Sprint 80 review/todo refinement work can now build against one stable
fence instead of reopening what Epic 8 is allowed to claim.
