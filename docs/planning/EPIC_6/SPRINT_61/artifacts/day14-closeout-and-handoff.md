# Sprint 61 Day 14: Closeout and Handoff

Date: 2026-06-09
Branch: sprint-61

## Purpose

Package Sprint 61 into a clean Phase 1 configuration-modernization handoff so
later Epic 6 work can build on the landed typed control surface without
reopening Sprint 60 architecture/validation contracts or Sprint 61 precedence
and compatibility decisions.

## Sprint 61 Closed Outcomes

Sprint 61 now hands off one coherent Phase 1 configuration package across:

- Sprint 61 scope and validation-baseline freeze
- ranked env-var/control inventory
- typed-options and precedence design
- reorder/ND typed-option integration
- analysis/postorder typed-option integration
- compatibility/default regression tightening
- docs and maintainer-story follow-through
- full Day 13 validation sweep

The strongest concrete landed outcomes are:

- public typed `sparse_analysis_opts_t.reorder_opts` coverage for:
  - `SPARSE_SUPERNODAL_POSTORDER`
  - `SPARSE_ND_ROOT_BISECT`
  - `SPARSE_ND_ROOT_BISECT_MAX_N`
  - `SPARSE_ND_COARSENING`
  - `SPARSE_ND_COARSEST_BISECTION`
  - `SPARSE_ND_SEP_LIFT_STRATEGY`
  - `SPARSE_ND_SEP_LIFT_WEIGHT`
  - `SPARSE_ND_COARSEN_FLOOR_RATIO`
- internal typed-policy ownership for:
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- explicit precedence:
  1. explicit typed value
  2. legacy compatibility env var when unspecified
  3. internal default
- preserved public compatibility wrapper:
  - `sparse_reorder_nd(...)` stayed API-stable

## Final Frozen Sprint 61 Contract

Sprint 61 closes with the following rules fixed explicitly:

- `sparse_analysis_opts_t.reorder_opts` is the preferred advanced
  analysis/reorder control surface
- explicit typed values win over env-var compatibility inputs
- legacy env vars remain compatibility overrides only when the typed field is
  left unspecified
- `SPARSE_ND_SUPERNODAL_POSTORDER` remains compatibility-only
- `SPARSE_ND_COARSENING_CV_FALLTHROUGH` remains internal-policy-only
- debug/profile controls remain deferred and non-public:
  - `SPARSE_ND_PROFILE`
  - `SPARSE_QG_PROFILE`
  - `SPARSE_HCC_DEBUG`
- the `SPARSE_FM_*` family remains deferred for later Epic 6 work
- the repeated-run workflow fence from Sprint 60 remains untouched

## Sprint 61 Validated Close Baseline

Sprint 61 closes from the Day 13 validated baseline:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed
- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity stayed `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- reviewed CMake total time from `make quality-review-full`:
  - `368.17 sec`

Targeted workflow-proof follow-ons also passed:

- direct lifecycle and CSC:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- graph/reorder-sensitive proof:
  - `./build/test_graph`
  - `./build/test_graph_fm_buckets`
  - `./build/test_reorder_nd`
  - `./build/test_reorder_amd_qg`
- adjacent repeated-run solver proof:
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
- representative examples and benchmarks:
  - `./build/example_analysis`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

## Sprint 62+ Starting Queue

The remaining configuration queue after Sprint 61 is now smaller and more
concrete:

1. later FM-family control rationalization:
   - `SPARSE_FM_*`
2. later debug/profile control treatment:
   - `SPARSE_ND_PROFILE`
   - `SPARSE_QG_PROFILE`
   - `SPARSE_HCC_DEBUG`
3. compatibility-only alias cleanup only if later work justifies it:
   - `SPARSE_ND_SUPERNODAL_POSTORDER`
4. later broader configuration/policy rationalization outside the Sprint 61
   analysis/reorder seam

The immediate Epic 6 handoff priority remains:

1. direct-solver usability and lifecycle coherence
2. later configuration Phase 2 only where it stays compatible with the landed
   Phase 1 model
3. later backend/AUTO policy rationalization

## PROJECT_PLAN Check

`docs/planning/EPIC_6/PROJECT_PLAN.md` does not need a Sprint 61 correction.

The landed sprint still matches the project-plan intent:

- replace the highest-value process-global analysis/reorder controls with
  typed options
- make precedence explicit
- preserve bounded compatibility behavior
- add regression/docs support
- close from a fully validated baseline

## Preserved Non-Goal Fence

Sprint 61 closes without widening Epic 6 beyond its intended Phase 1 scope:

- no public FM-family tuning surface
- no debug/profile migration into the public API
- no repo-wide configuration helper layer
- no backend/AUTO rewrite in the same sprint
- no packaging/platform spillover disguised as configuration work
- no reopening of the repeated-run workflow fence

## Day 14 Exit State

Sprint 61 is now closed from a validated and bounded Phase 1
configuration-modernization package.

Sprint 62+ can build from:

- a real typed analysis/reorder control surface
- an explicit precedence and compatibility contract
- a smaller and more intentional deferred env-var queue
- a reviewed local validation baseline that remained exact through the final
  landed tree
