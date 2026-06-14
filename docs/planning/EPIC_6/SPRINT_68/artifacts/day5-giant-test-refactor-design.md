# Sprint 68 Day 5: Giant-Test Refactor Design

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Turn the Day 4 first-landing boundary into one explicit ownership and
helper-extraction contract so the first Sprint 68 implementation batch stays
bounded to the highest-value `test_chol_csc` maintenance seam.

## First-Landing Ownership Contract

The first landing remains fixed to:

- `tests/test_chol_csc.c`

But Day 5 now makes the intended durable ownership more explicit.

`tests/test_chol_csc.c` should converge toward:

- one canonical family-local Cholesky CSC proof owner
- explicit scenario assertions and proof bodies in the main file
- a narrower supernodal/writeback/dispatch support tail

So the first landing is not a broad test-suite redesign. It is a bounded
ownership extraction inside the largest remaining giant test.

## Keep One Canonical Test Owner, Not Multiple New Binaries

The live file is large, but the first batch should not split it into several
new `tests/test_chol_csc_*.c` binaries.

Why that stays out:

- the current file is already the clear family-local owner for CSC Cholesky
  behavior
- late proof sections share local builders, comparison helpers, and
  internal-family context
- multiplying binaries immediately would widen into build-list churn, runner
  churn, and proof-ownership ambiguity

Design consequence:

- keep one canonical `test_chol_csc` binary
- reduce local maintenance pressure by extracting bounded support helpers, not
  by multiplying permanent owners

## Extract the Supernodal/Writeback/Dispatch Support Lane, Not the Scalar/Core Proof Lane

The live file still owns two different categories of logic.

Durable proof sections that should stay in `tests/test_chol_csc.c`:

- CSC alloc/grow/conversion/validation proof
- scalar elimination and solve proof
- scenario bodies and final assertions

Support-heavy seams that are better extraction candidates:

- supernode detection allocation helpers
- scalar-vs-batched factored CSC comparison helpers
- large SPD fixture builders for dispatch/backend-path checks
- repetitive writeback round-trip scaffolding

Current support helpers already living in the family-local header:

- `detect_supernodes_alloc(...)`
- `day8_count_supernodes(...)`
- `day9_assert_batched_matches_scalar(...)`
- `day11_build_spd(...)`

Likely next helper candidates if Day 6 needs them:

- `day8_chol_csc_match(...)`
- `day7_chol_csc_get(...)`
- `day10_factored_matches(...)`
- `day10_roundtrip_check(...)`

Design consequence:

- keep proof bodies in the main file
- move only bounded family-local support scaffolding where that materially
  clarifies the supernodal/writeback/dispatch tail

## Keep the Runner Surface Explicit

The giant `RUN_TEST(...)` tail is part of the maintenance burden, but the
first landing should not replace it with opaque registration indirection.

The safe first-batch contract is:

- keep one explicit `RUN_TEST(...)` owner in `tests/test_chol_csc.c`
- allow bounded regrouping only where helper extraction would otherwise make
  chronology harder to follow
- avoid new macro-driven or data-driven runner abstractions

## Exact Day 6-7 Touched-File Fence

Required first-batch implementation surface:

- `tests/test_chol_csc.c`

Support only if the landed extraction truly needs it:

- `tests/test_chol_csc_supernodal_helpers.h`

Proof/support surfaces that stay out unless the landed refactor unexpectedly
moves ownership wording:

- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`

This keeps the first implementation batch family-local by default and leaves
oracle/docs widening explicitly conditional.

## Explicit Non-Touch Set

The first giant-test landing should not widen into:

- new `tests/test_chol_csc_*.c` binaries
- shared cross-family test helper layers
- `tests/test_integration.c`
- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- implementation `src/` files
- benchmark or maintained-doc truth surfaces

That non-touch set matters because Sprint 68 still has real later lanes after
the first refactor landing:

- ND chronology follow-through
- oracle/parity expansion
- property/fuzz expansion
- platform-confidence follow-through

## Exit State

Sprint 68 Day 5 closes with one exact first implementation contract:

1. required first batch:
   - `tests/test_chol_csc.c`
2. support only if needed:
   - `tests/test_chol_csc_supernodal_helpers.h`
3. keep as durable owner:
   - one canonical `test_chol_csc` binary
   - scenario assertions and proof bodies in the main file
4. likely extraction lane:
   - supernodal/writeback/dispatch support helpers only
5. explicit non-touch set:
   - oracle lane
   - other giant tests
   - implementation files
   - benchmark/docs truth surfaces

That gives Day 6 one exact job:

- land one bounded `test_chol_csc.c` helper-extraction batch without widening
  into a broader test-suite redesign
