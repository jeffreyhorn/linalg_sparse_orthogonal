# Sprint 70 Day 8: Public-Surface and Proof-Surface Audit II

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Convert the Day 7 contradiction ranking into one explicit Epic 7 cleanup fence
so later work starts from the highest-value public, header, and proof lanes
instead of reopening every documentation or giant-test surface at once.

## Refined Cleanup Ranking

Re-ranking the Day 7 surfaces against:

- user-facing value
- contradiction density
- likelihood of durable simplification
- risk of reopening already-stabilized Epic 6/69 ownership work

produces the following later cleanup order:

1. public-product cleanup:
   - `README.md`
   - `INSTALL.md`
2. header/reference cleanup:
   - `include/sparse_cholesky.h`
3. proof-surface cleanup:
   - `tests/test_reorder_nd.c`
   - `tests/test_chol_csc.c`

## Why README + INSTALL Are the Public Cleanup Pair

The strongest later public cleanup lane is the front-door plus install pair:

- `README.md`
- `INSTALL.md`

Why this pair is stronger than a broader multi-doc batch:

- README still carries the densest mixed public-story burden
- INSTALL remains the strongest operator/support drift surface
- both are high-signal user-facing surfaces where simplification should have a
  durable product payoff
- widening immediately into tutorial/examples/benchmarks risks reopening
  already-bounded ownership surfaces for lower value

## Why the Header Lane Should Stay Narrow

The strongest header/reference target remains:

- `include/sparse_cholesky.h`

Why the lane should stay narrow:

- it concentrates the strongest mixed reference burden in one place
- broadening immediately to all public headers would turn a ranked cleanup seam
  into generic header churn
- the other public headers do not currently present the same contradiction
  density

So the correct next header/reference reading is:

- one exact first header candidate
- broader header cleanup only if later work proves it is justified

## Why the Proof Lane Is ND-First

The proof-surface queue is now explicit:

1. first giant-test cleanup candidate:
   - `tests/test_reorder_nd.c`
2. strongest second giant-test cleanup candidate:
   - `tests/test_chol_csc.c`
3. support only if later cleanup truly needs it:
   - `tests/test_integration.c`
   - `tests/test_fuzz.c`

Why this ordering holds:

- `tests/test_reorder_nd.c` still carries the densest live chronology burden
- `tests/test_chol_csc.c` remains large and history-heavy, but with clearer
  family-local ownership
- integration and fuzz remain important proof owners, but they are not the
  strongest current contradiction centers

## Support-Surface Fence

Sprint 70 Day 8 fixes the likely later support-surface fence as:

Likely future public-product cleanup:

- `README.md`
- `INSTALL.md`

Likely future header/reference cleanup:

- `include/sparse_cholesky.h`

Likely future giant-test cleanup:

- `tests/test_reorder_nd.c`
- `tests/test_chol_csc.c`

Support only if a later batch truly forces it:

- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- `tests/test_integration.c`
- `tests/test_fuzz.c`

This is the key Day 8 clarification:

- the remaining Epic 7 cleanup problem is no longer "all docs and all giant
  tests are too big"
- it is now a bounded set of likely lanes with explicit support-only surfaces

## Explicit Non-Goals

The later cleanup fence explicitly does not imply:

- a repo-wide rewrite of every public doc surface in one sprint
- broad public-header cleanup without a ranked header center
- giant-test breakup across every proof owner at once
- reopening benchmark-governance or maintainer-policy authority just to
  simplify wording elsewhere
- mixing public cleanup, proof cleanup, and platform/install contract work into
  one generic "documentation pass"

## Exit State

Sprint 70 Day 8 closes with one explicit cleanup fence:

1. public-product cleanup:
   - `README.md`
   - `INSTALL.md`
2. header/reference cleanup:
   - `include/sparse_cholesky.h`
3. proof-surface cleanup:
   - `tests/test_reorder_nd.c`
   - `tests/test_chol_csc.c`
4. support-only unless later work proves otherwise:
   - tutorial / examples / benchmarks / maintainer guide
   - integration / fuzz

That gives Day 9 one exact job:

- move from surface ranking to the validation/platform contract audit without
  reopening the newly fixed public/proof cleanup fence
