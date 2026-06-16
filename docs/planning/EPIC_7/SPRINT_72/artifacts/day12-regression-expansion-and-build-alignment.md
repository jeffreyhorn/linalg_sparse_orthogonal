# Sprint 72 Day 12: Regression Expansion and Build Alignment

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Tighten the maintained proof/reference ownership surface around the Sprint 72
product-model boundary without inventing extra regression churn where the live
tests already cover the landed behavior.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_72/artifacts/day11-public-contract-and-example-adoption-batch.md`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`
- `docs/maintainer_guide.md`

## Day 12 Alignment Results

### 1. No new regression code was required

The reread against the post-Day-11 state shows that the landed Sprint 72
boundary already has focused proof in the strongest existing homes:

- `tests/test_integration.c` owns the Day 6 matrix-shell reset boundary:
  `sparse_reset_perms()` invalidates stale reordered one-shot solve
  compatibility and recovers a plain matrix shell
- `tests/test_chol_csc.c` owns the Day 9 family-local Cholesky CSC publish-back
  boundary: writeback produces a solve-ready shell with the expected reorder
  payload and identity internal row/column permutation shells

That means Sprint 72 Day 12 should not add redundant proof in weaker or more
generic surfaces.

### 2. The real gap was maintained proof-ownership wording

The strongest remaining mismatch was the policy/reference surface:

- `docs/maintainer_guide.md` still described the current maintained proof map
  only through the older Sprint 68 Cholesky lifecycle lanes
- it did not yet name the new Sprint 72 proof owners for the matrix-shell
  reset boundary and the Cholesky CSC publish-back ownership boundary

The Day 12 batch therefore lands as a docs-only alignment pass in the
maintainer guide.

### 3. The maintained proof map now includes the Sprint 72 ownership boundary

The landed `docs/maintainer_guide.md` update now states directly:

- `tests/test_chol_csc.c` owns the family-local Cholesky CSC publish-back
  ownership proof surface
- `tests/test_integration.c` owns the matrix-shell reset boundary through
  `sparse_reset_perms()`

This keeps the maintainer policy aligned with the actual implementation and
regression surfaces now carrying the Sprint 72 boundary.

### 4. Broader support surfaces remained coherent and did not move

The reread confirmed that these support surfaces already match the landed
ownership story and therefore remained untouched:

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`

## Validation

This was a docs-only Day 12 alignment pass, so no `make format`, `make lint`,
or `make test` rerun was required.

Targeted sanity checks passed:

- touched-surface diff review
- terminology/alignment `rg`
- touched-surface `wc -l`
- branch-status recheck

Touched-surface raw `wc -l` count:

- `docs/maintainer_guide.md` = `585`

## Exit State

Sprint 72 Day 12 closes with:

1. one explicit confirmation that no new regression code was needed
2. one maintained proof-ownership update in `docs/maintainer_guide.md`
3. one final touched proof surface fixed to:
   - `tests/test_integration.c`
   - `tests/test_chol_csc.c`
4. one preserved non-move of the broader README/tutorial/example/benchmark
   support surfaces
