# Sprint 57 Day 6 - post-Day-5 direct-test seam map

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Re-audit the direct-solver giant-test queue from the landed Day 5 state and
decide which remaining seam is actually worth pursuing versus which direct
proof surfaces should intentionally stay dense while Sprint 57 pivots to the
solver-family giant tests and later lifecycle coverage.

## Landed-state baseline

Direct-solver giant-test sizes from the live branch:

- `tests/test_chol_csc.c` = `4552`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_integration.c` = `1803`

Day 5 already landed:

- `tests/test_chol_csc_supernodal_helpers.h` = `96`

That means the direct-test queue must be judged from the post-extraction
ownership state, not from the pre-Day-5 line-count ranking alone.

## Main finding

The highest-value direct-solver follow-through seam is no longer another
immediate `test_chol_csc.c` move.

After Day 5:

- `test_chol_csc.c` is still large, but it is now mainly dense proof-body
  territory
- the largest accidental helper-clutter seam in that file is already gone
- the cleaner remaining helper-heavy seam is now in `test_ldlt_csc.c`

This changes the queue shape materially.

## Post-Day-5 `test_chol_csc.c` assessment

The Cholesky giant test still contains large sections, but they are now mostly
cohesive proof groups:

- supernode detection + postorder corpus-safety proof
- dense/supernodal elimination proof
- writeback proof
- dispatch proof

The Day 5 extraction already moved the main local helper layer out:

- `detect_supernodes_alloc(...)`
- `day8_count_supernodes(...)`
- `day9_assert_batched_matches_scalar(...)`
- `day11_build_spd(...)`

Maintainability conclusion:

- another immediate Cholesky helper peel would now have lower value
- keeping the remaining Cholesky proof groups in one file is still reasonable
  for Sprint 57
- `test_chol_csc.c` is intentionally dense after Day 5 rather than
  accidentally cluttered

## `test_ldlt_csc.c` as the best remaining direct seam

The strongest remaining direct-solver helper-heavy region is the front-loaded
LDLT CSC cluster that combines:

- 2x2-aware supernode detection
- supernode extract / writeback round-trips
- batched supernodal cross-checks
- analysis-aware indefinite two-pass workflow proof

The most obvious local helper seam is centered on:

- `build_dense_ldlt_with_pivots(...)`
- `snapshot_supernode_state(...)`
- `ldlt_csc_factor_state_matches(...)`
- `build_kkt_5x5(...)`
- `build_kkt_10x10(...)`
- `s20_two_pass_indefinite_factor(...)`
- `s20_solve_residual(...)`

If Sprint 57 needed a second direct-solver test refactor, the best next owned
boundary would now be a local helper header for this LDLT CSC proof family.

## `test_integration.c` should stay intact

`test_integration.c` is not the right direct-test follow-through target:

- it is smaller than the CSC giant tests
- it is the highest-value cross-family caller-story surface
- Sprint 57 still needs it later for:
  - lifecycle sequencing coverage
  - factor-many / refactor failure preservation
  - one-shot versus repeated-run parity

Maintainability conclusion:

- keep `test_integration.c` intact
- do not split or normalize it as part of the direct-test follow-through
- use it later as the central regression-expansion surface

## Updated landing boundary

### Completed and sufficient for Sprint 57 direct Cholesky work

- `tests/test_chol_csc.c`
  - helper extraction already landed
  - no second immediate follow-through patch required

### Best remaining direct-test seam, but deferred

- `tests/test_ldlt_csc.c`
  - likely future seam:
    - local LDLT CSC helper header for the supernode / analysis-aware
      indefinite proof family

### Intentionally preserved as dense

- `tests/test_integration.c`
  - reserved for later lifecycle and factor-many coverage batches

## Why the LDLT seam is deferred anyway

Even though LDLT CSC is now the best remaining direct-test seam, Sprint 57
should still defer it after Day 6 because:

- Day 5 already delivered a real direct-solver maintainability win
- the sprint still needs to cover the large iterative/eigensolver test surfaces
- the highest-value remaining functional work is the later lifecycle and
  factor-many regression expansion

So the correct Day 6 outcome is to make the defer explicit, not to force a
second direct-solver patch merely because a seam exists.

## Conclusion

Day 6 closes the direct-solver refactor follow-through queue cleanly:

- Cholesky CSC got the intended first maintainability seam
- the remaining Cholesky density is acceptable and mostly proof-body density
- LDLT CSC is now the best remaining direct-test seam, but explicitly deferred
- integration stays intact for later lifecycle and factor-many proof work

That leaves Sprint 57 ready to pivot to Day 7's solver-family giant-test audit
without ambiguity about the direct-test queue.
