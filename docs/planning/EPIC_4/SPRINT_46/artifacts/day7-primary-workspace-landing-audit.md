# Sprint 46 Day 7 Artifact: Primary Workspace Landing Audit

## Purpose

Audit the post-Day-6 eigensolver state so Sprint 46's remaining queue is driven
by the live code after the shared Lanczos workspace landings rather than by the
broader pre-implementation plan labels.

## Main Day 7 Conclusion

After Day 6, Sprint 46 no longer has a generic “eigensolver workspace
migration” queue.

It now has three distinct buckets:

- primary families already on the shared workspace seam:
  - grow-m Lanczos
  - thick-restart Lanczos
- one real remaining direct workspace migration target:
  - LOBPCG
- later follow-on surfaces:
  - one-shot wrapper/public dispatch edges
  - repeated-run benchmark evidence
  - maintainer memory-behavior closeout

That is the important Day 7 narrowing.

## Post-Day-6 State by Eigensolver Family

### 1. Grow-m Lanczos

Now on the shared workspace seam:

- `sparse_eigs_sym(...)` grow-m branch
- `sparse_eigs_workspace_prepare_growm(...)`
- `sparse_eigs_growm_workspace_view_t`

Interpretation:

- grow-m is no longer a remaining Sprint 46 migration target
- it remains the simpler proof path that Day 5 already closed

### 2. Thick-restart Lanczos

Now on the shared workspace seam:

- `s21_thick_restart_outer_loop(...)`
- `sparse_eigs_workspace_prepare_thick_restart(...)`
- `sparse_eigs_thick_restart_workspace_view_t`

Still family-local by design:

- `lanczos_restart_state_t`
- restart assembly / lock-selection choreography
- local recurrence helper scratch inside `lanczos_thick_restart_iterate(...)`

Interpretation:

- the main thick-restart repeated bundle is no longer a remaining Sprint 46
  workspace target
- the retained local state is the right kind of solver-family-local ownership,
  not a missed migration

### 3. LOBPCG

Still outside the shared seam:

- `s21_lobpcg_rr_step(...)`
- `s21_lobpcg_solve(...)`

But the current helper layer already contains the intended landing seam:

- `sparse_eigs_workspace_prepare_lobpcg(...)`
- `sparse_eigs_lobpcg_workspace_view_t`

Interpretation:

- LOBPCG is the honest next migration target
- Day 8 does not need another helper redesign before it starts

## Remaining Allocation Churn

### Strongest remaining direct target

LOBPCG still owns the clearest live repeated-allocation bundles.

In `s21_lobpcg_rr_step(...)`:

- `Q`
- `AQ`
- `G`
- `Y`
- `theta_full`
- `sel_idx`
- `X_new`
- optional `P_new`

In `s21_lobpcg_solve(...)`:

- `X`
- `R`
- `W`
- `AX`
- `theta`
- `converged`
- lazily allocated `P`

Interpretation:

- this aligns directly with the already-landed LOBPCG prepare helper
- Day 8 should migrate these bundles first

### Lower-priority or keep-local helper allocations

- `lanczos_iterate_op(...)` local `w`
- `lanczos_thick_restart_iterate(...)` local `w`
- `s29_refine_eigenpairs(...)` local `Av` / `y`
- `s21_arrowhead_to_tridiag(...)` dense helper scratch
- `lanczos_restart_state_t` owned buffers

Interpretation:

- these are not the best next Sprint 46 migration target
- they are either solver-local by design or smaller helper scratch rather than
  the main repeated bundle

## Wrapper vs Real Workspace Targets

### Real workspace migration target

- LOBPCG:
  - `s21_lobpcg_rr_step(...)`
  - `s21_lobpcg_solve(...)`

### Mostly wrapper/composition or later proof surfaces

- `sparse_eigs_sym(...)` as the public one-shot compatibility entry point
- backend AUTO/explicit dispatch
- `benchmarks/bench_eigs.c`
- `examples/example_eigs.c`

Interpretation:

- Day 8 should reduce real repeated allocation churn, not shift wrapper or
  benchmark code around early

## Internal Helper / Header Cleanup Notes

The current private helper surface already contains:

- `sparse_eigs_workspace_prepare_growm(...)`
- `sparse_eigs_workspace_prepare_thick_restart(...)`
- `sparse_eigs_workspace_prepare_lobpcg(...)`

Live adoption is now asymmetric in a useful way:

- grow-m and thick-restart are already on the seam
- LOBPCG prepare support exists ahead of live adoption

Interpretation:

- no new internal-header redesign is needed before Day 8
- the main cleanup rule is to avoid widening the helper surface further until
  LOBPCG actually lands on it

## Confirmed Day 8 Target Set

Day 8 should be bounded to:

1. `s21_lobpcg_rr_step(...)`
2. `s21_lobpcg_solve(...)`
3. adoption of `sparse_eigs_workspace_prepare_lobpcg(...)`
4. preservation of current one-shot public behavior
5. no wrapper/benchmark churn unless something truly trivial falls out

## Wrapper / Benchmark Follow-On Notes

After LOBPCG lands, the remaining Sprint 46 queue should sequence as:

1. wrapper/public compatibility review
2. repeated-run benchmark evidence
3. maintainer memory-contract closeout

That keeps the sprint focused on measurable repeated-allocation reduction before
it shifts into comparison/closeout proof.

## Bottom Line

Day 7 confirms:

- the primary Lanczos-family workspace migration is complete
- LOBPCG is the only strong remaining direct workspace target
- wrapper/public dispatch and benchmark/example work are later follow-ons
- no new helper redesign is needed before Day 8

That is the right narrowed handoff for the next Sprint 46 batch.
