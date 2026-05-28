# Sprint 46 Day 8 Artifact: LOBPCG Workspace Migration Batch

## Purpose

Migrate the remaining primary eigensolver-family workspace target, LOBPCG, onto
the shared reusable internal owner/view seam while preserving existing
one-shot/public behavior and keeping the batch bounded away from wrapper,
benchmark, and closeout churn.

## Main Day 8 Conclusion

Sprint 46 now has all three primary eigensolver families on the shared reusable
workspace seam:

- grow-m Lanczos from Day 5
- thick-restart Lanczos from Day 6
- LOBPCG from Day 8

The Day 8 batch stayed intentionally bounded:

- migrated the LOBPCG RR-step temporary bundle
- migrated the LOBPCG outer-loop block bundle
- preserved current public one-shot behavior
- did not widen into wrapper cleanup, benchmark work, or public API changes

## Landed Migration

### `s21_lobpcg_rr_step(...)` now consumes a typed shared workspace view

The RR-step path no longer allocates and frees its own per-call bundle for:

- `Q`
- `AQ`
- `G`
- `Y`
- `theta_full`
- `sel_idx`
- `X_new`
- optional `P_new`

Instead, the function now receives those slices through
`sparse_eigs_lobpcg_workspace_view_t`.

Interpretation:

- the RR-step remains algorithm-local in control flow
- repeated heap churn is now owned by the shared workspace seam rather than by
  the RR-step itself

### `s21_lobpcg_solve(...)` now prepares and owns one reusable shared owner

The outer-loop path now:

- initializes a private `sparse_eigs_workspace_t`
- prepares a typed `sparse_eigs_lobpcg_workspace_view_t`
- binds the former outer-loop block bundle through shared slices:
  - `X`
  - `R`
  - `W`
  - `P`
  - `AX`
  - `theta`
  - `converged`
- frees the shared owner on all exits instead of manually freeing the former
  per-call bundle

Interpretation:

- the last main LOBPCG repeated-allocation region is now on the shared seam
- outer-loop ownership is now aligned with the Lanczos-family migration pattern

### First-iteration `P` semantics stayed explicit after the migration

Day 8 intentionally preserved the current first-iteration behavior without
relying on lazy heap allocation.

The migrated path now uses explicit control:

- `have_p` in the outer loop
- `use_p` in the RR-step path

Interpretation:

- the first-iteration “no previous search directions yet” contract remains
  explicit and readable
- reuse semantics are no longer tied to ad hoc allocation timing

## Helper-Layer Scope

### What Day 8 widened

The existing LOBPCG view model now also carries the persistent `P` slice needed
by the migrated outer loop.

### What Day 8 did **not** redesign

This batch did **not** reopen the broader helper-layer architecture:

- no new family beyond LOBPCG was added
- no new public API was introduced
- no new benchmark-specific helper layer was added

Interpretation:

- the helper-layer change was narrow and directly justified by the live LOBPCG
  migration
- Sprint 46 did not drift back into design churn after the Day 7 audit

## Batch Scope

### What Day 8 completed

- the remaining direct LOBPCG workspace migration target
- shared-owner adoption through
  `sparse_eigs_workspace_prepare_lobpcg(...)`
- completion of the primary eigensolver-family shared-workspace landing set

### Explicit non-goals for Day 8

This batch did **not** yet widen into:

- wrapper/public dispatch cleanup
- repeated-run benchmark work
- example/tutorial refresh
- maintainer memory-behavior closeout

Interpretation:

- the right Day 8 proof was to finish the final main family migration before
  shifting into evidence and closeout work

## Validation

Because `*.c` and `*.h` changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

The stronger reviewed baseline for this shared-layer/final-family migration
batch also passed:

```bash
make quality-review-full
```

Targeted touched-surface follow-ons also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`
- `./build/example_eigs`

## Sprint 46 Position After Day 8

The remaining Sprint 46 order is now clearer:

1. the shared owner already covers all three primary eigensolver families
2. wrapper/public compatibility review can now proceed from a fully migrated
   internal baseline
3. repeated-run benchmark evidence can compare from a stable migrated state
4. maintainer memory-contract closeout can summarize the final shared-workspace
   ownership model

## Bottom Line

Day 8 delivered:

- a workspace-backed LOBPCG RR-step path
- a workspace-backed LOBPCG outer loop
- explicit first-iteration `P` semantics without lazy heap ownership
- completion of the primary eigensolver-family workspace migration set
- a fully green reviewed validation baseline for the touched eigensolver paths

That is the right bounded Day 8 migration batch for Sprint 46.
