# Sprint 46 Day 5 Artifact: Shared Eigensolver Buffer Layer Batch 1

## Purpose

Land the first real Sprint 46 code batch by introducing the private reusable
eigensolver workspace/state owner and proving it in one bounded live grow-m
Lanczos path before widening the migration to thick-restart, LOBPCG,
examples, or benchmarks.

## Main Day 5 Conclusion

Sprint 46 now has a real shared eigensolver workspace/state layer, not just a
design.

That layer is intentionally narrow in this first batch:

- private internal owner:
  - `src/sparse_eigs_workspace_internal.h`
  - `src/sparse_eigs_workspace_internal.c`
- maintained build wiring:
  - `Makefile`
  - `CMakeLists.txt`
- first live adoption:
  - grow-m Lanczos inside `sparse_eigs_sym(...)`

The batch stayed within the Day 4 boundary:

- the shared layer owns contiguous storage, checked reserve logic, and typed
  view preparation
- eigensolver math kernels, convergence policy, restart policy, and
  result/reporting logic stayed in `src/sparse_eigs.c`
- thick-restart, LOBPCG, wrappers, and benchmark/example work remain later
  Sprint 46 batches

## Landed Internal Workspace Layer

### New shared owner

`sparse_eigs_workspace_t` now owns reusable backing storage for:

- double work buffers
- `idx_t` side buffers
- `int` side buffers
- cached shape/capacity metadata for:
  - `n`
  - Lanczos/restart capacity
  - block capacity

This makes the common seam capacity-centric rather than
algorithm-control-centric.

### New typed prepare helpers

The first shared helper surface now includes typed prepare helpers for:

- grow-m Lanczos
- thick-restart Lanczos
- LOBPCG

Interpretation:

- Sprint 46 does not need to widen public APIs to start getting reuse-ready
  internal structure
- the internal layer can prepare stable typed slices now and expand live
  adoption incrementally

## First Live Adoption

### Grow-m Lanczos migrated

The grow-m branch inside `sparse_eigs_sym(...)` now:

- initializes a private `sparse_eigs_workspace_t`
- prepares a typed `sparse_eigs_growm_workspace_view_t`
- binds the former local allocation bundle through shared typed slices:
  - `V`
  - `alpha`
  - `beta`
  - `v0`
  - `theta_long`
  - `subdiag`
  - `Y_long`
  - `sel_idx`
- frees the shared owner on all exit paths instead of managing the former
  grow-m heap bundle manually

### Explicit non-goals for Day 5

This batch did **not** yet migrate:

- thick-restart Lanczos call sites
- LOBPCG call sites
- public wrappers
- repeated-run benchmark/example surfaces

Interpretation:

- the right Day 5 proof was the smallest high-value live eigensolver path
- the batch proved the shared owner without broadening the migration surface too
  early

## Validation

Because `*.c` and `*.h` files changed, the required gate was:

```bash
make format
make lint
make test
```

All passed.

The stronger reviewed baseline for this shared-layer landing also passed:

```bash
make quality-review-full
```

Targeted touched-surface follow-ons also passed:

- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`

One small implementation issue surfaced during the first lint pass:

- the initial LOBPCG prepare helper formed `view->X` through a null
  `view->P_new` branch in the no-`P` case, which `cppcheck` flagged as a null
  pointer arithmetic path

That was fixed immediately, and the authoritative rerun from the top passed
fully.

## Sprint 46 Position After Day 5

The next migration order is now clearer:

1. grow-m already proves the shared owner in a live path
2. thick-restart can adopt the already-landed typed prepare seam next
3. LOBPCG can then join the reusable owner/view path
4. wrappers, benchmarks, and memory-contract closeout can follow once the main
   eigensolver families are on the shared seam

## Bottom Line

Day 5 delivered:

- the first real shared eigensolver workspace/state owner
- typed internal eigensolver workspace views
- maintained build wiring
- a successful grow-m Lanczos proof integration
- a fully green validation baseline for the touched eigensolver shared layer

That is the right bounded first code landing for Sprint 46.
