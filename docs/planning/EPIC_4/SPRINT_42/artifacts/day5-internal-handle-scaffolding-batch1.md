# Sprint 42 Day 5 Artifact: Internal Handle Scaffolding Batch 1

## Scope

Day 5 landed the first real internal lifecycle-handle scaffolding batch for the
Sprint 42 LU / Cholesky seam. The batch stayed deliberately narrow:

- add a private factor-state payload seam to `SparseMatrix`
- wire matrix copy / free / invalidation paths to that seam
- migrate the linked-list LU and linked-list Cholesky factor/solve paths onto
  the seam
- preserve public API shape and compatibility-visible matrix fields

This is internal ownership groundwork, not a public handle rollout.

## Delivered Code Changes

### 1. New private factor-state seam

Added a new private helper layer:

- `src/sparse_factor_state_internal.c`
- declarations in `src/sparse_matrix_internal.h`

The seam introduces:

- `sparse_factor_state_kind_t`
- `sparse_lu_factor_state_t`
- `sparse_cholesky_factor_state_t`
- `sparse_factor_state_t`
- bind / clear / clone helpers
- compatibility-preserving getters/setters for:
  - factored state
  - factor-norm cache

### 2. `SparseMatrix` private payload storage

`SparseMatrix` now carries:

- `sparse_factor_state_t *factor_state`

The public compatibility fields remain in place:

- `factored`
- `factor_norm`

Day 5 intentionally keeps those public-facing fields as compatibility mirrors
while the new internal seam becomes the preferred LU / Cholesky ownership path.

### 3. Matrix lifecycle wiring

`src/sparse_matrix.c` now:

- initializes `factor_state` on create
- frees it on matrix destruction
- clones it during `sparse_copy()`
- clears / invalidates it on touched mutation paths
- routes `sparse_mark_factored()` through the shared seam

This makes the seam a real lifecycle object instead of dead scaffolding.

### 4. LU seam adoption

`src/sparse_lu.c` now:

- binds LU payload state at factorization start
- uses shared setters for:
  - unfactored reset
  - cached factor norm
  - factored-success publication
- uses seam getters in solve / transpose-solve / condest / block-solve paths

This is still the same LU API contract. The change is internal ownership and
state publication, not a caller-visible behavior rewrite.

### 5. Cholesky seam adoption

`src/sparse_cholesky.c` now:

- binds Cholesky payload state at factorization start
- uses shared setters for:
  - unfactored reset
  - cached factor norm
  - factored-success publication
- uses seam getters in the linked-list solve path
- routes the touched reorder-path factor-norm reset through the shared seam

As with LU, this preserves the existing API contract while moving the internal
state boundary toward the Sprint 40 handle model.

## Build-System Wiring

The new helper source was added to both maintained build surfaces:

- `Makefile`
- `CMakeLists.txt`

That keeps the Day 5 seam present in both direct and reviewed CMake builds.

## Compatibility Boundary Preserved

Day 5 did **not** attempt any of the following:

- public explicit LU handle APIs
- public explicit Cholesky handle APIs
- `SparseMatrix` field removal
- `sparse_factors_t` bridge normalization
- tutorial / README public lifecycle wording changes

The batch is strictly internal-first, which matches the Sprint 40 contract and
the Sprint 42 Day 3 design boundary.

## Validation

Because `*.c` / `*.h` files changed, the required full gate was run:

- `make format`
- `make lint`
- `make test`

Authoritative result:

- all three passed

### Important validation note

The first `make test` attempt produced a false regression in
`test_writeback_roundtrip_nos4_amd` / `test_writeback_roundtrip_bcsstk04_amd`
inside `test_chol_csc`.

Root cause:

- the Day 5 batch changed the private `SparseMatrix` layout
- the local incremental build did not automatically rebuild every consumer of
  `src/sparse_matrix_internal.h`
- `src/sparse_chol_csc.c` was still linked against the pre-Day-5 layout during
  the first pass

Resolution:

- `make clean`
- rerun the full required gate from a clean tree

The clean authoritative rerun passed, including the previously failing
`chol_csc` writeback round-trip coverage. This was a local stale-object
validation issue, not a live logic regression in the Day 5 code.

## Day 5 Outcome

Sprint 42 now has a real first-phase internal handle seam:

- LU and Cholesky no longer rely exclusively on ad hoc direct publication
  through `SparseMatrix` compatibility fields
- matrix copy / free / invalidation paths understand the seam
- later Sprint 42 work can build guard helpers and compatibility bridges on top
  of a real internal ownership object instead of starting from raw matrix-field
  mutation alone

That is the intended Day 5 handoff into:

- Day 6 shared matrix-state guard helpers
- later `sparse_factors_t` bridge normalization
- later explicit-handle enrichment work
