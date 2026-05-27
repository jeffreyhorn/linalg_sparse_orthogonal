# Sprint 42 Day 10 Artifact: Cancellation & Mutation Contract Normalization

## Scope

Day 10 implemented the bounded contract-normalization batch chosen in the
Sprint 42 plan:

- direct LU cancellation / pre-mutation failure cleanup
- direct and CSC Cholesky state-entry timing cleanup
- analyze-once bridge output-commit cleanup
- focused lifecycle-contract regression coverage

The batch stayed intentionally narrow:

- normalize touched lifecycle semantics where the new internal seams already
  make that safe
- avoid public-handle redesign
- avoid broad documentation churn
- avoid unrelated factor-family work

This is contract cleanup across touched lifecycle paths, not a new lifecycle
phase.

## Delivered Code Changes

### 1. Added private compatibility-state restore support

Touched files:

- `src/sparse_matrix_internal.h`
- `src/sparse_factor_state_internal.c`

Day 10 changes:

- private factor-state payloads now snapshot:
  - previous `factored`
  - previous `factor_norm`
- added:
  - `sparse_factor_state_restore_compat(...)`

Result:

- the private factor-state seam can now restore compatibility mirrors on safe
  pre-mutation exits
- this gives Sprint 42 a bounded way to reduce cancellation drift without
  pretending every path can become bit-identical after mutation has already
  started

### 2. LU now restores compatibility mirrors on pre-mutation exits

Touched file:

- `src/sparse_lu.c`

Day 10 changes:

- immediate cancel before any inner-path mutation now restores the pre-entry
  compatibility mirrors
- pre-mutation singular/error exits do the same
- in-source cancellation comments now state the real contract:
  - restore mirrors when the inner factor path has not yet mutated the matrix
  - do not undo any earlier reorder done by `sparse_lu_factor_opts(...)`

Result:

- the direct LU path no longer leaves the private/public factored-state seam
  drifted on early exits that happen before any actual factor-body mutation

### 3. Cholesky now enters factor-state mode later

Touched file:

- `src/sparse_cholesky.c`

Day 10 changes:

- linked-list Cholesky now delays `sparse_factor_state_begin_cholesky(...)`
  until after:
  - symmetry validation
  - norm capture
  - local workspace allocation
- CSC Cholesky now delays `sparse_factor_state_begin_cholesky(...)` until after
  successful:
  - symbolic analysis
  - CSC conversion
  - CSC numeric elimination
- linked-list cancellation comments now say directly that the upper triangle is
  stripped before the first callback emission

Result:

- Cholesky still keeps its load-bearing in-place mutation semantics
- but preparatory failures no longer enter the factor-state seam earlier than
  necessary

### 4. The analyze-once bridge now commits factors only on success

Touched file:

- `src/sparse_analysis.c`

Day 10 changes:

- `sparse_factor_numeric(...)` now builds a local `new_factors` object first
- the caller-visible `sparse_factors_t` is freed/replaced only after full
  success

Result:

- bridge output mutation is now success-only
- failed analyze-once numeric factorization no longer leaves a partially
  rewritten factor object behind
- this aligns the bridge better with the already-success-only shape of
  `sparse_refactor_numeric(...)`

### 5. Focused lifecycle regressions now exercise the contract directly

Touched files:

- `tests/test_integration.c`
- `tests/test_etree.c`

Day 10 additions:

- LU cancel-at-step-0 regression now asserts the cancelled matrix is rejected
  by solve
- Cholesky cancel-at-step-0 regression now asserts the cancelled matrix is
  rejected by solve
- analyze-once failure regression now proves old factors remain usable after a
  failed replacement attempt

Result:

- the touched lifecycle contract is now encoded in tests rather than only
  inferred from comments

## What Stayed Intentionally Unchanged

Day 10 did **not** attempt:

- public `sparse_factors_t` redesign
- broad header/tutorial/README lifecycle rewrite
- QR / SVD / LDLT cancellation cleanup
- full reorder rollback
- full bit-identical restoration after already-mutating paths

That keeps the batch aligned with the Sprint 42 Day 10 contract:

- normalize touched lifecycle semantics where safe
- record later-phase work without reopening broad public churn

## Compatibility Boundary Preserved

The important Sprint 40 / Sprint 42 rules still hold:

- public APIs are unchanged
- installed bridge/object shapes are unchanged
- LU and Cholesky remain caller-facing matrix-centered routines
- Cholesky linked-list cancellation still does not promise a bit-identical
  matrix after the upper-triangle strip
- analyze-once still uses the same public `sparse_factors_t` bridge

The Day 10 change is that the touched internal lifecycle semantics are now more
consistent and more explicitly tested.

## Validation

Because `*.c` / `*.h` changed, the required full gate was run:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

## Day 10 Outcome

Sprint 42 now has a more coherent touched lifecycle contract:

- LU restores compatibility-state mirrors on safe early exits
- Cholesky enters the factor-state seam closer to the real mutation boundary
- the analyze-once bridge commits output only on success
- focused tests assert the touched cancellation and failure semantics directly

That is the intended Day 10 handoff into the later Sprint 42 compatibility and
focused-test planning work:

- compatibility wrapper and bridge design
- additional focused lifecycle misuse coverage only where still needed
