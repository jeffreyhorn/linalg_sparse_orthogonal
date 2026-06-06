# Sprint 56 Day 6 - Cholesky CSC residual ownership audit

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Reduce `src/sparse_chol_csc.c` to a concrete extraction map before code
movement begins, using the landed LDLT CSC split as a comparison point rather
than a mechanical template.

## Live ownership bands

The current `src/sparse_chol_csc.c` function map separates into six real
ownership bands:

1. lifecycle / storage / structural conversion
   - alloc / free
   - capacity growth
   - sparse-to-CSC conversion
   - analysis-aware sparse-to-CSC conversion
   - CSC-to-sparse conversion
   - validation
2. scalar workspace and native elimination/solve core
   - workspace alloc / free
   - scatter / cmod / cdiv / gather / end-column helpers
   - left-looking scalar elimination
   - CSC solve paths
3. wrapper / dispatch-specific glue
   - `chol_csc_factor(...)`
   - `chol_csc_factor_solve(...)`
4. Cholesky-owned supernodal backend
   - fundamental supernode detection
   - dense Cholesky primitives
   - supernodal elimination driver
   - supernode extract / diag / panel / writeback helpers
5. compatibility-facing CSC writeback seam
   - `chol_csc_writeback_to_sparse(...)`
6. shared dense indefinite primitive seam
   - `ldlt_dense_sym_swap(...)`
   - `ldlt_dense_factor(...)`

Interpretation:

- the file is large, but it is no longer ambiguous
- the main question is which ownership band should move first
- Cholesky has one important family-specific wrinkle:
  - the shared dense LDLT primitive still lives here even though it is not
    Cholesky-owned in the long term

## Strongest first extraction target

The strongest first extraction target is the Cholesky-owned supernodal backend
as one coherent file-owned slice.

Recommended moved set:

- `columns_in_same_supernode(...)`
- `chol_csc_detect_supernodes(...)`
- `chol_dense_factor(...)`
- `chol_dense_solve_lower(...)`
- `chol_csc_eliminate_supernodal(...)`
- `chol_csc_bsearch_row_map(...)`
- `chol_csc_supernode_extract(...)`
- `chol_csc_supernode_eliminate_diag(...)`
- `chol_csc_supernode_eliminate_panel(...)`
- `chol_csc_supernode_writeback(...)`

Why this outranks the scalar/native kernel for Batch 1:

- it is already contiguous and internally cohesive
- it carries a clean SPD-only vocabulary
- it is the clearest line-count relief in the file
- `tests/test_chol_csc.c` already treats it like a real backend boundary
- `bench_refactor_csc.c` already names the same analysis-aware CSC completion
  seam directly

Measured ownership value:

- approximate supernodal backend candidate band:
  - `1224..2093`
  - about `870` lines

Interpretation:

- unlike the LDLT batch, Cholesky can justify moving the top-level batched
  driver together with its helper cluster
- that gives the next batch a real backend-owned file rather than a narrower
  helper spillover

## Strongest second extraction target

The scalar workspace and native elimination/solve core is the strongest second
seam.

Why it should come second:

- it is still the highest-risk numerical ownership band
- it mixes workspace lifecycle, fill-sensitive gather logic, elimination, and
  solve semantics
- it is easier to reason about once the supernodal backend is no longer sharing
  the same file

Approximate ownership mass:

- scalar/workspace/elimination/solve band:
  - `714..1223`
  - about `510` lines

## Lower-priority seams

### Lifecycle / conversion / validation

This remains real ownership, but it is a weaker first-batch target:

- more compatibility-facing
- less numerically cohesive than the supernodal backend
- more likely to produce a low-value mechanical split

### Wrapper / dispatch glue

This stays intentionally secondary:

- already bounded
- not where most of the file's maintainability weight lives
- better preserved in the main file while the heavy backend band moves first

### CSC writeback-to-sparse seam

This is a real later target, but not a first extraction seam:

- it is compatibility-facing rather than numerically central
- it sits close to the supernodal cluster physically, but it belongs to the
  transparent dispatch/public-writeback story instead

### Shared dense LDLT primitive seam

This is explicitly not the right Batch 1 target:

- `ldlt_dense_factor(...)` is used by LDLT CSC, not only by Cholesky
- moving it into a Cholesky-owned supernodal file would blur ownership rather
  than sharpen it
- if it moves later, it should move into a shared dense-factor home, not into a
  Cholesky-specific module

## LDLT comparison and intentional family differences

The landed LDLT Batch 1 is still the right comparison point, but not the exact
template.

Shared pattern with LDLT:

- supernodal backend work is still the best first seam
- keep the existing private header
- avoid mixing source extraction with private-header taxonomy redesign
- preserve public CSC wrapper/direct behavior exactly

Intentional differences from LDLT:

1. Cholesky's first seam should be wider.
   - LDLT Batch 1 kept the top-level supernodal driver in the main file.
   - Cholesky's `chol_csc_eliminate_supernodal(...)` is more cleanly part of
     the same SPD backend cluster as its detect/extract/diag/panel/writeback
     helpers.
2. Cholesky should move its dense Cholesky primitives with the supernodal
   backend.
   - `chol_dense_factor(...)` and `chol_dense_solve_lower(...)` are naturally
     owned by the same backend slice.
3. Cholesky should leave the shared dense LDLT primitive behind.
   - `ldlt_dense_factor(...)` is not Cholesky-specific despite living in this
     file.

Interpretation:

- the right Day 6 outcome is not "repeat the LDLT split"
- it is "reuse the LDLT decision logic, then choose the narrower or wider seam
  that best matches the Cholesky file's real ownership"

## Proof-surface implications

The current proof surfaces already reinforce the Cholesky backend boundary:

- `tests/test_chol_csc.c` is the primary file-level proof surface
- it directly names:
  - supernode detection
  - dense Cholesky helpers
  - supernodal elimination
  - extract / diag / panel / writeback helpers
  - writeback-to-sparse
- `benchmarks/bench_refactor_csc.c` directly exercises:
  - `chol_csc_from_sparse_with_analysis(...)`
  - `chol_csc_eliminate_supernodal(...)`
- `examples/example_analysis.c` remains the high-signal caller-facing repeated
  direct workflow proof surface

That means the best extraction seam is the one the proof surfaces already imply
exists. Utility-first slicing across those boundaries would be harder to test
and harder to defend.

## Ranked extraction order

1. Cholesky-owned supernodal backend cluster
2. scalar workspace and native elimination/solve core
3. lifecycle / conversion / validation cluster
4. CSC writeback-to-sparse seam
5. wrapper / dispatch glue
6. shared dense indefinite primitive cleanup

## Recommended first extraction boundary

Start the Cholesky half of Sprint 56 implementation work by extracting the
full Cholesky-owned supernodal backend into an owned file:

- proposed file:
  - `src/sparse_chol_csc_supernodal.c`

Keep in `src/sparse_chol_csc.c` for the first batch:

- lifecycle/conversion entry points
- scalar workspace and native elimination/solve core
- wrapper/dispatch-specific glue
- `chol_csc_writeback_to_sparse(...)`
- shared dense LDLT primitive helpers

## Conclusion

Day 6 turns the Cholesky CSC large-file problem into a concrete decomposition
map with one clear first target:

- move the full Cholesky-owned supernodal backend first
- do not copy the LDLT split mechanically where the ownership differs
- keep shared dense LDLT primitive code out of the Cholesky-owned extraction
- leave lifecycle, scalar core, and compatibility-facing writeback in the main
  file initially

That gives Sprint 56 a maintainability-first Cholesky extraction target rather
than a generic "split the next large file" instruction.
