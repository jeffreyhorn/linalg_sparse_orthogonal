# Sprint 40 Day 8: Lifecycle Contract Map

## Objective

Turn the Day 7 taxonomy into an explicit contract map that records:

- caller-visible preconditions
- mutation boundaries
- cancellation-sensitive behavior
- which exposed behaviors are probably permanent public contract
- which exposed behaviors are likely implementation leakage that future
  explicit handles should internalize

This artifact is the direct input for Sprint 40 Day 9 handle-model design.

## Inputs

This contract map is grounded in:

- `docs/planning/EPIC_4/SPRINT_40/artifacts/day5-lifecycle-inventory-lu-cholesky-ldlt.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day6-lifecycle-inventory-qr-svd-analysis-iterative-eigs.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day7-state-model-taxonomy.md`
- `include/sparse_matrix.h`
- `include/sparse_types.h`
- `README.md`

## Contract Map Overview

The lifecycle contract now reduces to four caller-visible axes:

1. eligibility preconditions
2. mutation boundaries
3. cancellation artifact
4. ownership visibility

Those four axes separate stable public obligations from internal-state leakage
more clearly than the earlier subsystem-by-subsystem inventory.

## 1. Preconditions Map

### Contract classes

| Lifecycle class | Original matrix required? | Identity permutations required? | Matrix must already be factored? | Caller typically needs `sparse_copy()` first? | Notes |
|---|:---:|:---:|:---:|:---:|---|
| Matrix-mutating factor builders | If original coefficients are needed later | Not uniformly phrased as a public hard gate for every entry point, but original-state semantics are still load-bearing | No | Often yes | LU / Cholesky |
| Original-matrix consumers with separate result handles | Yes | Yes | No | Sometimes yes, when caller is unsure matrix state is still original | LDLT / QR / SVD / ILU / ILUT / IC |
| Analysis / factor handle pipelines | Yes | Yes | No | Sometimes yes, for caller certainty before analysis/factoring | `sparse_analyze`, `sparse_factor_numeric`, `sparse_refactor_numeric` |
| Read-only operator consumers | Usually no special “original-state” rule | Usually no explicit identity-permutation rule | No | No, unless building composed preconditioners/factors separately | iterative solvers, eigensolvers |
| Bridge / mixed-boundary workflows | Inherited from the builder/factor path they compose with | Inherited from the builder/factor path they compose with | Depends on composed handle | Often indirectly yes | solver+preconditioner, shift-invert, analysis+factors |

### Most important precondition conclusions

1. The strongest precondition burden is not on iterative/eigensolver entry
   points. It is on the builder/analysis families.
2. “Use `sparse_copy()` first” is not one single rule:
   - sometimes it preserves original coefficients before in-place mutation
   - sometimes it simply restores a known-good identity-permutation matrix view
3. Identity-permutation requirements are one of the biggest architecture
   friction points because they are easy for callers to violate after prior
   reorder/factor steps.

## 2. Mutation-Boundary Map

### Mutation classes

| Lifecycle class | Mutation boundary | What actually changes? | Where mutable state lives |
|---|---|---|---|
| Matrix-mutating factor builders | Fully in-place | matrix entries, permutations, `factored`, `factor_norm`, structural interpretation | `SparseMatrix` |
| Original-matrix consumers with separate result handles | Read-only on input matrix | result/factor struct only | `sparse_ldlt_t`, `sparse_qr_t`, `sparse_svd_t`, `sparse_ilu_t` |
| Analysis / factor handle pipelines | Read-only on input matrix; explicit handle mutation | analysis buffers, factor payload, internal wrapped factor matrix | `sparse_analysis_t`, `sparse_factors_t` |
| Read-only operator consumers | Read-only on input matrix | iterates, residual histories, result structs, workspaces | `x`, result structs, internal work arrays |
| Matrix utility/cache surfaces | Cache-mutating only | `cached_norm`, `factored` via explicit mark/reset paths | `SparseMatrix` internal fields |

### Important mutation distinctions

#### Fully in-place

Only the LU / Cholesky class fits the strongest mutation bucket:

- factorization changes the same matrix object that later becomes the solve
  handle

#### Explicit-handle mutation

Most advanced decomposition/preconditioner families already fall here:

- input matrix is stable
- factor/result state lives in dedicated objects

#### Cache-only mutation

There is a smaller but important mutation class that is easy to forget:

- `sparse_norminf()` can mutate internal cached norm state
- `sparse_mark_factored()` can mutate the internal factored-state flag and
  compute `factor_norm`

This class matters because it is not algorithmic mutation, but it is still
publicly observable state change on `SparseMatrix`.

## 3. Cancellation Contract Map

### Cancellation classes

| Lifecycle class | Input matrix after cancellation | Output/factor handle after cancellation | Caller-visible artifact |
|---|---|---|---|
| Matrix-mutating factor builders | May already be non-original | same matrix object survives in partially transitioned state | partially-mutated matrix |
| Original-matrix consumers with separate result handles | bit-identical | intermediate output freed / cleaned | clean cancellation with unchanged input |
| Analysis / factor handle pipelines | mostly inherit builder semantics of delegated factor path, but analysis itself is read-only on input | explicit handles remain the state carriers | mixed, depends on phase |
| Read-only operator consumers | bit-identical | latest iterate / partial convergence state remains in outputs | partial numerical progress |

### Strongest cancellation-sensitive paths

#### Highest-risk

- LU
- Cholesky

Reason:

- callback can arrive only after state has already shifted:
  - `factored` cleared
  - `factor_norm` cached
  - Cholesky may have stripped the upper triangle

#### Moderate-risk bridge case

- analysis / factor workflow when it delegates into concrete factor builders

Reason:

- public API is handle-oriented
- but the downstream numeric phase still inherits factor-family-specific
  cancellation behavior

#### Lowest-risk

- QR
- LDLT
- SVD-family handle builders
- iterative solvers
- eigensolvers

Reason:

- either clean handle cleanup or partial-progress semantics without matrix
  mutation

## 4. Ownership Visibility Map

This is the key Day 8 addition: which currently visible behaviors are likely
true public contract, and which look like implementation leakage that a future
handle model should hide.

### A. Behaviors that should likely remain public contract

These are semantically meaningful to callers independent of implementation:

- whether an API mutates the input matrix or not
- whether a solve/decomposition path requires the original matrix view
- whether identity permutations are required
- whether cancellation can preserve the input matrix
- whether outputs/factors require explicit free routines
- whether iterative/eigensolver cancellation returns the latest iterate or
  partial results
- solver/factor family matrix-class requirements:
  - SPD
  - symmetric
  - square
  - rectangular

### B. Behaviors that look like implementation leakage and should likely become internal-only

These are currently exposed or indirectly documented, but they are not good
long-term caller-facing design anchors:

- direct caller dependence on `mat->factored` as an internal state model
- public need for `sparse_mark_factored()` in normal workflows
- direct semantic dependence on `row_perm`, `col_perm`, `inv_row_perm`,
  `inv_col_perm` for routine everyday lifecycle management
- callback-time knowledge that LU/Cholesky have already cached `factor_norm`
  before first progress emission
- callers needing to reason about `cached_norm` mutation as part of ordinary
  matrix use
- `sparse_factors_t` internally storing `SparseMatrix *F` as the payload shape

These are the clearest candidates for future handle-layer encapsulation.

## 5. Bridge-Surface Contract Notes

### `SparseMatrix` as both value object and state carrier

Today `SparseMatrix` still mixes:

- coefficient storage
- permutation state
- factor-state flags
- cached norms
- occasionally solve-handle semantics

That is the deepest ownership ambiguity in the current model.

### `sparse_factors_t` as a transitional wrapper

`sparse_factors_t` is important because it already tells callers “this is a
factor handle,” but internally it still carries matrix-centric payloads.

This means the future handle design can probably preserve the public idea of
`factors`, while changing what those factors own underneath.

### Solver/preconditioner composition

The iterative and eigensolver entry points themselves are relatively clean, so
the real contract friction in those workflows is:

- what kind of preconditioner/factor context is allowed
- what matrix state the builder for that context required
- whether the context can be reused safely

That is a composition problem more than an entry-point API problem.

## 6. Contract Priorities For Day 9

The handle-model design should target these priorities in order:

1. eliminate or isolate matrix-as-factor-handle overloading
2. reduce caller dependence on internal matrix-state fields and permutation
   machinery
3. preserve clean operator-consumer semantics for iterative/eigensolver APIs
4. keep stable caller-visible distinctions that actually matter:
   - mutates input or not
   - original matrix required or not
   - cancellation preserves input or not
5. use analysis/factors as the bridge subsystem for staged migration

## Day 8 Conclusions

1. The current lifecycle contract is now explicit enough to separate semantic
   obligations from internal-state leakage.
2. The biggest public-facing burdens are:
   - in-place factor mutation
   - identity-permutation/original-matrix eligibility rules
3. The biggest internalization targets are:
   - `factored` / `factor_norm` / permutation-array lifecycle leakage
   - matrix-centric payload storage inside bridge handles
4. Day 9 can now design a future explicit handle model against a concrete
   contract map instead of against raw subsystem notes.
