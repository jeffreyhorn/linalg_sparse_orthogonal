# Sprint 40 Day 5: Lifecycle Inventory I

## Objective

Audit the first major stateful direct-factorization cluster:

- LU
- Cholesky
- LDLT

The goal is to record the actual lifecycle contract carried by public headers,
README/tutorial guidance, and implementation behavior before Epic 4 begins any
handle-model redesign.

## Authoritative Inputs

Primary inputs used for this inventory:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `README.md`
- `docs/tutorial.md`
- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`

## Contract Summary

### 1. LU and Cholesky still overload factor lifecycle onto `SparseMatrix`

The LU and Cholesky APIs both use `SparseMatrix *` as:

- original coefficient owner
- mutable factorization workspace
- post-factor solve handle
- permutation owner
- factored-state owner
- cached factor-norm owner

That makes their lifecycle model compact from the caller side, but it also
means the matrix object carries multiple roles over time.

### 2. LDLT already behaves much more like an explicit factor-handle API

LDLT separates concerns more cleanly:

- input matrix is `const SparseMatrix *A`
- factorization output is a separate `sparse_ldlt_t`
- solve consumes `const sparse_ldlt_t *`
- cancellation leaves `A` bit-identical
- the factor object, not the matrix, owns factor-state telemetry

The remaining lifecycle coupling is mostly caller discipline:

- `A` must still represent the original symmetric matrix state
- `A` must have identity permutations
- callers must free / reset `sparse_ldlt_t` correctly before reuse

## Mutation / Precondition Map

| Surface | Input mutated in factor step? | Requires original matrix view? | Requires identity perms on input? | Separate factor handle? | Solve reads factor state from |
|---|:---:|:---:|:---:|:---:|---|
| `sparse_lu_factor` / `sparse_lu_factor_opts` | Yes | Yes, if caller still needs original coefficients | Not for `_opts` entry because reorder is part of the factor path; yes in practice for “original coefficients” semantics | No | `SparseMatrix` |
| `sparse_cholesky_factor` / `sparse_cholesky_factor_opts` | Yes | Yes, if caller still needs original coefficients | Effective original-state expectation is strong; `_opts` also rejects already-factored / non-identity-perm inputs before reorder handling | No | `SparseMatrix` |
| `sparse_ldlt_factor` / `sparse_ldlt_factor_opts` | No | Yes | Yes | Yes, `sparse_ldlt_t` | `sparse_ldlt_t` |

## Per-Subsystem Findings

### LU

#### Public lifecycle contract

Public docs and headers are explicit that LU is in-place:

- normal usage is `SparseMatrix *LU = sparse_copy(A);`
- factorization overwrites the copied matrix with `L` and `U`
- row/column permutations are stored inside the matrix
- solve uses the factored matrix and automatically unpermutes the result

Header/docs sources align on the central rule:

- preserve the original coefficients with `sparse_copy()` before factoring if
  they are still needed later

#### Stateful fields owned by the matrix

The LU path uses `SparseMatrix` to carry:

- logical/physical row permutations
- logical/physical column permutations
- `factored`
- `factor_norm`
- the transformed matrix entries that now represent `L` and `U`

That means the same object shifts from “matrix” to “factor handle” after
factorization.

#### Cancellation / partial-mutation contract

The LU cancellation contract is not “maybe partially changed”; it is already
explicitly stateful before the first callback returns:

- `mat->factored` is cleared immediately
- `mat->factor_norm` is cached immediately
- a cancellation at `step == 0` can still leave the object in a non-original
  state

The implementation and header both agree that callers who need a bit-identical
preservation guarantee must factor a disposable `sparse_copy()` and discard it
on cancellation.

#### Architectural implication

LU is the clearest example of a lifecycle-overloaded `SparseMatrix` API:

- coefficient storage
- permutation state
- factor telemetry
- solve handle

all live in the same object.

### Cholesky

#### Public lifecycle contract

The Cholesky path mirrors LU’s lifecycle shape:

- normal usage is `SparseMatrix *L = sparse_copy(A);`
- factorization is in-place
- the matrix must represent an SPD system
- the upper triangle is discarded and the lower triangle is overwritten with
  the factor
- solve reads the factored matrix and auto-unpermutes if a reorder path ran

#### Stateful fields owned by the matrix

Like LU, Cholesky uses `SparseMatrix` as:

- original matrix copy
- mutable factor workspace
- factored solve handle
- permutation owner
- `factored` / `factor_norm` owner

The object-role transition is therefore:

1. fresh coefficient matrix
2. reordered and structurally transformed matrix
3. Cholesky factor handle

#### Cancellation / partial-mutation contract

Cholesky has the same early-state-mutation risk as LU, plus one extra
structural mutation:

- `mat->factored` is cleared before progress emission
- `mat->factor_norm` is cached before progress emission
- the upper triangle can already be stripped before a cancellation return

So the “discard a copied workspace if cancellation occurs” rule is even more
important for Cholesky than for LDLT.

#### Architectural implication

Cholesky has the same overloaded-handle issue as LU, but with a stronger
structural transformation of the underlying matrix representation. It is not
just “factored state in a matrix”; it is also “half the original structural
view intentionally destroyed.”

### LDLT

#### Public lifecycle contract

LDLT is the strongest existing example of the boundary Epic 4 is trying to
move toward:

- factorization accepts `const SparseMatrix *A`
- factorization writes to `sparse_ldlt_t`
- solve consumes `const sparse_ldlt_t *`
- cleanup is explicit through `sparse_ldlt_free()`

The input matrix remains the original coefficient owner instead of becoming the
factor handle.

#### Preconditions are still strict, but cleaner

LDLT still requires strong caller discipline:

- `A` must be symmetric and square
- `A` must have identity permutations
- callers should start from a fresh/original matrix view if the matrix may
  have been previously factored or reordered

But those preconditions are easier to reason about because the factorization
result does not overwrite `A`.

#### Cancellation / partial-mutation contract

This is the cleanest contract in the cluster:

- cancellation frees the partial `sparse_ldlt_t`
- input `A` is left bit-identical
- factor-state cleanup responsibility is explicit in the factor object

That is materially safer than the LU / Cholesky in-place model.

#### Architectural implication

LDLT is already close to the kind of explicit factor-handle separation Epic 4
is likely to use as its design reference:

- immutable coefficient input
- explicit factor output
- explicit cleanup
- read-only factor solve surface

The remaining complexity is mostly in factor-object ownership and precondition
clarity, not in hidden mutation.

## Cross-Subsystem Inconsistencies

### 1. “Direct factorization” does not mean one lifecycle model

This cluster already splits into two architectures:

- LU / Cholesky:
  - matrix-mutating
  - matrix-as-factor-handle
- LDLT:
  - input-preserving
  - separate factor object

That inconsistency is load-bearing for caller usability and for future API
cleanup planning.

### 2. Original-matrix preservation rules are not equally risky

All three subsystems care about an original matrix view, but for different
reasons:

- LU / Cholesky:
  - because factorization destroys or repurposes the matrix state
- LDLT:
  - because the input must represent the unfactored/original logical state,
    even though factorization itself does not mutate it

This means “use `sparse_copy()` first” is not one uniform pattern today. It is
serving different lifecycle failures across the APIs.

### 3. Cancellation semantics are materially different across the cluster

- LU / Cholesky:
  - callback cancellation can leave the matrix non-original immediately
- LDLT:
  - callback cancellation leaves the input matrix unchanged and cleans the
    partial factor object

That difference should be considered part of the architecture contract, not
just an incidental implementation detail.

### 4. Solve-handle identity is inconsistent

Caller-visible solve handle after factorization is:

- LU: `SparseMatrix *`
- Cholesky: `SparseMatrix *`
- LDLT: `sparse_ldlt_t *`

This is one of the strongest concrete examples supporting a later explicit
handle-model design.

## Current Documentation Ownership

The lifecycle contract for this cluster is spread across three layers:

- headers carry the precise API preconditions and cancellation caveats
- `README.md` teaches the copy-before-factor and solve-after-factor shape
- `docs/tutorial.md` shows the standard usage patterns

The good news is that these layers currently agree on the core user-facing
behavior. The harder part is that the conceptual model is still different
between LU/Cholesky and LDLT, so the documentation can only explain the split;
it cannot hide it.

## Day 5 Conclusions

1. LU and Cholesky are the clearest examples of `SparseMatrix` acting as both
   matrix object and factor handle.
2. LDLT is already much closer to the future explicit-handle direction Epic 4
   is likely to prefer.
3. The most important Day 5 risk is not just mutation; it is partially-mutated
   state under cancellation for the in-place paths.
4. The strongest design input for later handle-model work is the existing split
   inside this one factorization cluster:
   - matrix-mutating paths
   - separate-factor-object paths

## Next Input Needed

Day 6 should extend the same inventory method across:

- QR
- SVD
- analysis / reorder / refactorization surfaces
- iterative / preconditioner surfaces
- eigensolver surfaces

That will show whether LDLT is an outlier or the beginning of a broader
separate-handle pattern already present elsewhere in the library.
