# Sprint 40 Day 6: Lifecycle Inventory II

## Objective

Extend the lifecycle inventory beyond LU / Cholesky / LDLT across:

- QR
- SVD
- symbolic analysis / numeric factorization / refactorization
- iterative solvers
- iterative preconditioner builders
- eigensolvers

The goal is to determine which APIs already behave like explicit handle-based
subsystems, which still rely on strict original-matrix preconditions, and
which documentation surfaces currently compensate for lifecycle complexity.

## Authoritative Inputs

Primary inputs used for this inventory:

- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `include/sparse_ilu.h`
- `include/sparse_ic.h`
- `README.md`
- `docs/tutorial.md`
- `src/sparse_qr.c`
- `src/sparse_svd.c`
- `src/sparse_analysis.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`

## Contract Summary

### 1. QR, SVD, and analysis already prefer separate output handles

Unlike LU and Cholesky, the QR / SVD / analysis family generally keeps the
input matrix read-only and externalizes state into dedicated result objects:

- `sparse_qr_t`
- `sparse_svd_t`
- `sparse_analysis_t`
- `sparse_factors_t`

That is already much closer to the explicit-handle direction Epic 4 wants.

### 2. The main remaining lifecycle burden in this cluster is precondition strictness

Even where factor/result state is externalized, many APIs still require the
caller to provide:

- the original coefficient matrix view
- identity row/column permutations
- unfactored physical storage

So the lifecycle risk is less “hidden mutation” and more “hard-to-remember
matrix eligibility rules.”

### 3. Iterative and eigensolver routines are mostly operator consumers, not matrix-state owners

The iterative and eigensolver APIs are structurally different from the direct
factorization families:

- `A` is read-only
- the evolving mutable state lives in:
  - `x`
  - result structs
  - temporary workspaces
  - preconditioner contexts
  - callback-driven cancellation state

That makes them important for the taxonomy as “operator consumers” rather than
“matrix lifecycle owners.”

## Mutation / Precondition Map

| Surface | Input matrix mutated? | Requires original / unfactored matrix view? | Requires identity perms? | Separate factor / result handle? | Main mutable state lives in |
|---|:---:|:---:|:---:|:---:|---|
| QR (`sparse_qr_factor*`, `sparse_qr_solve*`) | No | Yes | Yes | Yes, `sparse_qr_t` | `sparse_qr_t`, solver outputs |
| SVD (`sparse_svd_*`) | No | Yes | Yes | Yes, `sparse_svd_t` | `sparse_svd_t` |
| Analysis (`sparse_analyze`, `sparse_factor_numeric`, `sparse_refactor_numeric`) | No on input matrix | Yes | Yes | Yes, `sparse_analysis_t` / `sparse_factors_t` | analysis / factor handles |
| Iterative solvers (`sparse_solve_*`) | No | No special original-state rule in public contract | No explicit identity-perm rule in public contract | No factor handle required; optional external preconditioner handle | `x`, result structs, workspaces |
| ILU / ILUT / IC builders | No | Yes | Yes | Yes, `sparse_ilu_t` | `sparse_ilu_t` |
| Eigensolvers (`sparse_eigs_sym`) | No | No explicit original-state rule in public contract | No explicit identity-perm rule in public contract | Caller-owned result buffers + optional external preconditioner/factor handles | result buffers, internal Krylov workspace, external preconditioner ctx |

## Per-Subsystem Findings

### QR

#### Lifecycle shape

QR is already a distinct factor-handle subsystem:

- factorization takes `const SparseMatrix *A`
- factorization writes to `sparse_qr_t`
- solve consumes `const sparse_qr_t *`
- cleanup is explicit through `sparse_qr_free()`

The input matrix is not repurposed into the solve handle.

#### Strong original-state precondition

QR still has a strict lifecycle gate:

- `A` must have identity permutations
- docs/tutorial explicitly tell callers to use the original unfactored,
  unreordered matrix view
- implementation rejects non-identity permutations because QR factors physical
  storage directly

That means QR is architecturally closer to LDLT than to LU, but it still
depends on caller discipline about matrix eligibility.

#### Cancellation semantics

QR cancellation is clean:

- intermediate state is freed
- caller does not need to clean partial QR state on cancellation
- input matrix remains bit-identical

This is the same general lifecycle class as LDLT rather than LU/Cholesky.

### SVD

#### Lifecycle shape

The SVD family is also explicit-handle oriented:

- `sparse_svd_compute`, `sparse_svd_partial`, rank/pinv/low-rank helpers all
  take `const SparseMatrix *A`
- main output lives in `sparse_svd_t`
- cleanup is explicit through `sparse_svd_free()`

The matrix is never the factor/result handle.

#### Hidden dependency on bidiagonalization preconditions

The main lifecycle constraint is again matrix eligibility:

- public headers require identity permutations
- implementation enforces that rule before bidiagonalization or partial SVD
  work proceeds
- tutorial prose explicitly teaches “use the original unfactored /
  unreordered matrix” and recommends starting from a fresh `sparse_copy()` if
  matrix state is uncertain

This is a documentation-visible workaround for a hard lifecycle rule rather
than an in-place mutation problem.

#### Architectural implication

SVD is already structurally clean in terms of handle ownership, but its
callability still depends on a fragile matrix-state precondition. That makes
it a strong example of “separate result handle, strict input-state rule.”

### Symbolic analysis / numeric factorization / refactorization

#### Lifecycle shape

The analyze-once workflow is the richest explicit-handle subsystem in this
cluster:

- `sparse_analyze()` writes `sparse_analysis_t`
- `sparse_factor_numeric()` writes `sparse_factors_t`
- `sparse_refactor_numeric()` overwrites `sparse_factors_t` on success
- `sparse_factor_solve()` consumes both handles

This is already a multi-object lifecycle model rather than a single overloaded
matrix-object flow.

#### Strict input eligibility

The analysis path still requires a very specific matrix state:

- `A` must not be factored
- `A` must have identity permutations
- for Cholesky / LDLT analysis, symmetry assumptions also apply

The implementation comments are direct about why:

- analysis and numeric factorization operate on physical index space
- non-identity permutations or prior factored state would make the symbolic
  and numeric interpretation wrong

#### Handle ownership is stronger, but still mixed internally

The public analysis/factor handles are conceptually clean, but
`sparse_factors_t` still wraps a `SparseMatrix *F` for factored state plus
LDLT-specific side arrays. So this subsystem is not free of the old model; it
has partially encapsulated it.

This is important for Epic 4 because it shows an intermediate migration shape:

- caller-visible handles are already explicit
- internal factor representation still partially depends on the matrix-centric
  model

### Iterative solvers

#### Lifecycle shape

The iterative solvers are fundamentally operator consumers:

- input matrix is `const SparseMatrix *A`
- the evolving state is:
  - current iterate `x`
  - residual/history/result bookkeeping
  - internal Krylov workspaces
  - optional preconditioner callback context

There is no persistent factor object created by the solver itself.

#### Input-state contract is lighter

Compared with QR/SVD/analysis, the iterative APIs do not publicly require:

- identity permutations
- unfactored/original matrix state

The key public contract is instead:

- matrix class requirements:
  - SPD for CG
  - general square for GMRES / BiCGSTAB
  - symmetric for MINRES
- read-only on `A`
- thread-safe across concurrent solves with distinct vectors/results

#### Cancellation semantics

Iterative cancellation is also structurally different from factorization:

- cancellation returns `SPARSE_ERR_CANCELLED`
- output `x` contains the latest iterate rather than a partially-built factor

So the cancellation artifact is “partial numerical progress,” not “partially
mutated matrix state.”

#### Lifecycle complication is externalized to preconditioners

The solver entry points themselves are clean. The lifecycle burden appears when
the caller supplies preconditioners built from other APIs.

### ILU / ILUT / IC preconditioner builders

#### Lifecycle shape

These builders externalize state into `sparse_ilu_t`, which is good:

- input matrix is `const SparseMatrix *A`
- factorization output is a separate preconditioner handle
- callback-compatible solve path is explicit

#### Strongest lifecycle gate in the iterative stack

These routines reintroduce the strict original-matrix rule:

- input matrix must have identity permutations
- input must not already be factored / pivoted / reordered
- docs explicitly recommend a fresh matrix or `sparse_copy()` of the original

So while iterative solvers themselves are read-only operator consumers, the
practical iterative workflow still inherits matrix-state fragility through the
preconditioner builders.

#### Architectural implication

For Epic 4 taxonomy purposes, these belong closer to QR/SVD/analysis than to
the iterative solvers that consume them:

- separate factor/preconditioner handle
- strict input eligibility
- read-only on input matrix

### Eigensolvers

#### Lifecycle shape

The eigensolver layer is another operator-consumer subsystem:

- input matrix is `const SparseMatrix *A`
- result storage is caller-owned (`sparse_eigs_t` buffers)
- mutable iteration state is internal Krylov / Rayleigh-Ritz workspace
- optional preconditioner or shift-invert factorization context is external

This is not a matrix-as-handle API.

#### Lighter public matrix-state rule

The public eigensolver contract focuses on:

- symmetry
- square shape
- valid `k`
- caller-owned result buffers
- shift-invert sigma validity

It does not publicly impose the same identity-permutation/original-matrix rule
that QR/SVD/analysis/preconditioner builders do.

#### External lifecycle dependencies still exist

The eigensolver stack can still depend on explicit external handles:

- LOBPCG preconditioners via `sparse_precond_fn`
- shift-invert mode through internal LDLT factorization of `A - sigma I`
- optional refinement pass via repeated internal LDLT solves

So the eigensolver API is read-only on `A`, but operationally it composes with
other handle-owning subsystems.

#### Cancellation semantics

Like iterative solvers, cancellation frees internal iteration state rather
than leaving a partially-mutated matrix or factor object behind.

## Documentation Compensation Hotspots

This cluster contains several places where docs are clearly compensating for a
hard-to-read lifecycle model:

### 1. Tutorial wording around “fresh copy of the original”

The tutorial now carries explicit warnings for:

- QR
- ILU / ILUT / IC
- SVD
- analyze-once workflow

That wording is truthful, but it is also evidence that callers must reason
carefully about matrix eligibility before using these APIs.

### 2. README explanation of preconditioner-family usage

The README has to teach not only which solver family fits which matrix class,
but also that preconditioner setup expects an original identity-permutation
matrix view.

That is not an algorithm choice only; it is lifecycle compensation.

### 3. README progress/cancellation note

The main progress/cancel section in `README.md` now acts as cross-subsystem
lifecycle documentation:

- LU / Cholesky mutate input on cancel
- LDLT / QR preserve input
- iterative and eigensolver routines do not write `A`

This is valuable, but it also shows that lifecycle semantics are spread across
subsystems rather than deriving from one consistent object model.

## Cross-Subsystem Inconsistencies

### 1. Separate handles exist, but input eligibility rules are inconsistent

The second-half surfaces already use separate output handles heavily, but their
input-state rules diverge:

- QR / SVD / analysis / ILU / ILUT / IC:
  - strict original / identity-permutation requirement
- iterative / eigensolver solve paths:
  - read-only operator contract without that same explicit rule

This is a major input for Day 7 taxonomy.

### 2. “Read-only on A” does not imply the same lifecycle simplicity

Several APIs are read-only on `A`, but in different ways:

- LDLT / QR / SVD:
  - read-only on input, explicit factor/result handle created
- analysis:
  - read-only on input, symbolic and numeric handles created
- iterative / eigs:
  - read-only on input, no persistent factor handle created by the solver
- preconditioner builders:
  - read-only on input, but strong input eligibility rules still apply

So “const SparseMatrix *A” is not one lifecycle class by itself.

### 3. Cancellation artifacts differ by subsystem role

- QR / LDLT / SVD-family style:
  - free intermediates, preserve input
- iterative / eigs:
  - preserve input, return latest iterate / partial convergence state
- LU / Cholesky from Day 5:
  - preserve neither original input state nor pure handle separation

This gives Epic 4 at least three distinct cancellation models today.

### 4. Analysis is already a transitional handle-model subsystem

The analyze-once workflow is probably the strongest transitional design in the
current codebase:

- public symbolic/factor handles already exist
- callers already think in terms of analysis + factors + solve
- but the factor payload still wraps `SparseMatrix *F`

That makes analysis a likely bridge subsystem for later Epic 4 handle work.

## Day 6 Conclusions

1. QR, SVD, analysis, and preconditioner builders already live closer to an
   explicit-handle architecture than LU/Cholesky do.
2. The biggest remaining lifecycle burden in this cluster is not hidden
   mutation; it is strict and inconsistent matrix-eligibility rules.
3. Iterative solvers and eigensolvers are best understood as read-only
   operator consumers whose mutable state lives in result buffers,
   workspaces, and external preconditioner/factor contexts.
4. The analysis subsystem is a particularly important bridge case:
   explicit public handles already exist, but internal factor ownership still
   partially depends on matrix-centric storage.

## Next Input Needed

Day 7 should convert the Day 5 and Day 6 inventories into a stable taxonomy,
at minimum separating:

- matrix-mutating factor builders
- original-matrix consumers with separate result handles
- analysis / factor handle pipelines
- read-only operator consumers
- external preconditioner / factor context consumers
