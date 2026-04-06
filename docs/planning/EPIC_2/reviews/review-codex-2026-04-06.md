# Sparse Linear Algebra Library Review

**Date:** 2026-04-06  
**Reviewer:** Codex  
**Scope:** Current `linalg_sparse_orthogonal` repository

## Findings

### 1. Critical: the direct-solver stack still has no symmetric-indefinite path

The package already covers general square systems with LU, SPD systems with Cholesky, and rectangular/rank analysis with QR and SVD, but it still has no sparse `LDL^T` path for symmetric indefinite matrices. The direct-solver surface in [README.md](README.md#L11) and [README.md](README.md#L29) lists LU, Cholesky, QR, bidiagonalization, and SVD-related features; [include/sparse_cholesky.h](include/sparse_cholesky.h#L5) is explicitly SPD-only; [include/sparse_lu.h](include/sparse_lu.h#L5) is the only general square direct solver.

This is the biggest remaining capability gap if the goal is a first-rate sparse package. In practice, symmetric indefinite systems from KKT matrices, saddle-point systems, constrained optimization, and many PDE formulations should not be pushed through generic LU if you want the library to look complete. The next direct factorization to add is sparse `LDL^T` with symmetric pivoting. If complex support ever appears, that becomes `LDL^H`.

### 2. High: factorization is still one-shot numeric work, not analysis plus refactorization

The public LU and Cholesky APIs are destructive, in-place factorizations with optional reordering, but there is no analysis object, symbolic factorization phase, elimination tree, or reusable numeric refactorization path. The API shape in [include/sparse_lu.h](include/sparse_lu.h#L35) and [include/sparse_cholesky.h](include/sparse_cholesky.h#L30) exposes only direct factor calls, and the implementations in [src/sparse_lu.c](src/sparse_lu.c#L29) and [src/sparse_cholesky.c](src/sparse_cholesky.c#L23) compute ordering, mutate storage, and perform numeric elimination as a single pass.

That is fine for a compact library, but it is the wrong architecture for serious sparse-direct work. Repeated solves on the same sparsity pattern need:

- symbolic analysis once,
- numeric refactorization many times,
- explicit storage of elimination structure,
- and predictable memory planning before numeric work starts.

Without that split, the package will remain useful for moderate problems but will not feel comparable to SuiteSparse-style sparse direct solvers.

### 3. High: the ordering layer is too limited for production-scale sparse factorization

You have reordering support, but not yet a production-grade ordering stack. The public enum in [include/sparse_types.h](include/sparse_types.h#L71) exposes only `NONE`, `RCM`, and `AMD`, and the documentation already notes the current AMD is a bitset-based method with `O(n^3/64)` time and `O(n^2/64)` memory. The implementation in [src/sparse_reorder.c](src/sparse_reorder.c#L387) really does build an `n x n` bitset adjacency and performs explicit fill modeling.

That is acceptable for the matrix sizes currently tested, but it is not the ordering layer you want if the package is aiming higher. The QR path also shows the limitation: [src/sparse_qr.c](src/sparse_qr.c#L165) forms an `A^T A` sparsity proxy and then reuses the same AMD/RCM machinery, rather than offering a true sparse QR column ordering such as COLAMD.

For a first-rate package, the ordering roadmap should include:

- COLAMD/CCOLAMD for QR and unsymmetric problems,
- nested dissection for larger sparse direct problems,
- elimination-tree based symbolic analysis,
- and eventually quotient-graph or external ordering integration rather than the current dense-bitset AMD core.

### 4. High: the core storage format is still optimized for editability, not top-end numeric factorization

The library is built around the orthogonal linked-list representation. That is a coherent design choice, and the docs are honest about its trade-offs: [docs/algorithm.md](docs/algorithm.md#L29) explicitly says it has higher per-entry overhead, worse cache behavior than CSR/CSC, and is not suitable for extremely large matrices. The main LU path in [src/sparse_lu.c](src/sparse_lu.c#L29) still factors directly on that linked structure. The repo already contains a strong signal that this is the bottleneck: [README.md](README.md#L11) advertises a CSR LU path for speed, and [src/sparse_lu_csr.c](src/sparse_lu_csr.c#L21) converts the matrix into a more numeric-friendly working format before elimination.

That points to the right long-term architecture. If the package is meant to become first-rate, the linked-list representation should remain the mutable construction/edit layer, while serious numeric work should increasingly happen in compressed or supernodal working formats. You do not need to abandon the orthogonal list, but you should stop treating it as the final execution format for every major factorization.

### 5. Medium: the QR least-squares path is rank-aware, but not yet complete

The QR implementation is better than the prompt suggested: it already supports column pivoting, rank estimation, null-space extraction, economy mode, and a sparse-mode factorization path. That said, the solve contract in [include/sparse_qr.h](include/sparse_qr.h#L128) explicitly states that underdetermined or rank-deficient problems return a basic solution and “not, in general, the minimum-norm least-squares solution.”

That is a real completeness gap for a sparse least-squares package. A stronger QR story should include:

- explicit minimum-norm least-squares for underdetermined problems,
- stronger rank-revealing behavior and diagnostics,
- and ideally a cleaner separation between “basic feasible solution” and “minimum-norm solution” APIs.

I would treat this as more important than adding exotic decompositions like polar, CS, or generalized Schur.

### 6. Medium: the SVD surface is broad, but parts of it are still dense-in-disguise

The SVD feature list is ambitious and well tested, but the implementation is not yet the sort of sparse SVD layer that should outrank missing direct-method fundamentals. The public API documents two important limits:

- [include/sparse_svd.h](include/sparse_svd.h#L31) says full `U`/`V^T` output is only implemented in economy mode.
- [include/sparse_svd.h](include/sparse_svd.h#L183) says `sparse_svd_lowrank_sparse()` still allocates an `m*n` dense accumulator internally.

That means some “sparse SVD” functionality is still using dense scaling behavior at the critical step. This is not a correctness problem, but it does affect prioritization: from a package-completeness standpoint, `LDL^T`, symbolic analysis, better ordering, and SPD/indefinite iterative support are more important than expanding into even more SVD-adjacent functionality.

### 7. Medium: the iterative layer still needs the symmetric-indefinite and SPD-preconditioner pieces

The iterative solver API in [include/sparse_iterative.h](include/sparse_iterative.h#L5) currently centers on CG and GMRES, and the README preconditioner list in [README.md](README.md#L31) and [README.md](README.md#L39) is ILU-focused. That is a solid baseline, but a more complete sparse package should add:

- incomplete Cholesky for SPD preconditioning,
- MINRES for symmetric indefinite systems,
- BiCGSTAB for nonsymmetric problems where restarted GMRES is a poor fit,
- and eventually sparse eigenpair routines built on Lanczos/LOBPCG rather than only the tridiagonal kernel infrastructure.

Given the current codebase, incomplete Cholesky plus MINRES is the highest-value next step on the iterative side.

## Open Questions / Assumptions

- I am assuming the target is broader than “moderate matrices that fit comfortably in RAM.” If the intended ceiling is roughly the current test scale, the present bitset AMD and linked-list-first approach are more defensible.
- I am assuming real-valued matrices only. The current APIs and Matrix Market support appear real-only, which makes `LDL^T` the right next direct factorization rather than complex `LDL^H`.
- I am treating “first-rate” as “credible against established sparse packages on matrix classes and workflow,” not “must match SuiteSparse performance immediately.”

## Priority Roadmap

1. Add sparse symmetric-indefinite `LDL^T` with symmetric pivoting.
2. Split symbolic analysis from numeric factorization for LU, Cholesky, and the future `LDL^T`.
3. Upgrade the ordering stack with COLAMD-style QR ordering and nested-dissection-capable infrastructure.
4. Make compressed/supernodal working formats the primary numeric backend, keeping the orthogonal list as the mutable front-end structure.
5. Add incomplete Cholesky and MINRES; then add BiCGSTAB if you want a broader iterative baseline.
6. Improve QR least-squares semantics with minimum-norm solves for underdetermined and rank-deficient problems.
7. Add sparse eigenpair routines only after the direct-solver and ordering stack are in better shape.

## What Not To Prioritize Yet

I would not spend the next cycle on polar decomposition, CS decomposition, generalized SVD, QZ, or other dense-style specialty decompositions. Relative to the current codebase, those are lower value than:

- `LDL^T`,
- symbolic analysis,
- better orderings,
- incomplete Cholesky,
- MINRES,
- and a stronger compressed-format numeric backend.

## Validation

I did not find a correctness regression that currently blocks release. The repository built cleanly and `make test` passed across the full suite during this review. The main issues are package-completeness and scaling architecture, not obvious functional breakage.
