# Day 2: Allocation Surface Inventory

## Purpose

Inventory allocation-heavy solver and shared subsystems before selecting one
failure-proof target. Day 2 identifies candidate surfaces, existing allocation
helpers, current test coverage, and decision inputs for Day 3. It does not
select the Sprint 176 implementation target.

## Inventory Commands

Day 2 used these repository scans and spot checks:

- `rg -n "malloc|calloc|realloc|free\\(|alloc|allocation" src include tests scripts`
- `rg -n "sparse_malloc_array|sparse_calloc_array|sparse_malloc_idx_array|sparse_calloc_idx_array" src include tests`
- `rg -n "fail|failure|fault|inject|cleanup|destroy|free" tests src include`
- allocation/free mention counts across `src/*.c`
- focused reads of:
  - `src/sparse_alloc_internal.c`
  - `src/sparse_alloc_internal.h`
  - `src/sparse_lu_csr.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd_partial.c`
  - `src/sparse_iterative_workspace_internal.c`
  - `src/sparse_eigs_workspace_internal.c`
  - Sprint 167 Day 4 source/header inventory

These scans are planning evidence. They rank failure-path audit candidates but
do not prove correctness, memory safety, or complete allocation-failure
coverage.

## Shared Allocation Helper Surface

| Helper | Location | Current behavior | Day 2 implication |
| --- | --- | --- | --- |
| `sparse_malloc_array` | `src/sparse_alloc_internal.c` | Checks null output, zero-size success, byte-count overflow, then calls `malloc`. | Central overflow-aware helper, but no deterministic fail-injection hook. |
| `sparse_calloc_array` | `src/sparse_alloc_internal.c` | Checks null output, zero-size success, byte-count overflow, then calls `calloc`. | Central zeroed allocation helper; suitable choke point if Sprint 176 chooses helper-level fault injection. |
| `sparse_malloc_idx_array` | `src/sparse_alloc_internal.c` | Validates signed `idx_t` count before delegating to `sparse_malloc_array`. | Useful for index-sized workspaces and negative-count failure behavior. |
| `sparse_calloc_idx_array` | `src/sparse_alloc_internal.c` | Validates signed `idx_t` count before delegating to `sparse_calloc_array`. | Useful for zeroed index-sized workspaces. |
| Overflow helpers | `src/sparse_alloc_internal.h` | Inline checked multiplication, addition, index-to-size, and size-to-index conversion. | Existing deterministic overflow paths can supplement but do not replace allocation-failure injection. |

## Existing Failure-Test And Hook Map

| Surface | Existing evidence | Gap |
| --- | --- | --- |
| Allocation wrappers | `tests/test_sparse_matrix.c` checks null output pointers, zero-size behavior, overflow rejection, and successful wrapper allocation. | No hook to fail the Nth real allocation deterministically. |
| Matrix pool/core | Edge tests cover sparse matrix lifecycle, copy, insert/remove, free-list reuse, and invalid arguments. | No deterministic slab/header/permutation allocation-failure proof. |
| Direct solvers | LU, LU CSR, LDLT, Cholesky, QR, and SVD tests cover success, singular, not-SPD, rank, residual, and edge cases. | Mostly functional/numerical failure evidence, not allocation-failure cleanup evidence. |
| Workspace helpers | Iterative and eigensolver workspace helpers centralize reserve/free behavior and reset structs on free. | No direct failure-injection test proving old buffers survive or cleanup is safe after reserve failure. |
| Public cleanup APIs | Headers document `*_free()` safety for many zeroed structs and caller-owned outputs. | Documentation is uneven across failure paths and not always paired with injected allocation proof. |
| Test framework | Existing tests use skips, environment gates, and helper cleanup patterns. | No generic test allocator/fault-injection framework found. |

## Allocation-Dense Source Ranking

The following counts are textual mentions of `malloc`, `calloc`, `realloc`,
and `free` in `src/*.c`. They are a heuristic for audit priority.

| Rank | File | Mentions | Candidate meaning |
| ---: | --- | ---: | --- |
| 1 | `src/sparse_lu_csr.c` | 128 | Highest allocation/free density; core direct solver with CSR factorization and solve paths. |
| 2 | `src/sparse_ldlt_csc.c` | 125 | Largest CSC factorization surface with symbolic/numeric cleanup complexity. |
| 3 | `src/sparse_ldlt.c` | 114 | Public LDLT orchestration and backend dispatch ownership surface. |
| 4 | `src/sparse_qr.c` | 101 | Active QR comparison and solve surface with multiple allocation-heavy factorization modes. |
| 5 | `src/sparse_lu.c` | 92 | Core linked-list LU surface with public lifecycle and refinement paths. |
| 6 | `src/sparse_etree.c` | 84 | Symbolic analysis/tree construction used by sparse direct workflows. |
| 7 | `src/sparse_svd_partial.c` | 66 | Active partial-SVD evidence surface with bounded output structs and dense work arrays. |
| 8 | `src/sparse_graph_coarsen.c` | 58 | Graph coarsening hierarchy and coarse graph ownership. |
| 9 | `src/sparse_svd.c` | 54 | Full SVD orchestration and result ownership. |
| 10 | `src/sparse_reorder.c`, `src/sparse_chol_csc.c` | 51 | Ordering and CSC Cholesky allocation surfaces. |

## Candidate Risk And Value Matrix

| Candidate | User value | Claim risk | Testability | Blast radius | Day 2 notes |
| --- | --- | --- | --- | --- | --- |
| LU CSR | High | Medium | Medium | High | Highest allocation/free density and direct-solver value. Uses many direct `malloc` calls, so deterministic injection may require either wrapper migration or test-only interception. |
| LDLT CSC | High | Medium | Medium | High | Large factorization surface and cleanup complexity. Good long-term target, but high review risk for one sprint if new fault injection is also needed. |
| QR | High | Medium | Medium | Medium-high | Active comparison and user-facing least-squares/minimum-norm scope. Already has `sparse_qr_free()` cleanup for partial structs, but multiple modes and ordering paths complicate first proof. |
| Partial SVD | High | Medium | High | Medium | Active Epic 15 comparison/convergence area, output struct is zeroed early, and allocation paths use shared wrappers in concentrated blocks. Good bounded candidate. |
| Iterative workspace | Medium | Low-medium | High | Low-medium | Workspace reserve/free helpers are small, centralized, and wrapper-backed. Lower user-facing claim value but easiest deterministic cleanup proof. |
| Eigensolver workspace | Medium | Low-medium | High | Low-medium | Similar to iterative workspace with double/idx/int buffers. Useful for shared subsystem proof but less directly tied to public solver claims. |
| Matrix core | Very high | High | Medium | High | Core lifecycle authority, but slab allocator uses direct `malloc` and affects nearly every solver; too broad for a first Sprint 176 target unless scoped narrowly. |
| Shared allocation wrappers | Cross-cutting | High | High | High | Existing wrapper tests cover overflow and basic allocation, but adding fail injection here could affect many subsystems. Better as enabling infrastructure only if scoped carefully. |

## Candidate Exclusions For Day 3 Consideration

| Candidate | Reason to avoid as first selected proof |
| --- | --- |
| Matrix core broad allocation failure | Too central; a broad claim would exceed Sprint 176's one-subsystem mandate. |
| Shared allocation wrappers as the only proof | Wrapper behavior alone would not prove a solver/shared subsystem cleanup path. |
| LDLT CSC broad factorization cleanup | High value, but high implementation and review risk if deterministic injection infrastructure must be added first. |
| LU CSR broad solve/factor lifecycle | High value, but many direct allocations may make deterministic proof more invasive than a wrapper-backed subsystem. |
| All solver families | Explicitly out of scope; Sprint 176 must close one selected gap completely. |

## Dependencies And Open Questions

| Topic | Dependency or question | Day 3 relevance |
| --- | --- | --- |
| Fault injection mechanism | No generic deterministic allocator hook is present today. | Day 3 must choose between wrapper-backed target, test-only allocator interception, or limited helper-level injection. |
| Public API exposure | Allocation controls must remain internal/test-only. | Reject any approach that adds unsupported public API. |
| Platform behavior | Tests must compile under existing Linux/macOS/Windows CMake/Make boundaries if registered broadly. | Candidate should avoid POSIX-only injection unless explicitly staged. |
| Cleanup observability | Many failures are observable only through return codes and safe repeated free. | Selected proof should have clear post-failure state assertions. |
| Full C gate | Any `.c` or `.h` implementation change requires `make format && make lint && make test`. | Day 5+ implementation must budget for full validation if source changes. |

## Day 2 Recommendation For Day 3

Day 3 should choose one of two practical target classes:

1. **Wrapper-backed shared workspace proof:** iterative or eigensolver
   workspace reserve/free behavior. This gives a small, deterministic,
   low-blast-radius proof of allocation-failure cleanup in a shared subsystem.
2. **Active solver proof with bounded output cleanup:** partial SVD. This has
   stronger user-facing value and active Epic 15 evidence relevance, but may
   require more careful injection design.

LU CSR, LDLT CSC, QR, and matrix core remain important future candidates, but
Day 2 does not recommend them as the first Sprint 176 proof unless Day 3 finds
a narrow deterministic failure point that avoids broad source churn.

## Day 2 Completion Record

- Allocation-heavy source files are ranked.
- Existing shared allocation wrappers and overflow helpers are inventoried.
- Existing failure-test coverage is separated from deterministic
  allocation-failure proof.
- Candidate subsystems are comparable by user value, claim risk, testability,
  and blast radius.
- No subsystem is selected on Day 2; selection remains a Day 3 decision.
