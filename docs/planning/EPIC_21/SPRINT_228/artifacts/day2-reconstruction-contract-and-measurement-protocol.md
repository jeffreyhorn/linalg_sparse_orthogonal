# Sprint 228 Day 2: Reconstruction Contract and Measurement Protocol

**Date:** 2026-11-25

**Source baseline:** `363600ea27e854a4334cff5fa06ace6529045ccd`

**Status:** Frozen before implementation

**Fixture registry:** [`day2-reconstruction-fixture-matrix.tsv`](day2-reconstruction-fixture-matrix.tsv)

## Decision

Sprint 228 will promote the existing per-cell outer reconstruction route as the maintained default only after the frozen differential, failure, memory, timing, and quality gates pass. The dense accumulator route remains temporarily available through an explicit compatibility selector for comparison and bounded rollback. No public declaration or structure layout changes.

The accepted capability is narrow: the default sparse low-rank reconstruction adds no `m×n` accumulator. Full SVD still uses dense bidiagonal work and retains dense economy factors during reconstruction.

## API validation and error precedence

`sparse_svd_lowrank_sparse(A, rank_k, drop_tol, result)` must apply checks in this order:

1. If `result == NULL`, return `SPARSE_ERR_NULL` without dereferencing it.
2. Set `*result = NULL` before inspecting any other argument.
3. If `A == NULL`, return `SPARSE_ERR_NULL`.
4. If `rank_k <= 0` or `rank_k > min(m,n)`, return `SPARSE_ERR_BADARG`.
5. If A has a factored or nonidentity row/column forward or inverse state, return `SPARSE_ERR_BADARG`.
6. If `drop_tol` is NaN or either infinity, return `SPARSE_ERR_BADARG`.
7. Parse `SPARSE_SVD_LOWRANK_OUTER`; any set value other than exact `on` or `off`, including an empty value, returns `SPARSE_ERR_BADARG`.
8. Compute the economy SVD. Propagate `SPARSE_ERR_ALLOC`, `SPARSE_ERR_NOT_CONVERGED`, or another documented hard error with `*result == NULL`.
9. If `sigma[0] == 0`, return a caller-owned empty matrix after the argument and selector checks above.
10. Resolve finite `drop_tol <= 0` to `DBL_EPSILON * sigma[0]`; otherwise use the caller's positive finite threshold.
11. Reconstruct through the selected route. Allocation failure returns `SPARSE_ERR_ALLOC`; any non-finite term or accumulated cell returns `SPARSE_ERR_NUMERIC`. Partial output is destroyed and `*result` remains NULL.

This ordering prevents an all-zero input from bypassing invalid tolerance or selector validation and avoids expensive SVD work for malformed calls.

## Selector contract

`SPARSE_SVD_LOWRANK_OUTER` is a process-environment compatibility control, not a new ABI symbol.

| Value after promotion | Route | Status |
| --- | --- | --- |
| Unset | Maintained per-cell outer route | Normal API status |
| `on` | Per-cell outer route | Normal API status |
| `off` | Dense accumulator compatibility route | Normal API status |
| Empty or any other value | No route | `SPARSE_ERR_BADARG`, NULL output |

Tests that mutate this variable must restore it and execute serially. Concurrent mutation of process environment is outside the API contract. Read-only calls with a stable environment remain governed by the library's normal thread-safety rules.

The `off` route is retained for the sprint's differential, rollback, and measurement evidence. Its future removal requires a separate compatibility decision; Sprint 228 does not silently delete it.

## Numerical and threshold contract

- Reconstruction is `A_k[i,j] = sum(s=0..min(rank_k,k)-1, sigma[s] * U[i,s] * Vt[s,j])`.
- Every input factor, product, running sum, and final cell used by reconstruction must be finite. A computed NaN or infinity fails with `SPARSE_ERR_NUMERIC`; successful output cannot contain a non-finite value.
- A cell is retained exactly when `fabs(value) >= effective_drop_tol`. Equality is retained.
- Values below the threshold are absent, not stored as explicit zeroes.
- Each coordinate is considered once; output has no duplicate coordinate.
- The all-zero SVD returns an empty matrix for any valid finite tolerance.
- Finite negative zero, zero, and negative tolerance values use the default threshold. This preserves the documented `<= 0` behavior.

### Independent oracle

One production route may never serve as the sole oracle for the other.

1. Analytical fixtures use exact matrices derived from diagonal entries, explicit rank-two column spans, or normalized power-of-two Hadamard factors.
2. General fixtures may use a separate long-double reference loop over a captured factor set. It must not call either reconstruction helper, `sparse_svd_lowrank_sparse()`, or `sparse_insert()` to determine expected values.
3. The reference records each cell's long-double sum and sum of absolute terms.
4. Cell comparison tolerance is `64 * DBL_EPSILON * max(sigma_0, sum_abs_terms)`. If both scales are zero, expected and actual must be exactly zero.
5. Normwise acceptance additionally requires relative Frobenius error at most `1e-12` for nonzero references. A zero reference requires no retained entry and no nonzero cell.
6. Threshold classification is mandatory only when the reference lies outside `[drop_tol - cell_tol, drop_tol + cell_tol]`. A generated fixture in that ambiguity band is invalid unless it is a separately named analytical power-of-two equality case.

The ambiguity rule prevents accumulation-order noise near zero from being mislabeled as a correctness failure while retaining an exact test that equality is included.

## Output, ownership, and structural contract

On success:

- `*result` is a new caller-owned `m×n` matrix freed with `sparse_free()`;
- A is byte/logically unchanged, including maps and factor metadata;
- row, inverse-row, column, and inverse-column maps are identity;
- factor/reorder metadata is fresh and unfactored;
- `nnz` equals the number of retained coordinates;
- every stored value is finite and nonzero;
- row chains are strictly increasing by column;
- column chains are strictly increasing by row; and
- the same node occurs exactly once in its row and exactly once in its column.

On every failure after a valid output pointer, `*result == NULL`, A is unchanged, and all SVD factors, route scratch, shell buffers, and partial node slabs are released. A successful retry after an injected failure must produce the accepted result.

## Pinned differential corpus

The TSV registry is normative. Its core cells are:

- square diagonal rank four;
- tall and wide rectangular diagonal rank three;
- all-zero rectangular input;
- exact rank-two dependent-column input;
- clustered diagonal singular values;
- exact cancellation from power-of-two Hadamard factors;
- exact threshold equality and below-threshold values;
- tiny finite powers of two; and
- NaN/infinite tolerance negatives plus NaN/infinite matrix-data negatives.

Day 3 implements the square/tall/wide core. Day 4 adds degenerate, threshold, cancellation, and scale cases. Day 5 completes bounded deterministic random coverage and status precedence.

## Measurement workloads

Four deterministic generated cells are frozen. Matrix entries and singular values are powers of two so correctness can be checked independently before accepting any sample.

| Workload | Shape | Requested rank | Effective numerical rank | `drop_tol` | Dense accumulator |
| --- | ---: | ---: | ---: | ---: | ---: |
| `measure-square-512-r16` | 512×512 | 16 | 32 | `1e-12` | 2,097,152 bytes |
| `measure-tall-1024x256-r8` | 1024×256 | 8 | 8 | default (`0`) | 2,097,152 bytes |
| `measure-wide-256x1024-r24` | 256×1024 | 24 | 32 | `1e-9` | 2,097,152 bytes |
| `measure-zero-512-r16` | 512×512 | 16 | 0 | default (`0`) | zero by all-zero short circuit |

The square input has diagonal values `2^5` through `2^-26`. The tall input has eight nonzero diagonal values `2^3` through `2^-4` and is rank deficient. The wide input has 32 nonzero diagonal values `2^4` through `2^-27`. The zero cell is a common-path control and must not be presented as evidence of route-specific memory removal.

All outputs must pass analytical value, `nnz`, structure, and checksum gates before timing or memory is accepted.

## Metrics and accounting

Each sample records:

- source SHA and dirty-state rejection;
- benchmark executable and static-library SHA-256;
- compiler identity and exact flags;
- `SPARSE_IDX_BITS`, dimensions, requested/effective rank, tolerance, route, and selector;
- exit status, timeout state, output `nnz`, deterministic checksum, and correctness verdict;
- full-call monotonic time in nanoseconds;
- exact peak live bytes in separate categories: bidiagonal work, retained SVD factors, reconstruction scratch, sparse shell/maps, and node slabs;
- process maximum RSS from a fresh process; and
- host, OS, architecture, thread controls, power-state note, UTC time, pair index, and order.

The exact category counters are the primary memory evidence. RSS is retained as contextual process evidence because allocator caching, slabs, and SVD phase overlap can obscure a 2 MiB route difference.

Required memory assertions:

- dense compatibility reconstruction scratch is exactly `m*n*sizeof(double)` for each nonzero cell;
- outer reconstruction scratch is zero heap bytes and O(1) scalar stack state;
- factor and output categories agree between routes for the same accepted result;
- the zero control reports zero reconstruction scratch on both routes; and
- no claim subtracts factor/output storage from the process or calls full SVD sparse-memory.

## Collection protocol

1. Build one optimized serial benchmark binary and static library from the exact candidate source. Both routes use those same bytes.
2. Set `OMP_NUM_THREADS=1`, a stable locale, and no unrelated benchmark process. Record but do not infer stable frequency from host metadata.
3. Execute each route in a fresh process. Run two unmeasured warmups per workload and route.
4. Collect ten measured pairs per workload. Pair order alternates dense/outer then outer/dense; the first order is fixed before collection.
5. Cap each invocation at 30 seconds and the complete collection at 15 minutes, including warmups and receipt finalization. A timeout invalidates the cell; it is never replaced selectively.
6. Retain raw stdout, stderr, status, timing, memory, and metadata separately for every invocation plus a SHA-256 manifest and machine-readable summary.
7. Do not discard outliers. A system interruption invalidates the whole affected workload; any full rerun is labeled and retained.

## Promotion gates

Promotion is allowed only when all gates pass on the exact implementation source:

1. Every differential/status fixture passes independently for explicit outer and dense routes.
2. Exhaustive reachable allocation and numeric failures satisfy NULL output, cleanup, unchanged input, and successful retry.
3. Unset/default execution is proven to select outer and records zero `m×n` reconstruction scratch.
4. Dense compatibility records exactly `8*m*n` reconstruction bytes on nonzero measurement cells.
5. No invocation times out; the outer/dense paired median time ratio is at most 1.25 for each nonzero workload and no individual pair ratio exceeds 1.50.
6. Raw hashes, correctness fields, category counters, and summaries validate without manual edits.
7. Focused tests, sanitizer, full C quality, and CMake parity pass.

Failure of a timing threshold blocks default promotion under this protocol; it does not authorize selective reruns or a revised threshold after seeing results. Passing timing permits only a no-material-regression statement for these cells, not a speed claim.

## Invalidation and claim boundaries

Evidence is invalid if source/binary hashes differ, routes use different builds, a correctness gate fails, output differs outside the frozen oracle rules, a sample is missing/replaced, a cap is exceeded, category counters do not reconcile, or raw files fail hash verification.

Allowed after all gates pass:

- the default sparse low-rank reconstruction no longer allocates an additional `m×n` accumulator;
- output/error behavior passed the named corpus; and
- selected exact-source cells met the frozen no-material-timing-regression gate.

Prohibited:

- full SVD is sparse-memory or memory-optimal;
- portable speedup, universal RSS reduction, or broad performance superiority;
- all SVD/low-rank algorithms share this behavior; or
- state-of-the-art sparse linear algebra performance.

## Historical evidence treatment

Sprint 29's sweep is useful design history but is not accepted Sprint 228 evidence. It used cumulative in-process RSS and described factor storage using requested `rank_k`, whereas the current full economy SVD retains factors at `k = min(m,n)`. Sprint 228 therefore requires fresh exact-source, fresh-process, component-separated measurements.
