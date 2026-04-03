# SVD Performance Profile Results

**Date:** 2026-04-02
**Platform:** macOS Darwin 24.6.0, Apple Silicon, cc -O2
**Build:** Static library, no OpenMP

---

## Timing Breakdown by Matrix

| Matrix | Size | nnz | Bidiag (ms) | QR iter (ms) | Full SVD σ (ms) | Full SVD U/V (ms) | Partial k=5 σ (ms) | Partial k=5 U/V (ms) |
|--------|------|-----|-------------|--------------|-----------------|-------------------|--------------------|-----------------------|
| nos4 | 100×100 | 594 | 1.79 | 0.44 | 2.08 | 5.46 | 0.59 | 0.65 |
| west0067 | 67×67 | 294 | 0.56 | 0.21 | 0.76 | 2.20 | 0.34 | 0.41 |
| bcsstk04 | 132×132 | 3648 | 3.99 | 0.39 | 4.65 | 10.84 | 1.94 | 2.26 |
| steam1 | 240×240 | 2248 | 25.6 | 0.85 | 26.3 | 61.4 | 2.06 | 1.46 |
| orsirr_1 | 1030×1030 | 6858 | 2768 | 20.6 | 2821 | 6758 | 5.12 | 6.06 |

## Phase Breakdown (% of Full SVD σ-only time)

| Matrix | Bidiag % | QR iter % | Overhead (other) % |
|--------|----------|-----------|-------------------|
| nos4 | 86% | 21% | -7% (measurement noise) |
| west0067 | 73% | 27% | 0% |
| bcsstk04 | 86% | 8% | 6% |
| steam1 | 98% | 3% | -1% |
| orsirr_1 | 98% | 1% | 1% |

**Key finding:** Bidiagonalization dominates, consuming 73–98% of full SVD time. QR iteration is fast (1–27%). For larger matrices (steam1, orsirr_1), bidiag is >97% of runtime.

## UV Extraction Overhead

| Matrix | Full σ (ms) | Full U/V (ms) | UV overhead |
|--------|-------------|---------------|-------------|
| nos4 | 2.08 | 5.46 | +162% |
| west0067 | 0.76 | 2.20 | +189% |
| bcsstk04 | 4.65 | 10.84 | +133% |
| steam1 | 26.3 | 61.4 | +134% |
| orsirr_1 | 2821 | 6758 | +140% |

**Key finding:** UV extraction adds 133–189% overhead (roughly triples the runtime). This is because extracting U/V from Householder reflectors requires O(m²k + n²k) work.

## Partial SVD Speedup

| Matrix | Full σ (ms) | Partial k=5 σ (ms) | Speedup |
|--------|-------------|--------------------|---------||
| nos4 | 2.08 | 0.59 | 3.5× |
| west0067 | 0.76 | 0.34 | 2.2× |
| bcsstk04 | 4.65 | 1.94 | 2.4× |
| steam1 | 26.3 | 2.06 | 12.8× |
| orsirr_1 | 2821 | 5.12 | 551× |

**Key finding:** Partial SVD (Lanczos) scales dramatically better than full SVD. For orsirr_1 (1030×1030), partial SVD is 551× faster because it avoids the O(m²n) dense bidiagonalization entirely.

## Top-5 Hot Functions (by expected contribution)

1. **`sparse_bidiag_factor()` — Householder bidiagonalization** (~85–98% of runtime)
   - Dominates for all matrix sizes
   - O(mn·min(m,n)) dense operations on a sparse-to-dense workspace
   - The workspace conversion + Householder reflector application is the bottleneck

2. **`sparse_svd_extract_uv()` — Householder reflector application** (~60% of UV overhead)
   - Applies stored reflectors to form explicit U and V matrices
   - O(m²k + n²k) work, mostly cache-unfriendly column operations

3. **`bidiag_svd_iterate()` — QR iteration** (~1–27% of runtime)
   - Fast for moderate sizes, grows with k
   - Givens rotation application is memory-bound for large k

4. **`sparse_matvec()` — in Lanczos iteration** (partial SVD only)
   - Dominates partial SVD for sparse matrices
   - Already efficient (O(nnz) per iteration)

5. **Memory allocation** (calloc for dense workspace)
   - Single large allocation in bidiag_factor (m×n workspace)
   - Not a bottleneck per se, but determines memory scaling

## Optimization Opportunities (ranked by expected impact)

### High Impact

1. **Cache-blocked Householder application in bidiagonalization**
   - Current: column-by-column application, poor cache reuse
   - Improvement: Block Householder (WY representation), process b columns at a time
   - Expected: 1.5–3× speedup on bidiag for larger matrices (steam1, orsirr_1)
   - Effort: Medium (need compact WY factorization)

2. **Avoid full dense workspace in bidiagonalization for sparse matrices**
   - Current: Converts entire matrix to dense m×n workspace
   - Improvement: Use sparse Householder application (already exists in sparse QR)
   - Expected: Significant memory savings, moderate speedup for very sparse matrices
   - Effort: High (need to adapt sparse Householder to bidiag pattern)

### Medium Impact

3. **Lazy UV extraction: only form U/V columns that are needed**
   - Current: Always forms full m×k U and n×k V
   - Improvement: For truncated SVD, only form the top-r columns
   - Expected: Proportional savings when r << k
   - Effort: Low

4. **Selective reorthogonalization in Lanczos**
   - Current: Full reorthogonalization at every step
   - Improvement: Monitor orthogonality loss, only reorthogonalize when needed
   - Expected: ~2× speedup on Lanczos for well-separated spectra
   - Effort: Medium (need orthogonality estimator)

### Low Impact

5. **SIMD-friendly Givens rotation loops**
   - Current: Scalar loop over vector elements
   - Improvement: Use SIMD intrinsics or ensure auto-vectorization
   - Expected: 2–4× on Givens apply, but QR iteration is only 1–8% of total
   - Effort: Low

6. **Precompute A^T for Lanczos instead of sparse_transpose()**
   - Current: Creates full transpose copy
   - Improvement: Use CSR/CSC duality (transpose is implicit in CSC)
   - Expected: Save one matrix copy in partial SVD
   - Effort: Medium
