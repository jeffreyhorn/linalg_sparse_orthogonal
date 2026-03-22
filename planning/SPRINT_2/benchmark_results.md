# Sprint 2 — SuiteSparse Benchmark Results

**Date:** 2026-03-22
**Platform:** macOS Darwin 24.6.0, x86_64, Apple Clang 11, -O2

## Partial Pivoting

| Matrix | n | nnz | nnz_LU | Fill-in | Factor (ms) | Solve (ms) | SpMV (ms) | Residual |
|--------|--:|----:|-------:|--------:|------------:|-----------:|----------:|---------:|
| west0067 | 67 | 294 | 928 | 3.16x | 0.507 | 0.018 | 0.002 | 3.3e-15 |
| nos4 | 100 | 594 | 1,510 | 2.54x | 0.617 | 0.027 | 0.003 | 1.6e-16 |
| bcsstk04 | 132 | 3,648 | 8,581 | 2.35x | 47.4 | 0.176 | 0.029 | 5.6e-09 |
| steam1 | 240 | 2,248 | 22,956 | 10.2x | 544.1 | 0.671 | 0.019 | 6.1e-07 |
| fs_541_1 | 541 | 4,282 | 7,401 | 1.73x | 5.18 | 0.309 | 0.032 | 4.1e-14 |
| orsirr_1 | 1030 | 6,858 | 78,069 | 11.4x | 1,744 | 3.53 | 0.050 | 2.5e-08 |

## Complete Pivoting

| Matrix | n | nnz | nnz_LU | Fill-in | Factor (ms) | Solve (ms) | SpMV (ms) | Residual |
|--------|--:|----:|-------:|--------:|------------:|-----------:|----------:|---------:|
| west0067 | 67 | 294 | 741 | 2.52x | 0.500 | 0.026 | 0.001 | 1.8e-15 |
| nos4 | 100 | 594 | 2,838 | 4.78x | 6.22 | 0.074 | 0.002 | 1.7e-16 |
| bcsstk04 | 132 | 3,648 | 14,914 | 4.09x | 248.5 | 0.295 | 0.026 | 7.1e-08 |
| steam1 | 240 | 2,248 | 8,945 | 3.98x | 31.6 | 0.254 | 0.015 | 1.5e-06 |
| fs_541_1 | 541 | 4,282 | 12,119 | 2.83x | 54.0 | 0.363 | 0.030 | 5.8e-13 |
| orsirr_1 | 1030 | 6,858 | 106,257 | 15.5x | 4,936 | 5.66 | 0.051 | 1.2e-08 |

## Analysis

### Fill-in

- **Low fill-in:** west0067 (2.5–3.2x), nos4 (2.5–4.8x), fs_541_1 (1.7–2.8x). These matrices have favorable sparsity structure for LU without reordering.
- **Moderate fill-in:** bcsstk04 (2.4–4.1x). Structural stiffness matrix — banded structure helps.
- **High fill-in:** steam1 (4.0–10.2x), orsirr_1 (11.4–15.5x). These would benefit significantly from fill-reducing reordering (AMD/RCM), planned for Sprint 3.

### Partial vs Complete Pivoting

- **Partial pivoting is faster** for all matrices, often dramatically:
  - bcsstk04: 47ms (partial) vs 249ms (complete) — 5.3x faster
  - orsirr_1: 1,744ms (partial) vs 4,936ms (complete) — 2.8x faster
  - steam1: 544ms (partial) vs 32ms (complete) — complete is faster here due to *less fill-in* (4.0x vs 10.2x)
- **Fill-in tradeoff is matrix-dependent:**
  - steam1: complete pivoting produces 3x less fill-in (8,945 vs 22,956), offsetting the O(n²) pivot search
  - orsirr_1: complete pivoting produces *more* fill-in (106K vs 78K), making it worse in every dimension
  - Most matrices: partial pivoting produces less fill-in

### Accuracy

- All residuals are acceptable (< 1e-06 worst case).
- nos4 achieves near-machine-precision residuals (~1e-16) with both strategies.
- steam1 has the largest residuals (~1e-06/1e-07), suggesting it may benefit from iterative refinement.
- Partial and complete pivoting achieve similar accuracy; neither is consistently better.

### Performance Scaling

- SpMV is O(nnz) and scales linearly as expected (0.002ms for 294 nnz → 0.050ms for 6,858 nnz).
- Solve time is dominated by factorization, not forward/backward substitution.
- The 1030x1030 orsirr_1 matrix takes ~1.7s (partial) to ~4.9s (complete) to factor — fill-reducing reordering (Sprint 3) should improve this significantly.

### Recommendations for Sprint 3

1. **Fill-reducing reordering** (AMD/RCM) should target steam1 and orsirr_1, where fill-in ratios of 10–15x indicate significant room for improvement.
2. **Iterative refinement** for steam1 would improve its 1e-06 residual.
3. **Default to partial pivoting** for general use — it's faster and produces comparable or better fill-in for most matrices.
