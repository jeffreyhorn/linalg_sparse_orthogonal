# Sprint 3 Benchmark Results: Reordering Effectiveness

## Summary

Fill-reducing reordering was tested on 6 SuiteSparse matrices with partial pivoting.
Three orderings compared: natural (none), Reverse Cuthill-McKee (RCM), and
Approximate Minimum Degree (AMD).

## Fill-in Comparison (partial pivoting)

| Matrix | n | nnz | None | RCM | AMD | Best |
|--------|--:|----:|-----:|----:|----:|------|
| west0067 | 67 | 294 | 928 | 982 (1.06x) | **819 (0.88x)** | AMD |
| nos4 | 100 | 594 | 1510 | 1676 (1.11x) | **1174 (0.78x)** | AMD |
| bcsstk04 | 132 | 3648 | **8581** | 8701 (1.01x) | 9153 (1.07x) | None |
| steam1 | 240 | 2248 | 22956 | **15340 (0.67x)** | 15402 (0.67x) | RCM ≈ AMD |
| fs_541_1 | 541 | 4282 | **7401** | 8875 (1.20x) | 7466 (1.01x) | None |
| orsirr_1 | 1030 | 6858 | 78069 | 96159 (1.23x) | **55212 (0.71x)** | AMD |

## Factorization Time (ms, partial pivoting)

| Matrix | None | RCM | AMD | Best |
|--------|-----:|----:|----:|------|
| west0067 | 0.45 | 0.50 | 0.53 | None |
| nos4 | 0.41 | 0.64 | 0.79 | None |
| bcsstk04 | 32.1 | 34.8 | 47.3 | None |
| steam1 | 385.6 | **70.2** | 152.0 | RCM (5.5x speedup) |
| fs_541_1 | 3.0 | 5.9 | 12.2 | None |
| orsirr_1 | 1153.6 | 1240.4 | **886.1** | AMD (1.3x speedup) |

## Key Findings

### 1. AMD excels on unstructured matrices

AMD produced the best fill-in reduction on 3 of 6 matrices:
- **orsirr_1**: 29% fill-in reduction (78k → 55k), 1.3x factorization speedup
- **nos4**: 22% fill-in reduction
- **west0067**: 12% fill-in reduction

### 2. RCM excels on thermal/banded matrices

RCM produced the best result on steam1:
- **steam1**: 33% fill-in reduction (23k → 15k), **5.5x factorization speedup**
- This is the most dramatic improvement, consistent with steam1's banded thermal structure

### 3. Some matrices don't benefit from reordering

- **bcsstk04** and **fs_541_1** are best with natural ordering
- Both have relatively low fill-in to begin with (2.4x and 1.7x)
- Reordering overhead outweighs any marginal improvement

### 4. Reordering overhead

For small matrices (n < 200), the reordering computation time is negligible.
For larger matrices, AMD's ordering cost is noticeable but justified when
fill-in reduction is significant (orsirr_1: reorder + factor still faster).

## Recommendations

1. **Default: no reordering** — For matrices with low fill-in ratio (< 3x), reordering adds overhead without benefit.

2. **Use AMD for unstructured matrices** — When fill-in ratio > 5x with natural ordering, AMD typically reduces it. Good default for general use.

3. **Use RCM for banded/thermal matrices** — When the matrix has a clear banded structure (bandwidth >> optimal), RCM provides the best bandwidth reduction and fastest factorization.

4. **Profile before committing** — Neither ordering universally dominates. For performance-critical applications, try both and compare.
