# Sprint 228 Day 4: Degenerate and Numerical Differential Corpus

## Outcome

Day 4 extends the maintained low-rank reconstruction differential harness with degenerate structure, difficult spectra, exact cancellation, threshold-boundary classification, and finite extreme scales. Every fixture executes both explicit reconstruction routes and is judged against an analytical expected matrix rather than against the other production route.

The existing Day 3 helper continues to enforce exact status, shape, `nnz`, row/column ordering, duplicate absence, identity maps, finite nonzero storage, fresh metadata, and unchanged input. Day 4 adds a dense analytical-fixture helper for non-diagonal expected matrices while retaining the same structural gate.

## Deterministic fixture matrix

| Fixture ID | Shape/rank | Construction | Drop tolerance | Expected output |
| --- | --- | --- | ---: | --- |
| `zero-6x4-r2` | 6×4, rank 2 | Explicit analytical all-zero array | `0` | Empty 6×4 matrix |
| `empty-pattern-6x4-r2` | 6×4, rank 2 | Fresh sparse shell with no nodes | `1e-12` | Empty 6×4 matrix |
| `clustered-diag-6-r4` | 6×6, rank 4 | `diag(8, 8(1-2^-40), 4, 4(1-2^-40), 1, 0.5)` | `0` | First four diagonal entries |
| `rankdef-6x4-r2` | 6×4, rank 2 | `c1=2c0`, `c3=c0+c2`; signed independent `c2` | `1e-12` | Exact 24-entry input matrix |
| `threshold-ulp-diag-4-r4` | 4×4, rank 4 | `diag(1, nextup(0.5), 0.5, nextdown(0.5))` | `0.5` | First three diagonal entries |
| `hadamard-cancel-4-r2` | 4×4, rank 2 | Two orthogonal power-of-two Hadamard terms with equal singular value 8 | `2^-40` | Eight values equal to 4; eight cancellation cells absent |
| `hadamard-cancel-large-4-r2` | 4×4, rank 2 | Same cancellation pattern at singular-value scale `2^500` | `2^450` | Eight values equal to `2^499`; cancellation cells absent |
| `tiny-diag-4-r2` | 4×4, rank 2 | `diag(2^-900, 2^-901, 2^-902, 0)` | `0` | First two diagonal entries |

Both `off` and `on` execute for every fixture, producing 16 independently judged Day 4 route cells. Together with Day 3, the maintained harness now has 34 deterministic explicit-route cells.

## Threshold and tolerance rationale

The accepted inclusion rule is `fabs(value) >= drop_tol`. The ULP fixture uses C `nextafter()` around the exactly representable threshold `0.5`:

- `nextafter(0.5, INFINITY)` must be retained;
- exactly `0.5` must be retained; and
- `nextafter(0.5, 0.0)` must be absent.

This is a named analytical boundary fixture, so its classifications are intentional and do not rely on an incidental traversal order. General value checks use the frozen cell bound `64 * DBL_EPSILON * oracle_scale` plus relative Frobenius error no greater than `1e-12`. The large cancellation fixture therefore uses a meaningful scale-relative bound instead of requiring a misleading small absolute error at magnitude `2^499`.

Every successful output must still contain only finite values. The ordinary and large Hadamard fixtures force signed terms to cancel to zero in half the cells, while the tiny fixture proves that valid finite values near `2^-900` are not mistaken for zero or non-finite data.

## Observed results

| Fixture | Expected `nnz` | Maximum absolute error | Relative Frobenius error |
| --- | ---: | ---: | ---: |
| Zero and empty pattern | 0 | `0.000e+00` | `0.000e+00` |
| Clustered diagonal | 4 | `0.000e+00` | `0.000e+00` |
| Rank deficient | 24 | `1.066e-14` | `6.968e-16` |
| One-ULP threshold | 3 | `0.000e+00` | `0.000e+00` |
| Ordinary cancellation | 8 | `1.332e-15` | `2.355e-16` |
| Large finite cancellation | 8 | `5.451e+134` | `2.355e-16` |
| Tiny finite diagonal | 2 | `0.000e+00` | `0.000e+00` |

Dense and outer routes produced the same listed observations, but each result was separately checked against its analytical matrix. The large absolute error is approximately 16 orders of magnitude below the `2^499` retained-cell scale and passes the frozen cell and normwise bounds; all output values are finite.

## Verification

| Command | Result |
| --- | --- |
| `make format` | Passed |
| `make build/test_svd && ./build/test_svd` | Passed: 122 tests, 16,581 assertions, 0 failures, 0 skips |
| `make lint` | Passed: strict compile, clang-tidy, and cppcheck |
| `make test` | Passed: all maintained test binaries |

No production source or public header changed. Day 4 adds test coverage and planning evidence only.
