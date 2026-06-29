# Day 8 Runtime/Fill Comparison Batch 1

## Purpose

Produce the first bounded Sprint 98 runtime/fill comparison artifact from the
Day 7 frozen workload. This artifact captures the selected `bench_reorder`
slice without widening canonical reporting, benchmark governance, workflow
checks, or timing thresholds.

## Command

```sh
make bench-reorder-sprint86
```

Expanded workload:

```sh
build/bench_reorder --sprint86-slice --skip-factor
```

## Raw Output

```text
=== Running bench_reorder --sprint86-slice --skip-factor ===
# nd_base_threshold=160, factor=no, via_analyze=no, slice=sprint86
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
bcsstk14,1806,none,190791,0.0,skip,direct,sprint86,160
bcsstk14,1806,rcm,178311,7.5,skip,direct,sprint86,160
bcsstk14,1806,amd,116071,68.9,skip,direct,sprint86,160
bcsstk14,1806,colamd,146037,107.5,skip,direct,sprint86,160
bcsstk14,1806,nd,132634,327.6,skip,direct,sprint86,160
Pres_Poisson,14822,none,5061932,0.0,skip,direct,sprint86,160
Pres_Poisson,14822,rcm,3187081,93.4,skip,direct,sprint86,160
Pres_Poisson,14822,amd,2668793,4385.2,skip,direct,sprint86,160
Pres_Poisson,14822,colamd,3415793,9247.2,skip,direct,sprint86,160
Pres_Poisson,14822,nd,2474435,4043.2,skip,direct,sprint86,160
```

## Fill Interpretation

`nnz_L` is the primary comparison field. The reduction values below are
computed against the `none` row for each fixture.

| Matrix | Reorder | `nnz_L` | Fill reduction vs `none` |
|---|---:|---:|---:|
| `bcsstk14` | `none` | 190791 | 0.00% |
| `bcsstk14` | `rcm` | 178311 | 6.54% |
| `bcsstk14` | `amd` | 116071 | 39.16% |
| `bcsstk14` | `colamd` | 146037 | 23.46% |
| `bcsstk14` | `nd` | 132634 | 30.48% |
| `Pres_Poisson` | `none` | 5061932 | 0.00% |
| `Pres_Poisson` | `rcm` | 3187081 | 37.04% |
| `Pres_Poisson` | `amd` | 2668793 | 47.28% |
| `Pres_Poisson` | `colamd` | 3415793 | 32.52% |
| `Pres_Poisson` | `nd` | 2474435 | 51.12% |

## Runtime Context

`reorder_ms` is included only as branch-local timing context. The values from
this run should not be used as portable performance thresholds or
cross-platform claims.

The selected workload intentionally keeps:

- `factor_ms=skip`
- `reorder_path=direct`
- `fixture_slice=sprint86`
- `nd_base_threshold=160`

## Reporting Limitations

- This artifact covers two fixtures, not the full SuiteSparse corpus.
- The run is threshold-free and does not define pass/fail performance limits.
- The artifact does not replace `make bench-canonical-report`.
- The artifact does not include `bench_amd_qg`, `bench_fillin`, or broad
  `bench_reorder` output.
- The timings are local to this development environment.

## Validation

Focused validation passed:

```sh
make bench-reorder-sprint86
```

Structural expectations observed:

- comment line included `nd_base_threshold=160`, `factor=no`,
  `via_analyze=no`, and `slice=sprint86`
- stable CSV header was present
- both selected fixtures were present
- all five selected reorder rows were present for each fixture
- `factor_ms` stayed `skip`
- `reorder_path` stayed `direct`
- `fixture_slice` stayed `sprint86`

## Day 9 Follow-Up

Day 9 should complete the runtime/fill lane by checking whether this artifact
needs a maintainer-guide or benchmark-doc guardrail. If no source or schema
change is needed, Day 9 should preserve this artifact as planning evidence and
write a closeout that ties the claim boundary back to the benchmark-governance
split.
