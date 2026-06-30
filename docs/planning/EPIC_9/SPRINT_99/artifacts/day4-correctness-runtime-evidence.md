# Sprint 99 Day 4: Correctness and Runtime Evidence

## Purpose

Day 4 executes the correctness and runtime/fill evidence commands frozen on
Day 3. The goal is to capture final closeout evidence for maintained external
correctness lanes and bounded runtime/fill calibration without widening the
claim surface.

## Environment

- Date: 2026-06-30
- Branch: `sprint-99`
- Baseline commit in generated benchmark manifest: `28cd0c1f`
- Host context: local macOS development tree

The runtime numbers below are local context only. They are not portable timing
thresholds, cross-platform performance claims, or benchmark pass/fail gates.

## Command Results

### LDLT External Dense-Reference Helper

Commands:

```sh
python3 tests/ldlt_external_dense_reference.py kkt5
python3 tests/ldlt_external_dense_reference.py kkt10
python3 tests/ldlt_external_dense_reference.py nope
```

Results:

| Fixture | Status | Captured output |
|---|---|---|
| `kkt5` | passed | `OK 5`, then reference solution `1`, `2`, `3`, `4`, `5` |
| `kkt10` | passed | `OK 10`, then solution entries matching `1..10` to floating-point precision |
| `nope` | failed closed as expected | `ERROR unknown fixture nope`, exit `1` |

Classification:

- closeout-ready for the two maintained deterministic KKT fixtures
- residual/non-claim for any broader LDLT external corpus

### Cholesky CSC External Correctness

Command:

```sh
make build/test_chol_csc && ./build/test_chol_csc
```

Result:

- 92 tests run
- 0 failed
- 0 skipped
- 20,844 assertions
- final line: `ALL TESTS PASSED`

External-reference rows:

| Fixture | Reorder path | `max|x-x_ref|` | `rel_residual` |
|---|---|---:|---:|
| `tests/data/suitesparse/nos4.mtx` | CSC | `4.690e-13` | `3.907e-15` |
| `tests/data/suitesparse/bcsstk04.mtx` | AMD CSC | `3.224e-11` | `3.010e-16` |

Classification:

- closeout-ready maintained external SPD dense-reference lane
- no final-fix candidate

### LDLT CSC External Correctness

Command:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
```

Result:

- 98 tests run
- 0 failed
- 0 skipped
- 2,288 assertions
- final line: `ALL TESTS PASSED`

External-reference rows:

| Fixture | `max|x-x_ref|` | `rel_residual` |
|---|---:|---:|
| `kkt5` | `0.000e+00` | `0.000e+00` |
| `kkt10` | `3.553e-15` | `2.292e-16` |

Classification:

- closeout-ready maintained deterministic KKT dense-reference lane
- no final-fix candidate

### Runtime/Fill Calibration

Command:

```sh
make bench-reorder-sprint86
```

Result:

```text
# nd_base_threshold=160, factor=no, via_analyze=no, slice=sprint86
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
bcsstk14,1806,none,190791,0.0,skip,direct,sprint86,160
bcsstk14,1806,rcm,178311,8.6,skip,direct,sprint86,160
bcsstk14,1806,amd,116071,64.2,skip,direct,sprint86,160
bcsstk14,1806,colamd,146037,90.2,skip,direct,sprint86,160
bcsstk14,1806,nd,132634,249.0,skip,direct,sprint86,160
Pres_Poisson,14822,none,5061932,0.0,skip,direct,sprint86,160
Pres_Poisson,14822,rcm,3187081,77.0,skip,direct,sprint86,160
Pres_Poisson,14822,amd,2668793,3700.8,skip,direct,sprint86,160
Pres_Poisson,14822,colamd,3415793,9115.2,skip,direct,sprint86,160
Pres_Poisson,14822,nd,2474435,2951.0,skip,direct,sprint86,160
```

Interpretation:

- `nnz_L` is the claim-bearing fill field.
- `reorder_ms` is local timing context only.
- `factor_ms=skip` confirms the sprint86 slice was run in skip-factor mode.

Classification:

- closeout-ready bounded runtime/fill calibration artifact
- residual/non-claim for portable timing thresholds, universal ordering
  superiority, and full-corpus runtime comparison

### Canonical Benchmark Report

Command:

```sh
make bench-canonical-report
```

Result:

```text
bench-canonical-report: wrote build/bench-reports/canonical
  - bench_refactor_csc.csv
  - bench_chol_csc.csv
  - bench_iterative_reuse.csv
  - bench_eigs_reuse.csv
  - index.tsv
  - manifest.txt
```

Manifest highlights:

- `generated_at_utc=2026-06-30T16:36:14Z`
- `report_dir=build/bench-reports/canonical`
- `git_commit=28cd0c1f`
- `git_branch=sprint-99`
- notes state that the report is threshold-free and not a pass/fail timing
  gate

CSV row counts:

| Artifact | Lines | Header owner |
|---|---:|---|
| `bench_refactor_csc.csv` | 2 | refactor CSC timing/residual schema |
| `bench_chol_csc.csv` | 2 | linked-list/CSC/supernodal Cholesky schema |
| `bench_iterative_reuse.csv` | 4 | iterative reuse schema |
| `bench_eigs_reuse.csv` | 4 | eigensolver reuse schema |

Classification:

- closeout-ready report-generation proof
- residual/non-claim for benchmark supremacy, portable timing, or thresholded
  performance gates

## Day 4 Classification Summary

| Evidence lane | Status | Reason |
|---|---|---|
| LDLT helper fixtures | Closeout-ready | positive fixtures pass; unknown fixture fails closed |
| Cholesky CSC external correctness | Closeout-ready | focused test passes with external dense-reference rows |
| LDLT CSC external correctness | Closeout-ready | focused test passes with deterministic KKT external rows |
| Reorder/fill calibration | Closeout-ready | bounded two-fixture artifact emitted with `nnz_L` field |
| Canonical benchmark report | Closeout-ready | maintained CSV report bundle generated with threshold-free manifest |
| broader solver-family external comparison | Residual/non-claim | not selected and not architected for Day 4 |
| portable timing or universal reorder superiority | Residual/non-claim | unsupported by bounded local evidence |

## Final-Fix Candidates

Day 4 produced no final-fix candidates.

The evidence supports the Day 3 allowed language:

- Cholesky CSC and LDLT CSC have maintained external dense-reference solve
  checks on named fixtures.
- Runtime/fill evidence is bounded, local, and calibration-oriented.
- Canonical benchmark reporting works as a threshold-free artifact generator.

The evidence does not support any widened runtime, solver-family, platform, or
benchmark superiority claim.
