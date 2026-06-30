# Day 13 Validation and Residual Queue

## Purpose

Run the strongest practical Sprint 98 validation set before closeout and turn
remaining assurance work into a bounded residual queue.

## Touched Surfaces Validated

Sprint 98 touched or added:

- `tests/test_ldlt_csc.c`
- `tests/ldlt_external_dense_reference.py`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_9/SPRINT_98/`

Sprint 98 also produced runtime/fill evidence from:

- `make bench-reorder-sprint86`

No workflow, Makefile, benchmark C, public README, install doc, or coverage
target changed.

## Focused Correctness Validation

Helper checks:

```sh
python3 tests/ldlt_external_dense_reference.py kkt5
python3 tests/ldlt_external_dense_reference.py kkt10
python3 tests/ldlt_external_dense_reference.py nope
```

Observed:

- `kkt5` emitted `OK 5` and solution `1..5`
- `kkt10` emitted `OK 10` and round-off-level values near `1..10`
- unknown fixture emitted `ERROR unknown fixture nope` and exited `1`

Focused LDLT CSC check:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
```

Observed:

- `test_ldlt_csc`: 98 tests passed, 0 failed, 0 skipped
- `kkt5`: `max|x-x_ref| = 0.000e+00`,
  `rel_residual = 0.000e+00`
- `kkt10`: `max|x-x_ref| = 3.553e-15`,
  `rel_residual = 2.292e-16`

## Runtime/Fill Validation

Focused runtime/fill command:

```sh
make bench-reorder-sprint86
```

Observed output preserved the selected Sprint 98 structure:

```text
=== Running bench_reorder --sprint86-slice --skip-factor ===
# nd_base_threshold=160, factor=no, via_analyze=no, slice=sprint86
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
bcsstk14,1806,none,190791,0.0,skip,direct,sprint86,160
bcsstk14,1806,rcm,178311,7.3,skip,direct,sprint86,160
bcsstk14,1806,amd,116071,63.2,skip,direct,sprint86,160
bcsstk14,1806,colamd,146037,98.3,skip,direct,sprint86,160
bcsstk14,1806,nd,132634,300.3,skip,direct,sprint86,160
Pres_Poisson,14822,none,5061932,0.0,skip,direct,sprint86,160
Pres_Poisson,14822,rcm,3187081,82.0,skip,direct,sprint86,160
Pres_Poisson,14822,amd,2668793,4041.7,skip,direct,sprint86,160
Pres_Poisson,14822,colamd,3415793,8098.1,skip,direct,sprint86,160
Pres_Poisson,14822,nd,2474435,3721.1,skip,direct,sprint86,160
```

Timing values remain local context only.

## Full Quality Validation

Because Sprint 98 modified a C test file, Day 13 reran the full required
quality chain:

```sh
make format && make lint && make test
```

Result:

- format completed
- lint completed, including strict compile, `clang-tidy`, and `cppcheck`
- full test suite completed
- final result: `All tests passed.`

Notable focused suite results inside the full run:

- `test_ldlt_csc`: 98 tests passed, 0 failed, 0 skipped
- new LDLT external-reference rows stayed at round-off residual levels
- `test_reorder_nd`: 35 tests passed, 0 failed, 1 expected skip

## Documentation Hygiene

Hygiene checks:

```sh
git diff --check
rg -n "[ \t]+$" tests/ldlt_external_dense_reference.py tests/test_ldlt_csc.c docs/planning/EPIC_9/SPRINT_98 docs/maintainer_guide.md
```

Result:

- no diff whitespace errors
- no trailing whitespace found in touched Sprint 98/code/doc surfaces

## Stale-Claim Scan

Claim scan:

```sh
rg -n "universal|portable timing|cross-platform timing|broad LDLT|broad indefinite|canonical reporting|external proof across all|every solver family|reviewed Windows Makefile|install-validation parity" docs/planning/EPIC_9/SPRINT_98 docs/maintainer_guide.md README.md INSTALL.md benchmarks/README.md
```

Result:

- hits were negative guardrails and boundary language
- no stale positive claim was found for:
  - broad LDLT external proof
  - every-solver-family external proof
  - portable timing
  - cross-platform timing parity
  - reviewed Windows Makefile parity
  - install-validation parity
  - canonical-report replacement

## Residual Queue

### External Correctness

1. Broader LDLT CSC Matrix Market or indefinite corpus coverage.
2. Iterative solver external comparison, with convergence semantics designed
   before implementation.
3. Eigensolver/LOBPCG external comparison, only after cluster/tolerance/runtime
   boundaries are explicit.
4. QR and SVD external comparison, each requiring separate reference and
   tolerance architecture.

### Runtime/Fill Evidence

1. Decide whether repeated Sprint 98-style reorder/fill artifacts need a small
   generated report target.
2. Decide whether `bench_amd_qg` should remain adjacent support evidence or get
   a separate bounded artifact lane.
3. Keep canonical report expansion deferred until a wider surface is proven
   cheap and stable.
4. Keep broad `make bench` and full-corpus timing comparison out of reviewed
   proof unless separately bounded.

### Coverage Topology

1. Keep coverage supplemental and tree-mutating.
2. Revisit coverage ownership only if a future sprint changes thresholds,
   artifact expectations, or workflow scope.
3. Continue requiring `make clean` before returning from coverage modes to
   normal reviewed paths.

### CI and Support Alignment

1. Keep Linux as the strongest reviewed source of truth.
2. Keep macOS as the enforced Apple Clang reviewed path with supplemental GCC
   and install confidence.
3. Keep Windows as the reviewed CMake-first consumer subset.
4. Do not add `bench-reorder-sprint86` to CI until its lane is classified as
   reviewed, supplemental, or artifact-only.
5. Do not widen public docs with maintainer-only proof details unless a user
   adoption path needs them.

## Closeout Readiness

Sprint 98 is ready for Day 14 closeout from a validation standpoint:

- focused correctness validation passed
- focused runtime/fill validation passed
- full quality validation passed
- docs hygiene passed
- no known overstated claim remains in touched surfaces
- residual work is ranked and bounded
