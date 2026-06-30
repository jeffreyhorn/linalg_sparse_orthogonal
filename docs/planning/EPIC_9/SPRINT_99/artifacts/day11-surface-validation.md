# Sprint 99 Day 11 - Surface Validation

## Purpose

Validate the install/export, example, consumer, benchmark, and reporting
surfaces that support the final Epic 9 closeout story. Day 11 re-runs the
package and reporting commands selected on Day 3 after the full reviewed
baseline passed on Day 10.

## Environment

- Date: 2026-06-30
- Branch: `sprint-99`
- Workspace: local macOS development tree
- Prior baseline: Day 10 `make quality-review-full` passed

Platform-specific workflow syntax and Windows reviewed subset behavior remain
CI-owned. This local Day 11 run validates the maintained local Make/CMake
install, consumer, example, and reporting surfaces without claiming symmetric
platform parity.

## Command Summary

| Surface | Command | Result |
|---|---|---|
| Make install/export | `bash tests/test_install.sh` | passed, 14 passed / 0 failed |
| CMake install/export and consumer proof | `bash tests/test_cmake_install.sh` | passed, 16 passed / 0 failed / 0 skipped |
| Example build surface | `make examples` | passed, 12 example binaries built |
| Representative example execution | selected `build/example_*` binaries | passed |
| Bounded reorder/fill calibration | `make bench-reorder-sprint86` | passed |
| Canonical benchmark reporting | `make bench-canonical-report` | passed |

## Install And Export Validation

### Make Install/Export

Command:

```sh
bash tests/test_install.sh
```

Result:

- 14 passed
- 0 failed
- final line: `ALL INSTALL TESTS PASSED`

Captured pass signals:

- static library installed
- no shared-library artifacts installed
- all 19 headers installed
- `pkg-config` file installed
- `pkg-config --cflags` returned the include path
- `pkg-config --libs` returned the library flag
- `pkg-config --modversion` returned `2.2.0`
- basic `pkg-config` consumer compiled, linked, and ran
- maintained example source compiled and ran with the installed package
- uninstall removed library, headers, and pkg-config file

Closeout classification:

- validated for static-first Make install/export proof
- still a non-claim for shared-library-first packaging and dynamic ABI
  guarantees

### CMake Install/Export And Consumer Proof

Command:

```sh
bash tests/test_cmake_install.sh
```

Result:

- 16 passed
- 0 failed
- 0 skipped
- final line: `ALL CMAKE INSTALL TESTS PASSED`

Captured pass signals:

- CMake configure, build, and install passed
- static library installed
- no shared-library artifacts installed
- 19 headers installed
- `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
  `SparseTargets.cmake` installed
- `sparse.pc` installed
- `examples/cmake_example` configured with `find_package(Sparse)`, built, and
  ran
- exact installed version lookup worked
- mismatched version lookup was rejected
- `pkg-config` version was `2.2.0`

Closeout classification:

- validated for maintained CMake install/export and consumer target proof
- still a non-claim for Windows install-validation and package-manager
  integration

## Example Validation

Command:

```sh
make examples
```

Result:

- built 12 example binaries
- final line: `All examples built.`

Representative execution smoke set:

```sh
./build/example_basic_solve
./build/example_ldlt
./build/example_eigs
./build/example_svd_lowrank
```

Result:

- `example_basic_solve` ran and reported zero residual for the 5x5 LU solve.
- `example_ldlt` ran and reported a KKT solve relative residual of
  `1.555e-16`.
- `example_eigs` ran through largest-eigenvalue, nearest-sigma, and LOBPCG
  sections and finished with `All done.`
- `example_svd_lowrank` ran and reported the low-rank approximation table and
  sparse low-rank compression summary.

Closeout classification:

- representative public examples are executable from the local project root
- this remains a smoke proof, not exhaustive example-output specification

## Benchmark And Reporting Validation

### Bounded Reorder/Fill Calibration

Command:

```sh
make bench-reorder-sprint86
```

Result:

- command passed
- emitted CSV-style rows for `bcsstk14` and `Pres_Poisson`
- retained `nnz_L` as the claim-bearing fill field
- retained `reorder_ms` as local runtime context
- used `--skip-factor`, so `factor_ms` was reported as `skip`

Closeout classification:

- validated for bounded two-fixture reorder/fill calibration
- still a non-claim for portable timing superiority or universal reorder/fill
  superiority

### Canonical Benchmark Report

Command:

```sh
make bench-canonical-report
```

Result:

- command passed
- wrote `build/bench-reports/canonical`

Generated files:

- `build/bench-reports/canonical/bench_refactor_csc.csv`
- `build/bench-reports/canonical/bench_chol_csc.csv`
- `build/bench-reports/canonical/bench_iterative_reuse.csv`
- `build/bench-reports/canonical/bench_eigs_reuse.csv`
- `build/bench-reports/canonical/index.tsv`
- `build/bench-reports/canonical/manifest.txt`

Manifest notes:

- generated at `2026-06-30T17:04:27Z`
- branch: `sprint-99`
- commit field from current `HEAD`: `28cd0c1f`
- report label: `unlabeled`
- report category: `proof`
- manifest explicitly states that the report is threshold-free and must not be
  interpreted as a portable timing claim

Closeout classification:

- validated for reproducible local canonical report generation
- still a non-claim for benchmark supremacy, portable timing gates, or broad
  external performance comparison

## Skipped Or Unvalidated Platform Lanes

The following remain explicit non-claims or residuals:

- Windows Makefile parity
- Windows install-validation lane
- shared-library-first package contract
- dynamic ABI guarantee
- package-manager integration
- symmetric Linux/macOS/Windows reviewed parity
- portable timing or universal reorder/fill superiority

Day 11 did not edit workflow files or platform-specific build scripts. The
Windows reviewed CMake subset and platform-specific syntax remain CI-owned.

## Implementation-Day Check Decision

Day 11 changed Sprint 99 planning documentation only.

No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
modified. A separate `make format && make lint && make test` chain is not
required for the docs-only Day 11 changes.

The validation commands above exercised the selected package, consumer,
example, benchmark, and reporting surfaces.

## Day 11 Conclusion

The package and consumer surfaces match the final documentation claims:
static-first Make install/export and CMake install/export consumer proof are
maintained and currently passing.

The reporting outputs are reproducible enough for closeout evidence:
`bench-reorder-sprint86` emits the bounded reorder/fill rows, and
`bench-canonical-report` writes the maintained CSV, index, and manifest files.

Closeout package writing can proceed with the platform, package, and benchmark
non-claims preserved.
