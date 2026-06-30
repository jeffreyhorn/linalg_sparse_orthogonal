# Sprint 99 Day 12 - Closeout Evidence Package

## Purpose

Consolidate Sprint 99 Days 1-11 into one final Epic 9 closeout evidence
package. This artifact is the source for Day 13 retrospective drafts and Day
14 closeout handoff writing.

The package cites validated evidence and keeps claim limits next to the
supporting proof. It does not promote residual architecture work, platform
non-claims, or benchmark context into final product claims.

## Evidence Index

| Day | Artifact | Closeout role |
|---:|---|---|
| 1 | `day1-authoritative-inputs.txt` | Captures the authoritative Sprint 99 project-plan scope and validation rules. |
| 1 | `day1-closeout-baseline.md` | Bounds the closeout workstreams, evidence expectations, and landing order. |
| 2 | `day2-end-state-contradiction-reaudit.md` | Maps the original Epic 9 contradiction classes to current resolved, partial, residual, and non-claim status. |
| 3 | `day3-final-comparison-scope.md` | Freezes final evidence lanes, allowed language, disallowed language, and deferred comparison work. |
| 4 | `day4-correctness-runtime-evidence.md` | Captures external correctness, bounded reorder/fill calibration, and canonical report evidence. |
| 5 | `day5-package-usability-workflow-evidence.md` | Captures static-first install/export, CMake consumer, docs, and workflow-scope evidence. |
| 6 | `day6-final-fix-decision.md` | Records that no final fix batch is required and rejects broad residual work as outside Sprint 99 closeout. |
| 7 | `day7-final-fix-batch1-noop.md` | Confirms no source, public-doc, build, workflow, benchmark, script, or test edits are needed for a final fix batch. |
| 8 | `day8-final-fix-closeout-noop.md` | Closes the no-op final-fix period and prepares residual queue finalization. |
| 9 | `day9-final-residual-queue.md` | Finalizes carry-forward work, deliberate non-claims, unsupported-claim status, and resolved items. |
| 10 | `day10-reviewed-validation.md` | Records the strongest reviewed local baseline: `make quality-review-full` passed. |
| 11 | `day11-surface-validation.md` | Records install/export, consumer, example, benchmark, and reporting validation. |

## Resolved Contradiction Summary

Sprint 99 does not claim that every Epic 9 contradiction is fully eliminated.
It closes Epic 9 by separating validated improvements from durable residuals
and explicit non-claims.

| Class | Closeout status | Evidence | Claim limit |
|---:|---|---|---|
| 1. Product model ownership | Partially resolved | Day 2 notes CSR/CSC export and compressed-first construction paths are public and documented. | The linked-list shell remains the mutable compatibility owner. |
| 2. Backend maturity ceiling | Partially resolved | Day 2 notes stronger dense/backend seams and LDLT CSC backend adoption. Day 4 validates named direct-family external lanes. | Broad portable backend maturity remains residual. |
| 3. Capability breadth ceiling | Partially resolved | Day 2 notes scalar/index seams and selected solver/eigensolver/SVD/QR surface growth. | Broad complex and mixed-precision maturity remain non-claims. |
| 4. Runtime/threading and ABI/index follow-through | Partially resolved | Days 4 and 11 validate bounded benchmark/reporting commands. | OpenMP and runtime conclusions remain localized; no portable timing claim is made. |
| 5. Large mixed-role sources | Active residual | Day 2 identifies remaining large source owners; Day 9 carries continued extraction forward. | Epic 9 improved selected owners but did not eliminate hotspot concentration. |
| 6. Giant proof owners | Active residual | Day 2 identifies remaining giant tests; Day 9 carries continued extraction forward. | Epic 9 improved selected proof owners but did not eliminate concentration. |
| 7. Chronology and naming | Partially resolved | Day 2 records cleaner public surfaces and remaining lower-level chronology. | Historical planning artifacts remain historical; lower-level cleanup is future work where useful. |
| 8. Build/package/workflow duplication | Partially resolved | Days 5 and 11 validate static-first install/export and CMake consumer proof. Day 10 validates Make/CMake test-count parity. | Make, CMake, and CI remain separate proof surfaces; symmetric platform parity is not claimed. |
| 9. Maintained comparison depth | Partially resolved | Days 4 and 11 validate Cholesky CSC and LDLT CSC external dense-reference lanes plus bounded reorder/fill/reporting. | Broader solver-family and ecosystem comparisons remain residual architecture work. |
| 10. Invalid broad claims | Deliberate non-claim | Days 5 and 9 find no live unsupported positive broad claims and preserve explicit guardrails. | No final closeout text may imply fake platform symmetry, shared-library maturity, dynamic ABI, broad complex/mixed precision, or benchmark supremacy. |

## Competitive Calibration Summary

Sprint 99 preserves the Sprint 90 comparison contract: calibration evidence is
bounded, reproducible, and claim-limited.

### External Correctness

Closeout-ready evidence:

- LDLT external dense-reference helper:
  - `kkt5` passed
  - `kkt10` passed
  - unknown fixture failed closed
- Cholesky CSC focused test passed with external dense-reference rows.
- LDLT CSC focused test passed with deterministic KKT external rows.
- Day 10 `make quality-review-full` revalidated focused and aggregate test
  surfaces.

Claim limit:

- Maintained external correctness comparison covers named Cholesky CSC and
  LDLT CSC lanes.
- It does not prove every solver family against an external reference.

### Reorder/Fill And Benchmark Reporting

Closeout-ready evidence:

- `make bench-reorder-sprint86` passed on Day 4 and again on Day 11.
- The command emitted bounded `bcsstk14` and `Pres_Poisson` rows.
- `nnz_L` remains the claim-bearing fill field.
- `reorder_ms` remains local runtime context.
- `make bench-canonical-report` passed on Day 4 and again on Day 11.
- Day 11 regenerated:
  - `build/bench-reports/canonical/bench_refactor_csc.csv`
  - `build/bench-reports/canonical/bench_chol_csc.csv`
  - `build/bench-reports/canonical/bench_iterative_reuse.csv`
  - `build/bench-reports/canonical/bench_eigs_reuse.csv`
  - `build/bench-reports/canonical/index.tsv`
  - `build/bench-reports/canonical/manifest.txt`

Claim limit:

- The benchmark/reporting evidence supports reproducible local calibration and
  maintained report generation.
- It does not support portable timing superiority, universal reorder/fill
  superiority, or benchmark supremacy.

## Package, Consumer, Example, And Workflow Summary

Closeout-ready evidence:

- `bash tests/test_install.sh` passed on Day 5 and again on Day 11:
  - 14 passed
  - 0 failed
  - static library and 19 headers installed
  - no shared-library artifacts installed
  - `pkg-config` consumer compiled, linked, and ran
  - uninstall removed installed artifacts
- `bash tests/test_cmake_install.sh` passed on Day 5 and again on Day 11:
  - 16 passed
  - 0 failed
  - 0 skipped
  - CMake install/export files installed
  - `examples/cmake_example` configured with `find_package(Sparse)`, built,
    and ran
  - exact installed version lookup worked
  - mismatched version lookup was rejected
- `make examples` passed on Day 11 and built 12 example binaries.
- Representative example execution passed on Day 11:
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
- Day 5 workflow review found the Windows expected CTest count current:
  54 CMake tests minus three staged Windows exclusions equals expected 51.

Claim limit:

- The package story is static-first.
- The CMake consumer proof is maintained.
- Windows remains a reviewed CMake-first subset.
- Windows Makefile parity, Windows install-validation, package-manager
  integration, shared-library-first packaging, and dynamic ABI guarantees are
  not Epic 9 closeout claims.

## Final Validation Summary

Day 10 ran the strongest reviewed local baseline:

```sh
make quality-review-full
```

Result:

- passed
- Makefile `quality-review` passed:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`
- CMake `quality-review-cmake` passed:
  - configure
  - clean rebuild
  - `ctest -N`
  - Makefile/CMake test-count parity
  - full `ctest`
- CMake test registration: 54
- Makefile test count: 54
- full CTest result: 100% tests passed, 0 failed out of 54

Day 11 then ran the selected surface-validation commands:

- `bash tests/test_install.sh`: passed
- `bash tests/test_cmake_install.sh`: passed
- `make examples`: passed
- representative examples: passed
- `make bench-reorder-sprint86`: passed
- `make bench-canonical-report`: passed

## Final Residual Summary

Day 9 is the authoritative post-Epic-9 residual queue.

Carry forward:

- broader LDLT CSC Matrix Market or indefinite corpus comparison
- iterative solver external comparison architecture
- eigensolver/LOBPCG external comparison architecture
- QR/SVD external comparison architecture
- generated reorder/fill report target if repeated captures justify it
- continued large-source extraction
- continued giant-test extraction
- lower-level chronology cleanup where useful

Preserve as non-claims:

- full compressed-first replacement of the linked-list shell
- broad complex support
- broad mixed-precision maturity
- broad backend-neutral acceleration maturity
- shared-library-first package contract
- dynamic ABI guarantee
- symmetric Linux/macOS/Windows reviewed parity
- Windows Makefile parity or install-validation lane
- portable timing superiority or universal reorder/fill superiority
- every-solver-family external correctness comparison

Unsupported claims to remove:

- none found

## Closeout Language

Supported final language:

- Epic 9 materially improved compressed-first entry paths while retaining the
  linked-list shell as the mutable compatibility owner.
- Epic 9 improved backend and direct-family maturity on bounded maintained
  lanes.
- Cholesky CSC and LDLT CSC have maintained external dense-reference solve
  checks on named fixtures.
- Static-first install/export and CMake consumer proof are maintained and
  validated.
- Runtime/fill evidence is bounded, local, and calibration-oriented.
- Linux remains the strongest reviewed source of truth; macOS and Windows have
  intentionally narrower reviewed or supplemental proof roles.
- Sprint 99 ended with a passing reviewed local baseline and a validated
  package/reporting surface sweep.

Unsupported final language:

- the whole library is now compressed-first
- the project has broad complex or mixed-precision maturity
- the project has broad backend-neutral acceleration maturity
- the package story is shared-library-first or dynamically ABI-stable
- Linux, macOS, and Windows have symmetric reviewed parity
- benchmark output proves portable timing superiority
- reorder/fill output proves universal best choice
- every solver family has maintained external correctness comparison
- Epic 9 eliminated all large-source or giant-test concentration

## Implementation-Day Check Decision

Day 12 changed Sprint 99 planning documentation only.

No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
modified. A separate `make format && make lint && make test` chain is not
required for the docs-only Day 12 changes.

Day 10 already passed the strongest reviewed local baseline, and Day 11 passed
the selected package, consumer, example, benchmark, and reporting validation
commands.

## Day 12 Conclusion

The final closeout evidence package is ready for Day 13 retrospective drafting.
It cites artifact-backed evidence, keeps claim boundaries visible, and carries
forward residual work without weakening the validated Sprint 99 baseline.
