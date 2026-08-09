# Day 5 Final Local Validation Command Log

## Scope

Day 5 runs the strongest feasible local validation baseline designed on Day 4.
The executed checks cover corpus schema, report normalization and freshness,
static-first package deferral, Make install/`pkg-config`, CMake
install/export, maintained examples, focused QR proof, focused partial-SVD
proof, local corpus/oracle generation, and Markdown hygiene. Hosted Linux,
macOS, and Windows evidence remains external and is reserved for Sprint 146
Days 6-7 reconciliation.

## Summary

| Surface | Result | Evidence |
| --- | --- | --- |
| Corpus schema | pass | `validate-corpus-schema: .../tests/corpus ok` |
| Report-index unit tests | pass | `test-normalize-report-index: ok` |
| Source-controlled report normalization | pass | `normalize-report-index: 47 rows ok` |
| Generated-aware report normalization | pass | `normalize-report-index: 47 rows ok` |
| Report freshness | pass | `normalize-report-index: freshness ok (47 rows)` |
| Selected support-family normalization | pass | `normalize-report-index: 9 rows ok` |
| Selected support-family freshness | pass | `normalize-report-index: freshness ok (9 rows)` |
| Static package deferral | pass | `static-package-deferral-check: passed` |
| Make install/`pkg-config` proof | pass | 23 passed, 0 failed |
| CMake install/export proof | pass | 26 passed, 0 failed, 0 skipped |
| Maintained examples | pass | 14 example binaries built |
| Focused QR corpus proof | pass | 4 tests, 0 failures, 83 assertions |
| Focused partial-SVD corpus proof | pass | 6 tests, 0 failures, 140 assertions |
| Local oracle/report refresh | pass | generated 15-line oracle TSV, 16-line report index, 2-line skips file, 16-line manifest |
| C/header quality gate | skipped | no `.c` or `.h` files changed in Sprint 146 |

## Command Results

### 1. Corpus Schema

```text
Command: python3 scripts/validate_corpus_schema.py
Surface: corpus, report metadata
Required: yes
Result: pass
Exit code: 0
Evidence captured: validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok
Generated artifacts: none
Claim impact: Corpus manifests, generator metadata, expected rows, report-family metadata, and schema guardrails remain valid.
Notes: This supports corpus/report evidence contracts, not observed solver pass proof by itself.
```

### 2. Report-Index Unit Tests

```text
Command: python3 tests/test_normalize_report_index.py
Surface: report
Required: yes
Result: pass
Exit code: 0
Evidence captured: test-normalize-report-index: ok
Generated artifacts: temporary test-local artifacts only
Claim impact: Normalization, generated-ingestion, freshness, advisory, skip, defer, and required-generated behaviors remain covered.
Notes: No source-controlled generated report files were created.
```

### 3. Source-Controlled Report Normalization

```text
Command: python3 scripts/normalize_report_index.py --no-generated --check
Surface: report
Required: yes
Result: pass
Exit code: 0
Evidence captured: normalize-report-index: 47 rows ok
Generated artifacts: none
Claim impact: Source-controlled report rows normalize deterministically without local generated artifacts.
Notes: This is navigation and metadata evidence, not generated pass proof.
```

### 4. Generated-Aware Report Normalization

```text
Command: python3 scripts/normalize_report_index.py --check
Surface: report
Required: yes
Result: pass
Exit code: 0
Evidence captured: normalize-report-index: 47 rows ok
Generated artifacts: none
Claim impact: Default generated-aware normalization remains valid when no additional generated reports are required.
Notes: Missing generated rows remain advisory/warning freshness diagnostics rather than pass evidence.
```

### 5. Report Freshness

```text
Command: python3 scripts/normalize_report_index.py --check-freshness
Surface: report
Required: yes
Result: pass
Exit code: 0
Evidence captured: normalize-report-index: freshness ok (47 rows)
Generated artifacts: none
Claim impact: Freshness diagnostics remain coherent for source-controlled rows, optional-data skips, and absent local generated reports.
Notes: Warnings for missing generated oracle, guardrail, and sentinel reports do not become pass evidence.
```

### 6. Selected Support-Family Normalization

```text
Command: python3 scripts/normalize_report_index.py --family documentation --family package --family ci --family runtime_backend --check
Surface: documentation, package, CI, runtime/backend
Required: yes
Result: pass
Exit code: 0
Evidence captured: normalize-report-index: 9 rows ok
Generated artifacts: none
Claim impact: Source-controlled support-tier advisory and lane-definition rows remain normalizable.
Notes: CI rows identify lanes; they do not replace hosted CI logs.
```

### 7. Selected Support-Family Freshness

```text
Command: python3 scripts/normalize_report_index.py --family documentation --family package --family ci --family runtime_backend --check-freshness
Surface: documentation, package, CI, runtime/backend
Required: yes
Result: pass
Exit code: 0
Evidence captured: normalize-report-index: freshness ok (9 rows)
Generated artifacts: none
Claim impact: Selected support rows remain source-controlled advisory metadata governed by schema and Git review.
Notes: The check confirms row coherence, not fresh hosted CI execution.
```

### 8. Static Package Deferral Guard

```text
Command: bash scripts/static_package_deferral_check.sh
Surface: package, ABI
Required: yes
Result: pass
Exit code: 0
Evidence captured: BUILD_SHARED_LIBS rejection, static target declaration, static install metadata, no shared export/ABI metadata, no package selector, support wording, final passed line.
Generated artifacts: temporary CMake rejection probe under a script-managed temp directory
Claim impact: Static-first package posture and shared-library/dynamic ABI/package-manager non-claims remain guarded.
Notes: This supports static-first claims and explicit shared-library deferral only.
```

### 9. Make Install And `pkg-config` Proof

```text
Command: bash tests/test_install.sh
Surface: package, adoption
Required: yes
Result: pass
Exit code: 0
Evidence captured: Passed: 23; Failed: 0; ALL INSTALL TESTS PASSED
Generated artifacts: temporary install prefix only
Claim impact: Static library install, header install, pkg-config metadata, downstream compile/link/run, maintained example compile/run, and uninstall behavior remain locally valid.
Notes: This is local proof on the current host; hosted platform promotion still requires CI evidence.
```

### 10. CMake Install/Export Proof

```text
Command: bash tests/test_cmake_install.sh
Surface: package, adoption
Required: yes
Result: pass
Exit code: 0
Evidence captured: Passed: 26; Failed: 0; Skipped: 0; ALL CMAKE INSTALL TESTS PASSED
Generated artifacts: temporary CMake build/install prefixes only
Claim impact: Static CMake target export, package metadata, downstream CMake example, exact-version build/run, mismatched-version rejection, and pkg-config version behavior remain locally valid.
Notes: This is local proof on the current host; Windows CMake-first and macOS reviewed support still require hosted reconciliation.
```

### 11. Maintained Examples

```text
Command: make examples-build
Surface: adoption
Required: yes
Result: pass
Exit code: 0
Evidence captured: Built 14 example binaries (no execution).
Generated artifacts: example binaries under ignored build directory
Claim impact: Maintained example sources continue to compile against the local static archive.
Notes: This does not prove every example workflow output, only the maintained build surface.
```

### 12. Focused QR Corpus Proof

```text
Command: make build/test_qr_corpus && ./build/test_qr_corpus
Surface: corpus, QR
Required: yes
Result: pass
Exit code: 0
Evidence captured: 4 tests, 0 failures, 0 skips, 83 assertions; solver nullspace normalized residual = 2.220e-16; reference direction normalized residual = 0.000e+00.
Generated artifacts: test binary under ignored build directory
Claim impact: The fixture-local QR claim for `qr_rank_deficient_6x4_nullspace_v1` remains locally proved.
Notes: This does not widen QR parity, raw basis, SuiteSparse, platform, performance, or state-of-the-art claims.
```

### 13. Focused Partial-SVD Corpus Proof

```text
Command: make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus
Surface: corpus, partial-SVD
Required: yes
Result: pass
Exit code: 0
Evidence captured: 6 tests, 0 failures, 0 skips, 140 assertions; top-k values, left/right projectors, residuals, orthogonality, tight-budget failure, recovery after failure, and full-rank truncate path passed.
Generated artifacts: test binary under ignored build directory
Claim impact: The fixture-local partial-SVD claim for `partial_svd_clustered_repeated_diag8x6_k3_v1` remains locally proved.
Notes: This does not widen broad SVD parity, repeated-spectrum coverage, convergence-rate, platform, performance, or state-of-the-art claims.
```

### 14. Local Oracle/Report Refresh

```text
Command: python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
Surface: corpus, QR, partial-SVD, report
Required: yes
Result: pass
Exit code: 0
Evidence captured: wrote build/corpus/oracle/corpus.oracle.tsv, build/corpus-reports/index.tsv, build/corpus-reports/skips.tsv, and build/corpus-reports/manifest.txt.
Generated artifacts: 15-line oracle TSV, 16-line report index, 2-line skips TSV, 16-line manifest; all under ignored build paths.
Claim impact: Local generated oracle/report evidence is refreshed for current command, commit, host, compiler, configuration, and support tier.
Notes: Generated rows remain reproducibility evidence only and are not source-controlled pass proof.
```

### 15. C/Header Quality Gate

```text
Command: make format && make lint && make test
Surface: C sources and public headers
Required: conditional
Result: skip
Exit code: not run
Evidence captured: git diff --name-only -- '*.c' '*.h' produced no changed files.
Generated artifacts: none
Claim impact: Full C gate is not required for Sprint 146 Day 5 because the branch has only planning Markdown changes so far.
Notes: Focused QR and partial-SVD proof owners were still run because final evidence refresh needs them.
```

## Environment Constraints

| Constraint | Day 5 Handling |
| --- | --- |
| Local host is macOS, not Linux CI | Linux reviewed source-of-truth status remains hosted-CI-only for Days 6-7 reconciliation. |
| Local host is macOS, not Windows/MSVC | Windows reviewed CMake subset and supplemental install/downstream confidence remain hosted-CI-only for Days 6-7 reconciliation. |
| Generated benchmark/coverage/dead-code/sentinel reports were not refreshed | Day 5 makes no freshness claim for those report families. |
| Optional external corpus data remains unavailable/default-disabled | Optional rows remain skip/defer evidence, not solver pass evidence. |
| Full C quality gate was conditional | No `.c` or `.h` files changed, so the required gate did not apply. |

## Day 6 Handoff

Day 6 should reconcile hosted platform and CI evidence against this local
baseline. The local package, report, corpus, QR, partial-SVD, and adoption
checks passed, but Linux source-of-truth, macOS reviewed install/export, and
Windows CMake-first/supplemental lanes still require hosted CI status rather
than local report rows.
