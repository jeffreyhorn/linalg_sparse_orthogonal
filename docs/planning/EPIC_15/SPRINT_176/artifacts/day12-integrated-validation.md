# Day 12: Integrated Validation

## Purpose

Day 12 runs the final Sprint 176 integrated validation pass for the selected
allocation-failure proof, claim recalibration, documentation guard surfaces,
and the full required C quality gate. Because Sprint 176 changed `.c` and
`.h` files earlier in the sprint, the full gate is required before closeout.

## Validation Scope

| Surface | Reason | Validation |
| --- | --- | --- |
| Iterative repeated-run allocation-failure proof | Sprint 176 added private fail injection and selected CG/GMRES/MINRES handle cleanup tests. | `make iterative-allocation-failure-gate` |
| Package-manager and static-first claim boundaries | Sprint 176 claim recalibration must preserve Epic 15 package and ABI non-claims. | `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh` |
| Report index and selected comparison guard surfaces | Epic 15 closeout carries selected comparison/report freshness claims. | `python3 tests/test_normalize_report_index.py`; `python3 tests/test_selected_comparison_workflow.py`; `python3 tests/test_bench_canonical_freshness.py` |
| Full C source/header quality gate | Sprint 176 modified source and public headers. | `make format && make lint && make test` |
| Patch hygiene | Day 12 added planning artifacts after validation. | `git diff --check` |

## Commands And Results

Focused allocation/package/report guard command:

```sh
make iterative-allocation-failure-gate &&
bash scripts/package_manager_deferral_check.sh &&
bash scripts/static_package_deferral_check.sh &&
python3 tests/test_normalize_report_index.py &&
python3 tests/test_selected_comparison_workflow.py &&
python3 tests/test_bench_canonical_freshness.py
```

Result: passed.

Observed focused outputs:

- `iterative-allocation-failure-gate: passed`.
- `test_iterative`: `Tests run: 85`, `Tests failed: 0`,
  `Tests skipped: 0`, `Assertions: 743`.
- `package-manager-deferral-check: passed`.
- `static-package-deferral-check: passed`.
- `test-normalize-report-index: ok`.
- `test-selected-comparison-workflow: ok`.
- `tests/test_bench_canonical_freshness.py`: all hosted/local selected
  freshness checks passed.

Required full gate:

```sh
make format && make lint && make test
```

Result: passed.

Observed full-gate outputs:

- `make format` completed.
- `make lint` completed, including strict warning compilation, clang-tidy,
  and cppcheck.
- `make test` completed with `All tests passed.`

Patch hygiene:

```sh
git diff --check
```

Result: passed.

## Skipped Or Not Repeated Checks

| Check | Day 12 reason |
| --- | --- |
| `tests/test_install.sh` / CMake install scripts | Day 12 did not change install rules, package metadata templates, installed headers list, or package-manager provider posture. Static/package deferral guards were run instead to protect the claim boundary. |
| Report generator freshness commands such as `make report-index-comparison-freshness`, `make bench-canonical-report-freshness`, and `make api-docs-freshness` | Day 12 did not change generator scripts, report manifests, benchmark result rows, API-doc generation inputs, or workflow target inventories. Focused Python guard tests covered the selected report/workflow contracts carried into closeout. |
| Hosted CI artifact publication proof | Local validation cannot prove GitHub-hosted artifact upload behavior. Final PR CI remains the source of hosted evidence activation. |
| Generated output staging under `build/`, `coverage/`, or `docs/api/` | No generated outputs were staged for Day 12 closeout. |

## Closeout Readiness

Day 12 confirms the selected iterative allocation-failure proof is still
reachable through the maintained focused gate, the public claim boundaries
remain guarded, and the repository passes the full required C quality gate.

Day 13 can proceed to final closeout claim reconciliation and residual-queue
polishing using this integrated-validation record as the evidence baseline.
