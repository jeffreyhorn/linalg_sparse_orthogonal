# Sprint 186 Day 9: Integrated Validation Plan

## Purpose

Define the final Epic 16 validation suite before execution. This artifact maps
commands to closeout claims, separates required local checks from
environment-dependent checks, and gives Days 10 and 11 failure-triage rules.

## Required Local Validation Matrix

| Validation group | Command | Protects | Failure owner |
| --- | --- | --- | --- |
| Whitespace and Markdown hygiene | `git diff --check` | All Sprint 186 documentation and planning artifacts. | Edited files reported by Git. |
| Generated API coverage | `make api-docs-validate` | Checked-in public headers have generated Doxygen pages and local-only generated output remains untracked/unstaged. | `Doxyfile`, `docs/api_reference.md`, `docs/maintainer_guide.md`, `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh`. |
| Generated API freshness | `make api-docs-freshness` | Sprint 179 local-only generated API product decision and Sprint 186 generated API claim calibration. | Same surfaces as `api-docs-validate`; generated output remains ignored local state. |
| QR header/docs coherence | `make qr-header-docs-guard` | Sprint 184 declaration-preserving QR header cleanup and Day 7 QR claim calibration. | `include/sparse_qr.h`, QR-facing README/API/tutorial/cookbook/solver-selection/example docs, `scripts/check_qr_header_docs_guard.sh`. |
| Static package boundary | `bash scripts/static_package_deferral_check.sh` | Static-first install/package claims and shared-library/ABI non-claims. | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, CMake/pkg-config package templates. |
| Package-manager boundary | `bash scripts/package_manager_deferral_check.sh` | Package-manager support non-claim, Homebrew local proof-path wording, provider deferral boundaries. | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `packaging/homebrew/README.md`, package metadata templates. |
| Corpus schema | `python3 scripts/validate_corpus_schema.py` | Report-family schema, selected target manifest schema, support-tier vocabulary, and metadata validity. | `tests/corpus/manifests/*.tsv`, `tests/corpus/schemas/*.md`. |
| Selected target manifest tests | `python3 tests/test_selected_report_targets_manifest.py` | Selected target uniqueness, expected rows, workflow metadata, Windows deferral guard. | `tests/corpus/manifests/selected_report_targets.tsv`, Windows deferral artifact. |
| Selected workflow guard | `python3 tests/test_selected_comparison_workflow.py` | Hosted workflow selected-target scope, Linux/macOS selected comparison uploads, Windows freshness deferral. | `.github/workflows/*.yml`, selected target manifest, Windows deferral artifact. |
| Report index normalizer tests | `python3 tests/test_normalize_report_index.py` | Report-index generation, selected freshness diagnostics, package rows, deferred/optional rows. | `scripts/normalize_report_index.py`, corpus manifests, report schema docs. |
| Selected oracle freshness | `make report-index-oracle-freshness` | Selected local oracle freshness for QR and partial-SVD rows. | `scripts/run_corpus_oracle.py`, generated oracle rows under ignored build output, manifest metadata. |
| Selected comparison freshness | `make report-index-comparison-freshness` | Selected QR, partial-SVD, LU, and Cholesky comparison freshness. | `scripts/run_external_comparison.py`, selected generated comparison rows, manifest metadata. |
| External comparison tests | `python3 tests/test_run_external_comparison.py` | Runner behavior, selected comparison row metadata, optional package non-selection behavior. | `scripts/run_external_comparison.py`, comparison fixtures/helpers. |
| Matmul allocation gate | `make matmul-allocation-failure-gate` | Sprint 178 selected `sparse_matmul()` allocation-failure cleanup, stale-output suppression, and retry evidence. | `tests/test_matmul.c`, gate registration test, Make/CMake test registration. |
| LDLT CSC helper guard | `make ldlt-csc-helper-guard` | Sprint 185 family-local helper ownership, include ownership, and registration boundaries. | `tests/test_ldlt_csc*.h`, `tests/test_ldlt_csc.c`, `scripts/check_ldlt_csc_helper_guard.sh`. |
| Source list guard | `make source-list-check` | Make/CMake source registration remains synchronized after helper/build-surface work. | `Makefile`, `CMakeLists.txt`, `scripts/check_library_sources.py`. |

## Full C Quality Gate

Run the full C gate when `.c` or `.h` files are modified:

```sh
make format && make lint && make test
```

Sprint 186 Days 1-9 have not changed `.c` or `.h` files, so this gate is not
required by touched-surface policy yet. Day 9 still schedules it as the final
Day 11 confidence gate because Epic 16 closeout references prior C validation,
allocation-failure proof, QR header coherence, and LDLT CSC helper evidence.

## Environment-Dependent Or Deferred Checks

| Check | Day 9 local status | Closeout handling |
| --- | --- | --- |
| PowerShell parse or workflow validation | `pwsh` is unavailable in the local environment. | Do not block local closeout while Windows report freshness remains formally deferred. Carry `R186-WIN-PWSH` to Day 13 with a closure target for a `pwsh`-equipped environment or hosted validation owner. |
| Windows selected report freshness | No selected Windows freshness lane exists; Sprint 182 selected guarded deferral. | Do not run or invent a Windows selected freshness proof. Carry `R186-WIN-REPORT-FRESHNESS` until a future sprint selects and proves a Windows-safe freshness lane. |
| Full Homebrew formula proof success | `brew` is available locally, but no root `LICENSE`, `COPYING`, or `NOTICE` file exists. | Treat full proof success as blocked by `R186-PKG-LICENSE`. Run deferral/static package guards locally; run `bash scripts/homebrew_local_formula_proof.sh` only as an expected-failure diagnostic or after license metadata is added. |
| Hosted or retained generated API publication | No product path selected; generated API HTML remains local-only. | Do not require hosted or artifact-retention checks. Carry `R186-HOSTED-API` until a later product decision selects that route. |
| Broad external comparison family expansion | No broad parity target is selected. | Do not require unselected comparison families. Carry `R186-BROAD-COMPARISON` for future bounded-family additions. |

## Execution Order For Days 10 And 11

Day 10 should run focused documentation, package, manifest, workflow, generated
API, and selected report freshness checks first:

```sh
git diff --check
make api-docs-validate
make api-docs-freshness
make qr-header-docs-guard
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
make report-index-oracle-freshness
make report-index-comparison-freshness
python3 tests/test_run_external_comparison.py
```

Day 11 should run focused C-adjacent guards and the final broad confidence
gate:

```sh
make matmul-allocation-failure-gate
make ldlt-csc-helper-guard
make source-list-check
make format && make lint && make test
git diff --check
```

## Failure Triage Rules

1. Stop on the first failing required check and record the command, exit
   status, and owner surface.
2. Prefer the smallest owner-surface fix named by the failing diagnostic.
3. Rerun the failed command before resuming the remaining validation queue.
4. For generated report freshness failures, regenerate only the selected
   family named by the diagnostic, then rerun the selected freshness check.
5. For generated API failures, keep `docs/api/html/` ignored and untracked;
   fix `Doxyfile`, public header documentation, or local-only guard wording
   rather than committing generated HTML.
6. For package-provider failures, preserve package-manager non-support wording
   unless standalone license metadata and full proof evidence are added.
7. For unavailable `pwsh` or unselected Windows freshness proof, keep the
   residual open instead of weakening the deferral guard.

## Validation

Day 9 changed planning documentation only. No `.c` or `.h` files were modified,
so the full C quality gate is not required.

Required validation:

```sh
git diff --check
```
