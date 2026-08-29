# Sprint 187 Day 4: Owner Surface and Evidence Inventory

## Purpose

Tie each reconciled Epic 17 closure candidate to concrete source files, docs,
tests, scripts, CI jobs, report manifests, benchmark drivers, current
validation commands, missing validation, and environment dependencies.

## Inventory Inputs

| Input | Role |
| --- | --- |
| `day2-review-intake-matrix.md` | Initial 16-row gap ledger. |
| `day3-residual-reconciliation.md` | Deduplicated residual mapping and closure candidate split. |
| `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md` | Inherited residual owner surfaces and validation commands. |
| `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `benchmarks/README.md`, `docs/api_reference.md` | Current public and maintainer claim/support surfaces. |
| `Makefile`, `CMakeLists.txt`, `.github/workflows/*.yml`, scripts, tests, manifests, benchmarks, examples, `include/`, and `src/` | Concrete implementation, validation, workflow, and documentation owner surfaces. |

## Closure Owner Matrix

| Closure candidate | Gap IDs | Primary owner files | Tests and scripts | CI or hosted owner | Current validation | Missing validation or decision | Environment dependencies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Homebrew proof completion | `E17-GAP-001`, `R186-PKG-LICENSE` | Root `LICENSE`/`COPYING`/`NOTICE` decision; `packaging/homebrew/sparse-lu-ortho.rb.in`; `packaging/homebrew/README.md`; `INSTALL.md`; `README.md`; `docs/maintainer_guide.md` | `scripts/homebrew_local_formula_proof.sh`; `scripts/package_manager_deferral_check.sh`; `scripts/static_package_deferral_check.sh`; `tests/test_install.sh`; `tests/test_cmake_install.sh` | Linux/macOS package-contract jobs; macOS install/package jobs | Package guards and static install proofs exist; local `brew` is available. | Approved license metadata or alternate formula license strategy; full Homebrew render/install/test/uninstall/cleanup proof. | `brew`; `cmake`; compiler; local source archive/checksum tooling. |
| PowerShell validation ownership | `E17-GAP-002`, `R186-WIN-PWSH` | `.github/workflows/windows-ci.yml`; Windows workflow snippets; maintainer Windows support docs; selected report workflow docs | Future PowerShell parse/workflow validation script or test; `tests/test_selected_comparison_workflow.py`; `tests/test_selected_report_targets_manifest.py`; `scripts/validate_corpus_schema.py` | Windows hosted CI | Windows CMake/MSVC and install/downstream jobs exist. | Owned `pwsh` parse/workflow command and hosted wiring; local skip/availability semantics. | Hosted Windows runner; local `pwsh` is unavailable. |
| Windows selected report freshness | `E17-GAP-003`, `R186-WIN-REPORT-FRESHNESS` | `.github/workflows/windows-ci.yml`; `tests/corpus/manifests/selected_report_targets.tsv`; report schemas; report docs; `docs/maintainer_guide.md`; README support text | `scripts/normalize_report_index.py`; report generator commands selected later; `tests/test_normalize_report_index.py`; manifest/schema/workflow tests | Future Windows hosted report job if promoted | Manifest, schema, and normalizer tests exist; Windows freshness remains formally deferred. | Product decision to promote one Windows-safe lane or renew deferral; exact artifact scope, row IDs, generator command, and freshness check. | Hosted Windows runner; `pwsh`; generated report artifact upload. |
| Bounded external comparison family | `E17-GAP-004`, `R186-BROAD-COMPARISON` | `scripts/run_external_comparison.py`; `tests/corpus/manifests/selected_report_targets.tsv`; `tests/corpus/manifests/report_families.tsv`; comparison docs; solver docs; `docs/maintainer_guide.md` | `tests/test_run_external_comparison.py`; `tests/test_selected_comparison_workflow.py`; `tests/test_selected_report_targets_manifest.py`; `make report-index-comparison-freshness`; source-controlled dense reference helpers | Linux/macOS selected comparison freshness jobs | Selected QR, partial-SVD, LU, and Cholesky comparison freshness exists. | One new family with fixture, reference, dependency policy, metrics, tolerances, manifest rows, docs, and freshness proof. | Optional Python dependencies such as NumPy/SciPy if selected; hosted Linux/macOS runners. |
| Methodology-bound performance lane | `E17-GAP-005` | `benchmarks/`; `scripts/bench_canonical_report.sh`; `scripts/check_bench_canonical_freshness.py`; `scripts/performance_sentinels.sh`; `benchmarks/README.md`; `docs/maintainer_guide.md`; README benchmark guidance | `make bench-canonical-report-freshness`; `make performance-sentinels`; `tests/test_bench_canonical_freshness.py`; report-index normalizer tests | Linux hosted benchmark/report freshness job | Canonical benchmark reports and local sentinels exist with threshold-free or bounded interpretation. | One methodology-bound hosted lane with exact metadata, artifact retention, variance/threshold decision, and support wording. | Hosted Linux runner; compiler metadata; stable runtime budget; benchmark fixture availability. |
| Selected review-surface reduction | `E17-GAP-006`, `R186-REVIEW-SURFACE-NEXT`, narrowed `E17-GAP-011` | Candidate large tests and source files; likely owners include `tests/test_qr.c`, `tests/test_integration.c`, `tests/test_svd.c`, `tests/test_ldlt.c`, `tests/test_etree.c`, `src/sparse_ldlt_csc.c`, `src/sparse_lu_csr.c`, `src/sparse_iterative.c`, `src/sparse_qr.c` | `scripts/check_library_sources.py`; future helper guard; focused selected tests; `make source-list-check`; `make format && make lint && make test` | Linux/macOS quality jobs; CMake parity if source lists change | Source-list guard and LDLT CSC helper guard pattern exist. | One selected cluster, no-behavior-change invariant, helper/source ownership guard, and focused validation. | Local compiler/lint tools; CMake if source lists change. |
| Adoption and API coherence | `E17-GAP-007`, `E17-GAP-008`; retained `R186-HOSTED-API` context | `README.md`; `INSTALL.md`; `docs/tutorial.md`; `docs/cookbook.md`; `docs/solver_selection.md`; `docs/api_reference.md`; `docs/maintainer_guide.md`; `examples/README.md`; public headers | `make docs-check`; `make api-docs-validate`; `make api-docs-freshness`; examples build; install checks; local markdown link checks | Linux/macOS docs and quality jobs where selected | Adoption map, local-only API docs, and examples exist. | Compact support/readiness matrix; diagnostics wording alignment; decision on whether hosted API remains long-horizon. | Doxygen for generated API checks; local example/build tools. |
| Selected reliability proof | `E17-GAP-009`, narrowed `E17-GAP-013` | Allocation hook owners; selected future source/test owner; README quality section; maintainer guide evidence section | Existing `make iterative-allocation-failure-gate`; `make matmul-allocation-failure-gate`; future selected owner gate; full C gate | Linux quality jobs if wired later | Two selected deterministic allocation-failure proof lanes exist. | One additional owner with fail-at-count harness, cleanup invariant, stale-output suppression, retry proof, and focused Make/CTest target. | Local C compiler/lint/test tools; possible allocation-hook configuration. |

## Current Validation Command Inventory

| Command | Current owner | Relevant closure families |
| --- | --- | --- |
| `git diff --check` | Git whitespace hygiene | All documentation, script, workflow, and code changes. |
| `make format-check` | Makefile format target | C/header formatting validation. |
| `make source-list-check` | `scripts/check_library_sources.py` | Review-surface changes and source-list synchronization. |
| `make lint` | Makefile lint target with benchmark/example compile-only coverage | C/header, benchmark, and example compile/lint drift. |
| `make test` | Makefile full C test suite | C/header behavior changes. |
| `make quality-review` | Reviewed Makefile local quality path | Source/test changes and broad local confidence. |
| `make quality-review-cmake` | CMake configure/build/CTest parity | CMake/source-list/platform-sensitive changes. |
| `make api-docs-validate` | Doxygen/API docs validation | API/adoption/header documentation changes. |
| `make api-docs-freshness` | Local-only generated API freshness guard | API docs and generated API policy changes. |
| `bash scripts/static_package_deferral_check.sh` | Static-first package guard | Package and ABI claim changes. |
| `bash scripts/package_manager_deferral_check.sh` | Package-manager non-claim guard | Package-manager/Homebrew claim changes. |
| `bash scripts/homebrew_local_formula_proof.sh` | Homebrew local proof script | Sprint 188 package proof. |
| `python3 scripts/validate_corpus_schema.py` | Corpus schema validation | Report/comparison/Windows manifest changes. |
| `python3 tests/test_selected_report_targets_manifest.py` | Selected report target manifest test | Report/comparison/Windows selected target changes. |
| `python3 tests/test_selected_comparison_workflow.py` | Selected comparison workflow guard | Comparison and report workflow changes. |
| `python3 tests/test_normalize_report_index.py` | Report index normalizer test | Report freshness and generated report changes. |
| `make report-index-oracle-freshness` | Selected oracle freshness | QR/partial-SVD selected oracle report changes. |
| `make report-index-comparison-freshness` | Selected comparison freshness | Comparison family changes. |
| `make bench-canonical-report-freshness` | Canonical benchmark report freshness | Performance evidence changes. |
| `make performance-sentinels` | Bounded local performance sentinels | Performance sentinel changes. |
| `make iterative-allocation-failure-gate` | Iterative repeated-run allocation-failure proof | Reliability/failure-path reference pattern. |
| `make matmul-allocation-failure-gate` | `sparse_matmul()` allocation-failure proof | Reliability/failure-path reference pattern. |

## Missing Or Future Validation Inventory

| Missing validation | Needed by | Planning note |
| --- | --- | --- |
| Passing full Homebrew proof after license strategy | Sprint 188 | Existing proof script stops before success until root license metadata or alternate strategy is resolved. |
| Owned `pwsh` parse/workflow command | Sprint 189 | Local `pwsh` is unavailable; hosted Windows must own the proof or local skip semantics must be explicit. |
| Selected Windows report freshness command | Sprint 190 | Must name exact generator, row IDs, platform, artifact upload scope, and freshness check. |
| New comparison family freshness proof | Sprint 191 | Must be fixture-bound with dependency status and tolerance policy. |
| Methodology-bound performance hosted lane | Sprint 192 | Must add metadata, artifact retention, runtime budget, and claim-safe docs. |
| Selected review-surface helper/source guard | Sprint 193 | Must protect ownership boundaries for the chosen cluster only. |
| Support/readiness matrix and diagnostics coherence check | Sprint 194 | May be docs-only unless public headers/examples change. |
| New selected reliability/failure-path focused gate | Sprint 195 | Must prove cleanup, no stale output, and retry behavior for one owner. |

## Local Environment Snapshot

| Tool | Status |
| --- | --- |
| `brew` | Available at `/usr/local/bin/brew`. |
| `pwsh` | Not available locally. |
| `gh` | Available at `/usr/local/bin/gh`. |
| `cmake` | Available at `/usr/local/bin/cmake`. |
| `ctest` | Available at `/usr/local/bin/ctest`. |
| `pkg-config` | Available at `/usr/local/bin/pkg-config`. |

## Day 5 Ranking Inputs

Day 5 should rank closure candidates using these fields:

1. owner-surface clarity;
2. current evidence strength;
3. missing validation size;
4. environment dependency risk;
5. user/adoption value;
6. state-of-the-art relevance;
7. claim-risk reduction;
8. likelihood of complete closure inside one 14-day sprint.

## Validation

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
