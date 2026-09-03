# Sprint 194 Day 12: Documentation, Examples, and Install Validation

## Objective

Validate the adoption documentation, generated API documentation inputs,
example/tooling build surface, install/package proof scripts, selected report
freshness checks, and generated-output hygiene before the final quality gate.

## Commands Run

Documentation and guard validation:

```sh
make docs-check api-docs-local-only qr-header-docs-guard \
  source-list-check ldlt-csc-helper-guard qr-external-ref-helper-guard \
  windows-powershell-guard
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
```

Example, install, and report validation:

```sh
make tooling-build examples-build
bash tests/test_install.sh
bash tests/test_cmake_install.sh
make report-index-oracle-freshness report-index-comparison-freshness \
  bench-canonical-report-freshness
```

Generated-output hygiene:

```sh
git status --short --ignored build docs/api
git ls-files docs/api build
git diff --check
```

## Results

| Surface | Result | Notes |
| --- | --- | --- |
| Doxygen generation and API docs coverage | Passed | `make docs-check` generated local Doxygen HTML and confirmed 18 checked-in public headers have generated reference/source pages. |
| Generated API local-only guard | Passed | `docs/api/` and `docs/api/html/` remain ignored; no tracked, staged, or non-ignored generated API files were found. |
| QR header docs guard | Passed | Header sections, declarations, unsupported-claim absence, and docs alignment passed. |
| Source-list guard | Passed | `source-list-check` reported 49 library sources. |
| LDLT CSC helper guard | Passed | Proof-owner registration, helper headers, and header-only registration passed. |
| QR external-reference helper guard | Passed | Required files, helper boundary, selected cluster ownership, header-only registration, and maintainer docs passed. |
| Windows PowerShell ownership guard | Passed with local environment residual | Structural checks passed. Local `pwsh` execution is unavailable on this machine; the guard records that `--require-pwsh` is hosted-CI-owned evidence, not local pass evidence. |
| Corpus schema | Passed | `validate-corpus-schema` accepted `tests/corpus`. |
| Selected report target manifest | Passed | Manifest tests passed. |
| Selected performance docs | Passed | Claim-boundary docs tests passed. |
| Report normalizer and comparison workflow tests | Passed | Normalizer, external-comparison, and selected-comparison workflow tests passed. |
| Benchmark freshness tests | Passed | Canonical freshness unit tests passed, including threshold-free claim-boundary cases. |
| Tooling and examples build | Passed | Built 16 benchmark binaries and 14 example binaries; examples had no extra rebuild work after tooling build. |
| Make install and `pkg-config` proof | Passed | `tests/test_install.sh` passed 23 checks, including staged install, static archive metadata, downstream compile/link/run, and uninstall cleanup. |
| CMake install/export proof | Passed | `tests/test_cmake_install.sh` passed 27 checks, 0 failures, 0 skips, including installed package metadata and installed CMake consumers. |
| Selected oracle report freshness | Passed | Generated local oracle rows and `normalize_report_index.py` reported freshness ok for 54 rows. |
| Selected comparison report freshness | Passed | Generated selected comparison rows and `normalize_report_index.py` reported freshness ok for 46 rows. |
| Canonical benchmark report freshness | Passed | Generated the selected canonical benchmark report and passed threshold-free freshness checks. |
| Generated-output staging hygiene | Passed | `build/` and `docs/api/` are ignored; `git ls-files docs/api build` returned no tracked generated files. |
| Whitespace hygiene | Passed | `git diff --check` passed. |

## Environment Residuals

- No dedicated Markdown link-check target was found in the current Makefile or
  validation scripts. Day 12 therefore used the available documentation guards,
  Doxygen/API coverage, schema checks, claim-boundary tests, and install
  consumer scripts as the local documentation validation surface.
- Local PowerShell execution is unavailable because `pwsh` is not installed.
  The structural Windows PowerShell guard passed and explicitly preserves
  hosted CI as the owner for `--require-pwsh` execution evidence.

## Generated Artifact Status

- Doxygen HTML was regenerated under ignored `docs/api/`.
- Build, report, oracle, comparison, and benchmark outputs were generated under
  ignored `build/`.
- No generated files under `docs/api/` or `build/` are tracked or staged.
- A transient `scripts/__pycache__/` directory created by Python validation was
  removed before final status.

## Completion Criteria

- Item 194.6 documentation, example, install, package, corpus, and generated
  report checks were run locally where supported.
- Environment-unavailable PowerShell execution was explicitly recorded.
- Generated and ignored artifacts were not staged accidentally.
- No failing Day 12 check remains.
