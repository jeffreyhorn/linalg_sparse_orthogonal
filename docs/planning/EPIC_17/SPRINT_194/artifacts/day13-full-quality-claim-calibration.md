# Sprint 194 Day 13: Full Quality and Claim Calibration

## Objective

Run the final Sprint 194 quality gate, re-check affected documentation,
install, example, report, and guard surfaces, and confirm the adoption/API
coherence wording remains bounded to evidence.

## Changed Surface Reviewed

- User-facing documentation:
  - `README.md`
  - `INSTALL.md`
  - `docs/api_reference.md`
  - `docs/cookbook.md`
  - `docs/maintainer_guide.md`
  - `docs/solver_selection.md`
  - `docs/tutorial.md`
  - `examples/README.md`
- Public header comments:
  - `include/sparse_matrix.h`
  - `include/sparse_csr.h`
  - `include/sparse_iterative.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_eigs.h`
- Planning and evidence artifacts:
  - `docs/planning/EPIC_17/SPRINT_194/PLAN.md`
  - `docs/planning/EPIC_17/SPRINT_194/WORKING_NOTES.md`
  - Day 1 through Day 13 Sprint 194 artifacts.

No `.c` implementation files were changed by the sprint. Because Day 11 edited
public headers, Day 13 ran the full C quality gate.

## Full Quality Gate

Command:

```sh
make format && make lint && make test
```

Result: passed.

Evidence summary:

- `make format` completed.
- `make lint` completed strict build, tooling/example compile, `clang-tidy`,
  and `cppcheck` checks.
- `make test` completed with all tests passed.
- One existing reorder test was skipped by its own environment/feature guard;
  no failures were reported.

## Affected Documentation, Install, Example, and Guard Checks

Command:

```sh
make docs-check api-docs-local-only qr-header-docs-guard \
  source-list-check ldlt-csc-helper-guard qr-external-ref-helper-guard \
  windows-powershell-guard tooling-build examples-build && \
python3 scripts/validate_corpus_schema.py && \
python3 tests/test_selected_report_targets_manifest.py && \
python3 tests/test_selected_performance_docs.py && \
python3 tests/test_normalize_report_index.py && \
python3 tests/test_run_external_comparison.py && \
python3 tests/test_selected_comparison_workflow.py && \
python3 tests/test_bench_canonical_freshness.py && \
bash tests/test_install.sh && \
bash tests/test_cmake_install.sh
```

Result: passed.

Evidence summary:

- Doxygen/API docs coverage passed for the checked-in public headers.
- Generated API docs remained local-only.
- QR header docs guard passed.
- Source-list guard passed with 49 library sources.
- LDLT CSC helper and QR external-reference helper guards passed.
- Windows PowerShell structural ownership guard passed.
- Corpus schema validation passed.
- Selected report target manifest validation passed.
- Selected performance docs validation passed.
- Report normalizer tests passed.
- External comparison runner tests passed.
- Selected comparison workflow tests passed.
- Selected benchmark freshness tests passed.
- Tooling build produced 16 benchmark binaries and 14 example binaries.
- `examples-build` completed with examples already up to date.
- `tests/test_install.sh` passed 23 checks with 0 failures.
- `tests/test_cmake_install.sh` passed 27 checks with 0 failures and 0 skips.

## Report Freshness Gate

Command:

```sh
make report-index-oracle-freshness report-index-comparison-freshness \
  bench-canonical-report-freshness
```

Result: passed.

Evidence summary:

- Selected local oracle freshness passed for 54 rows.
- Selected local comparison freshness passed for 46 rows.
- Selected canonical benchmark report freshness passed.

## Claim Calibration Audit

Commands:

```sh
rg -n -i "state[- ]of[- ]the[- ]art|world[- ]class|best[- ]in[- ]class|production[- ]ready|fully supported|broad (windows|platform|package|performance)|homebrew/core|linuxbrew|vcpkg|conan|shared librar|dynamic ABI|runtime-loader|portable performance|performance guarantee|external-library parity|windows makefile parity|windows pkg-config" \
  README.md INSTALL.md docs/api_reference.md docs/cookbook.md \
  docs/maintainer_guide.md docs/solver_selection.md docs/tutorial.md \
  examples/README.md include

rg -n -i "supported|validated|not claimed|local-only|deferred|non-claim|claim boundary|package-manager|windows|performance" \
  README.md INSTALL.md docs/api_reference.md docs/cookbook.md \
  docs/maintainer_guide.md docs/solver_selection.md docs/tutorial.md \
  examples/README.md
```

Result: no unsupported promotion found.

Confirmed boundaries:

- The project does not claim state-of-the-art status, broad external-library
  parity, broad package-manager availability, portable performance guarantees,
  or production release readiness.
- Homebrew/core, Linuxbrew, vcpkg, Conan, pkgsrc, and distro package-manager
  support remain explicitly not claimed.
- Shared-library artifacts, dynamic ABI support, runtime-loader behavior, and
  shared/static selection remain deferred or not claimed.
- Windows support remains CMake-first and hosted-lane bounded; Windows Makefile
  and Windows `pkg-config` parity are not claimed.
- Windows PowerShell execution evidence remains hosted-CI-owned when local
  `pwsh` is unavailable.
- External comparison and performance evidence remains selected, threshold-free,
  fixture/local/hosted-lane scoped, not a broad portability or performance
  claim.
- Public headers retain declaration-adjacent API contracts without carrying
  tutorial-style adoption narrative.

## Generated-Output Hygiene

Commands:

```sh
find scripts/__pycache__ -type f -delete && rmdir scripts/__pycache__
git status --short --ignored build docs/api
git ls-files docs/api build
git diff --check
```

Result: passed.

Evidence summary:

- Python cache output generated by validation was removed.
- `build/` and `docs/api/` remain ignored and untracked.
- No files under `build/` or `docs/api/` are source-controlled.
- `git diff --check` passed.

## Residuals

- Local PowerShell execution remains unavailable because `pwsh` is not installed
  on this machine; hosted CI owns the `--require-pwsh` evidence path.
- No dedicated Markdown link-check target was found in the current Makefile or
  validation script surface.
- Generated `build/` and `docs/api/` outputs are expected ignored local
  artifacts.

## Completion Criteria

- Required full quality gate passed after public header edits.
- Affected docs, install, examples, selected report, and guard checks passed.
- Unsupported support, portability, package-manager, dynamic-library,
  performance, release, and state-of-the-art claims were not found.
- Residuals are recorded and bounded to environment or deferred scope.
