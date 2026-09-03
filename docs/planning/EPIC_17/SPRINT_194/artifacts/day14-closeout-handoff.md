# Sprint 194 Day 14: Closeout and Handoff

## Objective

Close Sprint 194 by summarizing the completed adoption/API coherence
simplification scope, final validation evidence, claim boundaries, residuals,
and handoff material for the retrospective, commit, push, and pull request.

## Completed Scope

Sprint 194 completed the selected adoption and API coherence simplification
work planned for Epic 17:

- Audited adoption friction across `README.md`, `INSTALL.md`, maintained user
  docs, examples, public headers, and existing Epic/Sprint artifacts.
- Added an evidence-bounded support/readiness matrix to `INSTALL.md`.
- Tightened install, package-manager, Windows, shared-library, dynamic ABI,
  report freshness, and performance wording so claims align with validated
  proof.
- Clarified the local source build, Unix Make install, CMake install/export,
  Windows CMake, selected comparison, and selected benchmark evidence paths.
- Improved first-solve and tutorial routing in maintained docs and examples.
- Cleaned diagnostic wording in solver selection, tutorial, cookbook, and API
  docs.
- Reduced public-header review surface by moving or shortening tutorial-style
  adoption narrative while preserving declaration-adjacent contracts.
- Ran final validation and claim-calibration checks after header edits.

The sprint closes simplification and calibration work only. It does not add or
claim broad package-manager support, broad platform parity, shared-library
support, dynamic ABI support, broad report freshness, broad external-library
parity, portable performance guarantees, release readiness, or state-of-the-art
status.

## Changed Files and Categories

User-facing documentation:

- `README.md`
- `INSTALL.md`
- `docs/api_reference.md`
- `docs/cookbook.md`
- `docs/maintainer_guide.md`
- `docs/solver_selection.md`
- `docs/tutorial.md`
- `examples/README.md`

Public header comments:

- `include/sparse_matrix.h`
- `include/sparse_csr.h`
- `include/sparse_iterative.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`
- `include/sparse_eigs.h`

Planning and sprint evidence:

- `docs/planning/EPIC_17/SPRINT_194/PLAN.md`
- `docs/planning/EPIC_17/SPRINT_194/WORKING_NOTES.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day1-adoption-intake.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day2-adoption-friction-audit.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day3-support-matrix-contract.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day4-support-matrix-implementation.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day5-consumer-tutorial-audit.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day6-consumer-tutorial-implementation.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day7-diagnostics-wording-contract.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day8-direct-iterative-diagnostics-cleanup.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day9-qr-svd-eigs-diagnostics-cleanup.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day10-header-narrative-audit.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day11-header-narrative-cleanup.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day12-docs-examples-install-validation.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day13-full-quality-claim-calibration.md`
- `docs/planning/EPIC_17/SPRINT_194/artifacts/day14-closeout-handoff.md`

No `.c` implementation files were intentionally changed.

## Final Metrics

- Full sprint duration planned: 14 days.
- Maximum single-day estimate in the plan: 12 hours.
- Public headers cleaned: 6.
- User-facing documentation files updated: 8.
- Sprint artifacts produced: 14 daily artifacts plus working notes and plan.
- Install proof checks passed: 23.
- CMake install/export proof checks passed: 27.
- Selected oracle freshness rows checked: 54.
- Selected comparison freshness rows checked: 46.
- Benchmark/tooling build output from validation: 16 benchmark binaries and 14
  example binaries.

## Final Validation Evidence

Day 13 full gate:

```sh
make format && make lint && make test
```

Result: passed.

Day 13 affected-surface gate:

```sh
make docs-check api-docs-local-only qr-header-docs-guard \
  source-list-check ldlt-csc-helper-guard qr-external-ref-helper-guard \
  windows-powershell-guard tooling-build examples-build
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
bash tests/test_install.sh
bash tests/test_cmake_install.sh
make report-index-oracle-freshness report-index-comparison-freshness \
  bench-canonical-report-freshness
```

Result: passed.

Day 14 closeout hygiene:

```sh
git status --short --ignored build docs/api
git ls-files docs/api build
git diff --check
find . -maxdepth 4 -type d -name __pycache__ -print
```

Result: passed.

Evidence summary:

- `build/` and `docs/api/` remain ignored and untracked.
- No generated `build/` or `docs/api/` files are source-controlled.
- No Python `__pycache__` directories remain under the checked depth.
- `git diff --check` passed.

## Final Claim Boundaries

The final claim scan confirmed the edited docs and headers keep these
boundaries explicit:

- Package-manager distribution is not claimed.
- Homebrew/core, bottles, Linuxbrew, vcpkg, Conan, pkgsrc, distro/system
  package, and broad provider support are not claimed.
- Static install support is documented; shared-library artifacts, dynamic ABI
  compatibility, runtime-loader behavior, and static/shared selection are
  deferred or not claimed.
- Windows support remains CMake-first and hosted-lane bounded; Windows Makefile
  parity, Windows `pkg-config` execution parity, broad Windows report
  freshness, selected oracle freshness on Windows, and selected benchmark
  freshness on Windows are not claimed.
- Selected comparison and selected performance freshness lanes remain
  threshold-free, fixture-local, or hosted-lane scoped, not broad
  external-library parity or portable performance evidence.
- Public headers retain API contracts and local diagnostic boundaries without
  advertising broad adoption, package, platform, performance, or release
  claims.
- The project does not claim state-of-the-art status.

## Retrospective Inputs

What worked:

- The support/readiness matrix gives consumers a single bounded entry point for
  install, package, Windows, and evidence status.
- Daily artifacts preserved traceability from audit to contract to
  implementation to validation.
- Header cleanup reduced API review surface without changing declarations.
- Running the full C gate after header edits kept comment-only public-header
  changes under the same review discipline as code-adjacent API work.

Constraints:

- Local PowerShell execution is unavailable because `pwsh` is not installed.
- Markdown link checking is still not represented by a dedicated target.
- Generated API and build artifacts remain local/ignored outputs rather than
  checked-in evidence.

Closeout conclusion:

- Sprint 194 completed adoption/API coherence simplification.
- Remaining gaps are explicitly documented as residuals or future work rather
  than implied support.

## PR-Ready Summary

Suggested pull request summary:

- Add and document the Sprint 194 adoption/API coherence plan and daily
  evidence artifacts.
- Add an evidence-bounded install/support readiness matrix.
- Clarify first-solve, install, package, Windows, selected comparison,
  selected benchmark, and diagnostic wording across maintained docs.
- Reduce tutorial-style narrative in selected public headers while preserving
  declaration-adjacent API contracts.
- Record final validation, generated-output hygiene, and claim-calibration
  evidence.

Suggested validation section:

- `make format && make lint && make test`
- `make docs-check api-docs-local-only qr-header-docs-guard source-list-check ldlt-csc-helper-guard qr-external-ref-helper-guard windows-powershell-guard tooling-build examples-build`
- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_performance_docs.py`
- `python3 tests/test_normalize_report_index.py`
- `python3 tests/test_run_external_comparison.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_bench_canonical_freshness.py`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- `make report-index-oracle-freshness report-index-comparison-freshness bench-canonical-report-freshness`
- `git diff --check`

## Residual Handoff

- Local `pwsh` is unavailable; hosted CI remains the owner for PowerShell
  execution evidence.
- Add a dedicated Markdown link-check target in a future sprint if link
  integrity becomes part of the acceptance gate.
- Keep package-manager, shared-library/dynamic ABI, broad Windows parity,
  portable performance, release readiness, external-library parity, and
  state-of-the-art claims closed until future sprints add corresponding
  implementation and validation.
