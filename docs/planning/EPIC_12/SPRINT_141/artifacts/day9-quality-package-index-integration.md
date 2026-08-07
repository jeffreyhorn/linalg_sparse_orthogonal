# Day 9 Quality And Package Index Integration

## Purpose

Day 9 extends the normalized report index to coverage, dead-code, and
package/install report families. The implementation distinguishes
source-controlled package proof owners from local generated quality reports,
and it keeps missing coverage/dead-code outputs deterministic rather than
silently omitting them.

## Implemented Surfaces

| Surface | Change | Purpose |
| --- | --- | --- |
| `scripts/normalize_report_index.py` | Added coverage, dead-code, and package proof-owner row emitters. | Normalizes stable quality/package families while preserving advisory/static-first boundaries. |
| `tests/test_normalize_report_index.py` | Added quality/package fixtures and required-missing coverage check. | Verifies package source-controlled rows, dead-code generated rows, and deterministic missing coverage diagnostics. |
| `docs/planning/EPIC_12/SPRINT_141/artifacts/day9-quality-package-index-integration.md` | Added this integration artifact. | Records Day 9 behavior, validation, and handoff. |
| `docs/planning/EPIC_12/SPRINT_141/WORKING_NOTES.md` | Updated Day 9 notes. | Keeps sprint evidence current. |

## Row Mapping

| Input | Normalized row behavior |
| --- | --- |
| `coverage/coverage-src.info` | Emits an advisory `coverage_*_v1` row when present, with unknown backend-from-artifact context and `generated_present_unchecked` freshness. |
| Missing coverage output | Emits deterministic `not_generated` rows by default; `--require-generated coverage --check` returns nonzero with a family-specific diagnostic. |
| `build/deadcode/report.tsv` | Emits `deadcode_*_v1` rows preserving bucket, tool, symbol/path, line, detail, disposition, artifact path, advisory status, and zero-dead-code non-claim. |
| `tests/test_install.sh` | Emits `package_make_install_pkg_config_v1` as a source-controlled static-first proof-owner row. |
| `tests/test_cmake_install.sh` | Emits `package_cmake_install_export_v1` as a source-controlled CMake install/export proof-owner row. |
| `sparse.pc.in` | Emits `package_pkg_config_template_v1` as source-controlled pkg-config metadata template evidence. |
| `cmake/SparseConfig.cmake.in` | Emits `package_cmake_package_config_v1` as source-controlled CMake package config template evidence. |
| `scripts/static_package_deferral_check.sh` | Emits `package_static_package_deferral_v1` as a source-controlled guardrail against unsupported shared-library or ABI claims. |

## Package Scope

Package rows are source-controlled proof-owner rows, not generated install
proof. They preserve the current static-first contract:

- Make install and pkg-config proof are owned by `tests/test_install.sh`;
- CMake install/export and `find_package(Sparse)` proof are owned by
  `tests/test_cmake_install.sh`;
- `sparse.pc.in` and `cmake/SparseConfig.cmake.in` are metadata templates;
- static-package deferral checks guard against unsupported shared-library,
  dynamic-linking, package-manager, or ABI claims.

## Quality Scope

Coverage and dead-code rows remain local or advisory:

- coverage artifacts are local tool output and do not claim coverage
  completeness, branch coverage parity, hosted platform proof, or product
  quality;
- dead-code report rows classify local static-analysis output and do not claim
  zero dead code, semantic correctness, release quality, or platform support;
- freshness comparison remains deferred to Sprint 141 Day 10/11.

## Test Coverage

The focused test suite now covers:

- generated dead-code rows from a synthetic `build/deadcode/report.tsv`;
- package proof-owner rows for Make install/pkg-config and CMake
  install/export;
- source-controlled package rows using `freshness_status=source_controlled`;
- coverage missing rows using `freshness_status=not_generated`;
- required generated coverage failure with diagnostic:
  `required generated family missing: coverage`.

## Validation Evidence

Commands run:

```sh
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --no-generated --check
python3 scripts/normalize_report_index.py --family coverage --no-generated --require-generated coverage --check
```

Results:

- focused normalized-index tests passed;
- quality/package no-generated check reported `10` rows;
- required coverage command returned the expected nonzero diagnostic:
  `normalize-report-index: required generated family missing: coverage`.

## Day 10 Handoff

Day 10 should define the freshness severity model for rows that currently use
`generated_present_unchecked` or `not_generated`. Coverage and dead-code should
remain advisory unless an explicit reviewed gate requires them. Package
proof-owner rows should remain source-controlled unless a later lane generates
and indexes install-run result logs.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Quality and package report families are represented without overstating freshness or platform coverage. | Complete | Coverage and dead-code rows are advisory/generated-local; package rows are source-controlled proof owners. |
| Install/package proof rows distinguish source-controlled checks from local generated output. | Complete | Package rows expand to maintained scripts/templates with `freshness_status=source_controlled`. |
| Stale or missing rows produce deterministic validation messages. | Complete | Missing coverage emits `not_generated`; required coverage check reports `required generated family missing: coverage`. |
