# Sprint 144 Day 8 Package And Report Integration

## Purpose

Connect the selected macOS reviewed static-first install/export lane to
source-controlled package/report semantics without turning report rows into
fresh hosted-run evidence or broad platform parity claims.

## Inputs Reviewed

| Input | Day 8 finding |
| --- | --- |
| `docs/planning/EPIC_12/SPRINT_144/artifacts/day7-ci-promotion-implementation.md` | macOS workflow jobs are promoted and proof commands are unchanged. |
| `tests/corpus/manifests/report_families.tsv` | Package and CI rows are source-controlled advisory rows with separate freshness semantics. |
| `scripts/normalize_report_index.py` | Package rows expand into proof-owner rows; CI rows identify hosted workflow definitions whose logs live outside source control. |
| `tests/corpus/schemas/report_index_fields.md` | Contract rows are advisory and cannot manufacture pass evidence. |
| Generated package/CI normalized index output | Package rows already list static-first proof owners; CI row needed clearer selected-lane discoverability. |

## Row Update Decision

Day 8 updated only the CI report-family contract row in
`tests/corpus/manifests/report_families.tsv`.

The updated CI `claim_scope` now identifies:

- Linux source-of-truth lanes;
- macOS reviewed static-first install/export proof;
- Windows reviewed CMake subset lanes;
- hosted CI logs as external evidence.

The row still preserves these non-claims:

- no local report freshness proof;
- no claim from absent logs;
- no unsupported platform closure;
- no benchmark release claim.

## Package Row Decision

Package report rows were not changed.

Reason:

- `scripts/normalize_report_index.py --family package` already expands the
  package contract into six proof-owner rows:
  - `package_make_install_pkg_config_v1`;
  - `package_cmake_install_export_v1`;
  - `package_static_package_deferral_v1`;
  - `package_pkg_config_template_v1`;
  - `package_cmake_package_config_v1`;
  - `report_contract_package_static_install_package_install_proof_owner_v1`.
- Those rows correctly describe maintained static-first package proof owners.
- They should remain source-controlled advisory rows governed by schema and Git
  review.
- They should not become fresh hosted macOS CI evidence.

## Report Wording Audit

| Report family | Day 8 interpretation |
| --- | --- |
| `package` | Identifies maintained static-first package proof-owner commands and templates. It does not claim package-manager availability, shared-library ABI support, dynamic linking, or broad platform support. |
| `ci` | Identifies reviewed hosted workflow lane definitions, including the selected macOS reviewed static-first install/export proof. Hosted logs remain external evidence. |
| `documentation` | Remains an advisory interpretation anchor; Day 9 owns README, INSTALL, and maintainer guide wording updates. |

## Freshness Boundary

Day 8 preserves the Sprint 141 report-index contract:

- source-controlled package rows are advisory proof-owner metadata;
- source-controlled CI rows identify hosted checks and workflow definitions;
- hosted logs live outside source control;
- absent hosted logs do not become local pass evidence;
- report rows do not replace workflow execution.

## Validation

| Check | Result |
| --- | --- |
| `python3 scripts/normalize_report_index.py --family package --check` | Passed: 6 rows |
| `python3 scripts/normalize_report_index.py --family ci --check` | Passed: 1 row |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Passed: 6 source-controlled advisory rows |
| `python3 scripts/normalize_report_index.py --family ci --check-freshness` | Passed: 1 source-controlled advisory row |
| `git diff --check` | Passed |

## Selected-Lane Evidence References

| Evidence | Reference |
| --- | --- |
| macOS workflow lane definition | `.github/workflows/macos-ci.yml` |
| Make install/`pkg-config` proof owner | `tests/test_install.sh` |
| CMake install/export proof owner | `tests/test_cmake_install.sh` |
| Static package deferral proof owner | `scripts/static_package_deferral_check.sh` |
| CI report-family contract | `tests/corpus/manifests/report_families.tsv` |
| Day 7 CI implementation evidence | `docs/planning/EPIC_12/SPRINT_144/artifacts/day7-ci-promotion-implementation.md` |

## Day 9 Handoff

Day 9 should align public and maintainer documentation with the selected-lane
report interpretation:

1. README should say macOS now carries reviewed static-first install/export
   proof while Homebrew GCC remains supplemental.
2. INSTALL should update the supported-platform table and install validation
   interpretation.
3. The maintainer guide should update current support-tier guidance without
   erasing historical Sprint 112/133/143 non-claim context.
4. Documentation should continue to preserve Windows supplemental
   install/downstream status and Linux source-of-truth ownership.

## Day 8 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected-lane evidence is discoverable from report artifacts. | Complete | CI report-family row now names macOS reviewed static-first install/export proof; package rows already identify proof-owner scripts. |
| Report rows do not claim unsupported platform parity. | Complete | CI and package non-claims preserve absent-log, unsupported-platform, package-manager, shared-library ABI, dynamic-linking, and broad-platform boundaries. |
| Affected report normalization and freshness checks pass. | Complete | Package and CI normalization/freshness checks passed. |
