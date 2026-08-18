# Sprint 165 Day 14: Closeout And Handoff

## Purpose

Day 14 closes Sprint 165 by summarizing the package-boundary hardening work,
final changed files, validation evidence, residual product decisions, PR
description bullets, and Sprint 166 handoff.

## Sprint Outcome

Sprint 165 hardened the static-first package boundary without widening the
supported product surface. The sprint improved guard coverage, downstream
install proof, public and maintainer documentation, and evidence traceability
while preserving explicit non-claims for shared-library support, dynamic ABI
compatibility, runtime-loader behavior, package-manager distribution, Windows
Makefile parity, and Windows `pkg-config` command execution parity.

## Final Changed Files

Tracked sprint implementation changes:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- `include/sparse_cholesky.h`
- `scripts/static_package_deferral_check.sh`
- `tests/test_install.sh`

Sprint planning and evidence artifacts:

- `docs/planning/EPIC_14/SPRINT_165/PLAN.md`
- `docs/planning/EPIC_14/SPRINT_165/WORKING_NOTES.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day1-sprint-intake.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day2-package-metadata-audit.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day3-static-deferral-guard-design.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day4-static-deferral-guard-implementation.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day5-abi-non-claim-audit.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day6-abi-package-wording-cleanup.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day7-downstream-proof-scope.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day8-downstream-proof-implementation.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day9-package-documentation-alignment.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day10-package-metadata-validation.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day11-install-downstream-validation.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day12-quality-gate-drift-scan.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day13-evidence-review-residual-register.md`
- `docs/planning/EPIC_14/SPRINT_165/artifacts/day14-closeout-and-handoff.md`

## Completed Deliverables

| Deliverable | Status | Evidence |
| --- | --- | --- |
| Hardened static-first package boundary | Closed | Day 3 guard design, Day 4 guard implementation, Day 12 static-package guard pass. |
| Updated ABI/package non-claim audit | Closed | Day 5 ABI non-claim audit, Day 6 public-header wording cleanup, Day 12 claim-drift scan. |
| Refreshed downstream static package proof | Closed | Day 7 proof scope, Day 8 install-script update, Day 10/11 install and CMake downstream validation. |
| Package documentation aligned with supported behavior | Closed | Day 9 README/INSTALL/maintainer/CMake alignment and Day 12 package report-index checks. |
| Sprint 166 closeout handoff | Closed | Day 13 residual register and this Day 14 closeout. |

## Validation Evidence

Strongest completed local validation was recorded on Day 12:

```sh
make format && make lint && make test
```

Result: passed; `make test` ended with `All tests passed.`

Focused package validation also passed on Day 12:

```sh
bash scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
```

Results:

- static package deferral check passed;
- Make install/`pkg-config` validation passed with 23 passes and 0 failures;
- CMake install/export validation passed with 27 passes, 0 failures, and
  0 skips;
- package report-index check passed with 6 rows;
- package freshness check passed with 6 source-controlled advisory rows.

Day 14 documentation consistency validation:

```sh
git diff --check
```

Result: passed.

## Claim Boundary Status

Supported package reading after Sprint 165:

- local Unix-like Make install and `pkg-config` proof covers the maintained
  static archive, headers, generated package metadata, exact version checks,
  downstream compile/link/run behavior, and uninstall removal;
- local CMake install/export proof covers the maintained static archive,
  installed CMake package metadata, exact-version downstream consumers, and
  mismatched-version rejection;
- Windows package confidence remains CMake-first and hosted, with
  metadata-only `sparse.pc` inspection and explicit Make/`pkg-config`
  non-claims;
- public docs and maintainer docs describe static-first support without
  promoting shared-library, dynamic ABI, runtime-loader, package-manager, or
  broader platform claims.

Unsupported claims that remain bounded:

- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager distribution;
- static/shared selector UX;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` command execution parity;
- broader platform package parity.

## Epic 12 Reference Check

Sprint 165 lives in `docs/planning/EPIC_14/SPRINT_165/`. The original prompt
referenced the older Epic 12 project-plan path, so the sprint plan and working
notes carry explicit source-note explanations. Day 1 records the same
correction. No unexplained stale Epic 12 artifact path was found in the Sprint
165 folder during closeout.

## PR Description Outline

Suggested PR summary bullets:

- Harden static-first package-boundary drift checks around
  `BUILD_SHARED_LIBS`, shared metadata, ABI/export macros, package selectors,
  public docs, and Windows package non-claims.
- Strengthen Make install/`pkg-config` downstream proof with installed path
  identity checks, static archive link-flag checks, semantic output checks, and
  unsupported package/ABI wording rejection.
- Align README, INSTALL, maintainer guide, and CMake comments around the
  maintained static archive package surface.
- Preserve explicit non-claims for shared-library support, dynamic ABI
  compatibility, runtime-loader behavior, package-manager distribution,
  Windows Makefile parity, and Windows `pkg-config` command execution parity.
- Publish Sprint 165 planning artifacts, evidence checklist, residual
  register, and Sprint 166 closeout handoff.

Suggested validation bullets:

- `make format && make lint && make test`
- `bash scripts/static_package_deferral_check.sh`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- `python3 scripts/normalize_report_index.py --family package --check`
- `python3 scripts/normalize_report_index.py --family package --check-freshness`
- `git diff --check`

## Review-Risk Notes

| Risk | Review focus |
| --- | --- |
| Guard regexes become too broad and block valid static-only wording. | Check `scripts/static_package_deferral_check.sh` patterns against the intended static-first metadata scope. |
| Documentation wording could be read as a wider package or ABI support claim. | Review README, INSTALL, and maintainer-guide language for explicit non-claim boundaries. |
| Install-script path identity checks could be brittle on unusual temp-prefix formatting. | Review `tests/test_install.sh` normalization and same-directory checks against the macOS/Linux CI failures previously seen. |
| Windows package proof could be misread as Makefile or `pkg-config` command parity. | Confirm Windows wording remains CMake-first and metadata-only for `sparse.pc`. |
| Exact package version checks could be misread as dynamic ABI compatibility. | Confirm docs distinguish package resolution metadata from binary ABI policy. |

## Sprint 166 Handoff

Sprint 166 should consume Sprint 165 as package-boundary evidence during Epic 14
final validation:

1. Include Day 4, Day 8, Day 10, Day 11, Day 12, Day 13, and Day 14 artifacts
   in the final evidence inventory.
2. Re-run strongest feasible package checks if package files are touched:
   static deferral guard, Make install validation, CMake install validation,
   package report-index checks, and claim-drift scan.
3. Reconcile hosted package evidence separately from local package evidence,
   especially for Windows CMake-first validation and metadata-only `sparse.pc`
   inspection.
4. Preserve the residual register as future product scope in Epic 14 closeout,
   not as a hidden failure of the static-first package contract.
5. During the final public claim audit, search specifically for shared-library,
   dynamic ABI, runtime-loader, package-manager, Windows Makefile, Windows
   `pkg-config`, broad platform, and state-of-the-art wording.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 165 deliverables are closed or explicitly deferred. | Complete | Completed deliverables table and residual register identify closed work and deferred product decisions. |
| Package, ABI, shared-library, and package-manager claims remain bounded. | Complete | Claim boundary section preserves static-first support and unsupported non-claims. |
| Branch is ready for review with validation evidence. | Complete | Day 12 full and package gates passed; Day 14 closeout records PR bullets, review risks, and handoff. |
