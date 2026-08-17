# Sprint 162 Day 14 Closeout

## Scope

Day 14 closes Sprint 162 by recording the final retained-guard decision,
validation status, changed-file review, retrospective inputs, and Sprint 163
handoff.

## Final Product Decision

Sprint 162 retained both Windows Makefile install/uninstall parity and Windows
`pkg-config` command execution parity as explicit non-claims.

The supported Windows package surface is:

- CMake-first;
- static-first;
- reviewed through hosted Windows CMake configure/build/CTest;
- reviewed through hosted Windows CMake install/downstream validation;
- allowed to install and inspect `sparse.pc` as metadata only;
- not a Windows Makefile, `pkg-config`, package-manager, shared-library,
  dynamic ABI, runtime-loader, or broad Windows parity claim.

## Selected Guard Closeout

The selected implementation is a retained non-claim guard:

| Guard Surface | Closeout State |
| --- | --- |
| Static package guard | `scripts/static_package_deferral_check.sh` now checks Windows package non-claim wording and rejects unselected Windows package execution in the Windows workflow. |
| Windows workflow | `.github/workflows/windows-ci.yml` now describes `sparse.pc` validation as metadata-only and labels downstream proof as installed CMake package evidence. |
| README | Windows package wording separates metadata-only `sparse.pc` inspection from Windows `pkg-config` execution proof. |
| INSTALL | Support-tier wording identifies Windows CMake install/downstream validation and retained Windows Makefile/`pkg-config` non-claims. |
| Maintainer guide | Package history records the Sprint 162 retained non-claim boundary. |
| Sprint evidence | Day 1-14 artifacts and working notes trace the decision from intake through validation. |

## Final Validation Record

| Command | Result |
| --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `bash tests/test_install.sh` | Passed 23 checks |
| `bash tests/test_cmake_install.sh` | Passed 27 checks |
| `rg -n "[ \t]+$" docs/planning/EPIC_14/SPRINT_162 README.md INSTALL.md docs/maintainer_guide.md .github/workflows/windows-ci.yml scripts/static_package_deferral_check.sh` | Passed |
| `git diff --check` | Passed |
| `git status --short -- '*.c' '*.h'` | Passed with no output |

## Static Guard Output

```text
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: static install metadata ok
static-package-deferral-check: no shared export/ABI metadata found ok
static-package-deferral-check: package metadata has no static/shared selector ok
static-package-deferral-check: support wording remains deferred ok
static-package-deferral-check: Windows package non-claim wording ok
static-package-deferral-check: Windows workflow has no unselected package execution ok
static-package-deferral-check: passed
```

## Install Validation Summary

```text
Passed: 23
Failed: 0
ALL INSTALL TESTS PASSED
```

## CMake Install Validation Summary

```text
Passed: 27
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

## Changed-File Review

Implementation changes are intentionally narrow:

- `.github/workflows/windows-ci.yml`: wording and hosted-output diagnostics
  only, no new commands;
- `scripts/static_package_deferral_check.sh`: retained non-claim wording and
  no-unselected-execution guards;
- `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`: support-tier wording
  alignment;
- `docs/planning/EPIC_14/SPRINT_162/*`: plan, working notes, and sprint
  artifacts.

No C source, public header, package template, CMake install rule, Make install
script, or CMake install test was changed.

## Retrospective Input Set

Use these Sprint 162 artifacts for the retrospective:

- `PLAN.md`;
- `WORKING_NOTES.md`;
- `artifacts/day1-sprint-intake.md`;
- `artifacts/day2-windows-package-audit.md`;
- `artifacts/day3-metadata-boundary.md`;
- `artifacts/day4-product-decision.md`;
- `artifacts/day5-proof-or-guard-design.md`;
- `artifacts/day6-implementation-foundation.md`;
- `artifacts/day7-implementation-completion.md`;
- `artifacts/day8-ci-alignment.md`;
- `artifacts/day9-downstream-evidence.md`;
- `artifacts/day10-focused-validation.md`;
- `artifacts/day11-docs-alignment.md`;
- `artifacts/day12-cross-platform-validation.md`;
- `artifacts/day13-evidence-claim-review.md`;
- `artifacts/day14-closeout.md`.

## Sprint 163 Handoff

Sprint 163 is methodology-bound performance publication. The handoff is ready
with these constraints:

1. Performance publication must not cite package validation as performance,
   scalability, superiority, or broad platform evidence.
2. Windows package evidence remains CMake install/downstream scoped.
3. Windows `sparse.pc` inspection remains metadata-only.
4. Windows Makefile and `pkg-config` execution parity remain retained
   non-claims.
5. Package-manager, shared-library, dynamic ABI, runtime-loader, and broad
   Windows parity non-claims remain guarded.

## Day 14 Conclusion

Sprint 162 deliverables are complete and traceable. The final local validation
record passed, the retained guard implementation is in place, and Sprint 163
has a clear performance-publication handoff boundary.
