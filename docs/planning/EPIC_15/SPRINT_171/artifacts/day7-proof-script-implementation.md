# Sprint 171 Day 7: Local Proof Script Implementation

## Purpose

Day 7 implements the local proof script designed on Day 6. Because Sprint 171
selected formal package-manager deferral, the implemented script validates the
deferral boundary rather than executing package-manager tooling.

## Implemented Script

`scripts/package_manager_deferral_check.sh`

The script is a focused package-manager non-claim guard. It follows the
existing guard-script style with `pass`, `fail`, `require_grep`, and
`require_absent_grep` helpers and emits `package-manager-deferral-check`
messages.

## Implemented Checks

| Check | Behavior |
| --- | --- |
| Deferral record | Requires `docs/planning/EPIC_15/SPRINT_171/artifacts/day5-package-manager-deferral.md`. |
| Deferral wording | Requires package-manager support deferral, provider non-claims, registry-readiness non-claim, evidence-to-revisit wording, downstream consumer proof, and guard coverage. |
| Provider recipe absence | Fails if unselected recipe artifacts appear outside planning, `.git`, build directories, or archive content. |
| Package metadata neutrality | Fails if `sparse.pc.in` or `cmake/SparseConfig.cmake.in` mention provider or package-manager distribution wording. |
| Public non-claims | Requires README, INSTALL, and maintainer guide to retain package-manager non-claim wording. |

## Provider Tool Policy

The script does not require or invoke:

- `vcpkg`;
- `brew`;
- `conan`;
- pkgsrc tooling;
- Debian, Fedora, RPM, or other distro packaging tools.

Absence of those tools is not a skip or failure because Sprint 171 selected
formal deferral, not provider support.

## Cleanup Behavior

The script does not create provider caches, package archives, build trees,
install prefixes, lockfiles, or binary packages. It only reads source files and
repository paths.

## Focused Validation

Day 7 validation commands:

```sh
bash -n scripts/package_manager_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

Expected result:

- package-manager deferral guard passes;
- static package/shared-library ABI guard still passes;
- diff hygiene passes.

## Day 7 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Local proof or deferral script | Complete | `scripts/package_manager_deferral_check.sh` was added. |
| Focused validation output | Complete | Validation commands and expected results are listed above. |
| Cleanup behavior | Complete | The script creates no provider or package outputs. |
| Day 7 proof-script artifact | Complete | This file. |

## Validation

Day 7 changed a shell script and planning artifacts. No `.c` or `.h` files
were modified, so the full C quality gate is not required for this day.

Validation command:

```sh
bash -n scripts/package_manager_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Provider proof or deferral enforcement is executable. | Complete | The new script is executable and validates formal deferral. |
| Failure output identifies unsupported claims clearly. | Complete | Failure messages name missing deferral wording, unselected recipe artifacts, metadata drift, or public non-claim drift. |
| Local validation passes or stops for user input. | Complete | Focused local validation passed. |
