# Sprint 171 Day 10: User Documentation Design

## Purpose

Day 10 designs the user-facing documentation update for the Sprint 171
package-manager decision. Sprint 171 selected formal package-manager deferral,
so the documentation must route users to the maintained static source-install
paths while making unsupported provider package-manager claims explicit.

## Current Documentation Map

| Document | Current Role | Package-Manager Treatment |
| --- | --- | --- |
| `README.md` | Short project front door and installation summary. | Already says install proof is not package-manager distribution evidence and Windows does not claim package-manager support. |
| `INSTALL.md` | Operational install, platform, validation, and package proof guide. | Already states package-manager distribution is out of scope and lists package proof owners. |
| `docs/maintainer_guide.md` | Maintainer-owned evidence, report-index, CI, and claim-boundary interpretation. | Already lists package-manager support as a non-claim and now owns the package-manager deferral guard. |
| `docs/tutorial.md` | First-use learning path. | Delegates static-first package details to `INSTALL.md` and avoids package-manager claims. |
| `docs/api_reference.md` | API reference index and claim-boundary summary. | States that the API reference does not imply package-manager distribution. |
| `docs/cookbook.md` | Workflow recipes. | Should stay workflow-focused and refer to `INSTALL.md` rather than duplicating package-manager policy. |

## Selected Documentation Posture

The Day 11 documentation implementation should use explicit deferral wording:

> Package-manager support is not currently provided. Use source install via
> Make or CMake and the maintained static archive package metadata until a
> future sprint selects and validates a specific provider recipe.

This wording should not imply that package-manager support is planned,
available, registry-ready, binary-package-ready, or supported on any platform.

## Quick-Start Versus Maintainer Detail

| Layer | Location | Intended Wording |
| --- | --- | --- |
| Quick start | `README.md` Installation section | One short sentence after the static install/downstream notes: no provider package-manager path is currently supported; use `INSTALL.md` for exact boundaries. |
| Operational guidance | `INSTALL.md` Start Here or Support Split | A concise package-manager deferral bullet that names unsupported provider families and points users to source install paths. |
| Validation ownership | `INSTALL.md` Verifying/Normalized Package Rows | Keep the new `scripts/package_manager_deferral_check.sh` proof-owner row and state it is a non-claim guard, not provider install proof. |
| Maintainer detail | `docs/maintainer_guide.md` package ownership section | Keep the guard ownership details and add a short rule: package-manager wording changes must run the package-manager deferral guard. |
| Tutorial/API/Cookbook | `docs/tutorial.md`, `docs/api_reference.md`, `docs/cookbook.md` | Avoid duplicating provider policy; only link to `INSTALL.md` when install/package boundaries matter. |

## Documentation Changes For Day 11

Day 11 should make these scoped edits:

1. Add a short package-manager deferral sentence to `README.md` near the
   installation summary, adjacent to the existing package proof/non-claim
   wording.
2. Add an explicit package-manager deferral bullet to `INSTALL.md#start-here`
   or `INSTALL.md#support-split`, naming vcpkg, Homebrew, Conan, pkgsrc,
   distro/system packages, provider registries, taps, recipes, and binary
   packages as unsupported.
3. Add one sentence to `INSTALL.md#normalized-package-rows` explaining that
   `scripts/package_manager_deferral_check.sh` guards non-claims and does not
   prove provider install behavior.
4. Add a maintainer rule in `docs/maintainer_guide.md`: changes to package
   manager wording, provider recipes, package metadata, or provider claims
   should run `bash scripts/package_manager_deferral_check.sh`.
5. Review `docs/tutorial.md`, `docs/api_reference.md`, and
   `docs/cookbook.md`; edit only if they imply package-manager support or lack
   a needed `INSTALL.md` handoff.

## Claim Scan Plan

Day 11 should run targeted scans after editing:

```sh
rg -n "package-manager|package manager|vcpkg|Homebrew|Conan|pkgsrc|apt|dnf|pacman|binary package|registry|tap" \
  README.md INSTALL.md docs/maintainer_guide.md docs/tutorial.md docs/api_reference.md docs/cookbook.md
rg -n "shared-library|dynamic ABI|runtime-loader|BUILD_SHARED_LIBS|static/shared selector|Windows Makefile|Windows.*pkg-config" \
  README.md INSTALL.md docs/maintainer_guide.md docs/tutorial.md docs/api_reference.md docs/cookbook.md
```

The scan is informational. Any allowed match must preserve one of these
meanings:

- unsupported package-manager provider support;
- maintained static source-install path;
- static CMake/`pkg-config` package metadata;
- explicit shared-library/dynamic ABI/runtime-loader deferral;
- explicit Windows Makefile or Windows `pkg-config` execution non-claim.

## Validation Commands For Day 11

Day 11 should run:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
```

If Day 11 unexpectedly changes any `.c` or `.h` files, it must also run:

```sh
make format
make lint
make test
```

No `.c` or `.h` changes are expected for the documentation implementation.

## Day 10 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Documentation update plan | Complete | Day 11 target documents and edit scope are listed above. |
| Quick-start and maintainer-detail split | Complete | README, INSTALL, maintainer guide, tutorial/API/cookbook ownership is separated. |
| Documentation claim-scan plan | Complete | Targeted scans and allowed interpretations are defined. |
| Day 10 documentation-design artifact | Complete | This file. |

## Validation

Day 10 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Documentation changes are scoped before editing. | Complete | Day 11 edits are limited to README, INSTALL, maintainer guide, and only necessary tutorial/API/cookbook handoffs. |
| User-facing wording cannot imply unsupported providers. | Complete | The selected wording states formal deferral and names unsupported provider families. |
| Performance, ABI, and runtime-loader claims remain separate. | Complete | Claim scans include shared-library, dynamic ABI, runtime-loader, and Windows parity boundaries. |
