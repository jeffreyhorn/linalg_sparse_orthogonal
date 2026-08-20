# Sprint 171 Day 6: Local Proof Script Design

## Purpose

Day 6 designs the local validation path for the selected Sprint 171 package
manager decision. Because Day 3 selected formal package-manager deferral, the
local proof should be a deferral-enforcement script, not a provider install
script.

The script should prove that package-manager support remains unsupported unless
a future provider-specific decision and proof are added.

## Selected Script Shape

Day 7 should add a focused shell script:

`scripts/package_manager_deferral_check.sh`

The script should be executable locally and usable from future package-claim
validation. It should avoid invoking package-manager tooling because no
provider is selected.

## Script Responsibilities

| Responsibility | Design |
| --- | --- |
| Deferral record presence | Require `docs/planning/EPIC_15/SPRINT_171/artifacts/day5-package-manager-deferral.md`. |
| Deferral wording | Require tokens proving package-manager support is formally deferred and no provider is supported. |
| Future-evidence gate | Require the deferral record to list provider selection, source/checksum, license/version metadata, dependency policy, recipe/manifest, isolated install proof, downstream consumer proof, cleanup proof, docs, and guard coverage. |
| Provider recipe absence | Fail if unselected in-tree provider recipe paths appear. |
| Package metadata neutrality | Fail if `sparse.pc.in` or CMake package templates mention provider names or package-manager distribution support. |
| Public doc boundary | Require README, INSTALL, and maintainer guide to keep package-manager support as an explicit non-claim after Day 11 documentation work. |
| Sprint 170 preservation | Keep `scripts/static_package_deferral_check.sh` as the shared-library/static-package guard and run it separately in integrated validation. |

## Proposed Checks

### Positive Checks

The script should require:

- Day 5 deferral record exists;
- Day 5 record contains:
  - `Package-manager support is formally deferred`;
  - `No vcpkg`;
  - `Homebrew`;
  - `Conan`;
  - `pkgsrc`;
  - `provider registry readiness`;
  - `Evidence Needed To Revisit`;
  - `Downstream consumer proof`;
  - `Guard coverage`;
- README, INSTALL, and maintainer guide contain package-manager non-claim
  wording after Day 11 documentation updates;
- package metadata templates remain provider-neutral.

### Negative Checks

The script should fail if these unselected provider artifacts appear:

| Provider | Paths Or Files To Reject |
| --- | --- |
| vcpkg | `vcpkg.json`, `vcpkg-configuration.json`, `ports/`, `portfile.cmake` |
| Homebrew | `Formula/`, `*.rb` formula files under package/formula locations |
| Conan | `conanfile.py`, `conanfile.txt` |
| pkgsrc | `pkgsrc/`, `PLIST`, `distinfo` under provider package paths |
| Debian | `debian/control`, `debian/rules`, `debian/changelog` |
| Fedora/RPM | `*.spec` |

The script should also fail if package metadata templates contain provider or
distribution wording such as:

- `vcpkg`;
- `Homebrew`;
- `Conan`;
- `pkgsrc`;
- `apt`;
- `dnf`;
- `pacman`;
- `package-manager support`;
- `registry-ready`;
- `binary package`.

These negative checks should be scoped to provider-relevant paths and package
metadata templates. They should not scan planning artifacts, because planning
must discuss candidate providers and deferrals.

## Provider Tool Availability Policy

No provider tooling is required for the selected deferral path.

| Tool | Day 6 Policy |
| --- | --- |
| `vcpkg` | Not required; absence is not a skip or failure. |
| `brew` | Not required; absence is not a skip or failure. |
| `conan` | Not required; absence is not a skip or failure. |
| pkgsrc tooling | Not required; absence is not a skip or failure. |
| distro packaging tools | Not required; absence is not a skip or failure. |

If a future sprint selects a provider, it must define provider-tool discovery,
skip behavior, hosted proof behavior, and failure messages separately.

## Expected Output

The script should follow the existing guard style:

```text
package-manager-deferral-check: deferral record ok
package-manager-deferral-check: provider recipe absence ok
package-manager-deferral-check: package metadata neutrality ok
package-manager-deferral-check: package-manager public non-claims ok
package-manager-deferral-check: passed
```

On failure, it should print:

```text
package-manager-deferral-check: FAIL: <specific failure>
```

Failure messages should name the unsupported support claim or unselected
provider artifact clearly.

## Cleanup And Generated-Output Policy

The deferral script should not create build directories, install prefixes,
package archives, provider caches, lockfiles, or binary packages. Its only
temporary outputs, if needed, should live under `mktemp -d` and be removed by
`trap`.

Because no provider tooling is executed, Day 7 implementation should normally
not need cleanup beyond shell temporary files.

## Validation Plan

Day 7 implementation should run:

```sh
bash -n scripts/package_manager_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

If Day 7 changes `.c` or `.h` files, also run:

```sh
make format && make lint && make test
```

No `.c` or `.h` changes are expected for the deferral script.

## Day 6 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Local proof script design | Complete | A separate `scripts/package_manager_deferral_check.sh` deferral guard is specified. |
| Provider-tool availability policy | Complete | No provider tooling is required for formal deferral. |
| Expected pass/fail messages | Complete | Output and failure-message style are defined. |
| Cleanup and generated-output policy | Complete | The script should not create provider artifacts or package outputs. |
| Day 6 proof-script design artifact | Complete | This file. |

## Validation

Day 6 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Local proof behavior is scoped before implementation. | Complete | Day 7 script responsibilities and checks are listed. |
| Provider-tool absence behavior is explicit. | Complete | Provider tools are not required for the selected deferral path. |
| Validation cannot silently broaden package-manager claims. | Complete | The script is designed to fail on unselected recipes and provider claims. |
