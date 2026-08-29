# Sprint 187 Day 7: Package Acceptance Gates

## Purpose

Define the exact Sprint 188 acceptance gates for the selected Homebrew proof
completion target. These gates decide when package-facing wording can move
from guarded non-claim to evidence-backed local Homebrew source formula proof.

## Selected Scope

Sprint 188 may close only this package gap:

- local Homebrew source formula proof for the maintained static archive
  package surface.

Sprint 188 must not turn that proof into broad package-manager availability,
Homebrew/core readiness, hosted binary distribution, shared-library support, or
dynamic ABI support.

## Gate 1: License Metadata Blocker

| Requirement | Acceptance criteria | Failure state |
| --- | --- | --- |
| Standalone license file | The repository root contains an approved `LICENSE`, `COPYING`, or `NOTICE` file that can be included in the local proof source archive. | `scripts/homebrew_local_formula_proof.sh` exits `2` with the missing standalone license message and public docs keep Homebrew support unclaimed. |
| Homebrew license identifier | `SPARSE_HOMEBREW_LICENSE` is set to accurate local-proof license metadata accepted by the rendered formula. | The proof exits `2` with the missing environment metadata message and package-manager support remains a guarded non-claim. |
| Archive inclusion | The local source archive produced by the proof script includes the selected standalone license file. | The proof cannot be counted as package evidence. |
| Claim update | Any README, INSTALL, or Homebrew README wording states exactly the proven local formula scope and the selected license strategy. | Package support wording must remain blocked or be reverted to non-claim language. |

The license gate closes only when the proof reaches formula render/install/test
success with accurate metadata. A continued `2` unavailable exit is acceptable
only as a guarded blocker outcome, not as package support evidence.

## Gate 2: Homebrew Formula Material

| Surface | Acceptance criteria |
| --- | --- |
| `packaging/homebrew/sparse-lu-ortho.rb.in` | Remains a source-controlled template, not an installable committed formula. |
| Formula placeholders | Keeps placeholders for homepage, local archive URL, archive SHA-256, project version, and Homebrew license metadata. |
| Formula install path | Builds with CMake, installs the maintained static archive package surface, and rejects shared artifacts. |
| Formula `test do` | Builds and runs a downstream CMake consumer with exact-version `find_package(Sparse ...)` and `Sparse::sparse_lu_ortho`. |
| Generated outputs | Rendered formula files, local taps, archives, logs, caches, build trees, install prefixes, and bottle outputs are not committed. |

The formula gate is failed by any committed provider recipe that implies a
public tap, Homebrew/core submission, bottle, Linuxbrew, or broad package
manager route.

## Gate 3: Proof Script Behavior

`scripts/homebrew_local_formula_proof.sh` must prove this sequence when the
local environment and license metadata are available:

1. Identify the project root, formula template, and `VERSION`.
2. Require `brew`, `cmake`, `ruby`, `tar`, a C compiler, and a SHA-256 tool.
3. Verify all required formula placeholders are present.
4. Create a temporary source archive with the source, package inputs, examples,
   and standalone license metadata.
5. Compute and inject the archive SHA-256 checksum.
6. Render a temporary local formula with no unresolved placeholders.
7. Install the formula from source with Homebrew.
8. Verify the installed static archive, headers, CMake package files, and
   `sparse.pc`.
9. Reject shared-library artifacts and unsupported provider/ABI wording in
   installed metadata.
10. Run `brew test` for the downstream CMake consumer.
11. Uninstall the temporary formula.
12. Clean temporary proof outputs unless `--keep-temp` is selected.

Accepted proof-script outcomes:

| Exit | Meaning | Package claim effect |
| ---: | --- | --- |
| `0` | Local Homebrew source formula proof passed for the static package surface. | Documentation may promote the exact local proof scope. |
| `2` | Required local environment or license metadata is unavailable. | Documentation must keep Homebrew support unclaimed and identify the blocker. |
| Any other nonzero exit | Proof failed. | Sprint 188 must stop or fix the failure before promotion. |

## Gate 4: Package Guards

| Guard | Required role |
| --- | --- |
| `scripts/package_manager_deferral_check.sh` | Preserves the package-manager deferral record, selected local Homebrew proof boundary, absence of unselected provider recipes, metadata neutrality, and public non-claim wording. |
| `scripts/static_package_deferral_check.sh` | Preserves the static-first package contract, rejects `BUILD_SHARED_LIBS=ON`, blocks shared artifacts/selectors/export macros, and keeps dynamic ABI support deferred. |
| `python3 scripts/normalize_report_index.py --family package --check` | Validates package report rows when Sprint 188 changes report-owner metadata. |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Validates package report freshness when Sprint 188 changes package report artifacts. |

Package support can be promoted only if the guard set still passes after the
Homebrew proof change.

## Gate 5: Documentation And Claim Wording

| Surface | Required wording |
| --- | --- |
| `README.md` | States source install and static package support first; mentions local Homebrew only as the evidence-backed local formula proof state. |
| `INSTALL.md` | Gives the exact package boundary, required proof command, retained non-claims, and static-first installed package behavior. |
| `packaging/homebrew/README.md` | Explains the template/proof workflow, temporary artifacts, license metadata requirement, and unsupported provider surfaces. |
| `docs/maintainer_guide.md` | Gives maintainers the proof command, claim boundary, and guard commands needed before changing package wording. |

Documentation must not describe the Homebrew template as a public install path
unless the exact local proof command has passed in the same support state.

## Required Sprint 188 Validation Commands

Minimum package gate commands:

```sh
SPARSE_HOMEBREW_LICENSE=<accurate-id> scripts/homebrew_local_formula_proof.sh
scripts/package_manager_deferral_check.sh
scripts/static_package_deferral_check.sh
```

Additional commands when matching files change:

```sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
make format
make lint
make test
```

`make format && make lint && make test` is required whenever Sprint 188 changes
`.c` or `.h` files. Documentation-only changes may use documentation and guard
checks instead.

## Retained Package Non-Claims

Sprint 188 must retain these non-claims unless a future epic selects separate
evidence:

- Homebrew/core submission or acceptance.
- Bottles or hosted binary artifacts.
- Linuxbrew support.
- Public tap maintenance.
- vcpkg, Conan, pkgsrc, apt, dnf, pacman, or distro packaging.
- Provider registry readiness.
- Binary package install/update/uninstall support.
- Shared-library package support.
- Dynamic ABI compatibility.
- Static/shared package selector support.
- Broad package-manager support.

## Claim Promotion Rule

Package wording may be promoted only to:

> Local Homebrew source formula proof for the maintained static archive package
> surface has passed.

It must still be paired with:

> This does not claim Homebrew/core, bottles, Linuxbrew, public taps, binary
> packages, other package managers, shared libraries, dynamic ABI support, or
> broad package-manager distribution.

If the proof exits `2`, wording must instead say the local Homebrew proof
remains blocked by missing local environment or license metadata and is not a
user-facing install route.

## Sprint 188 Completion Gate

Sprint 188 is complete when one of these states is true:

1. The local Homebrew proof passes, package guards pass, and public docs promote
   only the exact local static source formula proof.
2. The license/formula blocker remains, package guards pass, public docs keep
   Homebrew unclaimed, and the blocker is recorded with revisit criteria.

Any failed proof, failed guard, committed generated proof output, or widened
package-manager claim blocks completion.

## Validation

Day 7 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
