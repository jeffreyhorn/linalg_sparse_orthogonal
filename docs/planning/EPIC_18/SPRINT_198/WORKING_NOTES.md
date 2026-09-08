# Sprint 198 Working Notes

## Sprint Scope

Sprint 198 closes the Homebrew/package-manager blocker by adding approved
license metadata, proving the selected local Homebrew formula workflow, and
promoting only the earned support claim.

## Item Checklist

| Item | Description | Owner artifacts | Status |
| --- | --- | --- | --- |
| 198.1 | Record approved root license metadata and exact Homebrew formula license identifier. | Day 2 decision artifact; root `LICENSE`; `packaging/homebrew/README.md`; `scripts/homebrew_local_formula_proof.sh` | Complete for MIT path: root `LICENSE` exists and the selected local proof identifier is `SPARSE_HOMEBREW_LICENSE=MIT`. |
| 198.2 | Add or update root license files and formula metadata according to the approved decision. | Root metadata file; `packaging/homebrew/sparse-lu-ortho.rb.in`; proof-script render inputs; package docs | Complete for local proof metadata: the proof injects `MIT` into the rendered temporary formula and rejects placeholders. |
| 198.3 | Run and harden the local Homebrew proof through archive/checksum, render, install, `brew test`, uninstall, and cleanup. | `scripts/homebrew_local_formula_proof.sh`; `packaging/homebrew/sparse-lu-ortho.rb.in`; proof logs; Day 5 through Day 9 artifacts; Day 14 addendum | Complete for developer-mode local static source formula proof: `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT` reaches archive, checksum, temporary tap render, source install, installed static surface validation, downstream `brew test`, uninstall, cleanup, and exit `0` on macOS Intel x86_64 Tier 3 Homebrew. |
| 198.4 | Update package-manager and static-package guards for exact metadata and proof-backed wording. | `scripts/package_manager_deferral_check.sh`; `scripts/static_package_deferral_check.sh`; focused guard logs | Complete for bounded proof state: package guard asserts MIT metadata, temporary tap rendering, developer-mode proof invocation, proof-stage completion, and retained broader non-claims. |
| 198.5 | Update public and maintainer docs with exact earned support tier and retained non-claims. | `README.md`; `INSTALL.md`; `packaging/homebrew/README.md`; `docs/maintainer_guide.md` | Complete for bounded proof state: docs now reference the developer-mode local static source formula proof and retain all broader package-manager non-claims. |
| 198.6 | Run Homebrew proof, package guards, install checks, docs checks, and C gate if `.c` or `.h` files changed. | Validation logs; Day 13 artifact; Day 14 closeout artifact | Complete for current guarded surfaces: package/static guards pass, install/docs checks pass, the developer-mode MIT Homebrew proof exits `0`, and no C gate is required because no `.c` or `.h` files changed. |

## Evidence Ledger

| Date / Day | Evidence | Result | Notes |
| --- | --- | --- | --- |
| Day 1 | `bash scripts/homebrew_local_formula_proof.sh` | Exit `2` | Expected unavailable blocker: no standalone `LICENSE`, `COPYING`, or `NOTICE` exists for provider metadata. The script keeps local Homebrew proof unclaimed. |
| Day 1 | `bash scripts/package_manager_deferral_check.sh` | Exit `0` | Deferral record, provider recipe absence, selected Homebrew boundary, package metadata neutrality, and public non-claims pass. |
| Day 1 | `bash scripts/static_package_deferral_check.sh` | Exit `0` | Static-first package contract and shared-library/dynamic ABI non-claims pass. |
| Day 1 | Tool discovery for `brew`, `cmake`, `ruby`, `tar`, `shasum`, and `cc` | Available | Current blocker is metadata, not missing local tools. |
| Day 2 | Root metadata search | No files found | No root `LICENSE`, `COPYING`, or `NOTICE` exists. |
| Day 2 | License reference audit | No approved package identifier found | README has a research/educational purpose sentence, but no standalone license terms or Homebrew/SPDX-style identifier. |
| Day 2 | `day2-license-metadata-decision.md` | Decision blocker recorded | No exact Homebrew license identifier is selected; proof remains unavailable and unclaimed until approved inputs exist. |
| Day 3 | `find . -maxdepth 1 \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \)` | No files found | Root metadata implementation remains blocked. |
| Day 3 | `scripts/homebrew_local_formula_proof.sh` source review | Guarded | Missing root metadata stops proof before archive/render/install/test work; detected metadata would be added to the archive and verified. |
| Day 3 | `day3-root-metadata-implementation.md` | Blocker implementation record added | No guessed root license file or Homebrew identifier was added. |
| Day 4 | Formula template and render-path review | Guarded | Formula metadata remains placeholder-backed until approved license inputs exist; render rejects empty and unresolved replacements. |
| Day 4 | `day4-formula-metadata-wiring.md` | Formula wiring blocker recorded | No formula license promotion was made because no approved identifier exists. |
| Day 5 | Archive/checksum proof-path review | Guarded | Source archive creation remains correctly gated behind root metadata detection; approved metadata entries would be included and verified once available. |
| Day 5 | Generated Homebrew proof output scan | No files found | No rendered formula, archive, log, bottle, or nested `Formula` output exists under `packaging/homebrew`. |
| Day 5 | `day5-archive-checksum-proof.md` | Archive/checksum blocker record added | Required archive entries, checksum behavior, diagnostics, and generated-output boundaries are documented. |
| Day 6 | `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` | Exit `0` | Formula template syntax is valid. |
| Day 6 | Render placeholder contract review | Guarded | Required placeholders are present; render rejects empty replacements and unresolved `__SPARSE_*__` tokens. |
| Day 6 | Negative proof runs with unset, `NOASSERTION`, and `PLACEHOLDER` license values | Exit `2` | All stop at missing root metadata before render; placeholder-license rejection remains guarded behind root metadata availability. |
| Day 6 | `day6-formula-render-validation.md` | Render-validation blocker record added | Rendering is validated as protected, not as successful formula output. |
| Day 7 | Install-surface proof review | Guarded | Installed headers, static archive, CMake package files, `sparse.pc`, static target metadata, shared-artifact rejection, and cleanup behavior are owned by the proof script. |
| Day 7 | Generated Homebrew proof output scan | No files found | No rendered formula, archive, log, bottle, or nested `Formula` output exists under `packaging/homebrew`. |
| Day 7 | `day7-install-surface-proof.md` | Install-surface blocker record added | Install execution remains blocked until approved metadata allows render/install proof. |
| Day 8 | Formula `test do` downstream consumer review | Guarded | Exact-version `find_package`, imported target linkage, installed public headers, static artifact checks, and shared-artifact rejection remain present. |
| Day 8 | Proof-script formula test contract review | Guarded | The proof script checks the `test do` contract before metadata detection proceeds to archive/render/install/test work. |
| Day 8 | `day8-downstream-formula-test-proof.md` | Downstream test blocker record added | `brew test` execution remains blocked until a rendered and installed formula exists. |
| Day 9 | `bash scripts/homebrew_local_formula_proof.sh` | Exit `2` | End-to-end proof attempt stops at missing root metadata; archive, checksum, render, install, static surface validation, `brew test`, and uninstall are not reached. |
| Day 9 | Tool and version snapshot | Available / `2.2.0` | `brew`, `cmake`, `ruby`, `tar`, `shasum`, and `cc` are available; blocker is metadata, not environment setup. |
| Day 9 | Generated Homebrew proof output scan | No files found | No rendered formula, archive, log, bottle, or nested `Formula` output exists under `packaging/homebrew`. |
| Day 9 | `day9-end-to-end-proof-run.md` | End-to-end proof blocker record added | Item 198.3 remains incomplete because approved metadata is missing. |
| Day 10 | `scripts/package_manager_deferral_check.sh` update | Guard promoted | The guard now asserts Sprint 198 metadata-blocker records and retained non-claims. |
| Day 10 | Static package guard review | No change needed | Existing guard already preserves static package scope and shared-library/dynamic ABI non-claims. |
| Day 10 | `day10-package-guard-promotion.md` | Guard-promotion record added | Day 10 promotes guard coverage, not Homebrew support. |
| Day 11 | Public package docs update | Claim-safe blocker wording | README, INSTALL, and Homebrew README now reference Sprint 198 blocker evidence and retain package-manager non-claims. |
| Day 11 | `day11-public-package-docs.md` | Public documentation record added | No Homebrew support tier is promoted. |
| Day 11 | Public-doc validation | Passed | Package-manager guard, static-package guard, docs check, and diff whitespace check passed. |
| Day 12 | Maintainer guidance update | Claim-safe blocker wording | Maintainer guide now includes Sprint 198 blocker artifacts and the current package proof runbook status. |
| Day 12 | Item status ledger | Updated | Items 198.1 through 198.6 are classified as blocked, partial for blocker state, or in progress for final validation. |
| Day 12 | `day12-maintainer-planning-alignment.md` | Maintainer/planning record added | Residual package gaps and retrospective inputs are recorded. |
| Day 12 | Maintainer/planning validation | Passed | Package-manager guard, static-package guard, docs check, and diff whitespace check passed. |
| Day 13 | `bash scripts/homebrew_local_formula_proof.sh` | Exit `2` | Expected metadata blocker; full proof stages remain unearned. |
| Day 13 | Package/static guards | Passed | Package-manager and static-package guards passed. |
| Day 13 | Install checks | Passed | `tests/test_install.sh` passed 23/23; `tests/test_cmake_install.sh` passed 27/27 with 0 skipped. |
| Day 13 | Docs and whitespace checks | Passed | `make docs-check` and `git diff --check` passed. |
| Day 13 | `day13-integrated-validation.md` | Integrated validation record added | Current guarded surfaces are validated; Homebrew support remains unclaimed. |
| Day 14 | Sprint 198 item closeout | Closed with proof residual | Items 198.1 and 198.2 are complete for the MIT path, item 198.3 remains blocked at local install/test proof, items 198.4 and 198.5 are partial for the unpromoted proof state, and item 198.6 is complete for current guarded surfaces. |
| Day 14 | Generated artifact and staging review | Clean for Homebrew proof outputs | Root MIT license metadata now exists; no generated Homebrew proof outputs exist; `scripts/__pycache__/` remains untracked and must not be staged. |
| Day 14 | `day14-closeout-review.md` | Closeout review added | Retrospective inputs and retained non-claims are recorded. |
| Post-closeout | Root MIT metadata | Added | Root `LICENSE` now contains MIT metadata for Jeffrey Horn. |
| Post-closeout | `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh` | Exit `0` on this host | Metadata detection, archive creation, checksum calculation, temporary tap creation, formula rendering, source install, installed static surface validation, downstream `brew test`, uninstall, and cleanup complete on macOS Intel x86_64 Tier 3 Homebrew. |

## Owner Surface Inventory

| Surface | Owner files | Day 1 state |
| --- | --- | --- |
| Root license metadata | `LICENSE`, `COPYING`, `NOTICE` | Root `LICENSE` is present with MIT metadata. |
| Version metadata | `VERSION` | Present and available to the proof script. |
| Homebrew formula template | `packaging/homebrew/sparse-lu-ortho.rb.in` | Present. The template remains a temporary local formula and carries placeholder-driven metadata. |
| Homebrew proof script | `scripts/homebrew_local_formula_proof.sh` | Present and executable. It stops claim-safely before archive/render/install/test work when root license metadata is absent. |
| Package-manager guard | `scripts/package_manager_deferral_check.sh` | Present and passing for the current non-claim state. |
| Static-package guard | `scripts/static_package_deferral_check.sh` | Present and passing for the static-first package boundary. |
| Public docs | `README.md`, `INSTALL.md`, `packaging/homebrew/README.md` | Present. Current wording records the developer-mode local static source formula proof while keeping broad package-manager support unclaimed. |
| Maintainer docs | `docs/maintainer_guide.md` | Present. Current guidance records the selected Homebrew proof script, MIT metadata path, developer-mode Tier 3 Intel proof scope, and retained non-claims. |
| Prior package proof evidence | `docs/planning/EPIC_17/SPRINT_188/*`; `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Prior Sprint 188 proof work and Epic 18 residual queue both identify standalone license metadata as the near-term closure target. |

## Validation Matrix

| Validation | Trigger | Day 1 baseline |
| --- | --- | --- |
| `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh` | Any Homebrew proof or metadata change | Current developer-mode local proof exits `0` after archive/checksum/temporary-tap render, install, installed-surface validation, downstream `brew test`, uninstall, and cleanup on macOS Intel x86_64 Tier 3 Homebrew. |
| `bash scripts/package_manager_deferral_check.sh` | Package-manager docs, formula, proof, or guard edits | Passing. |
| `bash scripts/static_package_deferral_check.sh` | Package/install wording or package metadata edits | Passing. |
| `bash tests/test_install.sh` | Install/package behavior changes | Deferred until implementation changes. |
| `bash tests/test_cmake_install.sh` | CMake package metadata or install surface changes | Deferred until implementation changes. |
| `make docs-check` | Public or maintainer documentation changes | Deferred until documentation promotion days. |
| `make format && make lint && make test` | Any `.c` or `.h` change | Not required for Day 1 documentation-only changes. |

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| License metadata is guessed instead of approved. | Formula metadata could misrepresent the project license. | Day 2 must record the approved root metadata and exact Homebrew license identifier before implementation. |
| Local proof success is broadened into package-manager support. | Public docs could overclaim Homebrew/core, bottles, Linuxbrew, tap, or provider readiness. | Guards and docs must retain explicit non-claims outside the proven local formula path. |
| Generated proof outputs are committed. | Repository gains temporary archives, formulas, logs, or cache files. | Day 14 closeout must verify no generated proof artifacts are staged. |
| Proof depends on source-tree leakage. | The formula test would not prove installed package usability. | Day 8 must validate exact-version CMake consumer behavior against installed artifacts. |
| Environment unavailability is mistaken for proof failure or success. | Evidence interpretation becomes ambiguous. | Validation logs must classify unavailable Homebrew environments as residuals, not successes. |

## Open Questions

1. What exact root license metadata file is approved for this repository?
2. What exact Homebrew license identifier should populate
   `SPARSE_HOMEBREW_LICENSE`?
3. Should guards enforce one expected identifier or only reject placeholders?
4. What exact public support tier is earned if the local formula proof exits
   `0`?
5. Which residual package-manager claims must remain out of scope even after a
   local proof passes?

## Retained Non-Claims

Sprint 198 must not claim Homebrew/core readiness, bottles, Linuxbrew support,
public tap maintenance, binary package distribution, vcpkg, Conan, pkgsrc,
distro/system package support, provider registry readiness, shared-library
package support, dynamic ABI compatibility, runtime-loader behavior, broad
package-manager support, or state-of-the-art package ecosystem parity.

## Day Log

### Day 1: Package Metadata Intake

- Created the Sprint 198 working-notes scaffold.
- Mapped project-plan items 198.1 through 198.6 to owner artifacts.
- Inventoried current package-manager, Homebrew proof, static package, public
  docs, maintainer docs, and prior evidence surfaces.
- Confirmed local `brew`, `cmake`, `ruby`, `tar`, `shasum`, and `cc` are
  available.
- Ran the Homebrew proof and package guards in observation mode.
- Recorded that the current blocker is missing standalone root license
  metadata, not missing local tools.

### Day 2: License Metadata Decision

- Audited root metadata, README license wording, install guidance, Homebrew
  template metadata, proof-script requirements, package docs, maintainer
  guidance, and prior Sprint 188/Epic 18 residual records.
- Found no approved standalone root `LICENSE`, `COPYING`, or `NOTICE`.
- Found no exact Homebrew formula license identifier in source-controlled
  metadata.
- Recorded that Day 2 cannot select a license file or formula identifier
  without project-owner/legal approval.
- Preserved the current proof behavior as the correct claim-safe state: exit
  `2` before archive/render/install/test work and keep local Homebrew proof
  unclaimed.

### Day 3: Root Metadata Implementation

- Rechecked the repository root for `LICENSE`, `COPYING`, and `NOTICE`; none
  exists.
- Reviewed Homebrew proof archive handling and confirmed approved root metadata
  entries would be added to the source archive and verified once they exist.
- Did not add a root license file or formula identifier because Day 2 recorded
  no approved license inputs.
- Added a Day 3 implementation-blocker artifact that preserves the fail-safe
  package proof state.
- Updated the Epic 18 project-plan status row for Sprint 198 so it no longer
  says no Sprint 198 artifact directory exists on this branch.

### Day 4: Formula Metadata Wiring

- Reviewed the Homebrew formula template and render path.
- Confirmed homepage, archive URL, SHA-256, version, and license values remain
  explicit render-time placeholders.
- Confirmed the proof script requires `__SPARSE_HOMEBREW_LICENSE__`, rejects
  empty/unresolved render substitutions, and treats placeholder license values
  as unavailable blocker evidence.
- Did not promote formula license metadata because Day 2 and Day 3 found no
  approved root metadata or exact Homebrew identifier.
- Recorded the static local source proof boundary and Day 5 archive/checksum
  handoff.

### Day 5: Archive and Checksum Proof

- Reviewed `make_source_archive()`, `archive_contains()`,
  `verify_source_archive()`, checksum selection, formula URL/checksum
  injection, and generated-output guard behavior.
- Confirmed archive creation is still gated behind metadata detection, so the
  proof does not create an incomplete provider archive while root license
  metadata is absent.
- Confirmed detected root license metadata entries would be appended to the
  source archive and verified once approved metadata exists.
- Confirmed no generated Homebrew proof outputs exist under
  `packaging/homebrew`.
- Recorded required archive entries, checksum diagnostics, generated-output
  boundaries, and the Day 6 render-validation handoff.

### Day 6: Formula Render Validation

- Verified the Homebrew formula template parses with Ruby.
- Confirmed all required formula placeholders are present for homepage, local
  archive URL, SHA-256, version, and license metadata.
- Reviewed render-time checks for empty replacements and unresolved
  `__SPARSE_*__` placeholders.
- Ran the proof with unset, `NOASSERTION`, and `PLACEHOLDER` license values;
  all runs exited `2` at the missing root metadata gate before render.
- Recorded that Day 6 did not create placeholder-free rendered formula
  evidence because approved root metadata was still absent at that point.

### Day 7: Install Surface Proof

- Reviewed the formula `install` block and `check_installed_static_surface()`
  proof-script checks.
- Recorded installed artifact expectations for the Homebrew prefix, static
  archive, public headers, CMake package files, and `sparse.pc`.
- Confirmed the proof requires static imported target metadata and rejects
  shared-library, provider, selector, runtime-loader, and dynamic ABI wording.
- Confirmed cleanup expectations for failed post-install proof steps.
- Did not execute formula install because approved root license metadata is
  still absent and render cannot proceed claim-safely.

### Day 8: Downstream Formula Test Proof

- Reviewed the Homebrew formula `test do` block and proof-script contract
  checks.
- Confirmed the downstream test uses exact-version `find_package(Sparse ...)`
  and links `Sparse::sparse_lu_ortho`.
- Confirmed the test builds under Homebrew `testpath`, passes only
  `CMAKE_PREFIX_PATH=#{prefix}`, and does not reference source-tree include or
  library paths.
- Confirmed installed static archive, CMake config, pkg-config metadata, and
  shared-artifact rejection checks remain present.
- Did not execute `brew test` because the formula cannot be rendered or
  installed until approved license metadata exists.

### Day 9: End-to-End Local Formula Proof

- Ran `bash scripts/homebrew_local_formula_proof.sh`.
- Confirmed the command exits `2` at the missing standalone root metadata
  gate.
- Confirmed local `brew`, `cmake`, `ruby`, `tar`, `shasum`, and `cc` are
  available, and `VERSION` is `2.2.0`.
- Confirmed archive creation, checksum calculation, formula rendering,
  install, installed-surface validation, `brew test`, and uninstall are not
  reached.
- Confirmed no generated Homebrew proof outputs exist under
  `packaging/homebrew`.
- Recorded item 198.3 as blocked by missing approved root metadata and exact
  Homebrew formula license identifier.

### Day 10: Package Guard Promotion

- Reviewed the package-manager and static-package guards against the current
  Sprint 198 evidence state.
- Updated `scripts/package_manager_deferral_check.sh` to assert Sprint 198
  metadata-blocker artifacts and claim-safe Day 9 proof interpretation.
- Left `scripts/static_package_deferral_check.sh` unchanged because static
  archive scope and shared-library/dynamic ABI non-claims remain covered.
- Recorded that Day 10 guard promotion did not promote Homebrew support
  because the proof exited `2` at that point.

### Day 11: Public Package Documentation

- Updated README package-manager wording with the Sprint 198 proof blocker.
- Updated INSTALL package-manager deferral wording from Sprint 188 to Sprint
  198 and recorded the missing exact `SPARSE_HOMEBREW_LICENSE` value.
- Updated `packaging/homebrew/README.md` so proof-only status references
  Sprint 198 and requires both approved root metadata and the exact formula
  license identifier before any install-method wording.
- Preserved explicit non-claims for Homebrew/core, bottles, Linuxbrew, public
  taps, other package managers, binary packages, shared-library support, and
  dynamic ABI support.

### Day 12: Maintainer and Planning Alignment

- Updated maintainer guidance so Package/Homebrew proof ownership includes
  Sprint 198 blocker artifacts.
- Updated package-manager guard runbook wording to mention Sprint 198
  metadata-blocker record checks.
- Added a maintainer note that package-manager wording must remain
  blocker/provenance wording until approved metadata exists and proof exits
  `0`.
- Updated the Sprint 198 item status ledger for items 198.1 through 198.6.
- Recorded residual package-manager gaps and retrospective inputs for final
  closeout.

### Day 13: Integrated Validation

- Ran the Homebrew local formula proof and confirmed expected exit `2` at the
  missing standalone root metadata gate.
- Ran package-manager and static-package guards; both passed.
- Ran Make install validation; `tests/test_install.sh` passed 23 checks with
  0 failures.
- Ran CMake install validation; `tests/test_cmake_install.sh` passed 27 checks
  with 0 failures and 0 skips.
- Ran `make docs-check`; API docs generation and coverage checks passed.
- Ran `git diff --check`; no whitespace errors were reported.
- Confirmed no generated Homebrew proof outputs exist under
  `packaging/homebrew`.

### Day 14: Closeout Review

- Added the Day 14 closeout review artifact.
- Closed Sprint 198 as a blocker-safe sprint: metadata/proof work remains
  blocked, guard/docs alignment is partial for the blocker state, and current
  validation surfaces are complete.
- Recorded final retained non-claims for local Homebrew formula support,
  Homebrew/core readiness, bottles, Linuxbrew support, public taps, other
  package managers, binary packages, shared-library package support, dynamic
  ABI compatibility, and broad package-manager support.
- Confirmed no standalone root `LICENSE`, `COPYING`, or `NOTICE` file exists
  and no generated Homebrew proof outputs exist under `packaging/homebrew`.
- Recorded that `scripts/__pycache__/` is untracked generated cache and must
  not be staged.

### Post-Closeout MIT Metadata and Temporary Tap Proof Work

- Added root MIT license metadata in `LICENSE`.
- Selected `SPARSE_HOMEBREW_LICENSE=MIT` for the local Homebrew proof
  invocation.
- Updated the Homebrew proof script to render into a temporary local tap,
  install/test by formula reference, untap during cleanup, and preserve
  nonzero proof exit status.
- Confirmed the proof now reaches metadata detection, source archive creation,
  SHA-256 calculation, temporary tap creation, and formula rendering.
- Confirmed `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash
  scripts/homebrew_local_formula_proof.sh` completes source install,
  installed-surface validation, downstream `brew test`, uninstall, cleanup, and
  proof exit `0` on macOS Intel x86_64 Tier 3 Homebrew.
- Retained non-claims for Homebrew/core readiness, bottles, Linuxbrew support,
  public tap maintenance, binary package distribution, other package managers,
  shared-library package support, dynamic ABI compatibility, runtime-loader
  behavior, and broad package-manager support.
