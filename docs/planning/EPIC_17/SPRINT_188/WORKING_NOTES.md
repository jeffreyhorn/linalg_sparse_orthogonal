# Sprint 188 Working Notes

## Sprint Goal

Close the selected Homebrew local proof blocker by resolving standalone
license metadata and proving the full local formula workflow.

## Branch Baseline

- Branch: `sprint-188`
- Starting point: current `master` after PR #208 merge.
- Epic 17 owner gap: `E17-GAP-001 / R186-PKG-LICENSE`.
- Sprint 188 plan status: day-by-day plan exists at
  `docs/planning/EPIC_17/SPRINT_188/PLAN.md`.

## Planning Source

| Field | Value |
| --- | --- |
| Project plan | `docs/planning/EPIC_17/PROJECT_PLAN.md` |
| Section | `Sprint 188: Homebrew Proof Completion` |
| Sprint duration | 14 days, approximately 168 hours |
| Acceptance gate source | `docs/planning/EPIC_17/SPRINT_187/artifacts/day7-package-acceptance-gates.md` |
| Handoff source | `docs/planning/EPIC_17/SPRINT_187/artifacts/day13-implementation-handoffs.md` |
| Quality map | `docs/planning/EPIC_17/SPRINT_187/artifacts/day12-quality-surface-map.md` |

## Sprint 188 Item Boundaries

| Item | Name | Sprint 188 interpretation |
| --- | --- | --- |
| 188.1 | License Strategy Decision | Decide the approved root license metadata file and exact Homebrew license identifier, or record a guarded alternate strategy. |
| 188.2 | Metadata Implementation | Add or update root license metadata and align formula/proof metadata with the selected strategy. |
| 188.3 | Proof Script Hardening | Ensure render, archive, checksum, install, test, uninstall, and cleanup have clear pass/block/fail behavior. |
| 188.4 | Package Guards | Keep package-manager and static-package guards tied to the actual proof state and retained non-claims. |
| 188.5 | Documentation Calibration | Update README, INSTALL, Homebrew README, and maintainer guidance with exact earned support or blocker wording. |
| 188.6 | Validation | Run Homebrew proof, package guards, selected install/report checks, docs checks, and the full C gate if `.c` or `.h` files change. |

## Day 1 Package Proof Baseline

Day 1 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day1-package-proof-intake.md`
as the package proof baseline and owner-surface inventory.

### Day 1 Owner Surface Inventory

| Surface | Current owner files | Day 1 finding |
| --- | --- | --- |
| Root license metadata | `LICENSE`, `COPYING`, `NOTICE` | No root standalone license metadata file currently exists; this is the active Homebrew proof blocker. |
| Version metadata | `VERSION` | Present and non-empty. |
| Formula template | `packaging/homebrew/sparse-lu-ortho.rb.in` | Present as a temporary local formula template with required Homebrew placeholders and static-only install/test behavior. |
| Proof script | `scripts/homebrew_local_formula_proof.sh` | Present and executable; exits `2` on missing standalone license metadata while keeping local Homebrew support unclaimed. |
| Package-manager guard | `scripts/package_manager_deferral_check.sh` | Passes with the current missing-license blocker state. |
| Static package guard | `scripts/static_package_deferral_check.sh` | Passes and preserves static-first package support plus shared-library/dynamic ABI deferrals. |
| User docs | `README.md`, `INSTALL.md` | Current wording records local Homebrew proof artifacts but keeps the proof blocked and unclaimed as a user install route. |
| Maintainer/package docs | `packaging/homebrew/README.md`, `docs/maintainer_guide.md` | Current wording documents the local proof-only scope and missing license metadata blocker. |

### Day 1 Local Tool Snapshot

| Tool | Local status | Sprint 188 impact |
| --- | --- | --- |
| `brew` | Available at `/usr/local/bin/brew` | Local Homebrew proof can run once license metadata is resolved. |
| `cmake` | Available at `/usr/local/bin/cmake` | Formula build/install and downstream CMake test prerequisites are available. |
| `ruby` | Available at `/usr/bin/ruby` | Formula template rendering and syntax validation prerequisites are available. |
| `tar` | Available at `/usr/bin/tar` | Source archive creation prerequisite is available. |
| SHA-256 tool | `shasum` available at `/usr/bin/shasum` | Archive checksum prerequisite is available. |
| C compiler | `cc` available at `/usr/bin/cc` | Formula build prerequisite is available. |

### Day 1 Baseline Command Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Exit `2` | Proof is unavailable because no standalone `LICENSE`, `COPYING`, or `NOTICE` exists for provider metadata; Homebrew support remains unclaimed. |
| `scripts/package_manager_deferral_check.sh` | Exit `0` | Package-manager non-claims and selected local Homebrew proof boundary are intact. |
| `scripts/static_package_deferral_check.sh` | Exit `0` | Static-first package contract, shared-library deferral, and dynamic ABI deferral are intact. |

### Day 1 Risks

| Risk | Mitigation |
| --- | --- |
| Adding license metadata could be treated as a broad package-manager support decision. | Day 2 must separate legal/project license approval from Homebrew proof support wording. |
| A proof-script pass could be over-promoted to Homebrew/core, bottle, tap, Linuxbrew, or binary package support. | Package docs and guards must keep the claim limited to local source formula proof for the maintained static archive package surface. |
| Proof-script hardening could accidentally leave temporary formula, archive, tap, log, cache, build, install, or bottle outputs in the repository. | Day 5 through Day 8 must verify cleanup behavior and source-control status after proof runs. |
| Static/shared package wording could drift while updating docs. | `scripts/static_package_deferral_check.sh` remains required whenever package wording changes. |
| C/header edits are not expected but would expand validation requirements. | Any `.c` or `.h` change requires `make format && make lint && make test`. |

### Day 1 Open Questions

| Question | Day 1 disposition |
| --- | --- |
| What exact root license metadata file should be added? | Open for Day 2 license strategy decision. |
| What exact Homebrew license identifier should populate `SPARSE_HOMEBREW_LICENSE`? | Open for Day 2 license strategy decision. |
| Should Sprint 188 promote package wording if the proof exits `0` after metadata work? | Yes, but only to local static source formula proof and only after package guards pass. |
| Should Sprint 188 claim Homebrew/core, bottles, Linuxbrew, public taps, vcpkg, Conan, pkgsrc, distro packaging, shared libraries, or dynamic ABI support? | No. These remain explicit non-goals. |
| Is the current blocker local environment availability? | No. Required local tools are present; the active blocker is missing standalone license metadata. |

### Day 1 Validation

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 2 License Strategy Decision

Day 2 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day2-license-strategy-decision.md`
as the license strategy decision record. The repository scan found no root
`LICENSE`, `COPYING`, or `NOTICE` file and no authoritative source-controlled
project license text or SPDX identifier.

### Day 2 Decision Summary

| Decision point | Result |
| --- | --- |
| Add guessed root license metadata | Rejected. No authoritative license source or project-owner approval was found. |
| Select exact `SPARSE_HOMEBREW_LICENSE` value | Deferred. The value must match approved root license metadata. |
| Remove or bypass formula license metadata | Rejected. That would weaken provider metadata and avoid the selected blocker. |
| Use placeholder or `NOASSERTION` metadata | Rejected. A proof pass with inaccurate license metadata is not acceptable support evidence. |
| Selected strategy | Keep the Homebrew proof blocked and unclaimed until approved standalone root metadata exists; implement a claim-safe blocker record and guard/doc alignment. |

### Day 2 Day-3 Handoff

Day 3 should implement the safe metadata path selected on Day 2:

1. Add or update a source-controlled decision/blocker artifact for the missing
   approved root license metadata.
2. Keep `scripts/homebrew_local_formula_proof.sh` fail-safe on missing root
   metadata and missing `SPARSE_HOMEBREW_LICENSE`.
3. Add guard coverage only if a drift point is found where placeholder or
   inaccurate license metadata could pass.
4. Keep public docs calibrated to blocker status unless approved license
   metadata is provided before Day 3 implementation.
5. Re-run package guards after any documentation or guard updates.

### Day 2 Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 3 Metadata Implementation

Day 3 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day3-metadata-implementation.md`
as the safe metadata implementation record. Because no authoritative project
license source exists, Day 3 did not add a guessed root `LICENSE`, `COPYING`,
or `NOTICE` file and did not choose a guessed Homebrew license identifier.

### Day 3 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Added a placeholder-license rejection guard for `SPARSE_HOMEBREW_LICENSE` values such as `NOASSERTION`, `UNKNOWN`, `TBD`, `TODO`, `FIXME`, `PLACEHOLDER`, unresolved template text, and values containing `placeholder`. |
| `scripts/package_manager_deferral_check.sh` | Added static guard coverage requiring the proof script to retain placeholder-license rejection wording. |
| `packaging/homebrew/README.md` | Documented that placeholder license values are blocker evidence, not proof metadata. |
| `docs/maintainer_guide.md` | Documented that future standalone metadata must use an accurate matching Homebrew license identifier. |

### Day 3 Metadata State

| Field | State |
| --- | --- |
| Root standalone license metadata | Still absent; no approved project license text exists in source control. |
| Homebrew license identifier | Still unselected; no exact identifier can be chosen without approved root metadata. |
| Proof interpretation | Missing root metadata remains an expected unavailable blocker with exit `2`. |
| Claim interpretation | Homebrew support remains unclaimed; local proof material is not a user-facing install route. |

### Day 3 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable because no standalone root license metadata exists; Homebrew support remains unclaimed. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and selected Homebrew boundary remain guarded, including placeholder-license rejection. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |

Day 3 changed scripts and documentation but no `.c` or `.h` files. The full C
quality gate is not required.

## Day 4 Formula Template Audit

Day 4 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day4-formula-template-audit.md`
as the Homebrew formula template audit record. The audit found that
`packaging/homebrew/sparse-lu-ortho.rb.in` remains aligned with the selected
local static source formula proof boundary.

### Day 4 Audit Summary

| Area | Result |
| --- | --- |
| Template status | Pass. The source-controlled file remains a `.rb.in` template, not a committed installable `.rb` formula. |
| Required placeholders | Pass. Homepage, local archive URL, SHA-256, version, and Homebrew license placeholders are present. |
| License metadata | Pass. License metadata is injected only through `__SPARSE_HOMEBREW_LICENSE__`; no default or guessed identifier appears in the template. |
| Static install surface | Pass. The formula builds with CMake and checks for `lib/libsparse_lu_ortho.a`. |
| Shared artifact rejection | Pass. Install and test phases reject `.dylib`, `.so`, `.so.*`, and `.dll` outputs. |
| Downstream consumer test | Pass. `test do` uses exact-version `find_package(Sparse ...)` and links `Sparse::sparse_lu_ortho`. |
| Generated output hygiene | Pass. No rendered formula, archive, log, bottle, or local tap output was found under `packaging/homebrew`. |

### Day 4 Template Correction Decision

No formula template corrections are required before Day 5 proof-script
render/archive hardening. The active blocker remains approved standalone root
license metadata, not formula template shape.

### Day 4 Validation

Day 4 changed planning documentation only. Targeted checks confirmed required
placeholders, Ruby syntax, static/test markers, and generated-output hygiene.
No `.c` or `.h` files were modified, so the full C quality gate is not
required.

## Day 5 Render and Archive Proof Hardening

Day 5 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day5-render-archive-hardening.md`
as the proof-script render/archive hardening record. Day 5 changed the proof
script so license metadata is validated before temporary archive creation and
future successful archives are checked for required entries.

### Day 5 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Moves license metadata detection before temporary archive creation. |
| `scripts/homebrew_local_formula_proof.sh` | Tracks detected root license metadata entries for archive inclusion. |
| `scripts/homebrew_local_formula_proof.sh` | Adds `verify_source_archive` so successful archives must include source, package metadata, examples, and standalone license metadata. |
| `scripts/package_manager_deferral_check.sh` | Requires the proof script to retain required source archive entry verification. |
| `packaging/homebrew/README.md` | Documents that metadata is validated before archive creation and included in future successful archives. |

### Day 5 Proof-State Change

Before Day 5, the missing-license path created a temporary source archive and
then exited `2` before formula rendering. After Day 5, the proof exits `2`
before temporary archive creation when no root `LICENSE`, `COPYING`, or
`NOTICE` file exists. This keeps the blocker clearer and avoids producing
archive/checksum output that could be mistaken for partial proof evidence.

### Day 5 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable because no standalone root license metadata exists; it now exits before archive creation. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims, selected Homebrew boundary, placeholder-license rejection, and archive verification guard remain intact. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |

Day 5 changed shell scripts and documentation but no `.c` or `.h` files. The
full C quality gate is not required.

## Day 6 Install Surface Proof Hardening

Day 6 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day6-install-surface-hardening.md`
as the installed static package proof hardening record. Day 6 tightened the
future successful install path so installed package metadata cannot introduce
provider, shared-library, static/shared selector, SONAME/DLL/dylib, or dynamic
ABI wording.

### Day 6 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Verifies the Homebrew prefix and installed `lib` directory before checking installed static package files. |
| `scripts/homebrew_local_formula_proof.sh` | Keeps installed static archive, headers, CMake package files, target metadata, and `sparse.pc` checks. |
| `scripts/homebrew_local_formula_proof.sh` | Adds installed metadata scanning for unsupported provider, shared-library, selector, SONAME/DLL/dylib, `Libs.private`, and dynamic ABI wording. |
| `scripts/package_manager_deferral_check.sh` | Requires the proof script to retain installed metadata rejection. |
| `packaging/homebrew/README.md` | Documents installed static package metadata expectations and rejected installed metadata surfaces. |

### Day 6 Installed Surface Checklist

Future successful proof installs must include the temporary Homebrew prefix,
installed `lib` directory, `libsparse_lu_ortho.a`, sparse headers,
`SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`,
`SparseTargets-noconfig.cmake`, `sparse.pc`, static imported target metadata,
and target metadata pointing at the installed static archive.

### Day 6 Cleanup and Retry Notes

The proof still enables uninstall-on-exit before Homebrew install so a failure
after installation attempts to uninstall the temporary formula. Temporary proof
roots are removed by default unless `--keep-temp` is selected. With Day 5's
earlier license validation, the current missing-license blocker exits before
archive or install work begins.

### Day 6 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable because no standalone root license metadata exists; it stops before archive/install work. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims, Homebrew boundary, placeholder-license rejection, archive verification, and installed metadata rejection remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |

Day 6 changed shell scripts and documentation but no `.c` or `.h` files. The
full C quality gate is not required.

## Day 7 Downstream Consumer Test Proof

Day 7 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day7-downstream-consumer-proof.md`
as the Homebrew `test do` downstream consumer proof record. Day 7 added
preflight proof-script checks so the formula template cannot silently lose its
exact-version CMake consumer, imported target link, installed header exercise,
package metadata checks, successful output assertion, or shared-artifact
rejection.

### Day 7 Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Adds `verify_formula_test_contract` to check the template's downstream test contract before license-gated proof work. |
| `scripts/package_manager_deferral_check.sh` | Requires the proof script to retain downstream CMake consumer, imported-target link, and shared-artifact rejection guards. |
| `packaging/homebrew/README.md` | Documents the `test do` downstream consumer contract and its non-claim boundary. |

### Day 7 Test Contract

The formula `test do` block must keep exact-version `find_package(Sparse ...)`,
link `Sparse::sparse_lu_ortho`, compile against installed public headers,
assert successful executable output, verify static/CMake/pkg-config installed
metadata, and reject shared-library artifacts. A `brew test` failure blocks
package support promotion.

### Day 7 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` | Passed | Formula template parses as Ruby. |
| Template marker audit | Passed | Exact downstream CMake, imported target, installed header, output assertion, package metadata, and shared rejection markers are present. |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable before `brew test` because no standalone root license metadata exists. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and downstream consumer proof guards remain intact. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |

Day 7 changed shell scripts and documentation but no `.c` or `.h` files. The
full C quality gate is not required.

## Day 8 End-to-End Homebrew Proof Run

Day 8 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day8-end-to-end-proof-run.md`
as the full local Homebrew proof run record for the current license state. No
approved root `LICENSE`, `COPYING`, or `NOTICE` file exists, and no exact
`SPARSE_HOMEBREW_LICENSE` value has been selected.

### Day 8 Proof Classification

| Classification | Result |
| --- | --- |
| Pass | Not reached. A pass still requires approved root metadata, accurate license identifier, render, archive, checksum, install, installed-surface validation, `brew test`, uninstall, and cleanup success. |
| Block | Active. The proof exits `2` because no standalone root license metadata exists for provider metadata. |
| Fail | Not observed. The proof stops before render/archive/install/test work, so no later proof failure occurred. |

### Day 8 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable and Homebrew support remains unclaimed. |
| second `scripts/homebrew_local_formula_proof.sh` run | Expected exit `2` | Blocker result is reproducible. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and the selected local Homebrew boundary remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |
| Generated-output scan under `packaging/homebrew` | Clean | No rendered formula, archive, log, bottle, or local tap output was retained. |
| Homebrew formula install check | Clean | `sparse-lu-ortho-local` is not installed. |
| Recent proof temporary root scan | Clean | No recent `sparse-homebrew-proof.*` temp root remains under the local temp directory. |

### Day 8 Day-9 Handoff

Day 9 should align package guards with the blocker state rather than support
promotion. Guard wording should continue to allow local proof material while
requiring public docs to keep Homebrew support unclaimed until the proof exits
`0` with approved root license metadata and accurate Homebrew license
metadata.

### Day 8 Validation Scope

Day 8 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 9 Package Guard Alignment

Day 9 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day9-package-guard-alignment.md`
as the package guard calibration record. The package-manager guard now matches
the current blocker state more tightly: when root license metadata is absent,
the proof must exit `2`, name the missing standalone license blocker, keep
support unclaimed, and stop before archive/render/install/test work.

### Day 9 Guard Changes

| Surface | Change |
| --- | --- |
| `scripts/package_manager_deferral_check.sh` | Detects whether root standalone license metadata exists before running the Homebrew proof. |
| `scripts/package_manager_deferral_check.sh` | Requires missing-license exit `2` output to name the standalone metadata blocker. |
| `scripts/package_manager_deferral_check.sh` | Fails if missing-license proof output shows temp archive, render, install, or `brew test` work started. |
| `scripts/package_manager_deferral_check.sh` | Requires INSTALL and Homebrew README wording to keep blocked Homebrew proof out of user-facing install support. |

### Day 9 Package Report Decision

No selected package report metadata changed on Day 9, so package report
normalization and freshness checks are not required for this day.

### Day 9 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable and stops before archive/render/install/test work because root license metadata is absent. |
| `scripts/package_manager_deferral_check.sh` | Passed | Guard matches the current proof state and keeps public Homebrew support unclaimed. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain enforced. |

Day 9 changed shell scripts and planning documentation but no `.c` or `.h`
files. The full C quality gate is not required.

## Day 10 User-Facing Package Docs

Day 10 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day10-user-facing-package-docs.md`
as the README/INSTALL calibration record. The public docs now describe the
current Sprint 188 proof state directly: local Homebrew proof material exists,
but Homebrew install support remains unclaimed because approved standalone
root license metadata is absent.

### Day 10 Documentation Changes

| Surface | Change |
| --- | --- |
| `README.md` | Updated the package-manager paragraph to state that the proof exits before archive, render, install, or `brew test` work while approved standalone root license metadata is absent. |
| `INSTALL.md` | Updated the support split to distinguish local Homebrew proof material from user-facing Homebrew install support. |
| `INSTALL.md` | Added that no exact `SPARSE_HOMEBREW_LICENSE` value is selected until approved root license metadata exists and placeholder values are blocker evidence. |
| `INSTALL.md` | Updated package evidence wording to match the Day 9 blocker-state guard. |

### Day 10 Claim Boundary

Source install and installed static package support remain first. Local
Homebrew proof material may be described only as proof material. Homebrew/core,
bottles, Linuxbrew, public taps, vcpkg, Conan, pkgsrc, distro/system packages,
provider registry readiness, binary packages, shared-library package support,
dynamic ABI compatibility, static/shared selectors, and broad package-manager
distribution remain unsupported.

### Day 10 Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable and stops before archive/render/install/test work because root license metadata is absent. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and current Homebrew blocker wording remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain enforced. |

Day 10 changed user-facing documentation and planning documentation but no
`.c` or `.h` files. The full C quality gate is not required.

## Day 11 Homebrew and Maintainer Documentation

Day 11 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day11-maintainer-package-docs.md`
as the Homebrew and maintainer documentation record. The docs now state the
local proof command, exit-code meanings, generated-output policy, and required
validation by changed package surface.

### Day 11 Documentation Changes

| Surface | Change |
| --- | --- |
| `packaging/homebrew/README.md` | Added the proof command, exit-code interpretation, support wording rules, required package guards, and validation guidance. |
| `docs/maintainer_guide.md` | Updated proof wording so the missing-license blocker exits before archive, render, install, or `brew test` work. |
| `docs/maintainer_guide.md` | Added explicit validation ownership for package-manager guard, static-package guard, install checks, CMake install checks, and package report normalization checks. |

### Day 11 Validation Policy

Package-manager support wording changes require
`scripts/package_manager_deferral_check.sh`. Static/shared package wording or
install metadata changes require `scripts/static_package_deferral_check.sh`.
Install behavior, installed consumer docs, CMake package files, `sparse.pc`,
or downstream examples require `bash tests/test_install.sh` and
`bash tests/test_cmake_install.sh`. Package report metadata changes require
package report normalization and freshness checks.

### Day 11 Validation

Day 11 changed documentation and planning documentation but no `.c` or `.h`
files. The full C quality gate is not required.

## Day 12 Integrated Package Validation

Day 12 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day12-integrated-package-validation.md`
as the integrated validation record. The validation gate covers the selected
Homebrew proof blocker state, package-manager guard, static-package guard,
install proofs, CMake install proofs, documentation hygiene, generated-output
cleanup, package report applicability, and C quality-gate applicability.

### Day 12 Changed Surface Review

| Surface | Validation decision |
| --- | --- |
| Homebrew proof script | Validate with shell syntax, expected unavailable proof, progress scan, and package-manager guard. |
| Package-manager wording and provider proof boundary | Validate with `scripts/package_manager_deferral_check.sh`. |
| Static/shared package wording and installed metadata expectations | Validate with `scripts/static_package_deferral_check.sh`. |
| Install-facing package guidance | Validate with `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh`. |
| Package report metadata | Not changed; package report normalization and freshness checks are not required. |
| C source/header files | Not changed; `make format && make lint && make test` is not required. |

### Day 12 Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable because no standalone root `LICENSE`, `COPYING`, or `NOTICE` metadata exists. |
| Missing-license proof progress scan | Passed | The unavailable proof stops before temp archive, formula render, install, or `brew test` work. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims and selected local Homebrew proof boundary remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package support and shared-library/dynamic ABI deferrals remain guarded. |
| `bash tests/test_install.sh` | Passed | Make install/uninstall, installed headers, `pkg-config`, and downstream consumers remain valid. |
| `bash tests/test_cmake_install.sh` | Passed | CMake install/export, exact-version consumers, static imported target metadata, and package metadata remain valid. |
| `git diff --check` | Passed | Current diff has no whitespace errors. |
| Trailing-whitespace scan | Passed | Changed docs and scripts have no trailing whitespace. |
| Homebrew generated-output scan | Passed | No generated formula, archive, log, bottle, or local tap output exists under `packaging/homebrew`. |
| Sprint 188 markdown link check | Passed | Sprint-local markdown links resolve. |

### Day 12 Claim Boundary

The integrated gate confirms that Sprint 188 remains in a guarded blocker
state rather than a support-promotion state. The local Homebrew proof material
exists and is guarded, but Homebrew install support remains unclaimed until
approved standalone root license metadata exists and the proof exits `0`.

## Day 13 Package Claim Audit

Day 13 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day13-claim-audit.md` as the final
package claim audit and residual decision record.

### Day 13 Final State Decision

Sprint 188 retains a guarded residual blocker. The repository has local
Homebrew proof material and passing guards around the selected proof boundary,
but no standalone root `LICENSE`, `COPYING`, or `NOTICE` file exists and no
exact `SPARSE_HOMEBREW_LICENSE` value is selected. The Homebrew proof exits
`2` before archive, render, install, or `brew test` work, so the sprint cannot
promote Homebrew install support.

### Day 13 Touched Documentation Audit

| Surface | Audit result |
| --- | --- |
| `README.md` | Keeps package-manager support unprovided and frames Homebrew as a provider-proof blocker, not availability. |
| `INSTALL.md` | Keeps source install/static package support separate from package-manager support. |
| `packaging/homebrew/README.md` | Documents local proof-only scope, exit-code interpretation, generated-output policy, and retained non-claims. |
| `docs/maintainer_guide.md` | Documents proof ownership, validation commands, and support-promotion limits. |
| Sprint 188 artifacts | Keep the missing-license state as residual blocker evidence. |

### Day 13 Retained Non-Claims

Sprint 188 does not claim Homebrew install availability, Homebrew/core
readiness, bottles, Linuxbrew, public taps, vcpkg, Conan, pkgsrc,
distro/system packages, provider registry readiness, binary packages,
shared-library package support, dynamic ABI compatibility, static/shared
package selectors, or broad package-manager distribution.

### Day 13 Revisit Criteria

The Homebrew local proof can be reconsidered only after approved standalone
root license metadata exists, `SPARSE_HOMEBREW_LICENSE` is set to the accurate
matching Homebrew license identifier, the local proof exits `0`, and both
package guards pass. Any broader provider claim requires a separate product
decision and separate evidence.

## Day 14 Sprint Closeout

Day 14 created
`docs/planning/EPIC_17/SPRINT_188/artifacts/day14-closeout-summary.md` as the
final closeout summary, retrospective input record, and PR-ready evidence map.

### Day 14 Final Sprint State

Sprint 188 closes as a guarded residual, not as promoted Homebrew install
support. The sprint hardened the selected local Homebrew proof path, aligned
guards, calibrated user and maintainer docs, and validated the existing static
package install surface. The license metadata blocker remains because no
approved standalone root `LICENSE`, `COPYING`, or `NOTICE` file exists and no
exact `SPARSE_HOMEBREW_LICENSE` value is selected.

### Day 14 Item Disposition

| Item | Name | Disposition |
| --- | --- | --- |
| 188.1 | License Strategy Decision | Complete: retain the missing standalone license metadata blocker rather than invent license terms. |
| 188.2 | Metadata Implementation | Residual: no authoritative root license metadata or exact Homebrew identifier exists to implement safely. |
| 188.3 | Proof Script Hardening | Complete: proof script validates metadata, archive contents, installed metadata, downstream test contract, uninstall, and cleanup boundaries. |
| 188.4 | Package Guards | Complete: guards enforce the selected Homebrew proof boundary, missing-license blocker behavior, public non-claims, generated-output absence, and static-first package scope. |
| 188.5 | Documentation Calibration | Complete: README, INSTALL, Homebrew README, maintainer guide, and sprint artifacts carry consistent blocker/support wording. |
| 188.6 | Validation | Complete: integrated proof, guard, install, CMake install, docs hygiene, generated-output, and C/header applicability checks passed. |

### Day 14 Closeout Checks

| Check | Result |
| --- | --- |
| Sprint artifacts | Days 1 through 14 are represented by source-controlled planning artifacts. |
| Root license metadata scan | No root `LICENSE`, `COPYING`, or `NOTICE` file exists; residual remains explicit. |
| Generated Homebrew output scan | Clean; no generated formula, archive, log, bottle, or tap output is present under `packaging/homebrew`. |
| Stale TODO scan | No actionable stale TODO/FIXME/TBD markers were found; placeholder examples are intentional blocker examples. |
| Unsupported claim audit | Touched docs consistently keep Homebrew/package-manager support unclaimed. |
| Validation baseline | Day 12 integrated validation and Day 13 guard/proof rechecks passed. |

### Day 14 PR-Ready Summary

The branch improves local Homebrew proof rigor and package-claim calibration.
It keeps the support boundary intentionally narrow: local proof material is
present, but Homebrew install support remains unclaimed until approved
standalone root license metadata exists, an accurate `SPARSE_HOMEBREW_LICENSE`
is selected, the proof exits `0`, and package guards pass.

### Day 14 Retrospective Inputs

- Proof hardening and guard alignment landed without inventing license terms.
- The remaining residual is narrow, explicit, and externally owned by the
  need for approved root license metadata.
- Existing Make and CMake static package install proofs continue to pass.
- Future provider claims should be split from this local proof unless a later
  product decision adds separate evidence for Homebrew/core, bottles,
  Linuxbrew, public taps, other package managers, shared libraries, or dynamic
  ABI support.
