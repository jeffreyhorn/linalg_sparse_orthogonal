# Sprint 144 Working Notes

## Sprint

Sprint 144: Platform Promotion Lane Closure

## Goal

Fully promote one high-value platform support lane, or explicitly reject
promotion with source-level blockers and proof requirements.

## Starting Evidence

Sprint 144 starts from Sprint 143's static-first package/ABI decision:

- the maintained package surface is static-first only;
- `BUILD_SHARED_LIBS=ON` is rejected;
- Make install/`pkg-config` and CMake install/export proof both exercise the
  installed static archive surface;
- Linux CI carries the reviewed static-first package-contract lane;
- macOS package install/export jobs remain supplemental confidence paths;
- Windows carries the reviewed CMake subset plus supplemental CMake
  install/downstream confidence;
- shared-library build/install/export, dynamic ABI compatibility,
  runtime-loader compatibility, package-manager support, Windows Makefile
  parity, Windows `pkg-config` parity, Windows install-validation parity, and
  macOS full install/export parity remain non-claims.

## Candidate Platform Lanes

| Lane | Current status | Promotion question |
| --- | --- | --- |
| macOS reviewed install/export parity | Apple Clang reviewed path is enforced; Make install/`pkg-config` and CMake install/export jobs are supplemental. | Can the supplemental macOS package jobs become reviewed install/export parity for the static-first contract? |
| Windows reviewed install/downstream parity | MSVC CMake subset is reviewed; CMake install/downstream proof is supplemental. | Can supplemental Windows CMake install/downstream confidence become a reviewed Windows install-validation lane for the exact static-first scope? |
| Windows staged test portability | `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain outside the Windows reviewed subset. | Can one staged Windows blocker be closed with source-level portability work and an intentional CTest count update? |
| Linux source-of-truth strengthening | Linux is already the strongest reviewed source of truth, including package contract proof. | Is there a remaining source-owned Linux proof gap worth closing rather than promoting macOS or Windows? |

## Item-To-Day Owner Map

| Sprint 144 item | Day-level owner |
| --- | --- |
| Item 1: Platform Lane Selection | Days 1-2 |
| Item 2: Source/Script Portability Fixes | Days 3-5 |
| Item 3: CI Promotion Implementation | Days 6-7 |
| Item 4: Package/Report Integration | Day 8 |
| Item 5: Documentation Alignment | Day 9 |
| Item 6: Validation | Days 10-12 |
| Item 7: Closeout | Days 13-14 |

## Initial Promotion Criteria

A platform lane can be promoted only if:

- the selected lane has exact source-owned scope;
- hosted CI proof exists or the sprint explicitly records why local proof is
  the only feasible evidence;
- workflow comments, failure messages, README, INSTALL, maintainer guidance,
  report rows, and sprint artifacts all agree;
- expected CTest counts or staged exclusions have clear ownership;
- promoted proof is static-first package/platform proof and does not imply
  shared-library, dynamic ABI, loader, package-manager, or platform-wide parity
  unless those claims are directly tested.

## Initial Stop Conditions

Stop and ask before promotion if:

- the selected lane requires unplanned public C/header API changes;
- a `.c` or `.h` change causes `make format`, `make lint`, or `make test` to
  fail;
- the selected lane needs a hosted runner behavior that cannot be expressed in
  CI or locally reviewed with confidence;
- the work would promote more than one platform lane;
- the evidence would widen shared-library, dynamic ABI, package-manager,
  Windows Makefile, Windows `pkg-config`, or runtime-loader claims without
  direct proof.

## Daily Log

### Day 1: Platform Promotion Intake

Created the Sprint 144 working-notes baseline and Day 1 intake artifact.
Captured the Sprint 143 static-first package handoff, inventoried candidate
platform lanes, mapped Sprint 144 Items 1-7 to days, and recorded initial
promotion criteria, evidence requirements, non-claims, and stop conditions.

Validation:

- `git diff --check`
- `rg -n "^## Day|^\\*\\*Time estimate:\\*\\*" docs/planning/EPIC_12/SPRINT_144/PLAN.md`

No `.c` or `.h` files changed.

### Day 2: Platform Lane Selection

Scored the four candidate platform lanes and selected macOS reviewed
install/export parity as the primary Sprint 144 closure lane. The selected
scope is narrow: promote the existing macOS static-first Make
install/`pkg-config` and CMake install/export jobs from supplemental confidence
to reviewed install/export parity for the static-first package contract only.

Backup lane:

- Windows reviewed CMake static install/downstream parity, limited to the
  existing CMake-first static package scope.

Deferred lanes:

- Windows staged test portability remains source-risk-heavy because
  `test_threads`, `test_sprint4_integration`, and `test_fuzz` still have
  pthread/POSIX blockers.
- Linux source-of-truth strengthening remains lower value because Linux already
  owns reviewed package-contract proof.

Validation:

- `git diff --check`
- `rg -n "selected|macOS|Windows|Linux|Completion Criteria" docs/planning/EPIC_12/SPRINT_144/artifacts/day2-platform-lane-selection.md`

No `.c` or `.h` files changed.

### Day 3: Blocker Reproduction And Evidence Baseline

Captured the selected macOS lane's before-change evidence baseline. The package
proof scripts already pass locally, and the primary blocker is support-tier
bookkeeping drift: macOS workflow comments, job names, README, INSTALL, and
maintainer guidance still describe the static-first package jobs as
supplemental rather than reviewed macOS install/export proof.

Focused local validation baseline:

- `bash tests/test_install.sh`: passed, 23 passed and 0 failed
- `bash tests/test_cmake_install.sh`: passed, 26 passed, 0 failed, 0 skipped
- `bash scripts/static_package_deferral_check.sh`: passed

Hosted macOS CI remains the final proof owner for promotion, because local
script success is not the same as a GitHub-hosted `macos-latest` run.

Validation:

- `git diff --check`
- `rg -n "support-tier|bookkeeping|source-level|CI-only|expected post" docs/planning/EPIC_12/SPRINT_144/artifacts/day3-blocker-reproduction-evidence-baseline.md`

No `.c` or `.h` files changed.

### Day 4: Portability Design

Designed the selected-lane portability and support-tier update. The design keeps
Sprint 144 source-neutral unless hosted macOS CI later exposes a real
portability defect. The Day 5 implementation should update macOS workflow
comments, job names, and step names to promote the two static-first package
jobs to reviewed macOS install/export proof while preserving static-first
package boundaries and leaving Linux/Windows support tiers unchanged.

Planned implementation surfaces:

- `.github/workflows/macos-ci.yml`
- later documentation alignment in README, INSTALL, and maintainer guide
- later package/report alignment only if Day 8 finds a source-owned row change
  is needed

Surfaces intentionally not planned for Day 5:

- `.c` and `.h` files
- Make install implementation
- CMake install/export implementation
- `sparse.pc.in`
- Windows workflow/package status
- Linux package-contract workflow status

Validation:

- `git diff --check`
- `rg -n "Implementation Order|Portability Rules|Promoted|Staged|Stop Conditions|Completion Criteria" docs/planning/EPIC_12/SPRINT_144/artifacts/day4-portability-design.md`

No `.c` or `.h` files changed.

### Day 5: Source And Script Fix Batch

Implemented the selected-lane workflow promotion in
`.github/workflows/macos-ci.yml`. The package proof commands are unchanged, but
the macOS workflow now names the Make install/`pkg-config` and CMake
install/export package jobs as reviewed macOS static-first proof. The workflow
comments preserve non-claims for shared-library packaging, dynamic ABI,
runtime-loader behavior, package-manager support, static/shared selectors, and
broader macOS platform parity.

Changed proof owners:

- `.github/workflows/macos-ci.yml` top support-tier comment
- `install-and-pkgconfig` job name and proof step name
- `cmake-install-export` job name and proof step name

Unchanged proof commands:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- `bash scripts/static_package_deferral_check.sh`

Validation:

- `ruby -e 'require "yaml"; ARGV.each { |p| YAML.load_file(p) }' .github/workflows/macos-ci.yml`
- `rg -n "supplemental .*install|install.*supplemental|confidence path|not a reviewed macOS install/export|do not claim reviewed install/export" .github/workflows/macos-ci.yml`
- `rg -n "shared-library packaging|dynamic ABI compatibility|runtime-loader compatibility|package-manager support|static/shared selectors|broader macOS platform parity|reviewed static-first" .github/workflows/macos-ci.yml`
- `git diff --check`

No `.c` or `.h` files changed.

### Day 6: CI Promotion Design

Designed CI promotion follow-through for the selected macOS static-first
install/export lane. The design keeps Day 7 focused on workflow hardening and
consistency checks rather than expanding scope: Linux remains the strongest
reviewed source of truth, Windows remains reviewed CMake subset plus
supplemental CMake install/downstream confidence, and no expected CTest counts
change for macOS promotion.

Day 7 should verify the macOS workflow definition, add only targeted
failure-context comments if needed, and run cross-workflow scans that prove
the selected lane is promoted while non-selected lanes remain unchanged.

Validation:

- `git diff --check`
- `rg -n "Expected Count|Staged|Failure Message|Artifact Proof|Day 7|Completion Criteria" docs/planning/EPIC_12/SPRINT_144/artifacts/day6-ci-promotion-design.md`

No `.c` or `.h` files changed.

### Day 7: CI Promotion Implementation

Completed the CI promotion implementation review for the selected macOS
static-first install/export lane. Day 5 already applied the necessary workflow
support-tier changes, and Day 7 found no need for additional command, artifact,
matrix, expected-count, Linux, or Windows workflow changes. The selected macOS
proof commands remain unchanged and locally feasible mirrored checks passed.

Focused local package proof:

- `bash tests/test_install.sh`: passed, 23 passed and 0 failed
- `bash tests/test_cmake_install.sh`: passed, 26 passed, 0 failed, 0 skipped
- `bash scripts/static_package_deferral_check.sh`: passed

Workflow validation:

- Ruby YAML parse passed for Linux, macOS, and Windows workflows
- macOS stale selected-lane supplemental wording scan passed
- macOS reviewed static-first/non-claim scan passed
- Linux reviewed source-of-truth preservation scan passed
- Windows reviewed/supplemental/staged preservation scan passed
- `git diff --check` passed

No `.c` or `.h` files changed.

### Day 8: Package And Report Integration

Integrated the selected macOS lane into source-controlled report semantics by
updating the CI report-family contract row in
`tests/corpus/manifests/report_families.tsv`. The package rows remain unchanged:
they continue to identify static-first proof-owner commands and templates, not
fresh hosted-run evidence. The CI row now makes the promoted macOS reviewed
static-first install/export lane discoverable alongside Linux source-of-truth
and Windows reviewed CMake subset lane definitions, while keeping hosted CI logs
external.

Validation:

- `python3 scripts/normalize_report_index.py --family package --check`
- `python3 scripts/normalize_report_index.py --family ci --check`
- `python3 scripts/normalize_report_index.py --family package --check-freshness`
- `python3 scripts/normalize_report_index.py --family ci --check-freshness`
- `git diff --check`

No `.c` or `.h` files changed.

### Day 9: Documentation Support-Tier Alignment

Aligned README, INSTALL, and maintainer guidance with the selected macOS
reviewed static-first install/export lane. The documentation now identifies
macOS reviewed static-first Make install/`pkg-config` and CMake install/export
proof while keeping Homebrew GCC supplemental, Linux as the strongest reviewed
source of truth, and Windows install/downstream confidence supplemental.

Documentation updates:

- `README.md` CI summary
- `INSTALL.md` validation story, supported-platform table, and install
  validation interpretation
- `docs/maintainer_guide.md` current package/platform support-tier guidance

Validation:

- `rg -n "macOS.*supplemental.*install|supplemental.*macOS.*install|macOS supplemental package confidence|not claim full reviewed|do not become reviewed macOS install/export|macOS full install/export parity|static-first install/export confidence" README.md INSTALL.md docs/maintainer_guide.md`
- `rg -n "macOS.*reviewed static-first|reviewed static-first.*macOS|shared-library packaging|dynamic ABI compatibility|runtime-loader|package-manager|static/shared selectors|broader macOS platform parity|Windows.*supplemental CMake install/downstream|Linux.*strongest reviewed" README.md INSTALL.md docs/maintainer_guide.md`
- `git diff --check`

No `.c` or `.h` files changed.

### Day 10: Selected-Lane Validation Pass

Ran the focused selected-lane validation pass after workflow, report, and
documentation updates. Locally feasible install/export, downstream consumer,
workflow syntax, support-tier, report normalization, and freshness checks all
passed. No implementation defects were found.

Focused validation:

- `bash tests/test_install.sh`: passed, 23 passed and 0 failed
- `bash tests/test_cmake_install.sh`: passed, 26 passed, 0 failed, 0 skipped
- `bash scripts/static_package_deferral_check.sh`: passed
- Ruby YAML parse passed for Linux, macOS, and Windows workflows
- stale macOS supplemental install/export wording scan passed
- package report normalization/freshness passed
- CI report normalization/freshness passed

Hosted-runner boundary:

- local proof validates the commands and support-tier wording available in this
  checkout
- reviewed macOS hosted proof is fully earned only when GitHub Actions runs the
  promoted macOS jobs on `macos-latest`

No `.c` or `.h` files changed.

### Day 11: Cross-Platform Non-Regression Review

Completed a cross-platform non-regression review for the macOS selected-lane
promotion. The review confirmed that Linux source-of-truth wording and package
proof ownership remain intact, Windows reviewed/supplemental/staged boundaries
remain unchanged, static-first package deferral guards still pass, and package
plus CI report rows still normalize and pass freshness checks.

Validation:

- Linux source-of-truth preservation scan passed
- Windows reviewed CMake subset, supplemental install/downstream, and staged
  exclusion scan passed
- stale macOS supplemental install/export wording scan passed
- package and CI report normalization/freshness checks passed
- Ruby YAML parse passed for Linux, macOS, and Windows workflows
- `bash scripts/static_package_deferral_check.sh` passed
- unsupported package/platform claim scan found only explicit non-claims
- `git diff --check` passed

No `.c` or `.h` files changed.

### Day 12: Quality Gate Execution

Ran the formal quality gate for Sprint 144 changed surfaces. All required
checks passed. The changed surfaces are workflows, documentation, a report
manifest TSV, and planning artifacts; no `.c` or `.h` files changed, so
`make format && make lint && make test` was not required.

Quality gate results:

- Ruby YAML parse for Linux, macOS, and Windows workflows: passed
- `python3 scripts/validate_corpus_schema.py`: passed
- package and CI report normalization/freshness checks: passed
- `bash scripts/static_package_deferral_check.sh`: passed
- `bash tests/test_install.sh`: passed, 23 passed and 0 failed
- `bash tests/test_cmake_install.sh`: passed, 26 passed, 0 failed, 0 skipped
- stale macOS supplemental install/export wording scan: passed
- unsupported package/platform claim boundary scan: matches are explicit
  non-claims
- `git diff --check`: passed

Environment constraint:

- hosted `macos-latest` execution remains external to local validation and
  must be confirmed by PR CI.

### Day 13: Promotion Evidence And Residual Non-Claims

Consolidated the selected-lane evidence and marked the Sprint 144 selected lane
as promoted: macOS now has reviewed static-first Make install/`pkg-config` and
CMake install/export proof for the maintained static archive package contract.
The promotion remains scoped to hosted `macos-latest` workflow execution of the
existing proof commands, with PR CI as the external hosted proof owner.

Residual boundaries remain explicit:

- Linux remains the strongest reviewed source-of-truth baseline.
- Windows remains reviewed CMake subset plus supplemental CMake
  install/downstream confidence.
- Windows staged `test_threads`, `test_sprint4_integration`, and `test_fuzz`
  remain source-level pthread/POSIX blocker work.
- Shared-library packaging, dynamic ABI compatibility, runtime-loader
  compatibility, package-manager support, static/shared selectors, Windows
  Makefile parity, Windows `pkg-config` parity, Windows reviewed
  install-validation parity, and broader macOS platform parity remain
  non-claims.

Validation:

- support-tier consistency scan passed
- stale macOS supplemental install/export wording scan passed
- package/CI report normalization and freshness checks passed
- unsupported package/platform claim scan found only explicit non-claims
- `git diff --check` passed

No `.c` or `.h` files changed.

### Day 14: Closeout And Handoff

Finalized Sprint 144 closeout. The selected platform lane is closed as macOS
reviewed static-first Make install/`pkg-config` and CMake install/export proof
for the maintained static archive package contract. The closeout artifact maps
Sprint 144 Items 1-7 to evidence, inventories all Day 1-14 artifacts, records
touched surfaces, preserves residual non-claims, and prepares Sprint 145
adoption handoff inputs.

Final support-tier contract:

- Linux remains the strongest reviewed source-of-truth baseline.
- macOS now carries reviewed static-first install/export proof for the
  maintained static archive package contract.
- Windows remains reviewed CMake subset plus supplemental CMake
  install/downstream confidence.
- Windows staged `test_threads`, `test_sprint4_integration`, and `test_fuzz`
  remain source-level pthread/POSIX blocker work.
- Shared-library, dynamic ABI, runtime-loader, package-manager, static/shared
  selector, Windows Makefile, Windows `pkg-config`, Windows reviewed
  install-validation, and broad platform parity claims remain unsupported.

Validation:

- Ruby workflow YAML parse passed for Linux, macOS, and Windows workflows
- `python3 scripts/validate_corpus_schema.py` passed
- package/CI report normalization and freshness checks passed
- `bash scripts/static_package_deferral_check.sh` passed
- stale macOS supplemental install/export wording scan passed
- Linux source-of-truth preservation scan passed
- Windows reviewed/supplemental/staged preservation scan passed
- unsupported package/platform claim scan found only explicit non-claims
- `git diff --check` passed

No `.c` or `.h` files changed, so `make format && make lint && make test` was
not required.
