# Sprint 207 Working Notes

## Sprint Goal

Promote one exact package distribution path, or close the package distribution
residual with stronger deferral guards and no user-facing package claim.

## Current Branch

- Branch: `sprint-207`
- Plan: `docs/planning/EPIC_19/SPRINT_207/PLAN.md`
- Epic source: `docs/planning/EPIC_19/PROJECT_PLAN.md`, Sprint 207

## Item Checklist

| Item | Name | Initial Day 1 status | Primary surfaces | Expected evidence |
| --- | --- | --- | --- | --- |
| 207.1 | Provider Scope Decision | Complete: continued deferral selected | `packaging/homebrew/`, README, INSTALL, maintainer guide, package guards, Sprint 198 artifacts | Provider decision artifact; selected or deferred provider path with rejected-option rationale |
| 207.2 | Formula And Metadata Audit | Baseline audited; decision follow-up pending | `packaging/homebrew/sparse-lu-ortho.rb.in`, `packaging/homebrew/README.md`, `scripts/homebrew_local_formula_proof.sh`, root `LICENSE`, source archive/checksum behavior | Formula and metadata audit; installed static surface inventory |
| 207.3 | Proof Path Implementation | Complete for selected continued-deferral path | Homebrew proof script, formula template, package proof fixtures, environment gates, cleanup behavior | Updated proof command or strengthened deferral behavior; focused regression evidence |
| 207.4 | Package Guard Alignment | Complete for selected continued-deferral tier | `scripts/package_manager_deferral_check.sh`, `scripts/static_package_deferral_check.sh`, README, INSTALL, maintainer guide | Guard updates that enforce the selected support tier and retained package non-claims |
| 207.5 | User And Maintainer Docs | Complete for user and maintainer docs | README, INSTALL, `packaging/homebrew/README.md`, `docs/maintainer_guide.md`, Epic 19 planning docs | Claim-safe provider status and maintainer runbook wording |
| 207.6 | Validation And Closeout | Complete | Package proof command, guard scripts, install tests, docs checks, full C gate if needed | Integrated validation artifact; closeout status ledger and retrospective inputs |

## Day 1 Evidence Map

| Evidence source | Day 1 interpretation |
| --- | --- |
| `docs/planning/EPIC_18/SPRINT_198/RETROSPECTIVE.md` | Sprint 198 closed the developer-mode local Homebrew static source formula proof and retained broad package-manager non-claims. |
| `docs/planning/EPIC_18/SPRINT_198/artifacts/day14-closeout-review.md` | Current proof evidence includes `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT`, archive/checksum, temporary tap render, source install, installed static package surface validation, downstream `brew test`, uninstall, cleanup, and proof exit `0` on macOS Intel x86_64 Tier 3 Homebrew. |
| `packaging/homebrew/README.md` | Homebrew material remains proof-only and is not a Homebrew/core formula, tap, bottle, Linuxbrew claim, or general package-manager support. |
| `scripts/homebrew_local_formula_proof.sh` | Local proof script validates MIT metadata, archive contents, installed static package surface, downstream formula test behavior, cleanup, and retained non-claims. |
| `scripts/package_manager_deferral_check.sh` | Provider guard enforces Sprint 171 deferral, Sprint 198 proof record, absence of unselected provider recipes, local Homebrew proof boundary, package metadata neutrality, and public non-claims. |
| `scripts/static_package_deferral_check.sh` | Static package guard enforces static-first install, rejects shared-library packaging, and preserves dynamic ABI/package-manager non-claims. |
| README and INSTALL package sections | Public docs describe source install and static package surfaces while keeping broad package-manager distribution unclaimed. |
| `docs/maintainer_guide.md` package sections | Maintainer docs list exact package proof and guard commands and warn against inferring Homebrew/core, bottles, Linuxbrew, public tap, binary distribution, or broad package support. |

## Current Package Claim Boundary

Earned evidence currently covers only:

- root MIT license metadata;
- developer-mode local Homebrew static source formula proof;
- local source archive and checksum generation;
- temporary local tap rendering;
- source install of the local formula;
- installed static archive/header/CMake/pkg-config surface validation;
- downstream `brew test`;
- uninstall and cleanup;
- package and static-package guard coverage for the bounded proof state.

The following remain explicit non-claims:

- public Homebrew tap maintenance;
- Homebrew/core readiness or acceptance;
- bottle support;
- Linuxbrew support;
- hosted binary packages;
- vcpkg, Conan, pkgsrc, distro/system packages, or other package managers;
- broad package-manager distribution;
- shared-library package support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager platform parity;
- release readiness or binary release support;
- state-of-the-art package ecosystem claims.

## Initial Validation Matrix

| Validation | Purpose | Day 1 status |
| --- | --- | --- |
| `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh` | Re-run selected local Homebrew proof when implementation or documentation claims require proof evidence. | Candidate Day 4 or later command; not run on Day 1 intake. |
| `bash scripts/package_manager_deferral_check.sh` | Enforce package-manager non-claims and Sprint 198 local proof boundary. | Candidate guard for Days 9, 12, and 14. |
| `bash scripts/static_package_deferral_check.sh` | Enforce static-first package contract and shared-library/dynamic ABI deferrals. | Candidate guard for Days 9, 12, and 14. |
| `bash tests/test_install.sh` | Validate Make install/uninstall and downstream static package consumer behavior. | Required if install metadata changes. |
| `bash tests/test_cmake_install.sh` | Validate CMake install/downstream static package consumer behavior. | Required if CMake package metadata changes. |
| `make docs-check` | Validate docs and generated API coverage when docs surfaces change. | Candidate Day 12 validation. |
| `make format && make lint && make test` | Full C quality gate. | Required only if `.c` or `.h` files change. |
| `git diff --check` | Whitespace validation for all changed files. | Required before closeout. |

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Public package wording overstates the current local proof. | Users may infer Homebrew install support or package-manager distribution that is not earned. | Keep Day 1 non-goals visible and require guard coverage before any promotion. |
| Homebrew/core readiness is selected without release, URL, checksum, audit, and maintenance evidence. | Provider claim becomes unreproducible or rejected by downstream maintainers. | Day 2 option scoring must include exact provider evidence and maintenance costs. |
| Local proof passes on one environment but is described as broad platform support. | Package claim becomes platform-parity overclaim. | Record Homebrew tier, host, compiler/toolchain, and environment blockers separately. |
| Formula proof changes leave generated taps, archives, logs, or bottle outputs under source-controlled paths. | Review noise and accidental artifact publication. | Preserve generated-output scans and cleanup checks. |
| Static package proof is confused with shared-library or ABI support. | Users may depend on unsupported dynamic ABI behavior. | Keep `static_package_deferral_check.sh` in the validation matrix. |
| Future docs edits remove package non-claims while proof remains local-only. | Claim drift across public and maintainer surfaces. | Align package guards with selected support tier on Day 9. |
| Stale temporary Homebrew taps can survive outside the repository from earlier proof attempts. | Local environment state can confuse proof reruns and cleanup claims. | Day 4 removed stale `sparse-lu-ortho/local-proof-12578` and verified no temporary proof taps remained. |

## Open Questions

1. Should Sprint 207 target a public Homebrew tap/source formula, Homebrew/core
   readiness, or continued deferral?
2. If promotion is selected, what release URL, versioning, checksum, audit,
   hosted validation, and maintenance owner evidence is required?
3. Is the existing macOS Intel x86_64 Tier 3 Homebrew proof sufficient for any
   user-facing support tier, or only for local proof evidence?
4. Should the selected path retain developer-mode requirements or reject them
   as insufficient for provider support?
5. Which package docs should become the authoritative user-facing package
   status surface after Sprint 207?
6. Which guard should own public tap/Homebrew/core/bottle/Linuxbrew non-claims
   if continued deferral is selected?

## Day 2 Provider Option Matrix

| Option | User value | Evidence cost | Maintenance burden | Platform risk | Claim risk | Review surface | Day 2 disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Public Homebrew tap/source formula | High | Medium | Medium | Medium | Medium | Medium | Leading promotion candidate if Day 3-5 audit proves a public tap formula can be owned without implying Homebrew/core, bottles, or Linuxbrew. |
| Homebrew/core readiness | High | Very high | High | High | High | High | Not a Day 2 leading candidate; requires release URL, stable checksum provenance, formula audit readiness, upstream submission discipline, and support policy beyond current evidence. |
| Continued deferral with stronger guards | Medium | Low to medium | Low | Low | Low | Low | Safe fallback if no provider path can be proven without overclaiming current package support. |

### Provider Evidence Requirements

| Option | Required evidence before claim change |
| --- | --- |
| Public Homebrew tap/source formula | Source-controlled public tap formula or documented tap ownership model; stable source archive URL or release artifact; SHA-256 provenance; MIT license metadata; formula render/audit/install/test/uninstall proof on a supported macOS environment; cleanup behavior; public docs that name only the tap/source formula tier; guards retaining Homebrew/core, bottle, Linuxbrew, binary package, shared-library, ABI, and other package-manager non-claims. |
| Homebrew/core readiness | All public tap/source formula evidence plus Homebrew/core formula-style audit readiness, stable release archive discipline, supported upstream versioning, upstream metadata completeness, test block compatibility, no local developer-mode dependency, submission/maintenance owner, and explicit non-claims for acceptance, bottles, Linuxbrew, and binary distribution until separately proven. |
| Continued deferral | Stronger package-manager guard coverage; explicit public wording that no user-facing package-manager install path is claimed; residual queue entry naming evidence needed to revisit public tap, Homebrew/core, bottles, Linuxbrew, and other providers; validation that provider recipes and broad package claims remain absent. |

### Preliminary Decision Notes

- The public Homebrew tap/source formula is the most plausible promotion path
  because it builds on Sprint 198 local proof while avoiding Homebrew/core and
  bottle claims.
- Homebrew/core readiness is intentionally high risk for Sprint 207 because
  the repository does not currently have release/archive provider evidence or
  upstream submission readiness artifacts.
- Continued deferral remains a valid closeout path if the Day 3 formula and
  metadata audit finds that the existing proof cannot be converted into a
  user-facing package tier safely.
- No option may promote shared-library package support, dynamic ABI support,
  Linuxbrew, bottles, or broad package-manager distribution without separate
  evidence.

## Day 3 Formula And Metadata Baseline

| Surface | Current state | Provider implication |
| --- | --- | --- |
| `packaging/homebrew/sparse-lu-ortho.rb.in` | Template class is `SparseLuOrthoLocal`; formula name is local-proof oriented; URL, sha256, version, license, homepage, and local CMake bindir are rendered placeholders. | Suitable for local proof. Not yet suitable as a public tap formula or Homebrew/core readiness artifact without renaming, stable URL/checksum policy, and local-path removal. |
| `scripts/homebrew_local_formula_proof.sh` | Builds a local source archive from selected repository entries, computes SHA-256, renders a temporary local tap formula, installs from source, validates static installed surface, runs `brew test`, uninstalls, and cleans up. | Strong local proof owner. Public provider support still needs stable archive provenance and non-developer-mode/public-provider behavior. |
| Root `LICENSE` | MIT license metadata exists. | License blocker is closed for the MIT path. |
| `VERSION` | Current version is `2.2.0`. | Formula version can be rendered, but public provider support needs release/archive policy tied to versioning. |
| `CMakeLists.txt` install surface | Installs static archive, public headers under `include/sparse`, generated `sparse_version.h`, CMake package files, and `sparse.pc`. | Static package surface is well-defined; shared-library and ABI support remain outside package scope. |
| `sparse.pc.in` | Provider-neutral static archive metadata with `Libs: -L${libdir} -lsparse_lu_ortho -lm ...`. | Good installed static surface; must stay free of provider, shared-library, and ABI claims. |
| `cmake/SparseConfig.cmake.in` | Minimal package config that imports `SparseTargets.cmake`. | Good static CMake consumer surface; no provider claim by itself. |

### Day 3 Promotion Gaps

- No source-controlled public tap formula exists.
- No stable public source archive URL or release artifact is recorded for a
  provider formula.
- The proof formula depends on `file://` archive URLs generated under a temp
  root.
- The proof injects `__SPARSE_LOCAL_CMAKE_BINDIR__`, which is appropriate for
  local proof but not ideal provider formula behavior.
- The formula class/name intentionally includes `Local`.
- Homebrew/core audit readiness has not been established.
- Bottle, Linuxbrew, hosted binary, shared-library package, and dynamic ABI
  evidence remain absent.

## Day 4 Environment And Proof Baseline

| Field | Captured value |
| --- | --- |
| Host | macOS `15.7.9-x86_64` on Intel `kabylake` CPU |
| Homebrew | `6.0.22-231-gb5dc864`, prefix `/usr/local` |
| CLT / Xcode | CLT `26.3.0.0.1.1769666919`; Xcode `N/A`; selected developer dir `/Library/Developer/CommandLineTools` |
| SDK | `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` |
| C compiler | Apple clang `17.0.0 (clang-1700.6.4.2)` |
| CMake | `4.4.3` |
| Ruby | `2.6.10p210` |
| Proof command | `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh` |
| Proof result | Passed with exit `0`. |

### Day 4 Proof Stages Reached

- local-proof scope banner;
- temporary local tap creation: `sparse-lu-ortho/local-proof-53394`;
- temp root creation:
  `/var/folders/fw/8vs_6b6d407_pchh720y5h280000gn/T//sparse-homebrew-proof.YVpPB2`;
- local source archive creation;
- SHA-256 calculation:
  `c8003d621ef13919c4e47c615b5a4ea877bc9ad05992a5e9b733b108416dc5ed`;
- temporary formula rendering under the temporary tap;
- source install;
- static installed package surface validation;
- downstream `brew test`;
- uninstall;
- success line:
  `passed: local Homebrew formula proof completed for static source formula scope only`.

### Day 4 Cleanup Findings

- The Day 4 proof temp root was removed after success.
- No `sparse-lu-ortho-local` formula remained installed after success.
- No `sparse-lu-ortho/local-proof-*` tap remained after cleanup.
- No generated Homebrew archive, formula, log, bottle, or `Formula/` output
  appeared under `packaging/homebrew`.
- A stale `sparse-lu-ortho/local-proof-12578` tap from a prior environment
  state was found and removed with
  `brew untap --force sparse-lu-ortho/local-proof-12578`.

### Day 4 Classification

The existing local proof is reproducible on this host with developer mode and
MIT metadata. It remains a local static source formula proof only. The host is
Intel macOS with Homebrew on a constrained support tier, so this evidence does
not independently promote Homebrew/core readiness, bottles, Linuxbrew, public
tap maintenance, binary packages, or broad package-manager support.

## Day 5 Provider Decision

Sprint 207 selects **explicit continued deferral with stronger guards** as the
implementation path.

### Decision Rationale

| Decision input | Day 5 interpretation |
| --- | --- |
| Public tap/source formula evidence | Insufficient. The current formula is local-proof named, rendered from placeholders, installed through a temporary local tap, and backed by a temporary `file://` archive. |
| Homebrew/core readiness evidence | Insufficient. There is no Homebrew/core-ready formula, stable release/archive checksum policy, audit-readiness record, or upstream submission owner. |
| Existing local proof | Strong enough to retain and guard as developer-mode local static source formula evidence. |
| User-facing package claim risk | Too high for promotion without public-provider archive and formula ownership evidence. |
| Safest complete Sprint 207 outcome | Close the provider-support residual as intentionally deferred, with stronger guard coverage and exact future evidence requirements. |

### Selected Implementation Boundary

Allowed Sprint 207 implementation work after Day 5:

- strengthen `scripts/package_manager_deferral_check.sh`;
- strengthen `scripts/static_package_deferral_check.sh` only if static package
  non-claims need alignment;
- update package docs in README, INSTALL, `packaging/homebrew/README.md`, and
  `docs/maintainer_guide.md`;
- add planning residuals and closeout evidence for public tap, Homebrew/core,
  bottles, Linuxbrew, binary packages, and other package managers;
- add focused regression checks for forbidden package-provider overclaims;
- rerun the local proof and guard checks as validation evidence.

Out of scope after the Day 5 decision:

- public tap formula publication;
- Homebrew/core readiness wording;
- bottle support;
- Linuxbrew support;
- vcpkg, Conan, pkgsrc, distro/system package support;
- binary package distribution;
- shared-library package support;
- dynamic ABI compatibility;
- release readiness claims.

### Rejected Option Residuals

| Rejected path | Reason rejected for Sprint 207 | Future evidence required |
| --- | --- | --- |
| Public Homebrew tap/source formula | No public tap formula, stable public archive URL, public checksum provenance, or non-local formula proof exists. | Add provider formula ownership, stable source archive policy, formula render/audit/install/test/uninstall proof, cleanup proof, and claim-safe docs. |
| Homebrew/core readiness | Current evidence is local/developer-mode and lacks Homebrew/core audit readiness, upstream release discipline, submission owner, and non-local formula behavior. | Add all public tap evidence plus Homebrew/core-style audit readiness, release/archive process, and readiness wording that does not imply acceptance, bottles, or Linuxbrew. |

### Day 5 Validation Direction

The rest of Sprint 207 should validate that the repository clearly says:

- local Homebrew static source formula proof exists;
- no user-facing package-manager install path is claimed;
- public tap, Homebrew/core, bottles, Linuxbrew, other package managers,
  binary packages, shared-library packaging, and dynamic ABI support remain
  unclaimed until future evidence closes the residual.

## Day 6 Proof And Deferral Design

### Selected Design

Sprint 207 will preserve the existing local proof command and strengthen the
deferral boundary around it. The proof command remains:

```sh
HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh
```

The proof should continue to produce pass, unavailable, or fail semantics:

| State | Meaning | Support interpretation |
| --- | --- | --- |
| Pass, exit `0` | Local static source formula proof completed. | May cite only developer-mode local static source formula proof. |
| Unavailable, exit `2` | Required local tool, host condition, or approved metadata is missing. | Package support remains unclaimed. |
| Fail, other nonzero | Project proof or cleanup defect. | Fix defect before changing support wording. |

### Guard Design

| Guard area | Required Day 7-9 design target |
| --- | --- |
| Provider artifact absence | Continue rejecting unselected provider files such as vcpkg, Conan, pkgsrc, distro package specs, and committed Homebrew formula/tap outputs outside the selected proof template. |
| Positive provider wording | Reject public docs or maintainer docs that state package-manager distribution, Homebrew/core, bottles, Linuxbrew, public taps, binary packages, vcpkg, Conan, pkgsrc, or distro/system packages are supported. |
| Local-proof wording | Require docs to say the Homebrew proof is developer-mode local static source formula proof only, not a user-facing install path. |
| Future evidence requirements | Require Sprint 207 planning artifacts or maintainer docs to list exact evidence needed to reopen public tap/source formula and Homebrew/core readiness. |
| Generated proof output hygiene | Continue rejecting generated archives, rendered formula files, logs, bottles, and `Formula/` trees under `packaging/homebrew`. |
| Static package boundary | Keep shared-library packaging, dynamic ABI, and static/shared selector checks under `static_package_deferral_check.sh`. |

### Documentation Design

Docs should converge on this wording model:

- user-facing install path: source install via Make or CMake;
- package-manager distribution: not claimed;
- Homebrew proof: developer-mode local static source formula proof only;
- public tap/Homebrew/core/bottles/Linuxbrew: residual future work, not
  current support;
- evidence to reopen: stable public archive, checksum provenance, provider
  formula ownership, install/test/uninstall proof, cleanup proof, and
  claim-safe docs/guards.

### Validation Mapping

| Change type | Required validation |
| --- | --- |
| Package guard changes | `bash scripts/package_manager_deferral_check.sh` |
| Static package wording or install metadata changes | `bash scripts/static_package_deferral_check.sh` |
| Proof script or formula template changes | Local Homebrew proof command plus package guard. |
| Public or maintainer docs changes | Package guard, static package guard if static boundary wording changes, and `make docs-check`. |
| Install or CMake package metadata changes | `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh`. |
| `.c` or `.h` changes | `make format && make lint && make test`. |

## Daily Log

### Day 1: Package Intake

- Created this working-notes scaffold.
- Mapped Sprint 207 items 207.1 through 207.6 to expected surfaces and
  evidence.
- Reviewed Sprint 198 closeout and retrospective evidence.
- Identified current package proof and non-claim boundaries.
- Recorded initial validation matrix, risk register, and open questions.
- Added Day 1 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day1-package-intake.md`.

### Day 2: Provider Scope Options

- Compared public Homebrew tap/source formula, Homebrew/core readiness, and
  continued-deferral paths.
- Recorded evidence requirements for each option before any provider decision
  is made.
- Scored options by user value, evidence cost, maintenance burden, platform
  risk, claim risk, and review surface.
- Identified public Homebrew tap/source formula as the leading promotion
  candidate only if later audit/proof work can avoid overclaiming; retained
  continued deferral as the safe fallback.
- Added Day 2 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day2-provider-scope-options.md`.

### Day 3: Formula And Metadata Baseline

- Audited the Homebrew template, proof script, root MIT metadata, version
  metadata, source archive/checksum behavior, and installed static package
  surface.
- Confirmed the current Homebrew implementation is local-proof oriented, not a
  public provider formula.
- Recorded installed static surface ownership: static archive, public headers,
  generated version header, CMake package files, and `sparse.pc`.
- Identified promotion gaps for public tap/source formula and Homebrew/core
  readiness.
- Added Day 3 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day3-formula-metadata-baseline.md`.

### Day 4: Environment And Proof Baseline

- Captured host, Homebrew, CLT/Xcode, SDK, compiler, CMake, and Ruby
  environment details.
- Ran
  `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh`.
- Recorded proof success through archive/checksum, temporary tap render,
  source install, installed static surface validation, downstream `brew test`,
  uninstall, cleanup, and exit `0`.
- Verified no generated proof outputs under `packaging/homebrew`, no installed
  `sparse-lu-ortho-local` formula, and no remaining `sparse-lu-ortho/local-proof-*`
  taps after cleanup.
- Removed a stale temporary proof tap from earlier local environment state:
  `sparse-lu-ortho/local-proof-12578`.
- Added Day 4 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day4-environment-proof-baseline.md`.

### Day 5: Provider Decision

- Applied the Day 2 option criteria to Day 3 formula/metadata evidence and
  Day 4 proof/environment evidence.
- Selected explicit continued deferral with stronger guards as the Sprint 207
  implementation path.
- Rejected public Homebrew tap/source formula promotion for Sprint 207 because
  stable public archive provenance, public formula ownership/naming, and
  non-local formula proof are absent.
- Rejected Homebrew/core readiness because release/archive discipline,
  Homebrew/core audit readiness, non-local formula behavior, and submission
  ownership are absent.
- Defined the implementation boundary for Days 6-14: stronger guards, docs,
  residual evidence, focused regression checks, and validation of retained
  package non-claims.
- Added Day 5 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day5-provider-decision.md`.

### Day 6: Proof Design

- Designed Sprint 207 implementation around stronger continued deferral,
  not provider promotion.
- Preserved the current local proof command and pass/unavailable/fail support
  semantics.
- Defined guard targets for forbidden provider artifacts, positive provider
  wording, local-proof-only wording, future evidence requirements, generated
  proof output hygiene, and static package boundaries.
- Defined documentation wording and validation mapping for Days 7-14.
- Added Day 6 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day6-proof-deferral-design.md`.

### Day 7: Proof Implementation Batch One

- Updated `scripts/package_manager_deferral_check.sh` with a Sprint 207
  provider-decision check.
- Added guard checks requiring the Day 5 continued-deferral decision and Day 6
  proof/deferral design to stay present.
- Added public/maintainer documentation scans rejecting unsupported positive
  claims for package-manager distribution, Homebrew/core readiness, public
  taps, bottles, Linuxbrew, vcpkg, Conan, pkgsrc, distro/system packages, and
  binary packages.
- Preserved the existing local proof behavior; no formula template or proof
  script behavior was changed.
- Ran `bash -n scripts/package_manager_deferral_check.sh`.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed, including
  the selected Homebrew local proof boundary and new Sprint 207 checks.
- Verified no `sparse-lu-ortho-local` formula, no
  `sparse-lu-ortho/local-proof-*` tap, and no generated Homebrew output under
  `packaging/homebrew` remained after validation.
- Added Day 7 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day7-deferral-guard-implementation.md`.

### Day 8: Proof Implementation Batch Two

- Added `tests/test_package_manager_deferral_guard.py` with focused
  regression fixtures for unsupported package-provider wording.
- Moved Sprint 207 provider-decision and forbidden-provider-claim checks
  before the embedded local Homebrew proof in
  `scripts/package_manager_deferral_check.sh`, so unsupported wording fails
  quickly before expensive provider proof work starts.
- Covered positive Homebrew/core readiness, public tap support, and binary
  package support wording failures.
- Ran `python3 tests/test_package_manager_deferral_guard.py`.
- Ran `bash -n scripts/package_manager_deferral_check.sh`.
- Ran `python3 -m py_compile tests/test_package_manager_deferral_guard.py`.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed, including
  the selected Homebrew local proof boundary.
- Verified no `sparse-lu-ortho-local` formula, no
  `sparse-lu-ortho/local-proof-*` tap, and no generated Homebrew output under
  `packaging/homebrew` remained after validation.
- Added Day 8 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day8-proof-regression-cleanup.md`.

### Day 9: Package Guard Alignment

- Reviewed package-manager and static-package guard ownership against the Day
  5 continued-deferral decision.
- Extended `scripts/package_manager_deferral_check.sh` forbidden provider
  claim patterns to reject positive release-readiness, release artifact,
  release package, and package release wording.
- Extended `tests/test_package_manager_deferral_guard.py` with bottle support,
  Linuxbrew support, and release package overclaim regressions.
- Ran `python3 tests/test_package_manager_deferral_guard.py`.
- Ran `python3 -m py_compile tests/test_package_manager_deferral_guard.py`.
- Ran `bash -n scripts/package_manager_deferral_check.sh`.
- Ran `bash scripts/static_package_deferral_check.sh`.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed, including
  embedded local Homebrew proof.
- Verified no `sparse-lu-ortho-local` formula, no
  `sparse-lu-ortho/local-proof-*` tap, and no generated Homebrew output under
  `packaging/homebrew` remained after validation.
- Added Day 9 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day9-package-guard-alignment.md`.

### Day 10: User Package Docs

- Updated README package guidance to name the Sprint 207 continued-deferral
  decision and to describe the Homebrew proof as developer-mode local static
  source formula proof only.
- Updated INSTALL support/readiness wording and the support matrix package row
  so users can distinguish supported source installs from unclaimed provider,
  binary, release, bottle, Linuxbrew, tap, and package-manager readiness paths.
- Updated `packaging/homebrew/README.md` to clarify that the template remains
  proof-only and is not a user-facing Homebrew install method or provider
  distribution artifact.
- Ran `bash scripts/static_package_deferral_check.sh`.
- Ran `make docs-check`.
- Ran `python3 tests/test_package_manager_deferral_guard.py`.
- Ran `python3 -m py_compile tests/test_package_manager_deferral_guard.py`.
- Ran `bash -n scripts/package_manager_deferral_check.sh`.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed, including
  embedded local Homebrew proof boundary checks.
- Verified no `sparse-lu-ortho-local` formula, no
  `sparse-lu-ortho/local-proof-*` tap, and no generated Homebrew output under
  `packaging/homebrew` remained after validation.
- Added Day 10 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day10-user-package-docs.md`.

### Day 11: Maintainer Package Docs

- Updated `docs/maintainer_guide.md` package ownership to name Sprint 207 as
  the current continued-deferral decision layer on top of Sprint 198 local
  Homebrew proof evidence.
- Updated the maintainer package runbook so package-provider claim changes
  require stable archive/checksum provenance, provider formula or recipe
  ownership, non-local proof, cleanup proof, docs, residuals, and guard
  coverage before wording can change.
- Added exact maintainer evidence requirements for Homebrew/core readiness,
  bottles, Linuxbrew, binary packages, release packages, other package
  providers, shared-library packages, and dynamic ABI behavior.
- Updated `packaging/homebrew/README.md` with a maintainer claim-change
  checklist for public Homebrew tap/source formula and Homebrew/core readiness
  reopening.
- Ran `bash scripts/static_package_deferral_check.sh`.
- Ran `make docs-check`.
- Ran `python3 tests/test_package_manager_deferral_guard.py`.
- Ran `python3 -m py_compile tests/test_package_manager_deferral_guard.py`.
- Ran `bash -n scripts/package_manager_deferral_check.sh`.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed, including
  embedded local Homebrew proof boundary checks.
- Verified no `sparse-lu-ortho-local` formula, no
  `sparse-lu-ortho/local-proof-*` tap, and no generated Homebrew output under
  `packaging/homebrew` remained after validation.
- Added Day 11 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day11-maintainer-package-docs.md`.

### Day 12: Integrated Validation

- Ran the selected standalone local Homebrew proof command:
  `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh`.
- Ran package and documentation guards:
  `bash scripts/static_package_deferral_check.sh`,
  `make docs-check`, `make support-docs-guard`, and
  `bash scripts/package_manager_deferral_check.sh`.
- Ran Sprint 207 package-manager regression checks:
  `python3 tests/test_package_manager_deferral_guard.py`,
  `python3 -m py_compile tests/test_package_manager_deferral_guard.py`,
  and `bash -n scripts/package_manager_deferral_check.sh`.
- Ran installed static package checks:
  `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh`.
- Confirmed no `.c` or `.h` files are changed, so the full C gate is not
  required by the Day 12 plan.
- Ran `git diff --check`.
- Verified no `sparse-lu-ortho-local` formula, no
  `sparse-lu-ortho/local-proof-*` tap, and no generated Homebrew output under
  `packaging/homebrew` remained after validation.
- Added Day 12 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day12-integrated-validation.md`.

### Day 13: Review Hardening

- Audited the changed package guard, Python regression fixtures, README,
  INSTALL, maintainer guide, Homebrew proof notes, and Sprint 207 artifacts
  for package-provider overclaim risk.
- Confirmed changed docs consistently present source install as the
  user-facing path and Homebrew as developer-mode local static source formula
  proof only.
- Hardened `scripts/package_manager_deferral_check.sh` against plural
  `public taps` support wording and direct `package-manager support` or
  `broad package-manager support` overclaims.
- Extended `tests/test_package_manager_deferral_guard.py` with regressions for
  plural public-tap support wording and package-manager support wording.
- Ran `python3 tests/test_package_manager_deferral_guard.py`.
- Ran `python3 -m py_compile tests/test_package_manager_deferral_guard.py`.
- Ran `bash -n scripts/package_manager_deferral_check.sh`.
- Ran `bash scripts/static_package_deferral_check.sh`.
- Ran `make docs-check` and `make support-docs-guard`.
- Ran `bash scripts/package_manager_deferral_check.sh`; it passed, including
  embedded local Homebrew proof boundary checks.
- Verified no `sparse-lu-ortho-local` formula, no
  `sparse-lu-ortho/local-proof-*` tap, and no generated Homebrew output under
  `packaging/homebrew` remained after validation.
- Confirmed no `.c` or `.h` files are changed, so the full C gate is still not
  required.
- Added Day 13 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day13-review-hardening.md`.

### Day 14: Package Closeout

- Updated `docs/planning/EPIC_19/PROJECT_PLAN.md` with an Epic 19 current
  status snapshot that closes Sprint 207 and leaves Sprints 208-216 pending.
- Recorded final item dispositions for 207.1 through 207.6: all complete for
  the selected continued package-provider deferral path.
- Recorded residual package work for public Homebrew tap/source formula,
  Homebrew/core readiness, bottles, Linuxbrew, vcpkg, Conan, pkgsrc,
  distro/system packages, binary packages, release packages, shared-library
  packages, dynamic ABI behavior, and broad package-manager distribution.
- Prepared retrospective inputs covering accomplishments, changed surfaces,
  validation, residuals, risks, and claim-boundary summary.
- Re-ran focused closeout validation after Day 14 documentation edits:
  `python3 tests/test_package_manager_deferral_guard.py`,
  `python3 -m py_compile tests/test_package_manager_deferral_guard.py`,
  `bash -n scripts/package_manager_deferral_check.sh`,
  `bash scripts/static_package_deferral_check.sh`,
  `make docs-check`, `make support-docs-guard`, and
  `bash scripts/package_manager_deferral_check.sh`.
- Verified no `sparse-lu-ortho-local` formula, no
  `sparse-lu-ortho/local-proof-*` tap, and no generated Homebrew output under
  `packaging/homebrew` remained after validation.
- Confirmed no `.c` or `.h` files are changed, so the full C gate is not
  required for Sprint 207.
- Added Day 14 artifact:
  `docs/planning/EPIC_19/SPRINT_207/artifacts/day14-closeout-review.md`.
