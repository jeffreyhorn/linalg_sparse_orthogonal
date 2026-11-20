# Sprint 226 Retrospective

**Sprint:** 226 - Static Candidate And Installed First-Use Closeout

**Duration:** 14 planned days on branch `sprint-226`

**Status:** The static installed first-use route and Epic 20 claim reconciliation were completed locally. **The release-candidate outcome remains BLOCKED:** no clean candidate was selected, so no candidate-specific archive, platform proof, timing decision or owner-signed release approval exists. This is a technical closeout and handoff, not a release authorization.

## Source And Definition Of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) and [sprint plan](PLAN.md) allocate 140 estimated hours across six items and fourteen 10-hour days. These are scope estimates, not measured labor or CI duration. The [working notes](WORKING_NOTES.md), [Day 14 closeout](artifacts/day14-final-epic20-closeout-and-handoff.md) and [evidence index](artifacts/day14-closeout-evidence-index.md) identify what was actually checked. The intended product outcome was one exact static source-release candidate, with reproducible archive, installed consumer and named-platform evidence, a usable first-use route, claim calibration and owner signoff. The [Sprint 214 release checklist](../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md) requires the exact candidate and approvals before a release-ready label.

At closeout, the working tree was based on `c42d6715fad7e895bae4941a006df3563f1afc38` (tree `934a16b09ada33e12f726ec944fc675bd7d244f4`, `VERSION=2.2.0`). README and INSTALL were modified and Sprint 226 documents were untracked. Root `CHANGELOG.md` and the Sprint 225 accepted timing pointer were absent. The base commit was an inventory reference, **never a selected candidate**. Committing this retrospective and the sprint documents produces a new source SHA; the earlier archive, installed consumers and hosted receipts remain historical and cannot become exact-head release proof.

| Item | Planned | Retrospective state |
| --- | ---: | --- |
| 226.1 Candidate Freeze | 20 h | **Blocked:** inventoried source, version, license, tier and owner prerequisites and wrote a freeze rule; no approved changelog, clean reviewed candidate SHA or named release/rollback owners. |
| 226.2 Reproducible Source Archive | 28 h | **Exploratory method passed; candidate outcome blocked:** two independent same-host base archives matched byte for byte, extracted source was mapped, and a 25-file local CMake static install was hashed. The archive predates corrected docs. |
| 226.3 Candidate Platform Proof | 26 h | **Blocked:** local base-source quality/install/width checks passed and historical CI receipts were reconciled; no exact-candidate Linux, macOS or Windows proof. Historical Linux reviewed CMake parity ended at CTest 32/63. |
| 226.4 First-Use Audit | 24 h | **Locally complete, candidate replay blocked:** corrected README/INSTALL, exercised clean-prefix Make/`pkg-config` and CMake examples, 32/64 consumers, package version behavior, installed headers and links. The old archive does not contain these edits. |
| 226.5 Readiness Decision | 22 h | **Blocked technical decision, unsigned:** mapped Sprint 214 gates, proposed rollback procedure, approval register and exact-candidate index. No owner signature, timing-scope approval, destination or final go/no-go. |
| 226.6 Epic Closeout | 20 h | **Local handoff complete:** reconciled Sprint 217–226 item groups, finalized a scoped validation/claim ledger and assigned missing receipts to owner roles. No blocked outcome was counted as release-ready. |
| **Total** | **140 h** | **All planned scope accounted for; exact static-candidate and release-readiness completion criteria unmet.** |

## What Went Well

1. **The source boundary was kept exact.** The [Day 3 archive](artifacts/day3-exploratory-source-archive.md) and [Day 4 independent comparison](artifacts/day4-independent-archive-comparison.md) matched compressed bytes, ordered members, metadata and all 4,557 tracked blobs/modes. Both exploratory archives have SHA-256 `cdfceff5ef4fdf405f8bf3b15e6a7fe608d5f4566345ea3fd2f8d6ec0e27be8f`. The result demonstrated a repeatable procedure without promoting the old base to a release candidate.
2. **The installed route was exercised from clean prefixes.** [Days 5–6](artifacts/day6-local-installed-consumers.md) captured the static installed map, 32/64 CMake and pkg-config consumers and wrong-width diagnostics. [Days 9–10](artifacts/day10-first-use-audit.md) followed the README → INSTALL/examples → installed consumer → API/header path; local 32-bit CMake CTest passed 63/63, a separate 64-bit installed solve ran, and 18 checked-in headers plus generated `sparse_version.h` were accounted for. README/INSTALL now distinguish Make's pkg-config metadata from CMake's package exports.
3. **Historical hosted results were read at their actual scope.** [Day 8](artifacts/day8-platform-provenance-and-first-use-inventory.md) tied base-run Linux `44433727330`, macOS `44433727057` and Windows `44433727133` to run/job/log evidence and test inventories. macOS and Windows named subsets passed; Linux package and compile subsets passed, while its reviewed CMake parity stopped on runner shutdown. None was represented as candidate-specific proof.
4. **The readiness decision and claims are auditable.** [Day 12](artifacts/day12-static-candidate-readiness-decision.md) recorded **BLOCKED** with every signer pending. [Day 13 reconciliation](artifacts/day13-epic20-completion-reconciliation.md), [final claim ledger](artifacts/day13-validation-and-claim-ledger.md) and [Day 14 handoff](artifacts/day14-final-epic20-closeout-and-handoff.md) separate merged numerical/package/workflow work from blocked CHOLMOD, timing, platform and release outcomes. The local-only generated API HTML and static-first boundaries remain explicit.
5. **Focused documentation checks passed after the edits.** At Day 14, `make support-docs-guard`, the static package deferral script, API docs routing test and package-manager deferral policy test exited 0. The Day 14 file/anchor audit checked 272 local links across 22 Sprint 226 Markdown files with no errors; `git diff --check` passed. Day 7 exploratory base-source `make format-check`, `make lint`, `make test` (62 binaries) and CMake parity (63/63 CTests) passed, but were not substituted for candidate checks.

## What Did Not Go Well

1. **The candidate could not be frozen.** No version-specific root changelog, named release/rollback owner, approved scope or clean committed snapshot was available during the sprint. The Day 3 archive lacks later README/INSTALL fixes and Sprint 226 records; updating a local document could not repair its provenance. There was no candidate SHA or refreeze event to validate.
2. **Platform acceptance did not reach an exact candidate.** The historical Linux reviewed CMake parity run ended at CTest 32/63 because the runner shut down; there was no complete result. macOS and Windows historical subsets tested the base commit, not the intended package. The Day 7 optional Homebrew provider aggregate was interrupted while building a dependency from source, so it was not a pass. No Linux, macOS or Windows exact-candidate job or installed artifact exists.
3. **The timing gate remained unmeasured.** Sprint 225's baseline candidates were rejected for environment/dispersion mismatch. The base timing run `44433727385` returned `S225_BASELINE_MISSING`, with no raw candidate samples or threshold verdict; its blocked ZIP SHA-256 is `d89a5a06180b34e431a150daa71abf8412450b073eb65d48594657505b9353df`. No product/release owner approved a narrower static scope that excludes timing claims.
4. **The external comparison remained a local observation.** Sprint 222's two CHOLMOD/project pairs passed numerical checks on one 4×4 fixture, but the v1 observed records were rejected at `/environment`, canonical cell IDs are null, and exact baseline build/BLAS provenance plus independent attribution/rights approval are missing. An ignored local raw bundle is not an independently reviewed public artifact.
5. **A technical stop did not become a signed decision.** Release, product, API/ABI, build, platform CI, docs, license/legal and rollback roles remained unassigned or unsigned. Asset destination, retention, tag/signing and communications policy were unresolved. No release action or public claim could follow from the local checklist alone.

## Final Metrics And Claim

| Measure | Audited Sprint 226 result |
| --- | --- |
| Planned effort | 140 estimated hours, 14 planned days; actual effort not tracked |
| Inventory base | `c42d6715fad7e895bae4941a006df3563f1afc38`; tree `934a16b09ada33e12f726ec944fc675bd7d244f4`; no selected candidate |
| Exploratory source archive | Two same-host byte-identical archives; SHA-256 `cdfceff5ef4fdf405f8bf3b15e6a7fe608d5f4566345ea3fd2f8d6ec0e27be8f`; 4,557 tracked files |
| Local package and first use | 25-file CMake static install map; Make/`pkg-config` and CMake consumers, 32/64 width, opposite-width negatives, 19 installed headers; 63/63 local exploratory CTests |
| Historical hosted base | Linux `44433727330`: package/compile pass, reviewed CMake parity interrupted at 32/63; macOS `44433727057` and Windows `44433727133`: named subsets pass; all are base-only |
| Canonical timing | Accepted baseline absent; base run `44433727385` blocked at preflight; no candidate measurements or threshold verdict |
| Candidate and release | Zero exact-candidate archives/platform passes, zero signed approvals, no tag or published release |
| Day 14 focused checks | Support docs, static deferral, API routing and provider-policy tests passed; 272 local file/anchor links resolved; `git diff --check` passed |
| C/header change trigger at original closeout | The original Sprint 226 edits were README/INSTALL and sprint documentation only; the conditional C source gate was not triggered then |
| Later PR #250 C fixture revision | Commit `dc43f38fa07710dc5b4a8693c41478beca99f804` changed `tests/test_fuzz.c`; local `make format && make lint && make test` passed before the commit. The [exact-head Linux run 44794895083](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44794895083) passed direct runtime, UBSan, ASan and `bench-fast`; all 26 PR checks passed on that head. This is later review evidence, not a frozen release-candidate receipt. |

The supported Sprint 226 statement is: **“The maintained static source first-use route was exercised locally from clean prefixes, and Epic 20's bounded implementation and historical selected-evidence outcomes were reconciled. No exact static release candidate was approved; release readiness, candidate platform proof, accepted CHOLMOD external evidence and measured timing remain blocked.”** This does not authorize a tag, asset, package-manager/provider distribution, shared or cross-version ABI, portable speed, external-library parity or broad state-of-the-art claim. The merged source supports the maintained static package contract; this retrospective does not grant it a new release status.

## Daily Evidence

- [Day 1 candidate preconditions](artifacts/day1-candidate-preconditions.md)
- [Day 2 static-candidate definition](artifacts/day2-static-candidate-definition.md)
- [Day 3 exploratory archive and extraction](artifacts/day3-exploratory-source-archive.md)
- [Day 4 independent archive comparison](artifacts/day4-independent-archive-comparison.md)
- [Day 5 archive/install map and platform matrix](artifacts/day5-archive-install-map-and-platform-matrix.md) and [25-file manifest](artifacts/day5-exploratory-installed-manifest.tsv)
- [Day 6 local installed consumers](artifacts/day6-local-installed-consumers.md)
- [Day 7 quality and hosted intake](artifacts/day7-quality-and-hosted-intake.md)
- [Day 8 platform provenance and first-use inventory](artifacts/day8-platform-provenance-and-first-use-inventory.md)
- [Day 9 clean-prefix first use](artifacts/day9-clean-prefix-first-use.md)
- [Day 10 first-use audit](artifacts/day10-first-use-audit.md) and [draft decision packet](artifacts/day10-draft-readiness-decision-packet.md)
- [Day 11 checklist and rollback](artifacts/day11-checklist-review-and-rollback.md) and [approval register](artifacts/day11-approval-and-outstanding-gates.md)
- [Day 12 blocked readiness decision](artifacts/day12-static-candidate-readiness-decision.md) and [exact-candidate index](artifacts/day12-exact-candidate-evidence-index.md)
- [Day 13 completion reconciliation](artifacts/day13-epic20-completion-reconciliation.md) and [validation/claim ledger](artifacts/day13-validation-and-claim-ledger.md)
- [Day 14 closeout and residual handoff](artifacts/day14-final-epic20-closeout-and-handoff.md), [evidence index](artifacts/day14-closeout-evidence-index.md) and [retrospective input](artifacts/day14-retrospective-input.md)

## Residual Handoff

The [Epic 20 retrospective](../EPIC_20_RETROSPECTIVE.md) reconciles the ten sprint outcomes and carries their remaining product and release gaps forward.

The next release owner should follow the [Day 14 ordered gates](artifacts/day14-final-epic20-closeout-and-handoff.md#residual-handoff-with-acceptance-gates): approve version/changelog/static scope and name owners, commit/review intended inputs, select one clean candidate SHA, rebuild and independently verify two archives, install and run downstream consumers from the extracted candidate, complete exact-SHA Linux/macOS/Windows gates, decide the timing and external-evidence boundaries separately, and collect individual signoffs including rollback. A later source or checked-in documentation edit changes the candidate identity and requires affected proof to be repeated. Keep the readiness label **BLOCKED** until every adopted gate and the release maintainer's final approval pass; a passing local first-use route does not remove this stop.
