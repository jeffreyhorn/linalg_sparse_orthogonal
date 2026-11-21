# Sprint 226 Day 12: Static Candidate Readiness Decision

## Decision

**Readiness label: BLOCKED — no selected candidate is release-ready.** This is the documented Day 12 technical gate result under the [Sprint 214 checklist](../../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md), **pending release-owner and other required signatures**. It is not an owner-signed release decision. No tag, public release, source asset, provider package or binary artifact is authorized by this record.

The proposed source basis is still HEAD `c42d6715fad7e895bae4941a006df3563f1afc38` (tree `934a16b09ada33e12f726ec944fc675bd7d244f4`), but no one selected it as a candidate. `VERSION=2.2.0`; root `CHANGELOG.md` is missing; `README.md` and `INSTALL.md` are modified and Sprint 226 documentation is untracked. There is no `sprint-226` PR, reviewed clean commit, `v2.2.0` tag or release. The [exact-candidate evidence index](day12-exact-candidate-evidence-index.md) marks every missing candidate receipt. A historical base SHA is an inventory fact, not a release-candidate identity.

The Day 3–4 source archive hash `cdfceff5ef4fdf405f8bf3b15e6a7fe608d5f4566345ea3fd2f8d6ec0e27be8f` is a **same-host reproducible exploratory base artifact**, not an approved candidate archive. It lacks the Day 9–12 documentation corrections and changelog. The [Day 5 install map](day5-archive-install-map-and-platform-matrix.md), [Day 9 clean-prefix consumers](day9-clean-prefix-first-use.md) and [Day 10 first-use audit](day10-first-use-audit.md) establish useful local procedure and static-package behavior, but do not establish the provenance chain for a future candidate. No old hash or job result is carried forward as a candidate pass.

## Platform-specific evidence and disposition

| Proposed platform | Available source/run and artifact | Day 12 candidate result |
| --- | --- | --- |
| Linux GCC Make/`pkg-config` and CMake static routes, with selected width lane | [Historical base run 44433727330](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44433727330); [Day 8 provenance matrix](day8-platform-provenance-and-first-use-inventory.md) records package/compile job logs and hashes. Reviewed CMake parity stopped when the runner shut down at test 32/63. | **Missing / blocked.** No exact-candidate run; historical full CTest did not finish. |
| macOS Apple Clang Make/`pkg-config` and CMake static routes, with selected width lane | [Historical base run 44433727057](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44433727057) passed named jobs; [Day 8 matrix](day8-platform-provenance-and-first-use-inventory.md) identifies log hashes. Local Day 9 consumers passed on the exploratory base archive. | **Missing / blocked.** Historical hosted and local results do not include the edited candidate docs or a frozen candidate SHA. |
| Windows MSVC 2022 CMake static installed subset | [Historical base run 44433727133](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44433727133) passed named CMake/CTest and installed downstream jobs; [Day 8 matrix](day8-platform-provenance-and-first-use-inventory.md) identifies logs. | **Missing / blocked.** No exact-candidate run. No Windows Make/`pkg-config` execution parity is claimed. |

The platform matrix remains *proposed*, not release-owner-adopted. Hosted jobs checked out the historical Git SHA; they did not consume the exploratory tarball. No install-specific hosted ZIP or candidate archive-to-install digest connects those base runs to the final package. The [Day 11 checklist review](day11-checklist-review-and-rollback.md) maps each release gate to its reviewer and missing exact-candidate receipt.

## Timing decision, separate from static correctness

**Sprint 225 timing: BLOCKED / UNMEASURED.** The [base timing run 44433727385](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44433727385) failed preflight because accepted baseline `benchmarks/baselines/refactor-csc-nos4/active.json` was absent. Its downloaded ZIP, SHA-256 `d89a5a06180b34e431a150daa71abf8412450b073eb65d48594657505b9353df`, records no candidate samples or threshold verdict. This is not a measured performance regression, passing timing gate, or speed claim. Selected report freshness is threshold-free and cannot close it.

No product/release owner has approved whether a static source-only decision may proceed with the timing gate explicitly excluded. Until that scope decision exists, timing remains an **unresolved readiness input**. Readiness is independently blocked by the absent candidate, candidate-specific platform/archive proof and approvals even if a future owner excludes timing claims. An accepted baseline plus exact-candidate measured run is the other closure path.

## Claim boundary for this decision

The evidence supports only these narrow statements: on a local macOS host, the exploratory base source built a static package and the installed Make/CMake consumers ran with expected 32/64-bit width behavior; historical base-commit macOS and Windows named CI subsets passed, while historical Linux parity was interrupted. The checked-in first-use instructions have been corrected and verified in the working tree. These statements identify their source and do **not** certify a release candidate.

This record makes **no** tagged-release, published-source-archive, binary-package, package-manager/provider, shared-library, dynamic ABI, Windows Make/`pkg-config` parity, external CHOLMOD parity, portable speed, broad state-of-the-art, or release-ready claim. Sprint 222's two-sided local numerical work remains unaccepted for external publication because provenance and rights review are incomplete. Generated API HTML remains local-only. Product and release owners must approve any public wording against a future exact candidate.

## Outstanding actions and rollback

1. Release/API owners: approve comparison base, version-specific changelog, platform scope and a clean committed candidate ID/SHA; record tag/signing policy only after a reviewed source snapshot exists.
2. Build and independent verification owners: regenerate and compare two candidate archives, audit source/license manifest, install from an extracted candidate and map/have hashes for every installed file.
3. Linux/macOS/Windows platform owners: run the adopted exact-SHA jobs and retain toolchain, width, log, test-inventory and artifact receipts. Linux must complete its reviewed CMake parity; required provider gate must finish or receive an explicit scoped decision.
4. Documentation/API/license/product owners: review packaged first-use text, header/ABI/non-claim boundary, source rights and Sprint 225 timing scope; record actual approvals tied to that SHA.
5. Release maintainer and rollback owner: approve asset destination, retention, communications and [Day 11 trigger/recovery procedure](day11-checklist-review-and-rollback.md#proposed-rollback-triggers-and-recovery-sequence), then sign a final go/no-go record. No rollback was performed; no published asset exists to withdraw.

| Required signer | Recorded name and signature | Status |
| --- | --- | --- |
| Release maintainer / final decision owner | **None** | Pending; cannot authorize release. |
| Product and timing-scope owner | **None** | Pending. |
| Build/release and independent archive verifier | **None** | Pending. |
| Linux, macOS and Windows CI/platform reviewers | **None** | Pending. |
| API/ABI and documentation reviewers | **None** | Pending. |
| License/legal reviewer | **None** | Pending. |
| Rollback owner | **None** | Pending. |

The [Day 11 approval register](day11-approval-and-outstanding-gates.md) names the evidence each role must review. Silence, a green historical CI run and this technical record are not signatures. The Day 13 Epic 20 reconciliation should carry the candidate, timing, provider and approval gaps forward as blocked outcomes rather than describe partial evidence as a finished release.
