# Sprint 149 Day 13: Integrated Evidence Review

## Scope

Day 13 reviews the Sprint 149 evidence chain from product decision through
workflow, metadata checks, consumer checks, documentation, local validation,
and hosted-CI evidence status.

## Decision Alignment

The Day 4 product decision selected conditional promotion:

- Windows may claim reviewed CMake install/downstream validation for the
  maintained static-first package surface.
- The claim remains hosted MSVC CMake scoped.
- The claim does not include Windows Makefile parity, Windows `pkg-config`
  execution parity, package-manager support, shared-library support, dynamic
  ABI support, runtime-loader behavior, or broad Windows parity.
- Hosted Windows evidence is required before closeout treats the promotion as
  fully earned.

The final workflow and documentation align with that decision:

| Surface | Evidence | Status |
| --- | --- | --- |
| Workflow job name | `.github/workflows/windows-ci.yml` uses `Windows reviewed CMake install/downstream validation path` | Aligned |
| Workflow step name | `.github/workflows/windows-ci.yml` uses `Run reviewed CMake install/downstream validation proof` | Aligned |
| Workflow comments | Workflow header and job comments describe the lane as CMake install/downstream scoped and preserve non-claims | Aligned |
| README | Cross-platform CI contract names reviewed Windows CMake install/downstream validation and preserves non-claims | Aligned |
| INSTALL | Maintained install and verification sections name the reviewed Windows CMake install/downstream lane and preserve non-claims | Aligned |
| Maintainer guide | Package/platform ownership guidance lists the Windows reviewed CMake install/downstream checks and non-claims | Aligned |
| Report manifest | CI reviewed-lanes row includes Windows reviewed CMake install/downstream validation and records hosted logs as external | Aligned |

## Evidence Matrix

| Evidence Area | Artifact / File | Status |
| --- | --- | --- |
| Install-lane intake | `artifacts/day1-install-intake.md` | Complete |
| Existing Windows package audit | `artifacts/day2-windows-package-audit.md` | Complete |
| Promotion criteria | `artifacts/day3-promotion-criteria.md` | Complete |
| Product decision | `artifacts/day4-product-decision.md` | Complete |
| Workflow design | `artifacts/day5-workflow-design.md` | Complete |
| Workflow implementation | `artifacts/day6-workflow-implementation.md` and `.github/workflows/windows-ci.yml` | Complete |
| Metadata check design | `artifacts/day7-metadata-check-design.md` | Complete |
| Metadata implementation | `artifacts/day8-metadata-implementation.md` and `.github/workflows/windows-ci.yml` | Complete |
| Consumer proof design | `artifacts/day9-consumer-proof-design.md` | Complete |
| Consumer proof implementation | `artifacts/day10-consumer-implementation.md` and `.github/workflows/windows-ci.yml` | Complete |
| Public documentation alignment | `artifacts/day11-docs-alignment.md`, `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, `tests/corpus/manifests/report_families.tsv` | Complete |
| Local validation | `artifacts/day12-local-validation.md` | Complete |
| Hosted Windows validation | GitHub Actions run for `sprint-149` | Pending; no PR or branch run exists yet |

## Hosted CI Status

Local `gh` inspection found no hosted evidence yet:

```sh
gh pr view --json number,url,headRefName,baseRefName,state,statusCheckRollup
```

Result: no pull request found for branch `sprint-149`.

```sh
gh run list --branch sprint-149 --limit 10 --json databaseId,displayTitle,event,headBranch,status,conclusion,workflowName,url,createdAt
```

Result: no workflow runs found for branch `sprint-149`.

Conclusion: Sprint 149 can close local implementation and local validation, but
the reviewed Windows CMake install/downstream promotion remains pending until
the branch is pushed and GitHub Actions runs the
`Windows reviewed CMake install/downstream validation path` job.

## Stale-Wording Review

Focused search:

```sh
rg -n "supplemental CMake install/downstream|supplemental confidence|supplemental.*Windows.*install|Windows install-validation parity|no separate reviewed install-validation lane|separate reviewed install-validation lane" \
  README.md INSTALL.md docs/maintainer_guide.md tests/corpus/manifests/report_families.tsv .github/workflows/windows-ci.yml docs/planning/EPIC_13/SPRINT_149/artifacts/day11-docs-alignment.md docs/planning/EPIC_13/SPRINT_149/artifacts/day12-local-validation.md
```

Result: PASS for public docs and workflow. Remaining hits are limited to
historical sprint-artifact context or the recorded search command text itself.

Reviewed-lane wording search:

```sh
rg -n "Windows reviewed CMake install/downstream validation path|reviewed CMake install/downstream validation|reviewed CMake install/downstream" \
  .github/workflows/windows-ci.yml README.md INSTALL.md docs/maintainer_guide.md tests/corpus/manifests/report_families.tsv docs/planning/EPIC_13/SPRINT_149/artifacts/day*.md
```

Result: PASS. The workflow, public docs, report manifest, and sprint artifacts
consistently use the narrow reviewed CMake install/downstream wording.

## Residuals

| Residual | Disposition |
| --- | --- |
| Hosted Windows validation proof unavailable before PR/run creation | Carry to Day 14 closeout and PR validation; do not mark hosted evidence passed locally |
| Windows Makefile install/uninstall parity | Explicit non-claim; candidate only for a future sprint if product scope changes |
| Windows `pkg-config` execution and downstream parity | Explicit non-claim; candidate only after a maintained Windows `pkg-config` toolchain is selected |
| Package-manager installation or resolver behavior | Explicit non-claim; defer until package distribution is product scope |
| Shared-library packaging, dynamic ABI, and runtime-loader behavior | Explicit non-claims; require separate product decision before implementation |
| Broad Windows parity | Explicit non-claim; current supported claim remains hosted MSVC CMake scoped |

## Sprint 150 Handoff Candidates

- Preserve the Sprint 149 package-lane boundary when starting Sprint 150 QR
  work; QR changes should not depend on Windows Makefile or `pkg-config`
  package parity.
- After the Sprint 149 PR exists, inspect the hosted Windows job named
  `Windows reviewed CMake install/downstream validation path`; if it fails,
  fix the failing criterion or roll wording back to pending/supplemental before
  merge.
- If future QR or package work adds public headers, update the fixed Windows
  installed-header count intentionally in the reviewed install/downstream lane.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Sprint 149 claims are backed by explicit evidence or marked residual. | Complete | Evidence matrix and residual table separate local proof from hosted pending proof. |
| Hosted-only proof gaps are not hidden as local success. | Complete | Hosted CI status records no PR and no branch runs yet. |
| Sprint 150 QR work is not blocked by package-lane ambiguity. | Complete | Handoff preserves the static-first CMake install/downstream boundary and non-claims. |
