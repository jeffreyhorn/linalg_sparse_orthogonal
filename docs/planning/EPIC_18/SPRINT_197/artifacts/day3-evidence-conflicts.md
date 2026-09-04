# Day 3 Evidence Conflict and Gap Review

## Purpose

Day 3 classifies conflicts and stale-risk areas before claim edits begin. It
uses the Day 2 outcome ledger, Epic 18 planning artifacts, Epic 17 residuals,
and claim-sensitive public documentation as inputs.

## Evidence Conflict Matrix

| Area | Conflict risk | Classification | Resolution |
| --- | --- | --- | --- |
| Sprint numbering | Requested `SPRINT_197` final-validation path conflicts with the project-plan label `Sprint 206`. | Human-review-required planning mismatch. | Keep the requested path and cite final-validation items 206.1-206.6. |
| Sprint 198-205 outcomes | Implementation artifacts are not present yet. | Future-missing evidence. | Do not cite planned sprint work as completed proof. |
| Package/Homebrew | Sprint 198 plan could sound like current Homebrew support. | Human-review-required license and proof dependency. | Require approved license metadata and passing proof before promotion. |
| Windows selected Cholesky | Guarded workflow path could be overstated as promoted selected freshness. | Hosted-evidence dependency. | Require hosted artifact review and manifest metadata promotion. |
| PowerShell validation | PowerShell snippet ownership could be conflated with report freshness. | Hosted/local evidence boundary. | Keep PowerShell validation separate from CMake, CTest, and report generation. |
| Benchmark evidence | Linux selected benchmark freshness could be overstated as portable performance. | Selected hosted evidence boundary. | Require one additional hosted platform/row and preserve non-portable wording. |
| QR comparison | Local QR incompatible evidence could be overstated as Windows QR parity. | Selected platform evidence dependency. | Require MSVC/CMake proof and hosted artifacts for exact target promotion. |
| Generated API | Local Doxygen generation could be mistaken for hosted publication. | Product-decision dependency. | Require explicit publication or local-only policy before claim changes. |
| Reliability | One symbolic allocation proof could be overstated as broad reliability. | Selected-owner boundary. | Require a new owner-specific proof for each additional claim. |
| Support docs | Repeated support caveats could drift across docs. | Stale/duplication risk. | Treat `INSTALL.md#support-readiness-matrix` as current public support truth. |

## Evidence Type Rules

| Evidence type | Closeout use | Constraint |
| --- | --- | --- |
| Planning artifacts | Establish selected intent and scope. | Not implementation proof. |
| Public docs | Establish current claims and non-claims. | Must match evidence and support matrix. |
| Maintainer docs | Establish ownership and validation interpretation. | Must separate selected, local, hosted, generated, and advisory evidence. |
| Local commands | Validate current checkout surfaces. | Not hosted platform proof. |
| Hosted CI | Validate named platform/job/artifact scope. | No cross-platform inheritance. |
| Generated artifacts | Support local inspection/freshness when current. | Stale/advisory unless selected gates require them. |
| Optional dependencies | Support claims only when available and recorded. | Missing tools are residuals, not passes. |
| Human decisions | License, publication, support-tier, and promotion choices. | Block stronger claims until approved and recorded. |

## Stale-Risk And Gap List

| Gap | Current status | Required future evidence |
| --- | --- | --- |
| Homebrew/package-manager support | Not claimed. | Approved metadata, formula proof, package guards, install checks, docs. |
| Selected Windows Cholesky freshness | Guarded workflow only. | Hosted selected artifact review and manifest promotion. |
| Additional allocation-failure owner | Not selected yet. | Owner invariant, deterministic failure/retry tests, focused gate. |
| Additional review-surface reduction | Not selected yet. | Cluster ranking, behavior-preserving extraction, guard, focused tests. |
| Additional benchmark platform | Not selected yet. | Hosted platform/row evidence and benchmark freshness checks. |
| Windows QR incompatible freshness | Local-only baseline. | MSVC/CMake proof and selected hosted artifact review. |
| Generated API publication | Local-only. | Publication decision and freshness/link policy implementation. |
| Support/adoption consolidation | Planned only. | Quick-reference implementation, support truth consolidation, claim guards. |
| State-of-the-art status | Explicit non-goal. | Broad comparative correctness, performance, package, ABI, platform, and release evidence. |

## Claim Boundaries For Day 4

- Package-manager support remains unclaimed until provider-specific proof and
  approved metadata exist.
- Windows evidence remains selected and hosted/job-specific.
- PowerShell validation owns workflow snippet parsing, not report freshness or
  package/install execution.
- Benchmark evidence remains methodology-bound and non-portable.
- Comparison rows remain fixture-local and target-specific.
- Generated API HTML remains local-only until a publication decision changes
  that status.
- Reliability claims remain selected-owner scoped.
- State-of-the-art, release, broad ABI, broad platform, and broad ecosystem
  parity remain non-claims.

## Acceptance Evidence

- No contradicted or stale-risk evidence remains unclassified.
- Hosted, local, generated, checked-in, optional, and human-review-required
  evidence types are separated.
- Claim edits can proceed from explicit boundaries in Day 4.

