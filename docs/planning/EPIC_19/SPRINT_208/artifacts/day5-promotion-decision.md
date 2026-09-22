# Sprint 208 Day 5: Promotion Decision

## Purpose

Apply the Day 4 promotion criteria to the Day 2 hosted evidence and Day 3
row/path traceability, then select either promotion or continued re-deferral.

## Decision

Sprint 208 selects **continued re-deferral with stronger proof of absence**.

The latest hosted evidence is accepted as valid current evidence for the exact
bounded Windows selected Cholesky workflow path, but it does not satisfy the
full promotion threshold because source-controlled manifest metadata,
generated support tier, generated non-claim wording, and public/maintainer
documentation do not all promote Windows selected freshness together.

## Evidence Accepted

| Evidence | Day 5 status |
| --- | --- |
| Latest relevant hosted run `35731703320` on `master` | Accepted |
| Commit `d75118349269c6805654070eb44c9c57608b3e47` with clean hosted generated row provenance | Accepted |
| Job `Windows selected Cholesky comparison freshness (MSVC)` succeeded | Accepted |
| Artifact `sprint190-windows-selected-comparison-cholesky` with ID `10696020870` is unexpired and downloadable | Accepted |
| Artifact contains exactly the six expected selected Cholesky files | Accepted |
| `study.tsv` contains exactly six expected Cholesky row IDs | Accepted |
| All six selected rows have `status=pass` | Accepted |
| Row platform is `windows-amd64` and compiler is `cmake-probe:Visual Studio 17 2022:Release` | Accepted |
| Generated artifact path uses Windows backslashes and matches current normalizer behavior | Accepted as path evidence, not claim promotion |

## Promotion Criteria Evaluation

| Criterion | Result | Rationale |
| --- | --- | --- |
| Current hosted Windows run inspected | Met | Run `35731703320` inspected. |
| Exact selected job succeeded | Met | Selected Cholesky job completed with `conclusion=success`. |
| Artifact available | Met | Artifact ID `10696020870` is present, unexpired, and downloadable. |
| Artifact membership exact | Met | Six expected selected files are present. |
| Row identity exact | Met | Six expected Cholesky row IDs match manifest. |
| Row status clean | Met | All rows pass and record current source commit. |
| Platform and compiler bounded | Met | Rows record Windows AMD64 and MSVC/CMake probe. |
| Path handling guarded | Mostly met | Existing tests cover backslash, mixed separator, absolute Windows suffix, near-match, stale, and wrong-target behavior. Day 9-Day 10 should still review duplicate, extra-row, and artifact-mismatch cases. |
| Manifest metadata positive and aligned | Not met | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` still lists Linux/macOS workflow metadata only. |
| Support and claim fields agree | Not met | Manifest and generated rows remain `support_tier=local_only`. |
| Generated evidence wording agrees | Not met | Generated rows and `summary.md` still include `no hosted CI proof` and `no Windows report freshness`. |
| Public and maintainer docs agree with promotion | Not met | Current docs intentionally describe guarded workflow evidence and re-deferral. |
| Broad non-claims stay explicit | Met | Broad Windows, package/ABI, performance, release, and state-of-the-art non-claims remain explicit. |

## Why Promotion Is Rejected

Promotion would require adding Windows metadata to the selected target row and
claiming selected Windows freshness while the generated evidence still calls
the report `local_only` and explicitly says `no hosted CI proof` and
`no Windows report freshness`.

That would create a contradiction between the source-controlled manifest,
generated rows, generated summary, and public docs. Sprint 208 therefore
keeps the selected manifest unpromoted and treats the hosted run as reviewed
workflow evidence that can support a future promotion after generated claim
semantics are changed deliberately.

## Rejected Stronger Claims

Day 5 rejects these stronger claims for this sprint:

- promoted selected Windows Cholesky freshness;
- broad Windows report freshness;
- Windows selected oracle freshness;
- Windows selected benchmark freshness;
- QR incompatible Windows comparison freshness;
- unselected Windows comparison families;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support or package-manager platform parity;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- broad Windows parity;
- portable performance claims;
- release readiness;
- external-library ecosystem parity;
- state-of-the-art status.

## Selected Implementation Boundary

Day 6-Day 14 implementation should stay inside this boundary:

| Surface | Allowed work |
| --- | --- |
| Selected target manifest | Keep Windows platform absent; strengthen exact re-deferral and future-promotion tests. |
| Manifest tests | Add or adjust assertions for support tier, claim scope, non-claims, workflow metadata alignment, expected rows, and required files. |
| Windows workflow | Preserve bounded selected Cholesky job and six-file upload; strengthen drift guards if gaps are found. |
| PowerShell guards | Preserve `--require-pwsh` hosted owner and no-Windows selected manifest checks; strengthen diagnostics if needed. |
| Normalizer tests | Add duplicate, extra-row, artifact-mismatch, or support-tier/non-claim diagnostics only if review finds gaps. |
| Public docs | Say the latest hosted evidence was reviewed but selected freshness remains re-deferred. |
| Maintainer docs | Document exact blockers and validation commands for the retained re-deferral. |
| Planning docs | Record Sprint 208 as a current re-deferral path with evidence links and residual blockers. |

Day 6-Day 14 implementation must not:

- add `windows` to `workflow_platforms`;
- add `.github/workflows/windows-ci.yml` to the selected Cholesky manifest
  row's positive workflow metadata;
- add `sprint190-windows-selected-comparison-cholesky` to the selected
  Cholesky manifest row's positive workflow artifact metadata;
- remove generated `local_only`, `no hosted CI proof`, or
  `no Windows report freshness` wording without a promotion design;
- imply broad Windows support, package support, ABI support, performance
  claims, release readiness, external-library parity, or state-of-the-art
  status.

## Residual Blockers

| Residual blocker | Evidence needed to close later |
| --- | --- |
| Generated support tier remains `local_only` | Generator and manifest support-tier design for exact hosted selected Windows Cholesky evidence. |
| Generated rows and summary retain `no hosted CI proof` | Generated non-claim vocabulary that can distinguish hosted selected evidence from broad hosted proof. |
| Generated rows and summary retain `no Windows report freshness` | Narrow Windows selected Cholesky wording that does not imply broad Windows report freshness. |
| Selected manifest lacks Windows metadata | Platform-aligned manifest row update after generated semantics and docs agree. |
| Public docs intentionally preserve guarded-workflow wording | Docs update after all source-controlled evidence surfaces agree. |

## Day 5 Outcome

Item 208.2 is complete as a deliberate re-deferral decision. The sprint will
use the remaining days to preserve the reviewed hosted evidence, strengthen
absence and future-promotion guards, calibrate docs, and validate the
re-deferred claim boundary.

