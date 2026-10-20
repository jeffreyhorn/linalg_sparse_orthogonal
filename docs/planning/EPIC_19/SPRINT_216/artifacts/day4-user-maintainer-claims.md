# Sprint 216 Day 4: User And Maintainer Claim Calibration

## Scope

Applied the Day 3 [claim inventory](day3-claim-inventory.md) to
[README](../../../../../README.md), [INSTALL](../../../../../INSTALL.md), and
the [maintainer guide](../../../../maintainer_guide.md). This is a wording and
guard update, not a new platform, package, solver, ABI, release, benchmark, or
external-baseline proof. Day 5 still owns benchmark, API reference, corpus,
and planning wording; Days 6-7 own the stale Epic status snapshot.

## Edits And Evidence

| Claim | Edit and reason | Evidence boundary |
| --- | --- | --- |
| C01-C02 | README now calls CSR/CSC dispatch compressed-storage paths rather than universally faster large-matrix paths; the CSR LU API heading no longer says "high-performance." | [Sprint 212 policy](../../SPRINT_212/artifacts/day5-product-policy-decision.md) selected a threshold-free CSC refactor freshness row, not CSR LU superiority. [Benchmark guide](../../../../../benchmarks/README.md#report-index-handoff) owns measured context. |
| D4-1, newly found | README's LU complexity note formerly said sparse structured matrices are "usually much better" than the worst case. It now names sparsity, fill, and ordering as work drivers without a comparative claim. | The O(n^3) worst-case table is an algorithmic bound, not a portable measurement. No selected benchmark proves a general LU runtime relationship. |
| C05 | INSTALL retains static Unix Make/`pkg-config`, installed CMake, and Windows MSVC CMake install/downstream cells with their negative boundaries. A new paragraph distinguishes historical/local installed-consumer results and registered CI lanes from a fresh hosted candidate result or release approval. | [Sprint 214 validation](../../SPRINT_214/artifacts/day13-integrated-validation.md) and [release checklist](../../SPRINT_214/RELEASE_READINESS_CHECKLIST.md); no shared artifact or approved release. |
| C06-C07 | INSTALL now has a separate external-baseline row: Sprint 215 blueprint only, pilot blocked before either solver ran, no accepted correctness or timing. The broad ecosystem row cites this newer evidence instead of relying only on Epic 17 context. | [Sprint 215 blocked outcome](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md) and [claim matrix](../../SPRINT_215/artifacts/day11-evidence-and-claim-matrix.md). Missing v1 schema/validator remains a blocker. |
| C09 | Maintainer guide's evidence-owner table and generated API section now name Sprint 213 as the latest local-only publication decision, retaining Sprint 179/186/204 as policy history. | [Sprint 213 decision](../../SPRINT_213/artifacts/day5-product-policy-decision.md); no hosted, retained, or committed HTML. |
| C10-C11 | Maintainer guide keeps the Sprint 214 checklist as future gates and Sprint 215's planning/provisional/accepted evidence states separate. It adds explicit current-checkout static install rerun commands and says workflow registration is not a fresh hosted result. | [Sprint 214 checklist](../../SPRINT_214/RELEASE_READINESS_CHECKLIST.md), [Sprint 215 claim matrix](../../SPRINT_215/artifacts/day11-evidence-and-claim-matrix.md). |

The user-facing [solver-selection guide](../../../../solver_selection.md)
already routes platform/package support to INSTALL and excludes broad
performance and external-library parity. README's Windows comparison wording
still says Cholesky freshness is re-deferred and QR is evidence-collection
only; the selected performance and blocked pilot paragraphs remain unchanged.

## Focused Regression

[External-baseline claim tests](../../../../../tests/test_external_baseline_claim_docs.py)
now include INSTALL in their required blocked-pilot markers and unsupported
promotion mutations. Removing the blocked-before-solvers wording from either
README or INSTALL fails the test. Appending a broad positive pilot claim on
any guarded surface also fails. This is a textual consistency guard, not a
CHOLMOD run or evidence validator.

## Validation And Handoff

Day 4 changes only Markdown and one Python documentation test, not `.c` or
`.h`. The following local checks passed on the Day 4 wording and test changes:

| Check | Result and limit |
| --- | --- |
| `make docs-check`; `make support-docs-guard` | Passed; source-controlled docs coverage and support routing only. |
| `python3 -m unittest tests.test_external_baseline_claim_docs`; `python3 tests/test_selected_performance_docs.py` | Passed, including INSTALL missing-blocker and unsupported-promotion mutations; textual guards, not external or benchmark execution. |
| `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh` | Passed, including local Homebrew proof and static/shared deferral checks; no public provider or dynamic ABI claim. |
| `make api-docs-freshness` | Passed generated coverage, local-only staging, and routing; no hosted publication. |
| Local-link and whitespace scans; `git diff --check` | Passed after the root-document link checker was corrected to use each file's directory. |

The [working-notes tracker](../WORKING_NOTES.md#changed-surface-tracker)
records this as documentation/test-only validation, not a hosted or C gate.
Keep the external v1 validator, Windows selected promotion, public provider,
shared ABI, hosted API, timing threshold, and release approval as blockers or
residuals. Days 5-7 should complete their assigned claim/status corrections
without treating this wording pass as product validation.
