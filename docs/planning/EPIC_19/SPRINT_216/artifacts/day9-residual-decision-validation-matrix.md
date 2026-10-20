# Sprint 216 Day 9: Residual Decision And Validation Matrix

## Decision And Environment

The [finalized residual queue](../RESIDUAL_QUEUE.md) retains 21 ranked,
testable next-epic targets. Its [Day 8 source crosswalk](day8-residual-queue-intake.md)
maps every Sprint 207-215 residual and Epic 19 review/todo track. The
[Day 6](day6-project-plan-status-207-211.md) and
[Day 7](day7-project-plan-status-212-215.md) ledgers contain exactly one
selected-deliverable disposition for each of the 54 prior-sprint items.
All 54 are complete **for their selected scope**; no queued capability is
therefore promoted. The Sprint 216 items remain unassigned until the final
gate and closeout review.

Day 9 inspection ran on `sprint-216` under Darwin `24.6.0` x86_64, Apple
Clang `17.0.0`, Python `3.14.5`, with Doxygen at `/usr/local/bin/doxygen`
and Homebrew at `/usr/local/bin/brew`. This local host cannot substitute for
hosted Windows, candidate-specific Linux/macOS/Windows, or a configured
CHOLMOD external-evidence run. The commands below are the **selected future
validation matrix**, not Day 9 pass claims.

## Residual Reconciliation

| Completed selected outcome | Unclosed capability | Queue disposition |
| --- | --- | --- |
| 207 provider decision/local proof; 214 static package design | Public source provider, other providers, candidate release | **RQ-07**, **RQ-16**, **RQ-06**. The local proof is retained, not promoted. |
| 208 bounded hosted Cholesky evidence; 209 QR collection lane | Selected Windows freshness and broader platform parity | **RQ-04**, **RQ-05**, **RQ-15**. Sprint 208's older QR handoff is merged into the Sprint 209 QR target, not a second QR task. |
| 210 one LDLT allocation owner; 211 one test extraction | Further owner/cluster, shared parser upkeep | **RQ-09**, **RQ-10**, **RQ-19**. Sprint 211's missing full *local* CTest run was subsequently closed by Sprint 214's [60/60 local CTest](../../SPRINT_214/artifacts/day13-integrated-validation.md); candidate-hosted proof remains under RQ-06. |
| 212 threshold-free benchmark decision | Timing policy, broader/public/release benchmark evidence | **RQ-08**, **RQ-12**, **RQ-17**, **RQ-21**. Freshness is not speed or a timing gate. |
| 213 local-only API policy | Generated HTML publication, parser maintenance, broader API coverage | **RQ-14**, **RQ-19**, **RQ-20**. Hosted, retained-artifact and committed-HTML options share one future product-policy gate. |
| 214 ABI/release design | Shared artifact/ABI compatibility and candidate source release | **RQ-13** and **RQ-06**. Checklist completion does not prove either. |
| 215 blueprint and blocked pilot fallback | External v1 semantics, one correctness cell, publication, breadth and hosted/timing | **RQ-01/02/03/11/12/18**. Planning JSON and provisional `attempt.json` are not accepted `evidence.json`. |
| Epic review's storage, usability, oracle gaps | Compressed-first measurement, broader conformance, first-use/API route | **RQ-17/18/20** with revisit triggers, not claims attached to the prior sprint completions. |

No prior-sprint *item* remains status-less. The only supersession is of
duplicated **queue wording** (for example Sprint 208's QR handoff and Sprint
214's repeated provider/API references), not of the underlying acceptance
gate. Long-horizon RQ-11 to RQ-21 are intentionally deferred until their
named scope/dependency decision; they remain visible in the queue.

## State-Of-The-Art Evidence Decision

**No state-of-the-art claim is earned.** Sprint 215's pilot stopped before
either solver ran because the external v1 schema/validator are absent. There
is no accepted two-sided numerical cell, optional timing result, hosted
external artifact or publication approval. One validated local SPD/CHOLMOD
cell after RQ-01/02/03 could support only a bounded correctness statement.
The [queue's entry bar](../RESIDUAL_QUEUE.md#state-of-the-art-decision) for
reconsidering broader wording requires a pre-registered representative matrix
covering SPD, nonsymmetric and rectangular systems (and separately sampled
iterative/eigen/SVD families if named), pinned independent baselines,
multiple matrix structures/sizes, raw retained correctness and controlled
timing on at least two named hosted platform/toolchain classes, explicit
Windows proof if Windows is named, provenance/rights approval and product
review. This is necessary for review, not automatic approval or a fixed row
count. Day 13's assessment must preserve this no-claim decision unless an
actual new accepted evidence bundle changes it.

## Changed-Surface Inventory

At Day 9, tracked modifications are `README.md`, `INSTALL.md`,
`benchmarks/README.md`, `docs/api_reference.md`, `docs/maintainer_guide.md`,
`docs/planning/EPIC_19/PROJECT_PLAN.md`, `tests/corpus/README.md`,
`tests/corpus/schemas/report_index_fields.md`, and
`tests/test_external_baseline_claim_docs.py`. New Sprint 216 files are
`PLAN.md`, `WORKING_NOTES.md`, `RESIDUAL_QUEUE.md`, and Day 1-9 Markdown
artifacts. There is **no `.c` or `.h` diff or untracked source/header**.
The Python claim guard extends marker coverage across the changed public
and evidence docs, so its mutation cases are part of the required focused
suite. Docs and planning links/statuses need their own local audit. If later
edits add a C/header file, the gate changes immediately.

## Selected Validation Matrix

Run commands from the repository root. `PASS` means exit 0 and the stated
contract is inspected; merely producing ignored generated output is not a
publication result. Day 10 owns focused contracts; Day 11 owns install/full
quality decisions; Day 14 repeats affected checks after final edits.

| Surface / planned day | Exact command | Environment and expected outcome |
| --- | --- | --- |
| External claim guard / 10, 14 | `python3 -m unittest tests.test_external_baseline_claim_docs` | Python 3; all marker, mutation and negation cases pass. Text guard only, not solver proof. |
| Python syntax / 10 | `python3 -m py_compile tests/test_external_baseline_claim_docs.py` | Python 3; exit 0. |
| Selected benchmark wording / 10, 14 | `python3 tests/test_selected_performance_docs.py` | Python 3; exit 0 without threshold/performance promotion. |
| Support docs / 10, 14 | `make support-docs-guard` | Python 3; exit 0 with static/platform/release non-claims intact. |
| Generated API / 10, 14 | `make api-docs-freshness` | Doxygen and Python 3; `docs-check`, coverage, local-only workflow, routing and their regressions pass. `docs/api/html/` remains ignored local output. |
| Corpus/schema / 10, 14 | `python3 scripts/validate_corpus_schema.py` | Python 3; source-controlled schema and optional-data rules pass, no v1 external schema inferred. |
| Selected target manifest / 10, 14 | `python3 tests/test_selected_report_targets_manifest.py` | Python 3; selected Windows/benchmark/local-only metadata remains bounded. |
| Report tests / 10 | `python3 tests/test_normalize_report_index.py` and `python3 tests/test_run_external_comparison.py` | Python 3 and existing local fixtures; both exit 0. These exercise source-controlled-helper comparison contracts, not CHOLMOD. |
| Selected comparison freshness / 10, 14 | `make report-index-comparison-freshness` | Apple Clang/Make/Python; builds local library, generates ignored comparison output and passes selected freshness. Inspect generated statuses and paths; no external-baseline pass inference. |
| Canonical benchmark tests / 10 | `make bench-canonical-report-freshness-tests` | Local bench binaries; exit 0, including threshold-free metadata and negative fixtures. |
| Canonical benchmark freshness / 10, 14 | `make bench-canonical-report-freshness` | Local runner and bundled matrices; exit 0 with `threshold=n/a`, `--repeat 1`; inspect row and preserve timing non-claims. No timing acceptance threshold. |
| Blocked-pilot synthetic behavior / 10, 14 | `python3 -m unittest tests.test_run_cholmod_pilot` | Python 3; all synthetic/fail-closed tests pass. No external solver is run or accepted v1 result created. |
| Static/package guards / 11 | `python3 tests/test_static_install_contract.py`, `bash scripts/static_package_deferral_check.sh`, `python3 tests/test_package_manager_deferral_guard.py` | Python 3, Bash and CMake; each exit 0. Unsupported shared/provider claims remain rejected. |
| Local package-provider boundary / 11 | `bash scripts/package_manager_deferral_check.sh` | Homebrew installed; selected developer-mode proof may execute in a temporary tap with MIT metadata. Record exact result or an explicit environment blocker; never treat local proof as public provider support. Run serially. |
| Installed static consumers / 11 | `bash tests/test_install.sh` then `bash tests/test_cmake_install.sh` | Make, Apple Clang, CMake and `pkg-config`; exact static install/consumer and negative fixtures pass. The Make script invokes `make clean`; run serially after Day 10 report checks, then rebuild before any final report rerun. |
| Source gate / 11 or after any source edit | `make format && make lint && make test` | **Required only if any `.c` or `.h` file changes**, including newly untracked files. All three phases must pass in order; no such file changed through Day 9. Do not run on a tree left in an alternate sanitizer/coverage build mode without a clean reset. |
| Documentation coverage / 10, 14 | `make docs-check` | Doxygen; 18 checked-in public headers and matching reference/source pages expected. This is local generation, not hosted publication. |
| Planning status/links / 9, 14 | `git diff --check` plus local relative-link, 54-item status, 21-queue-ID and per-entry-field audits | `git diff --check` exits 0; all local links resolve, every 207.1-215.6 item has one status and every RQ-01 to RQ-21 entry has owner/dependency/acceptance/stop/claim. Untracked Markdown needs explicit whitespace scanning because `git diff --check` alone omits it. |

`bash tests/test_install.sh` mutates `build/` through `make clean` and must
not overlap Make/report/benchmark commands. The CMake install script uses a
temporary build but still runs serially for clear attribution. The focused
comparison and canonical benchmark checks create ignored local artifacts;
their row/status/provenance content must be inspected, not just exit status.

## Named Validation Gaps And Stop Conditions

| Gap | Required treatment before stronger wording |
| --- | --- |
| External v1 schema and validator absent | Expected **product blocker**, not a passing schema check. Do not run or publish the pilot as accepted evidence; RQ-01 first. |
| CHOLMOD actual solver, runtime provenance and fixture rights absent | `not_run`/blocked, not pass. RQ-02/03 own execution and publication approval. |
| Current hosted Windows QR and selected-freshness promotion evidence not inspected | Keep selected QR/Cholesky re-deferred; no hosted claim from local guard results. RQ-04/05 own exact run/job/artifact review. |
| Candidate-specific static release and shared-loader jobs absent | Preserve static-source support only. RQ-06/13 own candidate and shared proof. |
| No current hosted external/timing matrix | State-of-the-art, portable speed and platform parity remain unearned; RQ-11/12/17/18 are separate future gates. |
| Historical Sprint 212-215 checks predate PR review fixes | Current-tree commands above must be rerun and logged with exit status/platform on Days 10-14. A historical pass is not a new gate result. |

If any required Day 10/11 command fails, stop promotion and record the exact
failure; repair and rerun before closeout. Do not collapse an expected absent
dependency, local-only result or skipped hosted lane into `PASS`.

## Day 9 Validation

Day 9 itself edited Markdown only. On Darwin `24.6.0` x86_64, the queue audit
passed for 21 unique entries with all five required fields; the status audit
passed for 54 unique prior-sprint items; and local links passed. The focused
claim checks passed: `python3 -m unittest tests.test_external_baseline_claim_docs`
(5 tests), `python3 tests/test_selected_performance_docs.py`, and
`make support-docs-guard`. `make docs-check` passed with 18 public-header
reference pages, as did trailing-whitespace inspection and `git diff --check`.
The full changed-surface matrix above is scheduled for Days 10-14; this record
does not claim source-quality, CHOLMOD, or hosted execution.
