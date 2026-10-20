# Epic 19 Retrospective

**Epic:** 19 - Productization, Platform Proof and Evidence Maturity
**Sprints:** 207-216
**Status:** Sprint 216 selected deliverables closed on this branch; review and merge remain pending.

## Objective And Evidence Rule

The [project plan](PROJECT_PLAN.md) set out to close a small number of
productization and evidence gaps completely, or to make an explicit,
guarded deferral. The [Day 1 source inventory](SPRINT_216/artifacts/day1-epic-evidence-intake.md)
and [Day 2 reconciliation](SPRINT_216/artifacts/day2-outcome-validation-reconciliation.md)
compare merged Sprint 207-215 decisions, validation records and closeouts.
The [Day 6](SPRINT_216/artifacts/day6-project-plan-status-207-211.md) and
[Day 7](SPRINT_216/artifacts/day7-project-plan-status-212-215.md) ledgers give
each of their 54 project-plan items one evidence-backed disposition.

All 54 are **complete for their selected deliverables**. This includes
decisions to re-defer Windows freshness, timing thresholds, generated HTML
publication and shared-library work, plus the planned fail-closed external
pilot fallback. It does **not** mean those larger capabilities were promoted.
Sprint 216 items 216.1-216.6 are now complete for their selected audit,
documentation, validation, retrospective and queue deliverables after the
[Day 14 local closeout](SPRINT_216/artifacts/day14-integrated-validation-closeout.md).
The [Sprint 216 retrospective](SPRINT_216/RETROSPECTIVE.md) records its
process lessons and exact local validation scope.
Epic 19's selected work is closed on this branch, subject to review and merge;
none of the residual product capabilities is promoted by that status.

## Sprint Outcomes

| Sprint | Evidence-backed selected outcome | Boundary and next gate |
| --- | --- | --- |
| 207 | The [provider decision](SPRINT_207/artifacts/day5-provider-decision.md) kept public package providers deferred and retained a developer-mode local Homebrew static source formula proof; [validation](SPRINT_207/artifacts/day12-integrated-validation.md) checked that bounded path. | No public tap, Homebrew/core, bottles, Linuxbrew or binary distribution. [RQ-07/RQ-16](SPRINT_216/RESIDUAL_QUEUE.md) require separate provider approval and non-local proof. |
| 208 | The [hosted artifact intake](SPRINT_208/artifacts/day2-hosted-artifact-inventory.md) reviewed six selected Windows Cholesky files/rows; the [promotion decision](SPRINT_208/artifacts/day5-promotion-decision.md) re-deferred selected freshness because generated support tier and non-claims did not align. | Historical bounded hosted observation is not a current promoted row. [RQ-04/RQ-15](SPRINT_216/RESIDUAL_QUEUE.md) retain exact artifact and metadata/claim alignment. |
| 209 | A [Windows/MSVC QR evidence lane](SPRINT_209/artifacts/day5-workflow-implementation.md) and local inspection guards landed; the [manifest decision](SPRINT_209/artifacts/day8-manifest-decision.md) found no inspectable hosted QR artifact and re-deferred promotion. | Workflow registration is not execution. [RQ-05/RQ-15](SPRINT_216/RESIDUAL_QUEUE.md) require a hosted six-file artifact and selected support alignment. |
| 210 | [No-reorder linked-list LDLT](SPRINT_210/artifacts/day2-owner-ranking.md) became one additional deterministic allocation-failure owner. The [25-site sweep](SPRINT_210/artifacts/day6-failure-sweep.md) and [retry proof](SPRINT_210/artifacts/day9-retry-proof.md) tested cleanup and lifecycle invariants. | This is not all-LDLT, solver-wide, concurrent-hook or OS-OOM reliability. [RQ-09](SPRINT_216/RESIDUAL_QUEUE.md) names the next owner. |
| 211 | A [selected LDLT CSC native-parity test cluster](SPRINT_211/artifacts/day3-cluster-boundary.md) moved into a family-local helper with preserved registrations and [behavior regression](SPRINT_211/artifacts/day10-behavior-regression.md), reducing `tests/test_ldlt_csc.c` by 295 lines. | Test reviewability improved without a new solver behavior claim. [RQ-10/RQ-19](SPRINT_216/RESIDUAL_QUEUE.md) cover further clusters/tooling. |
| 212 | The [benchmark policy decision](SPRINT_212/artifacts/day5-product-policy-decision.md) retained selected canonical freshness as `--repeat 1`, `baseline=n/a`, `threshold=n/a`, without warmup or variance, with stronger [manifest/report guards](SPRINT_212/artifacts/day9-manifest-and-report-guards.md). | Freshness is not a timing gate, speed comparison or platform parity. [RQ-08/RQ-12/RQ-21](SPRINT_216/RESIDUAL_QUEUE.md) hold later methodology and publication choices. |
| 213 | The [API policy](SPRINT_213/artifacts/day5-product-policy-decision.md) kept Doxygen HTML ignored and local under `docs/api/html/`; [routing](SPRINT_213/artifacts/day9-routing-and-link-validation.md) and workflow guards strengthened the non-publication boundary. | No hosted, retained-artifact or committed generated HTML route. [RQ-14/RQ-20](SPRINT_216/RESIDUAL_QUEUE.md) retain publication and first-use decisions. |
| 214 | The [ABI/release scope decision](SPRINT_214/artifacts/day2-abi-release-scope-decision.md) selected static-first deferral with policy design and a [release checklist](SPRINT_214/RELEASE_READINESS_CHECKLIST.md). [Installed static consumers](SPRINT_214/artifacts/day9-installed-consumer-evidence.md) and [negative cases](SPRINT_214/artifacts/day10-consumer-negative-matrix.md) passed locally. | No shared artifact, dynamic ABI compatibility, candidate-hosted proof or approved release. [RQ-06/RQ-13](SPRINT_216/RESIDUAL_QUEUE.md) require distinct candidate and loader gates. |
| 215 | The [external blueprint](SPRINT_215/STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md) pinned a tiny SPD/CHOLMOD cell and validation order. The [pilot outcome](SPRINT_215/artifacts/day10-selected-pilot-outcome.md) stopped with `schema_validation_unavailable` before either solver ran, leaving `timing=not_run`. | Planning JSON and provisional `attempt.json` are not accepted `evidence.json`. [RQ-01/RQ-02/RQ-03](SPRINT_216/RESIDUAL_QUEUE.md) must close in prerequisite order before even one publishable correctness cell. |
| 216 | The [Day 14 closeout](SPRINT_216/artifacts/day14-integrated-validation-closeout.md) reconciled 54 prior items, calibrated public claims, reran changed-surface local gates, and published this retrospective and the 21-entry queue. | This is an evidence/claim closeout on the branch, not hosted Windows, external solver, shared ABI, release or state-of-the-art proof. |

## Capability And Claim Assessment

| Capability | What was earned | What remains unearned |
| --- | --- | --- |
| Package distribution | Local developer-mode static source formula proof, static install contracts, and guarded provider deferral ([207 decision](SPRINT_207/artifacts/day5-provider-decision.md), [Day 11 current checks](SPRINT_216/artifacts/day11-quality-gate-and-retro-outline.md)). | Public provider availability, binaries, other managers and candidate release. |
| Windows evidence | One historically inspected selected Cholesky workflow artifact and a QR collection lane ([208 decision](SPRINT_208/artifacts/day5-promotion-decision.md), [209 decision](SPRINT_209/artifacts/day8-manifest-decision.md)). | Promoted selected Cholesky/QR Windows freshness, current QR hosted result, broad Windows parity. |
| Reliability and review surface | One 25-site LDLT failure owner and one LDLT CSC test extraction ([210 sweep](SPRINT_210/artifacts/day6-failure-sweep.md), [211 regression](SPRINT_211/artifacts/day10-behavior-regression.md)). | Broad allocation-failure coverage, repository-wide review-surface cleanup or new solver behavior. |
| Benchmark evidence | Threshold-free selected freshness and negative claim guards ([212 policy](SPRINT_212/artifacts/day5-product-policy-decision.md), [Day 10 local output](SPRINT_216/artifacts/day10-focused-validation.md)). | Timing acceptance threshold, portable speed, comparative superiority or benchmark publication. |
| API documentation | Source-controlled API route and local generated reference for 18 public headers ([213 policy](SPRINT_213/artifacts/day5-product-policy-decision.md), [Day 10 coverage](SPRINT_216/artifacts/day10-focused-validation.md)). | Published generated API HTML, public retention/rollback path or API stability guarantee. |
| ABI and release | Static-first contracts, local installed Make/CMake consumers, provisional ABI policy and release acceptance checklist ([214 checklist](SPRINT_214/RELEASE_READINESS_CHECKLIST.md), [Day 11 current checks](SPRINT_216/artifacts/day11-quality-gate-and-retro-outline.md)). | Shared export/loader proof, dynamic compatibility, tagged candidate or release readiness. |
| External baseline | A pinned method and fail-closed pilot path with current synthetic tests ([215 blocked outcome](SPRINT_215/artifacts/day10-selected-pilot-outcome.md), [Day 10 current tests](SPRINT_216/artifacts/day10-focused-validation.md)). | Accepted two-sided CHOLMOD result, independent package parity, controlled comparative timing or publication approval. |

The [Day 3 claim inventory](SPRINT_216/artifacts/day3-claim-inventory.md),
[Day 4 public/maintainer edits](SPRINT_216/artifacts/day4-user-maintainer-claims.md),
and [Day 5 benchmark/API/corpus matrix](SPRINT_216/artifacts/day5-benchmark-api-corpus-claims.md)
calibrated README, INSTALL, maintainer, benchmark, API, corpus and planning
wording to these boundaries. The [project-plan snapshot](PROJECT_PLAN.md#epic-19-current-status-snapshot)
now separates merged Sprint 215 from branch-closed Sprint 216. A documentation
correction protects an evidence boundary; it is not proof of a new feature.

## Validation And Negative Evidence

| Record | Observed result | Interpretation |
| --- | --- | --- |
| [Day 10 focused validation](SPRINT_216/artifacts/day10-focused-validation.md) | All 14 selected commands exited `0` locally on Darwin x86_64. Generated comparison bundles have 40 passing `local_only` rows using source-controlled dense helpers; NumPy and SciPy dependency rows defer. The canonical benchmark is a single-repeat, threshold-free local measurement. Pilot synthetic suite has 13 passing tests; docs coverage has 18 reference/source pages for 18 public headers. | Current-tree contract and local freshness evidence, not CHOLMOD, hosted, portable timing or state-of-the-art proof. The historical Sprint 215 nine-test count describes its earlier tree, not the current suite. |
| [Day 11 installed-static gates](SPRINT_216/artifacts/day11-quality-gate-and-retro-outline.md) | Static contract unit tests (21), deferral and package guards, selected local Homebrew boundary, Make install (26/26), and CMake install (30/30) passed locally. | Installed static-consumer evidence, not a public package provider, shared ABI or release. `tests/test_install.sh` cleaned ignored generated report files after Day 10; regenerate before later freshness use. |
| [Sprint 208 promotion decision](SPRINT_208/artifacts/day5-promotion-decision.md) and [Sprint 209 manifest decision](SPRINT_209/artifacts/day8-manifest-decision.md) | Cholesky had bounded historical hosted files but contradictory promotion metadata; QR had no inspected hosted artifact at its decision point. | These are explicit re-deferrals, not failed attempts concealed as Windows passes. Current hosted run/artifact currency was not rechecked for this branch closeout. |
| [Sprint 215 blocked pilot](SPRINT_215/artifacts/day10-selected-pilot-outcome.md) | Exit `2` with `schema_validation_unavailable`; neither project nor CHOLMOD solver ran; `timing=not_run`. | A preflight stop and provisional attempt, not accepted external correctness. The missing v1 schema/validator is a product blocker, not a passing schema test. |
| [Day 14 integrated closeout](SPRINT_216/artifacts/day14-integrated-validation-closeout.md) | Focused claim, corpus, manifest, API, comparison, canonical benchmark, pilot-synthetic and docs checks passed locally; regenerated comparison rows remain `local_only` and benchmark `threshold=n/a`. | Current-tree selected contracts passed after final edits. Hosted, external v1, public distribution and release lanes remain unrun or unavailable. |

No `.c` or `.h` file changed during Sprint 216 through Day 14, so the
conditional `make format && make lint && make test` chain was **not run** for
this branch. Earlier sprint full C or CTest passes are historical; in
particular, Sprint 211 Day 12 CMake compile/registration parity was not full
CTest, while [Sprint 214 Day 13](SPRINT_214/artifacts/day13-integrated-validation.md)
later recorded 60/60 local CTest. The [Day 2 discrepancy register](SPRINT_216/artifacts/day2-outcome-validation-reconciliation.md#discrepancy-and-gap-register)
keeps review-commit chronology, missing external validation and hosted
currency visible. The [Day 13 claim audit](SPRINT_216/artifacts/day13-retrospective-review.md)
reviews the retrospective and its non-claims. Day 14 repeated the affected
local checks and assigned item status without changing the unsupported-claim
boundaries.

## State-Of-The-Art Assessment

**Day 13 assessment: no state-of-the-art sparse linear algebra claim is
earned.** The [selected blueprint](SPRINT_215/STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md)
specifies one real-float64, natural-order, 4-by-4 SPD solve against CHOLMOD
on a named local macOS build. It is a *correctness-only* pilot specification,
not representative matrix, solver-family or platform coverage.

| Evidence dimension | Observed state and claim boundary |
| --- | --- |
| Independent baseline | The [blocked pilot](SPRINT_215/artifacts/day10-selected-pilot-outcome.md) discovered local CHOLMOD `5.3.4` from SuiteSparse `7.12.2` and a candidate dylib hash, but did not capture the actually loaded library or execute either solver. The v1 schema/validator was unavailable. There are zero validator-accepted external cells. |
| Matrix and platform | Only `tests/data/symmetric_4.mtx` was pinned for the planned cell; no two-sided matrix result was validated. The preflight was local Darwin x86_64. The historical Sprint 208 Windows Cholesky artifact belongs to a different selected freshness lane and is not hosted external-baseline evidence. No current hosted external cell is accepted. |
| Numerical and timing comparison | No project or CHOLMOD solution, residual, backward error, or matched failure case was observed. The provisional `attempt.json` has `timing=not_run`; it is not `evidence.json`. Day 10's 40 passing comparison rows use source-controlled dense helpers and remain `local_only`, while NumPy/SciPy rows defer. The canonical benchmark is one repeat with `threshold=n/a`, not controlled comparative timing. |
| Publication | Matrix/RHS attribution, CHOLMOD module/dependency license review, artifact retention, exact claim approval and rollback ownership remain open. No external result is approved for publication. |

The earliest bounded correctness statement needs [RQ-01 through RQ-03](SPRINT_216/RESIDUAL_QUEUE.md#p0-evidence-integrity)
in order: an implemented v1 validator with negative fixtures; an accepted
two-sided cell with exact fixture/conversion hashes, project build and actual
loaded-library provenance, raw outputs and numerical gates; then written
attribution, license, retention and claim approval. Even that would speak
only to the exact matrix, operation, builds, baseline version and platform.

A broader assessment needs the [queue's entry bar](SPRINT_216/RESIDUAL_QUEUE.md#state-of-the-art-decision)
and [RQ-11/RQ-12/RQ-17/RQ-18](SPRINT_216/RESIDUAL_QUEUE.md#p2-long-horizon-decisions):
pre-registered representative SPD, nonsymmetric and rectangular systems;
separate iterative/eigen/SVD cells for any named families; multiple sizes and
structures, pinned independent baselines, correctness and failure cases;
controlled paired timing with raw samples and dispersion on at least two
named hosted platform/toolchain classes, including Windows only if claimed;
per-cell provenance and publication rights; and explicit product/reviewer
approval. This is an entry bar for review, **not** automatic state-of-the-art
status. The [Day 13 audit](SPRINT_216/artifacts/day13-retrospective-review.md)
records the exact evidence gaps and reviewer questions.

## Final Non-Claims

- No public package provider, binary distribution, shared library, dynamic
  ABI guarantee, tagged release candidate or release-ready status.
- No promoted selected Windows Cholesky/QR freshness or broad Windows parity;
  no solver-wide allocation-failure guarantee or repo-wide review cleanup.
- No timing acceptance threshold, portable or comparative speed claim,
  hosted Linux/macOS timing parity, or published generated API HTML.
- No accepted independent external correctness/parity result, published
  external bundle, or state-of-the-art ranking.

## Residual Handoff And Lessons

The [Sprint 216 residual queue](SPRINT_216/RESIDUAL_QUEUE.md) retains 21
deduplicated, prioritized acceptance targets. P0 is external v1 validation,
one actual two-sided correctness cell, then attribution/publication approval.
P1 contains bounded selected Windows, static candidate/provider, benchmark,
allocation and review-surface closures. P2 carries breadth, hosted/timing,
shared ABI, API publication, broader platform/provider, numerical conformance
and performance architecture decisions. Its [source crosswalk](SPRINT_216/artifacts/day8-residual-queue-intake.md)
explains duplicate historical wording; no incomplete capability disappears
because a narrower sprint item was completed.

- **Keep acceptance objects explicit.** The [blocked pilot](SPRINT_215/artifacts/day10-selected-pilot-outcome.md)
  shows that a preflight gate can preserve truth better than emitting a
  plausible but invalid comparison row. The next pilot must validate exact
  schema, hashes, runtime libraries and numerical outputs before a claim.
- **Align metadata, artifacts and wording together.** The [Windows Cholesky](SPRINT_208/artifacts/day5-promotion-decision.md)
  and [QR](SPRINT_209/artifacts/day8-manifest-decision.md) re-deferrals show
  that a workflow or isolated passing file cannot override support-tier and
  non-claim contracts.
- **Separate local proof from distribution.** The [static package gates](SPRINT_216/artifacts/day11-quality-gate-and-retro-outline.md)
  validate installed consumers on this host, while the [release checklist](SPRINT_214/RELEASE_READINESS_CHECKLIST.md)
  still needs a candidate, provenance, hosted jobs and approval.
- **Regenerate after build-state changes.** The [Day 11 record](SPRINT_216/artifacts/day11-quality-gate-and-retro-outline.md)
  notes that `make clean` removed Day 10's ignored reports. A recorded local
  digest is a snapshot, not a retained, current artifact or release input.

This assessment closes the Epic 19 selected work on `sprint-216`, subject to
review and merge. The named residuals remain open. No retrospective wording
can promote a product capability without its independent acceptance gate.
