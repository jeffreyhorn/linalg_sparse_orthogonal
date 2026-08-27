# Sprint 184 Working Notes

## Sprint Goal

Normalize one more high-impact public header family without declaration drift,
improving API usability and generated documentation input.

## Branch Baseline

- Branch: `sprint-184`
- Starting point: current `master` after PR #203 merge.
- Sprint 183 status: complete and merged.
- Sprint 184 plan status: day-by-day plan exists at
  `docs/planning/EPIC_16/SPRINT_184/PLAN.md`.

## Planning Source

| Field | Value |
| --- | --- |
| Project plan | `docs/planning/EPIC_16/PROJECT_PLAN.md` |
| Section | `Sprint 184: Public Header Coherence Batch 3` |
| Sprint duration | 14 days, approximately 168 hours |
| Selected residual | `S177-R09`: Public-header coherence breadth |
| Evidence matrix rows | `ESM-005`, `ESM-011` |

## Sprint 184 Item Boundaries

| Item | Name | Sprint 184 interpretation |
| --- | --- | --- |
| 184.1 | Header Family Selection | Confirm one selected family, likely QR/SVD or LDLT, and capture declaration baselines before edits. |
| 184.2 | Contract Cleanup | Normalize lifecycle, ownership, error-code, tolerance, workspace, option/result, and cancellation wording. |
| 184.3 | Declaration Organization | Reorder declarations only if declaration-preserving guardrails make the change reviewable. |
| 184.4 | Example and Docs Alignment | Align examples, tutorial, solver-selection, cookbook, or API reference for the selected family only. |
| 184.5 | Mechanical Guard | Add or extend declaration checksum, docs coverage, or unsupported-claim guard for the selected family. |
| 184.6 | Validation | Run the full C quality gate if `.c` or `.h` files change, plus docs and declaration guard checks. |

## Prior Evidence Carried Forward

| Input | Source | Sprint 184 use |
| --- | --- | --- |
| Epic 16 target selection | `docs/planning/EPIC_16/SPRINT_177/artifacts/day7-target-selection.md` | Confirms Sprint 184 should close `S177-R09` by selecting one header family, preserving declarations, and aligning docs. |
| Epic 16 residual audit | `docs/planning/EPIC_16/SPRINT_177/artifacts/day2-residual-audit.md` | Names QR, SVD, LDLT, IC, ILU, reorder, and analysis as remaining uneven public-header surfaces. |
| Prior header cleanup batch | `docs/planning/EPIC_15/SPRINT_172/WORKING_NOTES.md` and `docs/planning/EPIC_15/SPRINT_172/RETROSPECTIVE.md` | Provides the nearest declaration-preserving workflow: select one family, capture baselines, update docs, add a focused guard, and run the full C gate after header edits. |
| Earlier header selection criteria | `docs/planning/EPIC_14/SPRINT_164/artifacts/day2-header-selection.md` | Provides risk criteria: user impact, documentation ambiguity, claim risk, option/result complexity, downstream visibility, and feasibility of declaration-preserving cleanup. |
| Generated API coverage checks | `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh` | Existing API-docs guard surface that may constrain docs changes for the selected family. |
| Existing LU header guard precedent | `scripts/check_lu_header_docs_guard.sh` | Local model for a family-specific declaration/docs drift guard if QR, SVD, or LDLT needs similar protection. |

## Day 1 Candidate Family Inventory

Day 1 does not select the Sprint 184 family. It records the initial candidate
set called out by the project plan and Sprint 177 evidence so Day 2 can capture
declaration baselines before any cleanup.

| Candidate family | Public header | Header lines | User/docs visibility | Evidence and test surface | Day 1 disposition |
| --- | --- | ---: | --- | --- | --- |
| QR | `include/sparse_qr.h` | 373 | README API overview, tutorial QR section, solver-selection QR boundary, cookbook routing, examples README, API reference, least-squares/minimum-norm/COLAMD examples. | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_qr_corpus.c`, `tests/qr_external_dense_reference.py`, selected QR oracle/comparison freshness. | Strong candidate. High adoption visibility and sensitive rank/nullspace/minimum-norm claim boundaries. |
| SVD / partial SVD | `include/sparse_svd.h` | 243 | README SVD section, tutorial full/partial SVD sections, cookbook SVD workflows, solver-selection SVD boundary, examples README, API reference, SVD low-rank and condition examples. | `tests/test_svd.c`, `tests/test_svd_partial_corpus.c`, `tests/svd_external_dense_reference.py`, selected partial-SVD oracle/comparison freshness. | Strong candidate. Lower header size than QR but high evidence-sensitivity around rank, convergence, vectors, and low-rank output. |
| LDLT | `include/sparse_ldlt.h` | 315 | README direct-solver overview, tutorial direct-solver routing, solver-selection symmetric-indefinite routing, examples README, API reference, LDLT example. | `tests/test_ldlt.c`, `tests/test_ldlt_csc.c`, `tests/test_ldlt_backend_dispatch.c`, `tests/ldlt_external_dense_reference.py`, Sprint 183 deferred LDLT KKT comparison candidate. | Strong candidate. Direct-solver lifecycle and backend semantics are valuable, with symmetric-indefinite and inertia wording risks. |

## Related Documentation Surfaces

| Surface | QR | SVD / partial SVD | LDLT |
| --- | --- | --- | --- |
| `README.md` | API overview, one-shot solve guidance, QR public API list, selected QR evidence boundary. | SVD capability section, SVD public API list, partial-SVD fixture-local evidence boundary. | Direct-solver overview, LDLT public API list, backend and KKT report context. |
| `docs/api_reference.md` | `sparse_qr.h` summary row. | `sparse_svd.h` summary row. | `sparse_ldlt.h` summary row. |
| `docs/tutorial.md` | QR factorization walkthrough and least-squares/minimum-norm routing. | Full SVD, partial SVD, rank, condition, low-rank walkthroughs. | Direct-solver selection table and include overview. |
| `docs/cookbook.md` | Rectangular/rank-sensitive routing and QR evidence boundary wording. | SVD and low-rank workflow section. | Minimal direct-solver routing and benchmark context references. |
| `docs/solver_selection.md` | QR routing, QR evidence boundary, selected comparison non-claim wording. | SVD/low-rank workflows and partial-SVD evidence boundary. | Symmetric-indefinite routing and direct-solver selection table. |
| `examples/` | `example_least_squares.c`, `example_minnorm.c`, `example_colamd.c`, examples README. | `example_svd_lowrank.c`, `example_condition.c`, examples README. | `example_ldlt.c`, examples README. |

## Inherited Guardrails

- Select exactly one public header family for Sprint 184.
- Capture declaration baselines before public header edits.
- Keep header work declaration-preserving unless an artifact explicitly records
  an intentional organization-only change and its guard coverage.
- Do not change struct layout, enum values, typedef names, macro definitions,
  function signatures, include guards, or installed header names as part of
  comment cleanup.
- Keep header comments API-local. Use README, tutorial, cookbook,
  solver-selection, and maintainer docs for broader workflow guidance.
- Do not treat comment cleanup as evidence for dynamic ABI stability,
  shared-library support, runtime-loader behavior, package-manager support,
  broad platform parity, portable performance superiority, broad
  external-library parity, or state-of-the-art sparse linear algebra status.
- If `.c` or `.h` files change, run `make format && make lint && make test`.
- If only planning files change, `git diff --check` is sufficient for that
  day.

## Initial Risks And Open Questions

| ID | Topic | Risk or question | Day 1 disposition |
| --- | --- | --- | --- |
| S184-RISK-01 | QR evidence wording | QR has high doc visibility and selected comparison/oracle wording that can accidentally widen into broad QR or external-library parity. | Keep QR evidence boundary and non-claim scans in Day 2-3 selection criteria. |
| S184-RISK-02 | SVD vector semantics | SVD and partial-SVD comments can overpromise raw vector identity, convergence, sparse-output optimality, or broad partial-SVD correctness. | Require claim-boundary review if SVD is selected. |
| S184-RISK-03 | LDLT backend semantics | LDLT docs touch backend dispatch, inertia, KKT-style examples, and symmetric-indefinite semantics. | Require backend/performance non-claim review if LDLT is selected. |
| S184-RISK-04 | Declaration organization | Reordering declarations can be useful but raises review risk if baseline checks are weak. | Defer organization changes until guardrail design is recorded. |
| S184-RISK-05 | Generated docs | Header comments may feed generated API docs, but generated HTML remains local-only unless a separate publication decision changes that. | Keep generated API publication out of Sprint 184 scope. |

## Daily Log

### Day 1: Sprint Intake and Prior-Art Review

- Re-read the Sprint 184 section of
  `docs/planning/EPIC_16/PROJECT_PLAN.md`.
- Confirmed Sprint 184 closes Sprint 177 residual `S177-R09` for
  public-header coherence breadth.
- Reviewed Sprint 177 Day 2 residual audit and Day 7 target selection for the
  Sprint 184 handoff.
- Reviewed Sprint 172 working notes and retrospective as the nearest
  declaration-preserving public-header cleanup precedent.
- Reviewed Sprint 164 header selection criteria for user impact, ambiguity,
  claim risk, option/result complexity, downstream visibility, and cleanup
  feasibility.
- Inventoried initial QR, SVD, and LDLT candidate headers, docs, examples, and
  test surfaces without selecting a family.
- Recorded inherited guardrails, risks, and Day 2 baseline-capture handoff.
- Day 1 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day1-sprint-intake.md`.

### Day 2: Declaration Baseline Capture

- Captured declaration-order baselines for QR, SVD, and LDLT candidate public
  headers before selecting or editing a family.
- Recorded declaration-list checksums for the candidate headers:
  - `include/sparse_qr.h`: `77a53e6bc780d79907bad9a040310bb0d63f93dce3fdd3beb0ed8cfdfd0279bc`
  - `include/sparse_svd.h`: `51d334c7cc7681a3b53f0af3e5a3d0bdf4d890e0734fa4da1b424c54604c3025`
  - `include/sparse_ldlt.h`: `b99ed791daeb2e9a6d411cb0bccad486a897aa42b9c130f0f45eb58d0cf547a7`
  - combined QR/SVD/LDLT baseline:
    `765b4711e1a62006566b1a0a7f6187401b958753fbae4cb902f540c6e98ed45e`
- Inventoried current public type/function declaration starts and section
  grouping for each candidate.
- Identified existing guard surfaces: `make docs-check`,
  `make api-docs-coverage`, `make api-docs-local-only`,
  `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh`,
  and LU-specific precedent `scripts/check_lu_header_docs_guard.sh`.
- Defined Sprint 184 declaration-preservation rules: comments may change,
  declarations must remain stable unless an organization-only exception is
  recorded with before/after evidence.
- Narrowed the candidate list to a provisional order:
  1. QR: strongest Day 2 front-runner.
  2. SVD / partial SVD: close second.
  3. LDLT: viable but lower Day 2 priority than QR/SVD for this sprint.
- Day 2 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day2-declaration-baseline.md`.

### Day 3: Family Selection Decision and Contract Map

- Selected exactly one Sprint 184 public header family:
  `include/sparse_qr.h`.
- Rejected SVD/partial SVD and LDLT for Sprint 184 while preserving them as
  future public-header cleanup candidates.
- Recorded selected QR declaration baseline:
  `77a53e6bc780d79907bad9a040310bb0d63f93dce3fdd3beb0ed8cfdfd0279bc`.
- Mapped QR lifecycle, ownership/output, error-code, tolerance, workspace,
  option/result, and cancellation wording across `sparse_qr_opts_t`,
  `sparse_qr_t`, factorization, apply/form-Q, solve/refine, rank/nullspace,
  diagnostics, minimum-norm solve, and free declarations.
- Audited QR docs/example surfaces in README, API reference, tutorial,
  cookbook, solver-selection, examples README, `example_least_squares.c`,
  `example_minnorm.c`, and `example_colamd.c`.
- Identified the first cleanup checklist for Days 4-5:
  - clarify factor output lifecycle and overwrite/free expectations;
  - normalize `opts == NULL`, default tolerance, and caller-owned output
    wording;
  - make `sparse_qr_factor_opts()` error-code wording match
    `sparse_qr_factor()`;
  - clarify dense workspace and output sizes for `apply_q`, `form_q`,
    nullspace, and minimum-norm routines;
  - tighten rank-threshold wording without claiming a global rank policy;
  - preserve QR evidence boundaries as fixture-local documentation, not header
    correctness or external-parity claims.
- Day 3 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day3-family-selection-and-contract-map.md`.

### Day 4: Lifecycle, Ownership, and Error Contracts

- Updated `include/sparse_qr.h` comments for the first QR contract cleanup
  pass.
- Kept the Day 4 edit comment-only and declaration-preserving.
- Clarified file-level scope: QR header owns API-local contracts while
  runnable workflow and evidence boundaries stay in examples/docs.
- Clarified `sparse_qr_t` lifecycle and ownership: callers own the struct,
  successful factorization stores owned data inside it, and populated objects
  must be freed before reuse.
- Normalized borrowed-input and caller-owned-output wording for
  `sparse_qr_factor()`, `sparse_qr_factor_opts()`, `sparse_qr_apply_q()`,
  `sparse_qr_form_q()`, `sparse_qr_solve()`, and `sparse_qr_refine()`.
- Aligned `sparse_qr_factor_opts()` error-code documentation with
  implementation behavior for NULL, non-identity permutation, allocation, and
  cancellation errors.
- Verified return-code wording against `src/sparse_qr.c` before recording it.
- Focused declaration-preservation evidence:
  - comment-stripped QR declaration hash before edit:
    `5d20a4cf0cefb813c8eabc3d531d6ba31429f5023968c5d6f56c2506456d9a67`
  - comment-stripped QR declaration hash after edit:
    `5d20a4cf0cefb813c8eabc3d531d6ba31429f5023968c5d6f56c2506456d9a67`
  - comment-stripped QR declaration diff: no output
- The Day 2 line-numbered declaration checksum changed because comment edits
  shifted declaration line numbers; Day 4 uses the comment-stripped declaration
  comparison as the preservation check.
- Validation passed:
  - `make format && make lint && make test`
  - `git diff --check`
- Created `artifacts/day4-core-contract-cleanup.md`.

### Day 5: Tolerance, Workspace, Options, and Cancellation Contracts

- Updated `include/sparse_qr.h` comments for the second QR contract cleanup
  pass.
- Kept the Day 5 edit comment-only and declaration-preserving.
- Tightened `sparse_mode` wording to describe the lower-workspace path without
  claiming a generic performance tradeoff.
- Normalized rank tolerance wording for `sparse_qr_rank()` and
  `sparse_qr_rank_info()`:
  - `tol > 0` maps to `tol * |R(0,0)|`;
  - `tol <= 0` maps to `eps * max(m,n) * |R(0,0)|`;
  - the result is a QR-local rank estimate, not a global rank policy.
- Clarified null-space basis ownership and sizing:
  - `basis` is optional;
  - when provided and the null dimension is positive, callers allocate
    `n * (n - rank)` dense scalars;
  - `null_dim` is caller-owned and required.
- Clarified diagnostic outputs for R-diagonal extraction, rank-info, and
  condition estimation without promising a full condition-number guarantee.
- Clarified minimum-norm solve/refine option behavior: these routines build
  temporary QR factorizations internally, apply `opts` to those internal
  factorizations, and can return `SPARSE_ERR_CANCELLED` when the progress
  callback cancels.
- Cross-checked the updated wording against `src/sparse_qr.c` implementations
  for rank, rank-info, condest, nullspace, minimum-norm solve/refine, and
  QR progress cancellation.
- Focused declaration-preservation evidence:
  - comment-stripped QR declaration hash before edit:
    `e1ec290dca650506021b144f03039a2ab528e91602cfc8f8d4c2821b9d6de6c0`
  - comment-stripped QR declaration hash after edit:
    `e1ec290dca650506021b144f03039a2ab528e91602cfc8f8d4c2821b9d6de6c0`
  - comment-stripped QR declaration diff: no output
- Validation passed:
  - `make format && make lint && make test`
  - `git diff --check`
- Created `artifacts/day5-advanced-contract-cleanup.md`.

### Day 6: Declaration Organization Design

- Reviewed current `include/sparse_qr.h` declaration grouping after the Day 4
  and Day 5 comment cleanup passes.
- Decided not to reorder declarations on Day 6. This day records the allowed
  organization proposal and guardrails so Day 7 can make any ordering change
  intentionally.
- Current QR order:
  1. options struct;
  2. factor object struct;
  3. factorization;
  4. Q operations;
  5. standard solve/refine;
  6. rank/nullspace/free;
  7. rank-revealing diagnostics;
  8. minimum-norm solve/refine.
- Identified two organization candidates for Day 7:
  - move `sparse_qr_free()` into an explicit lifecycle section near
    factorization/factor object declarations;
  - move minimum-norm solve/refine beside standard solve/refine under a solve
    operations section.
- Deferred any larger reordering as unnecessary for Sprint 184 because it would
  increase review noise without improving API meaning.
- Designed the Day 7 guard policy:
  - preserve the complete public declaration set exactly;
  - record before/after declaration order if any reorder happens;
  - require a section-heading presence check if QR headings are added;
  - keep the comment-stripped QR declaration hash available as the no-drift
    check when comments only change.
- Current comment-stripped QR declaration hash:
  `e1ec290dca650506021b144f03039a2ab528e91602cfc8f8d4c2821b9d6de6c0`.
- Day 6 changed planning artifacts only. No new `.c` or `.h` edits were made
  for this day, so the full C quality gate was not rerun.
- Validation passed:
  - `git diff --check`
  - focused comment-stripped QR declaration diff against `HEAD`: no output
- Created `artifacts/day6-organization-guardrail-design.md`.

### Day 7: Coherent Header Sections

- Applied the Day 6-approved organization pass to `include/sparse_qr.h`.
- Added lightweight QR section headings:
  - `Options and factor object`;
  - `Factorization and lifecycle`;
  - `Q operations`;
  - `Solve operations`;
  - `Rank, nullspace, and diagnostics`.
- Moved `sparse_qr_free()` next to the factorization declarations under the
  lifecycle section.
- Moved `sparse_qr_solve_minnorm()` and `sparse_qr_refine_minnorm()` next to
  standard solve/refine declarations under the solve operations section.
- Preserved all public declarations and signatures; only declaration order and
  comments/section headings changed.
- Recorded the post-organization QR declaration order:
  1. `sparse_qr_opts_t`;
  2. `sparse_qr_t`;
  3. `sparse_qr_factor()`;
  4. `sparse_qr_factor_opts()`;
  5. `sparse_qr_free()`;
  6. `sparse_qr_apply_q()`;
  7. `sparse_qr_form_q()`;
  8. `sparse_qr_solve()`;
  9. `sparse_qr_refine()`;
  10. `sparse_qr_solve_minnorm()`;
  11. `sparse_qr_refine_minnorm()`;
  12. `sparse_qr_rank()`;
  13. `sparse_qr_nullspace()`;
  14. `sparse_qr_diag_r()`;
  15. `sparse_qr_rank_info_t`;
  16. `sparse_qr_rank_info()`;
  17. `sparse_qr_condest()`.
- Guard evidence:
  - ordered comment-stripped QR declaration hash after organization:
    `5650cb782761cdbaa18c75b29b477f7957a1893f80d85e3114c2158cbf7b1734`;
  - sorted comment-stripped QR declaration-set hash after organization:
    `d50272d2e12f03f0869c8514809359e2d76ab585bb35ec5a2a936cb348432ec3`;
  - sorted comment-stripped QR declaration-set diff against `HEAD`: no output.
- Validation passed:
  - `make format && make lint && make test`
  - `make docs-check`
  - `git diff --check`
- Created `artifacts/day7-coherent-header-sections.md`.

### Day 8: Documentation Alignment Map

- Audited the known downstream QR documentation and example surfaces against
  the cleaned and reorganized `include/sparse_qr.h` contracts.
- Reviewed surfaces:
  - `README.md` QR public API bullets and QR evidence block;
  - `docs/api_reference.md` public header summary row;
  - `docs/tutorial.md` QR factorization snippet and diagnostics handoff row;
  - `docs/cookbook.md` solver routing and QR evidence note;
  - `docs/solver_selection.md` QR routing row, examples list, COLAMD guidance,
    and QR evidence boundary;
  - `examples/README.md` QR example descriptions;
  - `examples/example_least_squares.c`;
  - `examples/example_minnorm.c`;
  - `examples/example_colamd.c`.
- Found no required source-code edits for Day 8.
- Captured prioritized Day 9 documentation alignment checklist:
  1. update README QR API bullets so `sparse_qr_factor_opts()` names COLAMD as
     the recommended unsymmetric/QR column-ordering path instead of implying
     AMD is the primary option;
  2. align `docs/solver_selection.md#qr-evidence-boundary` with the newer
     README/cookbook evidence scope for selected minimum-norm and compatible
     least-squares comparison rows while preserving the "not broad parity"
     boundary;
  3. tighten the tutorial QR snippet with return-code checking and caller-owned
     output/free wording that mirrors the header;
  4. add a short minimum-norm options/cancellation note in the example-facing
     docs if Day 9 updates the QR example text;
  5. check `example_colamd` for solve/rank-info return-code handling if Day 9
     touches executable examples.
- Deferred broader docs rewrites, generated API publication wording, SVD/LDLT
  docs, and any new evidence claims as out of Sprint 184 Day 8 scope.
- Day 8 changed planning artifacts only. No new `.c` or `.h` edits were made
  for this day, so the full C quality gate was not rerun.
- Validation passed:
  - `git diff --check`
- Created `artifacts/day8-documentation-alignment-map.md`.

### Day 9: Example Contract Alignment

- Updated QR-facing documentation and example narrative to match the cleaned
  `include/sparse_qr.h` lifecycle, ownership, option, and evidence wording.
- Changed `README.md` QR API bullets so `sparse_qr_factor_opts()` names COLAMD
  column reordering for unsymmetric/QR workflows instead of implying AMD is the
  primary QR reordering path.
- Updated `docs/tutorial.md` QR snippet to:
  - check `sparse_qr_factor()` return status;
  - check `sparse_qr_solve()` return status;
  - free QR factor data on solve failure;
  - describe `x` and `residual_norm` as caller-owned outputs;
  - call out `sparse_qr_free()` as releasing factor data stored inside the
    caller-owned QR object;
  - note that minimum-norm solve/refine options apply to temporary internal QR
    factorizations.
- Updated `docs/solver_selection.md#qr-evidence-boundary` so it includes the
  selected QR minimum-norm and compatible least-squares comparison rows while
  preserving the fixture-local, not-broad-parity boundary.
- Updated `examples/README.md` minimum-norm example text to note that options
  apply to internal QR factorizations, including progress cancellation.
- Did not edit executable examples on Day 9. The Day 8 audit found no required
  example-code contradiction, and keeping this pass Markdown-only avoids
  widening the sprint scope.
- Validation passed:
  - `make docs-check`
  - `git diff --check`
  - sorted comment-stripped QR declaration-set diff against `HEAD`: no output
- Day 9 made no new `.c` or `.h` edits, so the full C quality gate was not
  rerun for this day.
- Created `artifacts/day9-example-contract-alignment.md`.

### Day 10: Reference Documentation Alignment

- Updated higher-level QR reference documentation to agree with the cleaned
  `include/sparse_qr.h` header and Day 9 tutorial/example text.
- Updated `docs/api_reference.md` so the `sparse_qr.h` row names
  factorization/lifecycle, least-squares, minimum-norm, rank/nullspace,
  R-diagonal diagnostics, and cancellation contracts.
- Updated `docs/cookbook.md` QR routing note so selected comparison freshness
  covers selected QR minimum-norm and compatible least-squares rows from
  `tests/corpus/manifests/selected_report_targets.tsv`, while preserving the
  fixture-local and not-broad-parity boundary.
- Updated `docs/solver_selection.md` diagnostics handoff so QR diagnostics
  include minimum-norm output and R-diagonal diagnostics in addition to rank,
  residual, nullity, and nullspace output.
- Rechecked unsupported-claim boundaries: no broad QR parity, global
  rank-threshold policy, broad minimum-norm behavior, external-library parity,
  platform/package/ABI, performance, Windows freshness, or state-of-the-art
  claims were added.
- Validation passed:
  - `make docs-check`
  - `make api-docs-local-only`
  - `git diff --check`
  - sorted comment-stripped QR declaration-set diff against `HEAD`: no output
- Day 10 made no new `.c` or `.h` edits, so the full C quality gate was not
  rerun for this day.
- Created `artifacts/day10-reference-documentation-alignment.md`.

### Day 11: Declaration and Claim Guards

- Implemented `scripts/check_qr_header_docs_guard.sh` as the focused Sprint
  184 QR header/docs drift guard.
- Added a `qr-header-docs-guard` Makefile target that runs the guard with
  `bash scripts/check_qr_header_docs_guard.sh`.
- Guard coverage:
  - required QR header section headings from Day 7;
  - required QR public declaration tokens;
  - unsupported claim absence in `include/sparse_qr.h`;
  - README QR `sparse_qr_factor_opts()` COLAMD wording;
  - API reference QR lifecycle/diagnostics/cancellation row;
  - cookbook selected QR comparison evidence wording;
  - solver-selection QR diagnostics and evidence-boundary wording;
  - tutorial QR factor/solve return-code and caller-owned output wording;
  - examples README minimum-norm internal-factorization options note.
- Focused guard output:
  - `qr-header-docs-guard: header sections ok`
  - `qr-header-docs-guard: header declarations ok`
  - `qr-header-docs-guard: header unsupported claim absence ok`
  - `qr-header-docs-guard: docs alignment ok`
  - `qr-header-docs-guard: passed`
- Validation passed:
  - `bash -n scripts/check_qr_header_docs_guard.sh`
  - `make qr-header-docs-guard`
  - `make docs-check`
  - `make api-docs-local-only`
  - `git diff --check`
  - sorted comment-stripped QR declaration-set diff against `HEAD`: no output
- Day 11 made no new `.c` or `.h` edits, so the full C quality gate was not
  rerun for this day.
- Created `artifacts/day11-mechanical-guard-implementation.md`.

### Day 12: Focused Validation Pass

- Ran the focused validation bundle for changed QR header, docs, examples, and
  guard surfaces before the Day 13 full quality gate.
- Validation passed:
  - `bash -n scripts/check_qr_header_docs_guard.sh`
  - `git diff --check`
  - sorted comment-stripped QR declaration-set diff against `HEAD`: no output
  - `make qr-header-docs-guard`
  - `make api-docs-validate`
  - `make format-check`
  - `make examples-build`
  - `./build/example_least_squares`
  - `./build/example_minnorm`
  - `./build/example_colamd`
- Focused QR guard output remained:
  - `qr-header-docs-guard: header sections ok`
  - `qr-header-docs-guard: header declarations ok`
  - `qr-header-docs-guard: header unsupported claim absence ok`
  - `qr-header-docs-guard: docs alignment ok`
  - `qr-header-docs-guard: passed`
- QR example smoke outputs confirmed:
  - least-squares example factors with rank 3 and reports residual norm
    `0.1897`;
  - minimum-norm example verifies `A*x = b` and refinement residual
    `0.00e+00`;
  - COLAMD example reports QR+COLAMD residual `0.00e+00` and rank `10/10`.
- No focused validation failures were found, so no Day 12 fixes were needed.
- Day 12 made planning artifact updates only. No new `.c` or `.h` edits were
  made for this day.
- Day 13 full validation command list:
  1. `make format`
  2. `make lint`
  3. `make test`
  4. `make qr-header-docs-guard`
  5. `make api-docs-validate`
  6. `git diff --check`
  7. sorted comment-stripped QR declaration-set diff against `HEAD`
- Created `artifacts/day12-focused-validation-pass.md`.

### Day 13: Full Validation and Final Cleanup

- Ran the full Sprint 184 quality gate because `include/sparse_qr.h` changed
  earlier in the sprint.
- Validation passed:
  - `make format && make lint && make test`
  - `make qr-header-docs-guard`
  - `make api-docs-validate`
  - `git diff --check`
  - sorted comment-stripped QR declaration-set diff against `HEAD`: no output
- The QR guard continued to report:
  - `qr-header-docs-guard: header sections ok`
  - `qr-header-docs-guard: header declarations ok`
  - `qr-header-docs-guard: header unsupported claim absence ok`
  - `qr-header-docs-guard: docs alignment ok`
  - `qr-header-docs-guard: passed`
- `make api-docs-validate` passed API coverage and local-only checks, and
  reported no tracked, staged, or non-ignored generated API files.
- The full test suite completed with `All tests passed.`
- No validation failures were found, so no Day 13 source fixes were required.
- Day 14 handoff:
  1. confirm items 184.1 through 184.6 are traceable to artifacts and notes;
  2. review the final diff for accidental scope creep outside QR;
  3. recheck open risks and unsupported-claim boundaries;
  4. prepare retrospective-ready evidence and final handoff notes.
- Created `artifacts/day13-full-validation-and-final-cleanup.md`.

### Day 14: Retrospective-Ready Handoff

- Reviewed the final Sprint 184 diff against project-plan items 184.1 through
  184.6.
- Confirmed item outcomes:
  - 184.1 Header Family Selection: completed with QR selected and QR/SVD/LDLT
    declaration baselines captured.
  - 184.2 Contract Cleanup: completed for QR lifecycle, ownership, error-code,
    tolerance, workspace, option/result, cancellation, diagnostics, and
    minimum-norm wording.
  - 184.3 Declaration Organization: completed with bounded organization-only
    movement and declaration-set preservation evidence.
  - 184.4 Example and Docs Alignment: completed for README, tutorial,
    solver-selection, cookbook, API reference, and examples README QR wording.
  - 184.5 Mechanical Guard: completed with
    `scripts/check_qr_header_docs_guard.sh` and `make qr-header-docs-guard`.
  - 184.6 Validation: completed with Day 12 focused validation and Day 13 full
    quality gate.
- Confirmed final diff scope remains limited to Sprint 184 planning artifacts,
  QR public header comment/organization cleanup, QR-facing docs/example
  narrative alignment, and the QR header/docs guard script plus Make target.
- Rechecked stale markers across Sprint 184 artifacts and touched QR surfaces.
  No active stale TODO/FIXME/TBD markers were found; matches were planned
  Day 14 text, historical/project-level deferrals, or explicit Sprint 184
  deferred-scope notes.
- Day 14 validation passed:
  - `make qr-header-docs-guard`
  - `git diff --check`
  - sorted comment-stripped QR declaration-set diff against `HEAD`: no output
- Deferred work is explicitly separated from completed Sprint 184 scope:
  - SVD/partial SVD public-header coherence remains a future candidate;
  - LDLT public-header coherence remains a future candidate;
  - broader generated API publication, package/platform/ABI, Windows
    freshness, and broad external-comparison claims remain out of scope.
- Sprint 184 is ready for retrospective creation and PR preparation.
- Created `artifacts/day14-retrospective-ready-handoff.md`.
