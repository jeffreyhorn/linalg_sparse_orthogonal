# Sprint 176 Working Notes

## Sprint Goal

Add one targeted allocation-failure proof, reconcile all Epic 15 claims, and
close the epic with evidence-bound documentation.

## Source Artifact Note

The Sprint 176 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`,
but the active merged Sprint 176 planning source is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section
"Sprint 176: Allocation-Failure Evidence, Claim Recalibration & Epic
Closeout".

## Branch Baseline

- Branch: `sprint-176`
- Starting point: current `master` after PR #194 merge.
- Sprint 176 plan status: day-by-day plan exists at
  `docs/planning/EPIC_15/SPRINT_176/PLAN.md`.
- Sprint 176 Day 1 status: intake, source note, artifact layout, inherited
  evidence categories, stop conditions, and retained non-claims.

## Prior Evidence Carried Forward

| Input | Source | Sprint 176 use |
| --- | --- | --- |
| Epic 15 baseline ledger | `docs/planning/EPIC_15/SPRINT_167/` | Use the evidence ledger and non-claim register as the final closeout baseline. |
| Hosted selected performance lane | `docs/planning/EPIC_15/SPRINT_168/` | Carry forward the selected `bench_refactor_csc` hosted threshold-free performance publication boundary. |
| Performance methodology hardening | `docs/planning/EPIC_15/SPRINT_169/` | Preserve repeat, warmup, variance, threshold, runner, and methodology semantics during final claim recalibration. |
| Shared-library ABI decision | `docs/planning/EPIC_15/SPRINT_170/` | Retain static-first-only support and shared-library/dynamic ABI non-claims. |
| Package-manager deferral | `docs/planning/EPIC_15/SPRINT_171/` | Retain formal provider deferral and package-manager non-claim guard requirements. |
| Public LU header coherence | `docs/planning/EPIC_15/SPRINT_172/` | Include LU public-header cleanup and tutorial signature repair in final Epic 15 claim inventory. |
| Generated API HTML decision | `docs/planning/EPIC_15/SPRINT_173/` | Keep generated API HTML guarded local-only, not hosted or committed publication. |
| Bounded LU comparison family | `docs/planning/EPIC_15/SPRINT_174/` | Carry forward fixture-local LU comparison freshness without broad LU or external-library parity claims. |
| Cross-platform report freshness | `docs/planning/EPIC_15/SPRINT_175/` | Carry forward Linux/macOS selected comparison hosted artifact evidence and workflow guards. |

## Closeout Categories

| Category | Day 1 posture | Sprint 176 responsibility |
| --- | --- | --- |
| Allocation failure | Deferred broadly; no selected deterministic proof yet. | Select one allocation-heavy subsystem and add deterministic failure-path evidence. |
| Cleanup invariants | Partially documented through header/API comments and functional tests. | Document selected-subsystem ownership and cleanup invariants supported by the new proof. |
| Claim recalibration | Many scoped claims and non-claims exist across README, INSTALL, maintainer docs, corpus docs, benchmark docs, and planning artifacts. | Reconcile public wording to the final Epic 15 evidence set. |
| Evidence ledger | Sprint 167 created the baseline; Sprints 168-175 added new evidence. | Update final ledger posture for completed, narrowed, deferred, and retained non-claim surfaces. |
| Documentation | Public docs contain selected performance, package, API, comparison, and report freshness boundaries. | Ensure final docs do not imply broad performance, platform, package-manager, shared-library, external-parity, allocation-failure, or state-of-the-art support. |
| Validation | Prior sprints ran focused checks matching each changed surface. | Run the full required gate if C/header files change and focused guards for docs, package, report, and claim surfaces. |
| Residual queue | Prior retrospectives list deferred provider, ABI, hosted docs, platform, and broad claim work. | Produce final residual queue and next-epic handoff without hiding deferred product decisions. |

## Day 2 Allocation Inventory Summary

Day 2 inventory found no generic deterministic allocation-failure injection
framework. The existing shared helpers in `src/sparse_alloc_internal.c` and
`src/sparse_alloc_internal.h` provide overflow-aware `malloc`/`calloc`
wrappers, but they currently delegate directly to the system allocator.

Highest allocation/free mention counts in `src/*.c`:

| Rank | File | Mentions | Day 2 posture |
| ---: | --- | ---: | --- |
| 1 | `src/sparse_lu_csr.c` | 128 | High-value but high-blast-radius direct solver candidate. |
| 2 | `src/sparse_ldlt_csc.c` | 125 | High-value but broad CSC factorization cleanup surface. |
| 3 | `src/sparse_ldlt.c` | 114 | Public LDLT orchestration and backend ownership candidate. |
| 4 | `src/sparse_qr.c` | 101 | Active comparison/user workflow candidate with multiple allocation paths. |
| 5 | `src/sparse_lu.c` | 92 | Core linked-list LU candidate with public lifecycle relevance. |
| 6 | `src/sparse_etree.c` | 84 | Symbolic analysis/tree cleanup candidate. |
| 7 | `src/sparse_svd_partial.c` | 66 | Active Epic 15 partial-SVD candidate with bounded output ownership. |

Most practical Day 3 target classes:

- wrapper-backed shared workspace proof: iterative or eigensolver workspace
  reserve/free behavior;
- active solver proof with bounded output cleanup: partial SVD.

Day 2 intentionally does not select a subsystem. LU CSR, LDLT CSC, QR, and
matrix core remain important but riskier first targets because they either use
many direct allocator calls or touch broader solver behavior.

## Day 3 Selected Subsystem

Sprint 176 selects the iterative repeated-run workspace owner for deterministic
allocation-failure proof.

Selected scope:

- public handle lifecycle APIs in `include/sparse_iterative.h`;
- handle setup/free implementation in `src/sparse_iterative.c`;
- internal workspace owner and typed view helpers in
  `src/sparse_iterative_workspace_internal.c` and
  `src/sparse_iterative_workspace_internal.h`;
- focused tests for deterministic allocation failure during repeated-run
  handle preparation.

Targeted failure points for later design:

| Target | Expected behavior |
| --- | --- |
| Empty handle owner allocation failure | prepare returns `SPARSE_ERR_ALLOC`, handle remains zeroed, free remains safe. |
| CG workspace growth failure | prepare returns `SPARSE_ERR_ALLOC`; existing handle ownership remains cleanup-safe. |
| GMRES workspace growth failure | failed larger prepare does not discard previous smaller reusable capacity. |
| MINRES workspace growth failure | prepare returns `SPARSE_ERR_ALLOC`; handle remains cleanup-safe. |

Claim boundary if implementation succeeds:

> The iterative repeated-run workspace handle has deterministic
> allocation-failure cleanup evidence for selected prepare paths.

This does not claim broad allocation-failure cleanup across all iterative
solvers, one-shot iterative calls, direct solvers, QR, SVD, LDLT, LU CSR,
matrix core, graph paths, external libraries, package/install behavior, or
state-of-the-art memory reliability.

## Day 4 Failure Harness Design

Day 4 designs the deterministic failure harness for the selected iterative
repeated-run workspace owner.

Harness owner:

- primary test target: `tests/test_iterative.c`;
- existing repeated-run helper file: `tests/test_iterative_handle_helpers.h`;
- allocation hook location: private `src/sparse_alloc_internal.h` and
  `src/sparse_alloc_internal.c`;
- no installed public header should expose allocator-test controls.

Recommended private hook:

| Hook | Purpose |
| --- | --- |
| `sparse_alloc_test_fail_after(long remaining)` | Fail the next helper allocation when `remaining == 0`; otherwise fail after that many successful helper allocations. |
| `sparse_alloc_test_reset(void)` | Disable injection and reset hook state before/after each test. |

Minimum Day 5 proof:

- owner allocation failure keeps a zeroed public handle empty and free-safe;
- CG workspace allocation failure returns `SPARSE_ERR_ALLOC` and leaves cleanup
  safe.

Preferred extended proof:

- GMRES or MINRES growth failure after a successful smaller prepare preserves
  old handle ownership and allows recovery after hook reset.

## Day 5 Failure Harness Implementation

Day 5 implements the deterministic allocation-failure harness for the selected
iterative repeated-run workspace owner.

Implemented private hook:

| Hook | Location | Behavior |
| --- | --- | --- |
| `sparse_alloc_test_fail_after(long remaining)` | `src/sparse_alloc_internal.h` / `.c` | Fails the next non-empty helper allocation when `remaining == 0`; otherwise fails after that many successful helper allocations. |
| `sparse_alloc_test_reset(void)` | `src/sparse_alloc_internal.h` / `.c` | Disables injection and restores the default no-failure state. |

Implemented tests in `tests/test_iterative_handle_helpers.h`:

- owner allocation failure leaves an empty handle empty and free-safe;
- CG workspace allocation failure returns `SPARSE_ERR_ALLOC`, keeps the owner
  cleanup-safe, and recovers after hook reset;
- GMRES growth failure preserves the existing smaller workspace and recovers
  after hook reset;
- MINRES growth failure preserves the existing smaller workspace and recovers
  after hook reset.

Focused validation:

```sh
make build/test_iterative && build/test_iterative
```

Result: `Tests run: 84`, `Tests failed: 0`, `Tests skipped: 0`,
`Assertions: 734`.

Required full gate:

```sh
make format && make lint && make test
```

Result: passed.

## Day 6 Cleanup Invariants

Day 6 strengthens cleanup behavior for invalid iterative repeated-run handle
prepare calls.

Defect found:

- `sparse_iter_handle_prepare_gmres(&handle, n, restart <= 0)` validated the
  bad restart after ensuring the private workspace owner, which could publish
  `handle.internal_state` on an invalid call.

Fix:

- validate `restart <= 0` before `s49_iter_handle_ensure()` allocates the
  owner.

New invariant test:

- `test_iter_handle_invalid_prepare_calls_do_not_publish_state` verifies null
  cleanup, bad-argument prepare calls under active allocation-failure
  injection, empty-handle preservation, later GMRES recovery after hook reset,
  and repeated cleanup.

Focused validation:

```sh
make build/test_iterative && build/test_iterative
```

Result: `Tests run: 85`, `Tests failed: 0`, `Tests skipped: 0`,
`Assertions: 743`.

Required full gate:

```sh
make format && make lint && make test
```

Result: passed.

## Day 7 Regression Gate

Day 7 makes the selected allocation-failure proof explicitly reachable from
maintained validation surfaces without widening the proof beyond the iterative
repeated-run workspace owner.

Maintained surfaces:

- `make iterative-allocation-failure-gate` builds and runs
  `build/test_iterative`.
- CMake labels the existing `test_iterative` CTest row with
  `allocation_failure`, so focused CMake validation can run
  `ctest -L allocation_failure`.

Inventory impact:

- no Make test-count update is required because `test_iterative` was already
  in `TEST_SRCS`;
- no CMake test-count update is required because Day 7 labels an existing
  `add_test` registration instead of adding a new executable;
- Windows CTest-count guards are unchanged for the same reason.

Focused validation:

```sh
make iterative-allocation-failure-gate
```

Result: passed.

Focused CMake validation:

```sh
cmake -S . -B build/sprint176-day7-cmake
cmake --build build/sprint176-day7-cmake --target test_iterative --parallel 1
ctest -N --test-dir build/sprint176-day7-cmake -L allocation_failure
ctest --test-dir build/sprint176-day7-cmake -L allocation_failure --output-on-failure
```

Result: passed.

Required full gate:

```sh
make format && make lint && make test
```

Result: passed.

## Day 8 Invariant Documentation

Day 8 documents the selected cleanup invariant in public API guidance,
adoption docs, and maintainer proof ownership without widening the
allocation-failure claim.

Documentation updates:

- `include/sparse_iterative.h` now states that `sparse_iter_handle_free()` is
  safe on NULL, zeroed, and already-freed handles; invalid prepare arguments
  return before publishing internal state; and selected allocation failures
  leave either an empty handle or the prior usable capacity intact.
- `README.md` now records the same user-facing lifecycle boundary in the
  repeated-run handle section.
- `docs/maintainer_guide.md` now names the exact Sprint 176 allocation-failure
  tests, `make iterative-allocation-failure-gate`, and the
  `allocation_failure` CTest label.

Non-claim retained:

- the proof remains limited to the iterative repeated-run workspace owner and
  does not imply broad allocation-failure cleanup coverage across other solver
  families or allocation paths.

Focused validation:

```sh
make iterative-allocation-failure-gate
```

Result: passed.

Required full gate:

```sh
make format && make lint && make test
```

Result: passed.

## Day 9 Claim Surface Inventory

Day 9 inventories the public, maintainer, generated-report, package, platform,
benchmark, API, and planning claim surfaces before Day 10 edits any public
wording.

Reviewed surfaces:

- `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`,
  `docs/api_reference.md`, `docs/tutorial.md`, `docs/cookbook.md`, and
  `docs/solver_selection.md`;
- `benchmarks/README.md` and `examples/README.md`;
- selected workflow, guard, report-index, package, API docs, and recent Epic
  15 sprint closeout artifacts.

Day 9 claim classification:

- earned: selected iterative repeated-run handle allocation-failure proof;
  selected QR/partial-SVD/LU fixture-local comparison and oracle evidence;
  static-first install/export/package proof;
- local-only: generated report, benchmark, API HTML, coverage, dead-code, and
  the current focused allocation-failure gate unless explicitly promoted by a
  hosted lane later;
- hosted-only: selected Linux oracle/comparison/performance freshness, macOS
  selected comparison freshness, and Windows CMake-first downstream/package
  validation;
- unsupported: broad allocation-failure coverage, state-of-the-art status,
  portable performance superiority, broad external-library parity,
  shared-library/dynamic ABI/runtime-loader support, package-manager provider
  support, broad platform parity, Windows Makefile or `pkg-config` execution
  parity, broad report freshness, hosted generated API HTML, and release
  evidence.

Day 10 checklist:

- add the Sprint 176 allocation-failure proof only where discoverability
  improves;
- keep every allocation-failure claim family-local to CG, GMRES, and MINRES
  repeated-run handle prepare/growth paths;
- cite `make iterative-allocation-failure-gate`;
- preserve the broad allocation-failure non-claim for unrelated solvers,
  matrix construction, package/install flows, and generated-report tooling.

Validation:

```sh
git diff --check
```

Result: passed.

## Day 10 Claim Recalibration

Day 10 applies the Day 9 claim inventory. The only public claim promoted is
the selected Sprint 176 allocation-failure proof for CG, GMRES, and MINRES
repeated-run handle prepare/growth cleanup.

Public documentation updates:

- `README.md` quality summary now names the selected allocation-failure proof
  and cites `make iterative-allocation-failure-gate`.
- `README.md` command map now includes
  `make iterative-allocation-failure-gate` as a focused local proof command.
- `docs/maintainer_guide.md` now records the Day 9/Day 10 interpretation:
  this is an earned local focused proof, not hosted CI, package,
  report-index, performance, release, or state-of-the-art evidence.

Evidence ledger update:

| Evidence | Day 10 status | Boundary |
| --- | --- | --- |
| Selected iterative allocation-failure proof | Earned local focused gate | CG/GMRES/MINRES repeated-run handle prepare/growth cleanup only. |
| Public iterative cleanup invariant | Earned and documented | `sparse_iter_handle_free()` and invalid prepare state-publication behavior only. |
| Broad allocation-failure coverage | Still unsupported | Direct solvers, eigensolvers, matrix construction, package/install flows, generated-report tooling, and unrelated allocation paths remain non-claims. |
| Selected hosted report/performance freshness | Previously earned for selected lanes only | Not allocation-failure proof. |
| Static-first package contract | Previously earned, separate surface | Not allocation-failure proof. |

Guard decision:

- no new guard was added because Day 10 changed documentation only and the
  focused allocation-failure proof is already maintained through
  `make iterative-allocation-failure-gate` and `ctest -L allocation_failure`;
- no package, ABI, package-manager, report-index, generated-output,
  performance, workflow, C, or header surface changed on Day 10.

Validation:

```sh
make iterative-allocation-failure-gate
git diff --check
```

Result: passed.

## Day 11 Epic Retrospective Draft

Day 11 drafts the final Epic 15 retrospective structure from Sprint 167-176
evidence without creating the final Epic retrospective yet. The final
retrospective should wait for Sprint 176 final validation.

Retrospective draft sections:

- Epic summary and source artifact note;
- definition of done checklist across Sprints 167-176;
- completed objective summary by sprint;
- earned claims and evidence links;
- retained non-claims;
- what went well and what did not;
- validation summary;
- residual queue and Epic 16 candidates;
- final claim calibration.

Completed objective draft:

| Sprint | Draft closeout claim |
| --- | --- |
| 167 | Epic 15 baseline, evidence ledger, gap selection, and claim gates established. |
| 168 | One selected hosted performance publication lane created. |
| 169 | Selected performance methodology and sentinel policy hardened. |
| 170 | Static-first shared-library ABI product decision recorded and guarded. |
| 171 | Package-manager support formally deferred and guarded. |
| 172 | One public header family, `sparse_lu.h`, cleaned and guarded. |
| 173 | Generated API HTML kept local-only with freshness and staging guards. |
| 174 | One bounded LU external comparison family added. |
| 175 | Linux/macOS selected comparison freshness promoted for selected artifacts. |
| 176 | Selected iterative repeated-run allocation-failure proof added and claims recalibrated; final validation completed across Days 12-14. |

Residual queue draft:

- broad allocation-failure coverage beyond the selected iterative handle lane;
- Windows report freshness;
- selected oracle freshness outside Linux;
- hosted generated API HTML;
- package-manager provider support;
- shared-library and dynamic ABI support;
- broad external-library parity;
- portable performance superiority;
- broader public-header coherence;
- selected workflow target-list duplication.

Validation:

```sh
git diff --check
```

Result: passed.

## Day 12 Integrated Validation

Day 12 runs the Sprint 176 final validation pass across the selected
allocation-failure proof, package/ABI deferral guard surfaces, selected
report/workflow guard tests, and the full required C quality gate.

Focused allocation, package, and report guard command:

```sh
make iterative-allocation-failure-gate &&
bash scripts/package_manager_deferral_check.sh &&
bash scripts/static_package_deferral_check.sh &&
python3 tests/test_normalize_report_index.py &&
python3 tests/test_selected_comparison_workflow.py &&
python3 tests/test_bench_canonical_freshness.py
```

Result: passed.

Focused observations:

- `iterative-allocation-failure-gate: passed`;
- `test_iterative`: `Tests run: 85`, `Tests failed: 0`,
  `Tests skipped: 0`, `Assertions: 743`;
- package-manager deferral, static package deferral, report-index
  normalization, selected comparison workflow guard, and benchmark freshness
  tests passed.

Required full gate:

```sh
make format && make lint && make test
```

Result: passed.

Full-gate observations:

- formatting completed;
- lint completed, including strict warning compilation, clang-tidy, and
  cppcheck;
- the full test suite completed with `All tests passed.`

Skipped or not repeated locally:

- install scripts were not rerun because Day 12 did not change install rules,
  package metadata templates, installed headers, or package-manager provider
  posture;
- report freshness generator commands were not rerun because Day 12 did not
  change generator scripts, report manifests, benchmark rows, API-doc
  generation inputs, or workflow target inventories;
- hosted artifact publication can only be proven by PR CI, not local
  validation;
- no generated outputs under `build/`, `coverage/`, or `docs/api/` were
  staged.

Patch hygiene:

```sh
git diff --check
```

Result: passed.

## Day 13 Retrospective Finalization

Day 13 finalizes the Epic 15 retrospective from the Sprint 167-175
retrospectives, the Day 11 draft, and the Day 12 integrated-validation record.

Finalized artifact:

- `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md`

Final retrospective contents:

- Epic 15 objective and source artifact note;
- Sprint 167-176 outcome table;
- major outcomes across evidence publication, performance methodology,
  static-first package/ABI decisions, package-manager deferral, generated API
  local-only policy, comparison freshness, and allocation-failure proof;
- validation evidence with boundaries;
- earned claims and retained non-claims;
- prioritized residual queue with next-epic closure targets;
- state-of-the-art assessment;
- what went well, what could be better, key deliverables, and completion
  statement.

Day 11 draft reconciliation:

| Draft point | Day 13 disposition |
| --- | --- |
| Sprint 176 validation pending | Reconciled with Day 12 integrated validation. |
| Allocation-failure proof claim | Finalized as selected CG/GMRES/MINRES repeated-run handle prepare/growth cleanup only. |
| Broad allocation-failure non-claim | Preserved for unrelated solver, matrix, package, install, generated-report, and allocation paths. |
| Residual queue | Converted to prioritized next-epic closure targets. |
| State-of-the-art wording | Finalized as an explicit non-claim with a narrower evidence-discipline assessment. |

Validation:

```sh
git diff --check
```

Result: passed.

## Day 14 Sprint And Epic Closeout

Day 14 completes the Sprint 176 and Epic 15 closeout package.

Final closeout artifacts:

- `docs/planning/EPIC_15/SPRINT_176/RETROSPECTIVE.md`;
- `docs/planning/EPIC_15/SPRINT_176/artifacts/day14-sprint-closeout.md`.

Closeout state:

| Surface | Status |
| --- | --- |
| Day-by-day artifacts | Complete for Days 1-14. |
| Sprint 176 retrospective | Complete. |
| Epic 15 retrospective | Complete. |
| Selected allocation-failure proof | Complete and guarded by `make iterative-allocation-failure-gate` and `ctest -L allocation_failure`. |
| Public claim boundary | Selected CG/GMRES/MINRES repeated-run handle prepare/growth cleanup only. |
| Broad non-claims | Retained for allocation failure, state-of-the-art status, external parity, package-manager providers, shared libraries, dynamic ABI, runtime loading, broad platform parity, hosted generated API HTML, and release evidence. |
| Generated output staging | No generated outputs staged under `build/`, `coverage/`, or `docs/api/`. |

Validation:

```sh
git diff --check
```

Result: passed.

## Retained Claim Non-Claims

Sprint 176 starts with no support claim for:

- broad allocation-failure cleanup guarantees across all solvers;
- allocation-failure proof beyond the subsystem selected during this sprint;
- unqualified state-of-the-art sparse linear algebra status;
- broad external-library ecosystem parity;
- portable performance superiority;
- broad benchmark publication;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager provider availability;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- Windows generated report freshness;
- hosted publication of all generated reports;
- hosted generated API HTML publication;
- release evidence.

## Sprint 176 Stop Conditions

Stop and revise before proceeding if a change:

- selects more than one allocation-failure subsystem for implementation;
- converts one selected subsystem proof into a broad allocation-failure claim;
- changes `.c` or `.h` files without running `make format && make lint &&
  make test`;
- adds allocation hooks that are visible as unsupported public API;
- weakens package-manager, static package, shared-library ABI, performance,
  platform, external-parity, release, or state-of-the-art non-claims;
- updates public claim wording without mapping it to a named evidence source;
- treats local generated output as hosted publication evidence;
- stages generated output under `build/`, `coverage/`, or `docs/api/`;
- hides a deferred product decision instead of listing it in the residual
  queue.

## Working Assumptions

- Day 1 is intake and planning only.
- If only planning files change on a given day, `git diff --check` is
  sufficient for that day.
- If C or public header files change during allocation-failure implementation,
  run `make format && make lint && make test`.
- If scripts, Make targets, workflows, report manifests, docs, or generated
  output rules change, run the focused guard for the affected surface.
- Sprint 176 should close one allocation-failure gap completely and preserve
  evidence-bound claim language for Epic 15 closeout.

## Daily Log

### Day 1: Closeout Intake

- Re-read the active Sprint 176 section in
  `docs/planning/EPIC_15/PROJECT_PLAN.md`.
- Recorded the prompt path mismatch and active source path.
- Created the Sprint 176 artifact directory.
- Inventoried Sprint 167-175 retrospective inputs for final closeout.
- Defined closeout categories, retained non-claims, stop conditions, and
  working assumptions.
- Wrote `artifacts/day1-closeout-intake.md`.

### Day 2: Allocation Inventory

- Inventoried allocation-heavy solver and shared subsystem surfaces.
- Reviewed shared allocation/overflow helpers and confirmed there is no
  general deterministic fail-injection layer today.
- Ranked allocation/free dense source files and separated high-value broad
  candidates from lower-blast-radius proof candidates.
- Recorded existing failure-test coverage versus the deterministic
  allocation-failure proof gap.
- Wrote `artifacts/day2-allocation-inventory.md`.

### Day 3: Subsystem Selection

- Reviewed the Day 2 candidate matrix and focused source reads.
- Selected the iterative repeated-run workspace owner as the single Sprint 176
  allocation-failure proof target.
- Defined public APIs, internal setup paths, allocation points, cleanup paths,
  and expected error behavior in scope.
- Recorded out-of-scope solver families and retained non-claims.
- Wrote `artifacts/day3-subsystem-selection.md`.

### Day 4: Failure Harness Design

- Reviewed existing iterative repeated-run handle tests and registration.
- Designed a private helper-count allocation-failure hook in the internal
  allocation helper layer.
- Defined owner allocation, CG workspace, GMRES growth, and MINRES growth
  failure targets.
- Defined setup, teardown, return-code, cleanup, reuse, and public-API
  boundary assertions.
- Wrote `artifacts/day4-harness-design.md`.

### Day 5: Failure Harness Implementation

- Added private deterministic allocator fault injection in
  `src/sparse_alloc_internal.h` and `src/sparse_alloc_internal.c`.
- Added iterative repeated-run handle allocation-failure tests for owner
  allocation, CG workspace allocation, GMRES growth, and MINRES growth.
- Registered the tests inside the existing `test_iterative` executable.
- Ran the focused iterative validation successfully:
  `make build/test_iterative && build/test_iterative`.
- Ran the required full quality gate successfully:
  `make format && make lint && make test`.
- Wrote `artifacts/day5-harness-implementation.md`.

### Day 6: Cleanup Invariant Implementation

- Re-ran the iterative handle harness and inspected cleanup surfaces.
- Found and fixed the GMRES bad-restart prepare path so invalid restart
  arguments do not allocate or publish private handle state.
- Added `test_iter_handle_invalid_prepare_calls_do_not_publish_state` to cover
  null cleanup, invalid prepare calls under armed allocation failure, later
  recovery, and repeated cleanup.
- Ran the focused iterative validation successfully:
  `make build/test_iterative && build/test_iterative`.
- Ran the required full quality gate successfully:
  `make format && make lint && make test`.
- Wrote `artifacts/day6-cleanup-invariants.md`.

### Day 7: Allocation Failure Regression Gate

- Added the maintained Make focused gate:
  `make iterative-allocation-failure-gate`.
- Added the `allocation_failure` CTest label to the existing `test_iterative`
  registration.
- Confirmed no Make, CMake, or Windows CTest-count update is required because
  no new test executable was added.
- Ran the focused Make regression gate successfully.
- Ran focused CMake configure, build, label listing, and label execution
  successfully.
- Ran the required full quality gate successfully:
  `make format && make lint && make test`.
- Wrote `artifacts/day7-regression-gate.md`.

### Day 8: Cleanup Invariant Documentation

- Updated `include/sparse_iterative.h` with the selected repeated-run handle
  cleanup and allocation-failure invariant.
- Updated `README.md` repeated-run lifecycle guidance with the bounded
  cleanup behavior and proof scope.
- Updated `docs/maintainer_guide.md` with the exact test owners and focused
  validation commands for the Sprint 176 allocation-failure lane.
- Preserved the non-claim that this is not broad allocation-failure coverage.
- Ran the focused allocation-failure gate successfully:
  `make iterative-allocation-failure-gate`.
- Ran the required full quality gate successfully:
  `make format && make lint && make test`.
- Wrote `artifacts/day8-invariant-docs.md`.

### Day 9: Claim Surface Inventory

- Reviewed public and maintainer claim surfaces before final Sprint 176 claim
  recalibration.
- Classified earned, local-only, hosted-only, advisory, supplemental, and
  unsupported claim categories.
- Mapped the Sprint 176 allocation-failure proof to the exact test owner,
  focused Make gate, and CTest label.
- Preserved broad allocation-failure, state-of-the-art, package-manager,
  shared-library, dynamic ABI, runtime-loader, broad platform, Windows
  `pkg-config`, report freshness, generated API HTML, and release non-claims.
- Wrote `artifacts/day9-claim-surface-inventory.md`.

### Day 10: Claim Recalibration

- Updated `README.md` to surface the selected Sprint 176 allocation-failure
  proof without widening it beyond CG/GMRES/MINRES repeated-run handle
  prepare/growth cleanup.
- Added `make iterative-allocation-failure-gate` to the README command map as
  a focused local proof command.
- Updated `docs/maintainer_guide.md` to state that the Sprint 176
  allocation-failure lane is earned local focused proof, not hosted CI,
  package, report-index, performance, release, or state-of-the-art evidence.
- Preserved broad allocation-failure, state-of-the-art, package-manager,
  shared-library, dynamic ABI, runtime-loader, broad platform, Windows
  `pkg-config`, report freshness, generated API HTML, and release non-claims.
- Wrote `artifacts/day10-claim-recalibration.md`.

### Day 11: Epic Retrospective Draft

- Reviewed Sprint 167-175 retrospectives and closeout artifacts plus Sprint
  176 Day 1-10 artifacts.
- Drafted the final Epic 15 retrospective structure.
- Summarized completed objectives by sprint, earned claims, retained
  non-claims, validation families, and residual queue candidates.
- Preserved the final claim calibration: evidence-disciplined and
  static-first with selected hosted/report/performance/comparison proof, not a
  broad state-of-the-art replacement claim.
- Wrote `artifacts/day11-epic-retrospective-draft.md`.

### Day 12: Integrated Validation

- Ran the maintained focused allocation-failure gate successfully:
  `make iterative-allocation-failure-gate`.
- Ran package-manager and static-first deferral guards successfully.
- Ran report-index, selected comparison workflow, and benchmark freshness
  Python guard tests successfully.
- Ran the required full quality gate successfully:
  `make format && make lint && make test`.
- Recorded skipped local checks and hosted-CI-only evidence boundaries.
- Wrote `artifacts/day12-integrated-validation.md`.

### Day 13: Retrospective Finalization

- Reconciled the Day 11 Epic 15 retrospective draft against the Day 12
  integrated-validation record.
- Created `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md`.
- Represented Sprint 167-176 outcomes as complete, narrowed, or residualized.
- Published the final residual queue and next-epic closure targets.
- Preserved the final state-of-the-art non-claim and evidence-bound Epic 15
  completion statement.
- Wrote `artifacts/day13-retrospective-finalization.md`.

### Day 14: Sprint And Epic Closeout

- Reviewed all Sprint 176 artifacts for consistency and completeness.
- Created `docs/planning/EPIC_15/SPRINT_176/RETROSPECTIVE.md`.
- Confirmed the Epic 15 retrospective, claim docs, and allocation-failure
  evidence agree on selected proof scope.
- Recorded the final Sprint 176 and Epic 15 handoff state.
- Preserved explicit residuals and non-claims.
- Wrote `artifacts/day14-sprint-closeout.md`.
