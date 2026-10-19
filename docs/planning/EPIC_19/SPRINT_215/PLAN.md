# Sprint 215 Plan: External Baseline And State-Of-The-Art Evidence Blueprint

**Sprint Duration:** 14 days
**Goal:** Define an executable, reproducible external-baseline evidence blueprint
and run one selected pilot only if its dependency, correctness, and provenance
gates can be met. Do not promote a pilot into a broad state-of-the-art claim.

**Time budget:** No day exceeds 12 hours. Twelve 12-hour days and two 11-hour
days total `166` hours, matching the Epic 19 project-plan estimate.

**Primary scope:** Select exact external baselines, solver families, versions,
and matrix subset; specify correctness and measurement methodology; design or
extend report fields for external identity and evidence status; make a gated
pilot decision; calibrate public claims; and validate the selected surfaces.

**Non-goals:** A portable speed ranking, broad solver-family parity,
state-of-the-art performance, hosted timing threshold, Linux/macOS timing
parity, Windows evidence, release benchmark status, and endorsement of an
external package are not assumed. Existing selected canonical freshness is
threshold-free; local performance sentinels remain a separate policy. A pilot
on one toolchain and matrix does not establish general correctness or speed.

**Pilot decision rule:** Day 8 records a go/no-go for one named baseline,
version, solver operation, matrix, and platform. Proceed only if the external
package and fixture can be pinned and used legally, both solvers expose
comparable operations, correctness checks can be stated before timing, and the
run can emit the required provenance. Otherwise spend the pilot allocation on
an executable harness specification, blocker reproduction, and residual queue;
mark the pilot `not_run` or `blocked`, never `pass` or `parity`.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 215.1 Baseline Selection | 26 h | Days 1-3: 12 + 12 + 2 |
| 215.2 Methodology Specification | 34 h | Days 3-5: 10 + 12 + 12 |
| 215.3 Report Schema Design | 28 h | Days 6-8: 12 + 12 + 4 |
| 215.4 Selected Pilot | 34 h | Days 8-11: 8 + 12 + 12 + 2 |
| 215.5 Claim Documentation | 20 h | Days 11-12: 10 + 10 |
| 215.6 Validation And Closeout | 24 h | Days 12-14: 2 + 11 + 11 |
| **Total** | **166 h** | **Days 1-12: 12 h each; Days 13-14: 11 h each** |

---

## Day 1: Evidence And Contract Intake

**Title:** Evidence And Contract Intake
**Theme:** Establish what the current benchmark and comparison surfaces prove.
**Time estimate:** 12 hours

### Tasks

1. Map items 215.1-215.6 to artifacts, potential code owners, validation
   commands, and explicit claim boundaries.
2. Read Sprint 212's threshold-free benchmark decision, current selected
   comparison/report contracts, selected-target manifest, report-family schema,
   benchmark guide, and relevant Epic 19 residuals.
3. Inventory current matrix fixtures, corpus provenance, solver operations,
   correctness fields, timing fields, manifests, generated report paths, and
   local versus hosted evidence lanes.
4. Create `WORKING_NOTES.md` with an item ledger, candidate register, decision
   log, risk register, changed-surface tracker, and validation matrix.
5. Write `artifacts/day1-external-baseline-evidence-intake.md` with existing
   evidence, unknowns, and non-claims.

### Deliverables

- Sprint 215 working-notes scaffold and item-to-evidence map.
- Current benchmark/comparison contract inventory and Day 1 intake artifact.

### Completion Criteria

- Each Epic item has an owner, evidence artifact, and provisional acceptance
  check.
- Existing threshold-free selected freshness is separated from external
  baseline correctness and performance evidence.
- No external or state-of-the-art comparison is claimed from current internal
  benchmark rows.

---

## Day 2: External Candidate And Fixture Survey

**Title:** External Candidate And Fixture Survey
**Theme:** Screen feasible baselines before committing to a comparison.
**Time estimate:** 12 hours

### Tasks

1. Inventory candidate external libraries or tools by exact project name,
   release/version, distribution source, license, build prerequisites, solver
   family, operation, and supported platform/compiler.
2. Map candidate operations to the repository's corresponding solver paths;
   flag differences in ordering, pivoting, preconditioning, precision,
   stopping criteria, and backend dependencies.
3. Select a small candidate matrix-suite subset from source-controlled
   fixtures, recording dimensions, sparsity, matrix properties, provenance,
   license, and content hashes where applicable.
4. Record package availability, optional-data requirements, reproducibility
   risks, and any reason an external result could not be compared fairly.
5. Write `artifacts/day2-candidate-baselines-and-matrices.md` with candidates,
   exclusions, and a pilot-feasibility shortlist.

### Deliverables

- Versioned external candidate and solver-family comparison table.
- Matrix-subset shortlist with provenance and exclusion reasons.
- Day 2 survey artifact.

### Completion Criteria

- Candidate versions and acquisition routes are explicit, not floating names.
- At least one potential narrow comparison cell is identified, or a concrete
  blocker is recorded.
- Licensing and matrix provenance are treated as pilot gates, not footnotes.

---

## Day 3: Baseline Decision And Methodology Inputs

**Title:** Baseline Decision And Methodology Inputs
**Theme:** Freeze comparison identity and begin the fairness contract.
**Time estimate:** 12 hours (215.1: 2 h; 215.2: 10 h)

### Tasks

1. Select the exact baseline version(s), solver family, operation, and
   matrix-suite subset for the blueprint; distinguish the single pilot cell
   from future expansion candidates.
2. Record rejected candidates and missing package, license, fixture, or
   operation evidence in `artifacts/day3-baseline-selection-decision.md`.
3. Define a comparison-cell identity tuple covering matrix hash, problem
   class, solver path, precision, backend, ordering, thread count, and build
   mode so unlike operations cannot share a result row.
4. Specify required input conversion checks and rules for singular,
   indefinite, rank-deficient, or otherwise inapplicable matrices.
5. Start `artifacts/day3-methodology-inputs.md` with assumptions, exclusions,
   and unresolved tolerance or measurement decisions.

### Deliverables

- Item 215.1 baseline-selection decision with exact versions and fixture keys.
- Comparison-cell identity and methodology-input record.

### Completion Criteria

- The baseline and matrix choices are reproducible or explicitly blocked by
  named missing inputs.
- A pilot cannot silently switch solver, dataset, or package version.
- Correctness and timing requirements are not conflated.

---

## Day 4: Correctness Metrics And Tolerances

**Title:** Correctness Metrics And Tolerances
**Theme:** Define numerical acceptance before measuring speed.
**Time estimate:** 12 hours

### Tasks

1. Specify status, convergence, residual or backward-error, solution error
   where an oracle exists, iteration, and factorization diagnostics for each
   selected solver family.
2. Define absolute and relative tolerances as functions of precision, matrix
   scale, conditioning, and problem class; do not infer equality from matching
   row labels or success exit codes.
3. Define how to handle different reorderings, pivots, warm starts, stopping
   conditions, and unsupported or failed external solves.
4. Specify negative fixtures for mismatched matrices, missing oracle data,
   failed convergence, nonfinite outputs, and divergent solver semantics.
5. Write `artifacts/day4-correctness-and-tolerance-contract.md` with formulas,
   acceptance tables, and explicit reasons a row is non-comparable.

### Deliverables

- Correctness-metric and tolerance specification for item 215.2.
- Failure and non-comparability matrix.
- Day 4 contract artifact.

### Completion Criteria

- Every selected cell has a predeclared correctness gate or an explicit
  `not_comparable` disposition.
- Timing cannot be treated as evidence when either solver fails its numerical
  gate.
- No tolerance is chosen after seeing a favorable pilot result.

---

## Day 5: Measurement And Provenance Methodology

**Title:** Measurement And Provenance Methodology
**Theme:** Make any later timing result interpretable and repeatable.
**Time estimate:** 12 hours

### Tasks

1. Define phase boundaries for setup, factorization, solve, reuse, and total
   runtime; specify which phases are comparable for the selected operation.
2. Specify warmup, repeat/sample count, aggregation, dispersion, timeout,
   memory/fill metrics, failure handling, and how unavailable metrics are
   represented without inventing zero values.
3. Record required platform, CPU/architecture, compiler and flags, library
   build options, BLAS/backend identity, thread environment, matrix hashes,
   package source/version, command, commit, and artifact hashes.
4. Separate local pilot measurements from hosted or cross-platform evidence;
   retain Sprint 212's threshold-free policy and prohibit speed rankings from
   single-sample or unmatched-environment data.
5. Complete `artifacts/day5-external-comparison-methodology.md` with a
   rerun checklist and acceptance/non-claim rules.

### Deliverables

- Full correctness/performance methodology for item 215.2.
- Environment and package-provenance capture checklist.
- Day 5 methodology artifact.

### Completion Criteria

- Required metrics and reporting fields are defined before schema or pilot
  implementation.
- A reviewer can reproduce the intended comparison cell from recorded inputs.
- Cross-platform parity, hosted threshold, and broad performance claims remain
  explicitly unearned.

---

## Day 6: External Evidence Schema Inventory

**Title:** External Evidence Schema Inventory
**Theme:** Choose a schema home without weakening current report contracts.
**Time estimate:** 12 hours

### Tasks

1. Inspect `tests/corpus/schemas/report_index_fields.md`, report-family and
   selected-target manifests, `scripts/normalize_report_index.py`, canonical
   benchmark output, and existing comparison rows for reusable identities.
2. Decide whether a compatible selected extension or a separate external
   evidence schema best preserves existing row meanings and freshness gates.
3. Define fields for baseline project/version/source, solver operation,
   package/build provenance, matrix identity, methodology version, platform,
   measurement units, and raw artifact references.
4. Define evidence status values such as `planned`, `blocked`, `not_run`,
   `measured`, and `non_comparable`; prevent absent optional dependencies from
   becoming successful comparisons.
5. Write `artifacts/day6-external-report-schema-design.md` with field types,
   authority boundaries, compatibility risks, and example rows.

### Deliverables

- Schema-home decision and external-row field dictionary.
- Evidence-status and provenance vocabulary.
- Day 6 schema design artifact.

### Completion Criteria

- External identity is not inferred from an internal benchmark target name.
- Every row can distinguish a blueprint, a blocked pilot, and observed
  numerical/timing evidence.
- Existing selected canonical freshness and report-index semantics remain
  unchanged unless a tested migration is explicitly selected.

---

## Day 7: Schema Validation And Compatibility Design

**Title:** Schema Validation And Compatibility Design
**Theme:** Make the proposed external fields enforceable.
**Time estimate:** 12 hours

### Tasks

1. Define required-field and allowed-value validation for identity,
   provenance, methodology, status, metrics, and artifact references.
2. Specify cross-field rules: a `measured` row needs both solvers' correctness
   outcomes and raw artifacts; `blocked` or `not_run` rows cannot carry a
   performance verdict; mismatched comparison-cell keys cannot be ranked.
3. Design schema or normalizer changes and focused mutation fixtures if the
   selected schema home is implemented this sprint; otherwise provide a
   complete parseable example and executable validation design.
4. Define migration/backward-compatibility checks for current manifest rows,
   report-index normalization, and selected workflow tests.
5. Write `artifacts/day7-schema-validation-design.md` with positive and
   negative fixture matrix and exact validation commands.

### Deliverables

- Testable external evidence schema contract for item 215.3.
- Compatibility and regression fixture design.
- Day 7 validation-design artifact.

### Completion Criteria

- Missing provenance, unsupported status, unit drift, and unjustified
  performance conclusions have specified rejection behavior.
- Existing report families can still validate unchanged or have a bounded,
  documented migration.
- Schema design does not claim that a pilot has run.

---

## Day 8: Schema Acceptance And Pilot Go/No-Go

**Title:** Schema Acceptance And Pilot Go/No-Go
**Theme:** Close the schema decision and select one executable pilot path.
**Time estimate:** 12 hours (215.3: 4 h; 215.4: 8 h)

### Tasks

1. Finalize the schema example, field ownership, validator design or bounded
   implementation, and regression expectations from Days 6-7.
2. Evaluate the Day 3 pilot cell against pinned dependency/source, license,
   fixture hash, equivalent operation, correctness oracle, environment capture,
   and time budget.
3. Record a go/no-go decision with exact baseline version, matrix, commands,
   expected outputs, owner, and stop conditions in
   `artifacts/day8-pilot-feasibility-decision.md`.
4. For go, scope one optional local pilot without changing canonical or hosted
   thresholds; for no-go, specify a runnable harness design and reproduce the
   blocker rather than reporting a skipped run as evidence.
5. Identify all code, data, schema, and docs files allowed to change during
   the pilot and the focused tests each change requires.

### Deliverables

- Item 215.3 schema acceptance record.
- Pilot go/no-go decision and bounded worklist for item 215.4.
- Day 8 feasibility artifact.

### Completion Criteria

- The schema can represent both executed and blocked outcomes without
  changing their meaning.
- Exactly one pilot cell is selected or a documented blocker activates the
  executable-design path.
- No external package is made mandatory for the maintained build or test path.

---

## Day 9: Pilot Harness Or Executable Fallback

**Title:** Pilot Harness Or Executable Fallback
**Theme:** Build the narrow comparison mechanism without broadening claims.
**Time estimate:** 12 hours

### Tasks

1. For a go decision, implement the smallest opt-in external baseline runner
   or adapter using the pinned dependency and selected matrix; keep default
   builds and CI independent of the external package.
2. Capture both solver commands, matrix and package hashes, build/compiler
   flags, environment, raw output, and failure logs before normalizing rows.
3. Add focused positive and negative fixtures for status, correctness,
   missing dependency, malformed output, and mismatched comparison keys.
4. For no-go, write the exact source layout, install/build commands, CLI
   invocation, output parser contract, fixtures, and dependency gate needed to
   run the pilot later; preserve the reproduced blocker log.
5. Record implementation and fallback choices in
   `artifacts/day9-selected-pilot-harness.md`.

### Deliverables

- Opt-in pilot harness and tests, or executable blocked-pilot specification.
- Raw-output and provenance capture contract.
- Day 9 harness artifact.

### Completion Criteria

- A missing optional dependency is a visible `blocked` or `not_run` outcome,
  not a passing comparison.
- Internal and external results retain their distinct identities and raw
  artifacts.
- No unselected matrix or solver family enters the pilot scope.

---

## Day 10: Selected Pilot Execution Or Blocker Proof

**Title:** Selected Pilot Execution Or Blocker Proof
**Theme:** Produce one honest outcome for the selected cell.
**Time estimate:** 12 hours

### Tasks

1. Run the opt-in pilot only after confirming the Day 8 dependency and
   comparability gates; otherwise execute the blocker reproduction and
   validate the fallback design against available local inputs.
2. Evaluate correctness for both solvers before comparing phase timings;
   record nonfinite values, solver failures, or semantic mismatch explicitly.
3. Capture raw outputs, commands, commit, toolchain, package versions,
   matrix hashes, environment, sample policy, and schema-validation result.
4. Test one negative mutation of baseline identity, matrix identity, or
   correctness outcome so the pilot cannot silently report a favorable
   comparison from incompatible inputs.
5. Write `artifacts/day10-selected-pilot-outcome.md` with measured,
   non-comparable, or blocked status and the exact claim it supports.

### Deliverables

- One selected pilot evidence bundle if feasible, or reproducible blocker
  evidence and executable fallback design.
- Day 10 outcome artifact.

### Completion Criteria

- Any measured result passes predeclared correctness and provenance gates.
- Missing samples, dependencies, or raw artifacts do not count as performance
  evidence.
- The outcome is bounded to the selected matrix, solver, version, and local
  environment.

---

## Day 11: Pilot Review And Claim Matrix

**Title:** Pilot Review And Claim Matrix
**Theme:** Separate what ran from what remains only a blueprint.
**Time estimate:** 12 hours (215.4: 2 h; 215.5: 10 h)

### Tasks

1. Audit the Day 10 result against the Day 3-5 frozen methodology and close
   item 215.4 as executed, non-comparable, or blocked with specific evidence.
2. Build a claim matrix for blueprint-only, selected local pilot,
   cross-matrix, cross-solver, hosted, cross-platform, and state-of-the-art
   wording; mark the last five as unearned unless separately proven.
3. Update benchmark documentation to distinguish current internal selected
   freshness from external-baseline design and any actual pilot observation.
4. Identify README, maintainer guide, solver-selection, and report-index text
   that could accidentally convert a pilot into broad parity or a speed claim.
5. Write `artifacts/day11-evidence-and-claim-matrix.md` with approved wording,
   rejected wording, source links, and remaining evidence gates.

### Deliverables

- Reviewed item 215.4 outcome and item 215.5 claim matrix.
- Initial benchmark and maintainer documentation updates.
- Day 11 claim artifact.

### Completion Criteria

- Docs say whether the pilot actually ran and identify its exact cell.
- Blueprint requirements are not presented as completed external evidence.
- No broad state-of-the-art, parity, or portable-performance statement is
  supported by a single selected pilot.

---

## Day 12: Documentation Closure And Validation Setup

**Title:** Documentation Closure And Validation Setup
**Theme:** Finish claim-boundary text and prepare the acceptance run.
**Time estimate:** 12 hours (215.5: 10 h; 215.6: 2 h)

### Tasks

1. Finish user and maintainer wording for external baseline selection,
   correctness prerequisites, provenance, raw artifacts, evidence statuses,
   pilot outcome, and future stronger-claim gates.
2. Add or adjust documentation regressions where current guards would allow
   a blueprint, blocked pilot, or single measurement to be described as
   state-of-the-art or cross-platform proof.
3. Reconcile the schema examples, pilot commands, and report field names with
   actual implementation or executable fallback design.
4. Prepare a changed-surface validation matrix for schema, report,
   comparison, docs, and focused pilot checks, with exact commands and
   expected outcomes.
5. Write `artifacts/day12-claim-documentation-and-validation-plan.md`.

### Deliverables

- Completed item 215.5 documentation and non-claim guard plan.
- Day 12 validation command matrix.

### Completion Criteria

- Public text distinguishes design, local observation, and unearned broad
  claims consistently.
- Every changed script or report contract has a focused validation command.
- Any `.c` or `.h` edits are identified for the mandatory full C gate.

---

## Day 13: Schema, Report, Comparison, And Pilot Validation

**Title:** Schema, Report, Comparison, And Pilot Validation
**Theme:** Run the relevant contract checks and inspect raw evidence.
**Time estimate:** 11 hours

### Tasks

1. Run source-controlled corpus/schema and selected-manifest validation,
   including `python3 scripts/validate_corpus_schema.py` and affected
   `tests/test_selected_report_targets_manifest.py` fixtures.
2. Run affected report-index, selected-comparison, and canonical benchmark
   freshness tests; use maintained commands such as
   `make report-index-comparison-freshness` and
   `make bench-canonical-report-freshness-tests` where relevant.
3. Run `make docs-check` and any new external-claim regression checks; run
   the focused pilot test or blocker-reproduction command from Day 8.
4. If any `.c` or `.h` file changed, run `make format && make lint && make test`
   and resolve failures before closeout. Preserve raw logs and exit statuses.
5. Write `artifacts/day13-integrated-validation.md` with exact commands,
   platform, passing/failing outcomes, and evidence limitations.

### Deliverables

- Schema, report, comparison, docs, and pilot validation record.
- Full C quality-gate evidence if applicable.
- Day 13 integrated-validation artifact.

### Completion Criteria

- Required checks pass; failures are fixed and rerun or explicitly block
  closeout rather than being reclassified as skips.
- Generated rows and raw artifacts agree on baseline identity, status, and
  methodology.
- The validation record distinguishes local runs from hosted evidence not
  executed in this sprint.

---

## Day 14: Blueprint And Sprint Closeout

**Title:** Blueprint And Sprint Closeout
**Theme:** Package the executable blueprint and decide what remains unearned.
**Time estimate:** 11 hours

### Tasks

1. Audit all six Epic items against their estimated scope, deliverables,
   changed files, and Day 13 validation evidence.
2. Write `STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md` with exact baseline
   selection, methodology, report schema, pilot status, rerun instructions,
   acceptance gates, and the minimum matrix/platform expansion needed for
   stronger claims.
3. Update `RESIDUAL_QUEUE.md` with blocked dependency work, additional
   baselines, matrix expansion, hosted provenance, and future claim gates;
   assign owners or explicit decision points.
4. Write `artifacts/day14-closeout-review.md` and `RETROSPECTIVE.md`,
   reconciling planned versus observed pilot evidence without inflating claims.
5. Recheck documentation links, generated-file exclusions, working-notes
   ledger, `git diff --check`, and any focused checks affected by closeout edits.

### Deliverables

- `STATE_OF_THE_ART_EVIDENCE_BLUEPRINT.md`, the executable external-baseline
  and state-of-the-art evidence blueprint.
- Selected pilot bundle or documented blocker and runnable design.
- Residual queue, Day 14 closeout artifact, and retrospective.

### Completion Criteria

- Items 215.1-215.6 each have a traceable outcome and evidence location.
- The selected pilot is identified as executed, non-comparable, or blocked;
  no absent result is promoted to pass evidence.
- All required validation passes, and public claims remain no stronger than
  the exact matrix, baseline, solver, and platform evidence earned.
