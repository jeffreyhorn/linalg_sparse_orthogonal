# Sprint 221 Plan: External Evidence V1 Validator

**Sprint Duration:** 14 days
**Goal:** Close the missing schema and validator prerequisite for the optional
independent CHOLMOD baseline pilot without treating validation as a solver
result.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds
12 hours; the Epic 20 sprint capacity is 168 hours. CI elapsed time is not
included in the estimates.

**Prerequisites:** Sprint 215's accepted v1 external-evidence design,
planning-only blueprint and blocked pilot record; Sprint 220's local-only
publication/retention conventions; the current corpus manifest and selected
report boundaries. The selected pilot remains one correctness-only,
opt-in local cell on `tests/data/symmetric_4.mtx`.

**Primary scope:** `tests/corpus/schemas/external_baseline_evidence_v1.json`,
`scripts/validate_external_baseline_evidence.py`, focused validator fixtures
and tests, `scripts/run_cholmod_pilot.py` and its focused tests, optional
preflight wiring, and maintainer guidance. Keep generated attempt bundles
under `build/external-baseline/`; do not alter selected report manifests,
normalization, canonical benchmark rows or the generated-API local-only rule.

**Evidence rule:** Freeze the v1 contract before accepting generated records.
Planning, blocked and synthetic observed fixtures are distinct from actual
CHOLMOD observations. Tests must assert both verdict and stable rule code;
mutations occur only in temporary copies. An unavailable dependency,
incomplete provenance, ambiguous identity, failed self-test or failed required
check blocks promotion. Publication approval is separate from local evidence
validation.

**Non-goals:** Running or publishing a new external solve, claiming solver
parity or speed, adding a default-build SuiteSparse dependency, or making
the selected report normalizer consume this evidence family. Sprint 222 owns
the first genuinely independent two-sided correctness cell.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 221.1 Schema Freeze | 24 h | Days 1-3: 10 + 10 + 4 |
| 221.2 Validator Implementation | 30 h | Days 3-6: 6 + 10 + 10 + 4 |
| 221.3 Fixture Matrix | 28 h | Days 6-9: 6 + 10 + 10 + 2 |
| 221.4 Pilot Preflight | 22 h | Days 9-11: 8 + 10 + 4 |
| 221.5 Mutation Proof | 18 h | Days 11-13: 6 + 10 + 2 |
| 221.6 Docs And Validation | 18 h | Days 13-14: 8 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Evidence Contract Inventory

**Title:** Evidence Contract Inventory
**Theme:** Establish the accepted v1 design and current pilot blocker.
**Time estimate:** 10 hours (221.1: 10 h)

### Tasks

1. Read the Sprint 215 Day 6-8 schema decisions, blueprint example, Day 10
   blocked attempt and current pilot preflight/tests; map each required field
   to its source and validation owner.
2. Inventory selected matrix bytes/hash, operation, RHS/oracle, project and
   CHOLMOD controls, current corpus manifest boundaries and bundle layout.
3. List status, provenance, artifact, unit and claim rules that require
   cross-field validation rather than JSON shape checks alone.
4. Start `WORKING_NOTES.md` with source revision, commands, owner map,
   unresolved decisions and the no-observed-result boundary.

### Deliverables

- Field/owner inventory and current pilot-blocker record.
- Initial working notes and validation risk ledger.

### Completion Criteria

- Every accepted Sprint 215 v1 requirement has a proposed validator or
  fixture owner.
- The current missing-schema/missing-validator gate is recorded as blocked,
  not as an external result.

---

## Day 2: Status, Units And Provenance Freeze

**Title:** Status, Units And Provenance Freeze
**Theme:** Make the v1 acceptance vocabulary unambiguous.
**Time estimate:** 10 hours (221.1: 10 h)

### Tasks

1. Freeze required and optional fields, exact types, unknown-field policy,
   lowercase SHA-256 syntax, UTC timestamps and complete generated-run Git
   identities; booleans must not count as integers.
2. Freeze `planned`, `not_run`, `blocked`, `non_comparable`,
   `correctness_failed`, `correctness_observed`, `timing_invalid` and
   `measured` cross-field relationships, including null-with-reason rules.
3. Specify residual, oracle-error and backward-error units and the `ns`
   timing/sample contract; prohibit missing values disguised as zero or
   nonfinite numbers.
4. Freeze artifact roots, safe relative paths, hash coverage, publication
   review and forbidden speed/winner/platform claims.

### Deliverables

- Versioned v1 field and status matrix with stable rule codes.
- Explicit provenance, unit, path and claim-boundary decisions.

### Completion Criteria

- Each accepted status has a positive shape and contradictory negative
  shapes.
- No blueprint or execution-blocked record can imply an observed pass.

---

## Day 3: Canonical Identity And Schema Authority

**Title:** Canonical Identity And Schema Authority
**Theme:** Complete the freeze and start the validator boundary.
**Time estimate:** 10 hours (221.1: 4 h; 221.2: 6 h)

### Tasks

1. Freeze the exact complete `cell_id` identity object, dependency sort by
   role/absolute loaded path, sorted-key compact ASCII JSON bytes and SHA-256
   prefix; add a canonical byte/hash test vector.
2. Specify which fields are immutable selection controls, which are run
   provenance, and why a partial identity retains `cell_id=null` with reason.
3. Create the v1 schema authority at
   `tests/corpus/schemas/external_baseline_evidence_v1.json` with version,
   enums, field shapes and claim tokens tied to the frozen contract.
4. Establish the stdlib-only validator interface: `--schema`, `--input`,
   optional `--bundle-root` and `--self-test`, with stable error codes and
   source locations.

### Deliverables

- Frozen cell-ID serialization/vector and v1 schema file.
- Validator interface and rule-code map.

### Completion Criteria

- Item 221.1's 24-hour freeze is closed before any generated record can be
  accepted.
- Changing a selected identity field requires a new selection decision, not
  reuse of the old cell ID.

---

## Day 4: Strict JSON And Structural Validation

**Title:** Strict JSON And Structural Validation
**Theme:** Reject malformed or ambiguous evidence at the boundary.
**Time estimate:** 10 hours (221.2: 10 h)

### Tasks

1. Implement UTF-8 JSON loading with duplicate-key and NaN/Infinity
   rejection, including nested objects, with deterministic diagnostics.
2. Enforce required fields, exact scalar/container types, unsupported enums,
   nonempty identifiers, unknown-field rejection and valid version selection.
3. Check the pinned selection ID, matrix path/hash, dimensions, operation,
   RHS/oracle, precision and solver controls before status promotion.
4. Add focused parser and shape tests for missing, extra, mistyped,
   duplicate and unsupported fields.

### Deliverables

- Initial `validate_external_baseline_evidence.py` implementation.
- Structural positive and negative tests with stable rule codes.

### Completion Criteria

- Syntax-valid but ambiguous or structurally incomplete records fail closed.
- The planning blueprint can be parsed without being promoted to observed
  evidence.

---

## Day 5: Semantic Status And Cell Validation

**Title:** Semantic Status And Cell Validation
**Theme:** Enforce relationships that shape validation cannot express.
**Time estimate:** 10 hours (221.2: 10 h)

### Tasks

1. Implement the full v1 status matrix and side/timing transitions; reject
   `measured` without two verified passes and `blocked` with a performance
   verdict.
2. Recompute complete `cell_id` values from canonical identity bytes;
   reject mismatches, duplicate dependency roles/paths and incomplete
   observed identities.
3. Validate exact metric units, finite raw values, declared limits, sample
   counts and timing aggregates against preserved raw values where present.
4. Keep claim scope and publication review independent of local correctness;
   reject unearned speed, parity or state-of-the-art fields.

### Deliverables

- Semantic validator rules and canonical cell-ID tests.
- Contradictory-status and unearned-claim regressions.

### Completion Criteria

- A status label alone cannot make an invalid observation pass.
- Locally valid evidence does not acquire publication rights or performance
  claims from the validator.

---

## Day 6: Safe Artifacts And Fixture Harness

**Title:** Safe Artifacts And Fixture Harness
**Theme:** Close validator I/O and begin reproducible fixtures.
**Time estimate:** 10 hours (221.2: 4 h; 221.3: 6 h)

### Tasks

1. Verify artifact references only within the declared repo/bundle root;
   reject absolute paths, traversal, symlink escapes, missing files and
   SHA-256 mismatches using actual bytes.
2. Finish validator error handling for inaccessible files and unavailable
   bundle roots without accepting partial evidence.
3. Build temporary-bundle fixture helpers that hash their own raw files and
   leave source-controlled manifests, corpus data and benchmark bundles
   untouched.
4. Capture the current blueprint as a positive planning fixture and freeze
   its `planned`/no-result assertions.

### Deliverables

- Complete first-pass semantic validator and safe-artifact checks.
- Fixture harness and positive planning test.

### Completion Criteria

- Item 221.2's 30-hour implementation is closed with focused tests.
- An artifact path or digest cannot be accepted without a real in-root file.

---

## Day 7: Planning And Blocked Fixtures

**Title:** Planning And Blocked Fixtures
**Theme:** Prove honest non-execution states are valid but non-promotable.
**Time estimate:** 10 hours (221.3: 10 h)

### Tasks

1. Add a canonical planning fixture with `run_id=null`, `cell_id=null`, two
   `not_run` sides, `timing=not_run` and no generated artifacts.
2. Add synthetic generated `not_run` and execution-`blocked` fixtures with
   named reasons and absent runtime observations.
3. Add a publication-blocked shape only if locally observed data is fully
   supported and separately hashed; otherwise record why it is excluded.
4. Test that absent CHOLMOD, missing module review or an unverified loaded
   binary cannot be represented as a baseline pass.

### Deliverables

- Positive planning, `not_run` and blocked fixtures/tests.
- Explicit non-promotion assertions for unavailable external evidence.

### Completion Criteria

- Every accepted non-execution fixture retains a reason and no numerical
  pass, timing sample or public result claim.
- Validation of a blocked record is not mistaken for a completed pilot.

---

## Day 8: Synthetic Observed And Measured Fixtures

**Title:** Synthetic Observed And Measured Fixtures
**Theme:** Exercise the strongest statuses without claiming an actual run.
**Time estimate:** 10 hours (221.3: 10 h)

### Tasks

1. Build a fully synthetic two-sided correctness fixture with complete
   identity, both solver-side raw values, Day 4 metrics and hashed artifacts.
2. Build a synthetic measured fixture with verified correctness, monotonic
   clock/thread provenance, five warmups and 50 alternating-order timed pairs
   per reported window, raw hashes and recomputed statistics.
3. Check selection/cell identity and methodology hashes against the fixture
   bytes; keep test-only provenance visibly synthetic.
4. Add negative neighbors for one-sided pass, missing raw output, invalid
   timing aggregates and an attempted speed/winner claim.

### Deliverables

- Canonical synthetic `correctness_observed` and `measured` fixtures.
- Paired accept/reject tests proving the raw-data requirements.

### Completion Criteria

- Both strong statuses require complete, internally consistent test data.
- No synthetic fixture is presented as a real CHOLMOD observation.

---

## Day 9: Fixture Closure And Pilot Gate Design

**Title:** Fixture Closure And Pilot Gate Design
**Theme:** Finish the fixture matrix and define the pre-solve stop point.
**Time estimate:** 10 hours (221.3: 2 h; 221.4: 8 h)

### Tasks

1. Close positive/negative fixture coverage for every status, required field
   group and named v1 rule code; identify unsupported shapes explicitly.
2. Trace `run_cholmod_pilot.py` from discovery through build and solver calls
   to locate the earliest fail-closed `--self-test` gate.
3. Define the preflight outcome record for missing schema, missing validator,
   failing self-test and invalid evidence: blocked with empty sides and no
   solve/timing output.
4. Define optional CI and local commands that exercise validator self-tests
   without installing CHOLMOD or changing selected reports.

### Deliverables

- Closed canonical fixture matrix.
- Pilot preflight control-flow and test plan.

### Completion Criteria

- Item 221.3's 28-hour fixture allocation is closed.
- The pilot cannot reach build or solver execution before a passing v1
  validator self-test.

---

## Day 10: Fail-Closed Pilot Preflight

**Title:** Fail-Closed Pilot Preflight
**Theme:** Make validation an executable prerequisite, not a result.
**Time estimate:** 10 hours (221.4: 10 h)

### Tasks

1. Wire the real schema and validator `--self-test` into the optional pilot's
   preflight before helper build or either solver command.
2. Preserve immutable attempt paths and raw preflight stdout/stderr/hash
   capture; reject overwrite or a missing/unreadable schema.
3. Keep solver-status, package-version, matrix-hash and provenance gates
   fail-closed after self-test succeeds.
4. Add focused tests proving that missing or invalid validator inputs leave
   `sides={}`, `timing=not_run` and no project/baseline execution.

### Deliverables

- Optional pilot preflight using the real v1 self-test.
- Pre-solve failure tests and raw-log assertions.

### Completion Criteria

- A failed or absent validation gate stops before building or solving.
- Passing self-test authorizes only the next pilot gate, not an observed
  correctness or timing claim.

---

## Day 11: Pilot Integration And Mutation Seeds

**Title:** Pilot Integration And Mutation Seeds
**Theme:** Finish preflight coverage and start adversarial proof.
**Time estimate:** 10 hours (221.4: 4 h; 221.5: 6 h)

### Tasks

1. Test the optional pilot with a passing self-test and mocked downstream
   gates, proving only the intended build/solver path is reached.
2. Register validator self-test in the optional/maintained Python CI route
   without making CHOLMOD or a generated evidence bundle a default dependency.
3. Create mutation helpers that copy valid fixtures into temporary roots,
   alter one field or byte sequence, invoke the CLI and assert stable error
   codes and nonzero exit status.
4. Seed mutations for duplicate keys, unknown fields, nonfinite values and
   missing execution provenance.

### Deliverables

- Completed pilot and CI preflight wiring.
- Mutation harness and first negative cases.

### Completion Criteria

- Item 221.4's 22-hour preflight allocation is closed.
- Optional self-test failure is visible in CI and cannot silently fall
  through to solver execution.

---

## Day 12: Semantic Mutation Proof

**Title:** Semantic Mutation Proof
**Theme:** Show syntax-valid corrupt evidence fails closed.
**Time estimate:** 10 hours (221.5: 10 h)

### Tasks

1. Mutate valid records' matrix hash, artifact bytes/hash, safe path,
   canonical cell ID, selected solver control and dependency identity.
2. Mutate status relationships: `planned` with metrics, `blocked` with pass,
   observed with one side, measured without raw paired samples.
3. Mutate units, sample count/aggregates, false performance claims and
   publication approval without a review artifact.
4. Check that each mutation fails for the intended rule rather than an
   unrelated parser error; restore and revalidate the original positive
   fixture after each case.

### Deliverables

- Mutation verdict/code matrix and deterministic test suite.
- Evidence that valid JSON syntax does not bypass semantic checks.

### Completion Criteria

- Hash, status, path and canonical-ID mutations all fail closed.
- No mutation writes to selected manifests, source corpus or canonical
  benchmark output.

---

## Day 13: Mutation Closure And Maintainer Guide

**Title:** Mutation Closure And Maintainer Guide
**Theme:** Finish adversarial coverage and document the v1 contract.
**Time estimate:** 10 hours (221.5: 2 h; 221.6: 8 h)

### Tasks

1. Close uncovered mutation classes, review stable diagnostics and record
   any deliberately unsupported evidence shape.
2. Document v1 schema versioning, status vocabulary, canonical cell-ID
   bytes, rule codes and the `--schema`/`--input`/`--bundle-root` CLI.
3. Document raw-artifact retention, immutable attempt directories, hash
   verification, local-only evidence and the independent publication review.
4. State explicitly that validator success proves format/consistency only;
   Sprint 222 must supply the first real two-sided CHOLMOD correctness cell.

### Deliverables

- Closed mutation ledger and maintainer-facing v1 guidance.
- Reproducible commands for blueprint, blocked and synthetic fixture checks.

### Completion Criteria

- Item 221.5's 18-hour mutation proof is closed.
- Guidance does not project planning, blocked or synthetic records as
  observed external results.

---

## Day 14: Integrated Validation And Handoff

**Title:** Integrated Validation And Handoff
**Theme:** Close the validator prerequisite and hand off a bounded pilot.
**Time estimate:** 10 hours (221.6: 10 h)

### Tasks

1. Run validator self-test, positive/negative fixture and mutation suites,
   pilot preflight tests, existing corpus/selected-report regressions and
   `make docs-check`; record exact commands, counts, versions and outcomes.
2. If any `.c` or `.h` file changed during the sprint, run
   `make format && make lint && make test`; repair failures before declaring
   completion. Run any applicable optional CI/quality route changed by the
   integration.
3. Verify selected manifests, normalizer outputs, canonical benchmark rows,
   generated-API local-only routing and ignored `build/external-baseline/`
   artifacts did not acquire a new publication or result claim.
4. Reconcile every 221.1-221.6 deliverable and open risk in working notes;
   prepare a precise Sprint 222 handoff listing remaining execution,
   provenance and publication gates.

### Deliverables

- Passing local validation ledger and v1 schema/validator/preflight package.
- Sprint 222 handoff with no unsupported solver or performance claim.

### Completion Criteria

- All required checks pass; failed, skipped or unavailable checks block
  Sprint 221 completion.
- The optional pilot can proceed only with valid v1 preflight, and no solver
  or parity result is inferred from validation alone.
