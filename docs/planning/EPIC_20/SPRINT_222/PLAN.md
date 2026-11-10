# Sprint 222 Plan: One Two-Sided CHOLMOD Correctness Cell

**Sprint Duration:** 14 days
**Goal:** Execute and validate one independent, exact-scope SPD solve against
CHOLMOD, retain reproducible two-sided evidence, and make a separate rights and
publication decision.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds
12 hours; Epic 20 allows at most 168 hours per sprint. CI elapsed time is not
included in the estimates.

**Prerequisites:** Sprint 221's v1 schema, validator, self-test and fail-closed
pilot preflight; Sprint 215's pinned symmetric-four fixture, RHS, solver
operation, methodology and rights questions; a usable, version-verified
CHOLMOD build. Read the Sprint 221 handoff before attempting a solve.

**Primary scope:** One local correctness-only cell for
`tests/data/symmetric_4.mtx` using the frozen project Cholesky path and
SuiteSparse v7.12.2 / CHOLMOD 5.3.4. Preserve the original input, fresh
per-side state, complete runtime identities, raw streams, solutions, metrics,
attempt manifests and validation diagnostics. Keep generated bundles under
`build/external-baseline/` until an explicit publication review approves an
exact bundle and claim.

**Evidence rule:** A pilot process exit, parser pass, synthetic fixture or
matching hash alone is not an observed comparison. Both independently run
solvers must pass the Day 4 status, factor-control, finite, residual, oracle
and cross-solver checks before the validator may accept a correctness cell.
Record failed or blocked attempts verbatim. Do not substitute a later success
for a failed attempt or change the frozen input after inspecting outputs.

**Non-goals:** Timing, speedup or winner claims; hosted or cross-platform
parity; a new default SuiteSparse dependency; changes to selected benchmark
reports, canonical rows or the generated-API local-only rule. A rights denial
or unavailable baseline leaves the cell local and blocked for publication.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 222.1 Fixture And Rights | 20 h | Days 1-2: 10 + 10 |
| 222.2 Runtime Provenance | 28 h | Days 3-5: 10 + 10 + 8 |
| 222.3 Two-Sided Execution | 34 h | Days 5-9: 2 + 10 + 10 + 10 + 2 |
| 222.4 Validation And Repeat | 24 h | Days 9-11: 8 + 10 + 6 |
| 222.5 Claim Review | 18 h | Days 11-13: 4 + 10 + 4 |
| 222.6 Test Closure | 16 h | Days 13-14: 6 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Freeze the Selected Fixture

**Title:** Freeze the Selected Fixture
**Theme:** Establish the exact problem and custody of its bytes.
**Time estimate:** 10 hours (222.1: 10 h)

### Tasks

1. Read the Sprint 215 Day 3-5 contracts and Sprint 221 validator handoff;
   list the selected matrix, RHS, oracle, operation, precision, index ABI,
   solver APIs and factor controls.
2. Verify the checked-in Matrix Market bytes and SHA-256, dimensions,
   symmetry, expanded nonzero count and CSC conversion against independent
   fixture checks. Preserve the untouched source and conversion description.
3. Identify the exact v1 record fields and source artifacts needed before an
   observed status is possible; record any missing evidence as a gate.
4. Open `WORKING_NOTES.md` with source revision, input hashes, commands,
   artifact owners, failure log and the local-only boundary.

### Deliverables

- Pinned problem/fixture inventory with byte hashes and conversion checks.
- Initial working notes and a field-to-evidence map.

### Completion Criteria

- Matrix, RHS, oracle and solver selection match the frozen methodology.
- No result is described as observed before either solver runs.

---

## Day 2: Rights and Retention Gate

**Title:** Rights and Retention Gate
**Theme:** Decide which inputs and outputs may be retained or disclosed.
**Time estimate:** 10 hours (222.1: 10 h)

### Tasks

1. Verify matrix attribution, source terms and any redistribution obligations
   against the exact selected fixture bytes.
2. Inventory CHOLMOD, SuiteSparse module and transitive-library license terms
   for the intended binary, logs and metadata; distinguish local retention
   from public redistribution.
3. Record allowed bundle contents, redaction constraints, reviewer and
   unresolved rights questions. Keep approval pending if evidence is missing.
4. Freeze input/control hashes and a proposed append-only attempt directory
   layout; do not copy generated outputs into selected reports.

### Deliverables

- Written fixture attribution and rights/retention ledger.
- Frozen input/control manifest and proposed local bundle layout.

### Completion Criteria

- Item 222.1's 20-hour scope is accounted for with explicit rights status.
- Unresolved rights block publication, while local work proceeds only within
  the recorded retention scope.

---

## Day 3: Project Build Identity

**Title:** Project Build Identity
**Theme:** Capture a reproducible project-side runtime.
**Time estimate:** 10 hours (222.2: 10 h)

### Tasks

1. Capture full Git commit, clean/dirty state and exact patch/status bytes if
   dirty; stop if untracked build inputs cannot be reconstructed.
2. Record compiler executable/version, complete compile and link flags,
   optimization mode, `SPARSE_IDX_BITS`, mutex/OpenMP settings and solver API.
3. Build the selected project binary and library from the recorded inputs;
   retain and hash the exact executable and archive used by the run.
4. Cross-check project backend and ordering controls with the selected cell;
   record failed build attempts without replacing their logs.

### Deliverables

- Project build manifest, raw build logs and retained binary/archive hashes.
- Project provenance entries prepared for the v1 cell identity.

### Completion Criteria

- A reviewer can identify the exact project source, flags and binary bytes.
- An unknown build setting or missing dirty-tree patch blocks an observed pass.

---

## Day 4: CHOLMOD Runtime Identity

**Title:** CHOLMOD Runtime Identity
**Theme:** Verify the library actually loaded by the helper.
**Time estimate:** 10 hours (222.2: 10 h)

### Tasks

1. Verify SuiteSparse tag/commit, CHOLMOD version, acquisition source,
   candidate package prefix and expected module composition.
2. Capture the runtime-loaded CHOLMOD path and SHA-256 plus the loaded paths,
   versions, roles and hashes of required transitive libraries.
3. Record CHOLMOD build options, index ABI, simplicial/natural-ordering and
   postorder controls; distinguish requested controls from observed controls.
4. Reconcile discovered candidate library and runtime-loaded library. Stop
   and log a new identity if bytes or paths differ.

### Deliverables

- CHOLMOD package, loaded-library and dependency inventory with hashes.
- Baseline ABI and factor-control proof or a named blocker.

### Completion Criteria

- The actual loaded CHOLMOD library and dependencies are evidenced, not
  inferred from `pkg-config` or a linker search path alone.
- A missing or mismatched baseline component prevents solver promotion.

---

## Day 5: Host Freeze and Execution Setup

**Title:** Host Freeze and Execution Setup
**Theme:** Close provenance before independent solves.
**Time estimate:** 10 hours (222.2: 8 h; 222.3: 2 h)

### Tasks

1. Capture OS/build, architecture, CPU, memory, compiler runtime, host
   fingerprint, working directory, UTC time, full commands and effective
   BLAS/OpenMP thread counts.
2. Verify matrix and methodology hashes again, finalize the v1 cell key and
   ensure all required project and baseline provenance artifacts are retained.
3. Run the validator self-test and pilot preflight before build/solve work;
   an unavailable CHOLMOD or failed preflight remains a blocked attempt.
4. Prepare two fresh, isolated input/state instances and raw output paths so
   one side cannot reuse the other's factor or solver output.

### Deliverables

- Complete host/execution manifest and preflight outcome.
- Independent project/baseline run recipe with append-only attempt paths.

### Completion Criteria

- Item 222.2's 28-hour provenance scope is complete or explicitly blocked.
- Both solver commands refer to the same frozen problem but separate state.

---

## Day 6: Execute the Project Side

**Title:** Execute the Project Side
**Theme:** Capture one genuine project solve without promotion by assertion.
**Time estimate:** 10 hours (222.3: 10 h)

### Tasks

1. Run the recorded project command on a fresh copy of the frozen input and
   preserve raw stdout/stderr bytes, exit code, command, timestamps and logs.
2. Record load/factor/solve statuses, effective factor path, raw solution and
   any warning or allocation failure without sanitizing the raw streams.
3. Independently recompute finite residual, backward error and oracle error
   from the untouched full matrix and original RHS.
4. Match retained executable/library bytes and thread controls to the frozen
   manifest; mark the attempt failed or blocked if any proof is absent.

### Deliverables

- Hashed project raw output, solution and independently checked metrics.
- Explicit project-side disposition with reasons for failures.

### Completion Criteria

- A project `pass` is supported by both raw status and numerical gates.
- Failed output remains available as its own attempt, without substitution.

---

## Day 7: Execute the Baseline Side

**Title:** Execute the Baseline Side
**Theme:** Obtain an independently evidenced CHOLMOD solve.
**Time estimate:** 10 hours (222.3: 10 h)

### Tasks

1. Run the baseline command from a fresh input/state instance with the same
   matrix, RHS and frozen host/thread controls; retain raw streams and exit
   status separately from the project run.
2. Record analyze, factor, solve and CHOLMOD statuses, factor minor, ordering,
   storage/supernodal and CSC conversion diagnostics, and raw solution.
3. Verify runtime-loaded library path/hash against Day 4; independently
   recompute finite residual, backward error and oracle error.
4. Preserve every failed or timed-out attempt and its reason. Never promote a
   baseline pass from a package version string or synthetic fixture.

### Deliverables

- Hashed baseline raw output, solution and independently checked metrics.
- Baseline disposition with runtime and factor-control evidence.

### Completion Criteria

- CHOLMOD success includes both factor-time and solve-time status checks.
- The baseline run is independent of the project factor and output state.

---

## Day 8: Two-Sided Numerical Reconciliation

**Title:** Two-Sided Numerical Reconciliation
**Theme:** Apply the frozen Day 4 correctness gates to both sides.
**Time estimate:** 10 hours (222.3: 10 h)

### Tasks

1. Recompute both solutions' residual infinity norm, backward error and
   oracle error from the untouched full matrix; inspect finite intermediate
   products and preserve full-precision raw values.
2. Apply the fixed, input-derived Day 4 tolerances and compare independently
   passing solutions using the declared cross-solver consistency bound.
3. Confirm input bytes, operation, precision, ABI, factor controls and host
   identity did not drift between sides; separate any mismatch into a new
   cell or `non_comparable` attempt.
4. Write an evidence table linking each assertion to raw bytes and hashes,
   with no timing or superiority interpretation.

### Deliverables

- Two-sided numerical and control-gate ledger.
- Explicit correctness disposition and unresolved mismatch list.

### Completion Criteria

- Each side passes its own checks before a paired consistency result exists.
- Any failed gate blocks a two-sided correctness claim.

---

## Day 9: Build and Validate the First Cell

**Title:** Build and Validate the First Cell
**Theme:** Convert raw observations into a checked v1 record.
**Time estimate:** 10 hours (222.3: 2 h; 222.4: 8 h)

### Tasks

1. Finish the raw two-side disposition, list every retained artifact and
   SHA-256, and verify no output was rewritten after observation.
2. Populate the v1 record with observed statuses, solution metrics,
   methodology hashes, host/runtime identity and exact cell ID.
3. Run the v1 CLI with `--schema`, `--input` and `--bundle-root`; resolve
   discrepancies by correcting evidence or recording failure, not by
   weakening the schema or fixed numerical thresholds.
4. Check generated files remain under the local bundle and outside selected
   report manifests and canonical benchmark rows.

### Deliverables

- First complete v1 record, validator output and byte-backed artifact index.
- Failed validation attempts retained with rule codes and pointers.

### Completion Criteria

- Item 222.3's 34-hour execution scope is accounted for.
- A `correctness_observed` label appears only if both independent solves and
  complete provenance pass the validator.

---

## Day 10: Repeat from Frozen Inputs

**Title:** Repeat from Frozen Inputs
**Theme:** Prove the cell can be rerun without changing its identity.
**Time estimate:** 10 hours (222.4: 10 h)

### Tasks

1. Start a new append-only attempt directory with the same frozen input,
   binary and methodology controls; capture fresh commands, streams and
   output hashes for both sides.
2. Recheck loaded library/dependency hashes, host/thread settings and factor
   controls before comparing the repeat to the first attempt.
3. Recompute Day 4 numerical gates and canonical cell ID; explain any
   differing metric, path, environment or identity rather than overwriting
   an earlier record.
4. Validate the repeat v1 record independently and retain both validator
   transcripts, including any failures.

### Deliverables

- Separate repeat attempt bundle and v1 validation transcript.
- First-versus-repeat identity and numerical comparison ledger.

### Completion Criteria

- The repeat uses frozen inputs and is independently valid or explicitly
  failed/blocked; no attempt is silently replaced.
- Any changed identity is recorded as a different cell, not presented as a
  repeat of the original.

---

## Day 11: Close Evidence and Prepare Review

**Title:** Close Evidence and Prepare Review
**Theme:** Finish validation and assemble an exact-bundle review packet.
**Time estimate:** 10 hours (222.4: 6 h; 222.5: 4 h)

### Tasks

1. Audit first and repeat bundles for complete raw streams, statuses,
   vectors, hashes, failure attempts, binary/dependency identity and stable
   validator diagnostics.
2. Run adversarial mutations for missing raw sides, changed loaded library,
   bad ABI, nonfinite metrics, stale cell ID and unsupported claim scope.
3. Freeze the exact bundle digest/index for review and distinguish local
   validation from independent publication approval.
4. Draft a one-cell correctness-only claim and a no-publication alternative;
   identify the reviewer, rights evidence and precise artifacts at issue.

### Deliverables

- Closed validation/retention ledger and mutation transcript.
- Exact-bundle attribution, rights and wording review packet.

### Completion Criteria

- Item 222.4's 24-hour validation/repeat scope is closed with first and
  repeat outcomes plainly labeled.
- The review packet makes no speed, winner, hosted or broad parity claim.

---

## Day 12: Independent Rights and Claim Decision

**Title:** Independent Rights and Claim Decision
**Theme:** Keep publication rights separate from numerical validity.
**Time estimate:** 10 hours (222.5: 10 h)

### Tasks

1. Obtain an explicit attribution, license/retention and exact-wording
   decision for the frozen bundle from the responsible reviewer.
2. Record approval, denial or pending status with reviewer identity,
   decision scope, retained evidence and the hashed review artifact.
3. Recheck that an approved statement names only this local correctness
   cell; deny any inferred speedup, state-of-the-art or platform claim.
4. If review is unavailable or rights are denied, mark public-claim work
   blocked while preserving permitted local evidence and the reason.

### Deliverables

- Written exact-bundle rights/attribution and claim decision.
- Approved wording or a documented no-publication disposition.

### Completion Criteria

- Numerical validity alone does not set `publication_review=approved`.
- Any approval cites a separate hashed review artifact and approved matrix
  attribution; unresolved review remains pending.

---

## Day 13: Claim-Scope Docs and Test Matrix

**Title:** Claim-Scope Docs and Test Matrix
**Theme:** Apply the decision and begin final regression closure.
**Time estimate:** 10 hours (222.5: 4 h; 222.6: 6 h)

### Tasks

1. Update public documentation only to the exact approved scope, or record
   the blocked/private disposition without exposing restricted bundle data.
2. Reconcile the first/repeat records, review artifact and claim wording;
   keep the Sprint 215 blueprint distinct from observed evidence.
3. Run focused pilot and validator positive/negative suites, rights/claim
   mutations and affected documentation checks; record exact commands and
   outcomes in working notes.
4. Check that selected reports, normalization and generated API docs retain
   their local-only/publication boundaries.

### Deliverables

- Claim-scope documentation or explicit private/blocked decision.
- Focused regression and documentation check ledger.

### Completion Criteria

- Item 222.5's 18-hour claim review is closed or explicitly pending.
- No public result appears without an approved exact-bundle review.

---

## Day 14: Quality Gates and Handoff

**Title:** Quality Gates and Handoff
**Theme:** Finish the correctness cell without overstating its reach.
**Time estimate:** 10 hours (222.6: 10 h)

### Tasks

1. Rerun all affected pilot, validator, CSC, installation and documentation
   checks; include mutation verdicts, attempt hashes, environment and
   first/repeat outcomes in the final ledger.
2. If any `.c` or `.h` file changed, run `make format && make lint && make test`
   and repair all failures before completion. Run applicable CMake/CI routes
   touched by the sprint and record platform-specific blockers honestly.
3. Verify generated bundles stay local, no timing rows or broad external
   claims were introduced, and all accepted artifacts match their hashes.
4. Reconcile items 222.1-222.6, prepare a clear Sprint 223 handoff and list
   remaining work separately from this cell's evidence status.

### Deliverables

- Final test/quality ledger, validated cell and repeat bundle or explicit
  failed/blocked attempt records.
- Sprint 223 handoff with rights, retention and claim boundaries.

### Completion Criteria

- All required checks pass before an observed/approved result is declared;
  any failed or unavailable gate is recorded as blocked or pending.
- Item 222.6's 16-hour closure is complete, and the 140-hour sprint plan
  does not imply timing, portable parity or state-of-the-art status.
