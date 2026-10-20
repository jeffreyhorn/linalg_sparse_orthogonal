# Sprint 217 Plan: Reliable Norms and ILUT Numerical Diagnostics

**Sprint Duration:** 14 days
**Goal:** Close the shared vector-norm overflow and non-finite behavior gap,
and turn the `west0067` ILUT test into a meaningful numerical regression.

**Time budget:** 10 hours per day, 140 hours total. No day exceeds 12 hours;
the Epic 20 capacity limit is 168 hours per sprint.

**Primary scope:** `vec_norm2`, `vec_norminf`, their iterative-solver call
sites, the selected ILUT solve/test path, focused extreme-scale regressions,
and exact public documentation. The implementation must preserve the current
NULL/zero-length vector contract unless an explicit compatibility decision is
recorded.

**Evidence rule:** A passing test that only checks `norm > 0` cannot establish
a finite solution. Record whether an infinite norm comes from norm arithmetic
or from non-finite solution entries; require an independent residual or
backward-error gate before calling the selected ILUT result valid. Treat
failed, skipped, or unrun checks as blockers, not as passing evidence.

**Non-goals:** This sprint does not claim all-solver numerical conformance,
external-library parity, portable performance, broad non-finite input support
for every API, or state-of-the-art status.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 217.1 Contract And Call-Site Audit | 20 h | Days 1-2: 10 + 10 |
| 217.2 Scaled Norm Implementation | 28 h | Days 5-8: 6 + 10 + 10 + 2 |
| 217.3 ILUT And Iterative Hardening | 28 h | Days 8-10: 8 + 10 + 10 |
| 217.4 Extreme-Scale Regression | 24 h | Days 3-5: 10 + 10 + 4 |
| 217.5 Public Contract Update | 20 h | Days 11-12: 10 + 10 |
| 217.6 Validation And Evidence | 20 h | Days 13-14: 10 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Norm Consumer Inventory

**Title:** Norm Consumer Inventory
**Theme:** Identify every selected arithmetic and convergence dependency.
**Time estimate:** 10 hours (217.1: 10 h)

### Tasks

1. Read `src/sparse_vector.c`, `include/sparse_vector.h`, and the existing
   vector tests; record current NULL, nonpositive-length, finite, infinity,
   and NaN behavior without assuming the behavior is intentional.
2. Enumerate all `vec_norm2` and `vec_norminf` callers in direct, iterative,
   eigensolver, test, and example code; identify which callers use the result
   for convergence, diagnostics, or only reporting.
3. Trace CG, GMRES, MINRES, and BiCGSTAB residual normalization, initial
   convergence, and final status paths; mark division-by-zero and non-finite
   comparisons.
4. Establish a baseline with focused vector, ILUT, and iterative test runs;
   retain command, exit status, toolchain, and the `west0067` norm output.
5. Start `WORKING_NOTES.md` with the consumer map, decision log, test matrix,
   and evidence ledger.

### Deliverables

- Norm call-site and convergence-dependency inventory.
- Baseline results and working-notes scaffold.

### Completion Criteria

- Every selected norm consumer has a location, purpose, and risk classification.
- Baseline output is recorded without labeling `norm=inf` a solver defect.
- Current NULL/empty-vector behavior is explicit.

---

## Day 2: Numerical Behavior Contract

**Title:** Numerical Behavior Contract
**Theme:** Decide exact results and error behavior before editing kernels.
**Time estimate:** 10 hours (217.1: 10 h)

### Tasks

1. Define the expected 2-norm and infinity-norm results for finite,
   zero-length, NULL, infinity, and NaN inputs, including mixed NaN/infinity
   vectors and representable versus unrepresentable mathematical norms.
2. Choose a scaled sum-of-squares algorithm and rounding/accuracy tolerances
   suitable for the shipped `double` scalar contract; specify when an actual
   result may be infinite.
3. Define how each selected solver will report invalid initial or updated
   residual norms using existing error/status APIs, without changing unrelated
   solver contracts.
4. Define the `west0067` acceptance calculation: finite entries, status,
   factor-product residual `L(Uz) - P r` using the effective row permutation,
   a scale-aware error threshold, and a reference computed independently
   from the solve implementation. ILUT is approximate, so do not require
   `A*z = r` as an exact-solve condition.
5. Record compatibility decisions and acceptance cases in the working notes.

### Deliverables

- Norm and convergence behavior matrix.
- Selected ILUT numerical acceptance specification.

### Completion Criteria

- Each unusual input has one documented expected result or status.
- The ILUT test will be able to distinguish bad norm arithmetic from bad
  solution data.
- No change to public error semantics is assumed without a recorded decision.

---

## Day 3: Finite Extreme-Scale Fixtures

**Title:** Finite Extreme-Scale Fixtures
**Theme:** Expose overflow and underflow in the current norm algorithm.
**Time estimate:** 10 hours (217.4: 10 h)

### Tasks

1. Add focused vector cases with finite components around `1e200` and
   `1e-200` whose mathematical norms are representable, plus mixed-scale and
   cancellation-adjacent values.
2. Use independently derived expectations, such as `hypot` for two-element
   cases or analytically known powers of two; avoid calculating expectations
   with the same square-and-sum algorithm under test.
3. Check relative error, not just nonzero or sign, and include zero and
   singleton controls that preserve the existing API contract.
4. Run the new cases against the pre-change implementation, recording expected
   failures and the exact observed values in the evidence ledger.

### Deliverables

- Reproducible finite extreme-scale norm fixtures.
- Before-fix failing-case record.

### Completion Criteria

- At least one representable finite norm demonstrably overflows the old
  implementation, and one tiny nonzero norm is exercised.
- Test expectations are independent and deterministic.
- Baseline failures are retained rather than waived.

---

## Day 4: Non-Finite And Solver Fixtures

**Title:** Non-Finite And Solver Fixtures
**Theme:** Make invalid data and false convergence observable.
**Time estimate:** 10 hours (217.4: 10 h)

### Tasks

1. Add vector tests for NaN, positive/negative infinity, mixed finite and
   non-finite entries, and `vec_norminf`'s NaN comparison behavior.
2. Add deterministic seeded small-system cases for selected CG, GMRES,
   MINRES, and BiCGSTAB norm/convergence paths; include finite control cases
   and invalid residual or right-hand-side cases.
3. Add an ILUT diagnostic fixture that records solution-entry finiteness and
   the chosen independent residual metric before the existing weak norm
   assertion is replaced.
4. Register new tests in the existing Make/CTest test surfaces where needed,
   and retain their pre-fix outcomes.

### Deliverables

- Non-finite vector and seeded iterative regression fixtures.
- Baseline ILUT entry/residual diagnosis.

### Completion Criteria

- The suite contains a case for each non-finite behavior chosen on Day 2.
- Seeded tests expose an invalid-norm path without accepting a false success.
- Fixture registration and test ownership are clear.

---

## Day 5: Fixture Closure And Algorithm Design

**Title:** Fixture Closure And Algorithm Design
**Theme:** Finish red tests and finalize the stable norm implementation shape.
**Time estimate:** 10 hours (217.4: 4 h; 217.2: 6 h)

### Tasks

1. Complete the cross-product of finite extreme-scale, non-finite, and seeded
   solver cases; remove duplicate or implementation-coupled assertions.
2. Review current failing cases against the Day 2 contract and revise only
   incorrect expectations, not cases that expose the defect.
3. Design the scaled accumulator's zero, subnormal, large, infinity, and NaN
   branches and show that intermediate square/sum operations stay bounded.
4. Decide whether a shared internal helper is warranted by actual duplicate
   call-site logic; keep the public vector API unchanged.
5. Record the final test/implementation matrix and scope for Day 6.

### Deliverables

- Complete pre-change regression set with expected failure list.
- Reviewed scaled-norm implementation design.

### Completion Criteria

- Item 217.4's 24-hour fixture allocation is complete.
- Every planned implementation branch has a corresponding fixture or a
  documented reason it cannot be exercised.
- No failing test is hidden by a blanket skip.

---

## Day 6: Scaled 2-Norm Implementation

**Title:** Scaled 2-Norm Implementation
**Theme:** Compute representable Euclidean norms without intermediate overflow.
**Time estimate:** 10 hours (217.2: 10 h)

### Tasks

1. Implement the reviewed scale/sumsq norm algorithm in `vec_norm2`, keeping
   the existing NULL and nonpositive-length behavior.
2. Make NaN/infinity handling follow the Day 2 precedence decision rather
   than relying on incidental floating-point comparisons.
3. Check subnormal, zero, singleton, mixed-magnitude, and very large finite
   cases against the independent expectations.
4. Run focused vector tests and inspect compiler warnings, sanitizer output
   if practical, and numerical tolerances before changing solver call sites.

### Deliverables

- Scaled `vec_norm2` implementation.
- Focused passing finite-scale tests and any remaining failure list.

### Completion Criteria

- A representable large norm no longer becomes infinity merely because a
  component was squared.
- Tiny finite norms are not lost to unnecessary underflow.
- The existing NULL/empty-vector tests still pass.

---

## Day 7: Infinity Norm And Norm Integration

**Title:** Infinity Norm And Norm Integration
**Theme:** Complete deliberate non-finite behavior for both helpers.
**Time estimate:** 10 hours (217.2: 10 h)

### Tasks

1. Update `vec_norminf` so NaN does not silently disappear behind a false
   comparison; preserve finite maximum and infinity behavior.
2. Check both norm helpers on every Day 3-5 fixture and evaluate scale-aware
   error at normal, subnormal, and near-maximum magnitudes.
3. Inspect callers that compare results with `<=`, `>`, or `==` for behavior
   changes and note any required solver guards for Days 8-10.
4. Run the complete vector suite plus focused call-site smoke tests; record
   remaining failures without broadening the sprint to unrelated vector ops.

### Deliverables

- Explicit `vec_norminf` non-finite handling.
- Norm-helper regression record and solver guard checklist.

### Completion Criteria

- Every Day 2 norm result is implemented or has a named blocker.
- No finite control regresses.
- The solver work list identifies concrete branches, not a vague sweep.

---

## Day 8: Norm Finalization And ILUT Diagnosis

**Title:** Norm Finalization And ILUT Diagnosis
**Theme:** Freeze shared helper behavior and begin the selected solve proof.
**Time estimate:** 10 hours (217.2: 2 h; 217.3: 8 h)

### Tasks

1. Finish the norm implementation review, rerun vector tests, and close item
   217.2's remaining warnings or precision issues.
2. Capture the actual `west0067` ILUT solution entries, factor/solve status,
   computed norm, and independent residual/backward error under the new norm.
3. Determine whether the earlier `norm=inf` was caused by norm overflow,
   non-finite solution entries, or a mathematical norm outside `double` range.
4. Replace the old `znorm > 0` test with finite/status/scale-aware numerical
   assertions matching the Day 2 contract; keep diagnosis in working notes.

### Deliverables

- Reviewed norm helper implementation.
- Explicit cause of the `west0067` infinity observation.
- Initial strengthened ILUT assertion.

### Completion Criteria

- Item 217.2's 28-hour allocation is complete with vector regressions green.
- The ILUT case cannot pass on an unexplained infinite norm.
- Any genuine solver failure is recorded as a defect, not relabeled as a test
  expectation.

---

## Day 9: ILUT Validity And Failure Behavior

**Title:** ILUT Validity And Failure Behavior
**Theme:** Close the selected factor/solve numerical contract.
**Time estimate:** 10 hours (217.3: 10 h)

### Tasks

1. Complete `west0067` solution and factor-product residual assertions using
   an independent multiplication/reference route and the effective row
   permutation, with justified tolerance; do not treat the inexact ILUT
   preconditioner as an exact factorization of the original matrix.
2. If ILUT returns success with non-finite output, identify the first invalid
   factor or substitution value and make the selected path return a suitable
   existing numeric failure status or produce a valid finite result.
3. Add a negative fixture for non-finite factor/solve data and a finite
   positive control; verify cleanup and output-status behavior on failure.
4. Run the focused ILUT and GMRES-preconditioned tests; retain raw outputs,
   error metrics, and any blocked cases.

### Deliverables

- ILUT test that checks both status and numerical result.
- Fix or explicit numeric failure for any confirmed non-finite selected path.

### Completion Criteria

- `west0067` passes only with a justified finite numerical result, or the
  path correctly reports a numeric failure and the test reflects that
  documented contract.
- No false positive remains from the old `norm > 0` assertion.
- Selected preconditioned GMRES behavior is covered.

---

## Day 10: Iterative Convergence Hardening

**Title:** Iterative Convergence Hardening
**Theme:** Prevent invalid norms from becoming successful solver statuses.
**Time estimate:** 10 hours (217.3: 10 h)

### Tasks

1. Review initial, updated, preconditioned, and final true-residual checks in
   CG, GMRES, MINRES, and BiCGSTAB against Day 2's invalid-norm contract.
2. Add the narrow guards required before relative-residual division and
   convergence comparisons; preserve valid zero-right-hand-side behavior.
3. Exercise seeded finite and invalid cases for each selected family, checking
   status, finite outputs where promised, and no false `SPARSE_OK`.
4. Review error cleanup and convergence result fields on numeric failure,
   rerun focused iterative tests, and close item 217.3's discrepancy ledger.

### Deliverables

- Finite-aware convergence handling in selected iterative paths.
- Passing family-local regressions and failure-status record.

### Completion Criteria

- Invalid norm or residual data cannot be reported as convergence in the
  selected paths.
- Finite, zero-RHS, and preconditioned control cases retain prior behavior.
- Item 217.3's 28-hour allocation is complete or has a named blocking defect.

---

## Day 11: Public Vector And Solver Contract

**Title:** Public Vector And Solver Contract
**Theme:** Make numerical behavior discoverable at the API boundary.
**Time estimate:** 10 hours (217.5: 10 h)

### Tasks

1. Update `include/sparse_vector.h` with scaled-norm, representability,
   NULL/empty, NaN, and infinity semantics actually implemented.
2. Update affected iterative and ILUT public-header contracts for invalid
   numeric inputs, failure statuses, and result-field validity on failure.
3. Check wording against actual tests and code; avoid claiming universal
   non-finite behavior or independent solver parity.
4. Run focused header/API documentation checks and inspect generated local
   reference pages for the touched declarations.

### Deliverables

- Precise public vector and selected solver header documentation.
- Header-to-regression traceability notes.

### Completion Criteria

- Public return/status descriptions match tested behavior.
- Ownership and historical NULL/empty contracts remain clear.
- No unsupported all-solver claim appears in the API text.

---

## Day 12: User Guidance And Claim Calibration

**Title:** User Guidance And Claim Calibration
**Theme:** Explain the selected fix without widening numerical claims.
**Time estimate:** 10 hours (217.5: 10 h)

### Tasks

1. Update the smallest relevant README, solver-selection, API-reference, or
   maintainer pages with the new norm and selected ILUT diagnostic contract.
2. Distinguish stable arithmetic from broad ill-conditioned or all-family
   convergence guarantees; retain the Epic 20 state-of-the-art non-claim.
3. Add or update focused documentation guards only where a repeatable
   overclaim could otherwise be introduced; avoid brittle prose matching.
4. Verify links, local generated API coverage, and the working-notes claim
   matrix; freeze item 217.5's documentation list.

### Deliverables

- Updated user/maintainer guidance and claim-boundary record.
- Documentation validation results.

### Completion Criteria

- User-facing claims describe the exact selected implementation and tests.
- Generated docs and local links resolve.
- Item 217.5's 20-hour allocation is complete.

---

## Day 13: Focused And Instrumented Validation

**Title:** Focused And Instrumented Validation
**Theme:** Verify the changed numerical owner under stress and sanitizers.
**Time estimate:** 10 hours (217.6: 10 h)

### Tasks

1. Run focused vector, ILUT, and iterative tests with the final fixtures;
   retain commands, exit codes, solver metrics, and skip reasons.
2. Run the appropriate sanitizer target(s) on changed C paths and inspect
   numeric-error cleanup, input preservation, and failure output behavior.
3. Run affected CMake/CTest registrations and selected corpus tests where
   available; compare with the Day 1 baseline.
4. Record any failures and fix the cause before declaring validation done.
   Reset tree-mutating sanitizer build state before Day 14's normal build.

### Deliverables

- Focused and instrumented validation ledger with retained outputs.
- Clean normal-build precondition for the final quality chain.

### Completion Criteria

- All focused, sanitizer, and affected CMake checks pass, or a named blocker
  prevents completion.
- The earlier infinite-norm observation has an evidenced disposition.
- Failed or skipped checks are not silently counted as passes.

---

## Day 14: Full Quality Gate And Sprint Closeout

**Title:** Full Quality Gate And Sprint Closeout
**Theme:** Finish only with passing source checks and a scoped claim decision.
**Time estimate:** 10 hours (217.6: 10 h)

### Tasks

1. From a normal, non-sanitized build state, run the required sequence
   `make format && make lint && make test`; inspect the formatter's diff and
   rerun affected focused cases if it changes source text.
2. Run relevant documentation/API checks after final wording and verify
   changed-file/source-list hygiene.
3. Reconcile all six item estimates and evidence links in `WORKING_NOTES.md`;
   publish a Day 14 validation/claim record and identify any residual issue.
4. Review final `git diff`, exact passing commands, platform scope, skip
   statuses, and failure semantics; do not mark the sprint complete unless
   all required checks pass.

### Deliverables

- Passing `make format && make lint && make test` record and final focused
  numerical evidence.
- Sprint closeout with exact scope, non-claims, and unresolved residuals.

### Completion Criteria

- The full quality chain and affected documentation checks pass on the final
  files.
- Finite representable norms remain finite, invalid selected solver data
  cannot pass as convergence, and `west0067` has a meaningful numerical gate.
- Sprint status distinguishes a completed selected owner from any broader
  numerical-conformance work still open.
