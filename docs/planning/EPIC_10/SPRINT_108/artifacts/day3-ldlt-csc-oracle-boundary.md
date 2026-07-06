# Day 3 LDLT CSC Oracle Boundary

## Purpose

Day 3 selects at most one additional `tests/test_ldlt_csc.c` proof helper for
Sprint 108. The selected candidate must be narrower than the surrounding proof
logic, must not repeat Sprint 107's row-adjacency helper work, and must keep
direct CSC solve/oracle intent visible at the edited call sites.

## Inspected Proof Areas

### Factor-state equality

Candidate area:

- `ldlt_csc_factor_state_matches`
- call sites comparing scalar, batched, heuristic, and analysis-backed factor
  states

Decision: do not move or split in Sprint 108 Day 4.

Rationale:

- the helper already has specific mismatch messages for `L`, `D`,
  `D_offdiag`, and `pivot_size`;
- the surrounding tests deliberately distinguish structural equality from
  residual correctness;
- further splitting would add indirection without reducing repeated proof
  setup.

### External dense-reference oracle

Candidate area:

- `read_ldlt_external_dense_reference_solution`
- `assert_ldlt_external_dense_reference`
- Sprint 98 and Sprint 102 external dense-reference tests

Decision: defer.

Rationale:

- this helper includes skip behavior, subprocess parsing, dense reference
  comparison, solve reconstruction, and residual proof in one oracle lane;
- moving pieces from it risks hiding the reason a test skipped versus failed;
- it is better treated as a future oracle-lane review, not a Day 4 narrow
  helper.

### Row-adjacency exact-set proof

Candidate area:

- `assert_row_adj_matches_l_pattern`
- `test_row_adj_matches_reference`

Decision: explicitly exclude from Sprint 108 Day 4.

Rationale:

- Sprint 107 already extracted and refined this proof helper;
- repeating that cleanup would duplicate completed work;
- additional movement would not reduce the remaining broad direct-solver
  proof/oracle debt.

### Supernodal with-analysis residual assertions

Candidate area:

- existing `s20_solve_residual`
- repeated assertions such as `ASSERT_TRUE(s20_solve_residual(F2, A_perm) <
  1e-10)`
- KKT and random-indefinite supernodal with-analysis tests

Decision: selected for Day 4 follow-through.

Rationale:

- the existing residual computation is already factored into a local helper;
- multiple call sites still repeat the residual-threshold assertion without a
  labeled failure message;
- a small assertion helper can improve failure localization while preserving
  visible direct CSC solve intent at call sites.

## Selected Day 4 Candidate

Add a local helper near `s20_solve_residual`:

```c
static void assert_s20_solve_residual_below(const char *label, LdltCsc *F,
                                            const SparseMatrix *A_ref,
                                            double tol);
```

Expected behavior:

- call `s20_solve_residual(F, A_ref)`;
- print or fail with the provided label, actual residual, and threshold;
- assert that the residual is below the requested tolerance;
- avoid changing `s20_solve_residual` semantics;
- avoid changing fixture construction, factorization, or solve behavior.

Approved call sites:

- `test_s20_supernodal_with_analysis_kkt_5x5`
- `test_s20_supernodal_with_analysis_kkt_10x10`
- `test_s20_supernodal_with_analysis_random_indefinite_30x30`
- `test_s20_supernodal_heuristic_vs_with_analysis_residuals`

Non-approved call sites:

- external dense-reference helper assertions;
- row-adjacency exact-set tests;
- factor-state equality checks;
- unrelated LDLT solve and dispatch tests below the Sprint 20 section.

## Call-Site Readability Rules

The Day 4 edit must preserve:

- fixture identity at the call site (`build_kkt_5x5`, `build_kkt_10x10`,
  random-indefinite fixture);
- factorization path at the call site (`s20_two_pass_indefinite_factor` or
  `ldlt_csc_from_sparse_with_analysis`);
- validation of the factor object through `ldlt_csc_validate`;
- explicit tolerance value (`1e-10`) at the assertion call.

The helper may hide only the repeated "compute residual and compare with
threshold" mechanics.

## Assertion Specificity

Failure output should identify:

- the labeled proof case;
- actual residual;
- expected threshold.

The helper must not replace specific factor-state mismatch messages from
`ldlt_csc_factor_state_matches`.

## Placement and Target Rules

- Place the helper in `tests/test_ldlt_csc.c` near `s20_solve_residual`.
- Do not create a new test helper target.
- Do not move code to a shared header.
- Do not change CTest registration.
- Do not touch public headers or implementation sources.

## Focused Validation Plan

If Day 4 changes `tests/test_ldlt_csc.c`, run:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
make format && make lint && make test
git diff --check
```

Because a `.c` file would be modified, the full quality gate is required
before committing or proceeding.

## Day 3 Decision

Proceed to Day 4 with exactly one candidate:
`assert_s20_solve_residual_below`. All broader LDLT CSC oracle movement remains
deferred until a future sprint creates a dedicated oracle-lane boundary.

## Completion Criteria Status

- The selected helper is narrow and reviewable.
- Direct-solver proof intent remains visible at approved call sites.
- Validation commands are known before edits begin.

