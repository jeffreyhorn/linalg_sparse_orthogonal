# Sprint 200 Working Notes: Additional Allocation-Failure Owner Proof

## Sprint Goal

Prove one additional selected allocation-failure owner with deterministic
cleanup, stale-output suppression, and retry evidence.

## Day 1: Candidate Intake

### Scope Trace

| Epic item | Day 1 intake interpretation | Initial artifact |
| --- | --- | --- |
| 200.1 Owner Selection | Rank candidate allocation-failure owners and select exactly one on Day 2. | Candidate ledger and Day 2 scoring inputs. |
| 200.2 Invariant Record | Prepare cleanup, publication, stale-output, retry, caller-owned input, and unsupported-breadth categories before code edits. | Invariant checklist placeholder. |
| 200.3 Harness Integration | Reuse the deterministic allocation hook where possible; extend only if the selected owner cannot be reached. | Harness reachability notes. |
| 200.4 Regression Tests | Plan failed-allocation, cleanup, stale-output, retry, and caller-owned input tests for the selected owner. | Test requirement matrix. |
| 200.5 Focused Gate | Add a selected-owner gate and registration guard only after the owner is chosen. | Gate inventory and gap record. |
| 200.6 Docs And Validation | Keep reliability wording selected-owner-only and run focused, source-list, docs, format, lint, and test checks once implementation begins. | Validation matrix and claim-boundary notes. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Sprint 200 is a 168-hour sprint to prove one additional selected allocation-failure owner, not broad allocation reliability. |
| `docs/planning/EPIC_17/SPRINT_195/WORKING_NOTES.md` | Sprint 195 provides the pattern: candidate scoring, invariant record, narrow harness integration, regression tests, focused gate, claim docs, and full validation. |
| `docs/planning/EPIC_17/SPRINT_195/RETROSPECTIVE.md` | Remaining reliability candidates include `sparse_symbolic_lu()`, `sparse_analyze()`, helper-level etree/postorder/colcount paths, direct solvers, matrix construction, allocator/platform behavior, and CI ownership. |
| `Makefile` | Existing focused allocation-failure gates are `iterative-allocation-failure-gate`, `matmul-allocation-failure-gate`, and `symbolic-allocation-failure-gate`. |
| `tests/test_matmul_allocation_failure_gate_registration.py` | Existing registration guard pattern prevents focused matmul allocation-failure tests from silently dropping out. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Existing registration guard pattern prevents selected symbolic Cholesky allocation-failure tests from silently dropping out. |
| `src/sparse_alloc_internal.c` and `src/sparse_alloc_internal.h` | Private deterministic allocation hooks remain the preferred proof mechanism for wrapped allocation sites. |

### Existing Allocation-Failure Gates

| Gate | Owner surface | Evidence shape | Sprint 200 interpretation |
| --- | --- | --- | --- |
| `make iterative-allocation-failure-gate` | Iterative repeated-run handle allocation behavior. | Focused test binary plus deterministic allocation-failure cases. | Already covered; do not select as the additional owner. |
| `make matmul-allocation-failure-gate` | `sparse_matmul()` workspace allocation behavior. | Focused gate, Python registration guard, stale-output and retry coverage. | Existing proof model for stale-output and retry assertions. |
| `make symbolic-allocation-failure-gate` | Selected `sparse_symbolic_cholesky()` output allocation behavior. | Focused `test_etree` gate, Python registration guard, cleanup/stale-output/retry tests. | Primary reusable pattern; do not re-select symbolic Cholesky. |

### Current Allocation and Cleanup Surface Scan

The scan counted source lines matching allocation wrappers or direct
`malloc`/`calloc`/`realloc`, then cleanup/failure indicators such as
`cleanup:`, `goto cleanup`, `SPARSE_ERR_ALLOC`, `return NULL`, and `free(`.
The counts are ranking signals only, not proof of owner boundaries.

| Candidate source | Allocation signal | Cleanup/failure signal | Day 1 interpretation |
| --- | ---: | ---: | --- |
| `src/sparse_ldlt_csc.c` | 58 | 118 | Highest allocation signal and high direct-solver value; likely too broad unless a narrow entry point is isolated. |
| `src/sparse_lu_csr.c` | 37 | 145 | Strong public direct-solver candidate with dense cleanup paths; stale-output and retry semantics need tracing. |
| `src/sparse_qr.c` | 33 | 149 | High cleanup density, but recent QR review surface makes this risky unless a small owner can be isolated. |
| `src/sparse_etree.c` | 30 | 145 | Strong symbolic owner candidate; Sprint 195 already proved symbolic Cholesky, leaving symbolic LU and helper paths. |
| `src/sparse_ldlt.c` | 29 | 135 | Public LDLT owner candidate with meaningful user impact; may overlap with broader LDLT CSC complexity. |
| `src/sparse_chol_csc.c` | 28 | 71 | Direct Cholesky CSC candidate with existing correctness evidence; symbolic Cholesky proof already covers a related but distinct owner. |
| `src/sparse_lu.c` | 18 | 96 | Linked-list LU solve/factor owner candidate; solve workspace may be narrower than all factorization. |
| `src/sparse_matrix.c` | 14 | 56 | Highest general user impact, but broad constructor/insertion behavior could exceed one selected proof. |
| `src/sparse_svd_partial.c` | 14 | 97 | Valuable multi-output numerical owner; fixture and retry proof may be heavier than symbolic or direct-solver owners. |
| `src/sparse_analysis.c` | 7 | 77 | Explicit Sprint 195 residual; lower allocation count but important lifecycle owner because it composes symbolic and numeric analysis state. |

### Candidate Owner Ledger

| Candidate owner | User impact | Hook fit | Current gap shape | Day 1 disposition |
| --- | --- | --- | --- | --- |
| `sparse_symbolic_lu()` symbolic owner | Medium-high for LU analysis and direct-solver setup. | Mixed; may need selected wrapper conversion if direct allocation bypasses hooks. | Sprint 195 explicitly left this unproved; output publication and cleanup are likely auditable in `src/sparse_etree.c`. | Leading Day 2 candidate. |
| `sparse_analyze()` lifecycle owner | High because it owns analysis state used by multiple solver flows. | Unclear until lifecycle tracing confirms which allocations are wrapped. | Sprint 195 explicitly left this unproved; composed ownership may make stale-output and retry semantics more complex. | Leading Day 2 candidate with scope-risk warning. |
| Standalone etree/postorder/colcount helper owner | Medium for symbolic analysis internals. | Likely good for helper-local wrapped allocations. | Helper-level failures remain out of scope from Sprint 195 and may be more bounded than full symbolic LU. | Day 2 candidate if public-output semantics can be stated cleanly. |
| Direct-solver output publication owner | High for factor/solve users. | Mixed due direct allocations and multiple solver families. | Sprint 195 retained direct solvers as future owner class; select only one entry point if chosen. | Candidate only with a narrow entry point. |
| LU CSR factor/solve owner | High for compressed direct-solver users. | Mixed; many allocation and cleanup sites. | Dense failure surface with meaningful retry value, but broad factor/growth proof could exceed scope. | Candidate if narrowed to one public function. |
| Linked-list LU solve workspace owner | High for public LU solves. | Mixed due workspace allocation style. | Temporary solve buffers offer clearer cleanup and retry proof than full linked-list factorization. | Candidate for a narrow solver lane. |
| Core sparse matrix constructor owner | Very high for all users. | Existing hook smoke tests help, but constructor family is broad. | Public stale-output behavior is important, yet one path must be isolated to avoid broad matrix claims. | Lower priority unless Day 2 finds a precise owner boundary. |
| Partial SVD output owner | Medium-high for advanced users. | Likely reachable in selected paths. | Multi-output publication and numerical retry checks add review cost. | Defer unless other candidates fail feasibility. |

### Initial Invariant Categories

| Category | Day 1 placeholder |
| --- | --- |
| Cleanup | Failed allocation must release selected-owner temporary and partially owned resources. |
| Publication | Failed allocation must not publish a success-looking owner object or overwrite caller-visible outputs outside the selected contract. |
| Stale-output suppression | Caller-visible outputs should remain unset, unchanged, or explicitly documented after selected-owner allocation failure. |
| Retry | After resetting deterministic failure injection, the same fixture should succeed and produce fresh output. |
| Caller-owned input | Input matrices, options, vectors, and caller buffers must remain valid and unchanged unless the selected owner contract explicitly permits mutation. |
| Unsupported breadth | The proof will not imply broad allocation-failure, OS OOM, concurrent allocation-hook, platform, sanitizer, or all-solver reliability coverage. |

### Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Documentation-only changes. |
| Focused selected-owner gate | Not yet applicable. | Owner is selected on Day 2. |
| Source-list check | Not yet applicable. | No source or test registration changes on Day 1. |
| `make format && make lint && make test` | Not required for Day 1. | No `.c` or `.h` files modified. |

### Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Selecting too broad an owner | Sprint 200 should completely close one gap rather than partially touch many. | Day 2 must freeze exactly one owner and document rejected breadth. |
| Re-proving an already covered owner | Sprint 200 requires one additional owner beyond existing gates. | Exclude iterative, matmul, and selected symbolic Cholesky proof lanes. |
| Direct allocations bypass deterministic hooks | Fail-after tests only prove wrapped allocation paths unless direct allocation is converted or otherwise controlled. | Prefer wrapper-reachable owners or budget a narrow wrapper conversion. |
| Stale-output semantics are ambiguous | Tests can overclaim if the owner’s publication contract is not explicit. | Day 3 and Day 4 must define publication and stale-output invariants before code changes. |
| Process-global hook contamination | Early-return assertions can leave failure hooks active. | Store statuses locally, reset hooks before assertions, and use cleanup labels. |
| Overclaiming reliability | One selected proof is not broad allocation reliability. | Keep docs and retrospective wording selected-owner-only. |

### Open Questions For Day 2

1. Is `sparse_symbolic_lu()` narrow enough to prove without pulling in broad
   matrix construction or all symbolic-analysis behavior?
2. Does `sparse_analyze()` have a clean owner boundary, or is it too composed
   for one selected proof?
3. Which candidate has the best stale-output publication contract?
4. Which candidate can reuse an existing test binary and registration guard
   pattern with low review churn?
5. Which candidate can prove retry success with the least numerical ambiguity?

### Day 1 Validation

Commands run:

```sh
git status --short --branch
sed -n '1,95p' docs/planning/EPIC_18/SPRINT_200/PLAN.md
find docs/planning -path '*SPRINT_195*' -type f | sort
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_195/artifacts/day1-reliability-intake.md
sed -n '1,180p' docs/planning/EPIC_17/SPRINT_195/WORKING_NOTES.md
sed -n '275,310p' docs/planning/EPIC_18/reviews/review-codex-2026-09-04.md
sed -n '195,222p' docs/planning/EPIC_17/SPRINT_195/RETROSPECTIVE.md
find tests -maxdepth 1 -name '*allocation_failure*' -o -name '*alloc*failure*' | sort
rg -n "allocation|alloc|failure|fault|inject|owner proof|reliability" tests Makefile docs/planning/EPIC_17/SPRINT_195 docs/planning/EPIC_18 -g '*.{c,h,md,py}'
for f in src/sparse_etree.c src/sparse_analysis.c src/sparse_lu.c src/sparse_lu_csr.c src/sparse_ldlt.c src/sparse_ldlt_csc.c src/sparse_chol_csc.c src/sparse_matrix.c src/sparse_qr.c src/sparse_svd_partial.c; do alloc=$(rg -n "sparse_malloc|sparse_calloc|malloc|calloc|realloc" "$f" | wc -l | tr -d ' '); fail=$(rg -n "cleanup:|goto cleanup|goto fail|SPARSE_ERR_ALLOC|return NULL|free\\(" "$f" | wc -l | tr -d ' '); printf "%s\\t%s\\t%s\\n" "$f" "$alloc" "$fail"; done
```

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 2: Owner Selection

### Scoring Method

Day 2 scores use a 1-5 scale where 5 is strongest for Sprint 200. The scoring
prioritizes complete selected-owner closure over broad surface area.

| Criterion | Meaning |
| --- | --- |
| Review value | The fix materially improves a visible reliability gap and is easy to review. |
| Failure-path risk | Allocation failure could currently leave partial state, stale output, or unclear cleanup behavior. |
| Harness reachability | Deterministic fail-after hooks can reach the owner with minimal wrapper conversion. |
| Stale-output exposure | The owner has caller-visible outputs where failed publication must be proven safe. |
| Retry clarity | A failure can be followed by a successful retry with clear expected output. |
| Implementation size | The owner can be proven without broad solver rewrites or multi-owner churn. |

### Ranked Candidate Table

| Rank | Candidate owner | Review value | Failure-path risk | Harness reachability | Stale-output exposure | Retry clarity | Implementation size | Total | Disposition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `sparse_symbolic_lu()` symbolic owner | 5 | 5 | 4 | 5 | 4 | 5 | 28 | Selected. Explicit Sprint 195 residual, same source/test family as the completed symbolic Cholesky proof, and clear `sym_L`/`sym_U` publication semantics. |
| 2 | Helper-level etree/postorder/colcount owner | 3 | 4 | 5 | 2 | 3 | 5 | 22 | Fallback symbolic owner. More bounded, but weaker public-output and stale-output value. |
| 3 | `sparse_analyze()` lifecycle owner | 5 | 5 | 3 | 4 | 3 | 2 | 22 | Deferred. High user value, but composed analysis state risks turning one selected proof into broad lifecycle coverage. |
| 4 | Linked-list LU solve workspace owner | 4 | 3 | 3 | 3 | 4 | 4 | 21 | Deferred. Narrower than full LU, but caller-owned output mutation makes stale-output semantics less crisp. |
| 5 | LU CSR selected entry point | 5 | 4 | 3 | 3 | 4 | 2 | 21 | Deferred. Valuable direct-solver owner, but factor/growth breadth is larger than the symbolic LU target. |
| 6 | Direct-solver output publication owner | 5 | 4 | 2 | 4 | 4 | 2 | 21 | Deferred. Must be narrowed to one function in a later sprint or day if symbolic LU becomes infeasible. |
| 7 | Core sparse matrix constructor path | 5 | 4 | 3 | 4 | 5 | 1 | 22 | Rejected for Sprint 200. Too easy to imply broad sparse-matrix allocation coverage. |
| 8 | Partial SVD output owner | 3 | 4 | 3 | 5 | 3 | 2 | 20 | Deferred. Multi-output numerical retry proof is higher cost than symbolic LU. |

### Selected Owner

Sprint 200 selects `sparse_symbolic_lu()` in `src/sparse_etree.c`.

Selected boundary:

- owner function: `sparse_symbolic_lu(const SparseMatrix *A, const idx_t *perm,
  sparse_symbolic_t *sym_L, sparse_symbolic_t *sym_U)`;
- input owner: caller-owned `SparseMatrix` and optional caller-owned
  permutation array;
- output owners: caller-provided `sym_L` and `sym_U` symbolic structures;
- selected success modes: L+U, L-only, U-only, and optional valid
  permutation;
- primary test binary: `tests/test_etree.c`;
- expected focused gate: extend the existing symbolic allocation-failure gate
  or add a clearly named symbolic-LU sub-guard without claiming all symbolic
  analysis;
- primary reusable pattern: Sprint 195 symbolic Cholesky allocation-failure
  tests that reset hooks before assertions, prove empty failed output, and
  then retry successfully.

### Selection Rationale

`sparse_symbolic_lu()` is the highest-value complete closure candidate because
Sprint 195 explicitly left it unproved, while the existing symbolic Cholesky
proof already established local patterns for symbolic output cleanup,
deterministic fail-after testing, focused gate registration, and selected-owner
claim language. The function has delayed `sym_L` publication and immediate
`sym_U` initialization, which gives the sprint a concrete stale-output and
partial-publication surface to prove without expanding into all direct solvers
or the full analysis lifecycle.

### Rejected And Deferred Candidates

| Candidate | Disposition | Reason | Future follow-up |
| --- | --- | --- | --- |
| Iterative repeated-run handles | Excluded | Already covered by `make iterative-allocation-failure-gate`. | Maintain existing gate only. |
| `sparse_matmul()` workspace | Excluded | Already covered by `make matmul-allocation-failure-gate`. | Maintain existing registration guard only. |
| `sparse_symbolic_cholesky()` | Excluded | Already covered by `make symbolic-allocation-failure-gate`. | Use as proof model, not target. |
| `sparse_analyze()` | Deferred | Composes reorder, symbolic, and analysis-state publication; too broad for this selected-owner sprint. | Select explicitly in a later sprint with analysis lifecycle invariants. |
| Helper-level etree/postorder/colcount | Deferred | More bounded but less public-output value than symbolic LU. | Use if symbolic LU proves unreachable through deterministic hooks. |
| LU CSR selected entry point | Deferred | Dense direct-solver path with larger implementation and source-list risk. | Pick one function and define caller-output semantics first. |
| Linked-list LU solve workspace | Deferred | Caller-owned output mutation complicates stale-output assertions. | Scope to a solve-workspace proof with explicit output contract. |
| Core sparse matrix constructor path | Rejected for Sprint 200 | Too broad and likely to imply broad constructor reliability. | Split into one constructor/insertion owner in a future reliability sprint. |
| Partial SVD output owner | Deferred | Multi-output numerical retry proof has higher fixture cost. | Revisit after symbolic and direct-solver owners are stronger. |

### Owner Proof Checklist

| Proof area | Sprint 200 requirement for `sparse_symbolic_lu()` |
| --- | --- |
| Cleanup | Allocation failure releases temporary `B`, permutation workspaces, etree workspaces, symbolic Cholesky intermediates, and partially initialized `sym_U` data. |
| Stale-output suppression | Failed calls must not leave `sym_L` or `sym_U` in a success-looking state; any initialized failed output must be safe to free repeatedly. |
| Retry | After resetting deterministic failure injection, the same fixture must succeed and produce fresh symbolic LU structures. |
| Caller-owned input | The input matrix and optional permutation must remain valid and unchanged after injected allocation failure. |
| Unsupported breadth | The proof does not claim broad `sparse_analyze()`, all etree helpers, direct solvers, matrix construction, OS OOM, concurrent allocation hooks, or platform parity. |
| Gate ownership | Focused gate wording must name symbolic LU or selected symbolic allocation owners precisely enough to avoid implying all symbolic paths. |

### Day 2 Validation

Commands run:

```sh
git status --short --branch
sed -n '50,115p' docs/planning/EPIC_18/SPRINT_200/PLAN.md
sed -n '1,260p' docs/planning/EPIC_18/SPRINT_200/WORKING_NOTES.md
rg -n "sparse_symbolic_lu|sparse_symbolic_cholesky|sparse_analyze|sparse_analyze_" src include tests | head -80
nl -ba src/sparse_etree.c | sed -n '400,665p'
nl -ba src/sparse_analysis.c | sed -n '1,240p'
rg -n "symbolic_lu|symbolic.*allocation|alloc.*symbolic|sparse_analyze" tests/test_etree.c tests/test_integration.c tests/test_sparse_matrix.c tests/*.py
nl -ba tests/test_etree.c | sed -n '1000,1155p'
nl -ba tests/test_etree.c | sed -n '1300,1485p'
```

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 3: Lifecycle Trace

### Selected Owner Boundary

Day 3 traces `sparse_symbolic_lu()` in `src/sparse_etree.c` lines 412-665.
The selected owner remains exactly this function and its caller-visible
`sym_L`/`sym_U` publication behavior. The trace does not expand Sprint 200 to
all symbolic analysis, `sparse_analyze()`, direct solvers, matrix
construction, or OS allocation behavior.

### Lifecycle Map

| Phase | Implementation reference | Allocation/publication behavior | Cleanup path |
| --- | --- | --- | --- |
| Argument and shape validation | `src/sparse_etree.c:414-419` | Rejects null matrix, both outputs null, or non-square input before allocation. | No owner cleanup needed. |
| Interaction graph allocation | `src/sparse_etree.c:423-429` | Creates temporary `B` with `sparse_create(n, n)`. | `sparse_free(B)` on later failures. |
| Optional permutation workspace | `src/sparse_etree.c:431-465` | Allocates `seen` with direct `calloc` and `inv_perm` with direct `malloc` when `perm` is provided. | Frees `seen`/`inv_perm` and `B` on allocation or validation failure. |
| Row workspace | `src/sparse_etree.c:467-473` | Allocates `row_cols` through `sparse_malloc_idx_array`. | Frees `inv_perm` and `B` on failure. |
| `A^T * A` construction | `src/sparse_etree.c:475-510` | Inserts entries into temporary `B`; diagonal insertions may allocate through matrix internals. | Frees `row_cols`, `inv_perm`, and `B`; insertion failures return without publishing outputs. |
| Etree workspace allocation | `src/sparse_etree.c:513-525` | Allocates `parent`, `postorder`, and `cc` through `sparse_malloc_idx_array`. | Frees any allocated workspace and `B`. |
| Symbolic pipeline | `src/sparse_etree.c:527-543` | Computes etree, postorder, colcount, then builds local `sym_full` through `sparse_symbolic_cholesky`. | `cleanup` frees etree workspaces and `B`; `sym_full` is freed only on later U-building failures or L-not-requested success. |
| U workspace and output start | `src/sparse_etree.c:551-561` | If `sym_U` is requested, zeros caller-provided `sym_U`, sets `sym_U->n = n`, and allocates `u_cnt`. | On `u_cnt` failure, frees `sym_full`; `sym_U` remains zeroed except for `n`. |
| U `col_ptr` publication | `src/sparse_etree.c:572-590` | Allocates `sym_U->col_ptr` with direct `malloc`. | On failure, frees `u_cnt` and `sym_full`; `sym_U` has no allocated arrays. |
| U `col_ptr` fill and overflow checks | `src/sparse_etree.c:591-617` | Fills `sym_U->col_ptr` and `sym_U->nnz`; overflow path calls `sparse_symbolic_free(sym_U)`. | Frees `u_cnt`, partial `sym_U`, and `sym_full`. |
| U `row_idx` publication | `src/sparse_etree.c:619-626` | Allocates `sym_U->row_idx` through `sparse_malloc_idx_array`. | Frees `u_cnt`, partial `sym_U`, and `sym_full`. |
| U fill and sorting | `src/sparse_etree.c:628-649` | Writes final U rows and frees `u_cnt`. | No failure path after row writes in current code. |
| L publication | `src/sparse_etree.c:652-657` | Publishes `sym_full` into `*sym_L` only after all selected work succeeds; frees `sym_full` if L was not requested. | Success path transfers or releases `sym_full`. |
| Common cleanup | `src/sparse_etree.c:659-664` | Frees `parent`, `postorder`, `cc`, and `B`. | Returns final status. |

### Allocation Point Map

| Allocation point | Hook-controlled today | Day 3 reachability note |
| --- | --- | --- |
| Temporary `B = sparse_create(n, n)` | No direct selected-owner hook from `sparse_symbolic_lu()` call site. | Treat as owner setup failure only if existing matrix allocation hook reaches internals; otherwise avoid claiming this point until Day 5. |
| `seen = calloc(...)` | No. | Candidate for narrow wrapper conversion if valid-permutation failure coverage is required. |
| `inv_perm = malloc(...)` | No. | Candidate for narrow wrapper conversion if valid-permutation failure coverage is required. |
| `row_cols = sparse_malloc_idx_array(...)` | Yes. | Primary early selected-owner fail-after point. |
| `sparse_insert(B, ...)` growth | Mixed, through matrix internals. | May be reachable but belongs partly to matrix construction; avoid overclaiming unless tests isolate propagated allocation failures. |
| `parent`, `postorder`, `cc` | Yes. | Primary middle selected-owner fail-after points. |
| `sparse_symbolic_cholesky(...)` local `sym_full` | Yes for internal symbolic allocations. | Reuse existing symbolic Cholesky failure case expectations, but claim as propagated selected symbolic LU failure. |
| `u_cnt = sparse_calloc_idx_array(...)` | Yes. | Primary U-building fail-after point before U arrays are published. |
| `sym_U->col_ptr = malloc(...)` | No. | Important stale-output point; likely needs wrapper conversion for deterministic proof. |
| `sym_U->row_idx = sparse_malloc_idx_array(...)` | Yes. | Primary late U-building fail-after point after `col_ptr` and `nnz` publication. |

### Publication And Stale-Output Map

| Output mode | Publication behavior | Stale-output risk |
| --- | --- | --- |
| `sym_L` + `sym_U` | `sym_U` is zeroed and built before `sym_L` receives `sym_full`; `sym_L` is published only at final success. | Failed U-building must not leave `sym_U` success-looking; `sym_L` should remain caller sentinel or be explicitly unchanged until final publication. |
| L-only | `sym_U == NULL`; `sym_L` receives `sym_full` after symbolic Cholesky and cleanup work succeeds. | Failures before final assignment must not mutate caller `sym_L` into success-looking output. |
| U-only | `sym_L == NULL`; local `sym_full` is freed after U success. | Failed U-building must clean partial `sym_U` and free `sym_full`; success must not leak `sym_full`. |
| Permuted L+U | Valid `perm` drives `seen`/`inv_perm` setup before graph construction. | Failed permutation workspace allocation must preserve caller-owned `perm` and avoid output publication. |

### Caller-Owned Input Preservation

| Caller-owned input | Expected Day 3 preservation rule |
| --- | --- |
| `SparseMatrix *A` | Read-only during selected-owner execution; failure tests should assert dimensions and representative entries remain unchanged. |
| `const idx_t *perm` | Read-only during selected-owner execution; valid-permutation failure tests should compare the full array before and after failure. |
| `sparse_symbolic_t *sym_L` | Caller owns the struct storage; selected proof should decide whether failure leaves sentinel fields unchanged or requires zero/free-safe state. |
| `sparse_symbolic_t *sym_U` | Caller owns the struct storage; selected proof must handle early zeroing, partial `col_ptr`, partial `nnz`, and free-safe cleanup after failure. |

### Retry Entry Points

| Retry scenario | Expected proof shape |
| --- | --- |
| Natural-order L+U retry | Inject selected failure, reset hook, call `sparse_symbolic_lu(A, NULL, &retry_L, &retry_U)`, and compare against existing containment or basic symbolic expectations. |
| L-only retry | Inject failure before final `sym_L` publication, reset hook, call L-only path, and assert fresh L output. |
| U-only retry | Inject U-building failure, reset hook, call U-only path, and assert fresh U output. |
| Valid-permutation retry | Inject reachable valid-permutation path failure after Day 5 harness decision, reset hook, call with same `perm`, and assert `perm` and `A` remain intact. |

### Unsupported Breadth

| Not covered by Day 3 trace | Reason |
| --- | --- |
| Broad `sparse_analyze()` allocation failures | `sparse_analyze()` composes multiple solver and reorder lifecycles and remains deferred. |
| All `src/sparse_etree.c` helpers | Helper-level etree/postorder/colcount ownership is a separate candidate. |
| Matrix construction allocation reliability | Temporary `B` and `sparse_insert` may allocate, but broad matrix allocation proof is outside Sprint 200. |
| Direct solver factor/solve output publication | Direct solvers remain future selected-owner candidates. |
| OS OOM or concurrent allocation-hook behavior | The deterministic hook is private test infrastructure, not a platform OOM or concurrency claim. |
| Hosted CI ownership | Day 3 traces local proof behavior only; hosted lane ownership is not selected. |

### Day 4 Invariant Inputs

Day 4 should convert this trace into testable invariants, with special care for
three semantic decisions:

1. whether failed `sym_L` remains caller-sentinel unchanged until final
   publication or is required to be zero/free-safe;
2. whether failed `sym_U` must be zero/free-safe after early zeroing and
   partial U-building;
3. whether direct `malloc`/`calloc` paths for `seen`, `inv_perm`, and
   `sym_U->col_ptr` need wrapper conversion before deterministic proof.

### Day 3 Validation

Commands run:

```sh
git status --short --branch
sed -n '86,142p' docs/planning/EPIC_18/SPRINT_200/PLAN.md
sed -n '260,430p' docs/planning/EPIC_18/SPRINT_200/WORKING_NOTES.md
nl -ba src/sparse_etree.c | sed -n '120,665p'
rg -n "sparse_symbolic_free|typedef struct.*sparse_symbolic|sparse_symbolic_t|sparse_alloc_test" src include tests/test_etree.c tests/test_framework.h src/sparse_alloc_internal.* include -g '*.{c,h}'
nl -ba tests/test_etree.c | sed -n '760,1145p'
nl -ba tests/test_etree.c | sed -n '1280,1485p'
nl -ba src/sparse_analysis_internal.h | sed -n '70,155p'
nl -ba src/sparse_alloc_internal.c | sed -n '1,90p'
nl -ba src/sparse_alloc_internal.h | sed -n '1,90p'
rg -n "SparseMatrix \\*sparse_create|sparse_create\\(|sparse_insert\\(" src/sparse_matrix.c include/sparse_matrix.h | head -40
```

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 4: Invariant Record

### Selected Owner Contract

The Sprint 200 proof contract applies only to `sparse_symbolic_lu()` in
`src/sparse_etree.c`. It covers deterministic allocation-failure behavior for
the selected symbolic LU owner when callers satisfy the existing internal
precondition that non-null `sym_L` and `sym_U` are zeroed or previously freed
before the call.

### Pre-Edit Invariant Table

| ID | Invariant | Planned assertion |
| --- | --- | --- |
| S200-LU-ARG-01 | Null `A`, both outputs null, and non-square input return the existing argument/shape errors before selected-owner allocation proof begins. | Keep existing argument tests; do not mix argument validation with allocation-failure claims. |
| S200-LU-CLEAN-01 | Every deterministic allocation failure after owner entry releases temporary `B`, `row_cols`, `parent`, `postorder`, `cc`, `u_cnt`, local `sym_full`, and any partial `sym_U` arrays owned by the function. | Fail-at-count cases return allocation failure and requested outputs remain safe for `sparse_symbolic_free()`. |
| S200-LU-CLEAN-02 | Propagated allocation failure from the local symbolic Cholesky intermediate releases the temporary LU owner resources and does not transfer `sym_full` to `sym_L`. | Force intermediate symbolic failure and assert `sym_L`/`sym_U` are not success-looking. |
| S200-LU-PUB-01 | `sym_L` is published only after all requested LU symbolic work succeeds. | On every injected failure, `sym_L.col_ptr == NULL`, `sym_L.row_idx == NULL`, `sym_L.n == 0`, and `sym_L.nnz == 0`. |
| S200-LU-PUB-02 | `sym_U` may be initialized during U-building, but on allocation failure it must be returned to empty/free-safe state before the function returns. | On every injected U-building failure, `sym_U.col_ptr == NULL`, `sym_U.row_idx == NULL`, `sym_U.n == 0`, and `sym_U.nnz == 0`. |
| S200-LU-STALE-01 | Failed calls must not leave a requested output in a success-looking partial state. | Assert no non-null output arrays, no positive `nnz`, and no nonzero `n` after allocation failure. |
| S200-LU-RETRY-01 | After `sparse_alloc_test_reset()`, the same matrix and output mode must succeed and produce fresh symbolic output. | Failure-then-success tests for L+U, L-only, and U-only modes. |
| S200-LU-INPUT-01 | Caller-owned `SparseMatrix *A` is read-only across selected-owner allocation failures. | Snapshot dimensions and representative entries before failure; assert unchanged afterward. |
| S200-LU-INPUT-02 | Caller-owned valid `perm` is read-only across selected-owner allocation failures. | Snapshot every permutation entry before failure; assert unchanged afterward. |
| S200-LU-HOOK-01 | Process-global allocation hook state is reset before any assertion macro or early-return helper can exit a test. | Store status in a local, call `sparse_alloc_test_reset()`, then assert. |
| S200-LU-HOOK-02 | Direct selected-owner allocation sites needed for proof must be routed through existing private wrappers or explicitly excluded from the earned claim. | Day 5/6 wrapper decision for `seen`, `inv_perm`, and `sym_U->col_ptr`. |
| S200-LU-SCOPE-01 | The proof remains selected symbolic LU only. | Documentation and gate names must not imply broad `sparse_analyze()`, etree-helper, matrix, direct-solver, OS OOM, or concurrency coverage. |

### Output-State Decisions

| Output mode | Allocation-failure invariant |
| --- | --- |
| L+U | Both requested outputs must be empty/free-safe after every selected allocation failure. `sym_L` must not receive `sym_full`; `sym_U` must not retain partial U arrays or nonzero metadata. |
| L-only | `sym_L` must remain empty/free-safe on failure and receive fresh output only on retry success. |
| U-only | `sym_U` must be empty/free-safe on failure; local `sym_full` must be freed because no caller receives L. |
| Valid permutation | `perm` must remain byte-for-byte unchanged; outputs must remain empty/free-safe on selected allocation failure. |

### Cleanup Checklist

| Resource | Cleanup invariant |
| --- | --- |
| Temporary `B` | Freed on every failure after creation. |
| `seen` and `inv_perm` | Freed on permutation allocation failure, invalid permutation failure, and later selected failures after allocation. |
| `row_cols` | Freed after graph construction and on its allocation failure path. |
| `parent`, `postorder`, `cc` | Freed through direct failure cleanup or the common cleanup label. |
| Local `sym_full` | Freed on every failure after symbolic Cholesky succeeds unless transferred to `sym_L` on final success. |
| `u_cnt` | Freed on all U-building failure paths and after successful U row fill. |
| Partial `sym_U` | Cleared with `sparse_symbolic_free(sym_U)` on any failure after `sym_U` initialization. |
| Process-global hook | Reset before assertions, fixture frees, or retry calls. |

### Claim Boundary Wording

Earned Sprint 200 claim, once tests and gates exist:

`make symbolic-lu-allocation-failure-gate` or the selected symbolic gate proves
deterministic local allocation-failure status, requested-output cleanup,
stale-output suppression, caller-owned matrix/permutation preservation, and
retry-after-reset behavior for selected `sparse_symbolic_lu()` fixtures.

Retained non-claims:

- no broad allocation-failure coverage;
- no `sparse_analyze()` lifecycle or publication proof;
- no standalone etree/postorder/colcount helper proof;
- no direct-solver, eigensolver, graph, SVD, sparse-matrix constructor,
  conversion, or IO allocation-failure proof;
- no OS-level OOM guarantee;
- no concurrent allocation-hook behavior guarantee;
- no hosted CI, platform parity, package, ABI, performance, release, or
  state-of-the-art reliability claim.

### Test Mapping

| Planned test family | Invariants covered |
| --- | --- |
| Hook reachability smoke tests | S200-LU-HOOK-01, S200-LU-HOOK-02. |
| Failed allocation L+U table | S200-LU-CLEAN-01, S200-LU-CLEAN-02, S200-LU-PUB-01, S200-LU-PUB-02, S200-LU-STALE-01. |
| L-only failed allocation and retry | S200-LU-PUB-01, S200-LU-RETRY-01, S200-LU-INPUT-01. |
| U-only failed allocation and retry | S200-LU-PUB-02, S200-LU-RETRY-01, S200-LU-INPUT-01. |
| Valid-permutation failed allocation and retry | S200-LU-INPUT-02, S200-LU-RETRY-01. |
| Focused gate registration guard | S200-LU-SCOPE-01. |

### Day 5 Harness Inputs

Day 5 should decide the minimum wrapper/harness changes needed for these
specific allocation points:

1. direct `calloc` for `seen`;
2. direct `malloc` for `inv_perm`;
3. direct `malloc` for `sym_U->col_ptr`;
4. whether propagated `sparse_insert(B, ...)` allocation failures are inside
   or outside the selected symbolic LU proof.

### Day 4 Validation

Commands run:

```sh
git status --short --branch
sed -n '126,180p' docs/planning/EPIC_18/SPRINT_200/PLAN.md
sed -n '430,680p' docs/planning/EPIC_18/SPRINT_200/WORKING_NOTES.md
sed -n '1,220p' docs/planning/EPIC_18/SPRINT_200/artifacts/day3-lifecycle-trace.md
sed -n '1,240p' docs/planning/EPIC_17/SPRINT_195/artifacts/day3-selected-owner-invariant-record.md
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_195/artifacts/day11-claim-boundaries.md
```

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 5: Harness Reachability Design

### Harness Decision

Sprint 200 will reuse the existing private deterministic allocation hook:

- `sparse_alloc_test_fail_after(long remaining)`;
- `sparse_alloc_test_reset()`;
- `sparse_malloc_array(...)`;
- `sparse_calloc_array(...)`;
- `sparse_malloc_idx_array(...)`;
- `sparse_calloc_idx_array(...)`.

No new global hook, environment variable, allocator callback, public API, or
test binary is planned. The selected owner needs narrow wrapper conversion for
only the direct allocation sites that belong to `sparse_symbolic_lu()` itself.

### Required Minimal Code Changes For Day 6

| Site | Current code | Day 5 decision | Reason |
| --- | --- | --- | --- |
| `seen` permutation bitmap | direct `calloc(1, seen_bytes)` | Convert to `sparse_calloc_array(seen_bytes, sizeof(unsigned char), (void **)&seen)` or equivalent count-based wrapper use. | Makes valid-permutation allocation failure deterministic without broad allocator changes. |
| `inv_perm` permutation inverse | direct `malloc(inv_perm_bytes)` | Convert to `sparse_malloc_array((size_t)n, sizeof(idx_t), (void **)&inv_perm)`. | Keeps permutation workspace inside existing fail-after hook. |
| `sym_U->col_ptr` | direct `malloc(u_col_ptr_bytes)` | Convert to `sparse_malloc_array(u_col_ptr_len, sizeof(idx_t), (void **)&sym_U->col_ptr)`. | This is the main stale-output publication point and must be reachable. |
| `sym_U` early initialization | `memset(sym_U, 0, ...)` then `sym_U->n = n` before `u_cnt` allocation | Ensure every U-building allocation failure calls `sparse_symbolic_free(sym_U)` before returning. | Satisfies the Day 4 empty/free-safe output invariant. |

### Explicit Non-Changes

| Surface | Decision |
| --- | --- |
| `SparseMatrix *B = sparse_create(n, n)` | Do not convert the top-level `sparse_create` direct `malloc` in Sprint 200. Matrix shell allocation failures may be observed only as propagated setup failures, not as earned symbolic-LU owner proof. |
| `sparse_insert(B, ...)` allocation failures | Treat as propagated matrix-construction allocation failures. Tests may include them only if assertions state they are propagated setup failures, not a broad matrix allocation proof. |
| `sparse_symbolic_cholesky()` internals | Do not rework the already covered symbolic Cholesky owner. Symbolic LU tests may force failures inside this intermediate and claim only symbolic-LU cleanup/publication behavior around the propagated failure. |
| New CTest binary | Not needed. Reuse `tests/test_etree.c`. |
| New public allocator API | Not needed. The hook remains private test infrastructure. |

### Planned Failure-Index Families

Exact fail-after indices can shift with fixture shape and Day 6 wrapper
conversion, so tests should discover or document them through named case
tables tied to stable fixture paths. The planned deterministic order for the
small selected fixtures is:

| Family | Expected reachable allocation sequence after fixture setup | Test purpose |
| --- | --- | --- |
| Natural L+U | `row_cols`, temporary `B` insertion internals if reached, `parent`, `postorder`, `cc`, `sym_full` allocations, `u_cnt`, `sym_U->col_ptr`, `sym_U->row_idx`. | Main cleanup and stale-output proof for both outputs. |
| L-only | `row_cols`, temporary `B` insertion internals if reached, `parent`, `postorder`, `cc`, `sym_full` allocations. | Proves delayed `sym_L` publication and retry without U-building. |
| U-only | `row_cols`, temporary `B` insertion internals if reached, `parent`, `postorder`, `cc`, `sym_full` allocations, `u_cnt`, `sym_U->col_ptr`, `sym_U->row_idx`. | Proves partial U cleanup and local `sym_full` cleanup. |
| Valid permutation | `seen`, `inv_perm`, `row_cols`, temporary `B` insertion internals if reached, `parent`, `postorder`, `cc`, `sym_full` allocations, U allocations when U is requested. | Proves permutation input preservation and hook reachability for converted permutation workspaces. |

The Day 6 implementation should start with hook-reachability smoke tests for
the newly converted direct sites, then Day 7-Day 9 can lock in the final
failure case table after observing the stable local sequence.

### Expected Test Helpers

| Helper | Responsibility |
| --- | --- |
| `assert_symbolic_lu_failure_outputs_free_safe(...)` | Assert requested failed outputs are empty, call `sparse_symbolic_free()` repeatedly, and assert they remain empty. |
| `assert_unsym_3x3_matrix_intact(...)` | Check dimensions and representative entries after failure. |
| `assert_perm_intact(...)` | Compare every permutation entry before and after valid-permutation failures. |
| `expect_symbolic_lu_allocation_failure(...)` | Set hook, call selected owner, reset hook before assertions, assert failure and cleanup. |
| `expect_symbolic_lu_allocation_failure_recovers(...)` | Add retry success after reset and compare fresh output. |

### Focused Gate And Registration Plan

The preferred Day 10 gate shape is to keep `make symbolic-allocation-failure-gate`
as the umbrella symbolic allocation-failure gate and extend
`tests/test_symbolic_allocation_failure_gate_registration.py` so it also
requires symbolic-LU allocation-failure tests and named failure cases. This
keeps one `test_etree` CTest label set:

`etree;symbolic;allocation_failure`

If review clarity requires a separate target later, the fallback is a
`symbolic-lu-allocation-failure-gate` alias that still builds and runs
`$(BUILDDIR)/test_etree` plus a symbolic-LU-specific Python registration
guard. Day 5 does not require that split.

### Harness Cleanup Rules

| Rule | Enforcement plan |
| --- | --- |
| Reset before assertion | Tests must store `sparse_symbolic_lu()` status, call `sparse_alloc_test_reset()`, then assert. |
| Reset before retry | Retry calls occur only after `sparse_alloc_test_reset()`. |
| Reset before fixture cleanup | Failure helpers call reset before freeing fixtures if an assertion can early-return. |
| Free requested outputs after failure | Failure helpers call repeated `sparse_symbolic_free()` checks only after hook reset. |
| Avoid hidden hook state in loops | Table-driven tests reset at the beginning and end of each case. |

### Day 5 Risks

| Risk | Mitigation |
| --- | --- |
| Failure indices drift after wrapper conversion | Use named case tables after Day 6 observation and guard the case names in Python. |
| Matrix-internal allocations blur owner boundary | Mark `sparse_create`/`sparse_insert` failures as propagated setup behavior or exclude them from the earned claim. |
| `sym_U->n` remains nonzero after early U failure | Day 6 should explicitly clear `sym_U` on `u_cnt` allocation failure. |
| Direct allocation conversion changes zero-size behavior | Preserve the existing `seen_bytes != 0` and `inv_perm_bytes != 0` semantics or rely on wrapper zero-count no-op behavior only after verifying `n == 0` is not a valid `sparse_create` path. |
| Registration guard becomes too broad | Require symbolic-LU-specific test names while keeping CTest labels selected symbolic/allocation-failure scoped. |

### Day 5 Validation

Commands run:

```sh
git status --short --branch
sed -n '166,220p' docs/planning/EPIC_18/SPRINT_200/PLAN.md
sed -n '680,920p' docs/planning/EPIC_18/SPRINT_200/WORKING_NOTES.md
sed -n '1,220p' docs/planning/EPIC_18/SPRINT_200/artifacts/day4-invariant-record.md
nl -ba Makefile | sed -n '280,312p'
nl -ba CMakeLists.txt | rg -n "test_etree|allocation_failure|symbolic" -C 3
sed -n '1,120p' tests/test_symbolic_allocation_failure_gate_registration.py
nl -ba src/sparse_etree.c | sed -n '412,626p'
rg -n "sparse_alloc_test_fail_after|SymbolicFailureCase|RUN_TEST\\(test_symbolic_cholesky" tests/test_etree.c -C 2
nl -ba src/sparse_matrix.c | sed -n '178,230p'
nl -ba src/sparse_matrix.c | sed -n '347,430p'
sed -n '1,180p' tests/test_matmul_allocation_failure_gate_registration.py
```

Day 5 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 6: Harness Integration

### Implementation Summary

Day 6 implemented the selected Day 5 harness design inside
`sparse_symbolic_lu()` only.

| Site | Day 6 change | Invariant supported |
| --- | --- | --- |
| Optional `seen` bitmap | Replaced direct `calloc` with `sparse_calloc_idx_array(...)`. | S200-LU-HOOK-02 and valid-permutation failure reachability. |
| Optional `inv_perm` inverse permutation | Replaced direct `malloc` with `sparse_malloc_idx_array(...)`. | S200-LU-HOOK-02 and valid-permutation failure reachability. |
| Early `sym_U` allocation failure | Added `sparse_symbolic_free(sym_U)` before returning from `u_cnt` allocation failure. | S200-LU-PUB-02 and S200-LU-STALE-01. |
| `sym_U->col_ptr` | Replaced direct `malloc` with `sparse_malloc_array(...)`. | S200-LU-HOOK-02 and U `col_ptr` failure reachability. |
| `sym_U->col_ptr` length overflow | Clears partial `sym_U` before returning allocation failure. | S200-LU-CLEAN-01 and S200-LU-PUB-02. |

No new global hook, public API, environment variable, test binary, or broad
allocator behavior was introduced.

### Reachability After Day 6

| Allocation point | Reachability after integration |
| --- | --- |
| `seen` | Hook-controlled for valid-permutation fixtures. |
| `inv_perm` | Hook-controlled for valid-permutation fixtures. |
| `row_cols` | Already hook-controlled. |
| `parent`, `postorder`, `cc` | Already hook-controlled. |
| `sym_full` internal allocations | Already hook-controlled through the existing symbolic Cholesky path. |
| `u_cnt` | Already hook-controlled. |
| `sym_U->col_ptr` | Hook-controlled after Day 6. |
| `sym_U->row_idx` | Already hook-controlled. |

`sparse_create(n, n)` and `sparse_insert(B, ...)` remain matrix-construction
or propagated setup behavior. Sprint 200 still does not claim broad matrix
allocation-failure ownership.

### Harness Reset Discipline

Day 6 did not add failure tests yet, but the implementation preserves the
planned test discipline:

1. set fail-after immediately before `sparse_symbolic_lu()`;
2. store the return status in a local;
3. call `sparse_alloc_test_reset()` before assertions;
4. verify outputs and caller-owned inputs;
5. reset again before retry and at the end of each case.

### Day 7 Handoff

Day 7 can now add failed-allocation tests for:

- valid-permutation `seen` allocation;
- valid-permutation `inv_perm` allocation;
- natural-order `row_cols`;
- `parent`, `postorder`, and `cc`;
- propagated `sym_full` allocation failures;
- `u_cnt`;
- `sym_U->col_ptr`;
- `sym_U->row_idx`.

### Day 6 Validation

Commands run:

```sh
git status --short --branch
nl -ba src/sparse_etree.c | sed -n '430,626p'
sed -n '206,260p' docs/planning/EPIC_18/SPRINT_200/PLAN.md
git diff -- src/sparse_etree.c
make symbolic-allocation-failure-gate
make format
make lint
make test
git diff --check
```

Results:

- `make format`: PASS.
- `make symbolic-allocation-failure-gate`: PASS; `test_etree` reported 101
  tests, 0 failures, 0 skipped, and 1262 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.

Day 6 modified `src/sparse_etree.c`, so the full C quality gate was run.

## Day 7: Failed-Allocation Tests

### Implementation Summary

Day 7 added direct failed-allocation coverage for the selected
`sparse_symbolic_lu()` owner. The tests run through the existing `test_etree`
binary and are locked into the existing symbolic allocation-failure registration
guard.

| File | Day 7 change |
| --- | --- |
| `src/sparse_etree.c` | Clears requested symbolic LU outputs before the first internal allocation, after argument validation. |
| `tests/test_etree.c` | Adds `test_symbolic_lu_allocation_failures_clear_outputs` and table-driven failure cases. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Requires the symbolic LU allocation test and exact case table. |
| `docs/planning/EPIC_18/SPRINT_200/artifacts/day7-failed-allocation-tests.md` | Records failure-index coverage and retained non-claims. |

### Failure-Index Coverage

The internal `sparse_create(n, n)` shell consumes six hook-controlled
allocations before symbolic-LU-specific owner allocations are reached. Day 7
does not count those setup allocations as earned selected-owner coverage.

| Case | `fail_after` | Permutation path | Assertion |
| --- | ---: | --- | --- |
| `perm seen` | 6 | yes | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `perm inverse` | 7 | yes | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `row_cols` | 6 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `parent` | 7 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `postorder` | 8 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `cc` | 9 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full col_ptr` | 10 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full row_idx` | 11 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full child_head` | 12 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full child_next` | 13 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full marker` | 14 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full tmp` | 15 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full col_rows` | 16 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full col_nrows` | 17 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_full propagated row set` | 18 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_U u_cnt` | 19 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_U col_ptr` | 20 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |
| `sym_U row_idx` | 21 | no | `SPARSE_ERR_ALLOC`; `sym_L` and `sym_U` empty/free-safe. |

### Day 7 Validation

Commands run:

```sh
make format
make symbolic-allocation-failure-gate
make lint
make test
git diff --check
git diff --check
```

Results:

- `make format`: PASS.
- `make symbolic-allocation-failure-gate`: PASS; `test_etree` reported 102
  tests, 0 failures, 0 skipped, and 1880 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.

Day 7 modified `.c` files, so the full C quality gate was run.

## Day 8: Cleanup Proof

### Implementation Summary

Day 8 added a cleanup-focused symbolic LU allocation-failure sweep over the
same selected-owner failure sites covered by Day 7. The sweep repeats every
forced failure twice and verifies both failed-output cleanup and allocation
harness teardown after each case.

| File | Day 8 change |
| --- | --- |
| `tests/test_etree.c` | Refactors the selected symbolic LU failure table for reuse and adds `test_symbolic_lu_allocation_failures_cleanup_sweep`. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Requires the cleanup sweep registration and allocation-hook reset probe. |
| `docs/planning/EPIC_18/SPRINT_200/artifacts/day8-cleanup-proof.md` | Records selected-owner cleanup evidence, limits, and validation results. |

### Cleanup Invariants

For each selected symbolic LU failure site, Day 8 proves the following
test-visible cleanup invariants:

- `sparse_symbolic_lu()` returns `SPARSE_ERR_ALLOC`;
- `sym_L` and `sym_U` are empty after failure;
- failed outputs remain safe across repeated `sparse_symbolic_free()` calls;
- the unsymmetric 3x3 fixture matrix is unchanged;
- permutation input remains unchanged for permutation-path failures;
- the allocation-failure hook is reset before assertions;
- a normal wrapper allocation succeeds after each reset.

The cleanup proof intentionally avoids claiming allocator balance accounting.
The current test harness exposes failure injection and reset behavior, but not
live allocation counts.

### Day 8 Validation

Commands run:

```sh
make format
make symbolic-allocation-failure-gate
make lint
make test
git diff --check
```

Results:

- `make format`: PASS.
- `make symbolic-allocation-failure-gate`: PASS; `test_etree` reported 103
  tests, 0 failures, 0 skipped, and 3188 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.

Day 8 modified C test code, so the full C quality gate was run.

## Day 9: Retry Proof

### Implementation Summary

Day 9 added failure-then-success retry coverage for the selected
`sparse_symbolic_lu()` owner. The new test reuses the selected failure-site
table from Days 7 and 8, forces each allocation failure once, verifies cleanup
and caller-owned input preservation, then disables injection and reruns the same
owner successfully using the same output objects.

| File | Day 9 change |
| --- | --- |
| `tests/test_etree.c` | Adds `test_symbolic_lu_allocation_failures_recover_on_retry` plus fresh-output retry assertions. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Requires the symbolic LU retry test and fresh-output assertion helper. |
| `docs/planning/EPIC_18/SPRINT_200/artifacts/day9-retry-proof.md` | Records retry behavior, input snapshots, registration guard coverage, and validation. |

### Retry Invariants

For every selected symbolic LU failure site, Day 9 proves:

- the injected failure returns `SPARSE_ERR_ALLOC`;
- `sym_L` and `sym_U` are cleared and repeatedly free-safe after failure;
- the caller-owned unsymmetric 3x3 input matrix is unchanged after failure;
- caller-owned permutation input remains `{0, 1, 2}` for permutation-path
  failures;
- retry without injection succeeds using the same output objects;
- retry output is fresh, nonempty, column-pointer monotone, and bounded against
  a numeric partial-pivot LU factorization of the same matrix;
- the input matrix and permutation remain unchanged after retry success.

These assertions close the Day 9 goal that a failed allocation does not prevent
later successful use and that retry success does not depend on stale failed-call
state.

### Day 9 Validation

Commands run:

```sh
make format
make symbolic-allocation-failure-gate
make lint
make test
git diff --check
```

Results:

- `make format`: PASS.
- `make symbolic-allocation-failure-gate`: PASS; `test_etree` reported 104
  tests, 0 failures, 0 skipped, and 4316 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.

Day 9 modified C test code, so the full C quality gate was run.

## Day 10: Focused Gate Wiring

### Implementation Summary

Day 10 added a selected-owner-only reliability gate for the Sprint 200
`sparse_symbolic_lu()` allocation-failure proof. The gate reuses the existing
`test_etree` binary but restricts execution to the symbolic LU allocation
tests through an environment selector.

| File | Day 10 change |
| --- | --- |
| `Makefile` | Adds `make symbolic-lu-allocation-failure-gate` for the selected symbolic LU proof. |
| `tests/test_etree.c` | Adds `SPARSE_TEST_SYMBOLIC_LU_ALLOCATION_ONLY` handling so the selected proof runs independently. |
| `tests/test_symbolic_lu_allocation_failure_gate_registration.py` | Adds a registration guard for the selected symbolic LU gate, test list, failure-site table, cleanup assertions, and retry assertion. |
| `docs/planning/EPIC_18/SPRINT_200/artifacts/day10-focused-gate.md` | Records the focused gate wiring, validation evidence, and retained non-claims. |

### Focused Gate Contract

The selected gate is:

```sh
make symbolic-lu-allocation-failure-gate
```

It runs:

```sh
python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py
SPARSE_TEST_SYMBOLIC_LU_ALLOCATION_ONLY=1 build/test_etree
```

The gate intentionally names symbolic LU rather than broad symbolic analysis.
The existing `make symbolic-allocation-failure-gate` remains available and still
runs the broader `test_etree` symbolic allocation-failure coverage.

### Registration Guard Coverage

The Day 10 guard fails clearly if any of the following proof components drop
out:

- `Makefile` target and `.PHONY` declaration;
- the `SPARSE_TEST_SYMBOLIC_LU_ALLOCATION_ONLY` selector in `test_etree`;
- all three selected symbolic LU allocation-failure tests;
- the exact selected failure-site table from Days 7 through 9;
- failed-output free-safe assertions for `sym_L` and `sym_U`;
- the allocation-hook reset probe;
- the fresh retry-output assertion.

No new C test file was added, so no C source-list update was required.

### Day 10 Validation

Commands run:

```sh
make format && make symbolic-lu-allocation-failure-gate
make symbolic-allocation-failure-gate
make lint
make test
```

Results:

- `make format`: PASS.
- `make symbolic-lu-allocation-failure-gate`: PASS; selected `test_etree`
  gate reported 3 tests, 0 failures, 0 skipped, and 3054 assertions.
- `make symbolic-allocation-failure-gate`: PASS; full symbolic allocation gate
  reported 104 tests, 0 failures, 0 skipped, and 4316 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.

Day 10 modified C test code, so the full C quality gate was run.

## Day 11: Claim Documentation

### Implementation Summary

Day 11 updated the selected-owner claim surfaces so Sprint 200 documentation
matches the symbolic LU evidence from Days 2 through 10. The docs now say that
Sprint 200 adds one selected `sparse_symbolic_lu()` allocation-failure owner
proof, while broad allocation-failure and state-of-the-art reliability claims
remain unearned.

| File | Day 11 change |
| --- | --- |
| `README.md` | Adds symbolic LU to selected allocation-failure proof wording and lists `make symbolic-lu-allocation-failure-gate`. |
| `INSTALL.md` | Extends the local selected allocation-failure support matrix row with the symbolic LU gate and owner boundary. |
| `docs/maintainer_guide.md` | Adds Sprint 200 symbolic LU tests, gate, guard, artifacts, and non-claims to the reliability proof-owner ledger. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Updates Sprint 200 interim status to in progress through Day 11. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Updates the additional allocation-failure residual to in-progress selected symbolic LU proof. |
| `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` | Updates Sprint 200 and status metrics so they no longer say no Sprint 200 artifacts exist. |
| `docs/planning/EPIC_18/SPRINT_200/artifacts/day11-claim-documentation.md` | Records claim surfaces, earned claim, retained non-claims, evidence links, and validation commands. |

### Earned Claim

The current earned claim is selected-owner only:

`sparse_symbolic_lu()` has focused local deterministic allocation-failure proof
for selected bounded fixtures covering allocation-failure status,
requested-output cleanup, stale-output suppression, caller-owned
matrix/permutation preservation, repeated cleanup after failure, and
retry-after-reset behavior.

### Retained Non-Claims

The docs continue to reject broad claims for:

- broad allocation-failure coverage across the library;
- `sparse_analyze()` lifecycle cleanup;
- standalone etree, postorder, or colcount helpers;
- direct solvers, eigensolvers, graph routines, SVD, sparse matrix
  construction, conversion, or IO;
- package/install flows and generated tooling;
- OS OOM behavior, platform parity, hosted CI proof, concurrent allocation-hook
  behavior, release readiness, or state-of-the-art reliability support.

### Day 11 Validation

Commands run:

```sh
make docs-check
python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py
git diff --check
```

Results:

- `make docs-check`: PASS; Doxygen generation and API docs coverage completed
  with 18 checked-in public headers, 18 generated reference pages, and 18
  generated source pages verified.
- `python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py`:
  PASS.
- `git diff --check`: PASS.

Day 11 changed documentation only. No `.c` or `.h` files were modified by the
Day 11 claim-calibration edits, so the full C quality gate is deferred to Day
12 integrated validation.

## Day 12: Integrated Local Validation

### Implementation Summary

Day 12 completed the integrated validation pass for Sprint 200 item 200.6. The
branch includes C and test changes from earlier Sprint 200 days, so the full
required local quality path was run and recorded instead of treating the pass
as documentation-only.

The Day 12 artifact is:

- `docs/planning/EPIC_18/SPRINT_200/artifacts/day12-integrated-validation.md`

### Validation Results

Commands run:

```sh
make symbolic-lu-allocation-failure-gate
make symbolic-allocation-failure-gate
make source-list-check
make docs-check
make format
make lint
make test
git diff --check
```

Results:

| Command | Result | Evidence |
| --- | --- | --- |
| `make symbolic-lu-allocation-failure-gate` | PASS | Selected gate reported 3 tests, 0 failures, 0 skipped, and 3054 assertions. |
| `make symbolic-allocation-failure-gate` | PASS | Broader symbolic allocation gate reported 104 tests, 0 failures, 0 skipped, and 4316 assertions. |
| `make source-list-check` | PASS | Source-list guard reported 49 library sources. |
| `make docs-check` | PASS | Doxygen and API docs coverage completed with 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. |
| `make format` | PASS | Clang-format completed. |
| `make lint` | PASS | Strict compile, clang-tidy, and cppcheck completed successfully. |
| `make test` | PASS | Full test suite completed with `All tests passed.` |
| `git diff --check` | PASS | Whitespace validation completed before the Day 12 evidence artifact was added. |

### Risk Register Update

No Day 12 validation blockers remain. The selected symbolic LU
allocation-failure proof is covered by focused and broader symbolic gates, and
the branch has passed formatting, lint, docs, source-list, full tests, and
whitespace checks.

The retained claim boundary is unchanged: Sprint 200 proves one selected
`sparse_symbolic_lu()` allocation-failure owner. It does not claim broad
allocation-failure coverage across the library, OS OOM behavior, hosted
platform parity, package/install reliability, or state-of-the-art reliability
support.

## Day 13: Review-Surface Hardening

### Implementation Summary

Day 13 audited the selected-owner proof surface for unnecessary breadth,
invariant coverage, gate naming, documentation vocabulary, and planning-status
consistency. No implementation behavior was changed.

The Day 13 artifact is:

- `docs/planning/EPIC_18/SPRINT_200/artifacts/day13-review-hardening.md`

### Review Findings

| Surface | Finding |
| --- | --- |
| Code | Implementation changes remain limited to `sparse_symbolic_lu()` output clearing and allocation-hook reachability for selected owner allocations. |
| Tests | Regression coverage maps to Day 4 cleanup, publication, stale-output, retry, caller-input, hook-reset, and selected-scope invariants. |
| Gate | `make symbolic-lu-allocation-failure-gate` names the selected owner and does not replace the broader symbolic allocation gate. |
| Docs | README, INSTALL, maintainer guide, and planning docs retain selected-owner wording and broad reliability non-claims. |
| Planning status | Sprint 200 status rows now reference Day 12 integrated validation and Day 13 review hardening instead of stopping at Day 11. |

### Closeout Checklist Draft

| Item | Day 13 status |
| --- | --- |
| 200.1 Owner selection | Ready for Day 14 reconciliation; selected owner remains `sparse_symbolic_lu()`. |
| 200.2 Invariant record | Ready for Day 14 reconciliation; Day 13 artifact maps each Day 4 invariant to test or guard evidence. |
| 200.3 Harness reachability | Ready for Day 14 reconciliation; selected direct allocations are covered by existing allocation wrappers. |
| 200.4 Regression proof | Ready for Day 14 reconciliation; failed-allocation, cleanup, stale-output, input-preservation, and retry paths are covered. |
| 200.5 Focused gate | Ready for Day 14 reconciliation; selected gate and registration guard are present. |
| 200.6 Docs and validation | Ready for Day 14 reconciliation; claim docs, integrated validation, and review-hardening evidence are recorded. |

### Day 13 Validation

Commands run after the Day 13 documentation updates:

```sh
python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py
make symbolic-lu-allocation-failure-gate
make docs-check
git diff --check
```

Results:

- `python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py`:
  PASS.
- `make symbolic-lu-allocation-failure-gate`: PASS; selected gate reported 3
  tests, 0 failures, 0 skipped, and 3054 assertions.
- `make docs-check`: PASS; Doxygen and API docs coverage completed with 18
  checked-in public headers, 18 generated reference pages, and 18 generated
  source pages.
- `git diff --check`: PASS.

## Day 14: Closeout and Retrospective Inputs

### Implementation Summary

Day 14 reconciled Sprint 200 items 200.1 through 200.6 against the evidence
artifacts, validation results, known residuals, and retrospective inputs. The
Sprint 200 selected-owner proof is closed for `sparse_symbolic_lu()` only.

The Day 14 artifact is:

- `docs/planning/EPIC_18/SPRINT_200/artifacts/day14-closeout-review.md`

### Final Item Status

| Item | Final status | Evidence |
| --- | --- | --- |
| 200.1 Owner Selection | Complete | Day 1 candidate intake and Day 2 owner selection choose exactly one additional owner: `sparse_symbolic_lu()`. |
| 200.2 Invariant Record | Complete | Days 3, 4, and 13 record lifecycle, invariant, and invariant-to-test traceability evidence. |
| 200.3 Harness Integration | Complete | Days 5 and 6 record minimal wrapper-allocation reachability for selected owner-owned allocations. |
| 200.4 Regression Tests | Complete | Days 7, 8, and 9 record failed-allocation, cleanup, stale-output, caller-input, and retry proof. |
| 200.5 Focused Gate | Complete | Day 10 records `make symbolic-lu-allocation-failure-gate` and registration guard coverage. |
| 200.6 Docs And Validation | Complete | Days 11 through 14 record claim-safe docs, integrated validation, review hardening, and closeout. |

### Final Validation Summary

The required full local validation for this branch's C and test changes passed
on Day 12:

```sh
make symbolic-lu-allocation-failure-gate
make symbolic-allocation-failure-gate
make source-list-check
make docs-check
make format
make lint
make test
git diff --check
```

Additional Day 13 review-hardening validation also passed:

```sh
python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py
make symbolic-lu-allocation-failure-gate
make docs-check
git diff --check
```

### Known Residuals

Sprint 200 leaves these areas explicitly unclaimed:

- broad allocation-failure coverage across the library;
- `sparse_analyze()` lifecycle cleanup;
- standalone etree, postorder, and colcount helper allocation failures;
- direct solver, eigensolver, graph, SVD, sparse matrix construction,
  conversion, IO, package/install, generated-tooling, platform, hosted CI,
  OS OOM, or concurrent allocation-hook reliability;
- state-of-the-art reliability support.

### Retrospective Inputs

| Area | Input |
| --- | --- |
| Completed work | One additional selected allocation-failure owner proof was completed for `sparse_symbolic_lu()`. |
| Validation | Focused gate, broader symbolic gate, source-list, docs, format, lint, full tests, and whitespace checks passed before closeout. |
| Deviation | The selected symbolic LU proof received its own Make gate so the selected-owner boundary remains explicit. |
| Deferred breadth | Analysis lifecycle, symbolic helper families, direct solvers, matrix construction, platform, package/install, generated tooling, and hosted CI proof remain future work. |
| Recommendation | Continue requiring selected-owner invariants before future allocation-failure implementation work. |

### Day 14 Validation

Commands run after the closeout artifact and status updates:

```sh
python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py
make symbolic-lu-allocation-failure-gate
make docs-check
git diff --check
```

Results:

- `python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py`:
  PASS.
- `make symbolic-lu-allocation-failure-gate`: PASS; selected gate reported 3
  tests, 0 failures, 0 skipped, and 3054 assertions.
- `make docs-check`: PASS; Doxygen and API docs coverage completed with 18
  checked-in public headers, 18 generated reference pages, and 18 generated
  source pages.
- `git diff --check`: PASS.
