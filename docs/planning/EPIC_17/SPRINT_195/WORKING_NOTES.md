# Sprint 195 Working Notes: Selected Reliability and Failure-Path Proof

## Sprint Goal

Add deterministic reliability evidence for one selected allocation-heavy or
failure-prone owner beyond prior proof lanes.

## Day 1: Reliability Intake

### Scope Trace

| Epic item | Day 1 intake interpretation |
| --- | --- |
| 195.1 Owner Selection | Rank one allocation-heavy or failure-prone owner by allocation density, cleanup complexity, user impact, deterministic hook fit, and current test gaps. |
| 195.2 Invariant Record | Define cleanup ownership, output publication, stale-output behavior, retry semantics, global-state restoration, and unsupported breadth before implementation. |
| 195.3 Harness Extension | Reuse `sparse_alloc_test_fail_after(...)` and `sparse_alloc_test_reset()` where possible, or add a narrow owner-local deterministic fail-at-count mechanism. |
| 195.4 Regression Tests | Add selected-owner tests for allocation failure, cleanup, stale-output suppression, and successful retry behavior. |
| 195.5 Focused Gate And Docs | Add a focused Make/CTest gate and maintainer/user wording that states the exact selected proof and retained non-claims. |
| 195.6 Validation | Run the focused gate, source-list checks as needed, `make format && make lint && make test` after C/H edits, and relevant docs checks. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Sprint 195 is allocated 168 hours to select one reliability owner, document invariants, extend deterministic failure injection, add regression tests, add a focused gate/docs, and validate. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day10-maintainability-reliability-gates.md` | Sprint 195 must select exactly one owner and prove failure status, cleanup, stale-output behavior, retry, and global-state restoration without claiming exhaustive reliability. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day4-owner-surface-inventory.md` | Existing reliability evidence is limited to iterative repeated-run handles and `sparse_matmul()` workspace allocation; one new selected owner remains the planned closure. |
| `docs/planning/EPIC_17/SPRINT_193/artifacts/day3-selected-cluster-contract.md` | Recent review-surface work reinforced the rule that process-global state must be restored before assertion macros or early-return helpers can exit. |
| `src/sparse_alloc_internal.c` and `src/sparse_alloc_internal.h` | The private deterministic allocation hook already supports fail-after countdown and reset for `sparse_malloc_array`, `sparse_calloc_array`, and idx-count wrappers. |
| `Makefile` and `CMakeLists.txt` | Existing focused gates are `make iterative-allocation-failure-gate` and `make matmul-allocation-failure-gate`; CTest labels already include `allocation_failure`, `iterative`, and `matmul`. |
| `docs/maintainer_guide.md` | Maintainer docs already describe the two selected allocation-failure proof lanes and retain broad non-claims for other solver families and allocation paths. |

### Current Allocation and Cleanup Surface Scan

The Day 1 scan counted source lines matching internal allocation wrappers,
direct `malloc`/`calloc`/`realloc`, and cleanup/error-return markers. Counts
are approximate signals for Day 2 scoring, not final selection proof.

| Candidate owner | Allocation signal | Cleanup/failure signal | Initial Day 1 interpretation |
| --- | ---: | ---: | --- |
| `src/sparse_ldlt_csc.c` | 55 | 29 | Highest allocation signal and direct-solver user impact, but algorithmic complexity and prior helper work make scope control important. |
| `src/sparse_lu_csr.c` | 37 | 36 | Strong allocation and cleanup density with public CSR LU behavior; likely high payoff but may require careful source-list and retry semantics review. |
| `src/sparse_qr.c` | 33 | 37 | Large cleanup signal and high user impact; recent QR review/comparison work raises review risk if touched too broadly. |
| `src/sparse_etree.c` | 30 | 32 | Allocation-heavy structural owner with many direct failure exits; public analysis/reordering users may benefit from proof. |
| `src/sparse_ldlt.c` | 29 | 19 | Direct LDLT allocation surface with public solve/factor APIs; needs inspection for stale-output and retry points. |
| `src/sparse_chol_csc.c` | 19 | 25 | Direct Cholesky CSC owner with public solve/factor paths and existing external comparison evidence. |
| `src/sparse_lu.c` | 18 | 15 | Linked-list LU owner with public output and workspace paths; external LU comparison evidence exists but allocation proof breadth is unknown. |
| `src/sparse_graph_coarsen.c` | 18 | 4 | Allocation-heavy graph helper surface; lower direct solver user impact but possible contained proof. |
| `src/sparse_svd_partial.c` | 14 | 16 | Public numerical path with several output publication points; proof could be valuable but numerical fixture setup may be heavier. |
| `src/sparse_reorder_amd_qg.c` | 14 | 9 | Realloc/growth path and ordering user impact are relevant; existing tests already include some overflow/failure checks. |
| `src/sparse_matrix.c` | 14 | 22 | Core matrix construction has broad user impact and existing allocation-hook smoke tests, but broad constructor coverage could exceed one sprint. |

### Existing Deterministic Proof Models

| Existing owner | Proof pattern | Reusable lesson |
| --- | --- | --- |
| Iterative repeated-run handles | `tests/test_iterative_handle_helpers.h` uses `sparse_alloc_test_fail_after(...)` around CG, GMRES, and MINRES prepare/growth calls and resets the hook before proceeding. | Keep proof family-local, assert handle state after failure, and prove successful recovery after reset. |
| `sparse_matmul()` workspace | `tests/test_matmul.c` enumerates fail-after sites, checks `SPARSE_ERR_ALLOC`, verifies stale output remains caller-owned, resets the hook, and proves retry success through `make matmul-allocation-failure-gate`. | Use a small fail-site table, assert stale-output suppression, and add a registration guard for the focused gate. |
| Allocation helper smoke tests | `tests/test_sparse_matrix.c` directly exercises `sparse_malloc_array`, `sparse_calloc_array`, idx wrappers, overflow, countdown, and reset semantics. | Treat the hook as private test infrastructure and avoid exposing it as public API. |
| AMD/QG and bucket tests | `tests/test_reorder_amd_qg.c` and `tests/test_graph_fm_buckets.c` include selected allocation/error checks. | Some owners already have narrow negative-path checks, but not full cleanup/stale-output/retry proof. |

### Candidate Owner Inventory

| Candidate | User impact | Deterministic hook fit | Current gap shape | Day 1 disposition |
| --- | --- | --- | --- | --- |
| LU CSR factor/solve owner | High for compressed direct-solver users. | Mixed: many allocations are direct `malloc`/`calloc`/`realloc`, so it may need wrapper conversion or owner-local checkpoints. | Cleanup and fill-in growth paths are allocation dense; stale-output and retry behavior need tracing. | Strong Day 2 candidate if proof can be scoped to one public entry point. |
| Elimination tree / symbolic analysis owner | Medium-high because it feeds analysis and direct solvers. | Good where internal wrappers are used; several direct allocations also need review. | Many cleanup labels and partial symbolic structures create useful cleanup proof potential. | Strong Day 2 candidate if output publication is clear. |
| Linked-list LU solve workspace owner | High for public LU solve APIs. | Mixed because several workspace allocations use direct `malloc`. | Temporary solve buffers have clear cleanup and retry semantics but may not cover factor construction. | Strong Day 2 candidate as a narrower lane than all LU. |
| Cholesky CSC factor/solve owner | High for SPD direct users. | Good where internal wrappers are present; direct workspace allocations need tracing. | Existing correctness evidence can support retry oracle checks; failure-path breadth remains selected-only. | Day 2 candidate. |
| LDLT dense/public owner | High for symmetric indefinite users. | Likely mixed; needs closer allocation publication tracing. | Public output and solve workspaces create useful reliability scope but may overlap with complex LDLT CSC surfaces. | Day 2 candidate with caution. |
| Core sparse matrix construction | Very high because every user creates matrices. | Existing allocation helper tests already cover hook mechanics; constructor proof would have broad stale-output implications. | Could exceed one sprint if it tries to cover shell construction, insertion growth, conversion, and IO. | Candidate only if narrowed to one constructor path. |
| QR factor/solve owner | High. | Mixed and recently touched by review-surface work. | Allocation and cleanup density are high, but behavior/tolerance risk is also high. | Defer unless Day 2 shows a narrow, isolated solve-workspace lane. |
| Partial SVD owner | Medium-high for advanced users. | Good wrapper presence in several paths. | Numerical fixtures and multi-output publication increase proof complexity. | Candidate but likely higher test-design cost. |

### Initial Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Selecting too broad an owner | Sprint 195 must completely prove one selected owner, not partially touch several owners. | Day 2 must pick one public entry point or tightly bounded owner lane with explicit rejected breadth. |
| Direct allocator use bypasses the hook | `sparse_alloc_test_fail_after(...)` only controls internal wrapper allocations. | Prefer owners already using wrappers or explicitly budget wrapper conversion/owner-local checkpoints. |
| Stale output semantics are unclear | Public APIs differ between out-pointer publication, in-place handles, and caller-owned buffers. | Day 3 must define publication and stale-output invariants before tests. |
| Global hook contamination | Assertion macros can return early while fail hooks or backend overrides remain set. | Store status locally, reset hooks/overrides, then assert; use cleanup labels for multi-step tests. |
| Retry proof masks cleanup bugs | A successful retry can pass while leaked or caller-owned state is mishandled. | Pair retry assertions with cleanup/stale-output assertions and, where available, counter-sensitive checks. |
| Source-list drift | New test binaries or production source files require Make/CMake/source-list parity. | Prefer adding tests to an existing proof-owner binary unless Day 4 requires a new target; run `make source-list-check` when registration changes. |
| Overclaiming reliability | One more selected proof does not mean broad allocation-failure safety. | Keep README and maintainer wording selected-owner-only with explicit non-claims. |

### Day 2 Scoring Questions

1. Which candidate has the best ratio of allocation density to deterministic
   hook coverage without requiring broad wrapper conversion?
2. Which candidate has a clear public output publication point where
   stale-output suppression can be asserted?
3. Which candidate can prove successful retry against an existing oracle,
   fixture, or baseline with low numerical ambiguity?
4. Which candidate can reuse an existing test binary and focused Make target
   without large CMake/source-list churn?
5. Which candidate closes the highest user-impact reliability gap while
   staying strictly selected-owner-only?

### Day 1 Validation

Commands run:

```sh
git status --short --branch
sed -n '304,336p' docs/planning/EPIC_17/PROJECT_PLAN.md
sed -n '1,115p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
rg --files docs/planning/EPIC_17 | rg 'SPRINT_187|SPRINT_193|SPRINT_195'
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_187/artifacts/day10-maintainability-reliability-gates.md
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_187/artifacts/day4-owner-surface-inventory.md
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_193/artifacts/day3-selected-cluster-contract.md
rg -n "fail_at|fail at|allocation failure|alloc failure|OOM|out of memory|sparse_malloc|calloc|realloc|malloc" tests src include scripts
sed -n '1,180p' src/sparse_alloc_internal.c
sed -n '1,180p' src/sparse_alloc_internal.h
rg -n "sparse_alloc_test|allocation-failure|allocation_failure|fail_after|matmul-allocation|iterative-allocation" Makefile CMakeLists.txt tests src docs README.md
rg -n "cleanup:|goto cleanup|goto fail|return SPARSE_ERR_ALLOC|return NULL|SPARSE_ERR_ALLOC" src tests include
rg -n "sparse_alloc_test_fail_after|SPARSE_ERR_ALLOC|stale|recovers|allocation_failure" tests/test_matmul.c tests/test_iterative_handle_helpers.h tests/test_sparse_matrix.c tests/test_reorder_amd_qg.c tests/test_graph_fm_buckets.c
sed -n '440,510p' docs/maintainer_guide.md
git diff --check
```

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

## Day 2: Owner Selection Scoring

### Scoring Method

Day 2 scores use a 1-5 scale where 5 is strongest for Sprint 195 selection.
The score favors one owner that can receive complete deterministic proof in
this sprint, not the broadest or most complex reliability surface.

| Criterion | Meaning |
| --- | --- |
| Allocation density | Number and variety of allocations or growth paths worth proving. |
| Cleanup complexity | Amount of partial-state cleanup and owned output release that can be asserted. |
| Stale-output risk | Whether the API publishes an out pointer, handle, or caller-visible buffer that must not look successful after failure. |
| Retry clarity | Whether a failed call can be followed by a successful call against the same fixture with clear expected results. |
| User impact | Importance of the owner to maintained public workflows. |
| Hook feasibility | How much of the owner can be driven by deterministic fail-at-count hooks without broad allocator redesign. |
| Review cost | Lower score means broad, risky, or noisy implementation; higher score means bounded and reviewable. |

### Ranked Reliability Owner Table

| Rank | Candidate | Allocation density | Cleanup complexity | Stale-output risk | Retry clarity | User impact | Hook feasibility | Review cost | Total | Disposition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `sparse_symbolic_cholesky()` symbolic out-struct owner in `src/sparse_etree.c` | 4 | 5 | 5 | 4 | 4 | 4 | 5 | 31 | Selected. Narrow owner with clear `sparse_symbolic_t` publication, cleanup via `sparse_symbolic_free`, existing `test_etree` fixtures, and mostly wrapper-controlled allocations. |
| 2 | `chol_csc_alloc()` / `chol_csc_workspace_alloc()` allocation owner in `src/sparse_chol_csc.c` | 4 | 4 | 5 | 4 | 4 | 2 | 5 | 28 | Fallback. Very clear out-pointer semantics and tests, but direct `calloc` use means more harness conversion before fail-at-count proof. |
| 3 | Linked-list `sparse_lu_solve()` workspace owner in `src/sparse_lu.c` | 3 | 3 | 2 | 4 | 5 | 2 | 4 | 23 | Deferred. Caller-owned `x` is modified during solve, making stale-output expectations less clean; direct `malloc` paths need conversion. |
| 4 | `lu_csr_solve()` / `lu_csr_solve_block()` workspace owner in `src/sparse_lu_csr.c` | 4 | 4 | 2 | 4 | 5 | 2 | 3 | 24 | Deferred. Good public value, but direct allocation and caller-owned output mutation make the proof less crisp than symbolic out-struct publication. |
| 5 | Narrow `sparse_matrix` constructor path in `src/sparse_matrix.c` | 3 | 4 | 4 | 5 | 5 | 3 | 2 | 26 | Deferred. User impact is highest, but defining one constructor path without implying broad sparse-matrix allocation coverage is risky. |
| 6 | `sparse_symbolic_lu()` in `src/sparse_etree.c` | 5 | 5 | 5 | 3 | 4 | 3 | 2 | 27 | Deferred. Too broad for the first Sprint 195 owner because it composes matrix construction, insertion, etree, postorder, colcount, Cholesky symbolic, and optional U publication. |
| 7 | Broad `src/sparse_lu_csr.c` factor/growth owner | 5 | 5 | 3 | 3 | 5 | 2 | 1 | 24 | Rejected for Sprint 195. Valuable, but factorization/growth behavior is too wide for one selected proof lane. |
| 8 | Broad `src/sparse_ldlt_csc.c` owner | 5 | 4 | 4 | 3 | 5 | 3 | 1 | 25 | Rejected for Sprint 195. Allocation density is highest, but algorithm and backend complexity make it a poor first owner for this proof sprint. |
| 9 | Broad QR factor/solve owner in `src/sparse_qr.c` | 4 | 5 | 3 | 3 | 5 | 2 | 1 | 23 | Rejected for Sprint 195. Recent QR work and tolerance/behavior sensitivity make this too noisy for selected reliability proof. |

### Selected Owner

Sprint 195 selects `sparse_symbolic_cholesky()` in `src/sparse_etree.c`.

Selected lane:

- public/internal owner: symbolic Cholesky structure construction;
- primary output: caller-provided `sparse_symbolic_t *sym`;
- primary cleanup owner: `sparse_symbolic_free(sym)`;
- primary proof binary: `tests/test_etree.c`;
- likely focused gate: a new Make target that builds and runs `test_etree`
  with symbolic allocation-failure registration checks;
- expected harness approach: reuse the private allocation hook and convert the
  selected owner's remaining direct `malloc` allocation for `sym->col_ptr` to
  the wrapper path if Day 3 confirms that is needed for deterministic proof.

### Fallback Owner

The fallback owner is the Cholesky CSC allocation/workspace construction lane:
`chol_csc_alloc()` and `chol_csc_workspace_alloc()` in `src/sparse_chol_csc.c`.

Fallback rationale:

- out-pointer semantics are very clear;
- existing tests already cover null, badarg, zero-size, grow, and workspace
  basics;
- the proof would be tightly bounded to object/workspace construction rather
  than all Cholesky factorization;
- direct `calloc` use would require wrapper conversion or an owner-local hook,
  which is why it ranks behind `sparse_symbolic_cholesky()`.

### Current Coverage Notes

| Surface | Existing assertions | Gap for Sprint 195 |
| --- | --- | --- |
| `sparse_symbolic_cholesky()` null/bad-shape behavior | `test_symbolic_null_args` and existing shape-path tests cover invalid inputs. | No deterministic allocation-failure test currently forces every selected allocation class to fail. |
| `sparse_symbolic_cholesky()` success behavior | `test_symbolic_1x1`, diagonal, tridiag, arrow, dense, known 5x5, and SuiteSparse symbolic-vs-numeric tests cover normal output shape. | No retry-after-failure proof currently shows a later successful call after fail hook reset. |
| `sparse_symbolic_free()` cleanup | `test_symbolic_free_zeroed` proves zeroed and NULL free are safe. | No test currently proves partially populated `sym` is cleared after selected allocation failures. |
| Existing focused gates | `make iterative-allocation-failure-gate` and `make matmul-allocation-failure-gate` own prior lanes. | No focused symbolic reliability gate exists yet. |

### Rejected Breadth

Sprint 195 will not claim or attempt:

- all etree allocation-failure coverage;
- all `sparse_analyze()` allocation-failure coverage;
- `sparse_symbolic_lu()` L/U allocation-failure coverage;
- direct-solver factorization allocation-failure coverage;
- sparse matrix construction, insertion, conversion, or IO allocation-failure
  coverage;
- QR, SVD, LDLT CSC, or LU CSR broad failure-path proof;
- thread-safe or concurrent allocation-failure behavior.

### Day 2 Acceptance

Item 195.1 is accepted for planning purposes: the selected owner is
`sparse_symbolic_cholesky()`, the fallback owner is Cholesky CSC
allocation/workspace construction, and the rejected breadth is explicit enough
to constrain Day 3 invariant work.

### Day 2 Validation

Commands run:

```sh
git status --short --branch
sed -n '58,105p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
sed -n '1,180p' docs/planning/EPIC_17/SPRINT_195/WORKING_NOTES.md
rg -n "LuCsr|lu_csr|return SPARSE_ERR_ALLOC|malloc|calloc|realloc|cleanup:|goto cleanup|\\*out|out\\)" src/sparse_lu_csr.c include/sparse_lu_csr.h tests/test_lu_csr.c
rg -n "etree|symbolic|sparse_analyze|return SPARSE_ERR_ALLOC|sparse_malloc|sparse_calloc|malloc|calloc|cleanup:|goto cleanup|\\*out|out\\)" src/sparse_etree.c src/sparse_analysis.c include/sparse_analysis.h tests/test_etree.c
rg -n "sparse_lu_|return SPARSE_ERR_ALLOC|malloc|calloc|cleanup:|goto cleanup|\\*x|\\*out|out\\)" src/sparse_lu.c include/sparse_lu.h tests/test_sparse_lu.c
rg -n "sparse_chol|chol_csc|return SPARSE_ERR_ALLOC|sparse_malloc|sparse_calloc|malloc|calloc|cleanup:|goto cleanup|\\*out|out\\)" src/sparse_chol_csc.c include/sparse_cholesky.h tests/test_chol_csc.c
sed -n '235,410p' src/sparse_etree.c
sed -n '821,990p' tests/test_etree.c
rg -n "test_etree|allocation_failure|add_sparse_test\\(test_etree|TEST_SRCS|test_etree.c" Makefile CMakeLists.txt
sed -n '415,665p' src/sparse_etree.c
sed -n '665,760p' src/sparse_etree.c
sed -n '1029,1250p' src/sparse_lu_csr.c
sed -n '779,850p' src/sparse_lu.c
sed -n '54,150p' src/sparse_chol_csc.c
sed -n '630,665p' src/sparse_chol_csc.c
git diff --check
```

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

## Day 3: Selected Owner Invariant Record

### Selected Owner Boundary

Sprint 195 proves the selected `sparse_symbolic_cholesky()` owner only:

- implementation owner: `src/sparse_etree.c`;
- declaration owner: `src/sparse_analysis_internal.h`;
- test owner: `tests/test_etree.c`;
- output owner: caller-provided `sparse_symbolic_t *sym`;
- cleanup owner: `sparse_symbolic_free(sym)`;
- selected success fixture family: small symbolic Cholesky fixtures already in
  `test_etree` (`1x1`, diagonal, tridiagonal, arrow, dense, known 5x5).

The selected owner excludes `sparse_symbolic_lu()`, `sparse_analyze()`,
standalone etree/postorder/colcount proof, and all direct solver numeric
factorization paths.

### Allocation and Publication Map

| Step | Allocation or publication point | Current hook status | Failure behavior to prove |
| ---: | --- | --- | --- |
| 0 | Validate arguments and shape. | Not allocation-related. | Invalid calls return `SPARSE_ERR_NULL` or `SPARSE_ERR_SHAPE` without touching caller-owned allocation state beyond documented behavior. |
| 1 | `memset(sym, 0, sizeof(*sym))`. | Not allocation-related. | On entered calls, stale caller-provided symbolic output is cleared before any allocation. |
| 2 | `n == 0`: allocate one-entry `sym->col_ptr` through `sparse_calloc_array`. | Hook-controlled. | Failure returns `SPARSE_ERR_ALLOC` with `sym` still zeroed; success publishes only `col_ptr` with `n == 0`, `nnz == 0`, `row_idx == NULL`. |
| 3 | `n > 0`: compute `col_ptr_len` and `col_ptr_bytes`. | Overflow guard, not hook-controlled. | Overflow returns `SPARSE_ERR_ALLOC` before allocation and before any output pointer is published. |
| 4 | Allocate `sym->col_ptr`. | Currently direct `malloc`; Day 4 should convert to `sparse_malloc_array` or provide equivalent owner-local hook. | Failure must return `SPARSE_ERR_ALLOC` with `sym->col_ptr == NULL`, `sym->row_idx == NULL`, `sym->n == 0`, and `sym->nnz == 0`. |
| 5 | Set `sym->n`, build `sym->col_ptr`, and set `sym->nnz`. | Publication to `sym`. | Any overflow or monotonicity failure must call `sparse_symbolic_free(sym)` and leave no stale `col_ptr`, `row_idx`, `n`, or `nnz`. |
| 6 | Allocate `sym->row_idx` through `sparse_malloc_idx_array`. | Hook-controlled. | Failure must free `sym->col_ptr`, clear output fields, and return `SPARSE_ERR_ALLOC`. |
| 7 | Allocate `child_head`, `child_next`, `marker`, and `tmp` through wrapper calls. | Hook-controlled. | Failure must free temporary arrays allocated earlier in the expression, free symbolic output, and return `SPARSE_ERR_ALLOC`. |
| 8 | Allocate `col_rows` and `col_nrows` through wrapper calls. | Hook-controlled. | Failure must free column-row arrays, child arrays, marker/temp arrays, symbolic output, and return `SPARSE_ERR_ALLOC`. |
| 9 | Allocate per-column propagated `col_rows[j]` through `sparse_malloc_array`. | Hook-controlled. | Failure must free every propagated row set, all remaining temporaries, symbolic output, and return `SPARSE_ERR_ALLOC`. |
| 10 | Successful loop completion. | Not failure-related. | Success publishes a fully owned `sym` whose `col_ptr`, `row_idx`, `n`, and `nnz` match existing fixture expectations. |

### Cleanup Invariants

| Invariant | Required assertion or evidence |
| --- | --- |
| Entered calls clear stale output before allocation. | A test initializes `sym` with stale allocated fields, forces selected failure, and asserts `sparse_symbolic_free`-safe zeroed state after return. |
| Allocation failure returns `SPARSE_ERR_ALLOC`. | Fail-at-count tests cover each selected allocation class. |
| Partial symbolic publication is not observable after failure. | Tests assert `sym.col_ptr == NULL`, `sym.row_idx == NULL`, `sym.n == 0`, and `sym.nnz == 0` after selected failures. |
| Caller-owned inputs remain caller-owned and reusable. | Retry test reuses or reconstructs the same fixture inputs after failure and validates success. |
| Temporary arrays are not transferred to caller ownership. | Failure tests pair with normal teardown and, where available, allocator/leak tooling rather than exposing internal pointers. |
| `sparse_symbolic_free(NULL)` and `sparse_symbolic_free` on zeroed structs remain safe. | Existing `test_symbolic_free_zeroed` stays registered and should remain in the focused gate. |

### Retry Semantics

The selected retry proof should:

1. build a small fixture with nonzero propagated row-set allocations, likely
   the existing known 5x5 symbolic fixture;
2. force a selected allocation failure in `sparse_symbolic_cholesky()`;
3. reset the private allocation hook before any assertion macro can return;
4. assert the failed `sym` is zeroed and safe to free;
5. call `sparse_symbolic_cholesky()` again without injection;
6. assert the same output shape and row-index values as the existing success
   fixture; and
7. free the successful `sym` with `sparse_symbolic_free`.

### Harness Requirements

Day 4 must design one of these paths:

| Harness question | Preferred answer |
| --- | --- |
| How should `sym->col_ptr` allocation be made deterministic? | Convert direct `malloc(col_ptr_bytes)` to `sparse_malloc_array(col_ptr_len, sizeof(idx_t), ...)` so the existing private hook covers it. |
| How should hook state be restored? | Store the status in a local, call `sparse_alloc_test_reset()`, then assert the stored status. |
| Where should tests live? | Add selected symbolic allocation-failure tests to `tests/test_etree.c`, keeping `test_etree` as the proof-owner binary. |
| Is a new public API needed? | No. The allocation hook remains private/internal test infrastructure. |
| Is CMake/CTest parity required? | Yes if labels or new test binaries are added; otherwise Make-focused gate plus existing `test_etree` registration may be enough. |

### Unsupported Breadth And Non-Claims

Sprint 195 will not claim:

- exhaustive allocation-failure coverage for `src/sparse_etree.c`;
- deterministic failure proof for `sparse_symbolic_lu()`;
- deterministic failure proof for `sparse_analyze()` publication cleanup;
- standalone proof for `sparse_etree_compute()`, `sparse_etree_postorder()`, or
  `sparse_colcount()`;
- direct solver factorization, solve, refinement, or condition-estimator
  allocation-failure proof;
- sparse matrix construction/insertion/conversion/IO allocation-failure proof;
- QR, SVD, LDLT CSC, LU CSR, graph, package/install, report-generation, or
  benchmark allocation-failure proof;
- concurrent, thread-safe, or asynchronous allocation-hook behavior.

### Invariant-To-Test Map

| Planned test | Invariants covered |
| --- | --- |
| `test_symbolic_cholesky_allocation_failure_clears_stale_output` | Entered-call stale-output clearing, failed allocation status, zeroed output after failure. |
| `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `sym->row_idx`, child-array, column-row array, and propagated row-set allocation failures clean all partial state. |
| `test_symbolic_cholesky_allocation_failure_recovers` | Hook reset, retry success, caller-owned input reuse, and success output parity. |
| Focused gate registration guard | Ensures selected symbolic allocation-failure tests remain reachable through the maintained focused gate. |

### Day 3 Acceptance

Item 195.2 is accepted for planning purposes. The selected owner has a concrete
allocation/publication map, cleanup invariant list, retry contract, unsupported
breadth list, and planned assertion map sufficient to drive Day 4 harness
design.

### Day 3 Validation

Commands run:

```sh
git status --short --branch
sed -n '106,145p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
sed -n '235,410p' src/sparse_etree.c
sed -n '821,990p' tests/test_etree.c
rg -n "typedef struct.*sparse_symbolic|sparse_symbolic_t|sparse_symbolic_free|symbolic structure|col_ptr|row_idx" include/sparse_analysis.h src/sparse_etree.c src/sparse_analysis.c src/sparse_chol_csc_internal.h
sed -n '210,285p' include/sparse_analysis.h
sed -n '224,235p' src/sparse_etree.c
rg -n "sparse_symbolic_cholesky\\(" src tests include
git diff --check
```

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

## Day 4: Harness Design

### Design Decision

Sprint 195 will reuse the existing private deterministic allocation-failure
hook instead of adding an owner-local fail-at-count global.

Selected harness approach:

1. Convert the selected `sparse_symbolic_cholesky()` direct `sym->col_ptr`
   allocation from `malloc(col_ptr_bytes)` to
   `sparse_malloc_array(col_ptr_len, sizeof(idx_t), ...)`.
2. Keep `sparse_alloc_test_fail_after(...)` and `sparse_alloc_test_reset()` as
   private test-only controls declared in `src/sparse_alloc_internal.h`.
3. Add symbolic Cholesky allocation-failure tests to the existing
   `tests/test_etree.c` proof-owner binary.
4. Add a focused Make target tentatively named
   `symbolic-allocation-failure-gate`.
5. Add a Python registration guard tentatively named
   `tests/test_symbolic_allocation_failure_gate_registration.py`.
6. Add or update CTest labels for `test_etree` only if Day 5/Day 10 confirms
   the focused gate should be selectable through CMake as well as Make.

This avoids a second process-global failure-control mechanism and keeps all
Sprint 195 deterministic failures inside the existing allocation wrapper
model.

### Harness API Contract

| Surface | Contract |
| --- | --- |
| `sparse_alloc_test_fail_after(long remaining)` | Private test hook. When `remaining == 0`, the next wrapped allocation fails with `SPARSE_ERR_ALLOC`; positive values count down through wrapped allocations. |
| `sparse_alloc_test_reset()` | Private reset. Must be called before and after every failure-injection case. |
| `sparse_malloc_array` / `sparse_calloc_array` / idx wrappers | Only wrapped allocation calls are controllable. Direct `malloc`, `calloc`, or `realloc` remain outside the hook unless converted. |
| `sparse_symbolic_cholesky()` | Selected owner must use wrapper-controlled allocation for every allocation class claimed by Sprint 195. |

No public header, public API, ABI, allocator replacement, or caller-visible
failure-injection interface is part of the design.

### Reset and Early-Return Rules

Every Sprint 195 test helper that sets the allocation hook must follow this
pattern:

```c
sparse_alloc_test_reset();
sparse_alloc_test_fail_after(fail_after);
sparse_err_t err = sparse_symbolic_cholesky(A, parent, postorder, cc, &sym);
sparse_alloc_test_reset();
ASSERT_ERR(err, SPARSE_ERR_ALLOC);
```

Additional rules:

- reset before arming the hook so prior tests cannot affect countdown;
- store the status in a local variable before any assertion macro can return;
- reset immediately after the selected call, before asserting the status or
  symbolic state;
- call `sparse_symbolic_free(&sym)` in cleanup paths even when the expected
  failed state is zeroed;
- reset again in helper cleanup if a test uses multiple selected failures;
- do not nest fail-after scopes;
- do not run the selected allocation-failure tests concurrently because the
  hook is process-global test infrastructure.

### Planned Failure Cases

| Case name | Fail-after intent | Allocation class |
| --- | --- | --- |
| `empty col_ptr` | `0` on an empty fixture | `sparse_calloc_array` for `n == 0` `sym->col_ptr`. |
| `nonempty col_ptr` | `0` on a non-empty fixture after conversion | `sparse_malloc_array` for `sym->col_ptr`. |
| `row_idx` | first post-`col_ptr` selected allocation | `sparse_malloc_idx_array(sym->nnz, ...)`. |
| `child_head` / `child_next` / `marker` / `tmp` | representative countdowns through the chained wrapper expression | workspace allocations before row propagation. |
| `col_rows` / `col_nrows` | representative countdowns after child workspace | column-row workspace arrays. |
| `propagated row set` | countdown reaching a per-column `col_rows[j]` allocation | row-set allocation inside the postorder loop. |

Day 5/Day 7 may refine exact countdown values after the `col_ptr` conversion
and a small fixture are compiled.

### Fixture Design

Preferred fixtures:

| Fixture | Use |
| --- | --- |
| Empty `0x0` matrix with zero-length inputs | Proves empty-matrix `col_ptr` failure and success semantics. |
| Existing known 5x5 symbolic fixture | Proves non-empty `col_ptr`, `row_idx`, workspace, propagated row-set, stale-output clearing, and retry against explicit expected rows. |

The known 5x5 fixture should be factored into a local helper only if doing so
keeps the test readable and does not create a new shared helper surface.

### Build and Source Ownership

| Surface | Day 4 design |
| --- | --- |
| `src/sparse_etree.c` | One bounded behavior-preserving allocation-wrapper conversion for `sym->col_ptr`; no algorithm change. |
| `src/sparse_analysis_internal.h` | No declaration change expected. |
| `src/sparse_alloc_internal.*` | No hook API change expected. |
| `tests/test_etree.c` | Add selected symbolic allocation-failure tests and `RUN_TEST(...)` entries. |
| `Makefile` | Add focused `symbolic-allocation-failure-gate` target after existing allocation-failure gates. |
| `CMakeLists.txt` | Consider adding an `allocation_failure` or `symbolic` label to `test_etree`; no new binary planned. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Add guard checking Make target, CMake/test registration if used, and required `RUN_TEST(...)` entries. |
| `build-metadata/library_sources.txt` | No change expected because no new library source file is planned. |
| `docs/maintainer_guide.md` | Later Day 11 update should document the focused gate and selected-only non-claim. |

### Review Risks

| Risk | Mitigation |
| --- | --- |
| The `sym->col_ptr` conversion changes behavior. | Use the same count and element size already computed; preserve existing overflow checks unless Day 5 shows duplication can be safely simplified. |
| Fail-after countdowns become brittle. | Name cases by allocation class and keep a registration guard, but allow Day 7 to adjust exact countdowns after implementation. |
| Assertion macros leak hook state. | Store status locally, reset immediately, then assert. |
| The focused gate overclaims all etree reliability. | Name the target and docs around symbolic Cholesky allocation failure only. |
| CMake labels broaden existing allocation-failure selectors unexpectedly. | If adding `allocation_failure` to `test_etree`, document that CTest selector includes selected lanes, not broad coverage. |

### Item 195.3 Status

Item 195.3 is designed but not yet implemented. The planned implementation is
bounded to wrapper conversion plus test/gate wiring for the selected
`sparse_symbolic_cholesky()` owner.

### Day 4 Validation

Commands run:

```sh
git status --short --branch
sed -n '146,185p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_195/artifacts/day3-selected-owner-invariant-record.md
rg -n "allocation-failure-gate|allocation_failure|test_.*allocation|sparse_alloc_test_fail_after|sparse_alloc_test_reset" Makefile CMakeLists.txt tests scripts docs/maintainer_guide.md
sed -n '1,120p' tests/test_matmul_allocation_failure_gate_registration.py
sed -n '282,305p' Makefile
sed -n '200,214p' CMakeLists.txt
sed -n '188,285p' tests/test_iterative_handle_helpers.h
sed -n '340,438p' tests/test_matmul.c
git diff --check
```

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

## Day 5: Harness Scaffold

### Implementation Summary

Day 5 implemented the minimal scaffold needed for the selected
`sparse_symbolic_cholesky()` proof:

| Surface | Change |
| --- | --- |
| `src/sparse_etree.c` | Converted the non-empty `sym->col_ptr` allocation from direct `malloc` to `sparse_malloc_array`, keeping the existing overflow guard and publishing `sym->n` only after successful allocation. |
| `tests/test_etree.c` | Included the private allocation hook header and added two smoke tests that prove fail-after reaches empty and non-empty selected-owner `col_ptr` allocation paths. |
| `Makefile` | Added `symbolic-allocation-failure-gate`, modeled on the existing allocation-failure gates. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Added a focused registration guard that checks Make wiring, `test_etree` CMake registration, and required `RUN_TEST(...)` entries. |

### Scaffold Tests Added

| Test | Purpose | Current status |
| --- | --- | --- |
| `test_symbolic_cholesky_allocation_hook_reaches_empty_col_ptr` | Forces the `n == 0` `sparse_calloc_array` path to fail and checks `sym` is zeroed after reset. | Passing through `make symbolic-allocation-failure-gate`. |
| `test_symbolic_cholesky_allocation_hook_reaches_nonempty_col_ptr` | Forces the newly wrapper-controlled non-empty `sym->col_ptr` allocation to fail and checks `sym` is zeroed after reset. | Passing through `make symbolic-allocation-failure-gate`. |

### Harness Reset Notes

Both scaffold tests use the Day 4 reset pattern:

1. reset before arming the private hook;
2. arm `sparse_alloc_test_fail_after(0)`;
3. store the `sparse_symbolic_cholesky()` status in a local variable;
4. reset the private hook before assertions;
5. assert status and symbolic empty state; and
6. reset again after cleanup.

This keeps the process-global hook from leaking if later assertions fail.

### Initial Failed Run

The first local focused-gate run failed because the empty-matrix scaffold used
`sparse_create(0, 0)`, which returns `NULL` before reaching the selected owner.
The test was corrected to use a zeroed stack `SparseMatrix` for the `n == 0`
contract. `sparse_symbolic_cholesky()` checks only `rows` and `cols` before
the empty allocation path, so this fixture reaches the selected owner without
depending on matrix storage.

### Day 6 Handoff

Day 6 should build on this scaffold by adding deterministic failure cases for
later selected-owner allocation classes:

- `sym->row_idx`;
- `child_head`, `child_next`, `marker`, and `tmp`;
- `col_rows` and `col_nrows`;
- at least one propagated `col_rows[j]` allocation in the postorder loop.

Day 6 should keep tests in `tests/test_etree.c`, preserve the immediate reset
pattern, and avoid expanding the claim to `sparse_symbolic_lu()` or
`sparse_analyze()`.

### Day 5 Validation

Commands run:

```sh
git status --short --branch
sed -n '186,225p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
sed -n '1,160p' docs/planning/EPIC_17/SPRINT_195/artifacts/day4-harness-design.md
sed -n '1,80p' tests/test_etree.c
sed -n '2818,2935p' tests/test_etree.c
rg -n "make_.*5x5|known_5x5|static SparseMatrix" tests/test_etree.c
sed -n '1,120p' tests/test_framework.h
sed -n '112,160p' Makefile
sed -n '96,115p' src/sparse_analysis_internal.h
clang-format -i src/sparse_etree.c tests/test_etree.c
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
make format-check
git diff --check
```

Validation results:

- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed after the empty-fixture
  correction; `test_etree` ran 99 tests, 0 failures, 0 skips, and 675
  assertions.
- `make format-check` passed.

Day 5 modified `.c` files, so final sprint validation will need
`make format && make lint && make test`. Day 5 ran the focused scaffold gate
and formatting check; full C validation remains scheduled for later sprint
days.

## Day 6: Selected Owner Harness Integration

### Implementation Summary

Day 6 extended the Day 5 scaffold into the selected owner's later allocation
checkpoints while preserving normal success behavior.

| Surface | Change |
| --- | --- |
| `tests/test_etree.c` | Added `make_known_5x5_symbolic_matrix()` so success and failure fixtures share one local known-5x5 symbolic matrix builder. |
| `tests/test_etree.c` | Added `SymbolicFailureCase` and `expect_symbolic_cholesky_allocation_failure()` to run fail-after cases with reset-before-assert behavior. |
| `tests/test_etree.c` | Added `test_symbolic_cholesky_allocation_failures_clear_partial_state` covering selected later allocation classes. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Updated required `RUN_TEST(...)` entries to include the Day 6 partial-state test. |

No additional production behavior change was needed beyond the Day 5
`sym->col_ptr` wrapper conversion. The selected owner still returns the same
success layouts for existing symbolic fixtures.

### Checkpoint Ordering Record

The known-5x5 fixture reaches these deterministic fail-after checkpoints
inside `sparse_symbolic_cholesky()` after parent/postorder/colcount preparation
has completed:

| Fail-after | Allocation class |
| ---: | --- |
| 0 | non-empty `sym->col_ptr` |
| 1 | `sym->row_idx` |
| 2 | `child_head` |
| 3 | `child_next` |
| 4 | `marker` |
| 5 | `tmp` |
| 6 | `col_rows` |
| 7 | `col_nrows` |
| 8 | first propagated row-set allocation in the postorder loop |

The Day 6 partial-state test covers fail-after values 1 through 8. Day 5
already covers fail-after 0 for both empty and non-empty `col_ptr` allocation
paths.

### Early-Return Restoration Checklist

| Rule | Day 6 status |
| --- | --- |
| Reset before arming hook. | Implemented in all symbolic allocation-failure helpers. |
| Store selected-owner status in a local before assertions. | Implemented. |
| Reset immediately after selected call and before `ASSERT_ERR`. | Implemented. |
| Free selected symbolic output after assertions. | Implemented with `sparse_symbolic_free(&sym)`. |
| Reset again during helper cleanup. | Implemented. |
| Do not arm hook during fixture setup. | Implemented; parent/postorder/colcount setup completes before injection. |

### Preserved Success-Path Validation

`make symbolic-allocation-failure-gate` runs the full existing `test_etree`
binary, including existing symbolic success fixtures before and after the new
allocation-failure tests. The Day 6 run completed with `100` tests, `0`
failures, `0` skips, and `748` assertions.

### Day 7 Handoff

Day 7 should turn the Day 6 integration into the formal failed-allocation
regression artifact by:

- deciding whether fail-after 0 should be included in the table-driven
  non-empty failure cases or remain separately tested;
- adding explicit allocation-site coverage documentation for every selected
  class;
- strengthening stale-output or partial-publication assertions if needed; and
- checking whether CMake labels should be updated now or left for the Day 10
  focused-gate definition.

### Day 6 Validation

Commands run:

```sh
git status --short --branch
sed -n '226,267p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
sed -n '560,650p' docs/planning/EPIC_17/SPRINT_195/WORKING_NOTES.md
sed -n '235,410p' src/sparse_etree.c
sed -n '980,1045p' tests/test_etree.c
clang-format -i tests/test_etree.c
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
make format-check
git diff --check
```

Validation results:

- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed; `test_etree` ran 100 tests,
  0 failures, 0 skips, and 748 assertions.
- `make format-check` passed.

Day 6 modified `.c` files, so final sprint validation will need
`make format && make lint && make test`. Day 6 ran the focused owner gate and
format check; full C validation remains scheduled for later sprint days.

## Day 7: Failed Allocation Regression Tests

### Implementation Summary

Day 7 converted the selected-owner scaffold into formal failed-allocation
regression coverage.

| Surface | Change |
| --- | --- |
| `tests/test_etree.c` | Renamed the partial-state checkpoint test to `test_symbolic_cholesky_allocation_failures_clear_partial_state`, matching the planned regression-test name from the invariant record. |
| `tests/test_etree.c` | Added `assert_known_5x5_symbolic_matrix_intact()` and called it after each selected known-5x5 allocation failure to prove caller-owned fixture input remains usable and unchanged. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Extended the guard to require the formal regression `RUN_TEST(...)` entry and every selected fail-after case name/count. |

### Allocation-Site Coverage Map

| Allocation class | Fail-after | Test owner | Assertions |
| --- | ---: | --- | --- |
| Empty `sym->col_ptr` | 0 on empty fixture | `test_symbolic_cholesky_allocation_hook_reaches_empty_col_ptr` | `SPARSE_ERR_ALLOC`, reset before assertion, `sym` cleared. |
| Non-empty `sym->col_ptr` | 0 on 1x1 fixture | `test_symbolic_cholesky_allocation_hook_reaches_nonempty_col_ptr` | `SPARSE_ERR_ALLOC`, reset before assertion, `sym` cleared. |
| `sym->row_idx` | 1 on known 5x5 fixture | `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `SPARSE_ERR_ALLOC`, `sym` cleared, caller-owned matrix intact. |
| `child_head` | 2 on known 5x5 fixture | `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `SPARSE_ERR_ALLOC`, `sym` cleared, caller-owned matrix intact. |
| `child_next` | 3 on known 5x5 fixture | `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `SPARSE_ERR_ALLOC`, `sym` cleared, caller-owned matrix intact. |
| `marker` | 4 on known 5x5 fixture | `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `SPARSE_ERR_ALLOC`, `sym` cleared, caller-owned matrix intact. |
| `tmp` | 5 on known 5x5 fixture | `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `SPARSE_ERR_ALLOC`, `sym` cleared, caller-owned matrix intact. |
| `col_rows` | 6 on known 5x5 fixture | `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `SPARSE_ERR_ALLOC`, `sym` cleared, caller-owned matrix intact. |
| `col_nrows` | 7 on known 5x5 fixture | `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `SPARSE_ERR_ALLOC`, `sym` cleared, caller-owned matrix intact. |
| Propagated row set | 8 on known 5x5 fixture | `test_symbolic_cholesky_allocation_failures_clear_partial_state` | `SPARSE_ERR_ALLOC`, `sym` cleared, caller-owned matrix intact. |

### Deferred Breadth

No selected `sparse_symbolic_cholesky()` allocation class is intentionally
deferred. The retained non-claims remain:

- `sparse_symbolic_lu()` allocation-failure behavior;
- `sparse_analyze()` publication cleanup;
- standalone etree, postorder, and colcount failure paths;
- direct solver, QR, SVD, graph, sparse matrix construction/conversion/IO, and
  package/report/benchmark allocation paths;
- real operating-system OOM and concurrent fail-hook behavior.

### Day 8 Handoff

Day 8 should strengthen cleanup and stale-output proof by seeding `sym` with
real stale owned allocations before selected failure paths, then confirming the
selected owner frees and clears that stale state before returning
`SPARSE_ERR_ALLOC`.

### Day 7 Validation

Commands run:

```sh
git status --short --branch
sed -n '268,310p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
sed -n '980,1095p' tests/test_etree.c
sed -n '1,120p' tests/test_symbolic_allocation_failure_gate_registration.py
clang-format -i tests/test_etree.c
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
make format-check
git diff --check
```

Validation results:

- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed.
- `make format-check` passed.

Day 7 modified `.c` files, so final sprint validation will need
`make format && make lint && make test`. Day 7 ran the focused owner gate and
format check; full C validation remains scheduled for later sprint days.

## Day 8: Cleanup and Stale-Output Proof

### Implementation Summary

Day 8 strengthened the selected-owner failure tests from empty-output checks to
cleanup-specific proof.

| Surface | Change |
| --- | --- |
| `tests/test_etree.c` | Added `assert_symbolic_failure_free_safe(...)`, which verifies an allocation-failed `sparse_symbolic_t` is empty, calls `sparse_symbolic_free(...)` twice, and rechecks the empty state after each cleanup. |
| `tests/test_etree.c` | Replaced direct one-shot cleanup calls in selected failure tests with the new repeated-cleanup helper. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Added a guard requirement for the cleanup helper call so the focused gate keeps cleanup assertions visible. |
| `artifacts/day3-selected-owner-invariant-record.md` | Added a Day 8 cleanup proof update tying the final cleanup assertions back to the invariant record. |
| `artifacts/day8-cleanup-stale-output-proof.md` | Recorded cleanup coverage, assertion trace, diagnostic notes, and retained non-claims. |

### Cleanup Proof Scope

The Day 8 proof covers:

- stale scalar output suppression for entered allocation-failure paths;
- empty `sym->col_ptr` and non-empty `sym->col_ptr` allocation failures;
- `row_idx`, child workspace, marker/temp workspace, column-row workspace, and
  propagated row-set allocation failures;
- caller-owned known-5x5 fixture preservation before cleanup validation;
- `sparse_symbolic_free(&sym)` after allocation failure;
- repeated `sparse_symbolic_free(&sym)` after allocation failure.

### Diagnostic Boundary

No allocator-counter assertion was added because the private allocation harness
does not expose a reliable per-test outstanding-allocation counter. The Day 8
diagnostic proof therefore stays bounded to deterministic failure injection,
explicit empty-state assertions, repeated cleanup calls, and the focused gate.

### Day 8 Validation

Commands run:

```sh
git status --short --branch
sed -n '900,1160p' tests/test_etree.c
rg -n "sanitize|asan|valgrind|allocation-failure-gate|symbolic-allocation" Makefile
clang-format -i tests/test_etree.c
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
make format-check
git diff --check
```

Validation results:

- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed; `test_etree` ran 100 tests,
  0 failures, 0 skips, and 884 assertions.
- `make format-check` passed.
- `git diff --check` passed.

Day 8 modified `.c` files, so final sprint validation will need
`make format && make lint && make test`. Day 8 ran the focused owner gate and
format check; full C validation remains scheduled for later sprint days.

## Day 9: Successful Retry Proof

### Implementation Summary

Day 9 added a selected-owner retry proof for
`sparse_symbolic_cholesky()`.

| Surface | Change |
| --- | --- |
| `tests/test_etree.c` | Added `assert_known_5x5_symbolic_output(...)` and reused it in the baseline known-5x5 success fixture. |
| `tests/test_etree.c` | Added `test_symbolic_cholesky_allocation_failures_recover_on_retry`, which forces each selected allocation failure, resets the hook, validates failed-output cleanup, then retries with the same fixture inputs. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Added guard coverage for the retry test and the `{"col_ptr", 0}` retry checkpoint. |
| `artifacts/day3-selected-owner-invariant-record.md` | Added a Day 9 retry proof update tying the implemented retry sequence back to the invariant record. |
| `artifacts/day9-successful-retry-proof.md` | Recorded retry sequence, selected checkpoints, ordering assumptions, and retained non-claims. |

### Retry Proof Scope

The retry test covers fail-after checkpoints 0 through 8 on the known-5x5
symbolic fixture:

- `sym->col_ptr`;
- `sym->row_idx`;
- `child_head`;
- `child_next`;
- `marker`;
- `tmp`;
- `col_rows`;
- `col_nrows`;
- first propagated row-set allocation.

For each checkpoint, the test verifies `SPARSE_ERR_ALLOC`, confirms the
caller-owned matrix is intact, proves the failed output remains free-safe, then
reruns `sparse_symbolic_cholesky()` using the same matrix, parent, postorder,
and column-count arrays. The retry output must match the exact known-5x5
symbolic row-index oracle.

### Ordering Assumptions

The retry proof keeps the existing failure-hook ordering rule: reset before
arming, reset immediately after the selected call, then assert the stored
status. The focused gate runs the existing success fixtures, allocation
failure tests, cleanup tests, and retry test together in the same `test_etree`
process.

### Day 9 Validation

Commands run:

```sh
git status --short --branch
sed -n '310,355p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
sed -n '991,1115p' tests/test_etree.c
sed -n '1,90p' tests/test_symbolic_allocation_failure_gate_registration.py
clang-format -i tests/test_etree.c
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
make format-check
git diff --check
```

Validation results:

- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed; `test_etree` ran 101 tests,
  0 failures, 0 skips, and 1262 assertions.
- `make format-check` passed.
- `git diff --check` passed.

Day 9 modified `.c` files, so final sprint validation will need
`make format && make lint && make test`. Day 9 ran the focused owner gate and
format check; full C validation remains scheduled for later sprint days.

## Day 10: Focused Gate Definition

### Implementation Summary

Day 10 finalized the selected symbolic reliability focused gate.

| Surface | Change |
| --- | --- |
| `CMakeLists.txt` | Added `etree;symbolic;allocation_failure` labels to `test_etree` so CMake users can select the selected symbolic Cholesky reliability proof. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Added a guard requirement for the `test_etree` CTest label wiring. |
| `artifacts/day10-focused-gate-definition.md` | Recorded the primary Make gate, CMake selector, covered tests, drift guard, and retained non-claims. |

### Maintained Gate

The primary focused command is:

```sh
make symbolic-allocation-failure-gate
```

The target runs `tests/test_symbolic_allocation_failure_gate_registration.py`
and then runs the full `test_etree` binary. That keeps the selected failure,
cleanup, retry, and existing success fixtures in one process.

### CMake Selector

`test_etree` now has these labels:

```cmake
LABELS "etree;symbolic;allocation_failure"
```

Focused CMake validation can use:

```sh
ctest --test-dir <build-dir> -L symbolic --output-on-failure
```

The broader `ctest -L allocation_failure` selector now represents maintained
selected allocation-failure proof lanes only: iterative repeated-run handles,
`sparse_matmul()` workspace failures, and the selected symbolic Cholesky lane.

### Day 10 Validation

Commands run:

```sh
git status --short --branch
sed -n '355,405p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
sed -n '285,312p' Makefile
rg -n "add_sparse_test\\(test_etree\\)|set_tests_properties|LABELS|allocation_failure|test_etree" CMakeLists.txt
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
cmake -S . -B build/sprint195-day10-cmake
cmake --build build/sprint195-day10-cmake --target test_etree --parallel 1
ctest -N --test-dir build/sprint195-day10-cmake -L symbolic
ctest --test-dir build/sprint195-day10-cmake -L symbolic --output-on-failure
make format-check
git diff --check
```

Validation results:

- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed; `test_etree` ran 101 tests,
  0 failures, 0 skips, and 1262 assertions.
- `cmake -S . -B build/sprint195-day10-cmake` passed.
- `cmake --build build/sprint195-day10-cmake --target test_etree --parallel 1`
  passed.
- `ctest -N --test-dir build/sprint195-day10-cmake -L symbolic` passed and
  selected `test_etree` as the only `symbolic` test.
- `ctest --test-dir build/sprint195-day10-cmake -L symbolic --output-on-failure`
  passed with 1 of 1 tests passing.
- `make format-check` passed.
- `git diff --check` passed.

Day 10 modified `CMakeLists.txt` and test guard wiring. Final sprint
validation will still need `make format && make lint && make test` because
earlier sprint days modified `.c` files.

## Day 11: Documentation and Claim Boundaries

### Implementation Summary

Day 11 published the selected symbolic allocation-failure proof in public and
maintainer documentation with explicit claim boundaries.

| Surface | Change |
| --- | --- |
| `README.md` | Added `make symbolic-allocation-failure-gate` to selected allocation-failure proof wording and the command map. |
| `README.md` | Added repeated-run guidance text limiting symbolic allocation-failure proof to selected `sparse_symbolic_cholesky()` output cleanup, stale-output suppression, and retry behavior. |
| `INSTALL.md` | Added a local-only support-readiness row for selected allocation-failure proof lanes. |
| `docs/maintainer_guide.md` | Added the `test_etree` Sprint 195 bounded allocation-failure owner with test names, Make gate, CTest selector, registration guard, invariant artifacts, and retained non-claims. |
| `artifacts/day11-claim-boundaries.md` | Recorded earned claim, documentation surfaces, and retained non-claims. |

### Earned Claim

Sprint 195 now claims only that `make symbolic-allocation-failure-gate`
provides a focused local proof for selected `sparse_symbolic_cholesky()`
allocation-failure status, symbolic output cleanup, stale-output suppression,
repeated cleanup after failure, and retry-after-reset behavior on bounded
fixtures.

### Retained Non-Claims

The Day 11 wording keeps these out of scope:

- broad allocation-failure coverage;
- `sparse_symbolic_lu()` and `sparse_analyze()` allocation-failure proof;
- standalone etree, postorder, and colcount helper allocation-failure proof;
- direct-solver, eigensolver, graph, sparse matrix construction, conversion,
  IO, package/install, and generated-tooling allocation-failure proof;
- OS OOM behavior;
- concurrent allocation-hook behavior;
- hosted CI proof for the local gate unless a future hosted lane names it;
- platform, package, ABI, performance, release, or state-of-the-art
  reliability proof.

### Day 11 Validation

Commands run:

```sh
git status --short --branch
sed -n '120,155p' README.md
sed -n '292,310p' README.md
sed -n '580,602p' README.md
sed -n '94,110p' INSTALL.md
sed -n '440,520p' docs/maintainer_guide.md
rg -n "allocation-failure|allocation_failure|symbolic Cholesky|state-of-the-art|reliability|selected allocation" README.md docs/maintainer_guide.md INSTALL.md docs
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
git diff --check
```

Validation results:

- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed; `test_etree` ran 101 tests,
  0 failures, 0 skips, and 1262 assertions.
- Targeted claim-surface grep confirmed the new symbolic gate and non-claim
  wording in `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and Sprint
  195 artifacts.
- `make format-check` passed.
- `git diff --check` passed.

Day 11 changed documentation only, but the focused owner gate was rerun to
confirm the documented command remains valid. Final sprint validation still
needs `make format && make lint && make test` because earlier sprint days
modified `.c` files.

## Day 12: Focused Validation and Source Ownership

### Implementation Summary

Day 12 ran the focused Sprint 195 validation and ownership checks without
requiring implementation fixes.

| Surface | Result |
| --- | --- |
| Source-list ownership | `make source-list-check` passed with 49 library sources. |
| Focused registration guard | `python3 tests/test_symbolic_allocation_failure_gate_registration.py` passed. |
| Focused Make gate | `make symbolic-allocation-failure-gate` passed. |
| CMake ownership | `cmake -S . -B build/sprint195-day12-cmake` and targeted `test_etree` build passed. |
| CMake selector | `ctest -N -L symbolic` selected only `test_etree`; `ctest -L symbolic` passed. |
| Claim-boundary docs | Targeted grep found the selected symbolic gate and non-claim wording in public, maintainer, and Sprint 195 artifacts. |
| Formatting and whitespace | `make format-check` and `git diff --check` passed. |
| Artifact | `artifacts/day12-focused-validation-source-ownership.md` records the focused validation log and remaining Day 13 risk. |

### Day 12 Validation

Commands run:

```sh
git status --short --branch
sed -n '430,485p' docs/planning/EPIC_17/SPRINT_195/PLAN.md
make source-list-check
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
cmake -S . -B build/sprint195-day12-cmake
cmake --build build/sprint195-day12-cmake --target test_etree --parallel 1
ctest -N --test-dir build/sprint195-day12-cmake -L symbolic
ctest --test-dir build/sprint195-day12-cmake -L symbolic --output-on-failure
rg -n "symbolic-allocation-failure-gate|sparse_symbolic_cholesky\\(\\)|broad allocation-failure|state-of-the-art reliability|Local selected allocation-failure proof|ctest --test-dir <build-dir> -L symbolic" README.md INSTALL.md docs/maintainer_guide.md docs/planning/EPIC_17/SPRINT_195
make format-check
git diff --check
```

Validation results:

- `make source-list-check` passed with 49 library sources.
- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed; `test_etree` ran 101 tests,
  0 failures, 0 skips, and 1262 assertions.
- `cmake -S . -B build/sprint195-day12-cmake` passed.
- `cmake --build build/sprint195-day12-cmake --target test_etree --parallel 1`
  passed.
- `ctest -N --test-dir build/sprint195-day12-cmake -L symbolic` passed and
  selected `test_etree` as the only `symbolic` test.
- `ctest --test-dir build/sprint195-day12-cmake -L symbolic --output-on-failure`
  passed with 1 of 1 tests passing.
- Targeted claim-surface grep confirmed the new symbolic gate and non-claim
  wording in `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and Sprint
  195 artifacts.
- `make format-check` passed.
- `git diff --check` passed.

No Day 12 fixes were required. Day 13 still needs the full `make format`,
`make lint`, and `make test` pass because Sprint 195 modified `.c` files.

## Day 13: Full Quality Gate

### Implementation Summary

Day 13 ran the full Sprint 195 quality gate and reran the focused
allocation-failure proof checks. No implementation or documentation fixes were
required.

| Surface | Result |
| --- | --- |
| Formatting | `make format` passed. |
| Diff sanity | `git diff -- src/sparse_etree.c tests/test_etree.c | sed -n '1,220p'` confirmed the active implementation/test diff remained the intended Sprint 195 allocation-failure proof work. |
| Whitespace | `git diff --check` passed before and after the focused guard reruns. |
| Lint | `make lint` passed, including strict warning builds, clang-tidy, and cppcheck. |
| Full test suite | `make test` passed and ended with `All tests passed.` |
| Focused registration guard | `python3 tests/test_symbolic_allocation_failure_gate_registration.py` passed. |
| Focused Make gate | `make symbolic-allocation-failure-gate` passed; `test_etree` ran 101 tests, 0 failures, 0 skips, and 1262 assertions. |
| Claim-boundary docs | Targeted grep found the selected symbolic gate and non-claim wording in public, maintainer, and Sprint 195 artifacts. |
| Artifact | `artifacts/day13-full-quality-gate.md` records the full quality-gate evidence and remaining Day 14 risk. |

### Day 13 Validation

Commands run:

```sh
make format
git diff --stat
git diff -- src/sparse_etree.c tests/test_etree.c | sed -n '1,220p'
git diff --check
make lint
make test
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
rg -n "symbolic-allocation-failure-gate|sparse_symbolic_cholesky\\(\\)|broad allocation-failure|state-of-the-art reliability|Local selected allocation-failure proof|ctest --test-dir <build-dir> -L symbolic" README.md INSTALL.md docs/maintainer_guide.md docs/planning/EPIC_17/SPRINT_195
git diff --check
```

Validation results:

- `make format` passed.
- `git diff --check` passed before and after focused guard reruns.
- `make lint` passed.
- `make test` passed and ended with `All tests passed.`
- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `make symbolic-allocation-failure-gate` passed; `test_etree` ran 101 tests,
  0 failures, 0 skips, and 1262 assertions.
- Targeted claim-surface grep confirmed the symbolic gate and non-claim wording
  in `README.md`, `INSTALL.md`, `docs/maintainer_guide.md`, and Sprint 195
  artifacts.

No Day 13 fixes were required. Day 14 should focus on final review packaging,
retrospective preparation, and accumulated-diff inspection.

## Day 14: Closeout and Review Package

### Implementation Summary

Day 14 packaged Sprint 195 for review with item-to-evidence traceability,
scope-control notes, final residuals, and a reviewer checklist.

| Surface | Result |
| --- | --- |
| Selected owner | `sparse_symbolic_cholesky()` remains the only selected reliability owner. |
| Implementation scope | Code changes are limited to wrapper-controlling the selected non-empty `sym->col_ptr` allocation and adding focused tests/gate wiring. |
| Regression scope | `tests/test_etree.c` covers allocation hook reachability, partial-state cleanup, stale-output suppression, repeated cleanup, caller-owned input preservation, and retry-after-reset behavior. |
| Focused gate | `make symbolic-allocation-failure-gate` remains the documented focused local gate. |
| CTest selector | `test_etree` carries `etree;symbolic;allocation_failure` labels for selected CTest execution. |
| Documentation scope | `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` state the selected symbolic proof and retained non-claims. |
| Full validation | Day 13 recorded passing `make format`, `make lint`, `make test`, focused guard, focused Make gate, docs grep, and whitespace checks. |
| Artifact | `artifacts/day14-closeout-review-package.md` records item-to-evidence traceability, review checklist, closeout checks, and residuals. |

### Item-to-Evidence Traceability

| Item | Evidence |
| --- | --- |
| 195.1 Owner Selection | Day 1 and Day 2 artifacts plus working notes select `sparse_symbolic_cholesky()` by allocation density, cleanup complexity, user impact, and testability. |
| 195.2 Invariant Record | Day 3 artifact records cleanup, publication, stale-output, retry, caller-owned input, and unsupported-breadth invariants. |
| 195.3 Harness Extension | Days 4 through 6 record reuse of `sparse_alloc_test_fail_after(...)` and wrapper-controlled allocation for the selected owner. |
| 195.4 Regression Tests | Days 7 through 9 record failed allocation, cleanup, stale-output suppression, and retry-after-reset coverage in `tests/test_etree.c`. |
| 195.5 Focused Gate And Docs | Days 10 and 11 record `make symbolic-allocation-failure-gate`, CTest labels, registration guard, and claim-safe README/INSTALL/maintainer wording. |
| 195.6 Validation | Days 12 and 13 record focused source ownership and full `make format`, `make lint`, and `make test` validation. |

### Day 14 Validation

Commands run:

```sh
git status --short --branch
git diff --stat
git diff -- CMakeLists.txt Makefile src/sparse_etree.c tests/test_etree.c tests/test_symbolic_allocation_failure_gate_registration.py | sed -n '1,260p'
git diff -- README.md INSTALL.md docs/maintainer_guide.md | sed -n '1,260p'
make source-list-check
python3 tests/test_symbolic_allocation_failure_gate_registration.py
git ls-files --others --exclude-standard
rg -n "195\\.1|195\\.2|195\\.3|195\\.4|195\\.5|195\\.6|Owner Selection|Invariant Record|Harness|Regression Tests|Focused Gate|Validation" docs/planning/EPIC_17/SPRINT_195/WORKING_NOTES.md docs/planning/EPIC_17/SPRINT_195/artifacts
```

Validation results:

- `make source-list-check` passed with 49 library sources.
- `python3 tests/test_symbolic_allocation_failure_gate_registration.py`
  passed.
- `git ls-files --others --exclude-standard` listed only Sprint 195 day
  artifacts that should be included with the sprint package.
- Item-traceability grep confirmed evidence for items 195.1 through 195.6 in
  working notes and artifacts.

### Final Residuals and Non-Claims

- Sprint 195 proves only the selected `sparse_symbolic_cholesky()` output
  allocation-failure path on bounded fixtures.
- It does not prove broad allocation-failure coverage, OS OOM behavior,
  concurrent allocation-hook behavior, platform parity, package/install
  readiness, generated-tooling reliability, performance, release readiness, or
  state-of-the-art reliability.
- Future reliability proof work should select another explicit owner and record
  invariant, harness, test, gate, documentation, and validation evidence before
  widening claims.
