# Sprint 201 Working Notes: Additional Review-Surface Reduction

## Sprint Goal

Reduce one large QR, LDLT, SVD, etree, integration, or direct-solver review
surface without changing behavior.

## Day 1: Large Surface Intake

### Scope Trace

| Epic item | Day 1 intake interpretation | Initial artifact |
| --- | --- | --- |
| 201.1 Candidate Ranking | Rank current large source/test surfaces by line count, review risk, ownership clarity, and extraction feasibility on Day 2. | Candidate inventory and ranking inputs. |
| 201.2 Cluster Selection | Select exactly one cluster after ranking and record no-behavior-change boundaries before extraction. | Boundary checklist placeholder. |
| 201.3 Helper Or Module Extraction | Prefer behavior-preserving helper or module extraction only where it reduces review burden. | Extraction-shape decision placeholder. |
| 201.4 Ownership Guard | Add or update focused guards only after the selected cluster and registration surfaces are known. | Guard pattern inventory. |
| 201.5 Focused Regression | Run focused tests and add narrow regressions only if extraction safety needs more evidence. | Focused regression matrix. |
| 201.6 Validation And Docs | Run source-list/CMake checks as needed, full C quality gate for `.c`/`.h` changes, and maintainer docs updates. | Validation matrix and documentation checklist. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Sprint 201 is a 166-hour sprint to reduce one large review surface without behavior changes. |
| `docs/planning/EPIC_18/SPRINT_201/PLAN.md` | Day 1 is intake only; Day 2 ranks candidates and Day 3 selects the cluster. |
| `docs/planning/EPIC_17/SPRINT_193/RETROSPECTIVE.md` | Sprint 193 successfully reduced one QR external-reference cluster and preserved a focused proof-owner boundary. |
| `docs/planning/EPIC_17/SPRINT_193/artifacts/day1-review-surface-intake.md` | Reusable intake pattern: measure large files, function density, source-list owners, and no-behavior-change rules. |
| `docs/planning/EPIC_17/SPRINT_193/artifacts/day2-candidate-ranking.md` | Reusable ranking pattern: prioritize one cohesive test/helper extraction over production source movement unless production movement is clearly bounded. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Additional review-surface reduction remains a future selected-cluster closure target after Sprint 200. |

### Current Large Source/Test Inventory

The Day 1 inventory uses current line counts after Sprint 200 landed. Counts
are review-risk signals only; Day 2 will rank clusters by cohesion and
behavior-preservation risk before selecting any implementation target.

| File | Lines | Function-density signal | RUN_TEST count | Day 1 risk tag |
| --- | ---: | ---: | ---: | --- |
| `tests/test_ldlt_csc.c` | 3469 | 110 | 100 | Largest current C test surface; prior LDLT CSC extraction means incremental payoff must be chosen carefully. |
| `tests/test_etree.c` | 3306 | 118 | 107 | High helper density and recent Sprint 200 symbolic LU additions; likely fixture/helper clusters but risk of touching new reliability proof. |
| `tests/test_integration.c` | 3279 | 54 | 58 | Broad cross-solver scope; high risk of accidental multi-cluster extraction. |
| `tests/test_qr.c` | 3040 | 71 | 79 | Still large after Sprint 193; remaining QR clusters may be cohesive but must avoid duplicating prior external-reference work. |
| `tests/test_svd.c` | 3029 | 85 | 114 | Large numerical test surface with helper-friendly precedent and likely focused clusters. |
| `tests/test_ldlt.c` | 3006 | 92 | 89 | Large direct-solver surface with tolerance/status behavior that must be preserved. |
| `tests/test_iterative.c` | 2929 | 86 | 85 | Large iterative surface with convergence, allocation, and handle-lifetime risk. |
| `tests/test_graph.c` | 2764 | 65 | 61 | Large graph/reorder surface with subtle deterministic and heuristic behavior. |
| `tests/test_chol_csc.c` | 2554 | 108 | 92 | High helper density and direct-solver relevance. |
| `tests/test_chol_csc_supernodal.c` | 2504 | 72 | 62 | Focused supernodal surface with existing helper-header precedent. |
| `tests/test_normalize_report_index.py` | 2419 | N/A | N/A | Large Python test surface, but outside the primary C review-surface examples unless Day 2 finds stronger fit. |
| `scripts/run_external_comparison.py` | 2306 | N/A | N/A | Large Python implementation surface; extraction could affect generated evidence semantics. |
| `tests/test_reorder_nd.c` | 2304 | N/A | N/A | Large graph/reorder test surface with deterministic behavior risk. |
| `tests/test_eigs.c` | 2155 | N/A | N/A | Large eigensolver test surface; numerical behavior risk likely higher. |
| `src/sparse_ldlt_csc.c` | 2095 | 26 | N/A | Only current >2000-line production C source; high source-list and behavior risk. |
| `tests/test_colamd.c` | 2017 | N/A | N/A | Large ordering test surface; candidate only if Day 2 finds a clean cluster. |

### Helper/Header Inventory

| Helper surface | Lines | Day 1 interpretation |
| --- | ---: | --- |
| `tests/test_svd_partial_helpers.h` | 1519 | Largest helper header; may indicate existing extraction precedent or further split risk. |
| `tests/test_qr_external_ref_helpers.h` | 1004 | Sprint 193 extraction target; maintain rather than reselect unless Day 2 chooses a nearby QR cluster. |
| `tests/test_qr_helpers.h` | 343 | Existing QR helper surface. |
| `tests/test_iterative_handle_helpers.h` | 289 | Existing iterative handle helper surface. |
| `tests/test_svd_helpers.h` | 257 | Existing SVD helper surface. |
| `tests/test_chol_csc_supernodal_helpers.h` | 255 | Existing supernodal Cholesky helper surface. |
| `tests/test_solver_helpers.h` | 202 | Shared solver helper; avoid broadening selected-cluster scope. |
| `tests/test_graph_fixtures.h` | 195 | Existing graph fixture surface. |
| `tests/test_integration_fixtures.h` | 169 | Existing integration fixture surface. |
| `tests/test_ldlt_csc_oracle_helpers.h` | 151 | Existing LDLT CSC oracle helper surface. |

### Initial Candidate Set

Day 2 should rank these first:

1. A new `tests/test_ldlt_csc.c` helper cluster not already reduced by prior
   LDLT CSC work.
2. A `tests/test_etree.c` fixture/helper cluster that does not disturb Sprint
   200 symbolic LU reliability proof ownership.
3. A `tests/test_svd.c` dense-reference or repeated fixture/helper cluster.
4. A remaining `tests/test_qr.c` cluster outside the Sprint 193
   external-reference extraction.
5. A `tests/test_chol_csc.c` helper or fixture cluster.
6. A `tests/test_chol_csc_supernodal.c` helper or fixture cluster.
7. A `tests/test_ldlt.c` selected direct-solver helper cluster.
8. A `tests/test_integration.c` fixture cluster only if it can be separated
   without broad cross-solver behavior changes.
9. A `src/sparse_ldlt_csc.c` production extraction only if test-helper
   candidates are unsuitable and source-list/CMake risk is justified.

### No-Behavior-Change Boundary

- Preserve public APIs, public headers, ABI, status codes, diagnostics, solver
  behavior, numerical tolerances, random seeds, skip behavior, and expected
  outputs.
- Preserve test names, `RUN_TEST(...)` ordering, fixture values, helper
  ownership, cleanup order, process-global restoration, and assertion intent.
- Prefer family-local helper extraction where it avoids new library
  source-list or CMake registration churn.
- If a new source, helper, or test registration is required, update Make,
  CMake, source-list, and guard evidence together.
- Do not add performance, package, platform, release, public API, ABI, broad
  review-surface, or state-of-the-art claims.

### Guard Pattern Inventory

| Guard pattern | Sprint 201 use |
| --- | --- |
| `make source-list-check` | Required if production source registration changes; useful confidence gate even for no production movement. |
| `make quality-review-cmake-compile` | Required if CMake test or source registration changes. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Existing shell guard pattern for helper/proof-owner boundaries. |
| `scripts/check_qr_external_ref_helper_guard.sh` | Existing QR helper guard pattern from Sprint 193. |
| Python guard tests under `tests/test_*guard*.py` | Reusable pattern for drift-sensitive ownership checks. |
| Focused proof-owner test binary | Preferred validation before full C quality gate. |

### Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Day 1 changes planning documentation only. |
| Focused selected-cluster tests | Not yet applicable. | Cluster selection occurs on Day 3. |
| Ownership guard | Not yet applicable. | Guard depends on selected cluster and extraction shape. |
| `make source-list-check` | Deferred unless implementation affects source registration. | Expected later if production source movement occurs. |
| `make quality-review-cmake-compile` | Deferred unless Make/CMake registration changes. | Expected later if new source/test registration changes. |
| `make format && make lint && make test` | Not required for Day 1. | Required later if `.c` or `.h` files change. |

### Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Selecting too broad a cluster | Sprint 201 should completely close one review-surface gap rather than partially touch many. | Day 2 ranking and Day 3 selection must freeze exactly one cluster. |
| Hidden behavior changes during extraction | Helper moves can accidentally change cleanup, ordering, tolerances, or diagnostics. | Day 4 invariants must be written before code edits. |
| Source-list or CMake drift | New source/test files can compile locally but drop from another build path. | Day 8 and Day 12 must run the relevant registration checks. |
| Reworking recent Sprint 200 tests | `tests/test_etree.c` is a candidate but now contains selected reliability proof. | Treat Sprint 200 proof ownership as a protected boundary unless selected deliberately. |
| Guard churn outweighs review reduction | A guard can add as much complexity as the extraction removes. | Keep guards selected-cluster scoped and proportional to registration risk. |
| Overclaiming reviewability | One cluster reduction is not broad maintainability completion. | Docs and retrospective must retain residual large surfaces. |

### Open Questions For Day 2

1. Which current >2000-line surface has the best combination of size payoff,
   helper cohesion, and existing focused tests?
2. Should Sprint 201 avoid `tests/test_etree.c` because Sprint 200 just added
   symbolic LU allocation-failure proof there?
3. Is another QR cluster worth selecting after Sprint 193, or should Day 2
   prioritize SVD/LDLT/Cholesky surfaces?
4. Can the selected target be header-only, or will Make/CMake registration
   need to be updated?
5. Which guard pattern will provide drift protection without broadening the
   review surface?

### Day 1 Validation

Commands run:

```sh
git status --short --branch
sed -n '1,90p' docs/planning/EPIC_18/SPRINT_201/PLAN.md
sed -n '/## Sprint 201:/,/## Sprint 202:/p' docs/planning/EPIC_18/PROJECT_PLAN.md
sed -n '1,180p' docs/planning/EPIC_17/SPRINT_193/artifacts/day1-review-surface-intake.md
sed -n '1,160p' docs/planning/EPIC_17/SPRINT_193/artifacts/day2-candidate-ranking.md
sed -n '1,180p' docs/planning/EPIC_17/SPRINT_193/RETROSPECTIVE.md
rg --files src include tests benchmarks examples scripts -g '*.c' -g '*.h' -g '*.py' -g '*.sh' | xargs wc -l | sort -nr | head -60
for f in tests/test_qr.c tests/test_ldlt_csc.c tests/test_integration.c tests/test_svd.c tests/test_ldlt.c tests/test_etree.c tests/test_iterative.c tests/test_graph.c tests/test_chol_csc.c tests/test_chol_csc_supernodal.c tests/test_qr_external_ref_helpers.h src/sparse_ldlt_csc.c src/sparse_lu_csr.c src/sparse_ldlt.c src/sparse_iterative.c src/sparse_qr.c; do printf '%s\t' "$f"; rg -n '^(static[[:space:]]+)?[A-Za-z_][A-Za-z0-9_ *]+[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\([^;]*\)[[:space:]]*\{' "$f" | wc -l | tr -d ' '; done
find tests -maxdepth 1 \( -name '*helpers*.h' -o -name '*fixture*.h' -o -name '*fixtures*.h' -o -name '*oracle*.h' \) -print | sort | xargs wc -l | sort -nr | head -40
for f in tests/test_qr.c tests/test_svd.c tests/test_etree.c tests/test_integration.c tests/test_iterative.c tests/test_ldlt.c tests/test_chol_csc.c tests/test_chol_csc_supernodal.c tests/test_ldlt_csc.c tests/test_graph.c; do printf '%s\tRUN_TEST=' "$f"; rg --count 'RUN_TEST\(' "$f" || true; done
```

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 2: Candidate Ranking

### Ranking Method

Day 2 ranked candidate review surfaces by:

- size payoff;
- reviewer burden;
- ownership clarity;
- helper cohesion;
- behavior-risk exposure;
- current focused coverage;
- registration risk;
- sprint fit.

The Day 2 artifact is:

- `docs/planning/EPIC_18/SPRINT_201/artifacts/day2-candidate-ranking.md`

### Ranked Candidate Table

| Rank | Candidate | Decision | Rationale |
| ---: | --- | --- | --- |
| 1 | `tests/test_svd.c` selected full-SVD/helper cluster | Recommended | Large remaining test surface, high RUN_TEST density, existing `tests/test_svd_helpers.h` precedent, and likely header-only extraction without production or CMake source-list churn. |
| 2 | `tests/test_chol_csc.c` selected helper/fixture cluster | Alternate | High helper density and direct-solver value; likely family-local helper extraction, but more intertwined with CSC factor/solve behavior. |
| 3 | `tests/test_chol_csc_supernodal.c` selected helper/fixture cluster | Alternate | Focused supernodal surface with existing helper header and clear sections; backend/env contracts raise process-global cleanup risk. |
| 4 | Remaining `tests/test_qr.c` economy or sparse-mode cluster | Alternate | Still large and already has guard precedent, but Sprint 193 just reduced QR external-reference surface and remaining clusters are behavior-sensitive. |
| 5 | `tests/test_ldlt.c` selected direct-solver helper cluster | Deferred | Large and important, but tolerance, inertia, backend, and solve/refine coverage make no-behavior-change boundaries harder. |
| 6 | `tests/test_ldlt_csc.c` additional helper cluster | Deferred | Largest current test surface, but Sprint 185 already reduced the highest-value LDLT CSC helper areas and further movement may have lower clarity. |
| 7 | `tests/test_etree.c` selected fixture/helper cluster | Deferred | High helper density, but Sprint 200 just added selected symbolic LU reliability proof; avoid disturbing fresh proof ownership unless no better candidate exists. |
| 8 | `tests/test_integration.c` selected lifecycle fixture cluster | Deferred | Broad cross-solver lifecycle tests risk a multi-cluster refactor and harder behavior-preservation review. |
| 9 | `tests/test_iterative.c` selected helper cluster | Deferred | Large, but convergence, allocation-failure, repeated-run handles, and process-global behavior increase extraction sensitivity. |
| 10 | `src/sparse_ldlt_csc.c` production module extraction | Deferred | Only current >2000-line production source, but source-list, CMake, ABI-adjacent review, and behavior risks are higher than needed for this sprint. |
| 11 | `scripts/run_external_comparison.py` or `tests/test_normalize_report_index.py` Python split | Deferred | Large surfaces, but outside the primary C solver/test examples and tied to generated evidence semantics rather than solver reviewability. |

### Preferred Shortlist For Day 3

| Priority | Cluster | Day 3 decision question |
| ---: | --- | --- |
| 1 | `tests/test_svd.c` selected helper cluster | Can a cohesive SVD helper/test block move to a family-local helper header while preserving `test_svd` as the proof-owner binary? |
| 2 | `tests/test_chol_csc.c` selected helper cluster | Can a bounded fixture/helper section move without touching factor/dispatch semantics? |
| 3 | `tests/test_chol_csc_supernodal.c` selected helper cluster | Can a helper-only section move without changing backend env-contract cleanup or process-global behavior? |
| 4 | Remaining `tests/test_qr.c` economy/sparse-mode cluster | Is there a non-Sprint-193 QR cluster with clear guardable ownership and enough payoff? |

### Day 2 Decision

Item 201.1 is complete for candidate ranking. Day 3 should select exactly one
cluster from the shortlist and freeze behavior-preservation boundaries before
any extraction begins.

Recommended Day 3 target: a selected `tests/test_svd.c` helper cluster.

### Day 2 Validation

Commands run:

```sh
git status --short --branch
sed -n '50,130p' docs/planning/EPIC_18/SPRINT_201/PLAN.md
tail -n 180 docs/planning/EPIC_18/SPRINT_201/WORKING_NOTES.md
for f in tests/test_ldlt_csc.c tests/test_etree.c tests/test_integration.c tests/test_qr.c tests/test_svd.c tests/test_ldlt.c tests/test_chol_csc.c tests/test_chol_csc_supernodal.c; do printf '\n%s\n' "$f"; rg -n '^/\* [=-]|^static void test_|^static .*make_|^static .*assert_|RUN_TEST\(' "$f" | head -n 120; done
for f in tests/test_svd.c tests/test_qr.c tests/test_ldlt.c tests/test_ldlt_csc.c; do printf '\n%s markers\n' "$f"; rg -n '^/\*|^//|RUN_TEST\(' "$f" | head -n 180; done
sed -n '1,220p' scripts/check_qr_external_ref_helper_guard.sh
sed -n '1,220p' scripts/check_ldlt_csc_helper_guard.sh
sed -n '1,220p' tests/test_svd_helpers.h
sed -n '1,120p' tests/test_svd_partial_helpers.h
```

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 3: Cluster Selection

### Selected Cluster

Day 3 selected exactly one Sprint 201 implementation target:

- `tests/test_svd.c` rank, pseudoinverse, and low-rank helper cluster.

The Day 3 decision artifact is:

- `docs/planning/EPIC_18/SPRINT_201/artifacts/day3-selected-cluster-boundary.md`

### Decision Summary

| Decision Field | Selected Value |
| --- | --- |
| Selected source surface | `tests/test_svd.c` |
| Selected helper surface | `tests/test_svd_selected_helpers.h`; shared fixtures remain in `tests/test_svd_helpers.h`. |
| Selected proof owner | `test_svd` executable remains the proof owner |
| Selected cluster | rank, pseudoinverse, and low-rank SVD tests and their local helper logic |
| Preferred implementation | header-only helper extraction using existing family-local SVD helper patterns |
| Preferred registration impact | no Makefile or CMake registration changes |
| Public behavior posture | no behavior change |

This cluster was selected because it is cohesive, has meaningful review-surface
payoff, already has a family-local helper destination, and can preserve
`tests/test_svd.c` as the sole proof-owner binary.

### In-Scope Tests

The selected cluster includes these current `tests/test_svd.c` test families:

- rank tests: `test_svd_rank_full`, `test_svd_rank_deficient`,
  `test_svd_rank_nearly_singular`, `test_svd_rank_diagonal_threshold_fixture`,
  `test_svd_qr_rank_dependent_row_fixture`, and `test_svd_rank_null`;
- pseudoinverse tests: `test_pinv_diagonal`, `test_pinv_moore_penrose`,
  `test_pinv_null`, `test_pinv_rectangular`, and
  `test_pinv_underdetermined_minnorm_solution`;
- dense low-rank tests: `test_lowrank_diagonal`,
  `test_lowrank_error_bound`, and `test_lowrank_errors`;
- cluster-local fixture builders, reconstruction checks, Moore-Penrose checks,
  dense low-rank error helpers, and assertion wrappers used only by these
  selected tests.

Existing `RUN_TEST(...)` entries stay in `tests/test_svd.c` and retain their
current order.

### Out-Of-Scope Boundaries

The selected cluster explicitly excludes:

- production SVD implementation files;
- public headers, public API, ABI, status codes, exported names, and caller
  contracts;
- partial SVD helper surfaces and partial SVD vector/residual tests;
- external dense-reference fixture readers and generated evidence semantics;
- Golub-Kahan and bidiagonal test families;
- condition-number tests;
- later sparse low-rank corpus and outer-product proof blocks unless Day 4
  proves they are inseparable from the dense low-rank cluster;
- Makefile, CMake, source-list, and CI registration changes unless a later
  ownership guard requires a narrowly scoped registration update.

### Frozen Behavior Boundary

Sprint 201 implementation must preserve:

- public API and ABI exactly;
- production source behavior exactly;
- test names and `RUN_TEST(...)` order exactly;
- fixture dimensions, input values, deterministic construction, and sparsity
  patterns exactly;
- numerical tolerances and threshold comparisons exactly;
- expected status codes, assertion behavior, and emitted diagnostic text;
- skip behavior and unsupported-environment behavior exactly;
- cleanup order, allocation ownership, and failure-path behavior exactly;
- `test_svd` as the single proof-owner executable for the selected tests.

Any change requiring tolerance updates, fixture data updates, expected-output
updates, new public symbols, production solver edits, or a proof-owner split is
outside the selected Day 3 boundary.

### Focused Validation Checklist

After implementation, the selected cluster should be validated with:

- `git diff --check`;
- `make build/test_svd` or the repository's equivalent `test_svd` rebuild path;
- `./build/test_svd`;
- `make source-list-check` if any registration file changes;
- `make format && make lint && make test` once `.c` or `.h` files change.

### Item 201.2 Status

Item 201.2 is complete for Day 3. The selected cluster is frozen and can be
reviewed independently from broader solver work. Day 4 should turn the selected
boundary into extraction invariants and guard requirements before any code
movement begins.

### Day 3 Validation

Commands run:

```sh
rg -n "test_svd_rank|test_pinv|test_lowrank|test_cond|test_svd_qr_rank|test_svd_null_input|test_svd_rejects_factored|test_svd_external|test_svd_partial|RUN_TEST" tests/test_svd.c
sed -n '1,220p' docs/planning/EPIC_18/SPRINT_201/artifacts/day2-candidate-ranking.md
tail -n 120 docs/planning/EPIC_18/SPRINT_201/WORKING_NOTES.md
git status --short --branch
```

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 4: Behavior-Preservation Invariants

### Invariant Artifact

Day 4 converted the selected SVD cluster boundary into pre-edit invariants.

Artifact:

- `docs/planning/EPIC_18/SPRINT_201/artifacts/day4-preservation-invariants.md`

Selected cluster remains:

- `tests/test_svd.c` rank, pseudoinverse, and low-rank helper cluster.

Preferred extraction surface remains:

- `tests/test_svd_helpers.h`

Proof owner remains:

- `test_svd`, with selected `RUN_TEST(...)` registrations retained in
  `tests/test_svd.c`.

### Preservation Invariant Summary

| Invariant Area | Day 4 Requirement |
| --- | --- |
| Function signatures | Selected test names remain callable through the current `RUN_TEST(...)` registrations. |
| Proof ownership | `tests/test_svd.c` remains the proof-owner binary and keeps registration order. |
| Public API and ABI | No production source or public header changes are in scope. |
| Fixture dimensions and values | Existing dimensions, duplicate-column fixtures, dependent-row fixtures, diagonal thresholds, tridiagonal values, and underdetermined-system values are preserved exactly. |
| Sparse insertion behavior | `tf_svd_insert_or_free` semantics and cleanup ownership remain unchanged. |
| Error propagation | Existing `SPARSE_OK`, `SPARSE_ERR_NULL`, and `SPARSE_ERR_BADARG` expectations remain visible. |
| Numerical tolerances | Existing `0.0`, `1e-14`, `1e-12`, `1e-10`, `1e-8`, and `1e-6` paths are preserved exactly. |
| Dense layout | Pseudoinverse and low-rank buffers keep current column-major indexing. |
| Cleanup behavior | `free`, `sparse_free`, and `sparse_svd_free` calls remain on equivalent success and failure paths. |
| Helper scope | New or moved helpers stay SVD-test-local and use the existing `tf_svd_` helper naming pattern. |

### Selected Test Expectations

Day 4 pinned the expected behavior for these selected tests:

- `test_svd_rank_full`
- `test_svd_rank_deficient`
- `test_svd_rank_nearly_singular`
- `test_svd_rank_diagonal_threshold_fixture`
- `test_svd_qr_rank_dependent_row_fixture`
- `test_svd_rank_null`
- `test_pinv_diagonal`
- `test_pinv_moore_penrose`
- `test_pinv_null`
- `test_pinv_rectangular`
- `test_pinv_underdetermined_minnorm_solution`
- `test_lowrank_diagonal`
- `test_lowrank_error_bound`
- `test_lowrank_errors`

The selected expectations cover rank results, null and bad-argument error
codes, Moore-Penrose residual bounds, pseudoinverse column-major indexing,
low-rank diagonal entries, low-rank Frobenius error, and min-norm solution
checks.

### Registration Surface Disposition

No registration changes are planned.

| Surface | Disposition |
| --- | --- |
| `tests/test_svd.c` | Keep selected `RUN_TEST(...)` entries and current order. |
| `tests/test_svd_helpers.h` | Preferred static inline helper destination. |
| `Makefile` | No planned change. |
| `CMakeLists.txt` | No planned change. |
| CI workflows | No planned change. |

### Pre-Edit Checklist

Before implementation:

- confirm the exact helper/test movement pattern;
- preserve selected `RUN_TEST(...)` registrations in `tests/test_svd.c`;
- avoid production source and public header edits;
- copy fixture values, dimensions, and tolerances verbatim;
- preserve column-major indexing and leading-dimension assumptions;
- preserve cleanup on all allocation, SVD, pseudoinverse, and low-rank failure
  paths;
- validate with the focused `test_svd` proof owner;
- run the full C quality gate once `.h` or `.c` files are modified.

### Day 4 Status

Day 4 completes the pre-edit invariant work for the selected cluster and
prepares item 201.3 helper extraction. The selected cluster can now move into
implementation with explicit reviewability goals and behavior-preservation
checks.

### Day 4 Validation

Commands run:

```sh
sed -n '85,145p' docs/planning/EPIC_18/SPRINT_201/PLAN.md
sed -n '1,220p' docs/planning/EPIC_18/SPRINT_201/artifacts/day3-selected-cluster-boundary.md
rg -n "static .*tf_svd|static .*test_svd_rank|static .*test_pinv|static .*test_lowrank|tf_svd_" tests/test_svd.c tests/test_svd_helpers.h
sed -n '1700,2125p' tests/test_svd.c
sed -n '1,280p' tests/test_svd_helpers.h
sed -n '2960,2976p' tests/test_svd.c
git status --short --branch
```

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 5: Extraction Design

### Design Artifact

Day 5 converted the selected SVD cluster and Day 4 invariants into a concrete
helper extraction design.

Artifact:

- `docs/planning/EPIC_18/SPRINT_201/artifacts/day5-extraction-design.md`

Selected extraction shape:

- header-only test-helper split into `tests/test_svd_selected_helpers.h`;
- `tests/test_svd.c` remains the proof-owner binary;
- existing `RUN_TEST(...)` names and order remain in `tests/test_svd.c`;
- no Makefile, CMake, source-list, CI, public header, or production source
  changes are planned.

### Minimal Movement Pattern

The Day 5 design keeps one thin wrapper per selected test name in
`tests/test_svd.c` and moves selected implementation bodies behind
`static inline` `tf_svd_test_*` helpers in `tests/test_svd_helpers.h`.

| Existing Test Name | Planned Helper-Owned Implementation |
| --- | --- |
| `test_svd_rank_full` | `tf_svd_test_rank_full` |
| `test_svd_rank_deficient` | `tf_svd_test_rank_deficient` |
| `test_svd_rank_nearly_singular` | `tf_svd_test_rank_nearly_singular` |
| `test_svd_rank_diagonal_threshold_fixture` | `tf_svd_test_rank_diagonal_threshold_fixture` |
| `test_svd_qr_rank_dependent_row_fixture` | `tf_svd_test_qr_rank_dependent_row_fixture` |
| `test_svd_rank_null` | `tf_svd_test_rank_null` |
| `test_pinv_diagonal` | `tf_svd_test_pinv_diagonal` |
| `test_pinv_moore_penrose` | `tf_svd_test_pinv_moore_penrose` |
| `test_pinv_null` | `tf_svd_test_pinv_null` |
| `test_pinv_rectangular` | `tf_svd_test_pinv_rectangular` |
| `test_pinv_underdetermined_minnorm_solution` | `tf_svd_test_pinv_underdetermined_minnorm_solution` |
| `test_lowrank_diagonal` | `tf_svd_test_lowrank_diagonal` |
| `test_lowrank_error_bound` | `tf_svd_test_lowrank_error_bound` |
| `test_lowrank_errors` | `tf_svd_test_lowrank_errors` |

### Dependency Decisions

The existing helper header already provides most selected cluster dependencies:

- `sparse_matrix.h`
- `sparse_svd.h`
- `test_framework.h`
- `<math.h>`
- `<stdlib.h>`

Day 6+ must verify whether moved helpers also require `sparse_qr.h` for the QR
rank cross-check and whether `vec_norm2` visibility affects the underdetermined
pseudoinverse helper. If either dependency would broaden helper ownership too
much, the affected wrapper can remain local while the rest of the cluster moves.

### Registration And Build Plan

No registration changes are planned.

| Surface | Day 5 Plan |
| --- | --- |
| `tests/test_svd.c` | Keep proof-owner wrappers and `RUN_TEST(...)` registrations. |
| `tests/test_svd_selected_helpers.h` | Add selected `static inline` helper-owned implementations. |
| `Makefile` | No change. |
| `CMakeLists.txt` | No change. |
| Source-list manifests | No change expected. |
| CI workflows | No change. |

### Days 6 Through 8 Checklist

1. Move rank implementations first.
2. Build and run `test_svd` after the rank move.
3. Move pseudoinverse implementations after confirming helper dependencies.
4. Keep the underdetermined pseudoinverse test local if `vec_norm2` would
   create unnecessary helper leakage.
5. Move dense low-rank implementations after pseudoinverse checks pass.
6. Preserve `RUN_TEST(...)` registrations and order exactly.
7. Add a narrow ownership guard on Day 8.
8. Run the full C quality gate once `.h` or `.c` files change.

### Day 5 Status

Item 201.3 has a local extraction design that can be reviewed independently.
The design does not create public headers or public API and has explicit
registration-impact decisions before helpers move.

### Day 5 Validation

Commands run:

```sh
sed -n '130,205p' docs/planning/EPIC_18/SPRINT_201/PLAN.md
sed -n '1,240p' docs/planning/EPIC_18/SPRINT_201/artifacts/day4-preservation-invariants.md
rg -n "Day 5|Helper|Extraction|201\\.3" docs/planning/EPIC_18/SPRINT_201/PLAN.md docs/planning/EPIC_18/PROJECT_PLAN.md
sed -n '1700,2125p' tests/test_svd.c
sed -n '1,280p' tests/test_svd_helpers.h
sed -n '2960,2976p' tests/test_svd.c
git status --short --branch
```

Day 5 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.

## Day 6: First Extraction Pass

### Implementation Artifact

Day 6 implemented the first selected SVD extraction pass.

Artifact:

- `docs/planning/EPIC_18/SPRINT_201/artifacts/day6-first-extraction-pass.md`

Changed code files:

- `tests/test_svd.c`
- `tests/test_svd_helpers.h`

No production source, public header, Makefile, CMake, source-list, or CI files
were changed.

### Extracted Rank Subgroup

The rank subgroup moved into `tests/test_svd_selected_helpers.h` behind `static inline`
helper-owned implementations:

| Existing Test Wrapper | Helper-Owned Implementation |
| --- | --- |
| `test_svd_rank_full` | `tf_svd_test_rank_full` |
| `test_svd_rank_deficient` | `tf_svd_test_rank_deficient` |
| `test_svd_rank_nearly_singular` | `tf_svd_test_rank_nearly_singular` |
| `test_svd_rank_diagonal_threshold_fixture` | `tf_svd_test_rank_diagonal_threshold_fixture` |
| `test_svd_qr_rank_dependent_row_fixture` | `tf_svd_test_qr_rank_dependent_row_fixture` |
| `test_svd_rank_null` | `tf_svd_test_rank_null` |

`tests/test_svd.c` keeps the original test names as thin wrappers and retains
the existing `RUN_TEST(...)` registration order.

### Dependency Notes

`tests/test_svd_helpers.h` gained:

- `sparse_qr.h`, required by the dependent-row SVD/QR rank cross-check helper;
- `<stdio.h>`, required by moved `printf(...)` diagnostics.

This matches the Day 5 dependency plan. No broader helper or production
dependency was introduced.

### Behavior-Preservation Notes

The Day 6 move preserved:

- rank fixture dimensions and values;
- duplicate-column and dependent-row fixture semantics;
- rank tolerances and threshold literals;
- QR rank cross-check behavior;
- null-input error-code expectations;
- diagnostic text;
- cleanup and early-return behavior;
- `RUN_TEST(...)` names and order.

### Review-Surface Result

Post-format line counts:

| File | Lines |
| --- | ---: |
| `tests/test_svd.c` | 2914 |
| `tests/test_svd_helpers.h` | 394 |

Day 7 should continue with pseudoinverse and dense low-rank helper ownership if
the Day 6 pattern remains reviewable.

### Day 6 Validation

Focused commands:

```sh
make build/test_svd
./build/test_svd
```

Focused result:

- `make build/test_svd` passed.
- `./build/test_svd` passed: 114 tests run, 0 failed, 0 skipped.

Required quality gate because `.c` and `.h` files changed:

```sh
make format
make lint
make test
```

Required gate result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.

Additional inspection commands:

```sh
git diff --stat
git diff -- tests/test_svd.c tests/test_svd_helpers.h
wc -l tests/test_svd.c tests/test_svd_helpers.h
rg -n "tf_svd_test_rank_|test_svd_rank_full|test_svd_rank_deficient|test_svd_rank_nearly_singular|test_svd_rank_diagonal_threshold_fixture|test_svd_qr_rank_dependent_row_fixture|test_svd_rank_null" tests/test_svd.c tests/test_svd_helpers.h
rg -n "tf_svd_test_qr_rank" tests/test_svd_helpers.h tests/test_svd.c
```

### Day 6 Status

Day 6 completes the first extraction pass for item 201.3. The selected cluster
now has a proven rank-helper extraction path, local `test_svd` reachability, and
full quality-gate coverage.

## Day 7 - Cohesion Pass

### Summary

Day 7 completed the second extraction pass for the selected SVD review-surface
cluster. The pseudoinverse and dense low-rank test bodies now live in
`tests/test_svd_helpers.h`, while `tests/test_svd.c` keeps the registered
wrappers and `RUN_TEST(...)` ordering intact.

### Changed Files

- `tests/test_svd.c`
- `tests/test_svd_helpers.h`
- `docs/planning/EPIC_18/SPRINT_201/artifacts/day7-cohesion-pass.md`
- `docs/planning/EPIC_18/SPRINT_201/WORKING_NOTES.md`

### Helper Ownership Map

Day 7 added these helper-owned implementations:

| Registered test wrapper | Helper-owned implementation |
| --- | --- |
| `test_pinv_diagonal` | `tf_svd_test_pinv_diagonal` |
| `test_pinv_moore_penrose` | `tf_svd_test_pinv_moore_penrose` |
| `test_pinv_null` | `tf_svd_test_pinv_null` |
| `test_pinv_rectangular` | `tf_svd_test_pinv_rectangular` |
| `test_pinv_underdetermined_minnorm_solution` | `tf_svd_test_pinv_underdetermined_minnorm_solution` |
| `test_lowrank_diagonal` | `tf_svd_test_lowrank_diagonal` |
| `test_lowrank_error_bound` | `tf_svd_test_lowrank_error_bound` |
| `test_lowrank_errors` | `tf_svd_test_lowrank_errors` |

The Day 6 rank helper map remains in place, so the selected core cluster now
covers rank, pseudoinverse, and dense low-rank tests.

### Dependency Notes

`tests/test_svd_helpers.h` gained `sparse_vector.h` for `vec_norm2(...)` in the
underdetermined minimum-norm pseudoinverse fixture. The existing `sparse_qr.h`
dependency remains scoped to the dependent-row SVD/QR rank cross-check.

### Review-Surface Result

Post-format line counts:

| File | Lines |
| --- | ---: |
| `tests/test_svd.c` | 2657 |
| `tests/test_svd_helpers.h` | 677 |

The selected test body ownership moved out of `tests/test_svd.c` without
changing test registration names or order.

### Day 7 Validation

Focused commands:

```sh
make build/test_svd
./build/test_svd
```

Focused result:

- `make build/test_svd` passed.
- `./build/test_svd` passed: 114 tests run, 0 failed, 0 skipped.

Required quality gate because `.c` and `.h` files changed:

```sh
make format
make lint
make test
```

Required gate result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.

### Day 7 Status

Day 7 completes the core extraction pass for item 201.3. Day 8 should add an
ownership/registration guard so future edits do not move the selected helper
cluster back into the monolithic SVD test file unnoticed.

## Day 8 - Registration Alignment

### Summary

Day 8 aligned the selected SVD helper extraction with the build and test
registration surfaces. The extracted helper remains header-only and included by
`tests/test_svd.c`; `test_svd.c` remains the Make and CMake proof-owner binary.

### Changed Files

- `Makefile`
- `scripts/check_svd_helper_guard.sh`
- `docs/planning/EPIC_18/SPRINT_201/artifacts/day8-registration-alignment.md`
- `docs/planning/EPIC_18/SPRINT_201/WORKING_NOTES.md`

### Registration Decision

No CMake or library source-list registration changes were needed for the SVD
helper headers because they are included helper headers, not standalone tests or
library sources. PR #223 review follow-up added explicit Makefile prerequisites
for both helper headers on `build/test_svd` so helper-only edits rebuild the
proof-owner binary.

Day 8 added a focused guard target:

```sh
make svd-helper-guard
```

The guard enforces:

- `$(TESTDIR)/test_svd.c` remains in `TEST_SRCS`;
- `add_sparse_test(test_svd)` remains in `CMakeLists.txt`;
- `tests/test_svd.c` includes `test_svd_helpers.h` and
  `test_svd_selected_helpers.h` exactly once;
- the selected moved rank, pseudoinverse, and dense low-rank helper
  implementations remain in `tests/test_svd_selected_helpers.h`;
- the selected `RUN_TEST(...)` registrations remain in `tests/test_svd.c`
  exactly once and in frozen order;
- both helper headers remain absent from CMake registration and
  `build-metadata/library_sources.txt`;
- both helper headers remain explicit Makefile prerequisites for
  `build/test_svd`;
- no standalone SVD helper CMake test is introduced accidentally.

### Day 8 Validation

Registration guard:

```sh
make svd-helper-guard
```

Result:

- `svd-helper-guard: required files ok`
- `svd-helper-guard: proof-owner registration ok`
- `svd-helper-guard: helper boundary ok`
- `svd-helper-guard: selected cluster ownership ok`
- `svd-helper-guard: header-only registration ok`
- `svd-helper-guard: passed`

Source-list check:

```sh
make source-list-check
```

Result:

- `source-list-check: PASS (49 library sources)`

CMake registration configure check:

```sh
cmake -S . -B build/sprint201-day8-registration-check
```

Result:

- Configure and generate passed.

Generated-artifact hygiene:

```sh
git status --short --ignored build/sprint201-day8-registration-check
```

Result:

- `!! build/`

The CMake check wrote only under ignored `build/`; no generated build artifacts
were staged.

### Day 8 Status

Day 8 completes the registration-alignment portion of item 201.4 for the
selected SVD helper cluster. No residual build-registration risk is known for
the selected helper boundary.

## Day 9 - Ownership Guard

### Summary

Day 9 hardened the Sprint 201 SVD helper guard with fixture-based negative
tests. The new coverage proves that ownership, registration, helper dependency,
and header-only boundary drift fail with clear messages.

### Changed Files

- `tests/test_svd_helper_guard.py`
- `docs/planning/EPIC_18/SPRINT_201/artifacts/day9-ownership-guard.md`
- `docs/planning/EPIC_18/SPRINT_201/WORKING_NOTES.md`

### Guard Rules

The selected SVD helper guard now has regression coverage for:

- missing `test_svd_helpers.h` include in `tests/test_svd.c`;
- missing `sparse_svd.h` or `sparse_vector.h` helper dependency includes;
- moved helper-owned definitions reappearing in `tests/test_svd.c`;
- missing selected `RUN_TEST(...)` proof-owner registrations;
- missing Makefile or CMake proof-owner registration for `test_svd`;
- accidental Makefile or library-source registration of
  `tests/test_svd_helpers.h`.

The current-tree pass check remains part of the Python guard test, so the
fixture-based negative cases and real repo state are validated together.

### Day 9 Validation

New SVD guard regression:

```sh
python3 tests/test_svd_helper_guard.py
```

Result:

- Passed.

Focused guard command:

```sh
make svd-helper-guard
```

Result:

- `svd-helper-guard: required files ok`
- `svd-helper-guard: proof-owner registration ok`
- `svd-helper-guard: helper boundary ok`
- `svd-helper-guard: selected cluster ownership ok`
- `svd-helper-guard: header-only registration ok`
- `svd-helper-guard: passed`

Adjacent guard regression:

```sh
python3 tests/test_qr_external_ref_helper_guard.py
```

Result:

- Passed.

### Day 9 Status

Day 9 completes the drift-sensitive ownership-guard portion of item 201.4 for
the selected SVD helper cluster. The guard remains selected-cluster scoped and
does not introduce broad SVD, public API, performance, or repository-wide
review-surface claims.

## Day 10 - Focused Regression Review

### Summary

Day 10 ran the selected focused behavior-preservation checks for the Sprint 201
SVD helper extraction and recorded the invariant-to-test traceability table.
No additional C regression test was needed because the existing `test_svd`
proof owner already exercises every selected rank, pseudoinverse, and dense
low-rank invariant after extraction.

### Changed Files

- `docs/planning/EPIC_18/SPRINT_201/artifacts/day10-focused-regression.md`
- `docs/planning/EPIC_18/SPRINT_201/WORKING_NOTES.md`

### Focused Evidence

Proof-owner rebuild:

```sh
make build/test_svd
```

Result:

- `make: \`build/test_svd' is up to date.`

Focused behavior proof:

```sh
./build/test_svd
```

Result:

- 114 tests run.
- 0 failed.
- 0 skipped.
- 2067 assertions.
- All tests passed.

Ownership guard:

```sh
make svd-helper-guard
```

Result:

- `svd-helper-guard: required files ok`
- `svd-helper-guard: proof-owner registration ok`
- `svd-helper-guard: helper boundary ok`
- `svd-helper-guard: selected cluster ownership ok`
- `svd-helper-guard: header-only registration ok`
- `svd-helper-guard: passed`

Guard regression:

```sh
python3 tests/test_svd_helper_guard.py
```

Result:

- Passed.

### Invariant-To-Test Traceability

| Invariant | Day 10 Evidence |
| --- | --- |
| Function signatures and `RUN_TEST(...)` names remain callable. | `./build/test_svd` ran the selected wrappers; `make svd-helper-guard` verified selected registrations exactly once. |
| `tests/test_svd.c` remains proof owner. | `make build/test_svd`, `./build/test_svd`, and `make svd-helper-guard` passed. |
| Public API and ABI unchanged. | `git diff --name-only -- include src CMakeLists.txt build-metadata/library_sources.txt .github` produced no paths. |
| Fixture dimensions and values unchanged. | All selected rank, pseudoinverse, and dense low-rank tests passed with existing expected outputs. |
| Error propagation unchanged. | `test_svd_rank_null`, `test_pinv_null`, and `test_lowrank_errors` passed. |
| Numerical tolerances unchanged. | Rank threshold, Moore-Penrose, rectangular pseudoinverse, and low-rank error-bound tests passed. |
| Dense column-major layout preserved. | Pseudoinverse and dense low-rank selected tests passed after helper movement. |
| Cleanup behavior preserved. | Focused `test_svd` run completed all selected tests without failures after the extraction. |
| Helper remains SVD-test-local. | `make svd-helper-guard` verified header-only boundaries across Makefile, CMake, and library source manifest. |

### Untested Breadth

Day 10 intentionally does not claim broad SVD correctness, performance
improvement, public API change, partial-SVD helper changes, sparse low-rank
extended-cluster ownership changes, external-reference parity, platform support,
package-manager support, or release readiness.

### Day 10 Status

Day 10 completes item 201.5 focused behavior-preservation evidence for the
selected SVD rank, pseudoinverse, and dense low-rank helper extraction. No
focused behavior-preservation residual is known for the selected cluster.

## Day 11 - Documentation and Maintainer Alignment

### Summary

Day 11 updated maintainer and planning documentation so Sprint 201 is described
as a selected SVD helper review-surface reduction, not as broad SVD behavior,
public API, performance, platform, package, or repository-wide review-surface
cleanup.

### Changed Files

- `docs/maintainer_guide.md`
- `docs/planning/EPIC_18/PROJECT_PLAN.md`
- `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md`
- `docs/planning/EPIC_18/SPRINT_201/artifacts/day11-maintainer-alignment.md`
- `docs/planning/EPIC_18/SPRINT_201/WORKING_NOTES.md`

### Documentation Alignment

Maintainer guide updates:

- added Sprint 201 SVD helper ownership to the review-surface evidence family;
- added the selected SVD helper boundary near the existing LDLT CSC and QR
  helper boundary rules;
- recorded `make svd-helper-guard`, `python3 tests/test_svd_helper_guard.py`,
  and focused `test_svd` validation expectations;
- retained explicit non-claims for public API/ABI, algorithm behavior,
  performance, platform, package, partial-SVD ownership, broader external
  parity, and broad review-surface cleanup.

Planning updates:

- updated the Epic 18 interim project-plan snapshot for Sprint 201 from
  pending future execution to closed for the selected SVD helper cluster;
- updated E18-RQ-004 so it is closed for the selected SVD cluster while broader
  large-surface cleanup remains future selected-cluster work.

Public docs decision:

- no README, INSTALL, tutorial, cookbook, benchmark, example, or public-header
  update was needed because the change is maintainer/test ownership only.

### Day 11 Validation

Documentation changes were reconciled against:

- `docs/planning/EPIC_18/SPRINT_201/artifacts/day3-selected-cluster-boundary.md`
- `docs/planning/EPIC_18/SPRINT_201/artifacts/day4-preservation-invariants.md`
- `docs/planning/EPIC_18/SPRINT_201/artifacts/day8-registration-alignment.md`
- `docs/planning/EPIC_18/SPRINT_201/artifacts/day9-ownership-guard.md`
- `docs/planning/EPIC_18/SPRINT_201/artifacts/day10-focused-regression.md`

Runnable evidence remains:

```sh
make svd-helper-guard
python3 tests/test_svd_helper_guard.py
make build/test_svd
./build/test_svd
```

### Day 11 Status

Day 11 completes the documentation portion of item 201.6 for the selected SVD
helper ownership surface. Day 12 should run the integrated local validation
pass, including docs checks and the required full C quality gate for the
branch's `.c`/`.h` extraction changes.

## Day 12 - Integrated Validation

### Summary

Day 12 ran the focused selected-cluster checks, ownership guards, source/CMake
and documentation parity checks, and the required full C quality gate for the
branch's SVD helper extraction changes.

### Validation Commands

Focused SVD regression:

```sh
make build/test_svd && ./build/test_svd
```

Result:

- Passed.
- `./build/test_svd`: 114 tests run, 0 failed, 0 skipped, 2067 assertions.
- Final summary: `ALL TESTS PASSED`.

SVD helper ownership guard:

```sh
make svd-helper-guard
```

Result:

- `svd-helper-guard: required files ok`
- `svd-helper-guard: proof-owner registration ok`
- `svd-helper-guard: helper boundary ok`
- `svd-helper-guard: selected cluster ownership ok`
- `svd-helper-guard: header-only registration ok`
- `svd-helper-guard: passed`

Guard regression:

```sh
python3 tests/test_svd_helper_guard.py
```

Result:

- Passed.

Source-list parity:

```sh
make source-list-check
```

Result:

- `source-list-check: PASS (49 library sources)`

CMake configure parity:

```sh
cmake -S . -B build/sprint201-day12-validation-check
```

Result:

- Passed with AppleClang 17.0.0.17000604.
- Generated files stayed under ignored `build/sprint201-day12-validation-check`.

Docs check:

```sh
make docs-check
```

Result:

- Passed.
- `api-docs-coverage: PASS`.
- Checked-in public headers: 18.
- Generated reference pages: 18.
- Generated source pages: 18.
- Generated Doxygen output stayed under ignored `docs/api/html`.

Adjacent helper guard:

```sh
python3 tests/test_qr_external_ref_helper_guard.py
```

Result:

- Passed.

Required C quality gate:

```sh
make format
make lint
make test
```

Result:

- `make format`: passed.
- `make lint`: passed; built benchmark/example binaries, strict warning compile
  completed, `clang-tidy` completed for 49 production files, and `cppcheck`
  completed for 109 files.
- `make test`: passed; final summary was `All tests passed.`

### Risk Register

| Risk | Day 12 Status | Residual |
| --- | --- | --- |
| Selected SVD helper extraction changes behavior. | Focused `test_svd` and full `make test` passed. | No known selected-cluster behavior residual. |
| Selected SVD helper ownership drifts from `tests/test_svd.c`. | `make svd-helper-guard` and `python3 tests/test_svd_helper_guard.py` passed. | No known selected-cluster ownership residual. |
| Registration/source/CMake parity regresses. | `make source-list-check` and CMake configure passed. | No known parity residual. |
| Docs or maintainer guidance overclaim. | `make docs-check` passed and Day 11 documentation recorded selected-scope non-claims. | Broader review-surface cleanup remains future selected-cluster work. |
| Public API, ABI, or implementation changes are implied. | Sprint 201 changed selected SVD test/helper ownership, not public API or library implementation. | Public API/ABI non-claim remains unchanged. |
| Platform, package-manager, performance, or state-of-the-art claims are implied. | Day 12 validation is local selected-cluster evidence only. | These remain explicit non-claims for Sprint 201. |

### Day 12 Status

Day 12 completes item 201.6 validation evidence for the selected SVD rank,
pseudoinverse, and dense low-rank helper review-surface reduction. No Day 12
validation failure is known.

## Day 13 - Review-Surface Hardening

### Summary

Day 13 audited the selected SVD helper extraction for unnecessary breadth,
claim drift, and evidence consistency. No additional code change was needed
after the audit.

### Reviewed Surfaces

- `tests/test_svd.c`
- `tests/test_svd_helpers.h`
- `scripts/check_svd_helper_guard.sh`
- `tests/test_svd_helper_guard.py`
- `Makefile`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_18/PROJECT_PLAN.md`
- `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md`
- Sprint 201 artifacts and working notes

### Hardening Actions

- Confirmed the selected extraction remains limited to SVD rank,
  pseudoinverse, and dense low-rank test bodies.
- Confirmed `tests/test_svd.c` remains the proof-owner binary and keeps the
  selected `RUN_TEST(...)` registrations.
- Confirmed `tests/test_svd_selected_helpers.h` remains proof-owner-only and is
  not promoted into CMake or library-source registration, while both SVD helper
  headers are explicit Makefile prerequisites for `build/test_svd`.
- Confirmed the guard and guard-regression fixtures cover missing include,
  missing dependency include, duplicate moved ownership, missing proof-owner
  registration, and accidental helper registration drift.
- Updated project-plan and residual-queue evidence links through the Day 13
  review-hardening artifact.

### Final Invariant-To-Regression Traceability

| Invariant | Evidence | Status |
| --- | --- | --- |
| Selected moved bodies are helper-owned. | `make svd-helper-guard`; `python3 tests/test_svd_helper_guard.py`; moved marker checks. | Covered. |
| Proof-owner registrations stay in `tests/test_svd.c`. | `make svd-helper-guard`; guard negative fixtures; focused `./build/test_svd`. | Covered. |
| Helper dependencies are explicit. | Guard checks for `sparse_svd.h` and `sparse_vector.h`; negative fixtures. | Covered. |
| Helper remains header-only. | Guard non-registration checks; Day 12 source-list and CMake checks. | Covered. |
| Selected behavior is preserved. | Day 10 and Day 12 focused `./build/test_svd`; Day 12 full `make test`. | Covered. |
| Documentation remains claim-safe. | Day 11 maintainer alignment; Day 12 risk register; Day 13 hardening audit. | Covered. |

### Closeout Checklist Draft

- [x] Selected large-surface intake and ranking recorded.
- [x] Selected SVD rank/pseudoinverse/dense-low-rank cluster frozen.
- [x] Behavior-preservation invariants recorded.
- [x] Extraction design recorded.
- [x] Extraction and cohesion passes completed.
- [x] Build/registration alignment recorded.
- [x] Guard and guard regression tests added.
- [x] Focused regression evidence recorded.
- [x] Maintainer/planning docs aligned.
- [x] Integrated validation evidence recorded.
- [x] Review-hardening evidence recorded.
- [ ] Day 14 closeout and retrospective inputs prepared.

### Residuals

Remaining review surfaces are explicit future work: other `tests/test_svd.c`
clusters, `tests/test_svd_partial_corpus.c`, broader solver/graph/direct-solver
surfaces, and shared helper dependency tracking beyond the selected SVD guard.
No Sprint 201 artifact claims public API/ABI change, solver behavior change,
numerical tolerance change, performance improvement, package support, platform
support, release readiness, or state-of-the-art status.

### Day 13 Status

Day 13 completes review-surface hardening for the selected SVD helper cluster.
The remaining sprint work is Day 14 closeout and retrospective input
preparation.

## Day 14 - Closeout and Retrospective Inputs

### Summary

Day 14 reconciled items 201.1 through 201.6, recorded final validation and
residuals, and prepared retrospective inputs for Sprint 201. The sprint closes
with one selected SVD helper review-surface reduction complete.

### Final Item Status

| Item | Final status | Evidence |
| --- | --- | --- |
| 201.1 Candidate Ranking | Complete | Day 1 and Day 2 intake/ranking artifacts. |
| 201.2 Cluster Selection | Complete | Day 3 selected-cluster boundary and Day 4 behavior-preservation invariants. |
| 201.3 Helper Or Module Extraction | Complete | Day 5 design, Day 6 first extraction, and Day 7 cohesion pass. |
| 201.4 Ownership Guard | Complete | Day 8 registration alignment, Day 9 guard, `make svd-helper-guard`, and `tests/test_svd_helper_guard.py`; PR #223 review follow-up added selected-helper dependency checks, frozen registration-order checks, and explicit Makefile helper prerequisites. |
| 201.5 Focused Regression | Complete | Day 10 focused regression and Day 12 focused/full validation. |
| 201.6 Validation And Docs | Complete | Day 11 docs alignment, Day 12 integrated validation, Day 13 hardening, and Day 14 closeout. |

### Completed Work

- Selected one bounded SVD test cluster from the large review-surface queue.
- Moved selected rank, pseudoinverse, and dense low-rank test bodies to
  `tests/test_svd_selected_helpers.h`; shared fixture helpers remain in
  `tests/test_svd_helpers.h`.
- Preserved `tests/test_svd.c` as the proof-owner binary and retained selected
  `RUN_TEST(...)` registrations.
- Added `scripts/check_svd_helper_guard.sh`, `make svd-helper-guard`, and
  `tests/test_svd_helper_guard.py`.
- Recorded focused, integrated, and review-hardening evidence.
- Updated maintainer guide, project-plan, and residual-queue wording with
  selected-cluster scope and explicit non-claims.

### Final Validation Ledger

| Command | Result |
| --- | --- |
| `make build/test_svd && ./build/test_svd` | PASS; Day 12 reported 114 tests, 0 failures, 0 skipped, and 2067 assertions. |
| `make svd-helper-guard` | PASS on Day 9, Day 12, Day 13, and Day 14. |
| `python3 tests/test_svd_helper_guard.py` | PASS on Day 9, Day 12, Day 13, and Day 14. |
| `make source-list-check` | PASS on Day 12 with 49 library sources. |
| `cmake -S . -B build/sprint201-day12-validation-check` | PASS on Day 12. |
| `make docs-check` | PASS on Day 12, Day 13, and Day 14. |
| `python3 tests/test_qr_external_ref_helper_guard.py` | PASS on Day 12. |
| `make format` | PASS on Day 12. |
| `make lint` | PASS on Day 12. |
| `make test` | PASS on Day 12 with `All tests passed.` |
| `git diff --check` | PASS on Day 12, Day 13, and Day 14. |

### Retrospective Inputs

| Area | Input |
| --- | --- |
| Completed work | One selected SVD rank/pseudoinverse/dense-low-rank review surface is reduced and locally validated. |
| Validation | Focused SVD binary, SVD helper guard, guard regression test, source-list parity, CMake configure, docs check, adjacent QR guard, format, lint, full test suite, and whitespace checks passed. |
| Deviations | The sprint used a proof-owner-only selected helper header rather than a compiled helper module because this is a test-only selected surface and `tests/test_svd.c` remains the proof owner. Assertion source locations now follow the helper-owned implementation file; selected test names, status/error behavior, and emitted diagnostic text remain the preserved diagnostic surface. |
| Deferred breadth | Other SVD clusters, partial-SVD corpus ownership, broader test/helper dependency tracking, and other large solver/test files remain future selected-cluster work. |
| Recommendation | Keep future review-surface work bounded to one owner cluster, with invariants recorded before code movement and guard tests added before closeout. |

### Final Residuals and Non-Claims

Remaining review surfaces are explicit future work: other `tests/test_svd.c`
clusters, `tests/test_svd_partial_corpus.c`, broader solver/graph/direct-solver
surfaces, and shared helper dependency tracking beyond the selected SVD guard.

Sprint 201 does not claim broad SVD correctness, new SVD algorithm capability,
partial-SVD ownership changes, public API/ABI changes, library implementation
behavior changes, numerical tolerance changes, performance improvements,
package-manager support, platform support, release readiness, or state-of-the-art
status.

### Day 14 Validation

Commands run after closeout edits:

```sh
make svd-helper-guard
python3 tests/test_svd_helper_guard.py
make docs-check
git diff --check
```

Results:

- `make svd-helper-guard`: PASS.
- `python3 tests/test_svd_helper_guard.py`: PASS.
- `make docs-check`: PASS.
- `git diff --check`: PASS.

### Day 14 Status

Day 14 completes Sprint 201 closeout and retrospective input preparation. All
six Sprint 201 items have clear completion status for the selected SVD helper
review-surface reduction.
