# Sprint 211 Working Notes: Large Review-Surface Reduction

## Sprint Goal

Reduce one large high-risk implementation, test, or tooling surface with
behavior-preserving extraction and ownership guards.

## Scope Boundary

Sprint 211 is a selected review-surface sprint. It may close one bounded
large-surface reduction by ranking candidates, selecting one cluster, recording
no-behavior-change invariants, extracting helper or module ownership, and
adding guard coverage. It must not claim new solver behavior, public API or
ABI changes, numerical tolerance changes, performance improvement, package
support, broad platform parity, release readiness, external-library parity,
state-of-the-art status, or repository-wide review-surface cleanup.

## Day 1: Surface Intake

### Scope Trace

| Epic item | Day 1 intake interpretation | Initial evidence |
| --- | --- | --- |
| 211.1 Candidate Ranking | Measure and inventory large C tests, C implementations, helper headers, and Python tools before scoring. | Candidate ledger and Day 2 scoring inputs. |
| 211.2 Cluster Boundary | Prepare boundary categories for one selected cluster: owner file, non-owner files, preserved behavior, and non-goals. | Boundary checklist placeholder. |
| 211.3 Extraction Design | Reuse Sprint 201 helper-extraction patterns, source-list parity rules, and ownership guard expectations. | Design input notes. |
| 211.4 Extraction Implementation | No implementation on Day 1; extraction begins only after ranking, boundary, design, and baseline evidence. | Deferred to Days 6-8. |
| 211.5 Guard And Test Coverage | Identify existing guard patterns and likely new guard needs for helper ownership, registration, order, and source lists. | Guard inventory. |
| 211.6 Validation And Closeout | Plan focused checks, source-list/CMake checks when registration changes, docs checks, and full C gate when `.c`/`.h` files change. | Validation matrix. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Sprint 211 is a 166-hour sprint to reduce one large high-risk implementation, test, or tooling surface with behavior-preserving extraction and ownership guards. |
| `docs/planning/EPIC_19/reviews/review-codex-2026-09-20.md` | Epic 19 identifies large tests, large implementation files, and large Python report tools as still hard to review; broad maintainability needs targeted extraction, ownership guards, and focused tests. |
| `docs/planning/EPIC_19/reviews/todo-codex-2026-09-20.md` | Closure Track 5 names candidate surfaces and requires one selected cluster, no-behavior-change invariants, helper ownership or responsibility split, guard coverage, focused tests, and full C gate when C/header files change. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Additional review-surface reduction remains residual after Sprint 201; expected evidence includes candidate ranking, selected-cluster rationale, behavior-preservation notes, extraction diff, focused tests, guard coverage, and source-list/CMake parity when registration changes. |
| `docs/planning/EPIC_18/SPRINT_201/RETROSPECTIVE.md` | Sprint 201 closed one selected SVD helper surface and explicitly retained broad review-surface cleanup as residual. |
| `docs/planning/EPIC_18/SPRINT_201/artifacts/day1-large-surface-intake.md` | Reusable intake pattern: measure large files, keep no-behavior-change boundaries, prefer cohesive helper extraction, and defer code movement until ranking and selected-cluster boundary are recorded. |

### Current Large-Surface Inventory

The Day 1 scan used:

```sh
find src tests include scripts benchmarks examples -type f \
  \( -name '*.c' -o -name '*.h' -o -name '*.py' -o -name '*.sh' \) \
  -print0 | xargs -0 wc -l | sort -nr | head -35
```

Line counts are review-risk signals only. Day 2 must still consider cohesion,
ownership clarity, validation cost, and extraction safety.

| File | Lines | Surface type | Initial Day 1 risk tag |
| --- | ---: | --- | --- |
| `tests/test_ldlt_csc.c` | 3469 | C test | Largest current C test surface; direct-solver behavior is rich and high risk if cluster boundary is too broad. |
| `tests/test_ldlt.c` | 3444 | C test | Expanded by Sprint 210 allocation proof; candidate only if it avoids disturbing focused allocation-failure ownership. |
| `tests/test_etree.c` | 3306 | C test | Large symbolic/etree surface; must preserve Sprint 200 symbolic LU allocation proof if selected. |
| `tests/test_integration.c` | 3279 | C test | Broad cross-solver owner; extraction risk is high because clusters can become multi-feature refactors. |
| `tests/test_qr.c` | 3040 | C test | Large QR surface after prior external-reference extraction; remaining clusters may be cohesive but QR behavior is numerically sensitive. |
| `tests/test_iterative.c` | 2929 | C test | Iterative convergence and callback behavior make behavior-preservation subtle. |
| `tests/test_graph.c` | 2764 | C test | Large graph/reordering surface with deterministic but heuristic-heavy behavior. |
| `tests/test_normalize_report_index.py` | 2743 | Python test/tooling | Large validation surface; behavior is schema/CLI-sensitive rather than C-gate-sensitive. |
| `tests/test_svd.c` | 2658 | C test | Still large after Sprint 201; remaining clusters exist but selected-helper ownership must be preserved. |
| `tests/test_chol_csc.c` | 2554 | C test | Large direct-solver test surface with likely helper/fixture candidates. |
| `tests/test_chol_csc_supernodal.c` | 2504 | C test | Large supernodal surface with prior helper/header patterns available. |
| `scripts/run_external_comparison.py` | 2306 | Python tool | Large comparison implementation; semantic and artifact-routing behavior make extraction guard needs important. |
| `tests/test_reorder_nd.c` | 2304 | C test | Large nested-dissection/reorder surface; long focused tests can raise validation cost. |
| `tests/test_eigs.c` | 2155 | C test | Large eigensolver surface; numerical and vector-publication boundaries need care. |
| `src/sparse_ldlt_csc.c` | 2095 | C implementation | Largest production C source; behavior/source-list risk is higher than test helper extraction. |
| `tests/test_colamd.c` | 2017 | C test | Large ordering test surface; candidate if a cohesive helper cluster is found. |
| `tests/test_ilu.c` | 1974 | C test | Large preconditioner surface below 2000 lines but still review-heavy. |
| `scripts/normalize_report_index.py` | 1893 | Python tool | Large report-index implementation; schema and CLI compatibility must be preserved if selected. |
| `tests/test_lu_csr.c` | 1806 | C test | Large direct-solver test surface; candidate if a low-risk fixture/helper cluster is isolated. |
| `src/sparse_lu_csr.c` | 1594 | C implementation | Large production surface below 2000 lines; source extraction would require source-list parity. |
| `src/sparse_ldlt.c` | 1548 | C implementation | Sprint 210 touched related owner; Day 2 should avoid destabilizing new allocation proof unless selecting test-only follow-up. |
| `tests/test_api_docs_routing.py` | 1531 | Python guard test | Large guard suite; extraction would need routing-regression preservation. |
| `tests/test_svd_partial_helpers.h` | 1519 | Test helper header | Large helper header; selected extraction might reduce helper review surface, but header-only dependency guards are needed. |
| `src/sparse_iterative.c` | 1503 | C implementation | Large implementation surface; behavior risk and validation cost are high. |
| `tests/test_bicgstab.c` | 1483 | C test | Iterative solver test surface; selected helper cluster possible. |
| `src/sparse_qr.c` | 1448 | C implementation | QR production surface; recent QR evidence means extraction should be selected and conservative. |
| `tests/test_api_docs_local_only_guard.py` | 1435 | Python guard test | Large local-only guard suite; extraction could improve maintainability but preserve many negative fixtures. |
| `tests/test_eigs_lobpcg.c` | 1417 | C test | Large eigensolver test surface with focused algorithm boundary. |
| `tests/test_eigs_thick_restart.c` | 1377 | C test | Large eigensolver test surface with focused restart boundary. |
| `tests/test_stagnation.c` | 1361 | C test | Large iterative behavior test surface; selected helper extraction possible. |
| `src/sparse_eigs.c` | 1336 | C implementation | Production eigensolver surface; source extraction risk is higher than test/helper extraction. |
| `src/sparse_svd.c` | 1319 | C implementation | Production SVD surface; behavior and numerical validation risk are high. |

### Initial Candidate Families

| Candidate family | Candidate files | Day 1 note |
| --- | --- | --- |
| LDLT CSC test helper cluster | `tests/test_ldlt_csc.c` | Largest test file and likely high review value; must avoid production behavior changes and preserve direct-solver assertions. |
| LDLT selected proof-adjacent cluster | `tests/test_ldlt.c` | Large and recently changed; any selection must avoid weakening Sprint 210 focused allocation-failure gate and registration guard. |
| Etree/symbolic helper cluster | `tests/test_etree.c` | Large helper density is likely, but Sprint 200 symbolic LU proof must remain untouched or explicitly guarded. |
| QR remaining test cluster | `tests/test_qr.c` | High review value but must avoid overlapping prior QR external-reference extraction and Windows QR evidence lanes. |
| Integration fixture cluster | `tests/test_integration.c` | Potential fixture extraction, but broad cross-solver coupling makes cluster selection risky. |
| Iterative solver fixture/diagnostic cluster | `tests/test_iterative.c`, `tests/test_stagnation.c`, `tests/test_bicgstab.c` | Rich behavior surface; selected helper extraction may be safer than implementation extraction. |
| Graph/reorder test cluster | `tests/test_graph.c`, `tests/test_reorder_nd.c`, `tests/test_colamd.c` | Large deterministic test surfaces; validation can be slow and heuristic behavior must be preserved exactly. |
| Cholesky CSC helper cluster | `tests/test_chol_csc.c`, `tests/test_chol_csc_supernodal.c` | Strong helper extraction candidates with direct-solver relevance. |
| Report tooling/test cluster | `scripts/run_external_comparison.py`, `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py` | Python-only extraction may avoid full C gate but needs CLI/schema/artifact regression guards. |
| Production module extraction | `src/sparse_ldlt_csc.c`, `src/sparse_lu_csr.c`, `src/sparse_ldlt.c`, `src/sparse_iterative.c`, `src/sparse_qr.c`, `src/sparse_eigs.c`, `src/sparse_svd.c` | Higher-risk path; requires source-list/CMake parity and full C gate if selected. |
| Large helper-header reduction | `tests/test_svd_partial_helpers.h` | Header-only review surface; requires explicit include/dependency and stale-binary guard coverage. |

### Initial No-Behavior-Change Rules

- Preserve public APIs, public headers, ABI expectations, status codes,
  diagnostics, stdout/stderr text, fixture values, random seeds, tolerances,
  skip behavior, environment-variable behavior, and generated artifact formats.
- Preserve test names, `RUN_TEST(...)` order, registration counts, focused
  gate behavior, assertion intent, cleanup order, and process-global state
  restoration.
- Prefer family-local helper extraction where it reduces review surface without
  production source-list churn.
- If production source or compiled test registration changes, update Makefile,
  CMake, source-list, and focused parity guards in the same sprint.
- Do not add performance, package, platform, release, public API, ABI, broad
  review-surface, external-library parity, or state-of-the-art claims.

### Guard Pattern Inventory

| Guard pattern | Existing example | Sprint 211 reuse |
| --- | --- | --- |
| Helper ownership script and regression | `make svd-helper-guard`; `tests/test_svd_helper_guard.py` | Reuse for test-helper extractions with proof-owner registrations and helper-only boundaries. |
| Focused allocation proof registration guard | `tests/test_ldlt_allocation_failure_gate_registration.py` | Reuse active-line/comment-aware registration checks if selected cluster has a focused runner. |
| Source-list parity | `make source-list-check`; CMake quality-review paths | Required if compiled production source or registered test source changes. |
| Python guard fixtures | API docs, manifest, and workflow guard tests | Reuse mutation-style negative fixtures if selecting Python tooling. |
| Makefile prerequisite guard | SVD helper and LDLT gate guards | Required for helper headers or stale-binary-sensitive dependencies. |

### Validation Matrix

| Validation | Day 1 status | Notes |
| --- | --- | --- |
| `git diff --check` | Planned for Day 1 closeout. | Documentation-only Day 1 changes. |
| Focused selected-cluster tests | Not yet applicable. | Cluster selection is scheduled for Day 3 after Day 2 ranking. |
| Ownership guard | Not yet applicable. | Guard type depends on selected cluster. |
| Source-list/CMake parity | Not yet applicable. | Required only if source/test registration changes. |
| Documentation checks | Not yet applicable. | Day 1 creates planning evidence only. |
| `make format && make lint && make test` | Not required for Day 1. | No `.c` or `.h` files modified. |

### Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Selecting too broad a surface | Sprint 211 should close one selected reduction rather than partially touch many files. | Day 2 ranking and Day 3 boundary must freeze one cluster and retain broad cleanup residuals. |
| Production extraction changes behavior | Source movement can affect linkage, visibility, source lists, and subtle behavior. | Prefer test/helper extraction unless production owner boundary is clearly safer and fully validated. |
| Weak ownership guard | Moved code can drift back into the large file or wrong helper. | Add guard checks for owner file, non-owner absence, registrations, includes, prerequisites, and source lists. |
| Registration/order drift | Review-surface extraction can silently change test execution order or focused gate behavior. | Preserve `RUN_TEST(...)` order and add active-registration/order checks where relevant. |
| Stale documentation and metrics | Review comments often catch wrong file names and line-count ownership after extraction. | Keep changed-surface inventory and before/after metrics current through closeout. |
| Slow focused validation | Graph/reorder and some numerical tests can be slow. | Record focused command cost during baseline validation and choose bounded checks where possible. |
| Claim overreach | One selected reduction is not broad maintainability or state-of-the-art evidence. | Keep public and maintainer wording selected-cluster-only. |

### Open Questions For Day 2

1. Which candidate has the strongest combination of line-count reduction,
   ownership clarity, and low behavior risk?
2. Which candidate has a cohesive helper or module boundary that can be
   extracted without changing public behavior?
3. Which candidate has focused tests or guard patterns already close enough to
   extend in one sprint?
4. Which candidate avoids destabilizing recently closed Sprint 201 and Sprint
   210 proof-owner boundaries?
5. Which candidate can produce a meaningful before/after review-surface metric
   without requiring broad repository cleanup?

### Day 1 Validation

Commands planned for Day 1 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 1 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 2: Candidate Ranking

### Measurement Commands

Day 2 reused the Day 1 line-count scan and added function/registration density
signals for the largest C test candidates:

```sh
for f in tests/test_ldlt_csc.c tests/test_ldlt.c tests/test_etree.c \
  tests/test_integration.c tests/test_qr.c tests/test_iterative.c \
  tests/test_graph.c tests/test_svd.c tests/test_chol_csc.c \
  tests/test_chol_csc_supernodal.c tests/test_reorder_nd.c \
  tests/test_eigs.c tests/test_colamd.c; do
    printf '%s\t' "$f"
    rg -c '^static .*\(' "$f" | tr -d '\n'
    printf '\t'
    rg -c 'RUN_TEST\(' "$f" | tr -d '\n'
    printf '\n'
done
```

Helper/guard inventory commands:

```sh
rg --files tests | rg '(_helpers|_fixtures|guard).*\.(h|py)$' | sort
sed -n '1,220p' scripts/check_ldlt_csc_helper_guard.sh
wc -l tests/test_ldlt_csc_fixtures.h \
  tests/test_ldlt_csc_oracle_helpers.h \
  tests/test_ldlt_csc_supernode_helpers.h \
  tests/test_ldlt_csc.c
```

### Ranking Criteria

Scores use a 1-5 scale where 5 is strongest for Sprint 211. The scoring favors
complete selected-cluster closure over broad cleanup.

| Criterion | Meaning |
| --- | --- |
| Size payoff | Expected reduction in a large review surface. |
| Reviewer burden | How much the current file structure slows review. |
| Ownership clarity | Whether a cohesive helper/test family can be named and owned. |
| Helper cohesion | Whether candidate helpers naturally belong together. |
| Behavior-risk control | Lower risk of changing solver behavior, diagnostics, tolerances, registration order, or process-global state earns a higher score. |
| Existing guard leverage | Existing helper, registration, source-list, or mutation guard patterns reduce sprint risk. |
| Focused validation availability | A focused test/guard path exists or can be added without broad validation ambiguity. |
| Sprint fit | Likelihood of complete closure inside Sprint 211. |

### Function And Registration Density

| File | Lines | Static-function signal | `RUN_TEST` count |
| --- | ---: | ---: | ---: |
| `tests/test_ldlt_csc.c` | 3469 | 119 | 100 |
| `tests/test_ldlt.c` | 3444 | 115 | 95 |
| `tests/test_etree.c` | 3306 | 129 | 107 |
| `tests/test_integration.c` | 3279 | 53 | 58 |
| `tests/test_qr.c` | 3040 | 72 | 79 |
| `tests/test_iterative.c` | 2929 | 94 | 85 |
| `tests/test_graph.c` | 2764 | 68 | 61 |
| `tests/test_svd.c` | 2658 | 90 | 114 |
| `tests/test_chol_csc.c` | 2554 | 111 | 92 |
| `tests/test_chol_csc_supernodal.c` | 2504 | 72 | 62 |
| `tests/test_reorder_nd.c` | 2304 | 47 | 35 |
| `tests/test_eigs.c` | 2155 | 51 | 43 |
| `tests/test_colamd.c` | 2017 | 78 | 70 |

### Ranked Candidate Table

| Rank | Candidate family | Size payoff | Reviewer burden | Ownership clarity | Helper cohesion | Behavior-risk control | Existing guard leverage | Focused validation | Sprint fit | Total | Day 2 disposition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `tests/test_ldlt_csc.c` selected native/parity helper cluster | 5 | 5 | 4 | 4 | 4 | 5 | 5 | 5 | 37 | Selected primary. Largest file, high direct-solver review value, and existing LDLT CSC helper guard/header pattern. |
| 2 | `tests/test_chol_csc.c` selected fixture/helper cluster | 4 | 4 | 4 | 4 | 4 | 3 | 5 | 4 | 32 | Fallback. Strong helper density and focused direct-solver test binary, but guard coverage is less ready. |
| 3 | `tests/test_chol_csc_supernodal.c` helper/fixture cluster | 4 | 4 | 4 | 4 | 3 | 4 | 4 | 4 | 31 | Alternate. Existing helper header helps, but dense backend/env contracts increase process-global risk. |
| 4 | `tests/test_qr.c` remaining helper cluster | 4 | 4 | 3 | 3 | 3 | 4 | 5 | 3 | 29 | Alternate. Guard precedent exists, but QR behavior and prior QR extraction make cluster choice more delicate. |
| 5 | `tests/test_svd.c` remaining selected cluster | 3 | 4 | 3 | 3 | 4 | 5 | 5 | 3 | 30 | Deferred. Still large, but Sprint 201 already reduced one selected SVD cluster; preserve current helper guard. |
| 6 | `tests/test_ldlt.c` selected cluster outside allocation proof | 5 | 5 | 3 | 3 | 2 | 4 | 5 | 2 | 29 | Deferred. Large payoff, but Sprint 210 focused allocation proof is too fresh to risk in Sprint 211 unless no better candidate exists. |
| 7 | `tests/test_etree.c` non-symbolic-LU helper cluster | 5 | 5 | 3 | 3 | 2 | 4 | 5 | 2 | 29 | Deferred. High density, but Sprint 200 symbolic LU proof ownership should remain undisturbed. |
| 8 | Python report tooling cluster | 4 | 4 | 3 | 3 | 3 | 3 | 4 | 4 | 28 | Deferred. Avoids C quality gate if Python-only, but CLI/schema/artifact behavior is broad. |
| 9 | Graph/reorder test cluster | 4 | 4 | 3 | 3 | 3 | 2 | 2 | 3 | 24 | Deferred. Heuristic behavior and long focused validation make complete closure harder. |
| 10 | Production module extraction from `src/sparse_ldlt_csc.c` | 4 | 5 | 3 | 3 | 1 | 2 | 3 | 2 | 23 | Deferred. Highest production payoff, but source-list/CMake and behavior risks are too high for the first Epic 19 review-surface sprint. |

### Selected Primary Cluster

Day 2 selects a bounded `tests/test_ldlt_csc.c` native/parity helper cluster as
the primary Sprint 211 candidate for Day 3 boundary tracing.

Initial primary boundary hypothesis:

- proof-owner file: `tests/test_ldlt_csc.c`;
- likely selected section: native 1x1/2x2/mixed-pivot wrapper parity helpers
  and repeated comparison utilities around the native LDLT CSC parity tests;
- possible helper owner: existing family-local helper headers such as
  `tests/test_ldlt_csc_oracle_helpers.h` or a new selected helper header if
  Day 3 finds the existing helpers semantically too broad;
- existing guard starting point: `make ldlt-csc-helper-guard` and
  `scripts/check_ldlt_csc_helper_guard.sh`;
- focused behavior check: `./build/test_ldlt_csc` or the smallest available
  focused LDLT CSC test invocation if a selected runner is introduced;
- expected reduction type: move helper bodies or tightly coupled assertion
  utilities, not `RUN_TEST(...)` registrations, unless Day 3 explicitly
  changes the proof-owner model.

Selection rationale:

- `tests/test_ldlt_csc.c` is the largest current review surface and has high
  direct-solver value.
- The family already has three helper headers:
  `tests/test_ldlt_csc_fixtures.h`,
  `tests/test_ldlt_csc_oracle_helpers.h`, and
  `tests/test_ldlt_csc_supernode_helpers.h`.
- `scripts/check_ldlt_csc_helper_guard.sh` already proves helper headers are
  included by the proof-owner binary, remain header-only, and are absent from
  Makefile/CMake/library-source registration.
- The selected cluster can plausibly improve reviewability without production
  source movement, public API changes, or source-list churn.

### Fallback Cluster

If Day 3 cannot identify a cohesive LDLT CSC cluster that can move without
weakening current helper ownership or direct-solver behavior, Sprint 211 falls
back to a selected `tests/test_chol_csc.c` fixture/helper cluster.

Fallback rationale:

- `tests/test_chol_csc.c` is also large and helper dense.
- It has a focused proof-owner binary already registered in Make/CMake.
- It avoids the recently changed Sprint 210 LDLT allocation proof surface.
- It may need a new helper guard because current guard infrastructure is
  stronger for LDLT CSC and SVD than for Cholesky CSC.

Fallback trigger conditions:

- Day 3 finds the primary LDLT CSC candidate has no clean helper boundary.
- Moving the selected LDLT CSC helpers would require production source changes
  or broad registration changes.
- The current LDLT CSC helper headers cannot accept the selected cluster
  without confusing existing fixture/oracle/supernode ownership.
- Focused LDLT CSC validation proves too slow or too broad for a selected
  review-surface sprint.

### Deferred Candidate Notes

| Candidate | Why deferred on Day 2 | Future handoff |
| --- | --- | --- |
| `tests/test_ldlt.c` | Sprint 210 just added focused allocation-failure ownership, options preservation checks, and active registration guards. | Revisit after the allocation proof is stable and select only a cluster outside the focused gate. |
| `tests/test_etree.c` | Sprint 200 symbolic LU proof must remain isolated. | Select only with explicit guard preservation for symbolic LU allocation proof. |
| `tests/test_integration.c` | Cross-solver lifecycle behavior makes extraction too easy to broaden. | Choose only a fixture-only helper cluster with no registration changes. |
| `tests/test_qr.c` | QR has prior extraction and selected Windows QR evidence lanes; remaining clusters need careful boundary work. | Good future candidate if a non-overlapping sparse/economy cluster is identified. |
| `tests/test_svd.c` | Sprint 201 already closed one selected SVD helper cluster; remaining SVD work should not weaken `svd-helper-guard`. | Continue with a new selected SVD cluster only after preserving existing helper ownership. |
| Production sources | Source movement has higher behavior and source-list/CMake risk. | Reserve for a production-module sprint with deeper baseline validation. |
| Python report tooling | Large and valuable but broad CLI/schema/artifact behavior is a different closure shape. | Candidate for a tooling-specific review-surface sprint. |

### Item 211.1 Status

Item 211.1 is complete for Day 2. Candidate surfaces were measured, scored,
ranked, and narrowed to one primary and one fallback cluster. Day 3 should
trace the selected LDLT CSC native/parity helper cluster and freeze the exact
behavior-preservation boundary before any extraction edits begin.

### Day 2 Validation

Commands planned for Day 2 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 2 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 3: Cluster Boundary

### Purpose

Day 3 freezes the selected extraction boundary before implementation. The
selected review-surface reduction remains a behavior-preserving LDLT CSC test
helper extraction, not a production-source split or solver behavior change.

### Entry-Point Trace

| Surface | Current owner | Day 3 disposition |
| --- | --- | --- |
| Proof-owner binary | `tests/test_ldlt_csc.c` | Remains the only executable proof owner. |
| Build registration | `$(TESTDIR)/test_ldlt_csc.c` in `Makefile`; `add_sparse_test(test_ldlt_csc)` in `CMakeLists.txt` | Must remain unchanged unless a later guard explicitly proves equivalent registration. |
| Existing helper owners | `tests/test_ldlt_csc_fixtures.h`, `tests/test_ldlt_csc_oracle_helpers.h`, `tests/test_ldlt_csc_supernode_helpers.h` | Remain header-only family-local helper owners. |
| Guard owner | `scripts/check_ldlt_csc_helper_guard.sh` and `make ldlt-csc-helper-guard` | Must continue to bind helpers to `test_ldlt_csc.c`; may be extended on later days. |
| Selected test registrations | `RUN_TEST(test_native_1x1_...)` and `RUN_TEST(test_native_2x2_...)` block in `tests/test_ldlt_csc.c` | Must remain registered in the same relative order. |

### Selected Cluster

The selected cluster is the Sprint 18 Day 3/Day 4 native-kernel parity block
in `tests/test_ldlt_csc.c`.

Owned tests:

- `test_native_1x1_diagonal_matches_wrapper`;
- `test_native_1x1_tridiagonal_matches_wrapper`;
- `test_native_1x1_mixed_indefinite_matches_wrapper`;
- `test_native_1x1_with_swap_matches_wrapper`;
- `test_native_1x1_tridiag_large_matches_wrapper`;
- `test_native_detects_near_zero_1x1_pivot`;
- `test_native_1x1_identity_matches_wrapper`;
- `test_native_2x2_forced_matches_wrapper`;
- `test_native_2x2_nonadjacent_partner_matches_wrapper`;
- `test_native_mixed_pivots_matches_wrapper`;
- `test_native_mixed_pivots_larger_matches_wrapper`;
- `test_native_2x2_solve_matches_linked_list`;
- `test_native_2x2_inertia_matches_wrapper`.

The initial implementation direction is to move only these selected test bodies
or their tightly coupled native-parity assertion helpers into a family-local
helper header. The primary destination candidate is a new selected helper
header, tentatively `tests/test_ldlt_csc_native_parity_helpers.h`, because the
existing oracle helper already owns dense-oracle comparison utilities while the
proof-owner file currently owns the test registrations.

### Existing Dependencies

| Dependency | Current location | Boundary decision |
| --- | --- | --- |
| `check_native_matches_wrapper()` | `tests/test_ldlt_csc_oracle_helpers.h` | Reuse as-is; do not duplicate. |
| `ldlt_factorizations_match()` and `ldlt_column_nonzeros_match()` | `tests/test_ldlt_csc_oracle_helpers.h` | Reuse as-is; owner stays unchanged. |
| `rel_residual()` | `tests/test_ldlt_csc.c`, Day 9 solve block | Keep in proof-owner unless Day 4 designs a safe shared solve-helper destination. |
| `ldlt_csc_set_kernel_override()` state reset | selected native tests and oracle helper | Preserve every reset to `LDLT_CSC_KERNEL_DEFAULT`. |
| Sparse fixture construction | selected native tests | May move with selected test bodies; matrix values and insertion order must remain unchanged. |
| `RUN_TEST(...)` registrations | `tests/test_ldlt_csc.c` main | Keep in proof-owner and preserve order. |

### No-Behavior-Change Invariants

Day 4 and later edits must preserve:

- all selected test names and their registration order;
- all matrix dimensions, inserted entries, insertion order, and tolerances;
- status-code expectations, especially `SPARSE_ERR_SINGULAR` for the near-zero
  native pivot test;
- `LDLT_CSC_KERNEL_NATIVE`, `LDLT_CSC_KERNEL_WRAPPER`, and
  `LDLT_CSC_KERNEL_DEFAULT` override sequencing;
- `ldlt_csc_validate()` coverage inside wrapper/native comparisons;
- the residual threshold in `test_native_2x2_solve_matches_linked_list`;
- stdout/stderr text emitted by existing tests, including no new diagnostic
  output in selected native-parity tests;
- skip behavior, fixture data, and allocation ownership;
- Makefile/CMake test registration and CTest count;
- helper headers remaining header-only and absent from library source lists.

### Owner And Non-Owner Files

Proposed owner files for later extraction:

- `tests/test_ldlt_csc.c` remains proof-owner and registration owner;
- `tests/test_ldlt_csc_oracle_helpers.h` remains dense-oracle comparison owner;
- new `tests/test_ldlt_csc_native_parity_helpers.h` may own selected native
  parity test bodies if Day 4 confirms the split;
- `scripts/check_ldlt_csc_helper_guard.sh` may own new helper-registration and
  registration-order checks.

Explicit non-owner files:

- `src/sparse_ldlt_csc.c` and other production sources;
- public headers under `include/`;
- `tests/test_ldlt.c`, including Sprint 210 allocation-failure proof;
- `tests/test_svd.c` and SVD helper guards from Sprint 201;
- `tests/test_chol_csc.c`, unless the Day 2 fallback is explicitly activated;
- Make/CMake source lists beyond preserving or guarding `test_ldlt_csc`.

### Validation Expectations

Day 3 is documentation-only. Later implementation days should plan at least:

- `make ldlt-csc-helper-guard`;
- the focused LDLT CSC proof-owner test, likely `./build/test_ldlt_csc` after
  normal build setup or the equivalent Make/CMake test invocation;
- `ctest -N` or the repository parity checks if a CMake registration surface
  changes;
- full `make format && make lint && make test` if `.c` or `.h` files are
  modified.

### Item 211.2 Status

Item 211.2 is complete for Day 3. The selected boundary is concrete: the Sprint
18 Day 3/Day 4 LDLT CSC native-kernel parity block remains owned by
`test_ldlt_csc`, may move into a selected family-local helper header after Day
4 design, and must preserve all observable test behavior, registrations,
guarded header-only ownership, and non-claim scope.

### Day 3 Validation

Commands planned for Day 3 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 3 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 4: Extraction Design

### Design Decision

Day 4 selects a header-only extraction for the LDLT CSC native-parity test
cluster. The implementation days should create a new selected helper header:

```c
tests/test_ldlt_csc_native_parity_helpers.h
```

The header should use the include guard:

```c
#ifndef TEST_LDLT_CSC_NATIVE_PARITY_HELPERS_H
#define TEST_LDLT_CSC_NATIVE_PARITY_HELPERS_H
...
#endif
```

This file will own only the selected Sprint 18 Day 3/Day 4 native parity test
bodies. `tests/test_ldlt_csc.c` remains the proof-owner executable, include
owner, and `RUN_TEST(...)` registration owner.

### File Shape

| Design point | Decision |
| --- | --- |
| File type | Header-only test helper, matching existing LDLT CSC helper headers. |
| Visibility | `static` functions in the including translation unit; no external symbols. |
| Include direction | `tests/test_ldlt_csc.c` includes the new header between `test_ldlt_csc_fixtures.h` and `test_ldlt_csc_oracle_helpers.h`; the helper itself includes `test_ldlt_csc_oracle_helpers.h` for native/wrapper comparison helpers. |
| Source-list registration | Do not add the helper to `TEST_SRCS`, `TEST_BINS`, `CMakeLists.txt`, CTest registration, or `build-metadata/library_sources.txt`; the only Makefile mention should be the `build/test_ldlt_csc` prerequisite. |
| Test registration | Keep all selected `RUN_TEST(...)` calls in `tests/test_ldlt_csc.c`. |
| Public API impact | None. No production source or public header changes. |

### Planned Include Order

The implementation should use this include order in `tests/test_ldlt_csc.c`:

```c
#include "test_ldlt_csc_fixtures.h"
#include "test_ldlt_csc_native_parity_helpers.h"
#include "test_ldlt_csc_oracle_helpers.h"
#include "test_ldlt_csc_supernode_helpers.h"
```

The native-parity header depends on:

- `SparseMatrix` construction and access from already included public/internal
  sparse headers;
- `LdltCsc`, `LDLT_CSC_KERNEL_NATIVE`, `LDLT_CSC_KERNEL_WRAPPER`, and
  `LDLT_CSC_KERNEL_DEFAULT` from `sparse_ldlt_csc_internal.h`;
- test macros from `test_framework.h`;
- `check_native_matches_wrapper()` from `test_ldlt_csc_oracle_helpers.h`;
- a solve-residual helper for `test_native_2x2_solve_matches_linked_list`.

### Dependency Strategy

| Dependency | Day 4 design |
| --- | --- |
| Native/wrapper comparison | Reuse `check_native_matches_wrapper()` from `test_ldlt_csc_oracle_helpers.h`; do not move or duplicate it. |
| Dense factor comparison | Keep `ldlt_factorizations_match()` and `ldlt_column_nonzeros_match()` in `test_ldlt_csc_oracle_helpers.h`. |
| `rel_residual()` | Keep in `tests/test_ldlt_csc.c` for Day 6 unless implementation proves a lower-risk declaration strategy. The new helper header may use a forward declaration before selected test bodies. |
| Matrix fixtures | Move selected inline fixture construction with the selected test bodies; values and insertion order are immutable. |
| Kernel override resets | Keep each existing native/wrapper/default reset sequence intact. |
| `fabs()` | If the selected inertia test moves, the new helper header may include `<math.h>` or rely on prior includes; prefer including `<math.h>` explicitly for local readability. |

### Extraction Batches

Day 6 should move the Sprint 18 Day 3 1x1 native parity block:

- `test_native_1x1_diagonal_matches_wrapper`;
- `test_native_1x1_tridiagonal_matches_wrapper`;
- `test_native_1x1_mixed_indefinite_matches_wrapper`;
- `test_native_1x1_with_swap_matches_wrapper`;
- `test_native_1x1_tridiag_large_matches_wrapper`;
- `test_native_detects_near_zero_1x1_pivot`;
- `test_native_1x1_identity_matches_wrapper`.

Day 7 should move the Sprint 18 Day 4 2x2/native solve parity block:

- `test_native_2x2_forced_matches_wrapper`;
- `test_native_2x2_nonadjacent_partner_matches_wrapper`;
- `test_native_mixed_pivots_matches_wrapper`;
- `test_native_mixed_pivots_larger_matches_wrapper`;
- `test_native_2x2_solve_matches_linked_list`;
- `test_native_2x2_inertia_matches_wrapper`.

Day 8 should harden the guard and resolve any remaining include or helper
ownership drift before broader validation.

### Guard Strategy

Extend `scripts/check_ldlt_csc_helper_guard.sh` when the new helper is created:

1. Add `tests/test_ldlt_csc_native_parity_helpers.h` to `HELPERS`.
2. Require the new include guard and exactly one include in `tests/test_ldlt_csc.c`.
3. Require the helper headers as explicit prerequisites of the existing
   `build/test_ldlt_csc` proof-owner target and reject extra Makefile
   occurrences outside that rule.
4. Reject `add_sparse_test(test_ldlt_csc_native_parity_helpers)` or equivalent
   standalone proof-owner registration.
5. Add active-registration checks for the selected native parity `RUN_TEST(...)`
   calls in `tests/test_ldlt_csc.c`.
6. Add increasing-order checks so the Day 3 1x1 registrations stay before the
   Day 4 2x2 registrations and before the Day 9 solve block.
7. Avoid raw substring checks that can be satisfied by comments; active-line
   matching should ignore line comments and block comments.

### Build And Source-List Plan

No CMake test registration change is planned. The Makefile must list the
helper headers only as prerequisites of the existing `test_ldlt_csc`
proof-owner target so helper-only edits rebuild the proof-owner binary. Do not
create a new executable test target.

### Behavior Preservation Checks For Implementation

Before moving code, Day 5 should capture:

- `make ldlt-csc-helper-guard`;
- focused `test_ldlt_csc` behavior;
- selected registration order from `tests/test_ldlt_csc.c`;
- line counts for `tests/test_ldlt_csc.c` and the LDLT CSC helper headers.

After implementation, rerun the same focused checks. If any `.c` or `.h` file
changes, close the implementation day with `make format && make lint && make
test` before committing or PR closeout.

### Item 211.3 Status

Item 211.3 is complete for Day 4. The selected extraction has an
implementation-ready owner file, include strategy, dependency map, batch plan,
source-list plan, and guard strategy. Implementation remains blocked until Day
5 records pre-extraction baseline behavior and measurements.

### Day 4 Validation

Commands planned for Day 4 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 4 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 5: Baseline Validation

### Purpose

Day 5 captures the pre-extraction behavior and review-surface metrics for the
selected LDLT CSC native-parity cluster. This is the baseline that Days 6-8
must preserve while moving code into the selected helper owner.

### Environment Snapshot

| Field | Value |
| --- | --- |
| Branch | `sprint-211` |
| Baseline commit | `64cb32cc` |
| Selected proof owner | `tests/test_ldlt_csc.c` |
| Selected cluster span | Lines 2640-2934, `295` lines |
| Selected registrations | Lines 3439-3453 in `tests/test_ldlt_csc.c` |

### Commands And Results

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed. Proof-owner registration, helper headers, and header-only registration checks all passed. |
| `./build/test_ldlt_csc` | Passed. `100` tests run, `0` failed, `0` skipped, `3556` assertions, `ALL TESTS PASSED`. |
| `ctest -N --test-dir build/quality-review-cmake` | Available. Reported `Total Tests: 60`. |
| `ctest -N --test-dir build` | Not useful for this local baseline. The normal `build` tree reported `Total Tests: 0`; use the focused executable and quality-review CMake tree for this sprint baseline. |

### Baseline Line Counts

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3469 |
| `tests/test_ldlt_csc_fixtures.h` | 145 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 151 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 139 |

The planned new helper `tests/test_ldlt_csc_native_parity_helpers.h` does not
exist yet. Day 6 should create it, move the 1x1 native-parity batch, and record
the first before/after line-count delta.

### Selected Registration Baseline

The selected active registration order is:

| Line | Registration |
| ---: | --- |
| 3439 | `RUN_TEST(test_native_1x1_diagonal_matches_wrapper);` |
| 3440 | `RUN_TEST(test_native_1x1_tridiagonal_matches_wrapper);` |
| 3441 | `RUN_TEST(test_native_1x1_mixed_indefinite_matches_wrapper);` |
| 3442 | `RUN_TEST(test_native_1x1_with_swap_matches_wrapper);` |
| 3443 | `RUN_TEST(test_native_1x1_tridiag_large_matches_wrapper);` |
| 3444 | `RUN_TEST(test_native_detects_near_zero_1x1_pivot);` |
| 3445 | `RUN_TEST(test_native_1x1_identity_matches_wrapper);` |
| 3448 | `RUN_TEST(test_native_2x2_forced_matches_wrapper);` |
| 3449 | `RUN_TEST(test_native_2x2_nonadjacent_partner_matches_wrapper);` |
| 3450 | `RUN_TEST(test_native_mixed_pivots_matches_wrapper);` |
| 3451 | `RUN_TEST(test_native_mixed_pivots_larger_matches_wrapper);` |
| 3452 | `RUN_TEST(test_native_2x2_solve_matches_linked_list);` |
| 3453 | `RUN_TEST(test_native_2x2_inertia_matches_wrapper);` |

Day 8 guard hardening should assert these registrations as active code, not
commented text, and keep the 1x1 block before the 2x2 block and both before the
Day 9 solve block.

### Include And Build Registration Baseline

Current LDLT CSC helper includes in `tests/test_ldlt_csc.c`:

- `#include "test_ldlt_csc_fixtures.h"`;
- `#include "test_ldlt_csc_oracle_helpers.h"`;
- `#include "test_ldlt_csc_supernode_helpers.h"`.

Current proof-owner registrations:

- `Makefile` contains `$(TESTDIR)/test_ldlt_csc.c`;
- `CMakeLists.txt` contains `add_sparse_test(test_ldlt_csc)`.

Day 6 should add `#include "test_ldlt_csc_native_parity_helpers.h"` between
the fixture and oracle helper includes and should not create a new proof-owner
test target.

### Day 5 Risks And Fallbacks

| Risk | Baseline disposition |
| --- | --- |
| Normal `build` tree lacks CTest registration locally. | Use `./build/test_ldlt_csc` for focused proof and `build/quality-review-cmake` for CTest count baseline. |
| Existing guard does not yet know the planned native-parity helper. | Expected before Day 6. Day 8 must add the helper and active-registration checks. |
| Selected block uses `rel_residual()` defined later in the file. | Day 6/7 must keep a valid forward declaration or record a safer helper-local solve residual design. |
| Moving test bodies could accidentally change registration order. | Keep registrations in `tests/test_ldlt_csc.c`; Day 8 guard must check active order. |

### Item 211.4 Readiness

Day 5 completes the pre-extraction baseline required before Day 6
implementation. The selected proof-owner binary and helper guard pass, the CMake
quality-review surface is counted, and before-state line counts and
registration order are recorded.

### Day 5 Validation

Commands planned for Day 5 closeout:

```sh
git diff --check
git status --short
git diff --name-only -- '*.c' '*.h'
```

Day 5 changes planning documentation only. No `.c` or `.h` files are modified,
so the full C quality gate is not required.

## Day 6: Extraction Batch One

### Purpose

Day 6 implements the first behavior-preserving extraction batch from the
selected LDLT CSC native-parity cluster. The Sprint 18 Day 3 1x1 native parity
test bodies moved from `tests/test_ldlt_csc.c` into the new family-local helper
header `tests/test_ldlt_csc_native_parity_helpers.h`.

### Changed Files

| File | Day 6 change |
| --- | --- |
| `tests/test_ldlt_csc.c` | Added the new native-parity helper include and removed the moved 1x1 native parity test bodies. Kept all `RUN_TEST(...)` registrations in the proof-owner file. |
| `tests/test_ldlt_csc_native_parity_helpers.h` | New header-only selected helper owner for the Sprint 18 Day 3 1x1 native parity test bodies. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Recorded Day 6 implementation evidence. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day6-extraction-batch-one.md` | New Day 6 implementation artifact. |

### Moved Test Bodies

Day 6 moved these tests without changing their names or `RUN_TEST(...)`
registrations:

- `test_native_1x1_diagonal_matches_wrapper`;
- `test_native_1x1_tridiagonal_matches_wrapper`;
- `test_native_1x1_mixed_indefinite_matches_wrapper`;
- `test_native_1x1_with_swap_matches_wrapper`;
- `test_native_1x1_tridiag_large_matches_wrapper`;
- `test_native_detects_near_zero_1x1_pivot`;
- `test_native_1x1_identity_matches_wrapper`.

The native-parity helper includes `test_ldlt_csc_oracle_helpers.h` and reuses
`check_native_matches_wrapper()` rather than duplicating dense-oracle
comparison helpers.

### Include And Registration State

Current LDLT CSC helper include order:

```c
#include "test_ldlt_csc_fixtures.h"
#include "test_ldlt_csc_native_parity_helpers.h"
#include "test_ldlt_csc_oracle_helpers.h"
#include "test_ldlt_csc_supernode_helpers.h"
```

Selected registrations remain in `tests/test_ldlt_csc.c` and retain their
relative order. After the move, the selected 1x1 registrations begin at line
3316 and the Day 4 2x2 registrations begin at line 3325.

### Review-Surface Metrics

| File | Day 5 baseline | Day 6 after batch one | Delta |
| --- | ---: | ---: | ---: |
| `tests/test_ldlt_csc.c` | 3469 | 3346 | -123 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 0 | 133 | +133 |
| `tests/test_ldlt_csc_fixtures.h` | 145 | 145 | 0 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 151 | 151 | 0 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | 140 | 0 |

The net line count increased by 10 lines because the new header has its own
include guard, includes, and owner comment. The intended Day 6 reduction is
ownership separation and a smaller proof-owner file, not behavior or total
repository line reduction.

### Focused Validation

| Command | Result |
| --- | --- |
| `make build/test_ldlt_csc` | Passed; rebuilt `build/test_ldlt_csc`. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions, `ALL TESTS PASSED`. |
| `make ldlt-csc-helper-guard` | Passed with the existing guard. The guard does not yet cover the new helper and must be extended on Day 8 or Day 9. |

### Deferred Guard Work

The new helper is intentionally not yet wired into
`scripts/check_ldlt_csc_helper_guard.sh` on Day 6. Day 8 or Day 9 must add it
to `HELPERS`, enforce its include guard and single include, reject build/source
registration, and add active registration/order checks for the selected native
parity tests.

### Item 211.4 Status

Item 211.4 has begun. The first coherent extraction batch is complete, the
proof-owner executable still passes, and the selected `RUN_TEST(...)`
registrations remain in `tests/test_ldlt_csc.c`.

### Day 6 Validation

Because Day 6 modifies `.h` and `.c` files, the required closeout gate is:

```sh
make format && make lint && make test
```

Full-gate result: `make format && make lint && make test` passed after the
Day 6 extraction. The gate rebuilt the formatted C surface, completed strict
warning compile/lint checks, and passed the full test suite, including
`./build/test_ldlt_csc` with `100` tests, `0` failed, `0` skipped, and `3556`
assertions.

## Day 7: Extraction Batch Two

### Purpose

Day 7 completes the planned selected native-parity extraction by moving the
Sprint 18 Day 4 2x2/native solve parity test bodies from
`tests/test_ldlt_csc.c` into
`tests/test_ldlt_csc_native_parity_helpers.h`.

### Changed Files

| File | Day 7 change |
| --- | --- |
| `tests/test_ldlt_csc.c` | Removed the selected Sprint 18 Day 4 2x2/native solve parity test bodies. Kept all `RUN_TEST(...)` registrations in the proof-owner file. |
| `tests/test_ldlt_csc_native_parity_helpers.h` | Added the selected 2x2/native solve parity test bodies, `<math.h>` for local `fabs()` use, and a local forward declaration for `rel_residual()`. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Recorded Day 7 implementation evidence. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day7-extraction-batch-two.md` | New Day 7 implementation artifact. |

### Moved Test Bodies

Day 7 moved these tests without changing their names or `RUN_TEST(...)`
registrations:

- `test_native_2x2_forced_matches_wrapper`;
- `test_native_2x2_nonadjacent_partner_matches_wrapper`;
- `test_native_mixed_pivots_matches_wrapper`;
- `test_native_mixed_pivots_larger_matches_wrapper`;
- `test_native_2x2_solve_matches_linked_list`;
- `test_native_2x2_inertia_matches_wrapper`.

The native-parity helper now owns the complete selected native parity test-body
cluster from Sprint 18 Days 3-4. The proof-owner file still owns execution
order and active `RUN_TEST(...)` registration.

### Registration State

Selected registrations remain in `tests/test_ldlt_csc.c` and retain their
relative order:

- 1x1 native parity registrations: lines 3144-3150;
- 2x2/native solve parity registrations: lines 3153-3158.

### Review-Surface Metrics

| File | Day 6 after batch one | Day 7 after batch two | Delta |
| --- | ---: | ---: | ---: |
| `tests/test_ldlt_csc.c` | 3346 | 3174 | -172 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 133 | 303 | +170 |
| `tests/test_ldlt_csc_fixtures.h` | 145 | 145 | 0 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 151 | 151 | 0 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | 140 | 0 |

Compared with the Day 5 baseline, the proof-owner file is now 295 lines
smaller (`3469` to `3174`). The selected test surface has a net increase of 8
lines due to the new helper guard, local includes, and owner comments.

### Focused Validation

| Command | Result |
| --- | --- |
| `make build/test_ldlt_csc` | Passed; rebuilt `build/test_ldlt_csc`. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions, `ALL TESTS PASSED`. |
| `make ldlt-csc-helper-guard` | Passed with the existing guard. The guard does not yet cover the new helper and must be extended on Day 8 or Day 9. |

### Required Full Gate

Because Day 7 modifies `.h` and `.c` files, the required closeout gate ran:

```sh
make format && make lint && make test
```

Result: passed. The gate completed formatting, strict warning compile, lint,
cppcheck, and the full test suite. The full suite included
`./build/test_ldlt_csc`, which passed with `100` tests, `0` failed, `0`
skipped, and `3556` assertions.

### Item 211.4 Status

Item 211.4 is complete for the selected native-parity test-body extraction.
The original proof-owner file is measurably smaller, behavior remains covered
by the same test registrations, and Day 8/9 guard work remains the next
planned step.

## Day 8: Build And Source Wiring

### Purpose

Day 8 reconciles build wiring after the selected LDLT CSC native-parity helper
extraction. The helper remains header-only, but `build/test_ldlt_csc` now has
explicit Makefile prerequisites for the LDLT CSC helper headers so incremental
Makefile builds notice helper changes.

### Changed Files

| File | Day 8 change |
| --- | --- |
| `Makefile` | Added an explicit `$(BUILDDIR)/test_ldlt_csc` rule that lists `test_ldlt_csc.c`, the four LDLT CSC helper headers, and `$(LIB)` before the generic test rule. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Added the native parity helper to `HELPERS`, required the Makefile proof-owner prerequisite rule to list each helper, and added active/order-aware selected `RUN_TEST(...)` checks that strip line and block comments. |
| `tests/test_ldlt_csc_helper_guard.py` | Added a focused regression fixture for the LDLT CSC helper guard, covering missing includes, missing Makefile prerequisites, commented registrations, reordered registrations, standalone CMake registration, and library-source registration. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Recorded Day 8 wiring evidence. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day8-build-source-wiring.md` | New Day 8 build/source wiring artifact. |

### Build Wiring

`Makefile` now owns an explicit proof-owner rule:

```make
$(BUILDDIR)/test_ldlt_csc: $(TESTDIR)/test_ldlt_csc.c $(TESTDIR)/test_ldlt_csc_fixtures.h $(TESTDIR)/test_ldlt_csc_native_parity_helpers.h $(TESTDIR)/test_ldlt_csc_oracle_helpers.h $(TESTDIR)/test_ldlt_csc_supernode_helpers.h $(LIB) | $(BUILDDIR)
```

No new test executable was added. `CMakeLists.txt` still registers only
`add_sparse_test(test_ldlt_csc)`, and
`build-metadata/library_sources.txt` still excludes the helper header.

### Guard Wiring

The LDLT CSC helper guard now enforces:

- `tests/test_ldlt_csc_native_parity_helpers.h` exists with include guard
  `TEST_LDLT_CSC_NATIVE_PARITY_HELPERS_H`;
- `tests/test_ldlt_csc.c` includes each LDLT CSC helper header exactly once;
- the Makefile `build/test_ldlt_csc` prerequisite rule lists each helper;
- helper headers are not registered as CMake tests or library sources;
- all selected native parity `RUN_TEST(...)` registrations remain active and
  in increasing order;
- line-commented or block-commented `RUN_TEST(...)` text does not satisfy the
  active-registration check.

### Regression Fixture

`tests/test_ldlt_csc_helper_guard.py` creates an isolated minimal repository
fixture and mutates it to prove the guard fails clearly for:

- missing native helper include;
- missing native helper Makefile prerequisite;
- missing active registration;
- line-commented registration;
- block-commented registration;
- reordered selected registrations;
- standalone CMake test registration;
- library-source registration.

### Validation

| Command | Result |
| --- | --- |
| `make build/test_ldlt_csc` | Passed; proof-owner build path remained reachable under the explicit rule. |
| `make ldlt-csc-helper-guard` | Passed; now reports selected `RUN_TEST` registration coverage. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions, `ALL TESTS PASSED`. |

Day 8 did not modify `.c` or `.h` files beyond the already validated Day 6/7
extraction. The latest full C gate remains the Day 7
`make format && make lint && make test` pass. Day 8 focused on Makefile,
shell, Python, and planning artifacts.

### Item 211.5 Status

Item 211.5 is complete for build/source wiring. The extracted helper is
reachable through the intended proof-owner test path, no standalone build
target or library source was introduced, and the guard now fails closed for the
main omission and commented-registration risks.

## Day 9: Ownership Guard

### Purpose

Day 9 completes the ownership guard layer for the selected LDLT CSC
native-parity extraction. Day 8 already covered helper presence, build
prerequisites, header-only registration, and active registration order. Day 9
adds the remaining ownership invariant: the moved selected test-body
definitions must actively live in
`tests/test_ldlt_csc_native_parity_helpers.h` and must not drift back into the
proof-owner file or another helper.

### Changed Files

| File | Day 9 change |
| --- | --- |
| `scripts/check_ldlt_csc_helper_guard.sh` | Added moved-definition ownership markers and active-code checks that strip line and block comments before matching. |
| `tests/test_ldlt_csc_helper_guard.py` | Extended the isolated fixture so the native helper owns the moved definitions and added negative tests for missing, commented, proof-owner, and wrong-helper definitions. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Recorded Day 9 ownership-guard evidence. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day9-ownership-guard.md` | New Day 9 ownership-guard artifact. |

### Guard Coverage Added

The guard now enforces all 13 extracted native parity definition markers:

- `test_native_1x1_diagonal_matches_wrapper`;
- `test_native_1x1_tridiagonal_matches_wrapper`;
- `test_native_1x1_mixed_indefinite_matches_wrapper`;
- `test_native_1x1_with_swap_matches_wrapper`;
- `test_native_1x1_tridiag_large_matches_wrapper`;
- `test_native_detects_near_zero_1x1_pivot`;
- `test_native_1x1_identity_matches_wrapper`;
- `test_native_2x2_forced_matches_wrapper`;
- `test_native_2x2_nonadjacent_partner_matches_wrapper`;
- `test_native_mixed_pivots_matches_wrapper`;
- `test_native_mixed_pivots_larger_matches_wrapper`;
- `test_native_2x2_solve_matches_linked_list`;
- `test_native_2x2_inertia_matches_wrapper`.

For each marker, `scripts/check_ldlt_csc_helper_guard.sh` requires exactly one
active-code occurrence in `tests/test_ldlt_csc_native_parity_helpers.h`, zero
active-code occurrences in `tests/test_ldlt_csc.c`, and zero active-code
occurrences in the other LDLT CSC helper headers.

### Regression Fixture Added

`tests/test_ldlt_csc_helper_guard.py` now covers these additional Day 9
failure modes:

- moved definition missing from the native parity helper;
- moved definition present only inside a block comment;
- moved definition duplicated back into `tests/test_ldlt_csc.c`;
- moved definition duplicated into `tests/test_ldlt_csc_oracle_helpers.h`.

### Validation

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed; now reports moved definition ownership coverage. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `./build/test_ldlt_csc` | Passed: `100` tests, `0` failed, `0` skipped, `3556` assertions, `ALL TESTS PASSED`. |

Day 9 changed shell, Python, and planning artifacts only. No additional `.c`
or `.h` edits were made on Day 9, so the sprint-required full C quality gate
was not rerun. The latest full C gate remains the Day 7
`make format && make lint && make test` pass after the extraction.

### Item 211.5 Status

Item 211.5 is complete for ownership guard coverage. The extracted
native-parity cluster now has active guard coverage for helper ownership,
proof-owner registrations, Makefile reachability, header-only boundaries, and
wrong-owner regression cases.

## Day 10: Behavior Regression Coverage

### Purpose

Day 10 adds focused behavior-regression evidence for the extracted LDLT CSC
native-parity surface. Days 8-9 guard ownership and wiring; Day 10 proves the
proof-owner executable still runs the selected native parity tests in order and
still reports the Day 5 baseline summary counts.

### Changed Files

| File | Day 10 change |
| --- | --- |
| `tests/test_ldlt_csc_native_parity_behavior.py` | New focused behavior-regression runner for the extracted native parity cluster. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Recorded Day 10 behavior evidence. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day10-behavior-regression.md` | New Day 10 behavior-regression artifact. |

### Behavior Runner

`tests/test_ldlt_csc_native_parity_behavior.py` runs:

1. `make build/test_ldlt_csc`;
2. `./build/test_ldlt_csc`;
3. an ordered check for the 13 selected native parity `[PASS]` markers;
4. a summary check against the Day 5 baseline: `100` tests, `0` failed,
   `0` skipped, `3556` assertions, and `ALL TESTS PASSED`.

### Baseline Comparison

| Surface | Day 5 baseline | Day 10 observed | Status |
| --- | ---: | ---: | --- |
| `test_ldlt_csc` tests run | 100 | 100 | Unchanged. |
| `test_ldlt_csc` failures | 0 | 0 | Unchanged. |
| `test_ldlt_csc` skipped | 0 | 0 | Unchanged. |
| `test_ldlt_csc` assertions | 3556 | 3556 | Unchanged. |
| Selected native parity pass markers | 13 | 13 | Unchanged and in order. |

No assertion-count or selected-output change was observed.

### Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Passed. |
| `make ldlt-csc-helper-guard` | Passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |

Day 10 changed Python and planning artifacts only. No additional `.c` or `.h`
edits were made on Day 10, so the sprint-required full C quality gate was not
rerun. The latest full C gate remains the Day 7
`make format && make lint && make test` pass after the extraction.

### Item 211.5 Status

Item 211.5 now has focused behavior-regression coverage beyond ownership-only
checks. The selected native parity tests remain registered, execute in the
expected order, and preserve the baseline proof-owner summary counts.

## Day 11: Documentation Calibration

### Purpose

Day 11 calibrated documentation for the selected LDLT CSC native-parity helper
extraction. The goal was to make the current ownership model explicit without
turning the review-surface reduction into a solver, API, ABI, platform,
package, performance, release, or state-of-the-art claim.

### Changed Files

| File | Day 11 change |
| --- | --- |
| `docs/maintainer_guide.md` | Added `tests/test_ldlt_csc_native_parity_helpers.h` to the LDLT CSC helper-owner guidance, documented proof-owner registration ownership, and listed focused guard/behavior validation commands. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Recorded Day 11 documentation calibration, changed-surface inventory, and validation status. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day11-documentation-calibration.md` | New Day 11 documentation-calibration artifact. |

README and INSTALL were left unchanged because this sprint work is an internal
test ownership and guard refinement, not user-facing install/support/API
guidance.

### Current Ownership Summary

| Surface | Current owner |
| --- | --- |
| LDLT CSC proof-owner executable, `main`, and selected `RUN_TEST(...)` registrations | `tests/test_ldlt_csc.c` |
| Selected native parity test bodies moved during Sprint 211 | `tests/test_ldlt_csc_native_parity_helpers.h` |
| Dense/native comparison helpers | `tests/test_ldlt_csc_oracle_helpers.h` |
| KKT and analysis fixtures | `tests/test_ldlt_csc_fixtures.h` |
| Supernode fixtures and factor-state comparison helpers | `tests/test_ldlt_csc_supernode_helpers.h` |
| Helper ownership guard | `scripts/check_ldlt_csc_helper_guard.sh` and `make ldlt-csc-helper-guard` |
| Guard regression fixture | `tests/test_ldlt_csc_helper_guard.py` |
| Focused behavior regression | `tests/test_ldlt_csc_native_parity_behavior.py` |

### Changed-Surface Inventory

| Surface | Count |
| --- | ---: |
| Documentation/planning files changed on Day 11 | 3 |
| `.c` files changed on Day 11 | 0 |
| `.h` files changed on Day 11 | 0 |
| New Day 11 artifact files | 1 |

### Line-Count Snapshot

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3174 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 303 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 576 after PR #234 guard hardening |
| `tests/test_ldlt_csc_helper_guard.py` | 812 after PR #234 guard hardening |
| `tests/test_ldlt_csc_native_parity_behavior.py` | 92 |

The proof-owner test remains reduced by 295 lines from the Day 5 baseline of
3469 lines while preserving the Day 10 behavior-regression counts.

### Non-Goal Wording

The maintainer-guide update states that this boundary is a no-behavior-change
review-surface reduction. It does not claim new LDLT CSC solver behavior,
public API or ABI changes, numerical tolerance changes, performance
improvement, platform expansion, package support, release support, or
state-of-the-art evidence.

### Validation

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed; proof-owner registrations, helper headers, header-only boundaries, selected `RUN_TEST(...)` registrations, and moved-definition ownership all passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Passed. |
| `git diff --check` | Passed. |

Day 11 changed documentation only. No additional `.c` or `.h` edits were made
on Day 11, so the sprint-required full C quality gate is not rerun for this
documentation-only calibration. The latest full C gate remains the Day 7
`make format && make lint && make test` pass after the extraction.

### Item 211.6 Status

Item 211.6 is in progress. Documentation now names the correct selected helper
owner and guard surfaces, and the Day 11 focused validation passed. Day 12
should perform integrated validation across the full changed surface.

## Day 12: Integrated Validation

### Purpose

Day 12 ran the integrated validation suite for the Sprint 211 LDLT CSC
native-parity helper extraction. Because the branch changed
`tests/test_ldlt_csc.c` and added
`tests/test_ldlt_csc_native_parity_helpers.h`, the required full C quality
chain was rerun.

### Changed Files Covered

| Surface | Files |
| --- | --- |
| Build wiring | `Makefile` |
| Maintainer documentation | `docs/maintainer_guide.md` |
| Helper ownership guard | `scripts/check_ldlt_csc_helper_guard.sh` |
| Proof-owner C test | `tests/test_ldlt_csc.c` |
| New helper/test validation files | `tests/test_ldlt_csc_native_parity_helpers.h`, `tests/test_ldlt_csc_helper_guard.py`, `tests/test_ldlt_csc_native_parity_behavior.py` |
| Sprint planning artifacts | `docs/planning/EPIC_19/SPRINT_211/PLAN.md`, `WORKING_NOTES.md`, and Day 1-12 artifacts |

### Line-Count Snapshot

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3174 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 303 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 375 |
| `tests/test_ldlt_csc_helper_guard.py` | 317 |
| `tests/test_ldlt_csc_native_parity_behavior.py` | 92 |
| `Makefile` | 1104 |
| `docs/maintainer_guide.md` | 2206 |

The proof-owner test remains reduced by 295 lines from the Day 5 baseline of
3469 lines.

### Validation

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed; proof-owner registrations, helper headers, header-only boundaries, selected `RUN_TEST(...)` registrations, and moved-definition ownership all passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Passed. |
| `make source-list-check` | Passed; reported `49` library sources. |
| `make docs-check` | Passed; Doxygen coverage checked `18` public headers and generated `18` reference pages plus `18` source pages. |
| `make format && make lint && make test` | Passed. |
| `make quality-review-cmake-compile` | Passed; CMake configured, clean rebuilt, `ctest -N` reported `60` tests, and Makefile/CMake parity matched `59` Makefile tests plus `1` focused CTest-only selector. |

### CMake and Source-List Evidence

`make quality-review-cmake-compile` compiled the changed `test_ldlt_csc`
target and confirmed that the CTest registration surface remains aligned with
the Makefile test inventory plus the focused CTest-only selector.

`make source-list-check` confirmed the production library source list remains
unchanged at `49` entries. The new helper remains test-local and header-only;
it is not a library source or standalone CMake test.

### Residual Risks

- Day 12 ran CMake compile and CTest-registration parity, not the full CMake
  `ctest` suite.
- Platform-hosted validation remains outside the local Day 12 evidence.
- Day 13 review hardening later closed the active-include guard gap before
  closeout.

### Non-Claims

The integrated validation does not claim new LDLT CSC behavior, public API or
ABI support, package support, platform support, performance improvement,
release support, or state-of-the-art evidence.

### Item 211.6 Status

Item 211.6 is substantially complete for integrated validation. The focused
guards, behavior runner, docs/source-list checks, required full C quality
chain, and CMake registration parity path all passed. Day 13 review hardening
then added active-include guard coverage.

## Day 13: Review Hardening

### Purpose

Day 13 reviewed the Sprint 211 diff as a reviewer and closed one concrete
guard-coverage gap before final closeout. The hardening stayed scoped to the
selected LDLT CSC native-parity helper extraction and its evidence.

### Finding and Fix

| Finding | Fix |
| --- | --- |
| The LDLT CSC helper guard checked helper includes with a raw fixed-string count. A commented-out `#include "test_ldlt_csc_native_parity_helpers.h"` line could satisfy helper presence while the proof-owner test stopped including the helper. | `scripts/check_ldlt_csc_helper_guard.sh` now strips line/block comments before counting helper includes and requires exactly one active include for each LDLT CSC helper header. |
| PR #234 review identified additional guard gaps for single translation-unit ownership, complete Makefile boundary enforcement, conditional-preprocessor awareness, path-qualified includes, selected-native block boundary before the Day 9 solve block, duplicated scanner predicates, zero-valued preprocessor expressions, Makefile target wiring, multiline Makefile prerequisite rules, and commented-out target commands. | The guard now rejects extra active includes of the native helper outside `tests/test_ldlt_csc.c` across repository C/header trees, compares quoted include basenames so path-qualified includes are counted, counts every literal helper basename occurrence in `Makefile`, uses one shared branch-aware `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif` AWK prelude for active-code scans, treats `#if 00`, `#if 0x0`, and `#if (0)` as inactive zero expressions, rejects ambiguous non-constant `#if`/`#elif` ownership, accepts logical Makefile prerequisite rules split with continuations, verifies `ldlt-csc-helper-guard` runs both Python suites, verifies `quality-review-compile` runs the helper guard, ignores commented-out recipe commands when validating target wiring, and requires the selected native registrations to remain before `RUN_TEST(test_solve_null_args);`. |

### Regression Coverage Added

| Test | Coverage |
| --- | --- |
| `test_line_commented_native_helper_include_fails_clearly()` | Proves a line-commented native helper include no longer satisfies the guard. |
| `test_block_commented_native_helper_include_fails_clearly()` | Proves a block-commented native helper include no longer satisfies the guard. |
| `test_native_helper_extra_makefile_registration_fails_clearly()` | Proves the helper cannot be added to an extra Makefile registration surface. |
| `test_native_helper_duplicate_same_line_makefile_registration_fails_clearly()` | Proves duplicate helper prerequisites on one Makefile line are counted as duplicate occurrences. |
| `test_native_helper_bare_makefile_registration_fails_clearly()` | Proves a bare helper basename registration is still counted as an extra Makefile occurrence. |
| `test_native_helper_second_translation_unit_include_fails_clearly()` | Proves the native helper cannot be included by a second translation unit. |
| `test_native_helper_examples_translation_unit_include_fails_clearly()` | Proves the native helper cannot be included by a second translation unit outside `tests`, `src`, or `include`. |
| `test_native_helper_path_qualified_examples_include_fails_clearly()` | Proves path-qualified helper includes outside the proof owner are counted by basename and rejected. |
| `test_if_zero_run_test_registration_fails_clearly()` | Proves inactive selected registrations under `#if 0` do not satisfy the guard. |
| `test_parenthesized_zero_native_helper_include_fails_clearly()` | Proves an inactive helper include under `#if (0)` does not satisfy include ownership. |
| `test_if_zero_else_native_helper_include_passes_guard()` | Proves a helper include in the active `#else` branch of `#if 0` is accepted. |
| `test_path_qualified_native_helper_include_passes_guard()` | Proves path-qualified helper includes in the proof owner are counted by basename. |
| `test_ifdef_native_helper_include_passes_guard()` | Proves primary `#ifdef` branches remain scannable for helper includes. |
| `test_multiline_makefile_prerequisite_rule_passes_guard()` | Proves continued Makefile prerequisites are parsed as one logical rule. |
| `test_makefile_guard_target_runs_python_suite_fails_clearly()` | Proves the Makefile helper target must invoke the guard regression suite. |
| `test_makefile_guard_target_runs_behavior_suite_fails_clearly()` | Proves the Makefile helper target must invoke the native parity behavior suite. |
| `test_quality_review_compile_runs_helper_guard_fails_clearly()` | Proves the reviewed compile-quality path must invoke the helper guard target. |
| `test_makefile_guard_target_ignores_commented_python_suite_fails_clearly()` | Proves a commented-out guard-suite command does not satisfy helper target wiring. |
| `test_quality_review_compile_ignores_commented_helper_guard_fails_clearly()` | Proves a commented-out recursive helper-guard command does not satisfy reviewed compile wiring. |
| `test_octal_zero_run_test_registration_fails_clearly()` | Proves inactive selected registrations under `#if 00` do not satisfy the guard. |
| `test_if_zero_else_run_test_registration_passes_guard()` | Proves a selected registration in the active `#else` branch of `#if 0` is accepted. |
| `test_if_zero_elif_run_test_registration_passes_guard()` | Proves a selected registration in an active `#elif` branch after `#if 0` is accepted. |
| `test_if_one_run_test_registration_passes_guard()` | Proves explicit `#if 1` selected registrations remain accepted. |
| `test_unknown_primary_if_run_test_registration_fails_closed()` | Proves ambiguous non-constant primary `#if` selected registrations fail closed. |
| `test_if_zero_unknown_elif_run_test_registration_fails_closed()` | Proves ambiguous non-constant `#elif` ownership is rejected. |
| `test_selected_registration_after_solve_block_fails_clearly()` | Proves selected native registrations must remain before the Day 9 solve block. |
| `test_if_zero_moved_definition_fails_clearly()` | Proves inactive moved definitions under `#if 0` do not satisfy helper ownership. |
| `test_hex_zero_moved_definition_fails_clearly()` | Proves inactive moved definitions under `#if 0x0` do not satisfy helper ownership. |
| `test_if_zero_else_moved_definition_passes_guard()` | Proves a moved definition in the active `#else` branch of `#if 0` is accepted. |
| `test_ifndef_moved_definition_passes_guard()` | Proves primary `#ifndef` branches remain scannable for moved definitions. |

### Changed Files

| File | Day 13 change |
| --- | --- |
| `scripts/check_ldlt_csc_helper_guard.sh` | Added active include counting, shared conditional parsing, multiline Makefile prerequisite parsing, Makefile helper-target wiring checks, and comment-aware recipe-command matching. |
| `tests/test_ldlt_csc_helper_guard.py` | Added commented-include, commented-command, Makefile wiring, multiline prerequisite, zero-expression, and unknown-conditional regression cases. |
| `docs/planning/EPIC_19/SPRINT_211/WORKING_NOTES.md` | Recorded Day 13 review-hardening evidence. |
| `docs/planning/EPIC_19/SPRINT_211/artifacts/day13-review-hardening.md` | New Day 13 review-hardening artifact. |

### Current Line-Count Snapshot

| File | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3174 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 303 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 576 |
| `tests/test_ldlt_csc_helper_guard.py` | 812 |
| `tests/test_ldlt_csc_native_parity_behavior.py` | 92 |

### Validation

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Passed. |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Passed. |
| `git diff --check` | Passed. |

Day 13 changed shell, Python, and planning artifacts only. No `.c` or `.h`
files were edited during hardening, so the required full C quality chain was
not rerun on Day 13. The latest full C quality chain remains the Day 12
`make format && make lint && make test` pass.

### Residual Risks

- Day 13 did not run the full CMake `ctest` suite; Day 12 covered CMake
  compile and `ctest -N` registration parity.
- Platform-hosted validation remains outside local sprint evidence.
- Day 14 should reconcile final sprint status, close Item 211.6, and produce
  the closeout artifact.

### Non-Claims

The hardening does not claim new LDLT CSC behavior, public API or ABI support,
package support, platform support, performance improvement, release support, or
state-of-the-art evidence.

### Item 211.6 Status

Item 211.6 remains ready for final closeout. The Day 13 review pass closed the
active-include guard gap and reran the focused guard, fixture, behavior, and
diff checks successfully.

## Day 14: Closeout Review

### Purpose

Day 14 closes Sprint 211 by reconciling the implemented LDLT CSC
native-parity helper extraction against items 211.1 through 211.6, the final
changed surface, validation evidence, non-goals, and residual risks.

### Final Item Status

| Item | Status | Evidence |
| --- | --- | --- |
| 211.1 Candidate Ranking | Complete | Day 1-2 intake and ranking selected the LDLT CSC native/parity helper cluster as the bounded large review-surface target. |
| 211.2 Cluster Boundary | Complete | Day 3 froze `tests/test_ldlt_csc.c` as proof owner and scoped extraction to selected native parity test bodies only. |
| 211.3 Extraction Design | Complete | Day 4 designed `tests/test_ldlt_csc_native_parity_helpers.h`, proof-owner registrations, Makefile dependency wiring, and guard strategy. |
| 211.4 Extraction Implementation | Complete | Days 6-7 moved selected 1x1 and 2x2/native solve parity test bodies into the new helper without changing selected test names or registration order. |
| 211.5 Guard And Test Coverage | Complete | Days 8-10 added Makefile helper prerequisites, active registration/order checks, moved-definition ownership checks, guard fixtures, and focused behavior regression. Day 13 and PR #234 follow-up hardened active include detection, root-scan ownership, path-qualified include parsing, conditional parsing, and Makefile wiring for the Python guard/behavior suites. |
| 211.6 Validation And Closeout | Complete | Day 12 ran focused checks, source-list/docs checks, `make format && make lint && make test`, and CMake registration parity. Day 14 and PR #234 follow-up reconcile final status and wire `make ldlt-csc-helper-guard` into the reviewed `quality-review-compile` path. |

### Final Review-Surface Metrics

| Surface | Baseline | Final | Delta |
| --- | ---: | ---: | ---: |
| `tests/test_ldlt_csc.c` | 3469 | 3174 | -295 |
| `tests/test_ldlt_csc_native_parity_helpers.h` | 0 | 303 | +303 |
| Selected native parity test registrations | 13 | 13 | 0 |
| `test_ldlt_csc` tests run | 100 | 100 | 0 |
| `test_ldlt_csc` assertions | 3556 | 3556 | 0 |

### Final Ownership Boundary

| Surface | Owner |
| --- | --- |
| Proof-owner executable, `main`, and selected `RUN_TEST(...)` registrations | `tests/test_ldlt_csc.c` |
| Selected Sprint 211 native parity test bodies | `tests/test_ldlt_csc_native_parity_helpers.h` |
| Dense/native comparison helpers | `tests/test_ldlt_csc_oracle_helpers.h` |
| Family-local KKT and analysis fixtures | `tests/test_ldlt_csc_fixtures.h` |
| Supernode fixtures and factor-state comparison helpers | `tests/test_ldlt_csc_supernode_helpers.h` |
| Helper ownership guard | `scripts/check_ldlt_csc_helper_guard.sh` and `make ldlt-csc-helper-guard` |
| Guard regression fixture | `tests/test_ldlt_csc_helper_guard.py` |
| Focused behavior regression | `tests/test_ldlt_csc_native_parity_behavior.py` |

### Final Changed Surface

| Surface | Files |
| --- | --- |
| Build wiring | `Makefile` |
| Maintainer documentation | `docs/maintainer_guide.md` |
| Project planning status | `docs/planning/EPIC_19/PROJECT_PLAN.md` |
| Helper ownership guard | `scripts/check_ldlt_csc_helper_guard.sh` |
| Proof-owner C test | `tests/test_ldlt_csc.c` |
| New helper/test validation files | `tests/test_ldlt_csc_native_parity_helpers.h`, `tests/test_ldlt_csc_helper_guard.py`, `tests/test_ldlt_csc_native_parity_behavior.py` |
| Sprint planning artifacts | `docs/planning/EPIC_19/SPRINT_211/PLAN.md`, `WORKING_NOTES.md`, and Day 1-14 artifacts |

### Validation Summary

| Command | Result |
| --- | --- |
| `make ldlt-csc-helper-guard` | Passed; now runs the shell guard plus `tests/test_ldlt_csc_helper_guard.py` and `tests/test_ldlt_csc_native_parity_behavior.py`. |
| `python3 tests/test_ldlt_csc_helper_guard.py` | Covered by `make ldlt-csc-helper-guard`; latest PR #234 follow-up run through that target passed. |
| `python3 tests/test_ldlt_csc_native_parity_behavior.py` | Covered by `make ldlt-csc-helper-guard`; latest PR #234 follow-up run through that target passed. |
| `make source-list-check` | Passed on Day 12; reported `49` library sources. |
| `make docs-check` | Passed on Day 12. |
| `make format && make lint && make test` | Passed on Day 12. |
| `make quality-review-cmake-compile` | Passed on Day 12; `ctest -N` reported `60` tests and Makefile/CMake parity matched `59` Makefile tests plus `1` focused CTest-only selector. |
| `git diff --check` | Passed on Day 14 after closeout documentation updates. |

### Residual Risks

- The full CMake `ctest` suite was not rerun locally; Day 12 covered CMake
  configure, clean build, `ctest -N`, and Makefile/CMake registration parity.
- Platform-hosted validation remains outside local Sprint 211 evidence.
- Broader large-surface reductions remain future work for later sprints.

### Non-Claims

Sprint 211 does not claim new LDLT CSC solver behavior, public API or ABI
support, package support, platform support, performance improvement, release
support, external-library parity, or state-of-the-art evidence.

### Closeout Decision

Sprint 211 is closed for the selected large review-surface reduction. The
retrospective, PR follow-up hardening, Make/CI guard wiring, and closeout
status are recorded on the branch.
