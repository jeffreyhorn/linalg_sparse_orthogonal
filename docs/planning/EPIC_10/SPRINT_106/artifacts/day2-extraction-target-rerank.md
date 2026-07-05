# Sprint 106 Day 2 Extraction Target Re-rank

## Purpose

Day 2 re-ranks large source and giant-test files from the live repository
before Sprint 106 starts extraction. The ranking combines file size, recent
churn, helper density, ownership ambiguity, failure-localization value, API
impact, and Make/CMake/source-list follow-through cost.

## Inputs and Commands

Current inventory commands:

```sh
find src include tests benchmarks examples scripts -type f \
  \( -name '*.c' -o -name '*.h' -o -name '*.sh' -o -name '*.py' \) \
  -print0 | xargs -0 wc -l | sort -nr | head -40

git log --since='2026-06-01' --name-only --pretty=format: -- \
  src include tests benchmarks scripts CMakeLists.txt Makefile \
  build-metadata/library_sources.txt |
  sed '/^$/d' | sort | uniq -c | sort -nr | head -40

python3 scripts/check_library_sources.py
```

Approximate helper-density command:

```sh
rg -n '^[A-Za-z_][A-Za-z0-9_ *]+\([^;]*\)[[:space:]]*\{' <file>
```

The helper-density count is not a C parser. It is sufficient for ranking
review load and fixture-extraction pressure, not for API accounting.

## Current Large-File Inventory

### Largest Proof Owners

| rank | file | lines | approximate helpers | Day 2 interpretation |
|---:|---|---:|---:|---|
| 1 | `tests/test_ldlt_csc.c` | 3,848 | 122 | largest direct CSC proof owner; strongest Day 9 fixture candidate and closest proof surface to Day 3-5 CSC source extraction |
| 2 | `tests/test_integration.c` | 3,421 | 67 | broad cross-family lifecycle owner; good Day 11 integration helper candidate, but high blast radius |
| 3 | `tests/test_qr.c` | 3,234 | 76 | large direct-family proof owner; likely fixture/helper extraction candidate if QR source is selected |
| 4 | `tests/test_ldlt.c` | 3,006 | 96 | large linked-list/backend LDLT owner; useful if Day 6 selects linked-list LDLT follow-through |
| 5 | `tests/test_etree.c` | 2,962 | not sampled | large analysis owner; defer unless Day 2 source audit finds analysis ownership overlap |
| 6 | `tests/test_graph.c` | 2,925 | 75 | graph/reorder residual owner from Sprint 105; good fixture/comment cleanup candidate |
| 7 | `tests/test_svd.c` | 2,879 | 78 | large SVD proof owner; should preserve Sprint 103 comparison wording if touched |
| 8 | `tests/test_iterative.c` | 2,841 | 87 | large iterative proof owner with history-heavy sections; good fixture-boundary candidate |
| 9 | `tests/test_chol_csc.c` | 2,554 | 109 | high churn and high helper density; defer unless CSC extraction needs Cholesky comparison structure |
| 10 | `tests/test_chol_csc_supernodal.c` | 2,504 | 73 | existing helper header reduces some pressure; still a useful pattern for future LDLT CSC extraction |
| 11 | `tests/test_reorder_nd.c` | 2,340 | 48 | Sprint 105 residual history-heavy owner; defer until graph/reorder fixture boundary day |

### Largest Source Owners

| rank | file | lines | approximate helpers | Day 2 interpretation |
|---:|---|---:|---:|---|
| 1 | `src/sparse_ldlt_csc.c` | 2,174 | 28 | largest implementation owner; explicit Sprint 106 item; top extraction target |
| 2 | `src/sparse_lu_csr.c` | 1,665 | 11 | secondary direct solver target; high value but likely more API-sensitive |
| 3 | `src/sparse_qr.c` | 1,563 | 13 | secondary direct solver target paired with giant QR tests |
| 4 | `src/sparse_eigs.c` | 1,538 | 16 | secondary orchestration target; must preserve Sprint 103 evidence wording |
| 5 | `src/sparse_ldlt.c` | 1,535 | 8 | linked-list/backend dispatch LDLT owner; candidate if CSC extraction surfaces shared LDLT helpers |
| 6 | `src/sparse_iterative.c` | 1,495 | 13 | secondary orchestration target; paired with a giant convergence proof owner |
| 7 | `src/sparse_matrix.c` | 1,359 | 43 | central matrix shell; high API and compatibility impact, so defer unless a small internal seam is obvious |
| 8 | `src/sparse_svd.c` | 1,319 | 9 | lower source helper density, but large proof owner; fixture extraction may be higher value than source split |
| 9 | `src/sparse_chol_csc.c` | 1,279 | 24 | high recent churn and direct-family adjacency; defer because Sprint 106 already has a CSC LDLT source item |
| 10 | `src/sparse_lu.c` | 1,042 | 14 | linked-list LU has Sprint 102 external reference evidence; extraction should avoid disturbing oracle lanes |

## Recent Churn Signals

Top recent churn since `2026-06-01`:

| file | observed changes | interpretation |
|---|---:|---|
| `tests/test_integration.c` | 30 | high churn and broad ownership; high value for helper extraction but risky early in sprint |
| `benchmarks/README.md` | 28 | documentation/reporting churn; relevant to claim wording, not source extraction |
| `tests/test_chol_csc.c` | 23 | high direct-family proof churn; useful comparison signal for CSC work |
| `Makefile` | 20 | build follow-through cost is real for every source or test addition |
| `src/sparse_dense.c` | 16 | backend/runtime churn from Sprint 104; not a Sprint 106 first target |
| `src/sparse_chol_csc.c` | 14 | direct CSC churn adjacent to LDLT CSC; watch for shared patterns |
| `src/sparse_analysis.c` | 14 | analysis/reorder churn; defer unless extraction touches symbolic analysis |
| `CMakeLists.txt` | 14 | CMake parity must be checked after extraction |
| `src/sparse_ldlt_csc_internal.h` | 13 | internal CSC LDLT boundary has churn; include-boundary risk for Day 3 |
| `src/sparse_ldlt_csc.c` | 13 | largest source owner also has current churn, raising extraction value |
| `tests/test_ldlt.c` | 12 | linked-list LDLT proof owner remains active |
| `src/sparse_reorder_nd.c` | 12 | Sprint 105 residual graph/reorder cleanup pressure |
| `tests/test_reorder_nd.c` | 10 | graph/reorder proof owner remains active |

The churn picture supports starting with LDLT CSC source extraction because it
combines high size, active implementation/header changes, and direct proof
coverage. It also argues against opening `tests/test_integration.c` first:
that file is large and active but has too much cross-family blast radius for
the first extraction seam.

## Source-List Starting State

The starting source-list state is clean:

```text
source-list-check: PASS (42 library sources)
```

Any new library source file must update all synchronized surfaces:

- `build-metadata/library_sources.txt`
- `Makefile`
- `CMakeLists.txt`
- `scripts/check_library_sources.py` expectations, if the checker needs a
  rule update

Test helper extraction may also require Make/CMake test target updates,
especially if a helper moves from header-only support to compiled support.

## Family Classification

| family | current owners | extraction pressure | first-pass validation cost |
|---|---|---|---|
| LDLT and CSC direct solver | `src/sparse_ldlt_csc.c`, `src/sparse_ldlt_csc_internal.h`, `src/sparse_ldlt.c`, `tests/test_ldlt_csc.c`, `tests/test_ldlt.c`, direct CSC regression tests | highest; largest source and largest proof owner both sit here | high: focused LDLT CSC tests, source-list check, full C gate |
| LU and QR direct solver | `src/sparse_lu_csr.c`, `src/sparse_lu.c`, `src/sparse_qr.c`, `tests/test_lu_csr.c`, `tests/test_sparse_lu.c`, `tests/test_qr.c`, `tests/test_colamd.c` | high; large sources and giant QR/LU tests | high: focused direct tests, source-list check, full C gate |
| eigensolver, SVD, iterative | `src/sparse_eigs.c`, `src/sparse_svd.c`, `src/sparse_iterative.c`, spectral/SVD/iterative tests | medium-high; large orchestration and proof owners with Sprint 103 comparison boundaries | high: family-focused tests and full C gate |
| graph, reorder, large-matrix | graph/reorder sources, `tests/test_graph.c`, `tests/test_reorder_nd.c`, `tests/test_reorder_amd_qg.c`, guardrail script | medium; Sprint 105 already added guardrail structure but left history-heavy owners | medium-high: graph/reorder tests, guardrail target if touched, full C gate for C changes |
| shared fixture, oracle, integration helpers | `tests/test_solver_helpers.h`, external-reference helpers, `tests/test_integration.c`, family-local helper headers | high for tests; best value is failure localization and reduced duplicate setup | high if `.c` tests change; otherwise docs/helper hygiene plus full C gate for C touches |

## Ranked Extraction Queue

### Fix-Now Candidates

| rank | candidate | reason | Day ownership | validation estimate |
|---:|---|---|---|---|
| 1 | LDLT CSC internal helper seam in `src/sparse_ldlt_csc.c` | largest source owner, recent implementation/header churn, explicit Sprint 106 item, and cohesive helper clusters already visible in comments | Days 3-5 | focused `test_ldlt_csc`/LDLT direct tests, source-list check, `make format && make lint && make test` |
| 2 | `tests/test_ldlt_csc.c` fixture/helper boundary | largest proof owner with 122 approximate helpers; closest validation surface for CSC extraction | Days 5, 9-10 | focused `test_ldlt_csc`, CTest registration check if needed, full C gate |
| 3 | secondary direct solver source seam in LU CSR or QR | `src/sparse_lu_csr.c` and `src/sparse_qr.c` remain the next largest direct-family source owners | Days 6-8 | focused LU CSR or QR tests, source-list check, full C gate |
| 4 | graph/reorder fixture or comment-heavy proof helper | Sprint 105 explicitly handed off `tests/test_graph.c` and `tests/test_reorder_nd.c`; both remain large and history-heavy | Days 9-10 | focused graph/reorder tests, guardrail target if affected, full C gate |
| 5 | integration/oracle helper extraction | `tests/test_integration.c` has the highest recent churn and broad helper pressure | Day 11 | focused integration tests, full C gate; avoid changing test registration unless planned |

### Candidate CSC Seams for Day 3

Day 3 should choose one of these, after reading the implementation in detail:

| seam | likely files | value | risk |
|---|---|---|---|
| row-adjacency allocation and append helpers | `src/sparse_ldlt_csc.c`, possibly a new internal owner | cohesive memory-management responsibility used by CSC conversion and elimination | low-medium; must preserve allocation/free invariants |
| conversion and symbolic-analysis-aware construction helpers | `src/sparse_ldlt_csc.c`, `src/sparse_ldlt_csc_internal.h` | large comment-heavy area with clear responsibility and direct proof value | medium-high; easy to disturb fill-pattern behavior |
| writeback/public LDLT payload helpers | `src/sparse_ldlt_csc.c` | contained helper cluster around public `L`, `D`, pivot, and perm publication | medium; proof surface is direct but output invariants are sensitive |
| wrapper rebuild/publish helpers | `src/sparse_ldlt_csc.c` | localized fallback/wrapper behavior and fewer public dependencies | medium; may overlap linked-list LDLT evidence |

The recommended Day 3 starting point is the row-adjacency or writeback helper
cluster because both are cohesive and less likely than conversion to alter
symbolic fill semantics. Day 3 must verify this before implementation.

## Deferred Candidates

| candidate | disposition | reason |
|---|---|---|
| `src/sparse_matrix.c` central matrix shell | defer to later API/compatibility sprint | high public API and compatibility-shell risk; not needed for Sprint 106's explicit CSC/direct extraction |
| broad `tests/test_integration.c` split | defer until Day 11 boundary | highest churn but too broad for early extraction; needs call-site intent rules first |
| SVD source extraction | defer unless Day 6 rerank selects it | `src/sparse_svd.c` is large but lower helper density; `tests/test_svd.c` fixture extraction may be higher value |
| Cholesky CSC source extraction | defer | high recent churn, but Sprint 106's CSC source item should focus on LDLT CSC unless Day 3 proves otherwise |
| graph/reorder source extraction | defer to fixture/comment cleanup unless Day 9 finds clear helper ownership | Sprint 105 just changed guardrail surfaces; avoid destabilizing reviewed lanes without a narrow seam |
| public header splitting | defer | high API/doc impact and not required for Day 3-8 implementation seams |

## First-Pass Validation Plan

| extraction type | minimum validation before closeout |
|---|---|
| LDLT CSC source split | `python3 scripts/check_library_sources.py`; focused LDLT CSC/direct tests; `make format && make lint && make test` |
| secondary source split | source-list check; focused family tests; full C gate |
| test fixture/helper split | focused affected test binaries; CTest count/registration check if target membership changes; full C gate |
| graph/reorder fixture changes | focused graph/reorder tests; `make large-matrix-guardrails` if guardrail owners are affected; full C gate |
| docs-only artifact day | `git diff --check`; trailing-whitespace scan on touched planning files |
| build-system follow-through | source-list check; focused Make/CMake configure or build check; full C gate if `.c`/`.h` changed |

## Day 2 Decision

Sprint 106 should start implementation design with LDLT CSC, specifically a
narrow helper seam inside `src/sparse_ldlt_csc.c`. The current best candidates
are row-adjacency ownership or writeback/public-payload ownership, with
conversion/symbolic-analysis construction held as a higher-risk fallback.

The secondary extraction queue should be reranked after the CSC boundary is
frozen, with LU CSR and QR currently ahead of eigensolver, iterative, and SVD
source seams because they are direct-family owners with clearer Sprint 102
handoff relevance.
