# Sprint 118 Day 9 Source And Test Hotspot Metrics

## Purpose

Day 9 captures current file-size, responsibility, and ownership metrics for
source, header, test, benchmark, example, and documentation surfaces. The goal
is evidence collection only. Day 10 will interpret these numbers into Sprint
119-123 source-boundary and proof-owner handoff guidance.

## Repository File-Count Summary

| Surface | Files |
|---|---:|
| `src` | 68 |
| `include` | 19 |
| `tests` | 89 |
| `benchmarks` | 19 |
| `examples` | 18 |
| `docs` | 2222 |
| `src include tests benchmarks examples docs` total | 2435 |

## File-Type Summary

| File type | Count |
|---|---:|
| C source files (`*.c`) | 134 |
| Headers (`*.h`) | 49 |
| Markdown files (`*.md`) | 1693 |
| Shell scripts (`*.sh`) | 2 |
| CMake files (`*.cmake` and `CMakeLists.txt`) | 1 |

## Largest Source Owners

| Rank | File | Lines | Day 10 interpretation hint |
|---:|---|---:|---|
| 1 | `src/sparse_ldlt_csc.c` | 2095 | Direct-solver CSC factor/solve owner; high split pressure. |
| 2 | `src/sparse_lu_csr.c` | 1594 | CSR LU owner; high proof-owner and helper-density pressure. |
| 3 | `src/sparse_ldlt.c` | 1535 | Mutable-shell LDLT owner; direct-solver split candidate. |
| 4 | `src/sparse_iterative.c` | 1495 | Multiple iterative methods and diagnostics; mixed-responsibility candidate. |
| 5 | `src/sparse_qr.c` | 1448 | QR, least-squares, rank, and related helper pressure. |
| 6 | `src/sparse_eigs.c` | 1412 | Eigensolver public/private boundary owner for Sprint 119. |
| 7 | `src/sparse_svd.c` | 1319 | SVD, low-rank, condition, and pseudoinverse owner. |
| 8 | `src/sparse_chol_csc.c` | 1279 | CSC Cholesky owner; direct-solver split candidate. |
| 9 | `src/sparse_matrix.c` | 1053 | Mutable shell, construction, mutation, I/O-adjacent compatibility surface. |
| 10 | `src/sparse_lu.c` | 1042 | Mutable-shell LU owner; direct-solver split candidate. |
| 11 | `src/sparse_chol_csc_internal.h` | 1017 | Large private-header owner; extraction risk needs proof. |
| 12 | `src/sparse_dense.c` | 955 | Dense helper and utility owner; review before moving. |
| 13 | `src/sparse_ldlt_csc_internal.h` | 948 | Large private-header owner; split only with compile-unit proof. |
| 14 | `src/sparse_eigs_thick_restart.c` | 915 | Eigensolver restart owner; Sprint 119 evidence target. |
| 15 | `src/sparse_graph_internal.h` | 894 | Graph private helper owner; large internal contract surface. |

## Largest Header Owners

| Rank | File | Lines | Day 10 interpretation hint |
|---:|---|---:|---|
| 1 | `include/sparse_iterative.h` | 773 | Broad iterative public API and options surface. |
| 2 | `include/sparse_eigs.h` | 651 | Eigensolver public API, options, and repeated-run surface. |
| 3 | `include/sparse_matrix.h` | 617 | Core mutable-shell API and compatibility contract. |
| 4 | `include/sparse_analysis.h` | 499 | Analysis and repeated direct lifecycle contracts. |
| 5 | `include/sparse_qr.h` | 391 | QR, least-squares, rank, and min-norm public surface. |
| 6 | `include/sparse_lu.h` | 360 | LU public surface. |
| 7 | `include/sparse_ldlt.h` | 332 | LDLT public surface. |
| 8 | `include/sparse_lu_csr.h` | 322 | CSR LU compressed-first direct public surface. |
| 9 | `include/sparse_types.h` | 316 | Cross-cutting status, type, and option definitions. |
| 10 | `include/sparse_svd.h` | 254 | SVD, pseudoinverse, and low-rank public surface. |

## Largest Test Owners

| Rank | File | Lines | Function/test proxy count | Day 10 interpretation hint |
|---:|---|---:|---:|---|
| 1 | `tests/test_ldlt_csc.c` | 3915 | 137 | Giant direct-solver proof owner; strongest split pressure. |
| 2 | `tests/test_integration.c` | 3279 | 58 | Cross-feature integration owner; high hidden-coupling risk. |
| 3 | `tests/test_qr.c` | 3234 | 89 | QR/rank/min-norm proof owner; split candidate. |
| 4 | `tests/test_ldlt.c` | 3006 | 95 | Mutable-shell LDLT proof owner; split candidate. |
| 5 | `tests/test_etree.c` | 2962 | 111 | Elimination-tree proof owner; dense helper and fixture pressure. |
| 6 | `tests/test_iterative.c` | 2924 | 94 | Multiple iterative workflows; split by solver/handle class. |
| 7 | `tests/test_svd.c` | 2823 | 93 | SVD/rank/pseudoinverse proof owner; split candidate. |
| 8 | `tests/test_graph.c` | 2764 | 68 | Graph proof owner; helper-coupling candidate. |
| 9 | `tests/test_chol_csc.c` | 2554 | 111 | CSC Cholesky proof owner; direct-solver split candidate. |
| 10 | `tests/test_chol_csc_supernodal.c` | 2504 | 74 | Supernodal proof owner; split only with fixture reuse plan. |
| 11 | `tests/test_reorder_nd.c` | 2304 | 50 | Nested-dissection proof owner; graph/reorder split candidate. |
| 12 | `tests/test_eigs.c` | 2155 | 51 | Eigensolver proof owner; Sprint 119/120 evidence target. |
| 13 | `tests/test_ilu.c` | 1974 | 36 | ILU proof owner; below giant threshold but dense. |
| 14 | `tests/test_colamd.c` | 1957 | 78 | COLAMD proof owner; many small tests. |
| 15 | `tests/test_bicgstab.c` | 1826 | 73 | BiCGSTAB proof owner; many targeted cases. |

## Benchmark And Example Size Summary

| Surface | Largest files |
|---|---|
| Benchmarks | `benchmarks/bench_eigs.c` 958 lines; `benchmarks/bench_main.c` 841; `benchmarks/bench_refactor_csc.c` 648; `benchmarks/bench_ldlt_csc.c` 516; `benchmarks/bench_chol_csc.c` 423. |
| Examples | `examples/example_eigs.c` 287 lines; `examples/example_ic_minres.c` 232; `examples/example_analysis.c` 210; `examples/example_ldlt.c` 186; `examples/example_iterative.c` 144. |
| Product docs excluding planning | `docs/algorithm.md` 1562 lines; `docs/maintainer_guide.md` 1044; `docs/tutorial.md` 515; `docs/solver_selection.md` 204; `docs/matrix_market.md` 192. |

## Mixed-Responsibility Source Candidates

| Candidate | Metric evidence | Why it is a candidate, not an immediate refactor mandate |
|---|---:|---|
| `src/sparse_eigs.c` | 1412 lines; 15 function-definition proxy matches | It owns public eigensolver orchestration and residual source-boundary work already mapped to Sprint 119. Day 10 should rank exact move candidates before any split. |
| `src/sparse_iterative.c` | 1495 lines; 13 function-definition proxy matches | It spans several iterative methods, options, diagnostics, and handle behavior. Split pressure exists, but shared solver infrastructure must remain coherent. |
| `src/sparse_qr.c` | 1448 lines; 9 function-definition proxy matches | It covers QR, least-squares, rank, minimum-norm, and refinement behavior. Movement should follow Sprint 121 oracle/proof-owner needs. |
| `src/sparse_svd.c` | 1319 lines; 9 function-definition proxy matches | It combines SVD, pseudoinverse, condition, and low-rank behavior. Any split needs SVD/QR/rank oracle proof first. |
| `src/sparse_ldlt_csc.c` | 2095 lines; 26 function-definition proxy matches | It is the largest source owner and likely has helper/factor/solve density. Direct-solver proof coverage must guide any extraction. |
| `src/sparse_lu_csr.c` | 1594 lines; 9 function-definition proxy matches | It is a compressed-first direct solver owner. Extraction must preserve CSR LU regression and package/source-list parity. |
| `src/sparse_matrix.c` | 1053 lines; 40 function-definition proxy matches | It has high public compatibility density across mutable-shell behavior. Because it is foundational, split work requires API and Matrix Market compatibility proof. |
| `src/sparse_chol_csc_internal.h` | 1017 lines | Large internal header with contract surface. Movement should be driven by compile-unit proof rather than line count alone. |
| `src/sparse_ldlt_csc_internal.h` | 948 lines | Large internal header with direct-solver helper contracts. Movement needs internal API ownership and source-list proof. |
| `src/sparse_graph_internal.h` | 894 lines | Large graph private surface. Graph/reorder split work should wait for Day 10 ranking and Sprint 123 guardrails. |

## Giant-Test Proof-Owner Candidates

| Candidate | Metric evidence | Likely split axis |
|---|---:|---|
| `tests/test_ldlt_csc.c` | 3915 lines; 137 function/test proxy matches | Factorization setup, solve behavior, update/refactor behavior, error paths, and oracle fixtures. |
| `tests/test_integration.c` | 3279 lines; 58 function/test proxy matches | Cross-family workflows, end-to-end examples, and compatibility smoke proofs. |
| `tests/test_qr.c` | 3234 lines; 89 function/test proxy matches | QR factorization, least-squares, rank, min-norm, and refinement proof owners. |
| `tests/test_ldlt.c` | 3006 lines; 95 function/test proxy matches | Mutable-shell factorization, solve, update, and failure behavior. |
| `tests/test_etree.c` | 2962 lines; 111 function/test proxy matches | Elimination tree helpers, fixtures, and edge cases. |
| `tests/test_iterative.c` | 2924 lines; 94 function/test proxy matches | CG/GMRES/MINRES-style behavior, preconditioners, diagnostics, and handle cases. |
| `tests/test_svd.c` | 2823 lines; 93 function/test proxy matches | Full/partial SVD, pseudoinverse, rank, condition, and low-rank cases. |
| `tests/test_graph.c` | 2764 lines; 68 function/test proxy matches | Graph construction, traversal, partition, and helper behavior. |
| `tests/test_chol_csc.c` | 2554 lines; 111 function/test proxy matches | CSC Cholesky factorization, solve, update, and error behavior. |
| `tests/test_chol_csc_supernodal.c` | 2504 lines; 74 function/test proxy matches | Supernodal path, fixture reuse, and numerical behavior. |
| `tests/test_reorder_nd.c` | 2304 lines; 50 function/test proxy matches | Nested dissection and graph/reorder behavior. |
| `tests/test_eigs.c` | 2155 lines; 51 function/test proxy matches | Lanczos, shift-invert, repeated handles, and eigensolver edge cases. |

## Ranked Day 10 Starting Targets

| Rank | Target | Reason |
|---:|---|---|
| 1 | `tests/test_ldlt_csc.c` | Largest proof owner and highest proxy count; direct-solver proof-density risk. |
| 2 | `src/sparse_ldlt_csc.c` | Largest source owner; direct-solver split pressure with paired giant test. |
| 3 | `tests/test_qr.c` and `tests/test_svd.c` | Large rank/QR/SVD proof owners aligned with Sprint 121. |
| 4 | `src/sparse_eigs.c`, `src/sparse_eigs_thick_restart.c`, and `tests/test_eigs.c` | Sprint 119-120 eigensolver source-boundary and oracle handoff. |
| 5 | `src/sparse_iterative.c` and `tests/test_iterative.c` | Iterative methods and repeated-handle evidence density aligned with Sprint 120. |
| 6 | `src/sparse_matrix.c` | Foundational compatibility owner with high API density; move only with strong compatibility proof. |
| 7 | `tests/test_integration.c` | Hidden-coupling risk; split only after feature-specific proof owners are stable. |
| 8 | Graph/reorder files and tests | Large enough to track, but likely Sprint 123 after solver-owner work. |

## Reproducibility Notes

The metrics were collected with these commands from the repository root:

```sh
find src include tests benchmarks examples docs -type f | wc -l

for d in src include tests benchmarks examples docs; do
  printf '%s ' "$d"
  find "$d" -type f | wc -l
done

find src include tests benchmarks examples docs -type f |
  sed 's#^#/#' |
  awk '
    /\.c$/ {c++}
    /\.h$/ {h++}
    /\.md$/ {md++}
    /\.sh$/ {sh++}
    /\.cmake$/ {cmake++}
    /CMakeLists\.txt$/ {cmake++}
    END {printf "c %d\nh %d\nmd %d\nsh %d\ncmake %d\n", c+0, h+0, md+0, sh+0, cmake+0}'

find src -type f \( -name '*.c' -o -name '*.h' \) -print0 |
  xargs -0 wc -l | sort -nr | head -25

find include -type f -name '*.h' -print0 |
  xargs -0 wc -l | sort -nr | head -20

find tests -type f -name '*.c' -print0 |
  xargs -0 wc -l | sort -nr | head -25

find benchmarks -type f -name '*.c' -print0 |
  xargs -0 wc -l | sort -nr | head -20

find examples -type f -name '*.c' -print0 |
  xargs -0 wc -l | sort -nr | head -20

find docs -path 'docs/planning' -prune -o -type f -name '*.md' -print0 |
  xargs -0 wc -l | sort -nr | head -20

for f in src/*.c; do
  printf '%5d %s\n' \
    "$(rg -n '^[A-Za-z_][A-Za-z0-9_ *]*\([^;]*\)[[:space:]]*\{' "$f" | wc -l | tr -d ' ')" \
    "$f"
done | sort -nr | head -25

for f in tests/*.c; do
  printf '%5d %s\n' \
    "$(rg -n '^static|^void test_|^int test_|^static void test_|^static int test_' "$f" | wc -l | tr -d ' ')" \
    "$f"
done | sort -nr | head -25
```

The function/test proxy counts are intentionally approximate. They are useful
for ranking proof-owner density, not for semantic API counting.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 4 has current numeric evidence. | Complete. |
| Repository file-count summary is recorded. | Complete. |
| Largest source and test owner tables are recorded. | Complete. |
| Mixed-responsibility source list is recorded. | Complete. |
| Giant-test proof-owner list is recorded. | Complete. |
| Metric commands are reproducible. | Complete. |
| Downstream source-boundary and proof-owner sprints have ranked targets for Day 10 interpretation. | Complete. |
