# Sprint 118 Day 10 Hotspot Owner Handoff

## Purpose

Day 10 interprets the Day 9 source/test hotspot metrics against the Epic 11
review and gap-closure todo. It converts line-count and proof-density evidence
into ranked owner guidance for Sprints 119-123.

This artifact is not a refactor directive. Future sprints should move code or
split tests only when there is a behavior boundary, a proof plan, and a
validation path that preserves source-list, CMake, CTest, and public-claim
truth.

## Interpretation Rules

| Rule | Day 10 interpretation |
|---|---|
| Line count is signal, not proof. | Large files receive review priority, but movement still needs behavior-boundary evidence. |
| Proof density matters more than size alone. | Tests with many assertions/helpers or multiple behavior families outrank large but coherent files. |
| Source movement must follow consumer proof. | No source split should proceed without old/new file plans, internal header contracts, focused tests, source-list updates, and CMake parity. |
| Giant-test splits must preserve failure locality. | Splits should make failures easier to diagnose, not just reduce line counts. |
| Public claims stay frozen. | Maintainability improvements do not create ecosystem-parity, portable-performance, or state-of-the-art claims. |

## High-Risk Owners

| Rank | Owner | Evidence | Risk | Primary sprint |
|---:|---|---|---|---|
| 1 | `tests/test_ldlt_csc.c` | `3915` lines; `137` function/test proxy matches | Largest proof owner, high direct-solver fixture density, likely hidden helper coupling. | Sprint 120 |
| 2 | `src/sparse_ldlt_csc.c` | `2095` lines; largest source owner | Factor/solve/update behavior and helper density in one direct-solver owner. | Sprint 120 |
| 3 | `tests/test_qr.c` | `3234` lines; `89` proxy matches | QR, least-squares, rank, min-norm, and refinement proof ownership in one file. | Sprint 121 |
| 4 | `tests/test_svd.c` | `2823` lines; `93` proxy matches | SVD, pseudoinverse, rank, condition, and low-rank proof density. | Sprint 121 |
| 5 | `src/sparse_eigs.c` | `1412` lines; Sprint 119 residual source-boundary owner | Known residual movements depend on exact consumer and rollback proof. | Sprint 119 |
| 6 | `src/sparse_iterative.c` and `tests/test_iterative.c` | `1495` source lines; `2924` test lines; `94` test proxy matches | Multiple solver families, repeated-handle behavior, preconditioners, and diagnostics. | Sprint 120 |
| 7 | `tests/test_integration.c` | `3279` lines | Cross-feature coupling risk; unsafe to split before feature-specific owners are clearer. | Sprint 120 or defer |
| 8 | `tests/test_etree.c`, `tests/test_graph.c`, `tests/test_reorder_nd.c` | `2962`, `2764`, and `2304` lines | Graph/reorder proof owners are large but should follow solver/corpus handoff and Sprint 123 guardrail work. | Sprint 123 |

## Acceptable Large-But-Coherent Owners

These owners should be tracked but not treated as first-move targets unless a
future sprint finds a clear behavior split:

| Owner | Evidence | Reason to avoid immediate movement |
|---|---:|---|
| `src/sparse_matrix.c` | `1053` lines; `40` function-definition proxy matches | It is foundational mutable-shell compatibility code. High API density makes broad movement risky without compatibility and Matrix Market proof. |
| `src/sparse_chol_csc_internal.h` | `1017` lines | Large private contract surface. Movement must follow compile-unit and internal API proof, not size alone. |
| `src/sparse_ldlt_csc_internal.h` | `948` lines | Internal helper header for direct-solver behavior. Move only with direct-solver source plan. |
| `src/sparse_graph_internal.h` | `894` lines | Large graph private contract. Track for Sprint 123, but avoid preempting solver proof-owner work. |
| `benchmarks/bench_eigs.c` | `958` lines | Large benchmark driver, but benchmark semantics should wait for Sprint 122-123 report/sentinel architecture. |
| `docs/algorithm.md` | `1562` lines | Adoption/docs density is real, but docs simplification is owned by Sprint 126, not Sprint 119-123. |

## Sprint 119 Handoff: Eigensolver Source Boundary

| Target | Handoff guidance |
|---|---|
| `src/sparse_eigs.c` | Start with a movement feasibility audit, not an extraction. Rank `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert setup/conversion, and `lanczos_iterate_op` by consumer count, public-result invariants, cleanup ownership, and rollback cost. |
| `src/sparse_eigs_thick_restart.c` | Treat thick-restart behavior as a consumer proof source for any selection/lifting movement. |
| `tests/test_eigs.c` | Use as focused behavior proof for basic Lanczos, shift-invert, repeated handle, and edge-case preservation. |
| `tests/test_eigs_thick_restart.c` and `tests/test_eigs_lobpcg.c` | Use as consumer proof for movement candidates that touch restart, grow-m, lifting, or LOBPCG-adjacent behavior. |

Required Sprint 119 proof:

- exact old/new file plan;
- internal header contract;
- source-list and CMake update plan;
- focused eigensolver consumer tests before and after movement;
- CTest registration count evidence;
- rollback instructions;
- explicit statement that no ARPACK, SciPy, or broad eigensolver parity claim
  was created.

## Sprint 120 Handoff: Direct And Iterative Owners

| Target | Handoff guidance |
|---|---|
| `tests/test_ldlt_csc.c` | First direct-solver split candidate. Split by behavior family only if fixture reuse and failure localization improve. Likely axes: factorization setup, solve behavior, update/refactor behavior, error paths, and oracle fixtures. |
| `src/sparse_ldlt_csc.c` | Highest-risk source owner. Extract only after direct-solver proof boundaries are visible from tests and helpers. |
| `tests/test_ldlt.c` | Pair with LDLT CSC work only where shared oracle/helper extraction is behavior-preserving. |
| `tests/test_iterative.c` | Split by solver/handle/progress class if the split preserves tolerance and expected-failure visibility. |
| `src/sparse_iterative.c` | Defer source movement until test proof and oracle fixture shape are known. |
| `tests/test_integration.c` | Treat as a guardrail after feature-specific splits. Do not split first unless a tiny, obvious smoke-test extraction is found. |

Required Sprint 120 proof:

- before/after responsibility map for every split;
- unchanged CTest membership unless an explicit test-count change is justified;
- focused direct/iterative reruns;
- Makefile/CMake parity;
- full `make format && make lint && make test` for `.c` or `.h` changes;
- oracle helper design that keeps solver-specific tolerances visible;
- non-claim statement preserving no SuiteSparse, PETSc, Trilinos, or every
  solver-family parity claim.

## Sprint 121 Handoff: SVD, QR, And Rank Owners

| Target | Handoff guidance |
|---|---|
| `tests/test_qr.c` | Split only along user-visible behaviors: QR factorization, least-squares, rank-deficient handling, min-norm, nullspace, and refinement. |
| `tests/test_svd.c` | Split around full/partial SVD, reconstruction, orthogonality, pseudoinverse, condition, and low-rank cases. |
| `src/sparse_qr.c` | Source movement should follow helper extraction and oracle visibility. |
| `src/sparse_svd.c` | Source movement should follow deterministic matrix taxonomy and SVD proof helper extraction. |
| `include/sparse_qr.h` and `include/sparse_svd.h` | Public API remains stable unless Sprint 121 explicitly designs and validates a contract change. |

Required Sprint 121 proof:

- fixture taxonomy covering rank, conditioning, rectangularity, scaling, and
  expected failure modes;
- reusable proof helpers that preserve reconstruction, orthogonality, rank,
  storage, leading-dimension, and tolerance assertions;
- bounded dense-reference or external-reference pilot before broader claims;
- focused QR/SVD reruns plus required full quality for code changes;
- explicit no LAPACK, NumPy, SciPy, or broad dense-linear-algebra parity claim.

## Sprint 122 Handoff: Corpus, Coverage, And Report Indexes

| Target | Handoff guidance |
|---|---|
| Corpus taxonomy | Build on Sprint 120-121 fixture decisions. Do not invent a taxonomy disconnected from active oracle owners. |
| Coverage architecture | Use Day 9 high-risk owners as risk inputs rather than pursuing vanity coverage percentages. |
| Report indexes | Prioritize recurring indexes for benchmark, coverage, dead-code, source-list, large-matrix, and oracle artifacts. |
| `docs/maintainer_guide.md` | Clarify reviewed/supplemental/local report interpretation after the index model is known. |

Required Sprint 122 proof:

- deterministic corpus tags for symmetry, definiteness, rank, conditioning,
  scaling, sparsity pattern, ordering, solver family, and expected failures;
- report classification as reviewed, supplemental, or local-only;
- stale-report handling or explicit no-stale-check rationale;
- generated or documented index contract;
- no coverage-percentage claim detached from owner risk.

## Sprint 123 Handoff: Performance, Backend, Graph, And Reorder Owners

| Target | Handoff guidance |
|---|---|
| Graph/reorder tests | Use Day 9 metrics to track `tests/test_graph.c`, `tests/test_reorder_nd.c`, `tests/test_etree.c`, and `tests/test_colamd.c`, but sequence work after corpus/report decisions. |
| Graph/reorder internals | Track `src/sparse_graph_internal.h`, `src/sparse_graph.c`, and reorder owners for sentinel/report ownership rather than broad source splits. |
| Benchmarks | Use `benchmarks/bench_eigs.c`, `benchmarks/bench_main.c`, and direct-solver benchmark drivers as report-index and sentinel candidates. |
| Backend/runtime docs | Keep OpenMP, dense backend, fallback, and local sentinel behavior bounded to local measurement truth. |

Required Sprint 123 proof:

- hot-path inventory connected to current benchmark/report surfaces;
- local sentinel design with machine/fixture interpretation;
- explicit non-portable wording;
- focused benchmark/report validation;
- no portable performance, universal reorder/fill, or vendor-backend parity
  claim.

## Source-Movement Prerequisites

No Sprint 119-123 source movement should start until the owner sprint records:

1. the behavior boundary being moved;
2. exact old and new files;
3. internal header and private API contract;
4. build-system impacts for Makefile/source-list and CMake;
5. expected CTest count before and after;
6. focused tests proving every consumer path;
7. rollback instructions;
8. public API and public-claim impact, including any explicit non-claims that
   remain unchanged.

## Giant-Test Split Prerequisites

No giant-test split should start until the owner sprint records:

1. current line count and proxy/test count;
2. before/after responsibility map;
3. fixture ownership and helper reuse plan;
4. expected CTest membership and naming;
5. focused rerun list;
6. full reviewed quality requirements if C files change;
7. failure-localization improvement;
8. residual proof-owner list for anything intentionally left unsplit.

## Defer Or No-Move Candidates

| Candidate | Day 10 disposition | Reason |
|---|---|---|
| `tests/test_integration.c` | Defer as first-move target. | It is large, but splitting it before feature-specific owners stabilize risks weakening cross-feature smoke proof. |
| `src/sparse_matrix.c` | Defer broad movement. | Foundational compatibility and mutable-shell API density make movement risky without stronger compatibility proof. |
| Internal direct-solver headers | Defer until paired direct-solver owner plan exists. | Header size alone does not identify a safe behavior boundary. |
| Graph/reorder source movement | Defer to Sprint 123. | Graph/reorder work should follow corpus/report and performance guardrail decisions. |
| Benchmark driver splits | Defer to Sprint 122-123. | Report/index semantics should be defined before changing benchmark structure. |
| Product docs split | Defer to Sprint 126. | Adoption simplification is real but outside Sprint 119-123 source/test ownership work. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 4 is complete. | Complete. |
| Ranked hotspot owner map is recorded. | Complete. |
| High-risk owners are separated from acceptable large-but-coherent owners. | Complete. |
| Eigensolver, direct/iterative, SVD, QR, corpus, report-index, graph/reorder, and performance candidates are mapped to future sprints. | Complete. |
| Source-movement prerequisites are defined. | Complete. |
| Giant-test split prerequisites are defined. | Complete. |
| No-move and defer candidates are recorded. | Complete. |
| Future sprints receive proof requirements, not broad refactor mandates. | Complete. |
