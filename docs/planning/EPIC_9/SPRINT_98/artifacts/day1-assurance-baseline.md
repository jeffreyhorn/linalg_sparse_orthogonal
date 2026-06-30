# Sprint 98 Day 1: Assurance Baseline

## Purpose

Day 1 opens Sprint 98 by making the live assurance, external comparison,
runtime/fill, coverage, documentation, and workflow-proof topology explicit.
The goal is not to widen comparison evidence yet. Day 1 defines the candidate
surfaces and validation expectations that Day 2 can audit and rank.

## Sprint 98 Scope

Sprint 98 implements the Epic 9 assurance, external comparison, and coverage
architecture phase centered on:

- comparison-surface reranking
- proof/comparison architecture design
- one maintained external correctness expansion beyond the current bounded SPD
  path
- bounded runtime/fill comparison evidence on meaningful touched workloads
- coverage-topology and proof-owner cleanup
- CI and support-surface alignment with the widened assurance model
- validation and closeout

Non-goals for Day 1:

- no test harness edits
- no benchmark edits
- no workflow edits
- no source or header edits
- no new external dependency
- no widened performance or platform claim
- no promotion of advisory ecosystem comparisons into maintained product truth
- no coverage-threshold or runtime-threshold policy change

## Live Assurance Inventory

| Surface | Current role | Day 1 reading |
|---|---|---|
| `tests/test_chol_csc.c` | maintained external SPD differential proof owner | current correctness claim-bearing lane |
| `tests/chol_external_dense_reference.py` | external-process dense reference solve helper | bounded external oracle for Cholesky CSC SPD fixtures |
| `tests/test_chol_csc_supernodal.c` | adjacent supernodal Cholesky CSC proof owner | related direct-family evidence, not the primary external dense lane |
| `tests/test_ldlt_csc.c` | largest adjacent direct CSC proof owner | likely correctness-expansion candidate, but currently internal/reference-heavy rather than external-differential maintained |
| `tests/test_iterative.c` | CG/GMRES and matrix-free proof owner | possible correctness-expansion candidate through iterative residual/reference comparison |
| `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c` | eigensolver and LOBPCG proof owners | possible correctness-expansion candidates, but tolerance and runtime cost need design |
| `tests/test_svd.c` | SVD proof owner | possible correctness-expansion candidate with higher numeric and runtime risk |
| `tests/test_qr.c` | QR proof owner | possible correctness-expansion candidate if a deterministic reference path is affordable |
| `tests/test_colamd.c`, `tests/test_reorder_nd.c`, `tests/test_reorder_amd_qg.c` | ordering/fill proof owners | likely runtime/fill comparison candidates rather than external correctness first lane |
| `benchmarks/bench_refactor_csc.c` | canonical maintained performance surface | direct-family repeated/refactor comparison owner |
| `benchmarks/bench_chol_csc.c` | canonical maintained performance surface | Cholesky CSC scalar/supernodal runtime comparison owner |
| `benchmarks/bench_iterative_reuse.c` | canonical maintained performance surface | iterative reuse runtime comparison owner |
| `benchmarks/bench_eigs_reuse.c` | canonical maintained performance surface | eigensolver reuse runtime comparison owner |
| `benchmarks/bench_fillin.c` | regression-sensitive runtime lane | fill-oriented signal, not current canonical maintained surface |
| `benchmarks/bench_reorder.c`, `benchmarks/bench_amd_qg.c` | reorder runtime/fill comparison owners | bounded branch-local runtime evidence with explicit non-superiority interpretation |
| `scripts/bench_canonical_report.sh` | threshold-free canonical report generator | artifact owner for bounded local before/after snapshots |
| `Makefile` | reviewed quality, benchmark, coverage, and install command topology | local validation and artifact-generation command owner |
| `CMakeLists.txt` | CMake test/benchmark registration and install/export owner | CMake proof registration owner, not a comparison oracle |
| `.github/workflows/ci.yml` | Linux reviewed and supplemental proof workflow | strongest CI source of truth plus supplemental runtime, TSan, dead-code, and coverage lanes |
| `.github/workflows/macos-ci.yml` | Apple Clang reviewed path plus supplemental GCC/install confidence | platform-specific proof, not parity with Linux |
| `.github/workflows/windows-ci.yml` | reviewed Windows CMake-first subset | platform-specific CMake consumer proof with staged exclusions |
| `README.md`, `INSTALL.md` | public narrative and package/support guidance | claim surfaces that must not outrun maintained evidence |
| `docs/maintainer_guide.md` | authoritative proof-owner and benchmark-governance narrative | maintainer policy owner for Sprint 98 alignment |

## Evidence-Class Split

### Correctness Evidence

Claim-bearing maintained correctness evidence currently remains narrow:

- bounded external SPD differential proof lives in `tests/test_chol_csc.c`
- external dense-reference execution lives in
  `tests/chol_external_dense_reference.py`
- fixture-backed maintained SPD coverage uses:
  - `tests/data/suitesparse/nos4.mtx`
  - `tests/data/suitesparse/bcsstk04.mtx`
- interpretation is through direct solve agreement and retained residual
  strength, not through benchmark timing or public docs wording

Adjacent correctness-heavy candidates exist, but they are not yet maintained
external-differential lanes:

- LDLT CSC factor/solve and row-adjacency reference comparisons
- iterative CG/GMRES residual and solver-family comparisons
- eigensolver and LOBPCG SuiteSparse fixtures
- SVD/QR numeric reconstruction and cross-path checks
- COLAMD/ND/AMD ordering and fill tests

Day 2 should rank these by external-reference availability, deterministic
reproducibility, maintenance cost, and CI suitability.

### Runtime/Fill Evidence

Runtime and fill evidence remains bounded and calibration-oriented:

- canonical maintained performance surface:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- threshold-free canonical artifact command:
  - `make bench-canonical-report`
- bounded reorder/runtime slice:
  - `make bench-reorder-sprint86`
- PR-time supplemental runtime signal:
  - `make bench-fast`
- fill and reorder-adjacent candidates:
  - `bench_fillin`
  - `bench_reorder --skip-factor`
  - `bench_amd_qg`
  - fill assertions inside ordering and direct-factor tests

This class may support local before/after calibration and meaningful touched
workload discussion. It should not be interpreted as broad speed leadership,
portable timing parity, or a pass/fail product guarantee.

### Coverage and Proof-Owner Topology

Coverage and topology signals are fragmented across:

- large proof owners:
  - `tests/test_ldlt_csc.c`
  - `tests/test_integration.c`
  - `tests/test_qr.c`
  - `tests/test_ldlt.c`
  - `tests/test_etree.c`
  - `tests/test_graph.c`
  - `tests/test_iterative.c`
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_chol_csc_supernodal.c`
- tree-mutating local coverage targets:
  - `make coverage`
  - `make coverage-lcov`
  - `make coverage-gcovr`
- Linux supplemental coverage workflow:
  - `.github/workflows/ci.yml`

Sprint 96 already reduced one Cholesky CSC proof-owner concentration by
splitting supernodal/writeback proof from core Cholesky CSC proof. Sprint 98
should avoid reopening broad proof-owner refactors unless the cleanup directly
supports external comparison or coverage-topology clarity.

### Documentation Claim Surfaces

Public and maintainer claim surfaces are layered:

- `README.md` owns compact public adoption and capability summaries
- `INSTALL.md` owns install/package guidance
- `benchmarks/README.md` owns benchmark-local CLI and schema explanation
- `docs/maintainer_guide.md` owns proof-owner, benchmark-governance, and
  support-surface interpretation
- `docs/algorithm.md` contains deeper algorithm and historical measurement
  context

Day 1 reading:

- public claims must stay behind maintained correctness/package evidence
- benchmark docs should stay threshold-free and calibration-oriented
- maintainer-only proof-owner detail should not leak into front-door claims
- old measurement context should not be promoted into current product truth

### Workflow Ownership

Workflow evidence remains intentionally asymmetric:

- Linux:
  - strongest reviewed source of truth
  - reviewed Make compile-quality path
  - reviewed CMake parity path
  - dead-code, supplemental runtime, TSan, and coverage lanes
- macOS:
  - Apple Clang reviewed path
  - supplemental Homebrew GCC direct build/test path
  - supplemental static-first Make install/pkg-config confidence path
- Windows:
  - reviewed CMake-first consumer subset
  - expected CTest count remains a proof assertion
  - staged exclusions remain explicit for pthread/fuzz lanes
  - no Makefile parity or reviewed install-validation lane

Sprint 98 CI/support alignment should preserve these fences unless later days
earn stronger evidence.

## Initial Day 2 Candidate Queue

| Candidate | Evidence class | Initial classification |
|---|---|---|
| LDLT CSC external correctness expansion | correctness | high-value candidate because it is adjacent to Cholesky CSC and already has dense/reference-heavy proof, but external oracle and tolerance design are required |
| Iterative solver external correctness comparison | correctness | high-value user-facing candidate, but runtime, convergence variability, and reference semantics need careful ranking |
| Eigensolver/LOBPCG external correctness comparison | correctness | high-value capability candidate with higher tolerance/runtime risk |
| QR or SVD external correctness comparison | correctness | potentially valuable but likely more expensive and numerically sensitive |
| Reorder/fill runtime comparison | runtime/fill | strong candidate because existing benchmark/test surfaces already expose fill-oriented evidence |
| Canonical benchmark report extension | runtime/fill | possible only if it remains threshold-free and cheap |
| Coverage/proof-owner naming cleanup | topology | useful if it clarifies external comparison ownership without starting a broad refactor |
| CI support-surface wording alignment | workflow/docs | likely needed after any widened assurance lane lands |

## Validation Expectations

Use this validation split during Sprint 98:

| Change type | Minimum validation expectation |
|---|---|
| Planning/docs-only artifacts | `git diff --check` and trailing-whitespace scan on touched Sprint 98 planning files |
| `.c` or `.h` changes | `make format && make lint && make test` |
| Correctness-test or fixture changes | focused touched test binary plus full source validation when C/header files are touched |
| External-reference helper changes | focused helper/test command plus docs hygiene; full source validation only if C/header files change |
| Benchmark C changes | relevant benchmark build/run command plus full source validation |
| Benchmark script/report changes | `make bench-canonical-report` or focused script command plus docs hygiene |
| Make/CMake registration changes | reviewed Make/CMake parity command and source validation when code registration changes |
| Workflow-only changes | docs hygiene plus local equivalent commands where possible; CI remains final proof for platform syntax |
| Coverage target changes | coverage command or focused dry-run equivalent plus `make clean` before returning to normal reviewed paths |

## Day 1 Result

Sprint 98 starts from a current assurance baseline. The maintained correctness
lane is still Cholesky CSC external SPD differential proof. Runtime/fill
evidence remains bounded and threshold-free. Coverage and proof-owner topology
are visible enough for Day 10 cleanup, but Day 2 should first rank external
correctness and runtime/fill candidates from evidence rather than broadening
claims opportunistically.
