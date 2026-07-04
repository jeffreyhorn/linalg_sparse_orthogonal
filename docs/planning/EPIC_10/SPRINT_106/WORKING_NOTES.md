# Sprint 106 Working Notes

## Sprint Context

Sprint 106 implements "Large-Source & Giant-Test Maintainability Phase 7" from
`docs/planning/EPIC_10/PROJECT_PLAN.md`. The sprint reduces maintenance risk in
large implementation and proof owners by extracting narrow, validated source
and test seams rather than rewriting solver families.

The sprint starts from the Sprint 100 hotspot baseline and the Sprint 102-105
handoffs. The highest-pressure owners are direct CSC solver implementation,
secondary LU/QR/eigensolver/iterative hotspots, and giant tests whose reusable
fixtures obscure failure localization. Every extraction must preserve source
membership parity across the manifest, Makefile, CMake, and source-list
checker.

## Validation Rules

Validation must scale with the touched surface:

| touched surface | required validation |
|---|---|
| planning documentation only | `git diff --check`; trailing-whitespace scan on touched planning files |
| public documentation only | `git diff --check`; trailing-whitespace scan on touched docs |
| source extraction in `.c` files | focused affected tests; source-list check; `make format`; `make lint`; `make test` |
| public or internal `.h` changes | focused affected tests; source-list check when library ownership changes; `make format`; `make lint`; `make test` |
| test helper extraction | focused affected test binaries; CTest registration check when registration changes; `make format`; `make lint`; `make test` |
| build or CMake surface | source-list checker; focused Make/CMake configure or build check; any code-touch gate |
| scripts or validation helpers | syntax check or focused helper invocation; docs hygiene |
| mixed source/test/build extraction | all applicable focused checks plus the full C quality gate |

If any `.c` or `.h` file is modified, the full required quality chain is:

```sh
make format && make lint && make test
```

All required checks must pass before closeout or PR creation.

## Extraction Boundaries

Sprint 106 may extract ownership seams that reduce file size, clarify helper
responsibility, or improve failure localization. It must not turn extraction
into broad solver redesign.

Sprint 106 must preserve:

- existing public API behavior unless a specific compatibility note and test
  plan exists;
- Make/CMake/source-list parity for every library source move;
- reviewed CTest shape unless a test registration change is explicitly
  planned and counted;
- fixture intent at call sites after helper extraction;
- claim boundaries from Sprint 100 through Sprint 105 around oracle,
  benchmark, runtime, graph, and reorder evidence.

Sprint 106 must not claim:

- state-of-the-art solver superiority from maintainability work;
- broad ecosystem parity from fixture extraction;
- portable performance changes from source reorganization;
- completed cleanup of all large owners unless metrics prove it;
- Windows, CMake, or install parity beyond the reviewed scope actually
  checked.

## Day 1 - Scope and Maintainability Baseline

### Goal

Convert the Sprint 106 project-plan section and prior Epic 10 handoffs into a
bounded extraction package with clear workstreams, validation expectations,
handoff risks, and a Day 2 audit starting point.

### Actions

- Re-read the Sprint 106 section of
  `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read the Sprint 106 day-by-day plan in
  `docs/planning/EPIC_10/SPRINT_106/PLAN.md`.
- Re-read Sprint 100 hotspot and source-list baseline:
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day4-source-test-maintainability-metrics.md`
- Re-read direct-solver, comparison, runtime, and reorder/graph handoffs:
  - `docs/planning/EPIC_10/SPRINT_102/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_10/SPRINT_103/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_10/SPRINT_104/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_10/SPRINT_105/artifacts/day14-closeout-and-handoff.md`
- Created the Sprint 106 artifacts directory.
- Recorded authoritative Day 1 inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Sprint 106 maintainability baseline, workstream ownership,
  validation matrix, and handoff constraints in
  `artifacts/day1-maintainability-baseline.md`.

### Findings

- Sprint 100 identified `src/sparse_ldlt_csc.c` as the largest implementation
  owner and `tests/test_ldlt_csc.c` as the largest proof owner.
- The largest remaining proof owners span direct, integration, graph, iterative,
  SVD, Cholesky CSC, supernodal, and nested-dissection surfaces.
- Source-list drift is a first-class risk because library source membership is
  mirrored across `build-metadata/library_sources.txt`, `Makefile`,
  `CMakeLists.txt`, and `scripts/check_library_sources.py`.
- Sprint 102 already extracted shared direct-solver oracle helper structure,
  so Sprint 106 should avoid duplicating that work and instead focus on the
  next high-value direct-family ownership seams.
- Sprint 103 and Sprint 104 reinforce that comparison, runtime, and benchmark
  language must remain evidence-bounded even when source ownership improves.
- Sprint 105 hands off graph/reorder history-heavy cleanup and exact
  Make/CMake reviewed-surface discipline as constraints for fixture and source
  extraction.

### Validation Expectations

- Day 1 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_106`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_106`: passed; no
  matches.

### Day 1 Exit State

Day 1 is complete. Sprint 106 now has working notes, authoritative inputs, a
maintainability baseline, workstream ownership, validation expectations, and a
Day 2 target re-rank starting point.

## Day 2 - Extraction Target Re-rank

### Goal

Re-rank large source and giant-test files by current size, recent churn,
ownership ambiguity, helper density, failure-localization value, API impact,
and Make/CMake/source-list follow-through cost.

### Actions

- Re-read the Day 2 plan section in
  `docs/planning/EPIC_10/SPRINT_106/PLAN.md`.
- Generated current line-count inventory for source, header, test, benchmark,
  example, and script files.
- Generated recent churn counts from git history since `2026-06-01`.
- Generated approximate function/helper counts for the largest source and test
  owners.
- Ran the source-list checker to confirm the starting build membership state.
- Scanned representative chronology, compatibility, fallback, and future-work
  comments across `src`, `include`, and `tests`.
- Wrote the Day 2 extraction target re-rank artifact in
  `artifacts/day2-extraction-target-rerank.md`.

### Findings

- `tests/test_ldlt_csc.c` remains the largest proof owner at 3,848 lines and
  has the highest helper density among the inspected test files.
- `src/sparse_ldlt_csc.c` remains the largest implementation owner at 2,174
  lines and has recent churn in both the implementation and internal header.
- The highest-value CSC extraction candidate is the LDLT CSC row-adjacency,
  conversion/writeback, or wrapper-rebuild helper cluster, with Day 3 owning
  the exact seam decision before edits.
- Secondary source candidates should be ranked after the CSC boundary is
  frozen; the strongest starting set is LU CSR, QR, eigensolver orchestration,
  iterative orchestration, and linked-list LDLT.
- Giant-test extraction pressure is broad. Direct CSC, QR, graph/reorder,
  iterative, SVD, and integration owners all have enough helper density to
  justify fixture-boundary review before moving code.
- Build-system follow-through remains a first-class cost because the source
  list checker currently passes and must stay exact after extraction.

### Validation Expectations

- Day 2 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_106`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_106`: passed; no
  matches.

### Day 2 Exit State

Day 2 is complete. Sprint 106 now has a live ranked extraction queue, a
selected CSC direct-family starting candidate for Day 3, deferred candidate
rationale, and first-pass validation cost estimates.

## Day 3 - LDLT/CSC Extraction Boundary

### Goal

Freeze the first LDLT CSC extraction seam before implementation, including
ownership responsibilities, include boundaries, Make/CMake/source-list updates,
test impact, and focused validation commands.

### Actions

- Re-read the Day 3 plan section in
  `docs/planning/EPIC_10/SPRINT_106/PLAN.md`.
- Inspected `src/sparse_ldlt_csc.c` around lifecycle, conversion, writeback,
  row-adjacency, native elimination, and supernodal call sites.
- Inspected `src/sparse_ldlt_csc_internal.h` for the private LDLT CSC contract
  already visible to `sparse_ldlt_csc.c`, `sparse_ldlt_csc_supernodal.c`, and
  `sparse_ldlt_dense.c`.
- Inspected direct CSC test references in `tests/test_ldlt_csc.c` and
  `tests/test_direct_csc_regression.c`.
- Inspected source-list/build membership references in
  `build-metadata/library_sources.txt`, `Makefile`, and `CMakeLists.txt`.
- Wrote the Day 3 boundary artifact in
  `artifacts/day3-ldlt-csc-extraction-boundary.md`.

### Findings

- The Day 4 extraction should target row-adjacency lifecycle and population,
  not conversion or writeback.
- The row-adjacency seam is cohesive because it owns allocation/free support,
  append growth, symmetric-swap slot exchange, and per-column population used
  by scalar and batched elimination.
- The conversion and writeback seams remain valuable but carry higher semantic
  risk because they touch symbolic fill, public `sparse_ldlt_t` materialization,
  and analysis-aware indefinite behavior.
- The new owner should be private implementation only:
  `src/sparse_ldlt_csc_rowadj.c`, with declarations kept in
  `src/sparse_ldlt_csc_internal.h`.
- No public API change is planned.
- Build follow-through must update `build-metadata/library_sources.txt`,
  `Makefile`, and `CMakeLists.txt`, then rerun
  `python3 scripts/check_library_sources.py`.

### Validation Expectations

- Day 3 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_106`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_106`: passed; no
  matches.

### Day 3 Exit State

Day 3 is complete. Sprint 106 now has the row-adjacency extraction boundary
frozen for Day 4 and all build/test follow-through points known before source
edits begin.

## Day 4 - LDLT/CSC Extraction Batch

### Goal

Extract the selected LDLT CSC row-adjacency helper owner, preserve direct
solver behavior, and keep Make/CMake/source-list parity exact.

### Actions

- Added `src/sparse_ldlt_csc_rowadj.c` as the private row-adjacency owner.
- Moved `ldlt_csc_row_adj_append(...)` out of `src/sparse_ldlt_csc.c`.
- Moved row-adjacency population into `ldlt_csc_populate_row_adj(...)` in the
  new source owner.
- Added `ldlt_csc_row_adj_swap_slots(...)` and replaced the inline
  row-adj-specific swap block in `ldlt_csc_symmetric_swap(...)`.
- Added internal declarations to `src/sparse_ldlt_csc_internal.h`.
- Updated synchronized library source membership in:
  - `build-metadata/library_sources.txt`
  - `Makefile`
  - `CMakeLists.txt`
- Ran source-list, focused direct solver, full Make quality, and CMake compile
  parity validation.
- Wrote the Day 4 implementation artifact in
  `artifacts/day4-ldlt-csc-extraction-batch.md`.

### Findings

- The extraction stayed private: no public headers, public APIs, test
  registrations, dispatch policy, conversion behavior, or writeback behavior
  changed.
- `src/sparse_ldlt_csc.c` now delegates row-adjacency slot swapping and
  per-column row-adj population through the internal private contract.
- Source-list parity now reports 43 library sources.
- CMake registration remained at 54 tests, matching the Makefile test count.

### Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `src/sparse_ldlt_csc.c` | 2,174 lines | 2,092 lines | -82 |
| `src/sparse_ldlt_csc_rowadj.c` | 0 lines | 82 lines | +82 |
| `src/sparse_ldlt_csc_internal.h` | 929 lines | 947 lines | +18 |

### Validation Results

- `python3 scripts/check_library_sources.py`: passed;
  `source-list-check: PASS (43 library sources)`.
- Focused direct solver validation passed:
  - `make build/test_ldlt_csc build/test_direct_csc_regression build/test_ldlt build/test_ldlt_backend_dispatch`
  - `./build/test_ldlt_csc`
  - `./build/test_direct_csc_regression`
  - `./build/test_ldlt`
  - `./build/test_ldlt_backend_dispatch`
- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Reviewed CMake compile/parity path passed:
  - `make quality-review-cmake-compile`
  - CMake tests: 54
  - Makefile tests: 54
  - test-count parity passed
- Final hygiene checks passed:
  - `git diff --check`
  - trailing-whitespace scan across touched source, build-list, and Sprint 106
    planning files
  - final source-list recheck

### Day 4 Exit State

Day 4 is complete. Sprint 106 now has an extracted LDLT CSC row-adjacency
private source owner, synchronized Make/CMake/source-list membership, focused
direct solver validation, the required full C quality gate, and CMake compile
parity validation.

## Day 5 - LDLT/CSC Test and Oracle Follow-Through

### Goal

Tighten direct proof and documentation ownership around the Day 4 LDLT CSC
row-adjacency extraction, adding focused helper-boundary coverage where it
improves failure localization without broad fixture churn.

### Actions

- Reviewed `tests/test_ldlt_csc.c` and `tests/test_direct_csc_regression.c`
  against the extracted row-adjacency helper boundary.
- Added `test_row_adj_swap_slots_moves_whole_row_state` to
  `tests/test_ldlt_csc.c` so the extracted slot-swap helper now has direct
  coverage in addition to indirect native-elimination and symmetric-swap
  coverage.
- Left existing local test helpers in place because the current direct CSC
  test fixture layout already localizes the row-adjacency proof clearly.
- Updated source ownership comments in `src/sparse_ldlt_csc.c` and
  `src/sparse_ldlt_csc_internal.h`.
- Updated direct solver maintainer documentation in `docs/maintainer_guide.md`.
- Updated the row-adjacency algorithm note in `docs/algorithm.md`.
- Wrote the Day 5 proof follow-through artifact in
  `artifacts/day5-ldlt-csc-proof-follow-through.md`.

### Findings

- `ldlt_csc_row_adj_append(...)` already had direct append, growth, and
  argument-check coverage.
- `ldlt_csc_populate_row_adj(...)` remains covered through
  `test_row_adj_matches_reference` and the SuiteSparse structural regression
  path in `tests/test_direct_csc_regression.c`.
- `ldlt_csc_row_adj_swap_slots(...)` had indirect coverage through
  `ldlt_csc_symmetric_swap(...)` and native factorization paths, but lacked a
  direct proof that pointer, count, and capacity ownership move together.
- The stale ownership comments and maintainer notes could otherwise point
  future maintainers back to `src/sparse_ldlt_csc.c` for row-adjacency support.

### Validation Results

- `python3 scripts/check_library_sources.py`: passed;
  `source-list-check: PASS (43 library sources)`.
- Focused LDLT CSC validation passed:
  - `make build/test_ldlt_csc && ./build/test_ldlt_csc`
  - `Tests run: 100`
  - `Tests failed: 0`
  - `Tests skipped: 0`
  - `Assertions: 2335`
- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Final hygiene checks passed:
  - `git diff --check`
  - trailing-whitespace scan across touched source, documentation, and Sprint
    106 planning files
  - final source-list recheck

### Day 5 Exit State

Day 5 is complete. Sprint 106 now has direct regression coverage for the
extracted LDLT CSC row-adjacency slot-swap helper, refreshed ownership notes
for the extracted source owner, and passing focused, full, and hygiene
validation.

## Day 6 - LU/QR/Eigs/Iterative Extraction Boundary

### Goal

Select the next secondary source extraction seam after the LDLT CSC extraction
and freeze the ownership, dependency, build, and validation plan before
editing source files.

### Actions

- Re-read the Day 6 plan section and Day 2 extraction target re-rank.
- Rechecked current source and proof-owner sizes for LU CSR, QR, eigensolver,
  iterative, SVD, and linked-list LDLT candidates.
- Inspected representative helper boundaries in:
  - `src/sparse_qr.c`
  - `src/sparse_lu_csr.c`
  - `src/sparse_eigs.c`
  - `src/sparse_iterative.c`
- Checked current QR references in public headers, tests, Makefile, CMake, and
  source-list metadata.
- Selected a QR private helper extraction for Day 7.
- Wrote the Day 6 secondary extraction boundary artifact in
  `artifacts/day6-secondary-extraction-boundary.md`.

### Findings

- `src/sparse_lu_csr.c` remains the largest secondary source owner, but its
  most obvious helpers are tightly coupled to CSR elimination and dense LU
  ownership.
- `src/sparse_qr.c` is the next-best direct-family owner and has a cohesive
  low-API-risk helper group at the top of the file:
  - monotonic QR progress timer
  - Householder compute/apply kernels
  - sparse column extraction for column-wise QR
  - column-sliced Householder application for sparse-mode QR
- The QR helper group is used by dense QR factorization, sparse-mode QR
  factorization, and Q application, but does not require any public API change.
- Eigensolver and iterative candidates remain valuable, but their visible
  helper seams are more orchestration-heavy and closer to public handle,
  workspace, backend-selection, or comparison-evidence behavior.
- Day 7 should create `src/sparse_qr_internal.h` and
  `src/sparse_qr_householder.c`, then update `src/sparse_qr.c`,
  `build-metadata/library_sources.txt`, `Makefile`, and `CMakeLists.txt`.

### Selected Seam

The Day 7 extraction target is the QR Householder and column utility seam:

| responsibility | planned owner |
|---|---|
| `s29_qr_now_s(...)` | `src/sparse_qr_householder.c` |
| `sparse_qr_householder_compute(...)` | `src/sparse_qr_householder.c` |
| `sparse_qr_householder_apply(...)` | `src/sparse_qr_householder.c` |
| `sparse_qr_extract_column(...)` | `src/sparse_qr_householder.c` |
| `sparse_qr_householder_apply_to_column(...)` | `src/sparse_qr_householder.c` |
| private declarations | `src/sparse_qr_internal.h` |
| public QR API and factor/solve orchestration | `src/sparse_qr.c` |

### Validation Expectations

Day 6 changes planning documentation only.

Required checks:

- `git diff --check`
- trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_106`

Planned Day 7 source-touch checks:

- `python3 scripts/check_library_sources.py`
- focused QR validation:
  - `make build/test_qr build/test_colamd build/test_sprint6_integration`
  - `./build/test_qr`
  - `./build/test_colamd`
  - `./build/test_sprint6_integration`
- required full C quality gate:
  - `make format && make lint && make test`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_106`: passed; no
  matches.

### Day 6 Exit State

Day 6 is complete. Sprint 106 now has a frozen secondary source extraction
boundary for QR Householder/column utilities, explicit build-system
follow-through, focused validation commands, and a residual queue for LU CSR,
eigensolver, iterative, SVD, and linked-list LDLT candidates.

## Day 7 - Secondary Source Extraction Batch 1

### Goal

Extract the selected QR Householder and sparse-column helper seam into a
private source owner while preserving QR public API behavior, Make/CMake
source membership, and focused QR proof coverage.

### Actions

- Added `src/sparse_qr_internal.h` as the private QR internal contract.
- Added `src/sparse_qr_householder.c` as the private QR Householder and
  sparse-column helper owner.
- Moved QR-local helper ownership out of `src/sparse_qr.c`:
  - `s29_qr_now_s(...)`
  - Householder vector computation
  - Householder vector application
  - sparse column extraction for sparse-mode QR
  - column-sliced Householder application for sparse-mode QR
- Renamed moved helpers into the private QR namespace:
  - `sparse_qr_householder_compute(...)`
  - `sparse_qr_householder_apply(...)`
  - `sparse_qr_extract_column(...)`
  - `sparse_qr_householder_apply_to_column(...)`
- Updated synchronized library source membership in:
  - `build-metadata/library_sources.txt`
  - `Makefile`
  - `CMakeLists.txt`
- Ran source-list, focused QR-family, full C quality, CMake compile/parity, and
  final hygiene validation.
- Wrote the Day 7 implementation artifact in
  `artifacts/day7-qr-householder-extraction.md`.

### Findings

- The extraction stayed private: no public QR API, public header, test
  registration, QR option, reorder behavior, solve behavior, or factorization
  algorithm changed.
- `src/sparse_qr.c` now owns QR factor/solve orchestration while
  `src/sparse_qr_householder.c` owns reusable Householder and sparse-column
  mechanics.
- Source-list parity now reports 44 library sources.
- CMake registration remained at 54 tests, matching the Makefile test count.

### Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `src/sparse_qr.c` | 1,563 lines | 1,448 lines | -115 |
| `src/sparse_qr_householder.c` | 0 lines | 79 lines | +79 |
| `src/sparse_qr_internal.h` | 0 lines | 16 lines | +16 |

### Validation Results

- `python3 scripts/check_library_sources.py`: passed;
  `source-list-check: PASS (44 library sources)`.
- Focused QR-family validation passed:
  - `make build/test_qr build/test_colamd build/test_sprint6_integration`
  - `./build/test_qr`
  - `./build/test_colamd`
  - `./build/test_sprint6_integration`
- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Reviewed CMake compile/parity path passed:
  - `make quality-review-cmake-compile`
  - CMake tests: 54
  - Makefile tests: 54
  - test-count parity passed
- Final hygiene checks passed:
  - `git diff --check`
  - trailing-whitespace scan across touched source, build-list, and Sprint 106
    planning files
  - final source-list recheck

### Day 7 Exit State

Day 7 is complete. Sprint 106 now has a private QR Householder helper owner,
synchronized Make/CMake/source-list membership, focused QR-family validation,
the required full C quality gate, and CMake compile parity validation.

## Day 8 - Secondary Source Extraction Batch 2

### Goal

Complete the second selected secondary source extraction by moving LU CSR
structural helpers into a private implementation owner while preserving CSR LU
factorization behavior, source-list parity, and focused proof coverage.

### Actions

- Re-checked the Day 7 QR Householder extraction against the Day 6 boundary and
  confirmed it was complete enough to move to the next selected seam.
- Selected the LU CSR structural helper seam from the deferred queue because
  `lu_csr_grow(...)` and `lu_csr_validate(...)` are shared by the CSR factor
  build, plain elimination, block elimination, and dense-block detection paths.
- Added `src/sparse_lu_csr_internal.h` as the private LU CSR internal contract.
- Added `src/sparse_lu_csr_struct.c` as the private owner for LU CSR growth and
  structural validation helpers.
- Removed the helper definitions from `src/sparse_lu_csr.c` and included the
  private internal header there.
- Updated synchronized library source membership in:
  - `build-metadata/library_sources.txt`
  - `Makefile`
  - `CMakeLists.txt`
- Ran source-list validation, focused LU CSR family validation, the required
  full C quality gate, reviewed CMake compile/parity validation, and final
  hygiene checks.
- Wrote the Day 8 implementation artifact in
  `artifacts/day8-lu-csr-structural-extraction.md`.

### Findings

- The extraction stayed private: no public LU CSR API, public header, test
  registration, factorization option, solve behavior, or numerical algorithm
  changed.
- `src/sparse_lu_csr.c` now owns factorization and solve orchestration while
  `src/sparse_lu_csr_struct.c` owns reusable CSR structural mechanics.
- Dense LU helper routines stayed in `src/sparse_lu_csr.c`; they remain a
  separate potential seam because they are part of the block-elimination path,
  not the shared CSR structure lifecycle.
- Source-list parity now reports 45 library sources.
- CMake registration remained at 54 tests, matching the Makefile test count.

### Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `src/sparse_lu_csr.c` | 1,665 lines | 1,594 lines | -71 |
| `src/sparse_lu_csr_struct.c` | 0 lines | 57 lines | +57 |
| `src/sparse_lu_csr_internal.h` | 0 lines | 9 lines | +9 |

### Validation Results

- `python3 scripts/check_library_sources.py`: passed;
  `source-list-check: PASS (45 library sources)`.
- Focused LU CSR-family validation passed:
  - `make build/test_lu_csr build/test_sprint10_integration`
  - `./build/test_lu_csr`
  - `./build/test_sprint10_integration`
- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Reviewed CMake compile/parity path passed:
  - `make quality-review-cmake-compile`
  - CMake tests: 54
  - Makefile tests: 54
  - test-count parity passed
- Final hygiene checks passed:
  - `git diff --check`
  - trailing-whitespace scan across touched source, build-list, and Sprint 106
    planning files
  - final source-list recheck

### Day 8 Exit State

Day 8 is complete. Sprint 106 now has private LU CSR structural helper ownership,
synchronized Make/CMake/source-list membership, focused LU CSR-family
validation, the required full C quality gate, and reviewed CMake compile parity
validation. Remaining maintainability seams include dense LU helper ownership,
the larger LU CSR elimination body, eigensolver/iterative/SVD seams, and the
Day 9 giant-test fixture boundary.

## Day 9 - Giant-Test Fixture Boundary

### Goal

Inventory the largest test owners and define fixture/helper extraction targets
for Day 10 without changing test behavior, registration, or reviewed CTest
shape.

### Actions

- Re-read the Day 9 plan section in
  `docs/planning/EPIC_10/SPRINT_106/PLAN.md`.
- Reused the Day 2 giant-test ranking and regenerated current line-count and
  helper-density samples for the largest direct, graph/reorder, integration,
  oracle, and generated-fixture owners.
- Inspected existing test helper-header conventions:
  - `tests/test_solver_helpers.h`
  - `tests/test_iterative_handle_helpers.h`
  - `tests/test_svd_partial_helpers.h`
  - `tests/test_chol_csc_supernodal_helpers.h`
- Classified candidate helper seams into behavior-preserving fixture builders,
  assertion/oracle helpers, SuiteSparse fixture cache helpers, and residual
  utilities.
- Defined naming and ownership rules for new helper headers.
- Mapped Make, CMake, include, and test-registration follow-through for both
  header-only and compiled-helper extraction styles.
- Wrote the Day 9 boundary artifact in
  `artifacts/day9-giant-test-fixture-boundary.md`.

### Findings

- Direct, graph, and integration test owners are all represented in the Day 10
  queue.
- The safest first Day 10 extraction is graph/reorder fixture ownership because
  `tests/test_graph.c` and `tests/test_reorder_nd.c` duplicate synthetic graph
  builders and partition invariant helpers while preserving local test intent.
- The second Day 10 extraction should be direct-solver assertion ownership,
  starting with LU CSR/direct CSC residual and factorization assertion helpers
  rather than changing test cases.
- `tests/test_integration.c` has strong helper pressure but broader blast
  radius; Day 9 keeps it as a Day 11 integration/lifecycle helper target.
- Header-only test helper files are the preferred default because existing
  helper owners use that pattern and it avoids CTest registration churn.

### Giant-Test Inventory Snapshot

| file | lines | local helper/macro blocks | test cases | Day 9 disposition |
|---|---:|---:|---:|---|
| `tests/test_ldlt_csc.c` | 3,884 | 137 | 100 | direct CSC assertion/oracle helper candidate |
| `tests/test_integration.c` | 3,421 | 68 | 53 | defer broad lifecycle helpers to Day 11 |
| `tests/test_qr.c` | 3,234 | 84 | 73 | QR oracle/residual helper candidate after direct/graph pass |
| `tests/test_graph.c` | 2,925 | 79 | 61 | primary Day 10 graph fixture candidate |
| `tests/test_reorder_nd.c` | 2,340 | 56 | 35 | primary Day 10 reorder fixture/cache candidate |
| `tests/test_lu_csr.c` | 1,899 | 59 | 53 | direct-solver assertion helper candidate |

### Selected Day 10 Targets

1. Add a graph/reorder fixture helper owner, tentatively
   `tests/test_graph_fixtures.h`, for reusable synthetic matrix builders and
   partition invariant helpers currently duplicated or localized in
   `tests/test_graph.c` and `tests/test_reorder_nd.c`.
2. Add a direct-solver assertion/helper owner, tentatively
   `tests/test_direct_solver_helpers.h`, for LU CSR/direct CSC residual,
   factorization, and matrix-comparison helpers that can move without changing
   test names or assertions.
3. Leave integration lifecycle helpers and QR oracle helpers deferred until
   after the Day 10 direct/graph extraction proves the helper naming and
   validation pattern.

### Validation Plan

- Header-only helper extraction:
  - focused affected test binaries
  - `make format && make lint && make test` when `.c` or `.h` files change
  - `ctest -N` or `make quality-review-cmake-compile` only if test target
    membership or registration changes
- Graph/reorder helper extraction:
  - `make build/test_graph build/test_reorder_nd`
  - `./build/test_graph`
  - `./build/test_reorder_nd`
  - `make large-matrix-guardrails` if guardrail-owned fixtures or policies are
    touched
- Direct-solver helper extraction:
  - `make build/test_lu_csr build/test_ldlt_csc build/test_direct_csc_regression`
  - `./build/test_lu_csr`
  - `./build/test_ldlt_csc`
  - `./build/test_direct_csc_regression`
- Docs-only Day 9 validation:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_106`

### Day 9 Exit State

Day 9 is complete. Sprint 106 now has a fixture/helper extraction boundary with
direct, graph/reorder, and integration owners represented, behavior-preserving
targets separated from test changes, naming rules defined, and validation
commands known before Day 10 edits.

## Day 10 - Direct and Graph Fixture Extraction

### Goal

Split reusable direct-solver and graph/reorder fixtures from giant test owners
while preserving test names, assertions, registration, and reviewed CTest shape.

### Actions

- Added `tests/test_graph_fixtures.h` as the graph/reorder fixture helper owner.
- Moved shared graph/reorder helper mechanics out of giant tests:
  - 2D grid fixture builder
  - 1D path fixture builder
  - 3D mesh fixture builder
  - two-cliques-with-bridge fixture builder
  - partition invariant checker
  - partition side counters
  - bipartition side counters
  - cut and side-weight helpers
- Updated `tests/test_graph.c` and `tests/test_reorder_nd.c` to include
  `test_graph_fixtures.h` and use the `tf_` helper names.
- Kept graph-specific local helpers, such as weighted edge mutation and the
  asymmetric boundary fixture, in `tests/test_graph.c` because they are not
  shared by reorder tests.
- Added `tests/test_direct_solver_helpers.h` as the direct-solver assertion
  helper owner.
- Moved LU CSR assertion/residual mechanics out of `tests/test_lu_csr.c`:
  - sparse matrix equality assertion
  - LU CSR factorization verification
  - infinity-norm residual helper
- Preserved all test target names, `RUN_TEST(...)` registrations, and CTest
  registration count.
- Wrote the Day 10 implementation artifact in
  `artifacts/day10-direct-graph-fixture-extraction.md`.

### Findings

- Header-only helpers were sufficient; no Makefile or CMake test target source
  lists needed changes.
- `tests/test_graph.c`, `tests/test_reorder_nd.c`, and `tests/test_lu_csr.c`
  are all smaller while keeping local proof intent at the call sites.
- The graph helper owner is shared by both graph partition tests and nested
  dissection tests, matching the Day 9 boundary.
- Direct CSC tests did not need direct edits for this batch because the first
  direct-solver extraction was the LU CSR assertion/residual seam. Focused
  LDLT CSC and direct CSC regression validation still passed.
- CMake registration remained at 54 tests, matching the Makefile test count.

### Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `tests/test_graph.c` | 2,925 lines | 2,758 lines | -167 |
| `tests/test_reorder_nd.c` | 2,340 lines | 2,304 lines | -36 |
| `tests/test_lu_csr.c` | 1,899 lines | 1,806 lines | -93 |
| `tests/test_graph_fixtures.h` | 0 lines | 195 lines | +195 |
| `tests/test_direct_solver_helpers.h` | 0 lines | 93 lines | +93 |

### Validation Results

- `python3 scripts/check_library_sources.py`: passed;
  `source-list-check: PASS (45 library sources)`.
- Focused Day 10 validation passed:
  - `make build/test_graph build/test_reorder_nd build/test_lu_csr build/test_ldlt_csc build/test_direct_csc_regression`
  - `./build/test_graph`: 61 tests, 0 failed, 0 skipped, 1,762 assertions
  - `./build/test_reorder_nd`: 35 tests, 0 failed, 1 skipped, 105 assertions
  - `./build/test_lu_csr`: 53 tests, 0 failed, 0 skipped, 1,062,184 assertions
  - `./build/test_ldlt_csc`: 100 tests, 0 failed, 0 skipped, 2,335 assertions
  - `./build/test_direct_csc_regression`: 8 tests, 0 failed, 0 skipped, 42 assertions
- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Reviewed CMake compile/parity path passed:
  - `make quality-review-cmake-compile`
  - CMake tests: 54
  - Makefile tests: 54
  - test-count parity passed
- Large-matrix guardrail target was not run because Day 10 moved header-only
  fixture/helper mechanics and did not change guardrail scripts, policies,
  thresholds, or registered lanes.
- Final hygiene checks passed:
  - `git diff --check`
  - trailing-whitespace scan across touched test and Sprint 106 planning files
  - final source-list recheck

### Day 10 Exit State

Day 10 is complete. Sprint 106 now has graph/reorder and direct-solver
header-only helper owners, smaller giant test files, unchanged test
registration, focused family proof, the required full C quality gate, and
reviewed CMake test-count parity.

## Day 11 - Integration and Oracle Fixture Extraction

### Goal

Split reusable integration and oracle fixtures out of the largest remaining
integration proof owner while keeping workflow intent readable at call sites.

### Actions

- Added `tests/test_integration_fixtures.h` as a focused header-only
  integration fixture owner.
- Moved reusable integration setup/oracle mechanics out of
  `tests/test_integration.c`:
  - progress callback counter and cancellation helper
  - shared SPD tridiagonal matrix fixture
  - unsymmetric 4x4 LU workflow fixture
  - indefinite KKT lifecycle fixture
  - KKT perturbation helper for same-pattern refactor proofs
  - CSR and CSC constructor round-trip fixture helpers
- Updated `tests/test_integration.c` to include the helper owner and use
  `integration_` helper names at LU, Cholesky, LDLT, direct lifecycle,
  compressed-first constructor, QR, iterative, and eigensolver call sites.
- Preserved all test names, `RUN_TEST(...)` registrations, and Make/CMake test
  target shape.
- Wrote the Day 11 implementation artifact in
  `artifacts/day11-integration-oracle-fixture-extraction.md`.

### Findings

- The extraction could remain header-only, so no Makefile, CMake, or library
  source-list membership changes were required.
- `tests/test_integration.c` still owns the workflow narratives, but repeated
  matrix construction and progress callback mechanics no longer obscure the
  proof sections.
- The same SPD fixture now serves direct, QR, iterative, and eigensolver
  progress proof call sites through one integration-owned helper.
- Day 12 should explicitly reconcile that this new helper is test-only and not
  part of the library source-list contract.

### Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `tests/test_integration.c` | 3,421 lines | 3,279 lines | -142 |
| `tests/test_integration_fixtures.h` | 0 lines | 140 lines | +140 |

### Validation Results

- Focused Day 11 validation passed:
  - `make build/test_integration && ./build/test_integration`
  - `test_integration`: 58 tests, 0 failed, 0 skipped, 16,763 assertions
- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Source-list validation passed:
  - `python3 scripts/check_library_sources.py`
  - `source-list-check: PASS (45 library sources)`
- Final hygiene checks passed:
  - `git diff --check`
  - trailing-whitespace scan across touched Day 11 test and planning files

### Day 11 Exit State

Day 11 is complete. Sprint 106 now has a dedicated integration fixture helper
owner, a smaller integration giant test, unchanged registration, passing
focused integration proof, and the required full C quality gate.

## Day 12 - Source-List and CMake Parity Reconciliation

### Goal

Audit every file added or split during Sprint 106 and reconcile ownership
across Make, CMake, source-list checking, test registration, and reviewed CI
assumptions.

### Actions

- Audited the Sprint 106 extracted source owners:
  - `src/sparse_ldlt_csc_rowadj.c`
  - `src/sparse_qr_householder.c`
  - `src/sparse_qr_internal.h`
  - `src/sparse_lu_csr_struct.c`
  - `src/sparse_lu_csr_internal.h`
- Audited the Sprint 106 extracted test helper owners:
  - `tests/test_graph_fixtures.h`
  - `tests/test_direct_solver_helpers.h`
  - `tests/test_integration_fixtures.h`
- Compared Makefile `LIB_SRCS`, CMake `add_library(...)`, and
  `build-metadata/library_sources.txt` ownership for the new library `.c`
  files.
- Confirmed the test helper headers remain header-only and are included through
  their owning test translation units rather than registered as separate
  Make/CMake sources.
- Ran reviewed CMake configure/build and `ctest -N` parity validation.
- Wrote the Day 12 reconciliation artifact in
  `artifacts/day12-source-list-cmake-reconciliation.md`.

### Findings

- All three new library implementation owners are present in Make, CMake, and
  `build-metadata/library_sources.txt`:
  - `src/sparse_ldlt_csc_rowadj.c`
  - `src/sparse_qr_householder.c`
  - `src/sparse_lu_csr_struct.c`
- Private headers remain private source-tree contracts and are not public
  install/export headers:
  - `src/sparse_qr_internal.h`
  - `src/sparse_lu_csr_internal.h`
  - `src/sparse_ldlt_csc_internal.h`
- Test helper headers are intentionally test-only:
  - `tests/test_graph_fixtures.h` is consumed by `test_graph.c` and
    `test_reorder_nd.c`.
  - `tests/test_direct_solver_helpers.h` is consumed by `test_lu_csr.c`.
  - `tests/test_integration_fixtures.h` is consumed by `test_integration.c`.
- No test executable was added, removed, or renamed during fixture extraction.
- CMake and Make reviewed test surfaces still agree at 54 tests.

### Extracted Owner Sizes

| file | lines |
|---|---:|
| `src/sparse_ldlt_csc_rowadj.c` | 82 |
| `src/sparse_qr_householder.c` | 79 |
| `src/sparse_qr_internal.h` | 16 |
| `src/sparse_lu_csr_struct.c` | 57 |
| `src/sparse_lu_csr_internal.h` | 9 |
| `tests/test_graph_fixtures.h` | 195 |
| `tests/test_direct_solver_helpers.h` | 93 |
| `tests/test_integration_fixtures.h` | 140 |

### Validation Results

- Source-list validation passed:
  - `python3 scripts/check_library_sources.py`
  - `source-list-check: PASS (45 library sources)`
- Reviewed CMake compile/parity path passed:
  - `make quality-review-cmake-compile`
  - CMake configure passed
  - clean CMake rebuild passed
  - `ctest -N --test-dir build/quality-review-cmake`: 54 tests
  - Makefile/CMake test-count parity: 54 tests on both surfaces
- Final documentation hygiene passed:
  - `git diff --check`
  - trailing-whitespace scan across Sprint 106 planning files

### Day 12 Exit State

Day 12 is complete. Sprint 106 extracted library sources are reconciled across
Make, CMake, and the source-list checker; extracted test helpers remain
test-only through their owning translation units; and reviewed CTest parity
remains exact at 54 tests.

## Day 13 - Maintainer Guidance and Metrics Update

### Goal

Update maintainer documentation for the Sprint 106 ownership model, compile
before/after metrics, and make the residual extraction queue explicit for
Sprint 107 planning.

### Actions

- Updated `docs/maintainer_guide.md` with Sprint 106 maintainability ownership
  guidance for:
  - `src/sparse_ldlt_csc_rowadj.c`
  - `src/sparse_qr_householder.c`
  - `src/sparse_qr_internal.h`
  - `src/sparse_lu_csr_struct.c`
  - `src/sparse_lu_csr_internal.h`
  - `tests/test_graph_fixtures.h`
  - `tests/test_direct_solver_helpers.h`
  - `tests/test_integration_fixtures.h`
- Added guidance that reusable helper mechanics should move to existing
  Sprint 106 helper owners rather than growing giant source or proof files.
- Added guidance that any future compiled test-helper target needs explicit
  Make/CMake registration and reviewed CTest-surface reconciliation.
- Compiled source, giant-test, helper-owner, source-list, and test-surface
  metrics.
- Wrote the Day 13 metrics and guidance artifact in
  `artifacts/day13-maintainer-guidance-and-metrics.md`.

### Metrics

| source owner | baseline | current | change |
|---|---:|---:|---:|
| `src/sparse_ldlt_csc.c` | 2,174 | 2,095 | -79 |
| `src/sparse_qr.c` | 1,563 | 1,448 | -115 |
| `src/sparse_lu_csr.c` | 1,665 | 1,594 | -71 |
| `src/sparse_ldlt_csc_rowadj.c` | 0 | 82 | +82 |
| `src/sparse_qr_householder.c` | 0 | 79 | +79 |
| `src/sparse_qr_internal.h` | 0 | 16 | +16 |
| `src/sparse_lu_csr_struct.c` | 0 | 57 | +57 |
| `src/sparse_lu_csr_internal.h` | 0 | 9 | +9 |

| test owner | baseline | current | change |
|---|---:|---:|---:|
| `tests/test_graph.c` | 2,925 | 2,758 | -167 |
| `tests/test_reorder_nd.c` | 2,340 | 2,304 | -36 |
| `tests/test_lu_csr.c` | 1,899 | 1,806 | -93 |
| `tests/test_integration.c` | 3,421 | 3,279 | -142 |
| `tests/test_graph_fixtures.h` | 0 | 195 | +195 |
| `tests/test_direct_solver_helpers.h` | 0 | 93 | +93 |
| `tests/test_integration_fixtures.h` | 0 | 140 | +140 |

| build/test surface | baseline | current |
|---|---:|---:|
| source-list checker library sources | 42 | 45 |
| reviewed Makefile test binaries | 54 | 54 |
| reviewed CMake tests | 54 | 54 |
| new compiled test helper targets | 0 | 0 |
| new public/install headers | 0 | 0 |

### Residual Queue

- `tests/test_ldlt_csc.c`: largest proof owner remains large; future movement
  should start with one row-adjacency or residual/oracle helper, not broad
  fixture relocation.
- `tests/test_qr.c`: QR source seam was extracted, but QR proof fixtures remain
  large; future extraction should keep solve/reconstruction call-site intent
  visible.
- `tests/test_iterative.c`: convergence evidence wording and external-reference
  helper behavior make broad cleanup risky; extract only reusable matrix/RHS
  builders first.
- `tests/test_svd.c`: rank/oracle evidence is claim-sensitive; defer until a
  dedicated SVD proof-owner cleanup with focused validation.
- `src/sparse_eigs.c`: still large, but tied to Sprint 103 comparison surfaces;
  needs a fresh boundary before any source split.
- `src/sparse_matrix.c`: central matrix-shell API/compatibility risk remains
  too high for opportunistic maintainability extraction.

### Validation Results

- Day 13 changed documentation only, so no C quality gate was required.
- Documentation hygiene passed:
  - `git diff --check`
  - trailing-whitespace scan across `docs/maintainer_guide.md` and
    `docs/planning/EPIC_10/SPRINT_106`

### Day 13 Exit State

Day 13 is complete. Maintainer guidance now points maintainers to the Sprint
106 source and test helper ownership layout, metrics show the impact and limits
of the extraction work, and residual cleanup is explicit for Sprint 107
planning.

## Day 14 - Validation and Sprint Closeout

### Goal

Run the final required quality gates, reconcile Sprint 106 artifacts and
metrics, and close the sprint with explicit Sprint 107 handoff evidence.

### Actions

- Ran the required full C quality gate because the branch contains `.c` and
  `.h` changes.
- Re-ran source-list validation.
- Re-ran reviewed CMake configure, clean rebuild, `ctest -N`, and Make/CMake
  test-count parity validation.
- Ran large-matrix guardrail validation because Sprint 106 touched graph and
  reorder fixture ownership.
- Reconciled final ownership metrics against Day 13 metrics after formatting.
- Wrote the Day 14 closeout artifact in
  `artifacts/day14-validation-closeout.md`.

### Final Metrics

| source owner | baseline | final | change |
|---|---:|---:|---:|
| `src/sparse_ldlt_csc.c` | 2,174 | 2,095 | -79 |
| `src/sparse_qr.c` | 1,563 | 1,448 | -115 |
| `src/sparse_lu_csr.c` | 1,665 | 1,594 | -71 |
| `src/sparse_ldlt_csc_rowadj.c` | 0 | 82 | +82 |
| `src/sparse_qr_householder.c` | 0 | 79 | +79 |
| `src/sparse_qr_internal.h` | 0 | 16 | +16 |
| `src/sparse_lu_csr_struct.c` | 0 | 57 | +57 |
| `src/sparse_lu_csr_internal.h` | 0 | 9 | +9 |

| test owner | baseline | final | change |
|---|---:|---:|---:|
| `tests/test_graph.c` | 2,925 | 2,758 | -167 |
| `tests/test_reorder_nd.c` | 2,340 | 2,304 | -36 |
| `tests/test_lu_csr.c` | 1,899 | 1,806 | -93 |
| `tests/test_integration.c` | 3,421 | 3,279 | -142 |
| `tests/test_graph_fixtures.h` | 0 | 195 | +195 |
| `tests/test_direct_solver_helpers.h` | 0 | 93 | +93 |
| `tests/test_integration_fixtures.h` | 0 | 140 | +140 |

| build/test surface | baseline | final |
|---|---:|---:|
| source-list checker library sources | 42 | 45 |
| reviewed Makefile test binaries | 54 | 54 |
| reviewed CMake tests | 54 | 54 |
| new compiled test helper targets | 0 | 0 |
| new public/install headers | 0 | 0 |

### Final Validation Results

- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Source-list validation passed:
  - `python3 scripts/check_library_sources.py`
  - `source-list-check: PASS (45 library sources)`
- Reviewed CMake compile/parity path passed:
  - `make quality-review-cmake-compile`
  - CMake configure passed
  - clean CMake rebuild passed
  - `ctest -N --test-dir build/quality-review-cmake`: 54 tests
  - Makefile/CMake test-count parity: 54 tests on both surfaces
- Large-matrix guardrails passed:
  - `make large-matrix-guardrails`
  - reports written under `build/bench-reports/large-matrix-guardrails`
- Final hygiene passed:
  - `git diff --check`
  - trailing-whitespace scan across `docs/maintainer_guide.md` and
    `docs/planning/EPIC_10/SPRINT_106`

### Sprint 107 Handoff

- `tests/test_ldlt_csc.c`: still the largest direct CSC proof owner; extract
  only one row-adjacency assertion or residual/oracle helper after a narrow
  boundary artifact.
- `tests/test_qr.c`: QR source seam is smaller, but QR proof fixtures remain
  broad; split repeated matrix/vector builders without hiding
  solve/reconstruction intent.
- `tests/test_iterative.c`: convergence evidence and external-reference
  wording are sensitive; start with reusable matrix/RHS builders only.
- `tests/test_svd.c`: SVD rank/oracle claims need a dedicated proof-owner
  cleanup day with focused validation.
- `src/sparse_eigs.c`: still large, but tied to Sprint 103 comparison surfaces;
  require a fresh boundary before splitting workspace or dispatch helpers.
- `src/sparse_matrix.c`: central matrix-shell API/compatibility risk remains
  too high for opportunistic maintainability work.

### Day 14 Exit State

Day 14 is complete. Sprint 106 now has passing final validation, complete
artifacts, reconciled metrics, maintainer guidance for the new ownership
layout, exact Make/CMake reviewed test parity, and a documented Sprint 107
handoff queue.
