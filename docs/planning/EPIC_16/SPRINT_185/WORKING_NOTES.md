# Sprint 185 Working Notes

## Sprint Goal

Reduce one large review surface by extracting helpers or proof-owner files
while preserving behavior and build registration.

## Branch Baseline

- Branch: `sprint-185`
- Starting point: current `master` after PR #204 merge.
- Sprint 184 status: complete and merged.
- Sprint 185 plan status: day-by-day plan exists at
  `docs/planning/EPIC_16/SPRINT_185/PLAN.md`.

## Planning Source

| Field | Value |
| --- | --- |
| Project plan | `docs/planning/EPIC_16/PROJECT_PLAN.md` |
| Section | `Sprint 185: Large Test and Solver Review-Surface Reduction` |
| Sprint duration | 14 days, approximately 168 hours |
| Selected residual | `S177-R13`: Large test/source review-surface reduction |
| Evidence matrix rows | `ESM-014`, `ESM-013` |

## Sprint 185 Item Boundaries

| Item | Name | Sprint 185 interpretation |
| --- | --- | --- |
| 185.1 | Cluster Selection | Select one large test/source cluster with high review cost and low behavior/refactor risk. |
| 185.2 | Extraction Design | Define helper boundaries, fixture ownership, internal declarations, and no-behavior-change validation before moving code. |
| 185.3 | Mechanical Extraction | Extract helpers, fixtures, or focused proof-owner files for the selected cluster only, updating build/test registration as needed. |
| 185.4 | Drift Guard Update | Update existing source-list/build registration checks or add a selected-cluster guard only if needed. |
| 185.5 | Maintenance Note | Document invariants and future contribution guidance for the selected cluster's new layout. |
| 185.6 | Validation | Run full C quality gates when `.c` or `.h` files change, plus affected tests, source-list checks, registration guards, and `git diff --check`. |

## Prior Evidence Carried Forward

| Input | Source | Sprint 185 use |
| --- | --- | --- |
| Large review-surface inventory | `docs/planning/EPIC_16/SPRINT_177/artifacts/day4-surface-inventory.md` | Names the largest test/source/tooling/doc files and duplicated registration risks. |
| Target selection | `docs/planning/EPIC_16/SPRINT_177/artifacts/day7-target-selection.md` | Confirms Sprint 185 closes `S177-R13` by selecting exactly one large review surface. |
| Sprint 177 retrospective | `docs/planning/EPIC_16/SPRINT_177/RETROSPECTIVE.md` | Carries the risk that large test/source files slow review and should be reduced one cluster at a time. |
| QR proof-owner precedent | `docs/planning/EPIC_12/SPRINT_139/artifacts/day14-closeout-validation-summary.md` | Local precedent for extracting a focused proof-owner test without weakening existing broad coverage. |
| Source-list guard | `scripts/check_library_sources.py` and `make source-list-check` | Guards library source membership and order across `build-metadata/library_sources.txt`, `Makefile`, and `CMakeLists.txt`. |
| CMake parity guard | `make quality-review-cmake-compile` | Checks CMake configure/build, `ctest -N`, and Makefile/CMake test-count parity. |

## Current Candidate Review Surfaces

Day 1 does not select the Sprint 185 cluster. It records the inherited
candidate set and current line counts so Day 2 can build a detailed baseline.

| Candidate | Current lines | Primary owner | Day 1 notes |
| --- | ---: | --- | --- |
| `tests/test_qr.c` | 3970 | QR factorization, rank, nullspace, projector, reorder, and refinement tests | Largest test file. Strong review-cost candidate, but Sprint 184 just touched QR docs/header and Sprint 139 already added `test_qr_corpus` as a focused proof owner. |
| `tests/test_ldlt_csc.c` | 3915 | LDLT CSC allocation, row adjacency, supernodal, native pivoting, solve, inertia, and dense-reference tests | Very large direct-solver proof surface. Good candidate if helper boundaries can isolate native pivot or dense-reference setup without behavior drift. |
| `tests/test_integration.c` | 3279 | Cross-solver lifecycle, progress callback, refactor, and matrix-shell integration tests | Broad integration owner with high risk of accidental scope expansion. Candidate only if a focused helper or fixture seam is obvious. |
| `tests/test_svd.c` | 3029 | Full SVD, partial SVD, rank, pseudoinverse, low-rank, and dense-reference tests | Large mixed SVD proof owner. Existing partial-SVD helper headers may provide extraction precedent. |
| `tests/test_ldlt.c` | 3006 | Public LDLT behavior, reorder, KKT, refinement, condest, backend, dense-helper tests | Large direct-solver family test. Candidate if KKT/dense-helper fixtures can move cleanly. |
| `tests/test_etree.c` | 2962 | Elimination tree, postorder, column count, analysis/writeback tests | Large ordering/analysis owner; helper extraction may be feasible but requires careful registration and fixture ownership. |
| `tests/test_iterative.c` | 2929 | Iterative solvers, allocation-failure proof lane, preconditioners, solver helpers | Important allocation-failure proof owner; behavior risk is higher because Sprint 178/176 gates depend on it. |
| `tests/test_graph.c` | 2764 | Graph/coarsen/bisect/FM/partition/large guardrail proof surface | Large graph surface with existing `tests/test_graph_fixtures.h`; possible candidate if fixture extraction remains family-local. |
| `tests/test_chol_csc.c` | 2554 | Cholesky CSC family-local analysis-backed and publish-back proof surface | Candidate direct-solver test, but less urgent than QR/LDLT CSC by line count. |
| `src/sparse_ldlt_csc.c` | 2095 | LDLT CSC implementation path | Large implementation file. Higher risk than test-only extraction because source behavior and library registration may change. |
| `scripts/run_external_comparison.py` | 2094 | Selected external comparison runner | Large Python tooling surface. Useful candidate only if Sprint 185 selects tooling rather than C/test extraction. |
| `tests/test_normalize_report_index.py` | 1861 | Report-index and selected-report regression tests | Large Python test surface with broad report ownership; good tooling candidate but not a solver/test extraction by default. |
| `docs/maintainer_guide.md` | 1761 | Maintainer evidence and non-claim guidance | Large documentation surface, but Sprint 185 goal emphasizes test/source clusters and build registration. |

## Existing Registration And Guard Surfaces

| Surface | Files or command | Day 1 interpretation |
| --- | --- | --- |
| Make library registration | `Makefile` `LIB_SRCS` | Must change if a new library `.c` file is extracted. |
| CMake library registration | `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)` | Must change with any new library `.c` file. |
| Library source manifest | `build-metadata/library_sources.txt` | Source of truth for library source-list parity. |
| Library source-list guard | `make source-list-check` / `scripts/check_library_sources.py` | Verifies manifest, Makefile, and CMake library source membership/order. |
| Make test registration | `Makefile` `TEST_SRCS` | Must change if a new test binary is created. |
| CMake test registration | `CMakeLists.txt` `add_sparse_test(...)` | Must change if a new CMake test binary is created. |
| CMake test-count parity | `make quality-review-cmake-compile` | Checks CMake `ctest -N` count against Makefile `TEST_BINS`. |
| Formatting/lint coverage | `make format`, `make format-check`, `make lint` | Uses registered source globs and explicit file lists; must cover any extracted C/H test or source files. |
| Focused family tests | `make build/<test>` and `./build/<test>` | Primary Day 12/13 validation path for the selected cluster. |

## Day 2 Candidate Baseline

Day 2 scores candidates on two separate axes:

- Review cost: 1 low to 5 high, based on line count, static/function count,
  mixed responsibilities, repeated fixtures, and likely review churn.
- Refactor risk: 1 low to 5 high, based on behavior sensitivity,
  registration needs, numerical tolerance risk, fixture ordering, and proof
  ownership.

The strongest Day 3 candidate should have high review cost, lower refactor
risk, an obvious helper seam, and focused validation that can prove no behavior
change.

| Candidate | Lines | Static/function count | Registration owner | Review cost | Refactor risk | Day 2 disposition |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `tests/test_ldlt_csc.c` | 3915 | 130 | `test_ldlt_csc` in Make/CMake | 5 | 3 | Front-runner. Large direct-solver surface with helper seams around dense/symmetric/KKT builders, row-adjacent checks, and assertion helpers. |
| `tests/test_svd.c` | 3029 | 90 | `test_svd` in Make/CMake | 4 | 3 | Strong alternate. Existing SVD helper-header pattern reduces extraction uncertainty, but full/partial SVD responsibilities are broad. |
| `tests/test_graph.c` | 2764 | 68 | `test_graph` in Make/CMake | 4 | 3 | Viable fallback. Existing graph fixtures help, but graph/FM environment interactions need careful containment. |
| `tests/test_qr.c` | 3970 | 83 | `test_qr` in Make/CMake | 5 | 4 | Defer unless stronger candidates fail; recent QR work and existing QR proof owners make the surface more sensitive. |
| `tests/test_integration.c` | 3279 | 58 | `test_integration` in Make/CMake | 5 | 5 | Defer. Broad cross-solver behavior owner is too easy to expand accidentally. |
| `tests/test_ldlt.c` | 3006 | 95 | `test_ldlt` in Make/CMake | 4 | 4 | Candidate, but less clear than `tests/test_ldlt_csc.c` because it mixes public behavior, backend, KKT, and refinement coverage. |
| `tests/test_etree.c` | 2962 | 111 | `test_etree` in Make/CMake | 4 | 3 | Viable but lower solver-priority than LDLT CSC or SVD for this sprint. |
| `tests/test_iterative.c` | 2929 | 94 | `test_iterative` in Make/CMake | 4 | 5 | Defer. Allocation-failure proof ownership raises behavior-preservation risk. |
| `tests/test_chol_csc.c` | 2554 | 111 | `test_chol_csc` in Make/CMake | 3 | 3 | Viable but lower impact by current review surface. |
| `src/sparse_ldlt_csc.c` | 2095 | n/a | library source lists | 4 | 5 | Defer. Implementation extraction would require source-list registration and carries behavior risk. |
| `scripts/run_external_comparison.py` | 2094 | n/a | Python tooling | 3 | 4 | Defer. Useful tooling surface, but Sprint 185 defaults to solver/test review-surface reduction. |
| `tests/test_normalize_report_index.py` | 1861 | n/a | Python test tooling | 3 | 3 | Defer. Lower fit for the selected solver/test cluster. |
| `docs/maintainer_guide.md` | 1761 | n/a | documentation | 2 | 2 | Defer. Documentation size is real but not the primary Sprint 185 target. |

## Day 2 Registration Map

| Extraction type | Registration impact | Required validation |
| --- | --- | --- |
| Test-only helper header | No new Make/CMake binary when included by an existing test. | Focused test binary plus full C gate if `.c` or `.h` files change. |
| New proof-owner test binary | Add to `Makefile` `TEST_SRCS` and `CMakeLists.txt` via `add_sparse_test(...)`. | Focused test, `make quality-review-cmake-compile`, and full C gate. |
| New library source file | Add to `Makefile`, `CMakeLists.txt`, and `build-metadata/library_sources.txt`. | `make source-list-check`, focused tests, and full C gate. |
| Python tooling extraction | No C registration unless C/H files also change. | `python -m py_compile ...` and focused Python tests. |

## Day 3 Selected Cluster Decision

Sprint 185 selects `tests/test_ldlt_csc.c` as the only active extraction
cluster.

| Field | Decision |
| --- | --- |
| Selected file | `tests/test_ldlt_csc.c` |
| Current size | 3915 lines |
| Static/function count | 130 |
| Existing test binary | `test_ldlt_csc` |
| Make registration | `Makefile` `TEST_SRCS` includes `$(TESTDIR)/test_ldlt_csc.c` |
| CMake registration | `CMakeLists.txt` includes `add_sparse_test(test_ldlt_csc)` |
| Preferred extraction type | Family-local test helper header included by `tests/test_ldlt_csc.c` |
| Registration target | Avoid a new binary unless Day 4 proves a proof-owner split is safer |
| Focused validation | `make build/test_ldlt_csc && ./build/test_ldlt_csc` |
| Required full gate after C/H edits | `make format && make lint && make test` |

The selected cluster has high review cost and clear family-local helper seams:
dense/KKT fixture builders, residual/reference helpers, supernode state
assertion helpers, dense symmetric-swap oracles, and native-wrapper comparison
helpers. The first design preference is to extract helpers into a
`tests/test_ldlt_csc_*` header so the existing `test_ldlt_csc` proof owner,
test names, and Make/CMake registration remain stable.

## Day 3 No-Behavior-Change Contract

- Preserve all existing `RUN_TEST(...)` entries, test names, assertion
  semantics, numerical tolerances, fixture values, random seeds, and skip
  behavior.
- Keep `test_ldlt_csc` as the proof-owner binary unless a later design
  artifact explicitly justifies a new registered test.
- Do not change public APIs, internal solver APIs, LDLT CSC implementation
  files, source-list metadata, or production behavior during helper
  extraction.
- Preserve external dense-reference semantics, including `_POSIX_C_SOURCE`,
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`, Windows skip behavior, and the
  Python helper command.
- Preserve process-global kernel override restoration behavior around native
  vs wrapper checks.
- Treat any `.c` or `.h` movement as requiring the full C quality gate before
  the sprint proceeds past implementation.

## Day 3 Extraction Checklist Draft

| Day | Planned focus | Day 3 checklist |
| --- | --- | --- |
| Day 4 | Helper boundary design | Inspect exact helper blocks in `tests/test_ldlt_csc.c`; choose the helper-header name and dependency order; decide what remains local. |
| Day 5 | Registration guardrail design | Confirm whether extraction is header-only; if so, record no Make/CMake/source-list changes and validation expectations. |
| Day 6 | First mechanical extraction | Move the lowest-risk fixtures or assertion helpers first, preserving static linkage through inclusion in the same test translation unit. |
| Day 7 | Fixture/setup extraction | Move the next approved KKT, dense-reference, or oracle helpers only if Day 6 validation is clean. |
| Day 8 | Proof-owner cleanup | Clean includes, declarations, and helper ordering in `tests/test_ldlt_csc.c` without changing `RUN_TEST(...)` coverage. |

## Day 4 Helper Boundary Design

Day 4 keeps Sprint 185 on a test-helper extraction path. No solver-internal
source files, public headers, library source manifests, or new test binaries
are planned for the first mechanical pass.

| Proposed file | Ownership | Planned contents | Registration impact |
| --- | --- | --- | --- |
| `tests/test_ldlt_csc_supernode_helpers.h` | Family-local helpers for supernode detection, extract/writeback, and supernodal factor comparison tests. | `build_dense_ldlt_with_pivots`, `cm_idx`, `snapshot_supernode_state`, `ldlt_csc_factor_state_matches`, and related dense-SPD/random fixture helpers if Day 6 confirms dependency order. | Header-only include from `tests/test_ldlt_csc.c`; no Make/CMake change. |
| `tests/test_ldlt_csc_fixtures.h` | Family-local KKT, scaled-KKT, random-indefinite, and two-pass factor fixtures. | `build_kkt_5x5`, `build_kkt_10x10`, `build_kkt_scaled_10x10`, `s20_two_pass_indefinite_factor`, and fixture-specific allocation cleanup helpers after Day 6 proves the header pattern. | Header-only include from `tests/test_ldlt_csc.c`; no Make/CMake change. |
| `tests/test_ldlt_csc_oracle_helpers.h` | Dense oracle and comparison helpers used by symmetric-swap and native-wrapper tests. | `ldlt_lower_to_dense`, `dense_sym_swap`, `dense_lower_equal`, `build_ldlt_from_triples`, `ldlt_column_nonzeros_match`, `ldlt_factorizations_match`, and `check_native_matches_wrapper`. | Header-only include from `tests/test_ldlt_csc.c`; no Make/CMake change. |

The first implementation pass should create only
`tests/test_ldlt_csc_supernode_helpers.h`. The other two headers are approved
candidate boundaries, not mandatory Day 6 movement.

### Day 4 Include And Dependency Rules

- Keep helper headers self-contained with include guards and only the headers
  their helpers need.
- Include helper headers from `tests/test_ldlt_csc.c` after
  `test_solver_helpers.h`, preserving `_POSIX_C_SOURCE` and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` placement.
- Preserve existing helper names where possible to keep test call sites
  readable and avoid needless churn.
- Use `static` helper definitions in the family-local headers so included
  helpers keep internal linkage in the `test_ldlt_csc` translation unit.
- Do not move `main`, `RUN_TEST(...)` ordering, public test functions, or
  comments that explain the chronological proof-owner sections unless a later
  implementation artifact records why.

### Day 4 First-Pass Validation Plan

After the first helper-header extraction, run:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
make format && make lint && make test
```

Because Day 6 is expected to add a `.h` file and modify `tests/test_ldlt_csc.c`,
the full C gate is required before proceeding past implementation.

## Day 5 Registration Guardrail Design

Day 5 confirms the planned first extraction remains header-only and does not
require Makefile, CMake, library source-list, or test-count registration
changes.

| Surface | Day 5 decision | Rationale |
| --- | --- | --- |
| `Makefile` `TEST_SRCS` | No change for `tests/test_ldlt_csc_supernode_helpers.h`. | The existing binary remains `test_ldlt_csc`; helper headers are included by `tests/test_ldlt_csc.c`. |
| `CMakeLists.txt` `add_sparse_test(...)` | No change. | No new proof-owner binary is planned for Day 6. |
| `Makefile` `LIB_SRCS` | No change. | No production `.c` source extraction is planned. |
| `CMakeLists.txt` `add_library(...)` | No change. | No library source file is planned. |
| `build-metadata/library_sources.txt` | No change. | `scripts/check_library_sources.py` covers library `.c` membership only. |
| Source-list guard | Existing guard remains sufficient for the unchanged library-source state. | Header-only test extraction is outside its manifest contract. |
| Format coverage | Existing `ALL_TEST_SRC = $(wildcard $(TESTDIR)/*.c) $(wildcard $(TESTDIR)/*.h)` covers the new helper header. | `make format` and `make format-check` will include `tests/test_ldlt_csc_supernode_helpers.h`. |
| Lint coverage | Existing `cppcheck ... $(SRCDIR) $(TESTDIR)` and focused compile/test checks cover the new helper header. | `make lint` scans the tests tree; focused rebuild compiles the included header. |

### Day 5 Build-Dependency Caveat

The generic Makefile test rule depends on the test `.c` file and library, not
on included test helper headers. After Day 6 adds or changes a test helper
header, the focused validation must force the `test_ldlt_csc` binary to rebuild
before execution:

```sh
rm -f build/test_ldlt_csc
make build/test_ldlt_csc && ./build/test_ldlt_csc
```

Before the full C gate, use a clean build state so `make test` compiles the
header-including test binary:

```sh
make clean
make format && make lint && make test
```

### Day 5 Expected Generated Files

| Path | Source | Staging decision |
| --- | --- | --- |
| `build/test_ldlt_csc` | Focused Makefile build | Generated; do not stage. |
| `build/*.o` and `build/libsparse_lu_ortho.a` | Focused/full Makefile builds | Generated; do not stage. |
| `build/include/sparse_version.h` | Build version header generation | Generated; do not stage. |
| `build/quality-review-cmake/` | Only if CMake parity is needed | Generated; do not stage. |

### Day 5 Rollback Criteria

- Stop and redesign if Day 6 needs a new test binary to keep helper ownership
  understandable.
- Stop and redesign if helper movement requires production source changes,
  public/internal API changes, or library source-list edits.
- Stop and ask if focused validation fails after a forced rebuild.
- Keep generated `build/` and report output unstaged.

## Day 6 Initial Helper Extraction

Day 6 performed the first no-behavior-change mechanical extraction for the
selected `tests/test_ldlt_csc.c` cluster.

| Field | Result |
| --- | --- |
| New helper header | `tests/test_ldlt_csc_supernode_helpers.h` |
| Included from | `tests/test_ldlt_csc.c` with local helper includes, normalized by `clang-format` include sorting |
| Proof owner | Existing `test_ldlt_csc` binary remains the only proof owner |
| Registration impact | No Makefile, CMake, library-source, or test-count changes |
| Production impact | No production source or public/internal API changes |
| Test-body impact | No `RUN_TEST(...)`, test name, fixture value, tolerance, or `main` changes |

Moved helpers:

| Helper | Role |
| --- | --- |
| `build_dense_ldlt_with_pivots` | Dense lower-triangular `LdltCsc` fixture builder for supernode detection tests. |
| `cm_idx` | Column-major indexing helper for dense panel buffers. |
| `snapshot_supernode_state` | Pre/post writeback state snapshot for supernode round-trip checks. |
| `ldlt_csc_factor_state_matches` | Exact factor-state comparison for scalar/supernodal cross-checks. |
| `build_dense_spd` | Dense SPD fixture builder for supernodal scalar/batched parity checks. |

Review-surface result:

| Path | Lines after Day 6 | Notes |
| --- | ---: | --- |
| `tests/test_ldlt_csc.c` | 3793 | Reduced from the Day 3 baseline of 3915 lines. |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | Family-local helper header included by the existing test binary. |

Validation completed after the source/header edit:

```sh
make format
if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
make build/test_ldlt_csc
./build/test_ldlt_csc
make lint
make test
```

Focused `test_ldlt_csc` validation passed with 100 tests, 0 failures, 0
skips, and 3556 assertions. The full C gate passed through `make format`,
`make lint`, and `make test`.

`clang-format` normalized the local include block alphabetically, placing
`test_ldlt_csc_supernode_helpers.h` before `test_solver_helpers.h`; the
focused and full gates confirm the helper remains dependency-clean in that
formatted order.

Day 7 handoff:

- Consider `tests/test_ldlt_csc_fixtures.h` as the next extraction boundary.
- Prefer KKT, scaled-KKT, or two-pass fixtures only if macro and external
  reference sensitivity remains contained.
- Keep the existing proof-owner binary and avoid registration changes unless a
  later artifact proves a new proof owner is necessary.

## Day 7 Fixture And Setup Extraction

Day 7 performed the second no-behavior-change mechanical extraction for the
selected `tests/test_ldlt_csc.c` cluster by moving KKT fixture construction and
the analysis-backed two-pass setup helper into a family-local fixture header.

| Field | Result |
| --- | --- |
| New helper header | `tests/test_ldlt_csc_fixtures.h` |
| Included from | `tests/test_ldlt_csc.c` with the local helper includes |
| Proof owner | Existing `test_ldlt_csc` binary remains the only proof owner |
| Registration impact | No Makefile, CMake, library-source, or test-count changes |
| Production impact | No production source or public/internal API changes |
| External-reference state | Remains local in `tests/test_ldlt_csc.c` |

Moved helpers:

| Helper | Role |
| --- | --- |
| `build_kkt_5x5` | Small KKT fixture used by with-analysis and external dense-reference tests. |
| `build_kkt_10x10` | Larger KKT fixture used by with-analysis, residual, external-reference, and min-size rejection tests. |
| `build_kkt_scaled_10x10` | Scaled KKT fixture used by the Sprint 102 external dense-reference lane. |
| `s20_two_pass_indefinite_factor` | Shared scalar-prepass plus analysis-backed supernodal factor setup helper. |

Review-surface result:

| Path | Lines after Day 7 | Notes |
| --- | ---: | --- |
| `tests/test_ldlt_csc.c` | 3639 | Reduced from 3793 after Day 6 and from the Day 3 baseline of 3915 lines. |
| `tests/test_ldlt_csc_fixtures.h` | 145 | New family-local KKT/two-pass fixture header. |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | Existing Day 6 family-local supernode helper header. |

Validation completed after the source/header edit:

```sh
make format
if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
make build/test_ldlt_csc
./build/test_ldlt_csc
make lint
make test
```

Focused `test_ldlt_csc` validation passed with 100 tests, 0 failures, 0
skips, and 3556 assertions. The full C gate passed through `make lint` and
`make test` after formatting and focused validation.

Day 8 handoff:

- Keep `ldlt_external_dense_reference_state_t`,
  `read_ldlt_external_dense_reference_solution`, and
  `assert_ldlt_external_dense_reference` local unless a later cleanup proves a
  separate external-reference helper boundary is lower risk.
- Prefer cleanup of include/declaration ordering and stale comments before
  extracting another broad helper block.
- Continue avoiding Makefile, CMake, and source-list changes unless a new
  proof-owner binary or production source file becomes necessary.

## Day 8 Proof-Owner Cleanup

Day 8 performed the final planned helper extraction for the selected
`tests/test_ldlt_csc.c` cluster by moving dense-oracle and native-wrapper
comparison helpers into a family-local oracle helper header.

| Field | Result |
| --- | --- |
| New helper header | `tests/test_ldlt_csc_oracle_helpers.h` |
| Included from | `tests/test_ldlt_csc.c` with the local helper includes |
| Proof owner | Existing `test_ldlt_csc` binary remains the only proof owner |
| Registration impact | No Makefile, CMake, library-source, or test-count changes |
| Production impact | No production source or public/internal API changes |
| External-reference state | Remains local in `tests/test_ldlt_csc.c` |

Moved helpers:

| Helper | Role |
| --- | --- |
| `ldlt_lower_to_dense` | Lower-triangle dense copy oracle. |
| `dense_sym_swap` | Dense symmetric permutation oracle. |
| `dense_lower_equal` | Lower-triangle dense comparison. |
| `build_ldlt_from_triples` | Sparse-to-LDLT fixture builder for symmetric-swap tests. |
| `ldlt_column_nonzeros_match` | Zero-tolerant column comparison. |
| `ldlt_factorizations_match` | Native-wrapper factor comparison. |
| `check_native_matches_wrapper` | Wrapper/native factor parity assertion helper. |

Review-surface result:

| Path | Lines after Day 8 | Notes |
| --- | ---: | --- |
| `tests/test_ldlt_csc.c` | 3469 | Reduced from 3639 after Day 7 and from the Day 3 baseline of 3915 lines. |
| `tests/test_ldlt_csc_fixtures.h` | 145 | Existing Day 7 family-local fixture header. |
| `tests/test_ldlt_csc_oracle_helpers.h` | 149 | New family-local oracle/native-wrapper helper header. |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | Existing Day 6 family-local supernode helper header. |

Validation completed after the source/header edit:

```sh
make format
if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
make build/test_ldlt_csc
./build/test_ldlt_csc
make source-list-check
make lint
make test
```

Focused `test_ldlt_csc` validation passed with 100 tests, 0 failures, 0
skips, and 3556 assertions. `make source-list-check` passed with 49 library
sources. The full C gate passed through `make format`, `make lint`, and
`make test`.

Day 9 handoff:

- Decide whether a selected-cluster guard is needed for the three helper
  headers.
- No library source guard changes are needed.
- If adding a guard, keep it focused on `test_ldlt_csc.c` including the three
  family-local headers.

## Day 9 Drift Guard Update

Day 9 added a selected-cluster guard for the Sprint 185 LDLT CSC helper-header
layout.

| Field | Result |
| --- | --- |
| Guard script | `scripts/check_ldlt_csc_helper_guard.sh` |
| Make target | `make ldlt-csc-helper-guard` |
| Proof-owner registration checked | `Makefile` `$(TESTDIR)/test_ldlt_csc.c` and `CMakeLists.txt` `add_sparse_test(test_ldlt_csc)` |
| Helper headers checked | `tests/test_ldlt_csc_fixtures.h`, `tests/test_ldlt_csc_oracle_helpers.h`, and `tests/test_ldlt_csc_supernode_helpers.h` |
| Library source-list impact | No change; helper headers remain outside `build-metadata/library_sources.txt` |
| Test binary impact | No change; no new Make/CMake test binary was added |

Guard coverage:

- verifies the registered `test_ldlt_csc` proof owner still exists in Make and
  CMake;
- verifies the three extracted family-local helper headers exist;
- verifies each helper header keeps its include guard;
- verifies `tests/test_ldlt_csc.c` includes each helper header exactly once;
- verifies the helper headers are not accidentally named as Makefile, CMake,
  or library-manifest registration entries;
- verifies no helper stem is registered as a standalone CMake test without a
  new proof-owner decision.

Limitations:

- The guard protects registration and helper layout, not numerical behavior.
- The guard intentionally does not require helper headers in
  `build-metadata/library_sources.txt` because that manifest covers library
  `.c` sources.
- The guard intentionally does not run CMake test-count parity because no new
  test binary was added.
- Behavior preservation remains covered by focused `test_ldlt_csc` execution
  and the full C gate from mechanical extraction and later validation days.

Validation completed:

```sh
bash -n scripts/check_ldlt_csc_helper_guard.sh
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

`bash -n scripts/check_ldlt_csc_helper_guard.sh`, `make
ldlt-csc-helper-guard`, and `git diff --check` passed. `make
source-list-check` passed with 49 library sources.

Day 10 handoff:

- Draft the selected-cluster maintenance note.
- Document where future LDLT CSC helper additions belong.
- Reference `make ldlt-csc-helper-guard` as the guard for the extracted
  helper-header layout.

## Day 10 Maintenance Invariants

Day 10 drafted the maintenance guidance for extending the Sprint 185 LDLT CSC
test layout without re-growing the selected proof-owner surface or drifting
registration.

| Field | Result |
| --- | --- |
| Maintenance draft | `docs/planning/EPIC_16/SPRINT_185/artifacts/day10-maintenance-invariants.md` |
| Selected proof owner | `tests/test_ldlt_csc.c` / `test_ldlt_csc` |
| Helper files covered | `tests/test_ldlt_csc_fixtures.h`, `tests/test_ldlt_csc_oracle_helpers.h`, and `tests/test_ldlt_csc_supernode_helpers.h` |
| Guard referenced | `make ldlt-csc-helper-guard` |
| Existing docs link | `docs/maintainer_guide.md` identified as the likely long-term maintainer-policy surface for Day 11 alignment |

The draft records these maintenance invariants:

- keep `test_ldlt_csc` as the only Sprint 185 LDLT CSC proof-owner binary
  unless a later reviewed artifact explicitly chooses a split;
- keep `main`, `RUN_TEST(...)` ordering, test names, fixture values,
  tolerances, `_POSIX_C_SOURCE`, and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` ownership in `tests/test_ldlt_csc.c`;
- keep extracted helper headers family-local and included by
  `tests/test_ldlt_csc.c`;
- keep helper definitions with internal linkage through `static` or
  `static inline`;
- keep helper headers out of Makefile test registration, CMake test
  registration, and `build-metadata/library_sources.txt`;
- preserve Make/CMake/source-list parity if a future reviewed change adds a
  new proof-owner binary or library source.

The draft also records preferred locations for future contributions:

| Contribution type | Preferred location |
| --- | --- |
| Public LDLT CSC proof case | `tests/test_ldlt_csc.c` with a `RUN_TEST(...)` entry. |
| Reused KKT/scaled-KKT/analysis-backed fixture | `tests/test_ldlt_csc_fixtures.h`. |
| Reused dense oracle or native-wrapper comparison helper | `tests/test_ldlt_csc_oracle_helpers.h`. |
| Reused supernode fixture/snapshot/factor-state helper | `tests/test_ldlt_csc_supernode_helpers.h`. |
| External-process dense-reference state or platform skip behavior | Keep local in `tests/test_ldlt_csc.c` unless a later boundary review approves extraction. |
| Broad random-matrix or solve residual helper | Keep local until a later artifact proves a tighter helper boundary. |

Validation guidance in the draft requires `make ldlt-csc-helper-guard` for
helper layout changes, `make source-list-check` for registration/source-list
concerns, and the full C gate after any `.c` or `.h` changes.

Day 10 changed planning artifacts only. No `.c`, `.h`, Makefile, CMake,
script, production source, public API, or internal API files changed for this
day.

Validation completed:

```sh
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

`make ldlt-csc-helper-guard` passed. `make source-list-check` passed with 49
library sources. `git diff --check` passed.

Day 11 handoff:

- Decide whether to promote the maintenance draft into
  `docs/maintainer_guide.md`, a test-local README, or both.
- Cross-link the final note from existing maintainer or testing documentation.
- Keep the final note aligned with `make ldlt-csc-helper-guard`.

## Day 11 Contributor Guidance Alignment

Day 11 promoted the selected-cluster maintenance guidance into the existing
maintainer-facing helper ownership surface.

| Field | Result |
| --- | --- |
| Maintainer doc updated | `docs/maintainer_guide.md` |
| Planning artifact added | `docs/planning/EPIC_16/SPRINT_185/artifacts/day11-contributor-guidance-alignment.md` |
| Test-local README | Not added; the maintainer guide already owns proof-ownership and helper-ownership interpretation. |
| Guard referenced | `make ldlt-csc-helper-guard` |
| Detailed provenance link | `docs/planning/EPIC_16/SPRINT_185/artifacts/day10-maintenance-invariants.md` |

The maintainer guide now documents that:

- `tests/test_ldlt_csc_fixtures.h` owns LDLT CSC family-local KKT,
  scaled-KKT, and analysis-backed two-pass fixture/setup helpers;
- `tests/test_ldlt_csc_oracle_helpers.h` owns LDLT CSC family-local dense
  oracles, symmetric-swap helpers, and native-wrapper comparison helpers;
- `tests/test_ldlt_csc_supernode_helpers.h` owns LDLT CSC family-local
  supernode fixtures, snapshots, dense-SPD setup, and factor-state comparison
  helpers;
- `tests/test_ldlt_csc.c` remains the registered LDLT CSC proof-owner binary;
- `main`, `RUN_TEST(...)` ordering, public test bodies, test names, fixture
  values, numerical tolerances, `_POSIX_C_SOURCE`, and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` ownership stay in
  `tests/test_ldlt_csc.c`;
- external-process dense-reference state and platform skip behavior remain
  local unless a later boundary review approves extraction;
- `make ldlt-csc-helper-guard` is the maintained selected-cluster guard after
  helper-layout changes.

Reviewed `tests/test_ldlt_csc.c` and the three Sprint 185 helper headers for
stale comments that conflict with the extracted layout. No conflicting stale
comments were found.

Day 11 changed documentation and planning files only. No `.c`, `.h`,
Makefile, CMake, script, production source, public API, or internal API files
changed for this day.

Validation completed:

```sh
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

`make ldlt-csc-helper-guard` passed. `make source-list-check` passed with 49
library sources. `git diff --check` passed.

Day 12 handoff:

- Run focused selected-cluster validation.
- Include `make ldlt-csc-helper-guard` and `make source-list-check`.
- Review the accumulated Sprint 185 diff for accidental solver behavior,
  fixture, tolerance, or scope changes before the Day 13 full gate.

## Day 12 Focused Cluster Validation

Day 12 ran the focused selected-cluster validation before the full Day 13
quality gate.

| Field | Result |
| --- | --- |
| Focused proof owner | `tests/test_ldlt_csc.c` / `test_ldlt_csc` |
| Focused artifact | `docs/planning/EPIC_16/SPRINT_185/artifacts/day12-focused-cluster-validation.md` |
| Selected-cluster guard | `make ldlt-csc-helper-guard` |
| Source-list guard | `make source-list-check` |

Current selected-cluster line counts:

| Path | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3469 |
| `tests/test_ldlt_csc_fixtures.h` | 145 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 149 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 134 |

Focused validation completed:

```sh
if [ -e build/test_ldlt_csc ]; then rm build/test_ldlt_csc; fi
make build/test_ldlt_csc
./build/test_ldlt_csc
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

`make build/test_ldlt_csc` passed after forcing a rebuild. `./build/test_ldlt_csc`
passed with 100 tests, 0 failures, 0 skips, and 3556 assertions. `make
ldlt-csc-helper-guard` passed. `make source-list-check` passed with 49 library
sources. `git diff --check` passed.

Diff review found the accumulated Sprint 185 changes remain limited to the
selected cluster, extracted helper headers, guard target/script, maintainer
guidance, and sprint artifacts. No `RUN_TEST(...)` ordering changes, public
test-body changes, solver source changes, production API changes,
fixture-value changes, or registration broadening were found.

Day 13 full validation command list:

```sh
make format
make lint
make test
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

Day 13 handoff:

- Run the full quality gate.
- Re-run the selected-cluster guard and source-list check.
- Record final validation notes and any cleanup required for review readiness.

## Day 13 Full Quality Gate

Day 13 ran the repository-level quality gate for the accumulated Sprint 185
LDLT CSC review-surface extraction.

| Field | Result |
| --- | --- |
| Full-gate artifact | `docs/planning/EPIC_16/SPRINT_185/artifacts/day13-full-quality-gate.md` |
| Selected proof owner | `tests/test_ldlt_csc.c` / `test_ldlt_csc` |
| Selected-cluster guard | `make ldlt-csc-helper-guard` |
| Source-list guard | `make source-list-check` |

Full validation completed:

```sh
make format
make lint
make test
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

Results:

- `make format`: passed.
- `make lint`: passed, including strict warning compile, clang-tidy, and
  cppcheck.
- `make test`: passed; the suite ended with `All tests passed.`
- `make ldlt-csc-helper-guard`: passed.
- `make source-list-check`: passed with 49 library sources.
- `git diff --check`: passed.

`test_ldlt_csc` passed inside the full test suite with 100 tests, 0 failures,
0 skips, and 3556 assertions.

Current selected-cluster line counts after formatting:

| Path | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3469 |
| `tests/test_ldlt_csc_fixtures.h` | 145 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 149 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 134 |

No unresolved formatting, lint, test, guard, source-list, or whitespace
failures remain after Day 13. No Day 13 cleanup edits were required beyond
formatting normalization from `make format`.

Day 14 handoff:

- Review all Sprint 185 artifacts and working notes against items 185.1
  through 185.6.
- Confirm the final selected-cluster extraction, guard, maintainer guidance,
  and validation evidence.
- Prepare the review-ready handoff for the retrospective and PR description.

## Day 14 Review-Ready Handoff

Day 14 closed Sprint 185 by reviewing the accumulated diff against
project-plan items 185.1 through 185.6 and preparing retrospective-ready
handoff notes.

| Item | Outcome |
| --- | --- |
| 185.1 Cluster Selection | Selected exactly one large review surface: `tests/test_ldlt_csc.c`. |
| 185.2 Extraction Design | Designed three family-local helper-header boundaries and a no-behavior-change validation contract. |
| 185.3 Mechanical Extraction | Extracted supernode, fixture/setup, and oracle/native-wrapper helpers into family-local headers included by `tests/test_ldlt_csc.c`. |
| 185.4 Drift Guard Update | Added `scripts/check_ldlt_csc_helper_guard.sh` and `make ldlt-csc-helper-guard`. |
| 185.5 Maintenance Note | Drafted `day10-maintenance-invariants.md` and promoted the guidance into `docs/maintainer_guide.md`. |
| 185.6 Validation | Ran focused selected-cluster validation and the full C quality gate. |

Final review-surface result:

| Path | Lines |
| --- | ---: |
| `tests/test_ldlt_csc.c` | 3469 |
| `tests/test_ldlt_csc_fixtures.h` | 145 |
| `tests/test_ldlt_csc_oracle_helpers.h` | 149 |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 |
| `scripts/check_ldlt_csc_helper_guard.sh` | 134 |

`tests/test_ldlt_csc.c` was reduced from the Day 3 baseline of 3915 lines to
3469 lines, a 446-line reduction in the selected proof-owner file.

Final file layout:

- `tests/test_ldlt_csc.c` remains the registered LDLT CSC proof-owner binary
  and owns public test bodies, `main`, `RUN_TEST(...)` ordering, external
  dense-reference state, and remaining broad/stateful helpers.
- `tests/test_ldlt_csc_supernode_helpers.h` owns family-local supernode
  fixture, snapshot, dense-SPD, and factor-state comparison helpers.
- `tests/test_ldlt_csc_fixtures.h` owns family-local KKT, scaled-KKT, and
  analysis-backed two-pass fixture/setup helpers.
- `tests/test_ldlt_csc_oracle_helpers.h` owns family-local dense-oracle,
  symmetric-swap, and native-wrapper comparison helpers.
- `scripts/check_ldlt_csc_helper_guard.sh` owns selected-cluster helper
  presence, include ownership, and registration-boundary checks.
- `docs/maintainer_guide.md` owns discoverable maintainer guidance for the new
  helper ownership split.

Scope review found no production source, public API, internal solver API,
CMake registration, library source manifest, or new test binary changes. The
only Makefile change is the selected-cluster guard target. The accumulated
diff remains limited to the selected LDLT CSC test cluster and helper headers,
the selected-cluster guard script/target, maintainer guidance, and Sprint 185
planning artifacts.

No stale TODOs, unresolved open questions, untracked generated files, or
accidental scope expansion were found during Day 14 closeout. Deferred
candidates remain explicitly outside Sprint 185 scope:

- `tests/test_qr.c`;
- `tests/test_svd.c`;
- `tests/test_graph.c`;
- `tests/test_integration.c`;
- `tests/test_iterative.c`;
- `src/sparse_ldlt_csc.c`;
- Python/report/documentation large surfaces not selected for this sprint.

Day 14 closeout validation completed:

```sh
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

`make ldlt-csc-helper-guard` passed. `make source-list-check` passed with 49
library sources. `git diff --check` passed.

Retrospective inputs:

- Primary win: the selected proof-owner file is smaller while keeping the
  existing `test_ldlt_csc` binary, test names, test order, fixture values, and
  tolerances stable.
- Most useful guardrail: `make ldlt-csc-helper-guard` now mechanically
  protects the helper-header layout.
- Main residual risk: future contributors could still add one-off helpers back
  into `tests/test_ldlt_csc.c`; the maintainer guidance now documents where
  reusable fixture/oracle/supernode helpers belong.
- Validation baseline: focused selected-cluster validation and the full C gate
  both passed before closeout.

Reviewers can verify the sprint with:

```sh
make format
make lint
make test
make ldlt-csc-helper-guard
make source-list-check
git diff --check
```

Created `artifacts/day14-review-ready-handoff.md`.

## Inherited Guardrails

- Select exactly one large test/source cluster for Sprint 185.
- Preserve behavior, fixture data, public APIs, numerical tolerances, and test
  names unless an artifact explicitly records a reviewed no-behavior-change
  rationale.
- Prefer test helper or proof-owner extraction over implementation extraction
  unless source-file ownership is clearer and lower risk.
- Do not use review-surface reduction as evidence for performance,
  correctness expansion, broader solver coverage, package/platform support,
  or state-of-the-art claims.
- Update Make and CMake registration together for any new test binary.
- Update Makefile, CMake, and `build-metadata/library_sources.txt` together
  for any new library source file.
- If `.c` or `.h` files change, run `make format && make lint && make test`.
- If only planning files change, `git diff --check` is sufficient for that
  day.

## Initial Risks And Open Questions

| ID | Topic | Risk or question | Day 1 disposition |
| --- | --- | --- | --- |
| S185-RISK-01 | Candidate selection | The largest files are not automatically the lowest-risk extraction targets. | Day 2 must score review cost and refactor risk separately. |
| S185-RISK-02 | Test registration | Creating a new test binary requires Make/CMake registration and parity validation. | Keep registration inventory central in Day 2-5 artifacts. |
| S185-RISK-03 | Library source extraction | New library `.c` files touch Make, CMake, and source manifest ordering. | Prefer test-only extraction unless library seam is clearly lower risk. |
| S185-RISK-04 | Behavior drift | Moving helpers can accidentally alter static state, fixture initialization, tolerances, or test ordering. | Define no-behavior-change validation before mechanical extraction. |
| S185-RISK-05 | Over-broad cleanup | Large files invite opportunistic refactors. | Limit Sprint 185 to one cluster and one planned extraction design. |
| S185-RISK-06 | Generated artifacts | Focused tests or report tooling may generate local build/report files. | Keep generated output unstaged and record validation commands. |

## Daily Log

### Day 1: Review-Surface Intake

- Re-read the Sprint 185 section of
  `docs/planning/EPIC_16/PROJECT_PLAN.md`.
- Confirmed Sprint 185 closes Sprint 177 residual `S177-R13` for large
  test/source review-surface reduction.
- Reviewed Sprint 177 Day 4 repository surface inventory and Day 7 target
  selection for the Sprint 185 handoff.
- Reviewed current large-file line counts after the Sprint 184 merge.
- Inventoried candidate large test/source/tooling surfaces without selecting a
  cluster.
- Reviewed existing registration and guard surfaces:
  - `Makefile` `LIB_SRCS`, `TEST_SRCS`, and focused build/test targets;
  - `CMakeLists.txt` `add_library(...)` and `add_sparse_test(...)`;
  - `build-metadata/library_sources.txt`;
  - `scripts/check_library_sources.py`;
  - `make source-list-check`;
  - `make quality-review-cmake-compile`;
  - `make format`, `make lint`, and `make test`.
- Recorded inherited guardrails, initial risks, and the Day 2 baseline
  handoff.
- Day 1 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day1-review-surface-intake.md`.

### Day 2: Large Surface Baseline

- Recomputed current large candidate line counts and static/function counts.
- Reviewed existing helper headers and build-registration guard surfaces.
- Scored candidates separately for review cost and refactor risk.
- Shortlisted `tests/test_ldlt_csc.c` and `tests/test_svd.c` for the Day 3
  cluster-selection decision, with `tests/test_graph.c` retained as a viable
  fallback.
- Kept `tests/test_qr.c`, `tests/test_integration.c`,
  `tests/test_iterative.c`, and `src/sparse_ldlt_csc.c` out of the top
  shortlist because their recent-change, breadth, allocation-failure, or
  implementation-registration risks are higher.
- Did not select the final Sprint 185 cluster; Day 3 should inspect the
  shortlisted seams in detail before choosing.
- Day 2 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day2-candidate-cluster-baseline.md`.

### Day 3: Selected Cluster Decision

- Selected exactly one Sprint 185 cluster: `tests/test_ldlt_csc.c`.
- Confirmed baseline evidence for the selected cluster:
  - 3915 current lines;
  - 130 static/function entries;
  - existing Makefile registration as `$(TESTDIR)/test_ldlt_csc.c`;
  - existing CMake registration as `add_sparse_test(test_ldlt_csc)`.
- Reviewed the selected cluster's responsibility groups, including alloc/free,
  row adjacency, supernode detection/extract/writeback, supernodal
  cross-checks, `from_sparse` and analysis-shim coverage, external dense
  references, permutation, validation, elimination, native kernel checks,
  symmetric swaps, solve, and inertia tests.
- Rejected `tests/test_svd.c` for this sprint because it is a good alternate
  but has lower review cost and broader full/partial SVD ownership.
- Rejected `tests/test_graph.c` as a fallback because graph/FM environment
  interactions are less direct-solver-local than the LDLT CSC helper seams.
- Deferred `tests/test_qr.c`, `tests/test_integration.c`,
  `tests/test_iterative.c`, and `src/sparse_ldlt_csc.c` for the higher risks
  recorded in Day 2.
- Defined the no-behavior-change contract and a Days 4-8 extraction checklist.
- Day 3 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day3-selected-cluster-decision.md`.

### Day 4: Helper Boundary Design

- Reviewed existing local helper-header style in:
  - `tests/test_qr_helpers.h`;
  - `tests/test_svd_helpers.h`;
  - `tests/test_direct_solver_helpers.h`;
  - `tests/test_chol_csc_supernodal_helpers.h`.
- Confirmed the selected cluster can stay on a header-only helper extraction
  path, preserving the existing `test_ldlt_csc` proof-owner binary.
- Designed three family-local helper boundaries:
  - `tests/test_ldlt_csc_supernode_helpers.h`;
  - `tests/test_ldlt_csc_fixtures.h`;
  - `tests/test_ldlt_csc_oracle_helpers.h`.
- Chose `tests/test_ldlt_csc_supernode_helpers.h` as the only planned Day 6
  first-pass extraction target.
- Recorded what must remain local in `tests/test_ldlt_csc.c`: `main`,
  `RUN_TEST(...)` ordering, public test bodies, chronological proof-owner
  comments, `_POSIX_C_SOURCE`, and `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`.
- Defined first-pass validation:
  `make build/test_ldlt_csc && ./build/test_ldlt_csc`, then
  `make format && make lint && make test` after C/H edits.
- Day 4 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day4-helper-boundary-design.md`.

### Day 5: Registration Guardrail Design

- Audited the selected cluster's registration in `Makefile` and
  `CMakeLists.txt`.
- Confirmed Day 6's planned `tests/test_ldlt_csc_supernode_helpers.h`
  extraction does not require:
  - a new `Makefile` `TEST_SRCS` entry;
  - a new `CMakeLists.txt` `add_sparse_test(...)` entry;
  - a new `Makefile` `LIB_SRCS` entry;
  - a new `CMakeLists.txt` library source entry;
  - a `build-metadata/library_sources.txt` change.
- Confirmed existing format coverage includes new test helper headers through
  `ALL_TEST_SRC = $(wildcard $(TESTDIR)/*.c) $(wildcard $(TESTDIR)/*.h)`.
- Confirmed existing lint coverage scans the tests tree through `cppcheck`
  and that focused rebuild compiles the included helper header.
- Recorded the Makefile header-dependency caveat: `build/test_ldlt_csc` must
  be removed before the focused Day 6 rebuild, and the full C gate should run
  from a clean build state.
- Recorded generated-file expectations and rollback criteria.
- Day 5 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day5-registration-guardrail-design.md`.

### Day 6: Initial Helper Extraction

- Created `tests/test_ldlt_csc_supernode_helpers.h` as the first family-local
  helper header for the selected LDLT CSC review surface.
- Moved `build_dense_ldlt_with_pivots`, `cm_idx`,
  `snapshot_supernode_state`, `ldlt_csc_factor_state_matches`, and
  `build_dense_spd` from `tests/test_ldlt_csc.c` into the helper header.
- Included the helper header from `tests/test_ldlt_csc.c`; `clang-format`
  normalized the local include block alphabetically.
- Preserved `main`, `RUN_TEST(...)` ordering, public test bodies, test names,
  fixture values, numerical tolerances, `_POSIX_C_SOURCE`, and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` behavior.
- Made no Makefile, CMake, library-source manifest, production source, public
  API, or internal API changes.
- Reduced `tests/test_ldlt_csc.c` from the Day 3 baseline of 3915 lines to
  3793 lines; the new helper header is 140 lines.
- Forced the focused rebuild and ran `./build/test_ldlt_csc`; it passed with
  100 tests, 0 failures, 0 skips, and 3556 assertions.
- Ran the required C gate after source/header edits: `make format`,
  `make lint`, and `make test`; all passed.
- Created `artifacts/day6-initial-helper-extraction.md`.

### Day 7: Fixture And Setup Extraction

- Created `tests/test_ldlt_csc_fixtures.h` as the family-local fixture/setup
  header for KKT and analysis-backed LDLT CSC tests.
- Moved `build_kkt_5x5`, `build_kkt_10x10`,
  `build_kkt_scaled_10x10`, and `s20_two_pass_indefinite_factor` from
  `tests/test_ldlt_csc.c` into the new helper header.
- Kept external dense-reference process state and assertions local in
  `tests/test_ldlt_csc.c` to avoid widening macro and platform-skip
  ownership.
- Preserved `main`, `RUN_TEST(...)` ordering, public test bodies, test names,
  fixture values, numerical tolerances, `_POSIX_C_SOURCE`, and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` behavior.
- Made no Makefile, CMake, library-source manifest, production source, public
  API, or internal API changes.
- Reduced `tests/test_ldlt_csc.c` to 3639 lines; the new fixture header is
  145 lines.
- Forced the focused rebuild and ran `./build/test_ldlt_csc`; it passed with
  100 tests, 0 failures, 0 skips, and 3556 assertions.
- Ran the required C gate after source/header edits: `make format`,
  `make lint`, and `make test`; all passed.
- Created `artifacts/day7-fixture-setup-extraction.md`.

### Day 8: Proof-Owner Cleanup

- Created `tests/test_ldlt_csc_oracle_helpers.h` as the family-local
  oracle/native-wrapper helper header.
- Moved `ldlt_lower_to_dense`, `dense_sym_swap`, `dense_lower_equal`,
  `build_ldlt_from_triples`, `ldlt_column_nonzeros_match`,
  `ldlt_factorizations_match`, and `check_native_matches_wrapper` from
  `tests/test_ldlt_csc.c` into the new helper header.
- Preserved `main`, `RUN_TEST(...)` ordering, public test bodies, test names,
  fixture values, numerical tolerances, `_POSIX_C_SOURCE`, and
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` behavior.
- Kept external dense-reference process state and assertions local in
  `tests/test_ldlt_csc.c`.
- Made no Makefile, CMake, library-source manifest, production source, public
  API, or internal API changes.
- Reduced `tests/test_ldlt_csc.c` to 3469 lines; the new oracle helper header
  is 149 lines.
- Forced the focused rebuild and ran `./build/test_ldlt_csc`; it passed with
  100 tests, 0 failures, 0 skips, and 3556 assertions.
- Ran `make source-list-check`; it passed with 49 library sources.
- Ran the required C gate after source/header edits: `make format`,
  `make lint`, and `make test`; all passed.
- Created `artifacts/day8-proof-owner-cleanup.md`.

### Day 9: Drift Guard Update

- Added `scripts/check_ldlt_csc_helper_guard.sh` as the selected-cluster guard
  for the three Sprint 185 LDLT CSC helper headers.
- Added the `make ldlt-csc-helper-guard` target.
- Guarded the intended ownership model:
  `tests/test_ldlt_csc.c` remains registered in Make/CMake and includes
  `tests/test_ldlt_csc_fixtures.h`,
  `tests/test_ldlt_csc_oracle_helpers.h`, and
  `tests/test_ldlt_csc_supernode_helpers.h` exactly once.
- Guarded against accidental helper-header registration in Makefile, CMake,
  or `build-metadata/library_sources.txt`.
- Made no CMake, library-source manifest, production source, public API,
  internal API, `.c`, or `.h` changes for Day 9.
- Ran `bash -n scripts/check_ldlt_csc_helper_guard.sh`; it passed.
- Ran `make ldlt-csc-helper-guard`; it passed.
- Ran `make source-list-check`; it passed with 49 library sources.
- Created `artifacts/day9-drift-guard-update.md`.

### Day 10: Maintenance Invariants

- Drafted `artifacts/day10-maintenance-invariants.md` for the selected LDLT
  CSC helper-header layout.
- Documented current file ownership for `tests/test_ldlt_csc.c`,
  `tests/test_ldlt_csc_fixtures.h`,
  `tests/test_ldlt_csc_oracle_helpers.h`,
  `tests/test_ldlt_csc_supernode_helpers.h`, and
  `scripts/check_ldlt_csc_helper_guard.sh`.
- Documented invariants for proof-owner registration, helper-header internal
  linkage, Make/CMake/source-list boundaries, fixture ownership, and
  validation commands.
- Documented where future LDLT CSC proof cases, reusable fixtures, dense
  oracles, native-wrapper helpers, supernode helpers, external dense-reference
  state, and broader random/residual helpers should live.
- Identified `docs/maintainer_guide.md` as the likely long-term policy surface
  for Day 11 alignment.
- Made no `.c`, `.h`, Makefile, CMake, script, production source, public API,
  or internal API changes for Day 10.
- Ran `make ldlt-csc-helper-guard`; it passed.
- Ran `make source-list-check`; it passed with 49 library sources.
- Created `artifacts/day10-maintenance-invariants.md`.

### Day 11: Contributor Guidance Alignment

- Updated `docs/maintainer_guide.md` in the existing test fixture/helper
  ownership section.
- Documented the Sprint 185 LDLT CSC helper-header ownership split for:
  `tests/test_ldlt_csc_fixtures.h`,
  `tests/test_ldlt_csc_oracle_helpers.h`, and
  `tests/test_ldlt_csc_supernode_helpers.h`.
- Documented that `tests/test_ldlt_csc.c` remains the registered LDLT CSC
  proof-owner binary and keeps `main`, `RUN_TEST(...)` ordering, public test
  bodies, test names, fixture values, numerical tolerances,
  `_POSIX_C_SOURCE`, and `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`.
- Cross-linked the detailed Day 10 maintenance-invariants artifact as the
  provenance record for helper-placement rules.
- Reviewed `tests/test_ldlt_csc.c` and the three Sprint 185 helper headers for
  stale comments that conflict with the extracted layout; none were found.
- Made no `.c`, `.h`, Makefile, CMake, script, production source, public API,
  or internal API changes for Day 11.
- Ran `make ldlt-csc-helper-guard`; it passed.
- Ran `make source-list-check`; it passed with 49 library sources.
- Created `artifacts/day11-contributor-guidance-alignment.md`.

### Day 12: Focused Cluster Validation

- Forced `build/test_ldlt_csc` to rebuild.
- Ran `make build/test_ldlt_csc`; it passed.
- Ran `./build/test_ldlt_csc`; it passed with 100 tests, 0 failures, 0 skips,
  and 3556 assertions.
- Ran `make ldlt-csc-helper-guard`; it passed.
- Ran `make source-list-check`; it passed with 49 library sources.
- Ran `git diff --check`; it passed.
- Reviewed the accumulated Sprint 185 diff and confirmed it remains limited to
  the selected cluster, extracted helper headers, guard target/script,
  maintainer guidance, and sprint artifacts.
- Found no `RUN_TEST(...)` ordering changes, public test-body changes, solver
  source changes, production API changes, fixture-value changes, or
  registration broadening.
- Recorded the Day 13 full validation command list.
- Created `artifacts/day12-focused-cluster-validation.md`.

### Day 13: Full Quality Gate

- Ran `make format`; it passed.
- Ran `make lint`; it passed, including strict warning compile, clang-tidy,
  and cppcheck.
- Ran `make test`; it passed and ended with `All tests passed.`
- Confirmed `test_ldlt_csc` passed inside the full suite with 100 tests, 0
  failures, 0 skips, and 3556 assertions.
- Ran `make ldlt-csc-helper-guard`; it passed.
- Ran `make source-list-check`; it passed with 49 library sources.
- Ran `git diff --check`; it passed.
- Confirmed current selected-cluster line counts remain:
  `tests/test_ldlt_csc.c` at 3469,
  `tests/test_ldlt_csc_fixtures.h` at 145,
  `tests/test_ldlt_csc_oracle_helpers.h` at 149,
  `tests/test_ldlt_csc_supernode_helpers.h` at 140, and
  `scripts/check_ldlt_csc_helper_guard.sh` at 134.
- Recorded no unresolved formatting, lint, test, guard, source-list, or
  whitespace failures after the full gate.
- Created `artifacts/day13-full-quality-gate.md`.

### Day 14: Review-Ready Handoff

- Reviewed Sprint 185 outcomes against project-plan items 185.1 through 185.6.
- Confirmed the final selected-cluster extraction, guard, maintainer guidance,
  and validation evidence are documented.
- Recorded the final selected-cluster line counts:
  `tests/test_ldlt_csc.c` at 3469,
  `tests/test_ldlt_csc_fixtures.h` at 145,
  `tests/test_ldlt_csc_oracle_helpers.h` at 149,
  `tests/test_ldlt_csc_supernode_helpers.h` at 140, and
  `scripts/check_ldlt_csc_helper_guard.sh` at 134.
- Confirmed `tests/test_ldlt_csc.c` was reduced from the Day 3 baseline of
  3915 lines to 3469 lines.
- Confirmed no production source, public API, internal solver API, CMake
  registration, library source manifest, or new test binary changes were made.
- Confirmed no stale TODOs, unresolved open questions, untracked generated
  files, or accidental scope expansion were found during closeout.
- Recorded deferred follow-up candidates separately from completed Sprint 185
  scope.
- Ran `make ldlt-csc-helper-guard`; it passed.
- Ran `make source-list-check`; it passed with 49 library sources.
- Ran `git diff --check`; it passed.
- Created `artifacts/day14-review-ready-handoff.md`.
