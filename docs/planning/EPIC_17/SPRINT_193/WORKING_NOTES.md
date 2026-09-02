# Sprint 193 Working Notes: Selected Large Review-Surface Reduction

## Sprint Goal

Reduce one high-risk implementation/test review surface while preserving
behavior and validation.

## Day 1: Review-Surface Intake

### Scope Trace

| Epic item | Day 1 intake interpretation |
| --- | --- |
| 193.1 Candidate Ranking | Rank large source/test candidates by size, helper density, algorithm risk, fixture ownership, current tests, source-list coupling, and user-facing importance. |
| 193.2 Cluster Selection | Select exactly one cluster and record no-behavior-change invariants before extraction work starts. |
| 193.3 Extraction Design | Design helper/source boundaries, cleanup ownership, override restoration, registration behavior, and guard scope before editing code. |
| 193.4 Implementation | Extract only the selected cluster with minimal behavior-preserving edits. |
| 193.5 Guard And Docs | Add focused ownership guard coverage and maintainer documentation for the new boundary. |
| 193.6 Validation | Run focused tests, source-list checks, full `make format && make lint && make test`, and CMake parity if source lists change. |

### Baseline Evidence Read

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Sprint 193 is allocated 168 hours to reduce exactly one high-risk implementation/test review surface while preserving behavior and validation. |
| `docs/planning/EPIC_17/SPRINT_187/artifacts/day5-gap-ranking-and-feasibility.md` | Selected review-surface reduction is the Sprint 193 closure target and should be complete only if limited to one cluster with explicit no-behavior-change invariants. |
| `docs/planning/EPIC_16/SPRINT_185/RETROSPECTIVE.md` | The prior extraction pattern selected one large proof-owner file, extracted family-local helper headers, kept registration stable, added a guard, updated maintainer docs, and ran the full C/H gate. |
| `docs/planning/EPIC_16/SPRINT_185/artifacts/day3-selected-cluster-decision.md` | Sprint 185 selected `tests/test_ldlt_csc.c`, defined test-order/fixture/tolerance/global-override invariants, and deferred QR, SVD, graph, integration, iterative, and production-source candidates. |
| `scripts/check_ldlt_csc_helper_guard.sh` | Existing guard pattern checks proof-owner registration, helper-header existence, include ownership, and absence from Make/CMake/library registration. |
| `scripts/check_library_sources.py` | Library source-list parity is enforced across `build-metadata/library_sources.txt`, `Makefile LIB_SRCS`, and the CMake `add_library` block. |
| `Makefile` | Test registration is explicit in `TEST_SRCS`; library, benchmark, example, formatting, lint, source-list, CMake parity, and full quality targets are already available. |
| `CMakeLists.txt` | Library source registration and test registration are separate; adding production sources or new test binaries requires coordinated CMake changes. |
| `docs/maintainer_guide.md` | Maintainer docs already describe the LDLT CSC helper boundary and quality-review/CMake validation paths that Sprint 193 can reuse for a new selected cluster. |

### Current Large File Inventory

Raw source/test/script line-count scan, excluding matrix data files:

| Rank | File | Lines | Approx. static/function entries | Initial Day 1 interpretation |
| ---: | --- | ---: | ---: | --- |
| 1 | `tests/test_qr.c` | 3970 | 78 | Largest remaining C test surface; QR has recent proof-owner work and selected comparison coverage, so extraction could be valuable but must avoid solver behavior drift. |
| 2 | `tests/test_ldlt_csc.c` | 3469 | 110 | Still large after Sprint 185; helper seams remain, but prior sprint already reduced this cluster and further work risks diminishing returns. |
| 3 | `tests/test_integration.c` | 3279 | 54 | Cross-solver integration surface with broad behavior exposure; high review value but high scope-control risk. |
| 4 | `tests/test_svd.c` | 3029 | 85 | Large numerical test surface with existing helper-header precedent; potentially good candidate if a bounded cluster can be isolated. |
| 5 | `tests/test_ldlt.c` | 3006 | 92 | Large direct-solver test surface; behavior and tolerance preservation would be central. |
| 6 | `tests/test_etree.c` | 2962 | 102 | Large graph/elimination-tree test surface with high helper density; may support fixture/helper extraction. |
| 7 | `tests/test_iterative.c` | 2929 | 86 | Large iterative-solver surface with allocation/failure and convergence behavior; higher risk than pure helper movement. |
| 8 | `tests/test_graph.c` | 2764 | 65 | Large graph/FM/coarsening proof surface; fixture extraction may be possible, but behavior can be environment-sensitive. |
| 9 | `tests/test_chol_csc.c` | 2554 | 108 | Large CSC Cholesky test surface; high helper density and direct-solver relevance. |
| 10 | `tests/test_chol_csc_supernodal.c` | 2504 | 72 | Large supernodal test surface with an existing helper header, likely candidate for focused helper-boundary reduction. |
| 11 | `tests/test_reorder_nd.c` | 2304 | not counted in Day 1 table | Large ordering proof surface; source-list and fixture ownership need review before selection. |
| 12 | `tests/test_eigs.c` | 2155 | not counted in Day 1 table | Large eigensolver test surface; convergence semantics increase behavior-preservation risk. |
| 13 | `src/sparse_ldlt_csc.c` | 2095 | 26 | Production implementation surface; high review value but higher source-list and behavior risk. |
| 14 | `tests/test_colamd.c` | 2017 | not counted in Day 1 table | Large ordering/test surface; candidate only if helper seams are clearer than top-ranked files. |
| 15 | `tests/test_ilu.c` | 1974 | not counted in Day 1 table | Large preconditioner test surface; behavior/tolerance risk needs scoring. |
| 16 | `scripts/normalize_report_index.py` | 1836 | Python owner | Large script surface already touched by recent sprints; not the default Sprint 193 C/test review-surface candidate. |
| 17 | `tests/test_lu_csr.c` | 1806 | not counted in Day 1 table | Large CSR LU test surface; possible helper extraction candidate. |
| 18 | `src/sparse_lu_csr.c` | 1594 | 9 | Production implementation surface; source-list unchanged if internal-only movement, but behavior risk is high. |
| 19 | `src/sparse_ldlt.c` | 1535 | 8 | Production direct-solver surface; broad behavior exposure. |
| 20 | `src/sparse_iterative.c` | 1503 | 11 | Production iterative-solver surface; convergence and failure behavior make extraction higher risk. |
| 21 | `src/sparse_qr.c` | 1448 | 9 | Production QR surface; high public behavior risk. |

### Existing Helper/Header Precedent

| Helper surface | Lines | Day 1 implication |
| --- | ---: | --- |
| `tests/test_svd_partial_helpers.h` | 1519 | Large helper-header precedent exists, but moving more into a large helper should not hide review complexity. |
| `tests/test_qr_helpers.h` | 343 | QR already has a family-local helper boundary that may support or complicate further QR test extraction. |
| `tests/test_iterative_handle_helpers.h` | 289 | Iterative helper ownership exists, but failure-hook and handle lifetime behavior need caution. |
| `tests/test_svd_helpers.h` | 257 | SVD already uses helper boundaries, making it a plausible candidate if one cohesive cluster remains in `tests/test_svd.c`. |
| `tests/test_chol_csc_supernodal_helpers.h` | 255 | Cholesky supernodal helper precedent may support a Sprint 193 extraction with focused ownership. |
| `tests/test_solver_helpers.h` | 202 | Shared solver helper exists; Sprint 193 should avoid turning a selected extraction into broad shared-helper churn. |
| `tests/test_graph_fixtures.h` | 195 | Graph fixture extraction precedent exists. |
| `tests/test_integration_fixtures.h` | 169 | Integration fixture extraction precedent exists, but cross-solver scope remains broad. |
| `tests/test_ldlt_csc_oracle_helpers.h` | 151 | Sprint 185 helper boundary remains a working model. |
| `tests/test_svd_partial_shared_helpers.h` | 148 | SVD partial helper sharing exists but may already be at a review-surface limit. |
| `tests/test_ldlt_csc_fixtures.h` | 145 | Sprint 185 fixture extraction model. |
| `tests/test_ldlt_csc_supernode_helpers.h` | 140 | Sprint 185 supernode extraction model. |

### Source-List and Registration Owners

| Surface | Owner | Day 1 notes |
| --- | --- | --- |
| Library source manifest | `build-metadata/library_sources.txt` | Source-list check treats this as the expected ordered library source list. |
| Make library sources | `Makefile` `LIB_SRCS` | Must match the manifest exactly if production `.c` files are added, removed, or moved into new compilation units. |
| CMake library sources | `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)` | Must match the manifest for library-source parity. |
| Make test sources | `Makefile` `TEST_SRCS` | Existing tests are explicitly registered. New proof-owner binaries require registration and parity review. |
| CMake test registration | `CMakeLists.txt` `add_sparse_test(...)` | New test binaries require coordinated CMake registration. Header-only test helper extraction can avoid this churn. |
| Benchmark sources | `Makefile` `BENCH_SRCS`, `CMakeLists.txt` benchmark executables | Out of default scope unless the selected cluster is a benchmark file. |
| Format/lint owners | `Makefile` `format`, `format-check`, `lint` | Any `.c` or `.h` extraction must pass formatting and lint. |
| Full test owner | `Makefile` `test` | Required after C/H behavior-preserving extraction. |
| CMake parity owner | `make quality-review-cmake-compile` / `make quality-review-cmake` | Required if source-list or test registration changes. |
| Focused guard precedent | `make ldlt-csc-helper-guard` | Pattern to reuse for selected helper-header registration and boundary ownership. |

### Initial Candidate Cluster List

| Candidate | Evidence value | Initial risk | Day 1 disposition |
| --- | --- | --- | --- |
| `tests/test_qr.c` selected helper cluster | Highest remaining test-file line count; QR has user-facing importance and existing `tests/test_qr_helpers.h`. | Recent QR comparison work and solve semantics make tolerance/API drift risky. | Strong Day 2 candidate if a single helper cluster can be isolated. |
| `tests/test_svd.c` selected helper cluster | Large file with existing `tests/test_svd_helpers.h`; likely review-surface payoff without production source-list changes. | Numerical tolerance and full/partial SVD overlap need careful boundary selection. | Strong Day 2 candidate. |
| `tests/test_chol_csc_supernodal.c` selected helper cluster | Large, focused supernodal proof surface with existing helper header and direct-solver relevance. | Supernodal fixture and tolerance behavior can be subtle. | Strong Day 2 candidate. |
| `tests/test_chol_csc.c` selected helper cluster | High helper density and direct-solver relevance; likely fixture/oracle extraction seams. | Direct solver behavior and public Cholesky coverage are user-facing. | Day 2 candidate. |
| `tests/test_etree.c` selected fixture/helper cluster | High helper density and contained elimination-tree behavior. | Graph/order semantics may be subtle but less public-facing than solver kernels. | Day 2 candidate. |
| `tests/test_integration.c` selected fixture cluster | Large cross-solver review surface; existing integration fixtures header. | Too broad and easy to turn into multi-cluster cleanup. | Candidate but likely lower priority unless a very small cluster emerges. |
| `tests/test_iterative.c` selected helper cluster | Large file with existing iterative helper header. | Allocation failure, convergence, and handle lifetime behavior make cleanup risk higher. | Candidate with caution. |
| Further `tests/test_ldlt_csc.c` extraction | Existing guard and helper boundaries from Sprint 185 lower setup cost. | Prior sprint already reduced the file; further extraction may be lower payoff and higher risk around remaining local tests. | Deferred unless Day 2 ranking shows a uniquely cohesive remaining seam. |
| Production `src/*.c` extraction | Could reduce implementation review surface directly. | Source-list, public behavior, ABI expectations, and validation risk are much higher. | Default defer for Sprint 193 unless test-only candidates prove unsuitable. |

### Initial Behavior-Preservation Constraints

- Preserve public API, public headers, exported symbols, status codes, and
  documented solver behavior.
- Preserve test names, `RUN_TEST(...)` ordering, assertion count semantics
  where practical, fixture values, random seeds, tolerances, skip behavior,
  and diagnostic wording.
- Preserve cleanup ordering and ownership for allocations, temporary files,
  external-helper state, and fixture buffers.
- Restore process-global overrides or registration state before any assertion
  macro or early-return path can exit.
- Keep generated artifacts, build outputs, and report outputs out of source
  control unless an existing corpus policy explicitly requires them.
- Prefer header-only, family-local test helper extraction when it avoids new
  Make/CMake registration and source-list drift.
- If a new `.c` source or test binary is selected, update Make, CMake, and
  any manifest/guard owner together.
- Do not change numerical algorithms, tolerances, baseline expected values, or
  comparison policy as part of review-surface reduction.

### Initial Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Selecting too broad a cluster | Broad cleanup would be hard to review and could leave several gaps only partially closed. | Day 2 must select one cohesive cluster and reject adjacent cleanup explicitly. |
| Helper extraction hides complexity | Moving code into a large helper header can reduce one file while creating a new opaque surface. | Measure both original-file reduction and new helper size; add maintainer ownership notes. |
| Source-list drift | New `.c` files or test binaries can compile locally while missing Make/CMake parity. | Use `make source-list-check` and CMake parity when source lists change. |
| Behavior drift through cleanup edits | Cleanup and early-return paths can change observable behavior. | Define invariants before edits and add focused cleanup/restoration tests where needed. |
| Process-global state contamination | Test failures can leave overrides set for subsequent tests. | Restore overrides before assertion macros or early returns; use local status variables when needed. |
| Formatting churn obscures review | Broad `make format` can touch unrelated code if not scoped carefully. | Inspect format diff and keep implementation edits tightly scoped. |
| Duplicate helper ownership | Shared helpers can become dumping grounds across solver families. | Prefer selected family-local ownership unless an existing shared helper already owns the abstraction. |

### Day 1 Validation

Source and planning checks:

```sh
git status --short --branch
sed -n '235,267p' docs/planning/EPIC_17/PROJECT_PLAN.md
sed -n '1,90p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
rg --files src include tests benchmarks scripts -g '*.c' -g '*.h' -g '*.py' -g '*.sh' | xargs wc -l | sort -nr | head -n 45
for f in tests/test_qr.c tests/test_ldlt_csc.c tests/test_integration.c tests/test_svd.c tests/test_ldlt.c tests/test_etree.c tests/test_iterative.c tests/test_graph.c tests/test_chol_csc.c tests/test_chol_csc_supernodal.c src/sparse_ldlt_csc.c src/sparse_lu_csr.c src/sparse_ldlt.c src/sparse_iterative.c src/sparse_qr.c; do printf '%s\t' "$f"; rg -n '^(static[[:space:]]+)?[A-Za-z_][A-Za-z0-9_ *]+[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\([^;]*\)[[:space:]]*\{' "$f" | wc -l; done
find tests -maxdepth 1 -name '*helpers*.h' -o -name '*fixture*.h' -o -name '*oracle*.h' | sort | xargs wc -l | sort -nr
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_187/artifacts/day5-gap-ranking-and-feasibility.md
sed -n '1,220p' docs/planning/EPIC_16/SPRINT_185/artifacts/day3-selected-cluster-decision.md
sed -n '1,220p' scripts/check_ldlt_csc_helper_guard.sh
sed -n '1,220p' scripts/check_library_sources.py
git diff --check
```

Day 1 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

### Day 2 Questions

1. Which single cluster has the best combination of review-surface reduction,
   behavior-preservation confidence, and focused validation coverage?
2. Should Sprint 193 prioritize the largest remaining test file
   (`tests/test_qr.c`) or a smaller file with clearer helper seams such as
   `tests/test_svd.c` or `tests/test_chol_csc_supernodal.c`?
3. Is a header-only helper extraction sufficient, or does the selected cluster
   justify a new compiled helper/source unit with Make/CMake registration?
4. Which invariants must be proven before moving code for the selected
   cluster: tolerances, fixture values, cleanup paths, global overrides,
   registration, diagnostics, or test order?
5. What guard shape should Sprint 193 add so future contributors cannot
   accidentally bypass the selected helper boundary?

## Day 2: Candidate Ranking

### Ranking Method

Scores use a 1-5 scale:

- 5 = strongest positive value or highest risk;
- 3 = meaningful but bounded;
- 1 = low value or low risk.

For payoff, helper cohesion, current coverage, and sprint fit, higher is
better. For algorithm risk, cleanup risk, and registration risk, higher means
more risk.

### Ranked Candidate Matrix

| Rank | Candidate cluster | Payoff | Helper cohesion | Current coverage | User-facing importance | Algorithm risk | Cleanup/global-state risk | Registration risk | Sprint fit | Day 2 decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `tests/test_qr.c` external dense-reference rank/nullspace/threshold block | 5 | 5 | 4 | 4 | 3 | 2 | 1 | 5 | Recommended selected cluster for Day 3 confirmation. |
| 2 | `tests/test_svd.c` external dense-reference fixture/helper block | 3 | 5 | 4 | 4 | 3 | 2 | 1 | 4 | Alternate if QR proves too coupled. |
| 3 | `tests/test_chol_csc_supernodal.c` dense backend/env-contract block | 3 | 4 | 4 | 4 | 3 | 4 | 1 | 3 | Alternate with higher environment cleanup risk. |
| 4 | `tests/test_chol_csc.c` external dense-reference block | 3 | 4 | 4 | 4 | 3 | 2 | 1 | 3 | Plausible but lower payoff than QR. |
| 5 | `tests/test_etree.c` selected fixture/helper cluster | 3 | 3 | 4 | 3 | 3 | 2 | 1 | 3 | Defer until a more cohesive seam is identified. |
| 6 | `tests/test_integration.c` fixture/lifecycle cluster | 4 | 3 | 4 | 5 | 4 | 4 | 1 | 2 | Defer because cross-solver scope is too broad for this sprint. |
| 7 | `tests/test_iterative.c` CG/GMRES helper cluster | 4 | 3 | 4 | 5 | 4 | 5 | 1 | 2 | Defer because convergence and failure-path cleanup risk are high. |
| 8 | Further `tests/test_ldlt_csc.c` helper extraction | 3 | 4 | 5 | 4 | 3 | 3 | 1 | 2 | Defer because Sprint 185 already closed the highest-value LDLT CSC extraction. |
| 9 | Production `src/*.c` extraction | 5 | 2 | 4 | 5 | 5 | 4 | 5 | 1 | Defer by default; too much source-list and public-behavior risk for Sprint 193. |

### Recommended Cluster

Day 2 recommends the `tests/test_qr.c` external dense-reference
rank/nullspace/threshold block for Day 3 selection.

The recommended block currently includes:

- the external QR basis reference reader near the top of `tests/test_qr.c`;
- the external QR threshold reference reader near the top of `tests/test_qr.c`;
- `test_qr_external_dense_reference_rank1_4x3_nullspace_projector`;
- `test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector`;
- `test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector`;
- the local `make_rankdef_wide_3x5` fixture helper;
- `test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace`;
- `test_qr_external_dense_reference_rank_threshold_diag4_family`;
- `test_qr_external_dense_reference_rank_threshold_diag4_scaled_family`;
- `test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family`;
- `test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family`;
- the existing `RUN_TEST(...)` registration block in `tests/test_qr.c`, which
  should remain in the proof-owner file.

### Selection Rationale

| Criterion | Evidence |
| --- | --- |
| Size payoff | `tests/test_qr.c` is currently the largest C test file at 3970 lines. The external dense-reference block spans a large contiguous section in the rank/nullspace area rather than scattered one-off helpers. |
| Cohesion | The block shares one Python dense-reference helper, the same Windows skip policy, projector/rank-threshold semantics, QR rank/nullspace assertions, and QR fixture helpers. |
| Behavior-preservation path | The existing `test_qr` binary, `RUN_TEST(...)` entries, public solver calls, tolerance values, fixture keys, and diagnostic strings can remain unchanged while static test bodies move into a family-local helper header. |
| Registration risk | A header-only extraction can avoid Make/CMake test registration changes and avoid library source-list changes. |
| Guard fit | Sprint 185's helper guard pattern can be adapted to check the QR proof-owner registration, helper-header include, and absence from standalone Make/CMake/library registration. |
| Validation fit | Focused validation can run `make build/test_qr` and `./build/test_qr`, followed by `make format && make lint && make test` after C/H changes. |
| User-facing importance | QR rank, nullspace, threshold, and external dense-reference evidence directly support numerical credibility while remaining test-only. |

### Recommended No-Behavior-Change Invariants

- Preserve all QR external dense-reference fixture keys and Python command
  strings exactly.
- Preserve Windows skip behavior and messages.
- Preserve expected rank/nullity values, threshold values, perturbation values,
  tolerance constants, projector layout, and residual/orthogonality assertions.
- Preserve test function names and `RUN_TEST(...)` ordering in
  `tests/test_qr.c`.
- Preserve `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` placement and
  `_POSIX_C_SOURCE` behavior.
- Preserve the existing `test_qr` proof-owner binary without adding a new test
  executable.
- Keep the new helper header out of `Makefile`, `CMakeLists.txt`, and
  `build-metadata/library_sources.txt` source registration.

### Alternatives and Rejections

| Candidate | Decision | Reason |
| --- | --- | --- |
| `tests/test_svd.c` external dense-reference block | Alternate | Very cohesive and lower risk, but the immediate line-reduction payoff is smaller than the QR external-reference block unless partial SVD helpers are included, which would broaden scope. |
| `tests/test_chol_csc_supernodal.c` dense backend/env block | Alternate | Cohesive and important, but environment variables such as `SPARSE_CHOL_DENSE_BACKEND` and postorder toggles add cleanup/restoration risk. |
| `tests/test_chol_csc.c` external dense-reference block | Alternate | Good helper-boundary fit, but lower line-count payoff and less urgent than the largest remaining QR surface. |
| `tests/test_etree.c` fixture/helper cluster | Deferred | High helper density but no Day 2 seam was as obviously cohesive as QR external reference ownership. |
| `tests/test_integration.c` lifecycle cluster | Deferred | Cross-solver lifecycle behavior is too broad for a single-cluster extraction unless a smaller proof-owner boundary is selected in a future sprint. |
| `tests/test_iterative.c` CG/GMRES cluster | Deferred | Iterative solver convergence, allocation, and handle lifetime behavior increase behavior-preservation risk. |
| Additional `tests/test_ldlt_csc.c` extraction | Deferred | Sprint 185 already extracted the highest-value LDLT CSC helper families and added a guard; another pass would likely provide lower incremental closure. |
| Production `src/*.c` extraction | Deferred | Production movement carries source-list, public behavior, ABI, and CMake parity risk that is unnecessary while strong test-only candidates exist. |

### Day 2 Validation

Inspection and planning checks:

```sh
git status --short --branch
sed -n '1,260p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '50,112p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
sed -n '1,220p' tests/test_qr.c
sed -n '1,220p' tests/test_svd.c
sed -n '1,220p' tests/test_chol_csc_supernodal.c
sed -n '1,220p' tests/test_chol_csc.c
for f in tests/test_qr.c tests/test_svd.c tests/test_chol_csc_supernodal.c tests/test_chol_csc.c tests/test_etree.c tests/test_integration.c tests/test_iterative.c tests/test_graph.c; do printf '\n%s\n' "$f"; rg -n '^/\* [=-]|^static void test_|RUN_TEST\(' "$f" | head -n 80; done
for f in tests/test_qr.c tests/test_svd.c tests/test_chol_csc_supernodal.c tests/test_chol_csc.c tests/test_etree.c tests/test_integration.c tests/test_iterative.c tests/test_graph.c tests/test_ldlt.c; do printf '%s\tRUN_TEST=' "$f"; rg --count 'RUN_TEST\(' "$f"; done
sed -n '1120,2225p' tests/test_qr.c
sed -n '1,220p' tests/test_qr_helpers.h
sed -n '2860,3975p' tests/test_qr.c
git diff --check
```

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

### Day 3 Handoff

Day 3 should confirm or reject the recommended QR external dense-reference
cluster, then write the invariant contract before any code movement. If Day 3
accepts the recommendation, the likely extraction target is a new family-local
header such as `tests/test_qr_external_ref_helpers.h`, included by
`tests/test_qr.c` while preserving the existing `test_qr` proof-owner binary
and `RUN_TEST(...)` order.

## Day 3: Cluster Selection and Invariant Contract

### Selection Decision

Sprint 193 selects exactly one review-surface reduction cluster:

`tests/test_qr.c` external dense-reference rank/nullspace/threshold block.

The selected cluster is test-only and should be extracted behind a
family-local helper header. The existing `test_qr` proof-owner binary remains
the only registered test executable for this cluster.

### Selected Owner Files

| Surface | Day 3 ownership decision |
| --- | --- |
| Proof-owner test | `tests/test_qr.c` remains the registered proof-owner binary and keeps `main` plus every `RUN_TEST(...)` entry. |
| Existing QR helper header | `tests/test_qr_helpers.h` remains the owner for generic QR fixture builders and reconstruction helpers already shared by several QR test groups. |
| New selected helper boundary | Planned as `tests/test_qr_external_ref_helpers.h`, pending Day 4 design. It should own only external dense-reference readers and the selected rank/nullspace/threshold test bodies. |
| External reference script | `tests/qr_external_dense_reference.py` remains the Python reference owner and is not modified by default. |
| Shared external helper | `tests/test_solver_helpers.h` continues to provide `tf_read_external_reference_vector` under `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`. |
| Make registration | `Makefile` keeps `$(TESTDIR)/test_qr.c` in `TEST_SRCS`; no new test binary is planned. |
| CMake registration | `CMakeLists.txt` keeps `add_sparse_test(test_qr)`; no new CMake test target is planned. |
| Library source manifest | `build-metadata/library_sources.txt` remains unchanged because no production source file is selected. |
| Maintainer docs | `docs/maintainer_guide.md` should document the QR helper ownership boundary after implementation. |

### Selected Function Inventory

The selected cluster includes these local helpers and tests:

| Symbol | Current location | Planned boundary |
| --- | --- | --- |
| `read_qr_basis_external_reference` | `tests/test_qr.c:31` | Move to the selected QR external-reference helper boundary. |
| `read_qr_threshold_external_reference` | `tests/test_qr.c:61` | Move to the selected QR external-reference helper boundary. |
| `test_qr_external_dense_reference_rank1_4x3_nullspace_projector` | `tests/test_qr.c:1153` | Move test body to helper boundary; keep `RUN_TEST` registration in `tests/test_qr.c`. |
| `test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector` | `tests/test_qr.c:1308` | Move test body to helper boundary; keep `RUN_TEST` registration in `tests/test_qr.c`. |
| `test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector` | `tests/test_qr.c:1407` | Move test body to helper boundary; keep `RUN_TEST` registration in `tests/test_qr.c`. |
| `make_rankdef_wide_3x5` | `tests/test_qr.c:1527` | Move with the wide nullspace external-reference test. |
| `test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace` | `tests/test_qr.c:1541` | Move test body to helper boundary; keep `RUN_TEST` registration in `tests/test_qr.c`. |
| `test_qr_external_dense_reference_rank_threshold_diag4_family` | `tests/test_qr.c:1875` | Move test body to helper boundary; keep `RUN_TEST` registration in `tests/test_qr.c`. |
| `test_qr_external_dense_reference_rank_threshold_diag4_scaled_family` | `tests/test_qr.c:1943` | Move test body to helper boundary; keep `RUN_TEST` registration in `tests/test_qr.c`. |
| `test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family` | `tests/test_qr.c:2033` | Move test body to helper boundary; keep `RUN_TEST` registration in `tests/test_qr.c`. |
| `test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family` | `tests/test_qr.c:2129` | Move test body to helper boundary; keep `RUN_TEST` registration in `tests/test_qr.c`. |

The economy-mode external dense-reference test at `tests/test_qr.c:2505`
remains outside the selected cluster. It may continue to call
`read_qr_basis_external_reference` after that reader moves into the new helper
header, but its test body, `RUN_TEST(...)` placement, and economy-mode
assertions are not Sprint 193 extraction scope unless Day 4 discovers a
compile-only requirement.

### No-Behavior-Change Invariants

| Invariant area | Required preservation |
| --- | --- |
| Public API | No changes to `include/*.h`, exported symbols, QR public APIs, or production solver behavior. |
| Test proof owner | `test_qr` remains the only proof-owner binary for this cluster; no new `test_qr_*` executable is added. |
| Test names | Selected `test_qr_external_dense_reference_*` function names remain unchanged so `RUN_TEST(...)` output and review history stay stable. |
| Test order | All existing `RUN_TEST(...)` ordering in `tests/test_qr.c` remains unchanged. |
| Fixture keys | Strings passed to `tests/qr_external_dense_reference.py` remain byte-for-byte unchanged. |
| Reference command | The command form `python3 tests/qr_external_dense_reference.py %s` remains unchanged. |
| Windows behavior | `#ifdef _WIN32` skip behavior and skip messages remain unchanged. |
| External-helper enablement | `_POSIX_C_SOURCE` placement and `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` behavior remain valid before `test_solver_helpers.h` use. |
| Numeric expectations | Expected ranks, nullities, dimensions, thresholds, perturbations, projector layouts, residual tolerances, and orthogonality tolerances remain unchanged. |
| Cleanup | Existing `sparse_qr_free`, `sparse_free`, `free`, and early-return cleanup ordering remains unchanged except for mechanical movement into the helper header. |
| Diagnostics | `TF_FAIL_`, `printf`, and command-overflow/error reason strings remain unchanged unless Day 4 identifies a typo that must be explicitly deferred. |
| Source registration | The new helper header remains header-only and absent from Make/CMake test targets and `build-metadata/library_sources.txt`. |

### Boundary Between Files

| Must remain in `tests/test_qr.c` | May move to `tests/test_qr_external_ref_helpers.h` |
| --- | --- |
| `_POSIX_C_SOURCE` feature-test macro block. | External QR basis and threshold reader helper bodies. |
| Includes needed by `tests/test_qr.c`, including the new helper include. | Selected QR external dense-reference rank/nullspace test bodies. |
| `main` and every `RUN_TEST(...)` entry. | Selected QR external dense-reference threshold test bodies. |
| General QR, economy, sparse-mode, reorder, and refinement tests. | `make_rankdef_wide_3x5`, because it only supports the selected wide nullspace external-reference test. |
| Existing generic helper include `test_qr_helpers.h`. | Helper-local constants if moving them does not change names or behavior. |

### Day 4 Design Questions

1. Should the new helper header be named `tests/test_qr_external_ref_helpers.h`
   or a more specific `tests/test_qr_external_dense_reference_helpers.h`?
2. Which includes belong in the new helper header versus remaining inherited
   from `tests/test_qr.c`?
3. Should the helper guard require the selected test symbols to be absent from
   `tests/test_qr.c`, or only require the new header include and registration
   boundaries?
4. How should the guard account for the economy external-reference test that
   remains in `tests/test_qr.c` but shares `read_qr_basis_external_reference`?
5. Should Day 5 start with a compile-only scaffold before moving any test
   bodies?

### Day 3 Validation

Inspection and planning checks:

```sh
git status --short --branch
sed -n '260,430p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '91,151p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
rg -n "test_qr.c|test_qr\)|test_qr_helpers|qr_external" Makefile CMakeLists.txt tests/test_qr.c tests/test_qr_helpers.h tests/qr_external_dense_reference.py
sed -n '1,260p' tests/qr_external_dense_reference.py
sed -n '2460,2565p' tests/test_qr.c
rg -n "read_qr_basis_external_reference|read_qr_threshold_external_reference|make_rankdef_wide_3x5|test_qr_external_dense_reference" tests/test_qr.c
sed -n '3896,3925p' tests/test_qr.c
git diff --check
```

Day 3 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

## Day 4: Extraction Boundary Design

### Boundary Decision

Use a new header-only, family-local QR helper boundary:

`tests/test_qr_external_ref_helpers.h`

The shorter `external_ref` name matches the existing local style of concise
test helper names while still making the ownership specific to QR external
dense-reference tests. The helper remains included by `tests/test_qr.c` and is
not registered as a standalone test binary or production source.

### Include Contract

| File | Required include behavior |
| --- | --- |
| `tests/test_qr.c` | Keeps `_POSIX_C_SOURCE`, production/test includes, `#define TF_ENABLE_EXTERNAL_REFERENCE_HELPER`, `#include "test_solver_helpers.h"`, and then includes the new helper header. |
| `tests/test_qr_external_ref_helpers.h` | Uses include guard `TEST_QR_EXTERNAL_REF_HELPERS_H`; includes only what it owns directly if needed, preferring inherited test-owner context for existing symbols. |
| `tests/test_qr_helpers.h` | Remains the generic QR fixture/helper owner and continues to be included before the selected external-reference helper. |
| `tests/test_solver_helpers.h` | Continues to provide `tf_read_external_reference_vector`; the feature flag remains owned by `tests/test_qr.c`. |

Preferred include order after implementation:

```c
#include "test_qr_helpers.h"
#define TF_ENABLE_EXTERNAL_REFERENCE_HELPER
#include "test_solver_helpers.h"
#include "test_qr_external_ref_helpers.h"
```

This keeps `tf_read_external_reference_vector`, `tf_qr_*` helpers, QR public
types, sparse matrix/vector APIs, test macros, `<math.h>`, `<stdio.h>`,
`<stdlib.h>`, and `<string.h>` visible to the moved static helper/test bodies
without adding a new compilation unit.

### Planned Moved Surface

| Group | Symbols | Design note |
| --- | --- | --- |
| Reader helpers | `read_qr_basis_external_reference`, `read_qr_threshold_external_reference` | Move first after scaffold. They support selected rank/nullspace/threshold tests and the out-of-scope economy external-reference test that remains in `tests/test_qr.c`. |
| Nullspace/projector tests | Four selected `test_qr_external_dense_reference_*nullspace*` tests | Move together after reader helper compile validation. Preserve fixture keys, dimensions, projector layout, residual tolerances, and messages. |
| Wide nullspace fixture | `make_rankdef_wide_3x5` | Move with the wide nullspace test because it supports only that selected test. |
| Threshold tests | Four selected rank-threshold external-reference tests | Move together after nullspace tests are stable. Preserve threshold arrays, perturbations, pivot-ratio reporting, and rank expectations. |
| Registration | Existing `RUN_TEST(...)` lines in `tests/test_qr.c` | Do not move. They remain the review-visible proof-owner list. |

### Explicit Non-Moved Surface

| Surface | Reason |
| --- | --- |
| `test_qr_external_dense_reference_economy_projector_5x3` | It is an economy-mode proof, not part of the selected rank/nullspace/threshold cluster. It can call the moved reader helper. |
| General Householder/basic QR tests | Different proof family and not external-reference specific. |
| Economy QR tests | Separate proof family even when they use the external basis reader. |
| Sparse-mode QR tests | Separate dense-vs-sparse mode proof family. |
| Reorder and refinement tests | Separate behavior surfaces. |
| `tests/qr_external_dense_reference.py` | Reference data owner; no behavior change required for review-surface extraction. |
| `src/sparse_qr.c`, public headers, and library source lists | Production behavior is out of scope. |

### Cleanup and Error-Path Ownership

| Pattern | Required behavior |
| --- | --- |
| External-reference skip | Preserve `TF_EXTERNAL_REFERENCE_SKIP` handling and `SKIP_TEST(reason)` behavior. |
| External-reference failure | Preserve `TF_FAIL_("external QR ... reference failed: %s", reason)` and immediate `return`. |
| Allocation failure | Preserve current `ASSERT_NOT_NULL` plus cleanup path for `SparseMatrix`, `sparse_qr_t`, heap arrays, and stack buffers. |
| QR factor failure | Preserve current `ASSERT_ERR`, cleanup, and return behavior. |
| Nullspace/basis failure | Preserve cleanup before return. |
| Matrix insert failure | Preserve `tf_qr_insert_or_free` cleanup semantics. |
| Process-global state | No selected QR external-reference tests mutate process-global overrides or environment variables. Day 7 should still audit this after movement. |
| Temporary files/processes | The only external process is through `tf_read_external_reference_vector`; command string and error handling remain unchanged. |

### Guard Design

Add a Sprint 193 QR helper guard after implementation. Tentative script:

`scripts/check_qr_external_ref_helper_guard.sh`

Tentative Make target:

`qr-external-ref-helper-guard`

Guard checks should cover:

- `tests/test_qr.c` exists;
- `tests/test_qr_external_ref_helpers.h` exists;
- `tests/test_qr.c` includes `test_qr_external_ref_helpers.h` exactly once;
- `tests/test_qr.c` remains registered in `Makefile` `TEST_SRCS`;
- `CMakeLists.txt` still contains `add_sparse_test(test_qr)`;
- the new helper header is absent from `Makefile`, `CMakeLists.txt`, and
  `build-metadata/library_sources.txt` registration;
- no `add_sparse_test(test_qr_external_ref_helpers)` target exists;
- selected `RUN_TEST(...)` lines remain in `tests/test_qr.c`;
- selected moved test definitions are absent from `tests/test_qr.c` after the
  move, while `test_qr_external_dense_reference_economy_projector_5x3`
  remains present in `tests/test_qr.c`;
- the helper header contains include guard
  `TEST_QR_EXTERNAL_REF_HELPERS_H`.

### Source-List Update Checklist

| Source-list surface | Day 4 design |
| --- | --- |
| `Makefile` `TEST_SRCS` | No test registration change planned; keep `$(TESTDIR)/test_qr.c`. Add only the guard target if implemented. |
| `CMakeLists.txt` tests | No CMake test registration change planned; keep `add_sparse_test(test_qr)`. |
| `build-metadata/library_sources.txt` | No change planned. |
| `Makefile` `LIB_SRCS` | No change planned. |
| `CMakeLists.txt` library sources | No change planned. |
| Format/lint source sets | Header will be picked up by Makefile wildcard formatting/lint inputs through `ALL_TEST_SRC`. |
| Maintainer docs | Add QR helper ownership guidance once implementation lands. |

### Review Checkpoints

1. Day 5: add an empty helper header with guard and include it from
   `tests/test_qr.c`; compile `test_qr`.
2. Day 6: move reader helpers and the selected nullspace/projector tests;
   compile and run focused `test_qr`.
3. Day 7: audit cleanup and early-return paths after movement.
4. Day 8: add the QR external-reference helper guard and integrate it through
   Make.
5. Day 9: add or adjust focused behavior coverage only if movement exposes an
   unguarded invariant.
6. Day 10: document QR external-reference helper ownership in the maintainer
   guide.
7. Day 11-14: run focused validation, full C quality gate, review-surface
   metrics, and closeout.

### Day 4 Validation

Inspection and planning checks:

```sh
git status --short --branch
sed -n '430,620p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '135,198p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
sed -n '1,220p' docs/planning/EPIC_17/SPRINT_193/artifacts/day3-selected-cluster-contract.md
rg -n "tf_qr_make_|tf_qr_insert_or_free|vec_norm2|sparse_qr_|sparse_matvec|sparse_create|sparse_insert|sparse_free|ASSERT_|REQUIRE_|SKIP_TEST|TF_FAIL_|tf_read_external_reference_vector|snprintf|strcmp|sqrt|fabs" tests/test_qr.c | sed -n '1,180p'
sed -n '1,90p' tests/test_ldlt_csc.c
sed -n '619,626p' Makefile
rg -n "test_qr_helpers|test_qr_external|test_qr.c|test_qr\)" Makefile CMakeLists.txt build-metadata/library_sources.txt docs/maintainer_guide.md tests/*.h tests/*.c | head -n 120
git diff --check
```

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

### Day 5 Handoff

Day 5 should add only the mechanical scaffold:

1. Create `tests/test_qr_external_ref_helpers.h` with include guard
   `TEST_QR_EXTERNAL_REF_HELPERS_H`.
2. Include it from `tests/test_qr.c` after `test_solver_helpers.h`.
3. Do not move large test bodies until the scaffold compiles.
4. Run `make build/test_qr` after removing any stale binary if needed.
5. Record scaffold validation before Day 6 helper movement.

## Day 5: Mechanical Extraction Scaffold

### Implemented Scaffold

Day 5 added the selected QR helper boundary without moving behavior:

| File | Change |
| --- | --- |
| `tests/test_qr_external_ref_helpers.h` | Added header-only scaffold with include guard `TEST_QR_EXTERNAL_REF_HELPERS_H`. |
| `tests/test_qr.c` | Added `#include "test_qr_external_ref_helpers.h"` after `test_solver_helpers.h`. |

The scaffold is intentionally empty except for the include guard and a short
ownership comment. Selected test bodies remain in `tests/test_qr.c` until Day
6 helper movement.

### Source-List Impact

| Surface | Day 5 result |
| --- | --- |
| Make test registration | Unchanged; `$(TESTDIR)/test_qr.c` remains the proof-owner entry. |
| CMake test registration | Unchanged; `add_sparse_test(test_qr)` remains the proof-owner target. |
| Library source manifest | Unchanged; no production source file was added. |
| Make/CMake library sources | Unchanged. |
| Header formatting/lint ownership | The new header is picked up by existing Makefile wildcard header/test-source sets. |
| Guard | Not added yet; Day 8 owns the QR helper guard after movement stabilizes. |

### Compile Validation

Focused validation command:

```sh
make build/test_qr && ./build/test_qr
```

Result:

| Metric | Result |
| --- | --- |
| Build | passed |
| Test binary | `./build/test_qr` |
| Tests run | 77 |
| Failures | 0 |
| Skips | 0 |
| Assertions | 960 |
| Runtime | 4.384 s |

The build command recompiled `tests/test_qr.c`, proving the new header include
is visible and the proof-owner binary remains intact.

### Review-Surface Metrics

| File | Day 5 line count |
| --- | ---: |
| `tests/test_qr.c` | 3971 |
| `tests/test_qr_external_ref_helpers.h` | 9 |

The net line count increases on Day 5 because this day only adds the scaffold.
Line reduction starts on Day 6 when selected helper bodies move.

### Day 6 Handoff

Day 6 should move the selected helper logic in small chunks:

1. Move `read_qr_basis_external_reference` and
   `read_qr_threshold_external_reference` into the new helper header.
2. Rebuild `test_qr` after reader movement.
3. Move selected nullspace/projector external-reference tests.
4. Move `make_rankdef_wide_3x5` with the wide nullspace test.
5. Move selected rank-threshold tests.
6. Preserve the economy external-reference test body in `tests/test_qr.c`,
   allowing it to call the moved basis reader through the included header.
7. Re-run `make build/test_qr && ./build/test_qr`.

### Day 5 Validation

Commands run:

```sh
git status --short --branch
sed -n '1,40p' tests/test_qr.c
sed -n '620,760p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '183,236p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
make build/test_qr && ./build/test_qr
wc -l tests/test_qr.c tests/test_qr_external_ref_helpers.h
git diff -- tests/test_qr.c tests/test_qr_external_ref_helpers.h
git diff --check
```

Day 5 changed `.c` and `.h` files. Focused scaffold validation passed. The
full `make format && make lint && make test` gate is required before final
Sprint 193 closeout and remains scheduled for Day 12/Day 14 after the selected
movement and guard work are complete.

## Day 6: Helper Movement and Call-Site Preservation

### Movement Summary

Day 6 moved the selected QR external-reference logic from `tests/test_qr.c`
into `tests/test_qr_external_ref_helpers.h` while preserving the existing
`test_qr` proof-owner binary and all `RUN_TEST(...)` registrations.

| Group | Day 6 result |
| --- | --- |
| Reader helpers | `read_qr_basis_external_reference` and `read_qr_threshold_external_reference` moved to the helper header. |
| Nullspace/projector tests | Four selected external dense-reference nullspace/projector tests moved to the helper header. |
| Wide nullspace fixture | `make_rankdef_wide_3x5` moved with the wide nullspace test. |
| Rank-threshold tests | Four selected external dense-reference rank-threshold tests moved to the helper header. |
| Proof-owner registration | `main` and every `RUN_TEST(...)` line stayed in `tests/test_qr.c`. |
| Economy external-reference test | `test_qr_external_dense_reference_economy_projector_5x3` stayed in `tests/test_qr.c` and continues to call the moved basis reader. |

### Preserved Behavior

The movement preserved:

- selected test function names;
- `RUN_TEST(...)` order in `tests/test_qr.c`;
- fixture keys and Python command strings;
- Windows skip branches and messages;
- external-reference skip/failure diagnostics;
- expected rank, nullity, threshold, perturbation, projector, residual, and
  orthogonality constants;
- cleanup ordering for `sparse_qr_free`, `sparse_free`, and `free`;
- the existing `test_qr` Make/CMake proof-owner registrations;
- absence of new production sources or test binaries.

### Review-Surface Metrics

| File | Day 5 lines | Day 6 lines | Change |
| --- | ---: | ---: | ---: |
| `tests/test_qr.c` | 3971 | 3038 | -933 |
| `tests/test_qr_external_ref_helpers.h` | 9 | 947 | +938 |

The selected proof-owner file is now significantly smaller, while the moved
logic is isolated behind a named QR external-reference helper boundary.

### Focused Validation

Command:

```sh
make build/test_qr && ./build/test_qr
```

Result:

| Metric | Result |
| --- | --- |
| Build | passed |
| Test binary | `./build/test_qr` |
| Tests run | 77 |
| Failures | 0 |
| Skips | 0 |
| Assertions | 960 |
| Runtime | 5.421 s |

### Day 7 Handoff

Day 7 should audit cleanup and error paths after the movement, with attention
to:

1. external-reference skip/failure returns;
2. allocation and QR-factor failure cleanup;
3. early returns after `ASSERT_*`/`TF_FAIL_`;
4. moved reader behavior for the economy external-reference test that remains
   in `tests/test_qr.c`;
5. absence of process-global override or environment mutation in the selected
   moved block;
6. whether any cleanup-focused regression or guard check should be added.

### Day 6 Validation

Commands run:

```sh
git status --short --branch
sed -n '1,115p' tests/test_qr.c
sed -n '1140,1675p' tests/test_qr.c
sed -n '1865,2235p' tests/test_qr.c
python3 - <<'PY'
from pathlib import Path
...
PY
sed -n '1,140p' tests/test_qr.c
sed -n '1,120p' tests/test_qr_external_ref_helpers.h
rg -n "read_qr_basis_external_reference|read_qr_threshold_external_reference|make_rankdef_wide_3x5|test_qr_external_dense_reference" tests/test_qr.c tests/test_qr_external_ref_helpers.h
wc -l tests/test_qr.c tests/test_qr_external_ref_helpers.h
make build/test_qr && ./build/test_qr
git diff --stat
git diff -- tests/test_qr.c tests/test_qr_external_ref_helpers.h | sed -n '1,220p'
git diff --check
```

Day 6 changed `.c` and `.h` files. Focused QR validation passed. The full
`make format && make lint && make test` gate remains required before Sprint
193 closeout.

## Day 7: Cleanup and Error-Path Ownership

### Cleanup Audit Summary

Day 7 audited the moved QR external-reference helper boundary after Day 6
movement. The selected moved block does not use `REQUIRE_*`, does not mutate
environment variables, and does not set process-global kernel overrides.

| Audit area | Result |
| --- | --- |
| `REQUIRE_*` early returns | None in `tests/test_qr_external_ref_helpers.h`; no assertion macro in the moved block exits before cleanup through `REQUIRE_*`. |
| External-reference skip/failure | Existing `TF_EXTERNAL_REFERENCE_SKIP`, `SKIP_TEST(reason)`, `TF_FAIL_`, and immediate-return behavior preserved. |
| Reader helpers | Invalid arguments, unsupported fixture keys, command overflow, and external process execution still return structured `TF_EXTERNAL_REFERENCE_*` statuses. |
| Allocation failure | `SparseMatrix` creation and heap allocation checks remain paired with existing cleanup or pre-ownership returns. |
| Insert failure | `tf_qr_insert_or_free` still owns sparse-matrix cleanup for fixture insertion failures. |
| QR factor failure | Existing `ASSERT_ERR`, `sparse_free(A)`, and return behavior preserved. |
| Nullspace/rank-info failure | Existing `sparse_qr_free(&qr)`, `sparse_free(A)`, and return behavior preserved. |
| Environment variables | No moved selected test calls `tf_setenv`, `tf_unsetenv`, `setenv`, or `unsetenv`. |
| Process-global overrides | No moved selected test sets kernel overrides or process-global registration state. |
| Temporary external process | Still isolated through `tf_read_external_reference_vector` with unchanged command text and reason buffer handling. |

### Scoped Code Cleanup

Day 7 made one non-behavioral cleanup:

| File | Change |
| --- | --- |
| `tests/test_qr_external_ref_helpers.h` | Updated the scaffold-era ownership comment so it now describes the moved helper boundary and the retained `tests/test_qr.c` proof-owner registration. |

No cleanup-path code changes were required. The moved test bodies already
restore owned QR/matrix/heap resources before early returns after ownership is
established.

### Validation Caveat Found

Running `make build/test_qr && ./build/test_qr` immediately after editing only
the helper header reported:

```text
make: `build/test_qr' is up to date.
```

That confirms the known focused-Make header dependency caveat for included
test helper headers. Day 7 forced the stale binary out with:

```sh
find build -maxdepth 1 -type f -name test_qr -delete
```

and then rebuilt and ran `test_qr`.

### Focused Validation

Forced rebuild validation command:

```sh
find build -maxdepth 1 -type f -name test_qr -delete && make build/test_qr && ./build/test_qr
```

Result:

| Metric | Result |
| --- | --- |
| Build | passed after forced rebuild |
| Test binary | `./build/test_qr` |
| Tests run | 77 |
| Failures | 0 |
| Skips | 0 |
| Assertions | 960 |
| Runtime | 4.995 s |

### Day 8 Guard Handoff

Day 8 should encode the Day 7 findings into a guard:

1. Require `test_qr_external_ref_helpers.h` to exist with include guard
   `TEST_QR_EXTERNAL_REF_HELPERS_H`.
2. Require `tests/test_qr.c` to include the helper exactly once.
3. Require selected moved test definitions to remain out of `tests/test_qr.c`.
4. Require selected `RUN_TEST(...)` entries to remain in `tests/test_qr.c`.
5. Require the economy external-reference test body to remain in
   `tests/test_qr.c`.
6. Require Make/CMake proof-owner registration to remain on `test_qr`.
7. Require the helper header to stay absent from standalone Make/CMake/library
   registration.
8. Document in guard output or maintainer guidance that focused header-only
   validation should force-rebuild `build/test_qr`.

### Day 7 Validation

Commands run:

```sh
git status --short --branch
sed -n '236,298p' docs/planning/EPIC_17/SPRINT_193/PLAN.md
sed -n '760,940p' docs/planning/EPIC_17/SPRINT_193/WORKING_NOTES.md
sed -n '1,220p' tests/test_qr_external_ref_helpers.h
rg -n "return;|return NULL|return TF_|SKIP_TEST|TF_FAIL_|ASSERT_|REQUIRE_|sparse_qr_free|sparse_free|free\(|tf_setenv|tf_unsetenv|setenv|unsetenv|override|kernel|global" tests/test_qr_external_ref_helpers.h
rg -n "static void test_qr_external_dense_reference|static int read_qr_|make_rankdef_wide_3x5|#ifdef _WIN32|#endif" tests/test_qr_external_ref_helpers.h
tail -n 80 tests/test_qr_external_ref_helpers.h
rg -n "REQUIRE_|tf_setenv|tf_unsetenv|setenv|unsetenv|override|kernel" tests/test_qr_external_ref_helpers.h tests/test_qr.c
make build/test_qr && ./build/test_qr
find build -maxdepth 1 -type f -name test_qr -delete && make build/test_qr && ./build/test_qr
git diff --check
```

Day 7 changed `.c` and `.h` files only through the helper-header ownership
comment. Focused QR validation passed after a forced rebuild. The full
`make format && make lint && make test` gate remains required before Sprint
193 closeout.

## Day 8: Source-List Guards

### Objective

Add a cluster-specific guard that prevents the Sprint 193 QR
external-reference helper extraction from drifting across source-list, test-list,
and ownership boundaries.

### Changes Made

- Added `scripts/check_qr_external_ref_helper_guard.sh`.
- Added `make qr-external-ref-helper-guard`.
- Added `tests/test_qr_external_ref_helper_guard.py`.
- Added `artifacts/day8-source-list-guards.md`.

### Guard Contract

The guard now requires:

1. `tests/test_qr.c`, `tests/test_qr_external_ref_helpers.h`, `Makefile`,
   `CMakeLists.txt`, and `build-metadata/library_sources.txt` to exist.
2. `test_qr.c` to remain registered in Make and CMake.
3. `test_qr.c` to include `test_qr_external_ref_helpers.h` exactly once.
4. `test_qr_external_ref_helpers.h` to keep include guard
   `TEST_QR_EXTERNAL_REF_HELPERS_H`.
5. Moved selected rank/nullspace/threshold definitions to stay in the helper and
   out of `test_qr.c`.
6. Selected `RUN_TEST(...)` entries to remain in `test_qr.c`.
7. The economy external-reference test body to remain in `test_qr.c`.
8. The helper to remain absent from standalone Make/CMake test registration and
   library-source manifests.

### Focused Validation

Commands run:

```sh
python3 tests/test_qr_external_ref_helper_guard.py
make qr-external-ref-helper-guard
find build -maxdepth 1 -type f -name test_qr -delete && make build/test_qr && ./build/test_qr
git diff --check
```

Results:

| Check | Result |
| --- | --- |
| Guard regression tests | passed |
| Make guard target | passed |
| Forced QR rebuild | passed |
| `./build/test_qr` | 77 tests, 0 failures, 0 skips, 960 assertions, 4.468 s |
| `git diff --check` | passed |

Day 8 added a `.h` helper and modified a `.c` proof-owner file earlier in the
sprint; the full `make format && make lint && make test` gate remains required
before Sprint 193 closeout.

## Day 9: Focused Behavior Regression Coverage

### Objective

Strengthen extraction-sensitive behavior coverage for the selected QR
external-reference helper cluster without broadening the algorithmic surface or
weakening existing numerical assertions.

### Changes Made

- Added `test_qr_external_reference_readers_reject_invalid_arguments` to
  `tests/test_qr_external_ref_helpers.h`.
- Added `test_qr_external_reference_readers_reject_unsupported_fixtures` to
  `tests/test_qr_external_ref_helpers.h`.
- Registered both tests through `tests/test_qr.c`.
- Extended the QR helper guard and its fixture tests to require the new
  helper-owned behavior tests and `RUN_TEST(...)` entries.
- Added `artifacts/day9-behavior-coverage.md`.

### Behavior Coverage

The new tests cover reader failure paths that are sensitive to the extraction
boundary:

1. Basis reader NULL fixture-key rejection.
2. Basis reader NULL output-buffer rejection.
3. Basis reader NULL reason-buffer rejection.
4. Threshold reader NULL fixture-key rejection.
5. Threshold reader NULL output-buffer rejection.
6. Threshold reader NULL reason-buffer rejection.
7. Unsupported basis fixture diagnostic text.
8. Unsupported threshold fixture diagnostic text.

The existing moved numerical tests remain unchanged and continue to cover the
selected cluster's success paths and tolerance/rank decisions.

### Focused Validation

Commands run:

```sh
python3 tests/test_qr_external_ref_helper_guard.py
make qr-external-ref-helper-guard
find build -maxdepth 1 -type f -name test_qr -delete && make build/test_qr && ./build/test_qr
git diff --check
```

Results:

| Check | Result |
| --- | --- |
| Guard regression tests | passed |
| Make guard target | passed |
| Forced QR rebuild | passed |
| `./build/test_qr` | 79 tests, 0 failures, 0 skips, 976 assertions, 4.693 s |
| `git diff --check` | passed |

Day 9 modified `.c` and `.h` files. The full
`make format && make lint && make test` gate remains required before Sprint
193 closeout.

## Day 10: Boundary Documentation

### Objective

Document the Sprint 193 QR external-reference helper/source boundary, owner
files, validation commands, and no-behavior-change expectations for future
maintainers.

### Changes Made

- Added a Sprint 193 QR external-reference helper boundary section to
  `docs/maintainer_guide.md`.
- Extended `scripts/check_qr_external_ref_helper_guard.sh` to require
  maintainer-guide boundary markers.
- Extended `tests/test_qr_external_ref_helper_guard.py` with maintainer-guide
  fixture text and a missing-doc-marker negative case.
- Added `artifacts/day10-boundary-documentation.md`.

### Maintainer Contract

The guide now states:

1. `tests/test_qr_external_ref_helpers.h` owns the selected QR
   rank/nullspace/threshold external-reference readers, moved selected test
   bodies, and reader failure-path behavior tests.
2. `tests/test_qr.c` remains the registered QR proof-owner binary.
3. `tests/test_qr.c` retains `main`, selected `RUN_TEST(...)` registrations,
   `_POSIX_C_SOURCE`, `TF_ENABLE_EXTERNAL_REFERENCE_HELPER`, and the economy
   external-reference test body.
4. `tests/test_qr.c` must define `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` before
   including `test_qr_external_ref_helpers.h`; the helper header includes
   `test_solver_helpers.h` itself so formatter-driven include sorting cannot
   hide the external-reference reader API.
5. The helper stays family-local and header-only, absent from Makefile test
   lists, CMake `add_sparse_test(...)`, and
   `build-metadata/library_sources.txt`.
6. `make qr-external-ref-helper-guard` owns focused boundary validation after
   helper-layout changes.
7. Focused QR behavior validation after helper-header edits should force-rebuild
   `build/test_qr`.

### Non-Goals

The maintainer text explicitly keeps this as a no-behavior-change
review-surface reduction. It does not claim new QR algorithm capability,
numerical tolerance changes, performance improvement, platform expansion, or
broader external parity.

### Focused Validation

Commands run:

```sh
python3 tests/test_qr_external_ref_helper_guard.py
make qr-external-ref-helper-guard
git diff --check
```

Results:

| Check | Result |
| --- | --- |
| Guard regression tests | passed |
| Make guard target | passed |
| `git diff --check` | passed |

Day 10 changed documentation, the QR helper guard script, and its Python guard
tests. It did not add new C behavior changes beyond the existing Sprint 193
branch changes; the full `make format && make lint && make test` gate remains
required before Sprint 193 closeout.

## Day 11: Integrated Build and Source-List Validation

### Objective

Run the integrated source-list, selected-cluster ownership, focused QR, and
reviewed CMake compile/registration checks after the Sprint 193 extraction,
guard, behavior, and maintainer documentation work stabilized.

### Commands Run

```sh
make source-list-check
python3 tests/test_qr_external_ref_helper_guard.py && make qr-external-ref-helper-guard
find build -maxdepth 1 -type f -name test_qr -delete && make build/test_qr && ./build/test_qr
make quality-review-cmake-compile
git diff --check
```

### Results

| Check | Result |
| --- | --- |
| `make source-list-check` | passed: 49 library sources |
| Guard regression tests | passed |
| `make qr-external-ref-helper-guard` | passed, including maintainer docs |
| Forced QR rebuild and run | passed: 79 tests, 0 failures, 0 skips, 976 assertions, 4.882 s |
| `make quality-review-cmake-compile` | passed |
| CMake `ctest -N` inventory | 59 tests, including `test_qr` as test #20 |
| Makefile/CMake test-count parity | passed: 59 vs 59 |
| `git diff --check` | passed |

### Interpretation

The selected QR helper extraction remains source-list neutral: no production
source files were added, and `tests/test_qr_external_ref_helpers.h` remains an
included, header-only test helper. The Makefile and CMake proof-owner surfaces
still register `test_qr`, and the reviewed CMake compile path built that target
successfully after the extraction.

The full `make format && make lint && make test` gate remains Day 12 work and
is still required before Sprint 193 closeout because this branch includes
`.c`/`.h` changes.

## Day 12: Full C Quality Gate

### Objective

Run the required full C formatting, lint, and test gate for the Sprint 193 QR
external-reference helper extraction.

### Formatter-Stable Fix

The first `make test` attempt after `make format` failed while compiling
`tests/test_qr.c` because clang-format sorted
`test_qr_external_ref_helpers.h` before `test_solver_helpers.h`. That hid the
`TF_EXTERNAL_REFERENCE_*` declarations used by the extracted helper.

Fix applied:

- `tests/test_qr_external_ref_helpers.h` now defines
  `TF_ENABLE_EXTERNAL_REFERENCE_HELPER` if needed before including
  `test_solver_helpers.h`.
- `docs/maintainer_guide.md` and the Day 10 artifact now describe the
  formatter-stable dependency contract.

### Required Gate

Final command:

```sh
make format && make lint && make test
```

Result: passed.

Observed details:

| Phase | Result |
| --- | --- |
| `make format` | passed |
| `make lint` | passed: strict warnings, clang-tidy, cppcheck |
| `make test` | passed: ended with `All tests passed.` |
| `test_qr` | 79 tests, 0 failures, 0 skips, 976 assertions |
| `test_reorder_nd` | 35 tests, 0 failures, 1 skip |
| `test_framework_optin` | 8 tests, 0 failures, 3 skips |

### Follow-Up Checks

Commands run after the required gate:

```sh
make source-list-check
python3 tests/test_qr_external_ref_helper_guard.py && make qr-external-ref-helper-guard
git diff --check
```

Results:

| Check | Result |
| --- | --- |
| `make source-list-check` | passed: 49 library sources |
| Guard regression tests | passed |
| `make qr-external-ref-helper-guard` | passed, including maintainer docs |
| `git diff --check` | passed |

Day 12 closes the full C quality-gate requirement for the current Sprint 193
branch state. Day 13 still owns the review-surface audit, and Day 14 still owns
final closeout/handoff.

## Day 13: Review-Surface Audit

### Objective

Confirm that the selected QR external-reference helper extraction reduced the
review surface while preserving behavior, ownership, source-list boundaries, and
validation evidence.

### Metrics

| Measure | Before Sprint 193 branch edits | Current branch state | Result |
| --- | ---: | ---: | --- |
| `tests/test_qr.c` line count | 3970 | 3040 | Main QR proof owner reduced by 930 lines |
| Selected helper line count | 0 | 1003 | Selected cluster isolated in `tests/test_qr_external_ref_helpers.h` |
| `test_qr` registered tests | 77 | 79 | Existing selected tests preserved; 2 reader failure tests added |
| Production source changes under `src/` | 0 | 0 | No production algorithm change |
| Public header changes under `include/` | 0 | 0 | No API or ABI surface change |
| Library source-list count | 49 | 49 | Source manifest ownership unchanged |

### Risk Register

| Risk Area | Audit Result |
| --- | --- |
| Public API/ABI | No `include/` changes were present |
| Production behavior | No `src/` changes were present |
| QR tolerance policy | No tolerance or rank-threshold policy changes were introduced |
| Build registration | `test_qr` remains the only test executable owner for the selected cluster |
| Source manifests | The helper remains absent from Make/CMake library source lists |
| Generated artifacts | No generated report or build artifact churn was added |
| Documentation drift | Maintainer docs, guard checks, and test registrations describe the same boundary |

### Audit Artifact

Created
`docs/planning/EPIC_17/SPRINT_193/artifacts/day13-review-surface-audit.md`.

The artifact records:

- before/after line-count and test-count metrics;
- helper/proof-owner/source-list boundary consistency;
- final diff-risk register;
- validation evidence from Days 11 and 12;
- residuals and deferred candidates for future sprints.

### Residuals

- `test_qr_external_dense_reference_economy_projector_5x3` remains in
  `tests/test_qr.c` by Sprint 193 scope decision.
- Other large QR/economy/sparse-mode/refinement clusters remain future
  candidates only after separate boundary review.
- Header-only QR helper edits still require forced rebuild validation for
  focused `test_qr` runs.
- Day 14 still owns final closeout and handoff confirmation.

## Day 14: Closeout

### Objective

Complete final validation, record closeout evidence, and prepare Sprint 193 for
retrospective, commit, push, and pull request handoff.

### Closed Scope

Sprint 193 closed exactly one selected review-surface reduction claim: the
selected QR external-reference rank/nullspace/threshold cluster now lives in
`tests/test_qr_external_ref_helpers.h`, while `tests/test_qr.c` remains the QR
proof-owner executable with `main`, `RUN_TEST(...)` registration, and the scoped
economy external-reference body.

No public API, production source, QR tolerance, rank policy, solver behavior, or
generated report surface was intentionally changed.

### Final Metrics

| Measure | Before Sprint 193 branch edits | Final branch state | Result |
| --- | ---: | ---: | --- |
| `tests/test_qr.c` line count | 3970 | 3040 | Main QR proof owner reduced by 930 lines |
| Selected helper line count | 0 | 1003 | Selected cluster isolated in `tests/test_qr_external_ref_helpers.h` |
| `test_qr` registered tests | 77 | 79 | Existing selected tests preserved; 2 reader failure tests added |
| Production source changes under `src/` | 0 | 0 | No production algorithm change |
| Public header changes under `include/` | 0 | 0 | No API or ABI surface change |
| Library source-list count | 49 | 49 | Source manifest ownership unchanged |
| CMake/Makefile test-count parity | 59/59 | 59/59 | Test registration parity preserved |

### Final Validation

Final command:

```sh
make source-list-check && \
python3 tests/test_qr_external_ref_helper_guard.py && \
make qr-external-ref-helper-guard && \
make quality-review-cmake-compile && \
make format && \
make lint && \
make test
```

Result: passed.

Observed details:

| Check | Result |
| --- | --- |
| `make source-list-check` | passed: 49 library sources |
| Guard regression tests | passed |
| `make qr-external-ref-helper-guard` | passed |
| `make quality-review-cmake-compile` | passed configure, clean rebuild, `ctest -N`, and test-count parity |
| CMake tests | 59 |
| Makefile tests | 59 |
| `make format` | passed |
| `make lint` | passed strict warnings, clang-tidy, and cppcheck |
| `make test` | passed with final `All tests passed.` |
| `test_qr` | 79 tests, 0 failures, 0 skips, 976 assertions |
| `test_reorder_nd` | 35 tests, 0 failures, 1 skip |
| `test_reorder_amd_qg` | 7 tests, 0 failures, 0 skips, 2068 assertions |
| `git diff --check` | passed after the final gate |

### Closeout Artifact

Created `docs/planning/EPIC_17/SPRINT_193/artifacts/day14-closeout.md`.

The artifact records:

- closed scope and changed files;
- final metrics;
- final validation command log;
- retrospective inputs;
- residuals and next-sprint handoff notes.

### Retrospective Inputs

- Narrow candidate selection kept the extraction behavior-preserving and easy
  to audit.
- The guard target and maintainer docs now make the helper/proof-owner boundary
  mechanically enforceable.
- The Day 12 formatter-stability issue was resolved by making the helper own
  its `test_solver_helpers.h` dependency.
- Header-only helper edits still require forced focused rebuild validation when
  the full Makefile gate is not run.

### Handoff

Sprint 193 is ready for retrospective, commit, push, and pull request creation.
