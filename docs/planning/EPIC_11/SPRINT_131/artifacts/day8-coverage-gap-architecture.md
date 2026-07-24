# Sprint 131 Day 8 - Coverage Gap Architecture

## Purpose

Day 8 ranks coverage gaps by product and numerical risk, then defines which
coverage signals can support reviewed work and which remain supplemental,
smoke, optional, expensive, or experimental evidence.

This is a documentation-only architecture artifact. It does not change
coverage targets, thresholds, CI behavior, tests, generated reports, or source
code.

## Authoritative Inputs

| Input | Coverage role |
| --- | --- |
| `Makefile` coverage targets | Own `make coverage`, `make coverage-lcov`, `make coverage-gcovr`, backend selection, source filters, report paths, and the 80% aggregate line threshold. |
| `.github/workflows/ci.yml` | Owns the Linux supplemental coverage report job that installs lcov and runs `make coverage`. |
| `docs/planning/EPIC_2/SPRINT_29/coverage_threshold_decision.md` | Records the measured 81.3% aggregate baseline, per-file groups, and the decision to lower `COV_THRESHOLD` from 95 to 80. |
| `docs/planning/EPIC_2/SPRINT_29/coverage_audit_day11.md` | Records local macOS lcov incompatibility and the need for backend-specific interpretation. |
| `docs/planning/EPIC_9/SPRINT_98/artifacts/day10-coverage-topology-audit.md` and Day 11 cleanup | Confirm coverage remained tree-mutating and supplemental after topology review. |
| `docs/maintainer_guide.md` | Current maintainer-facing policy: coverage remains a live supplemental signal and must not be treated as an active reviewed baseline. |
| Sprint 131 Day 6-7 report-index artifacts | Defer coverage index generation until reviewed-versus-supplemental coverage boundaries are explicit. |

## Coverage Output Inventory

| Surface | Command or owner | Outputs | Interpretation |
| --- | --- | --- | --- |
| Auto-selected local coverage | `make coverage` | Routes to `coverage-$(COV_BACKEND)` | Tree-mutating supplemental signal. Backend depends on compiler detection. Run `make clean` before returning to normal reviewed validation. |
| Linux/GCC lcov coverage | `make coverage-lcov` | `coverage/coverage.info`, `coverage/coverage-src.info`, `coverage/html/index.html`, lcov summary | Supplemental aggregate line threshold check. Filters out `tests/*` and `benchmarks/*`. CI Linux uses this path. |
| Apple Clang gcovr coverage | `make coverage-gcovr` | `coverage/html/index.html`, gcovr summary | Supplemental local macOS-compatible report. Uses `/usr/bin/gcov`, filters to `src/`, excludes `tests/` and `benchmarks/`, and tolerates known suspicious-hit parse warnings. |
| CI supplemental coverage report | `.github/workflows/ci.yml` coverage job | Uploaded `coverage/html/` artifact and printed summary | Supplemental freshness and regression signal. A missing or stale artifact means no current coverage evidence, not a solver behavior failure by itself. |
| Historical threshold decision | Sprint 29 Day 12 artifact | Per-file and group coverage table | Baseline risk inventory, not a current pass/fail result. |
| Report-index coverage rows | Future Sprint 131+ generated or curated index | Backend, threshold, tree-mutating flag, command, artifacts, freshness, aggregate percentage | Allowed only as supplemental report metadata unless a future sprint explicitly changes coverage policy. |

## Risk Ranking Rubric

Coverage gaps must be ranked by risk, not by convenience or uncovered-line
count alone.

| Rank | Criteria | Reviewed action | Supplemental or residual action |
| --- | --- | --- | --- |
| High | Public solver workflow, correctness or failure semantics, numerical fallback, deterministic fixture path, and direct claim impact. | Add or strengthen owner tests when code changes touch the path and fixtures are stable. | If fixture construction is not stable, record blocker and owner before using coverage as report-index evidence. |
| Medium | Important implementation path with bounded user impact, specialized numerical setup, platform-specific behavior, or report-only claim impact. | Add focused tests when reachable through maintainable fixtures. | Keep as residual debt when tests require fragile, synthetic, or expensive setup. |
| Low | Defensive error handling, unreachable-by-contract branches, error-string stubs, benchmark-only/report-only code, or tool noise. | No automatic reviewed test obligation. | Track only if a future API, enum, or workflow change makes the path user-facing. |

Risk factors:

- `solver_family`: direct, iterative, SVD, eigensolver, reordering, graph,
  data structure, report workflow.
- `user_workflow`: public solve/factor/analyze call versus internal fallback.
- `numerical_risk`: residual correctness, convergence, singularity,
  rank/nullity, or fallback stability.
- `platform_risk`: Linux lcov, Apple Clang gcovr, optional dependency, or
  environment-specific behavior.
- `corpus_availability`: checked-in deterministic fixture, generated fixture,
  optional external corpus, expensive corpus, or no stable fixture.
- `claim_impact`: reviewed correctness claim, supplemental report claim,
  documentation wording, or no external claim.
- `owner_readiness`: known test owner and stable validation command versus
  unresolved fixture design.

## Coverage Gap Inventory

| Group | Source area | Sprint 29 baseline | Risk | Current gap class | Future owner |
| --- | --- | --- | --- | --- | --- |
| A - core data structures | `sparse_matrix.c`, `sparse_csr.c`, `sparse_vector.c`, `sparse_dense.c` | 82-100% | Medium-low | Mostly defensive error paths such as invalid dimensions, NULL options, or caller-guarded cases. | Core sparse matrix, CSR, vector, and dense helper tests. |
| B - direct factorizations | LU, CSR LU, Cholesky, CSC Cholesky, LDLT, CSC LDLT, QR | 81-85% | High | Bunch-Kaufman fallback, supernodal AUTO threshold dispatch, degenerate progress-callback paths. | Direct solver owner tests and external-reference helper owners. |
| C - iterative solvers and preconditioners | `sparse_iterative.c`, `sparse_ilu.c`, `sparse_ic.c` | 77-82% | High | Breakdown handling, cancellation callbacks, matrix-free NULL preconditioner paths. | Iterative, ILU, IC, stagnation, and matrix-free tests. |
| D - eigensolvers | `sparse_eigs.c` | 88% | Medium | Singular retry-shift path in inverse-iteration refinement. | Eigensolver tests with clustered-spectrum or singular-shift fixtures. |
| E - SVD and bidiagonalization | `sparse_svd.c`, `sparse_bidiag.c` | 77% | High | Bidiag back-projection, `pad_orthonormal_basis` non-convergence branch, singular-vector convergence edge cases. | SVD, partial-SVD helper, and bidiag tests. |
| F - symbolic and reordering | `sparse_analysis.c`, `sparse_etree.c`, `sparse_reorder*.c`, `sparse_colamd.c` | 72-85% | Medium | COLAMD restart-on-overflow, etree post-order compaction, disconnected graph paths. | Analysis, etree, COLAMD, and reorder tests. |
| G - graph and multilevel ND | `sparse_reorder_nd.c`, `sparse_graph.c` | 54-78% | Medium-high | FM ensemble permutations, empty-coarse-graph fallback, optional supernodal-postorder edge cases. | Graph, FM bucket, reorder ND, and guardrail owners. |
| H - error-string stubs | `sparse_types.c` | 50% | Low | Error-string table branches and stubs. | Error enum/API owner only when public error surface changes. |

The largest uncovered percentage is not automatically the highest reviewed
priority. For example, graph/ND gaps carry a large percentage gap, but many
paths require adversarial graph construction or optional environment settings.
Direct solver, iterative, and SVD gaps rank higher when they affect public
solve correctness, convergence, or failure semantics.

## Reviewed Versus Supplemental Coverage

| Evidence class | Reviewed status | Allowed use | Not allowed |
| --- | --- | --- | --- |
| Focused owner tests tied to code changes | Reviewed when run through normal quality gates | Support correctness, failure, and regression claims for the exact tested path. | Do not imply broader corpus or solver-family parity. |
| External-reference fixture tests | Reviewed only within each helper's documented fixture protocol | Support bounded dense-reference, residual, singular-value, rank, or projector claims. | Do not become raw basis-vector, LAPACK, NumPy, SciPy, or broad external parity claims. |
| Checked-in Matrix Market tests | Reviewed, smoke, expensive, or supplemental depending on taxonomy tags | Support the specific tagged fixture and owner workflow. | Do not imply SuiteSparse ecosystem coverage or broad Matrix Market parity. |
| `make coverage*` aggregate reports | Supplemental | Track aggregate regression risk, per-file gap movement, backend health, and report freshness. | Do not serve as behavioral completeness, reviewed baseline, or public capability proof. |
| Linux supplemental coverage workflow | Supplemental | Provide CI-produced freshness for coverage artifacts. | Do not block non-coverage reviewed claims unless the claim explicitly depends on current coverage evidence. |
| Smoke and optional corpus paths | Smoke or optional | Show that optional or checked-in corpus lanes still execute when present. | Do not silently promote to reviewed coverage. |
| Expensive guardrail and large-matrix reports | Expensive reviewed or supplemental depending on lane | Support bounded structural guardrail interpretation. | Do not become line coverage, performance, scalability, or broad corpus claims. |
| Synthetic fault-injection coverage | Experimental until designed | May target cold fallback paths in a future code-quality sprint. | Do not require it for current Sprint 131 report-index work. |

## Coverage Owner Map

| Area | Source owner | Test or report owner | Coverage owner label |
| --- | --- | --- | --- |
| Core sparse structures | `src/sparse_matrix.c`, `src/sparse_csr.c`, `src/sparse_vector.c`, `src/sparse_dense.c` | `tests/test_sparse_matrix.c`, `tests/test_csr.c`, `tests/test_sparse_vector.c`, `tests/test_dense.c` | `coverage-core-structures` |
| Direct solvers | LU, Cholesky, LDLT, QR source files | `tests/test_sparse_lu.c`, `tests/test_lu_csr.c`, `tests/test_cholesky.c`, `tests/test_chol_csc.c`, `tests/test_ldlt*.c`, `tests/test_qr*.c`, external helper tests | `coverage-direct-solvers` |
| Iterative and preconditioners | `src/sparse_iterative.c`, `src/sparse_ilu.c`, `src/sparse_ic.c` | `tests/test_iterative.c`, `tests/test_bicgstab.c`, `tests/test_minres.c`, `tests/test_ilu.c`, `tests/test_ic.c`, `tests/test_stagnation.c` | `coverage-iterative-preconditioners` |
| Eigensolvers | `src/sparse_eigs.c` | `tests/test_eigs*.c` | `coverage-eigensolvers` |
| SVD and bidiag | `src/sparse_svd.c`, `src/sparse_bidiag.c` | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, `tests/test_bidiag.c`, SVD external helper tests | `coverage-svd-bidiag` |
| Symbolic, reorder, and graph | `src/sparse_analysis.c`, `src/sparse_etree.c`, `src/sparse_reorder*.c`, `src/sparse_colamd.c`, `src/sparse_graph.c` | `tests/test_reorder*.c`, `tests/test_colamd.c`, `tests/test_etree.c`, `tests/test_graph*.c` | `coverage-symbolic-graph` |
| Coverage workflow | `Makefile`, `.github/workflows/ci.yml`, `coverage/` outputs | Coverage CI job, maintainer guide, future report-index artifact | `coverage-workflow` |

## Report-Index Claim Gates

Coverage gaps should block a report-index claim only when the index row would
make coverage evidence mean more than the coverage architecture allows.

Blocking cases:

- A generated row presents line coverage as reviewed behavioral evidence.
- A row omits backend, command, threshold, tree-mutating status, source filter,
  freshness, or reset requirement.
- A stale, missing, or failed coverage artifact is displayed as a current pass.
- A row uses aggregate coverage to imply solver-family, corpus, platform, or
  numerical parity.
- A high-risk residual gap is referenced as covered without owner, blocker,
  and fixture notes.

Non-blocking cases:

- Large-matrix guardrail index work remains bounded to guardrail report
  semantics and does not claim coverage completeness.
- Benchmark report indexes remain timing/report artifacts without coverage or
  performance claims.
- Dead-code report architecture can reference coverage gaps as triage inputs
  once Day 9 defines dead-code bucket ownership.
- Documentation-only planning indexes can link coverage artifacts as
  supplemental historical context.

Future coverage index rows may be generated only with these minimum fields:

- `report_family`: `coverage`
- `support_tier`: `supplemental`
- `backend`: `lcov`, `gcovr`, or `auto`
- `command`
- `tree_mutating`: `yes`
- `threshold`
- `aggregate_line_percent`
- `source_filter`
- `artifact_path`
- `freshness_source`
- `reset_command`
- `owner_label`
- `claim_boundary`

## Residual Coverage Queue

| Residual gap | Risk | Blocker | Future owner |
| --- | --- | --- | --- |
| Direct solver Bunch-Kaufman fallback and degenerate progress callbacks | High | Need deterministic fixtures that reach fallback and callback paths without brittle internal forcing. | `coverage-direct-solvers` |
| Supernodal AUTO dispatch near threshold | Medium-high | Need stable size/structure fixtures that exercise selector boundaries without turning runtime tuning into a correctness claim. | `coverage-direct-solvers` |
| Iterative breakdown and cancellation callbacks | High | Need reproducible breakdown, stagnation, and cancellation fixtures for each affected loop. | `coverage-iterative-preconditioners` |
| Matrix-free NULL preconditioner variants | Medium-high | Need explicit matrix-free fixture taxonomy and owner tests. | `coverage-iterative-preconditioners` |
| SVD bidiag back-projection and basis-padding cold paths | High | Need reachable scenarios that do not contradict caller guards or rely on invalid public inputs. | `coverage-svd-bidiag` |
| Singular-vector convergence edge cases | High | Need fixtures that distinguish convergence residual behavior from raw vector-basis orientation. | `coverage-svd-bidiag` |
| Eigensolver singular retry-shift path | Medium | Need constructed clustered-spectrum or singular-shift fixture with stable expected behavior. | `coverage-eigensolvers` |
| COLAMD overflow restart and etree post-order compaction | Medium | Need adversarial but maintainable symbolic fixtures, especially disconnected graph cases. | `coverage-symbolic-graph` |
| FM ensemble permutations and empty-coarse-graph fallback | Medium-high | Need adversarial graph fixtures and runtime bounds; do not chase percentage alone. | `coverage-symbolic-graph` |
| Error-string stub coverage | Low | No current blocker worth spending test budget on; revisit only when public error enums change. | `coverage-core-structures` |

## Day 9 Handoff

Day 9 should keep dead-code coverage-gap notes separate from line-coverage
reports:

- line coverage answers which executed tests touched source lines;
- dead-code workflow answers which functions, translation units, or generated
  notes need triage;
- large-matrix guardrails answer bounded structural/report checks;
- none of those surfaces automatically proves reviewed numerical behavior.

The Day 9 architecture should reuse the owner labels above when dead-code
coverage-gap notes identify source families, but it should not turn dead-code
findings into removal-ready proof.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Coverage gaps are ranked by risk, not by convenience. | Complete | Risk rubric ranks public workflow, numerical risk, corpus availability, and claim impact above uncovered-line percentage. |
| Reviewed coverage does not absorb optional or smoke-only paths silently. | Complete | Reviewed-versus-supplemental table keeps aggregate coverage, optional corpus, smoke, expensive, and experimental paths separate. |
| Every residual coverage gap has blocker and future-owner notes. | Complete | Residual queue assigns each gap a blocker and owner label. |
