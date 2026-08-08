# Day 1 Runtime Governance Intake

## Purpose

Day 1 establishes the Sprint 142 baseline before changing runtime/backend
behavior. It consumes the Sprint 141 `runtime_backend` defer row as a
governance handoff, identifies the initial runtime/backend surfaces, maps
project-plan items to day owners, and records claim boundaries and validation
expectations.

## Authoritative Inputs

| Input | Day 1 use |
| --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 142 section | Sprint scope, items, estimates, and deliverables. |
| `docs/planning/EPIC_12/SPRINT_142/PLAN.md` | Day-by-day execution plan. |
| `docs/planning/EPIC_12/SPRINT_141/artifacts/day14-closeout-and-sprint142-handoff.md` | Primary handoff for runtime/backend governance. |
| `docs/planning/EPIC_12/SPRINT_141/RETROSPECTIVE.md` | Sprint 142 readiness fields and residual debt. |
| `tests/corpus/manifests/report_families.tsv` | Source-controlled `runtime_backend` defer row and report-family policy context. |
| `scripts/normalize_report_index.py` | Current normalized report-index and freshness behavior for runtime/backend rows. |
| Runtime/backend source, headers, tests, scripts, benchmarks, and docs | Initial surface map for Days 2-3 audit work. |

## Sprint 141 Handoff Interpretation

Sprint 141 closed report normalization and intentionally left one narrow
runtime/backend governance handoff:

- audit runtime controls;
- define precedence among typed options, compile-time flags, environment
  overrides, backend fallback, and deterministic behavior;
- promote selected high-value controls into typed options or explicitly defer
  them as maintainer-only;
- add local sentinel rows only where they improve regression visibility;
- preserve non-claims for portable performance, platform support, package/ABI
  support, hosted CI proof, and state-of-the-art status.

The `runtime_backend` row is therefore not missing generator work. It is a
policy/product-governance input for Sprint 142.

## Initial Runtime/Backend Surface Map

| Surface | Representative files or commands | Initial Day 1 interpretation |
| --- | --- | --- |
| Cholesky backend dispatch | `include/sparse_cholesky.h`, `include/sparse_matrix.h`, `src/sparse_cholesky.c`, `src/sparse_chol_csc.c`, `tests/test_chol_csc*.c` | Public typed backend selector with AUTO routing by `SPARSE_CSC_THRESHOLD` and `used_csc_path` observability. |
| LDLT backend dispatch | `include/sparse_ldlt.h`, `src/sparse_ldlt.c`, `src/sparse_ldlt_csc*.c`, `tests/test_ldlt*.c` | Public typed backend selector mirroring Cholesky with AUTO routing, forced backends, and fallback caveats. |
| Dense helper selection | `src/sparse_chol_csc_supernodal.c`, `src/sparse_ldlt_dense.c`, `tests/test_chol_csc_supernodal.c` | Environment-selected dense helpers with builtin/external/accelerate requests and fallback descriptors. |
| Eigensolver backend selection | `include/sparse_eigs.h`, `src/sparse_eigs*.c`, `tests/test_eigs*.c`, `benchmarks/bench_eigs.c` | Public typed backend selector with AUTO thresholds, LOBPCG/preconditioner routing, and `backend_used` telemetry. |
| OpenMP runtime controls | `Makefile`, CMake, README, iterative/eigs/matvec tests | Compile-time OpenMP enablement plus runtime thread context such as `OMP_NUM_THREADS`. |
| Graph/ND/FM controls | `src/sparse_graph*.c`, analysis/reorder tests, maintainer docs | Mix of typed analysis controls and compatibility/diagnostic environment variables. |
| Analysis typed controls | `include/sparse_analysis.h`, `src/sparse_analysis.c`, reorder and direct-solver tests | Public analysis-time controls for reorder/supernodal/ND behavior with compatibility env override history. |
| Runtime sentinels | `make performance-sentinels`, `scripts/performance_sentinels.sh`, `make large-matrix-guardrails`, `scripts/large_matrix_guardrails.sh` | Existing local report bundles with hard wall-check lane, threshold-free context rows, backend fields, and non-claim boundaries. |
| Normalized report index | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, `tests/corpus/manifests/report_families.tsv` | Maintained path for runtime/backend defer and future sentinel row visibility. |

## Item-To-Day Ownership

| Item | Day owners | Day 1 decision |
| --- | --- | --- |
| Item 1: Runtime Control Audit | Days 1-3 | Intake, inventory, dispatch/fallback audit. |
| Item 2: Precedence Contract | Days 4-5 | Design and implement selected precedence behavior. |
| Item 3: Typed-Control Batch | Days 6-7 | Select, implement, or explicitly defer controls. |
| Item 4: Sentinel Expansion | Days 8-9 | Design and implement bounded local sentinel rows. |
| Item 5: Docs and Examples | Day 10 | Align docs/examples with earned behavior. |
| Item 6: Validation | Days 11-12 | Focused and required full validation. |
| Item 7: Closeout | Days 13-14 | Claim closure, Sprint 143 handoff, final validation. |

## Initial Validation Register

| Change type | Day 1 validation expectation |
| --- | --- |
| C/header runtime changes | Focused backend/runtime tests first, then `make format && make lint && make test`. |
| Report-index or Python changes | Python compile, focused Python tests, corpus schema validation, normalized index checks, and freshness checks. |
| Sentinel script/report changes | Focused script validation, generated-output ignored checks, and normalized report-index/freshness checks. |
| Documentation-only changes | `git diff --check`, trailing-whitespace scans, and command/path consistency review. |
| Build registration changes | Source-list/build parity plus focused Make/CMake compile or test target. |

## Initial Non-Claims

Sprint 142 starts with these boundaries:

- no portable performance claim from benchmark or sentinel rows;
- no broad backend portability claim across platforms, compilers, dense
  libraries, or OpenMP runtimes;
- no package-manager, shared-library, dynamic-linking, or ABI support claim;
- no hosted CI proof from local runtime rows;
- no broad solver correctness or corpus-completeness claim from dispatch
  evidence;
- no environment variable becomes public API unless explicitly promoted;
- no state-of-the-art claim.

## Stop Conditions

- Stop if a control promotion would imply package/ABI decisions reserved for
  Sprint 143.
- Stop if a sentinel requires portable timing or machine-class claims.
- Stop if a backend fallback policy cannot be validated in the current sprint.
- Stop if generated local reports would need to be committed as proof.
- Stop if required validation fails.

## Day 1 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 142 project-plan item has a day-level owner. | Complete | Item-to-day table in this artifact and `WORKING_NOTES.md`. |
| The Sprint 141 runtime/backend defer row is represented as a handoff, not unfinished report-index work. | Complete | Handoff interpretation section. |
| Stop conditions are explicit before code or contract changes begin. | Complete | Stop conditions recorded in this artifact and working notes. |
