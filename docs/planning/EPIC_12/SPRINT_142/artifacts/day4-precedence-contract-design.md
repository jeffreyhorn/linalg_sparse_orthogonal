# Day 4 Precedence Contract Design

## Purpose

Day 4 defines the maintained precedence contract for runtime/backend controls
before Day 5 changes any implementation or tests. The contract is intentionally
bounded: it clarifies existing typed options, compatibility environment
variables, compile-time flags, fallback semantics, deterministic behavior, and
report interpretation without creating package, ABI, platform, portable
performance, or broad backend claims.

## Global Precedence Rules

| Rank | Control class | Contract |
| --- | --- | --- |
| 1 | Explicit public typed option | A valid explicit typed option wins over AUTO routing, compatibility environment variables, and internal defaults for the surface it controls. Invalid typed values fail with `SPARSE_ERR_BADARG` unless the public API documents a different error. |
| 2 | Public typed `AUTO` or `DEFAULT` value | AUTO/DEFAULT means "library selects by the maintained rule for this surface." It does not mean "environment variables always win." For backend selectors, AUTO resolves through compile-time thresholds and API inputs. For analysis/reorder controls, DEFAULT means compatibility env override may be consulted before the internal default. |
| 3 | Compatibility environment variable | Consulted only for surfaces that explicitly document env compatibility and only when the relevant typed field is unspecified. Unrecognized values fall back to the documented default unless the env surface explicitly documents a hard failure. |
| 4 | Compile-time threshold or feature flag | Compile-time constants and build flags shape AUTO/default behavior but do not override an explicit typed backend request. Examples: `SPARSE_CSC_THRESHOLD`, eigensolver AUTO thresholds, `SPARSE_OPENMP`, `SPARSE_MUTEX`, `SPARSE_DROP_TOL`. |
| 5 | Internal default | Used when no explicit typed option, documented env compatibility value, or compile-time override applies. Defaults must remain deterministic for a fixed binary, input, and environment. |
| 6 | Maintainer diagnostic/report context | Diagnostic env vars, local benchmark settings, `OMP_NUM_THREADS`, and report labels may be recorded by tools but are not public runtime API unless separately promoted. |

## Backend Selection Contract

| Surface | Explicit typed option | AUTO/default rule | Env interaction | Fallback/failure rule | Telemetry wording |
| --- | --- | --- | --- | --- | --- |
| Cholesky backend | `SPARSE_CHOL_BACKEND_LINKED_LIST` and `SPARSE_CHOL_BACKEND_CSC` force the top-level path; invalid enum returns `SPARSE_ERR_BADARG`. | `SPARSE_CHOL_BACKEND_AUTO` selects CSC when `n >= SPARSE_CSC_THRESHOLD`; otherwise linked-list. | No environment variable participates in top-level Cholesky backend selection. | No post-selection top-level fallback. Lower-level CSC contract or numeric failures propagate as errors. | `used_csc_path` reports top-level dispatch selection and may be published before later factor/reorder errors. Public docs should treat success-path telemetry as the stable claim and describe error-path publication as best-effort/diagnostic unless tests harden it further. |
| LDLT backend | `SPARSE_LDLT_BACKEND_LINKED_LIST` and `SPARSE_LDLT_BACKEND_CSC` force the top-level path except `n == 0`; invalid enum returns `SPARSE_ERR_BADARG`. | `SPARSE_LDLT_BACKEND_AUTO` selects CSC when `n >= SPARSE_CSC_THRESHOLD`; otherwise linked-list. | No environment variable participates in top-level LDLT backend selection. | `n == 0` routes linked-list even when CSC is requested. After CSC is selected, internal batched completion may fall back to the resolved scalar-prepass factor; that is an internal CSC completion path, not top-level fallback to linked-list. | `used_csc_path` reports top-level selected path. It does not distinguish batched CSC completion from scalar-prepass completion. |
| Eigensolver backend | Explicit `LANCZOS`, `LANCZOS_THICK_RESTART`, or `LOBPCG` bypasses AUTO; invalid enum returns `SPARSE_ERR_BADARG`. | AUTO priority: eligible preconditioned large/block solve routes to LOBPCG; otherwise `n >= SPARSE_EIGS_THICK_RESTART_THRESHOLD` routes to thick-restart; below threshold routes grow-m. | No environment variable participates in eigensolver backend selection. | Lanczos-family backends ignore user preconditioners outside shift-invert internals; LOBPCG honors valid preconditioners. Error-path `backend_used` is best-effort. | `backend_used` is success-path selected backend telemetry. `peak_basis_size` is backend-specific memory context. `used_csc_path_ldlt` reports the internal LDLT route for shift-invert. |

## Analysis And Reorder Precedence

| Typed field | Compatibility env value when typed field is DEFAULT/unspecified | Internal default | Failure rule |
| --- | --- | --- | --- |
| `supernodal_postorder` | `SPARSE_SUPERNODAL_POSTORDER`, then legacy `SPARSE_ND_SUPERNODAL_POSTORDER` | Off | Invalid typed enum returns `SPARSE_ERR_BADARG`; unrecognized env values resolve off. |
| `nd_root_bisect` | Existing `SPARSE_ND_ROOT_BISECT` parser when no typed override is provided through the lower-level policy path | Multilevel | Invalid typed enum returns `SPARSE_ERR_BADARG`; unrecognized env values resolve multilevel. |
| `nd_root_bisect_max_n` | `SPARSE_ND_ROOT_BISECT_MAX_N` when no positive typed value is provided through the lower-level policy path | `50000` | Negative typed value returns `SPARSE_ERR_BADARG`; out-of-range env values are ignored. |
| `nd_coarsening` | Existing `SPARSE_ND_COARSENING` parser when no typed override is provided through the lower-level policy path | HCC | Invalid typed enum returns `SPARSE_ERR_BADARG`; unrecognized env values resolve HCC. |
| `nd_coarsen_floor_ratio` | `SPARSE_ND_COARSEN_FLOOR_RATIO` when no positive typed value is provided through the lower-level policy path | `100` | Negative or too-large typed value returns `SPARSE_ERR_BADARG`; out-of-range env values are ignored. |
| `nd_coarsest_bisection` | Existing `SPARSE_ND_COARSEST_BISECTION` parser when no typed override is provided through the lower-level policy path | Default routing | Invalid typed enum returns `SPARSE_ERR_BADARG`; unrecognized env values resolve default routing. |
| `nd_sep_lift_strategy` | Existing `SPARSE_ND_SEP_LIFT_STRATEGY` parser when no typed override is provided through the lower-level policy path | Smaller-weight | Invalid typed enum returns `SPARSE_ERR_BADARG`; unrecognized env values resolve smaller-weight. |
| `nd_sep_lift_weight` | Existing `SPARSE_ND_SEP_LIFT_WEIGHT` parser when no typed override is provided through the lower-level policy path | Hybrid | Invalid typed enum returns `SPARSE_ERR_BADARG`; unrecognized env values resolve hybrid. |

The public documentation rule is simpler than the implementation history:
explicit typed analysis fields win, DEFAULT/zero means compatibility env may
be consulted, and internal defaults apply last. FM strategy, profile, and
debug env vars that do not have public typed fields remain maintainer-only
and should not be described as public API.

## Environment-Only Control Contract

| Surface | Current contract | Day 6 disposition needed |
| --- | --- | --- |
| `SPARSE_CHOL_DENSE_BACKEND` | Maintainer/compatibility selector for dense helper requests. Invalid, unavailable, or unsupported requests fall back to builtin. It is report context, not a public stable typed backend control. | Decide whether to explicitly defer or promote a narrow typed dense-helper request. |
| `SPARSE_LDLT_DENSE_BACKEND` | Maintainer/compatibility selector for LDLT dense helper requests. Invalid, unavailable, or unsupported requests fall back to builtin. | Decide whether to add focused invalid-env proof or keep it report-only. |
| `SPARSE_SVD_LOWRANK_OUTER` | Maintainer/runtime selector for low-rank sparse reconstruction path. `on` opts into the outer-product accumulator; unrecognized/off/unset uses dense-intermediate path. | Likely explicit deferral unless Day 6 selects SVD runtime governance. |
| `SPARSE_FM_*`, `SPARSE_ND_PROFILE`, `SPARSE_QG_PROFILE`, `SPARSE_HCC_DEBUG` | Maintainer diagnostic or strategy knobs. They affect local behavior and tests but are not public API. | Explicitly defer as maintainer-only unless a concrete typed user workflow is selected. |
| `SPARSE_TEST_*`, `RUN_BENCH`, temp-dir vars | Test/benchmark workflow controls only. | Keep out of runtime governance except validation docs. |

## Build-Time And Package Boundary

| Control | Contract |
| --- | --- |
| `SPARSE_OPENMP` | Compile-time feature flag. It enables OpenMP regions in selected kernels; it does not create a public per-call thread policy. |
| `OMP_NUM_THREADS` | Caller/OpenMP-runtime setting. Benchmark and sentinel tools may record it as local comparison context; docs must not present it as library-owned runtime control. |
| `SPARSE_EIGS_OMP_REORTH_MIN_N` | Compile-time threshold for OpenMP reorthogonalization bodies. It affects work partitioning inside Lanczos-family backends, not backend identity. |
| `SPARSE_MUTEX` | Compile-time matrix mutation safety option. It is not a backend selector and does not make factorization concurrently safe on the same matrix. |
| pkg-config/CMake link flags | Runtime governance must not change package/link metadata unless a Day 5+ implementation explicitly touches build/install behavior. Package, ABI, shared-library, and package-manager claims remain Sprint 143 handoff material. |

## Fallback And Failure Matrix

| Request/result | Required behavior | Claim boundary |
| --- | --- | --- |
| Invalid public typed enum | Return `SPARSE_ERR_BADARG` before mutating caller-visible state when current tests require preservation. | Public API validation behavior for covered surfaces. |
| Explicit typed backend unsupported by input shape | Return documented shape/badarg/backend error or documented exception (`LDLT n == 0` linked-list path). | Surface-local behavior only. |
| Environment value unrecognized | Fall back to documented default for compatibility env selectors. | Maintainer/compatibility behavior, not public API unless promoted. |
| Optional dense backend unavailable | Fall back to builtin and record request/selected/fallback where report tooling exposes it. | No BLAS, LAPACK, Accelerate, or platform availability claim. |
| Internal CSC completion fallback | Keep vocabulary as "internal completion fallback" under CSC-selected path. | No top-level backend fallback claim. |
| OpenMP runtime unavailable | Non-OpenMP builds remain serial; OpenMP builds rely on configured compiler/runtime. | No portable OpenMP speedup or hosted-platform parity claim. |
| Sentinel timing row | Existing `wall-check` is the only hard timing gate. Other sentinel/benchmark rows remain local/advisory unless separately reviewed. | No portable performance claim. |

## Determinism Requirements

- Public typed options must produce the same dispatch result for the same
  binary, input, and option struct.
- AUTO routes must be deterministic for the same compile-time thresholds,
  dimensions, preconditioner presence, and block-size inputs.
- Compatibility env values must be read consistently by the owning parser and
  must not silently override explicit typed fields.
- Tests that use env vars must unset them after the scoped assertion and must
  not depend on process-global leakage.
- Benchmark and sentinel rows must record build mode, command, artifact,
  backend request/selection/fallback, and thread context when those values
  affect interpretation.
- Examples should use explicit typed options when demonstrating public runtime
  behavior; env vars should appear only in maintainer or diagnostic contexts.

## Public Versus Maintainer-Only Language

| Audience | Allowed wording |
| --- | --- |
| Public API docs | "Set this typed option to force a backend"; "AUTO selects by documented threshold/input rule"; "result telemetry records selected backend on success." |
| Maintainer docs | "Compatibility env override"; "diagnostic/profile env var"; "local advisory sentinel row"; "selected backend versus internal completion path." |
| Benchmark docs | "Local comparison context"; "threshold-free row"; "request/selected/fallback fields"; "same machine/configuration only." |
| Disallowed unless later earned | "Portable performance"; "backend portability guarantee"; "OpenMP runtime policy"; "shared-library/ABI support"; "package-manager support"; "state-of-the-art proof." |

## Validation Scenarios For Day 5+

| Scenario | Proposed owner | Expected proof |
| --- | --- | --- |
| Cholesky explicit backend wins over threshold | Existing `tests/test_chol_csc.c` or dispatch sentinel helper | Small forced CSC reports `used_csc_path == 1`; large forced linked-list reports `0`. |
| Cholesky invalid enum preserves retry path | Existing integration invalid-backend test | `SPARSE_ERR_BADARG`, original matrix can still factor through a valid option. |
| LDLT explicit backend wins over threshold except empty matrix | Existing `tests/test_ldlt_backend_dispatch.c` plus possible focused empty forced-CSC test | Small forced CSC reports `1`; large forced linked-list reports `0`; empty forced CSC reports `0` if hardened. |
| LDLT CSC completion fallback vocabulary | Direct CSC regression tests plus docs review | Tests continue to pass while docs distinguish top-level CSC selection from scalar-prepass completion. |
| Eigensolver AUTO priority | Existing `tests/test_eigs_lobpcg.c` and `tests/test_eigs_thick_restart.c` | Small/no-precond `LANCZOS`, large/no-precond `THICK_RESTART`, large/precond/block>=4 `LOBPCG`. |
| Eigensolver explicit backend bypasses AUTO | Existing eigensolver tests | Explicit LOBPCG without preconditioner records `LOBPCG`; explicit Lanczos below/above thresholds records requested backend. |
| Analysis typed option overrides env | Existing `tests/test_reorder_nd.c` | Typed value wins when conflicting env var is set; DEFAULT permits env compatibility behavior. |
| Dense helper invalid env fallback | Existing Cholesky test; possible LDLT focused test | Invalid dense env resolves builtin and report fields preserve request/selected/fallback. |
| OpenMP context is report-only | `tests/test_omp.c`, sentinel/report docs | OpenMP build tests pass; docs and reports record `OMP_NUM_THREADS` without claiming per-call control. |
| Sentinel row boundaries | `tests/test_normalize_report_index.py`, report-family manifest | Hard-gate rows remain `local_wall_gate`; advisory rows remain local/advisory with backend fields preserved. |

## Day 5 Implementation Guidance

- Prefer documentation and focused tests unless a behavior mismatch is found.
- Do not change public ABI or package metadata for precedence wording alone.
- If a code change is needed, keep it to one precedence surface and run the
  full C/header quality gate.
- Do not promote dense helper, SVD, FM, or diagnostic env vars on Day 5; Day 6
  owns the explicit typed-control selection.
- Do not add timing thresholds or generated report artifacts as committed
  proof.

## Day 4 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Typed options and environment overrides have an explicit ordering. | Complete | Global precedence rules and analysis/reorder precedence table. |
| Fallback behavior is documented without broad platform or performance claims. | Complete | Backend selection contract and fallback/failure matrix. |
| Validation scenarios are concrete enough to implement. | Complete | Validation scenarios and Day 5 implementation guidance. |
