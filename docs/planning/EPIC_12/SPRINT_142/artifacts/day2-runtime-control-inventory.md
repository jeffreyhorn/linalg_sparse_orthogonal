# Day 2 Runtime Control Inventory

## Purpose

Day 2 builds the Sprint 142 canonical inventory of runtime and backend
controls before any precedence or typed-control changes. The inventory
separates public API controls from build-time flags, compatibility
environment variables, maintainer diagnostics, and local sentinel commands.
It also records the current owner, default behavior, validation signal, user
visibility, documentation status, and ambiguity that must be resolved before
Days 4-7 change any behavior.

## Inventory

| Control surface | Entry points | Control type | Default/current behavior | Current validation | User visibility | Doc status | Initial classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cholesky numeric backend selector | `sparse_cholesky_opts_t::backend`, `SPARSE_CHOL_BACKEND_AUTO`, `SPARSE_CHOL_BACKEND_LINKED_LIST`, `SPARSE_CHOL_BACKEND_CSC`, `SPARSE_CSC_THRESHOLD`, `used_csc_path` | Public typed option plus compile-time threshold | AUTO routes `rows >= SPARSE_CSC_THRESHOLD` to the CSC supernodal path and smaller matrices to the linked-list path; explicit backend forces the selected path; `used_csc_path` reports the selected path. | `tests/test_chol_csc.c`, `tests/test_chol_csc_supernodal.c`, `tests/test_direct_csc_dispatch.c`, integration/fuzz threshold coverage. | Public C API. | Header, README, maintainer docs, and benchmark docs describe the threshold and telemetry. | Public typed backend control. |
| LDLT numeric backend selector | `sparse_ldlt_opts_t::backend`, `SPARSE_LDLT_BACKEND_AUTO`, `SPARSE_LDLT_BACKEND_LINKED_LIST`, `SPARSE_LDLT_BACKEND_CSC`, `SPARSE_CSC_THRESHOLD`, `used_csc_path` | Public typed option plus compile-time threshold | AUTO mirrors Cholesky threshold routing; forced CSC selects the CSC pipeline except the `n == 0` empty case, which reports linked-list because the CSC scalar pre-pass has no meaningful empty input; CSC selected includes batched supernodal completion and scalar pre-pass fallback. | `tests/test_ldlt.c`, `tests/test_ldlt_backend_dispatch.c`, `tests/test_ldlt_csc.c`, `tests/test_direct_csc_dispatch.c`, `tests/test_direct_csc_regression.c`, integration threshold coverage. | Public C API. | Header, README, maintainer docs, and benchmark docs describe the selector and telemetry. | Public typed backend control. |
| Eigensolver backend selector | `sparse_eigs_opts_t::backend`, `SPARSE_EIGS_BACKEND_AUTO`, `SPARSE_EIGS_BACKEND_LANCZOS`, `SPARSE_EIGS_BACKEND_LOBPCG`, `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`, `SPARSE_EIGS_THICK_RESTART_THRESHOLD`, `SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`, `backend_used`, `peak_basis_size`, `used_csc_path_ldlt` | Public typed option plus compile-time thresholds and result telemetry | AUTO prefers LOBPCG when a preconditioner exists, `n >= SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`, and the effective block size is at least 4; otherwise AUTO chooses grow-m Lanczos below `SPARSE_EIGS_THICK_RESTART_THRESHOLD` and thick-restart Lanczos at or above it; successful calls record `backend_used`. | `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c`, `benchmarks/bench_eigs.c`, integration eigensolver coverage. | Public C API. | Header and README describe routing, thresholds, memory telemetry, and LOBPCG conditions. | Public typed backend control. |
| Cholesky dense helper backend | `SPARSE_CHOL_DENSE_BACKEND`, `src/sparse_dense.c`, supernodal dense helper descriptors | Environment selector | Unset or invalid values use builtin kernels; `external`, `blas`, and `lapack` request the external BLAS/LAPACK path; `accelerate` is accepted on Apple builds. Fallback descriptors distinguish requested, selected, and unavailable helper paths in reports. | `tests/test_chol_csc_supernodal.c`, Cholesky CSC/sentinel benchmark rows. | Environment compatibility and maintainer tuning surface, not public typed API. | Benchmark docs and maintainer guide document report fields and non-claims; public docs do not promote it as stable API. | Maintainer/compatibility env control; candidate for explicit deferral or typed promotion decision. |
| LDLT dense helper backend | `SPARSE_LDLT_DENSE_BACKEND`, `src/sparse_ldlt_dense.c`, LDLT dense helper descriptors | Environment selector | Unset or `builtin` uses builtin kernels; `external`, `blas`, and `lapack` request the external path; `accelerate` is accepted on Apple builds. Invalid values fall back to builtin. | LDLT CSC/refactor coverage, `benchmarks/bench_refactor_csc.c`, sentinel backend descriptor rows. | Environment compatibility and maintainer tuning surface, not public typed API. | Benchmark docs and maintainer guide document report fields and non-claims; public docs do not promote it as stable API. | Maintainer/compatibility env control; candidate for explicit deferral or typed promotion decision. |
| OpenMP enablement and thread context | Make `SPARSE_OPENMP`, CMake `SPARSE_OPENMP`, `OMP_NUM_THREADS`, `SPARSE_EIGS_OMP_REORTH_MIN_N` | Compile-time build option plus external runtime context | Serial builds are the default product path; enabling `SPARSE_OPENMP` compiles OpenMP SpMV and eigensolver reorthogonalization; OpenMP team size and affinity remain owned by the OpenMP runtime and caller process; eigensolver reorth parallel regions are gated by `SPARSE_EIGS_OMP_REORTH_MIN_N`. | `tests/test_omp.c`, `make omp`, benchmark/sentinel report fields, CI platform lanes where enabled. | Build/user configuration, but no public per-call thread-control API. | README, maintainer guide, Makefile, CMake, and benchmark docs describe scope and limits. | Build-time feature flag plus external runtime setting; not a typed runtime policy. |
| Optional matrix mutation mutex | Make `SPARSE_MUTEX`, CMake `SPARSE_MUTEX`, `src/sparse_matrix_internal.h` | Compile-time build option | Default off; when enabled, insert/remove paths use per-matrix mutex protection; factorization remains caller-synchronized and is not mutex-protected. | Thread/concurrency tests where supported; Windows pthread-related lanes are staged out. | Build configuration, not runtime/backend selector. | README, maintainer guide, Makefile, and CMake document the option and limits. | Build-time safety option; out of scope for backend precedence except documentation boundaries. |
| Analysis/reorder typed policy | `sparse_analysis_opts_t::reorder_opts`, supernodal postorder enum, ND coarsening/root/coarsest/separator enums, numeric ND fields | Public typed analysis controls | Zero-init leaves fields unspecified; explicit typed values win; legacy compatibility environment variables are consulted only for unspecified typed fields; internal defaults are final fallback. | `tests/test_reorder_nd.c`, `tests/test_graph.c`, `tests/test_reorder_amd_qg.c`, analysis/integration tests, `bench_reorder --reorder-via-analyze`. | Public C API for analysis/reorder behavior. | Header, README, maintainer guide, and sprint artifacts describe typed precedence and compatibility overrides. | Public typed analysis control with compatibility env fallback. |
| Supernodal postorder compatibility override | `SPARSE_SUPERNODAL_POSTORDER`, legacy `SPARSE_ND_SUPERNODAL_POSTORDER` | Compatibility environment variable | Canonical `SPARSE_SUPERNODAL_POSTORDER=on` enables etree postorder when the typed field is default; legacy `SPARSE_ND_SUPERNODAL_POSTORDER` is accepted only when canonical is absent; unrecognized/off/unset values resolve off. | Analysis/reorder tests and maintainer documentation. | Compatibility env only. | Maintainer docs and source comments describe canonical and legacy names. | Compatibility env; typed field owns public behavior. |
| ND and graph policy compatibility overrides | `SPARSE_ND_ROOT_BISECT`, `SPARSE_ND_ROOT_BISECT_MAX_N`, `SPARSE_ND_COARSENING`, `SPARSE_ND_COARSEST_BISECTION`, `SPARSE_ND_COARSEN_FLOOR_RATIO`, `SPARSE_ND_COARSENING_CV_FALLTHROUGH`, `SPARSE_ND_SEP_LIFT_STRATEGY`, `SPARSE_ND_SEP_LIFT_WEIGHT` | Compatibility environment variables and internal policy parsers | Defaults come from `sparse_reorder_nd_default_policy`; typed analysis fields override corresponding compatibility env values where a typed field exists; env values still cover some legacy/deferred internal knobs. | `tests/test_reorder_nd.c`, `tests/test_graph.c`, `src/sparse_reorder_nd.c`, graph/coarsen/separator tests. | Mixed: typed API where promoted, compatibility env elsewhere. | README and maintainer guide distinguish typed controls from env compatibility/deferred knobs. | Mixed public typed/compatibility env; Day 4 needs precedence table. |
| FM and graph diagnostics/tuning | `SPARSE_FM_FINEST_STRATEGY`, `SPARSE_FM_ENSEMBLE_STRATEGIES`, `SPARSE_FM_FINEST_PASSES`, `SPARSE_FM_INTERMEDIATE_PASSES`, `SPARSE_FM_ANNEALING_SCHEDULE`, `SPARSE_FM_THICK_RESTART_PERTURB`, `SPARSE_FM_GAIN_NOISE_SCHEDULE`, `SPARSE_FM_ENSEMBLE_DEBUG`, `SPARSE_FM_THICK_RESTART_DEBUG`, `SPARSE_FM_ANNEALING_DEBUG`, `SPARSE_FM_GAIN_NOISE_DEBUG`, `SPARSE_HCC_DEBUG` | Maintainer environment variables | Defaults use internal FM/graph heuristics; env vars opt into strategy variants, pass counts, perturbation/noise schedules, or stderr diagnostics. | `tests/test_graph.c`, `tests/test_reorder_nd.c`, graph helper coverage. | Maintainer-only/diagnostic unless explicitly promoted later. | Maintainer guide lists these as deferred/internal; public docs should avoid presenting them as stable API. | Maintainer-only env controls. |
| Reorder profiling diagnostics | `SPARSE_ND_PROFILE`, `SPARSE_QG_PROFILE` | Maintainer environment variables | Unset disables profiling; set values enable stderr/wall-clock instrumentation for local diagnostics. | `tests/test_reorder_nd.c`, `tests/test_reorder_amd_qg.c`, source instrumentation paths. | Maintainer diagnostic. | Maintainer guide lists profile knobs as internal/deferred. | Maintainer-only diagnostics; not public performance evidence. |
| SVD low-rank outer-loop selector | `SPARSE_SVD_LOWRANK_OUTER`, `src/sparse_svd.c` | Environment selector | Environment value selects the low-rank SVD outer strategy; currently outside Sprint 142's backend-governance priority list but still affects solver runtime behavior. | SVD and partial-SVD tests cover behavior; exact env coverage owner needs Day 3 verification. | Environment compatibility/maintainer surface. | Not yet part of the Day 1 primary map; needs Day 3 disposition. | Runtime env control to audit, likely maintainer-only unless promoted. |
| Test and benchmark opt-ins | `SPARSE_TEST_SLOW`, `SPARSE_TEST_EXPERIMENTAL`, `SPARSE_TEST_LARGE`, `RUN_BENCH`, test temp-dir variables | Test environment variables | Default tests avoid slow/large/benchmark-only lanes; env vars opt into wider local validation. | Test harness and SuiteSparse/integration tests. | Maintainer/developer workflow only. | README and test docs describe selected opt-ins. | Test-scope controls, not library runtime policy. |
| Runtime evidence commands | `make bench-canonical-report`, `make performance-sentinels`, `make wall-check`, `make large-matrix-guardrails`, `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL`, `BENCH_CANONICAL_REPORT_LABEL` | Maintainer commands and report controls | Canonical reports and most sentinel rows are threshold-free local context; `wall-check` is the hard reviewed wall-time gate; large-matrix supplemental rows remain skipped unless explicitly enabled. | Shell scripts, normalized report-index tests, freshness checks, existing CI/local lanes. | Maintainer evidence workflow, not public runtime API. | Benchmark docs and maintainer guide define support tiers, claim boundaries, and non-claims. | Maintainer evidence controls; no portable performance claim. |
| Package/build link surface | `SPARSE_OPENMP`, `SPARSE_MUTEX`, pkg-config extra libs, CMake install targets | Build and package metadata controls | Static-first install remains current product path; OpenMP/mutex builds add downstream link flags where applicable; no shared-library or broad package-manager claim is implied by runtime governance. | Install/pkg-config tests and CMake consumer lanes from earlier sprints. | Downstream build integration, not runtime behavior. | README, install docs, Makefile, CMake, and prior sprint artifacts document current scope. | Sprint 143 handoff if runtime decisions would affect package/ABI surface. |
| Core compile-time numeric constants | `SPARSE_NODES_PER_SLAB`, `SPARSE_DROP_TOL`, `SPARSE_CSC_THRESHOLD`, eigensolver AUTO thresholds | Compile-time constants | Defaults are built into headers; callers can override at compile time. `SPARSE_CSC_THRESHOLD` and eigensolver thresholds affect backend routing; `SPARSE_DROP_TOL` affects numeric dropping/pivot tolerance behavior. | Broad solver tests plus backend threshold tests. | Advanced build-time customization. | Header comments are the primary authority. | Build-time tuning; routing-related constants need explicit precedence wording. |

## Public vs Maintainer-Only Draft

| Classification | Controls |
| --- | --- |
| Public typed controls | Cholesky backend selector, LDLT backend selector, eigensolver backend selector, analysis/reorder typed policy. |
| Build-time controls | `SPARSE_OPENMP`, `SPARSE_MUTEX`, `SPARSE_CSC_THRESHOLD`, eigensolver AUTO thresholds, `SPARSE_DROP_TOL`, `SPARSE_NODES_PER_SLAB`. |
| Compatibility environment variables | `SPARSE_SUPERNODAL_POSTORDER`, `SPARSE_ND_SUPERNODAL_POSTORDER`, typed-analysis-compatible `SPARSE_ND_*` overrides. |
| Maintainer-only runtime/diagnostic controls | FM strategy/debug env vars, ND/QG profile env vars, dense-helper backend env selectors unless promoted, SVD low-rank outer env selector pending Day 3 disposition. |
| Maintainer evidence controls | `make performance-sentinels`, `make wall-check`, `make large-matrix-guardrails`, `make bench-canonical-report`, `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL`, `BENCH_CANONICAL_REPORT_LABEL`. |
| Test-only controls | `SPARSE_TEST_SLOW`, `SPARSE_TEST_EXPERIMENTAL`, `SPARSE_TEST_LARGE`, `RUN_BENCH`, temp-directory environment variables used by test harnesses. |

## Effect Map

| Effect area | Controls that matter |
| --- | --- |
| Deterministic behavior | Explicit typed backend selectors, analysis typed reorder policy, compatibility env overrides when typed fields are unspecified, FM/ND env strategy choices, OpenMP scheduling where enabled. |
| Performance behavior | CSC/Lanczos/LOBPCG thresholds, dense-helper env selectors, OpenMP enablement and `OMP_NUM_THREADS`, FM/ND strategy controls, `SPARSE_DROP_TOL`. |
| Backend selection | Cholesky/LDLT/eigensolver typed backend selectors, dense-helper env selectors, compile-time AUTO thresholds, shift-invert LDLT telemetry. |
| Package/build behavior | `SPARSE_OPENMP`, `SPARSE_MUTEX`, pkg-config extra link flags, CMake options, static-first install metadata. |
| Platform support | OpenMP runtime availability, Apple Accelerate-only dense-helper requests, Windows staged exclusions for pthread/POSIX-dependent lanes, package/install consumer lanes. |
| Evidence and claim boundaries | Sentinel commands, guardrail commands, normalized report indexes, support-tier fields, claim-boundary fields, freshness gates. |

## Command/Test/Doc Ownership Map

| Owner area | Primary files or commands | Current responsibility |
| --- | --- | --- |
| Public solver backend API | `include/sparse_cholesky.h`, `include/sparse_ldlt.h`, `include/sparse_eigs.h` | Declare typed backend options, defaults, telemetry, and ABI/source-compat notes. |
| Backend dispatch implementation | `src/sparse_cholesky.c`, `src/sparse_ldlt.c`, `src/sparse_eigs.c`, CSC backend sources | Select and publish backend routes, enforce invalid-option errors, and own fallback behavior. |
| Dense helper selection | `src/sparse_dense.c`, `src/sparse_ldlt_dense.c`, supernodal CSC sources | Parse dense-helper env requests and report selected/fallback helper descriptors. |
| Analysis/reorder policy | `include/sparse_analysis.h`, `src/sparse_analysis.c`, `src/sparse_reorder_nd.c`, graph policy internals | Own typed-vs-env precedence and ND/supernodal policy resolution. |
| Graph/FM internals | `src/sparse_graph*.c`, `src/sparse_reorder_amd_qg.c` | Own maintainer-only graph strategy, debug, and profile env behavior. |
| Build controls | `Makefile`, `CMakeLists.txt` | Own OpenMP/mutex enablement and downstream link metadata. |
| Sentinel/report controls | `scripts/performance_sentinels.sh`, `scripts/large_matrix_guardrails.sh`, `scripts/bench_canonical_report.sh`, `scripts/normalize_report_index.py` | Own local evidence generation, report indexing, freshness, and non-claim fields. |
| Validation tests | `tests/test_chol_csc*.c`, `tests/test_ldlt*.c`, `tests/test_eigs*.c`, `tests/test_reorder_nd.c`, `tests/test_graph.c`, `tests/test_omp.c`, report-index tests | Own regression proof for dispatch, env parsing, typed policy, OpenMP surfaces, and report rows. |
| User/maintainer docs | `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md` | Explain public controls, compatibility/deferred controls, sentinel boundaries, and non-claims. |

## Unknowns and Day 3 Risks

- Direct Cholesky and LDLT fallback behavior needs a focused Day 3 pass to
  distinguish "backend selected" from lower-level internal fallback and to
  ensure docs/tests use the same vocabulary.
- Dense-helper env controls are currently useful for local evidence but lack
  public typed equivalents. Day 6 should decide whether to promote a narrow
  typed control, explicitly defer them, or keep them maintainer-only.
- `SPARSE_SVD_LOWRANK_OUTER` affects runtime behavior but was not in the
  initial Sprint 142 priority list. Day 3 should classify it explicitly.
- Some ND/FM env knobs have typed replacements and some do not. Day 4 must
  publish an exact precedence table instead of relying on broad "typed wins"
  language.
- OpenMP `OMP_NUM_THREADS` is caller/runtime-owned. Sentinel rows may record
  it as context but must not treat it as a library-level per-call control.
- Package/link effects of `SPARSE_OPENMP` and `SPARSE_MUTEX` are real but
  belong to build/install ownership. Any change that affects pkg-config,
  CMake exports, ABI, or shared-library claims should be handed to Sprint 143.
- Windows lanes still have staged exclusions around pthread/POSIX-dependent
  tests. Sprint 142 evidence should stay explicit about platform scope.
- Sentinel expansion must preserve the existing local/advisory boundary and
  avoid new portable timing, max-RSS, or machine-class claims.

## Day 2 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| OpenMP, direct-solver, eigensolver, dense helper, backend dispatch, environment, and typed-option surfaces are accounted for. | Complete | Inventory, classification, and effect-map sections. |
| Each control has owner, current behavior, validation, visibility, and documentation status. | Complete | Inventory and command/test/doc ownership map. |
| Ambiguous controls are flagged before precedence design. | Complete | Unknowns and Day 3 risks section. |
