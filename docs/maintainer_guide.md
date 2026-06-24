# Maintainer Guide

This guide is the maintainer-facing policy home for repository-wide quality
contract interpretation, documentation ownership, and a few stable norms that
should not keep getting re-explained inside `README.md`, tutorial prose, or
public headers.

It is intentionally narrower than a full developer handbook. It explains how
to read the maintained command surfaces and where policy lives. It does not
replace the executable truth in `Makefile`, scripts, CI workflows, or API-local
header contracts.

## Audience

This document is for:

- maintainers
- high-context contributors doing repo-wide cleanup
- reviewers evaluating quality-contract or documentation-ownership claims

This document is not the primary entry point for:

- first-time library users
- API consumers learning one solver
- benchmark/example users looking for command syntax

Those audiences should start with:

- [README](../README.md)
- [tutorial](tutorial.md)
- [benchmarks/README](../benchmarks/README.md)
- [examples/README](../examples/README.md)

## Authoritative Surfaces

Repository policy and executable truth are not the same thing.

Executable truth stays with:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- CI workflows under `.github/workflows/`
- public headers for API-local call-site caveats
- `tests/test_framework.h` for live opt-in test wrapper semantics

This guide owns:

- how to interpret those surfaces
- which surface is authoritative for which kind of claim
- where maintainer-only policy should live instead of spreading through README

Command-detail boundary:

- keep wrapper expansion, rerun guidance, build-tree paths, and other
  executable command detail in `Makefile`
- keep dead-code workflow execution detail in `Makefile`,
  `scripts/deadcode_workflow.sh`, and `scripts/deadcode_report.py`
- use this guide for repository-wide interpretation of those surfaces, not as a
  shadow command reference

## Reviewed Baseline and Warning Authority

### Strongest local reviewed baseline

The strongest maintained local reviewed baseline is:

```bash
make quality-review-full
```

Interpretation:

- this is the strongest local reviewed baseline command
- it composes the reviewed Makefile path and the reviewed CMake parity path
- it is the right default proof point for local “current branch is in the
  reviewed baseline” claims unless a narrower claim is being made
- exact wrapper expansion and rerun guidance should stay with the
  `Makefile` target help

### Reviewed CMake parity

The maintained shared parity surface is the reviewed CMake path:

```bash
make quality-review-cmake
ctest -N --test-dir build/quality-review-cmake
```

Interpretation:

- use `ctest -N` to confirm the maintained suite count when truthfulness about
  the active parity surface matters
- use the full reviewed CMake path when claiming CMake parity still passes
- keep configure/build/ctest command detail in the `Makefile` target help

### Repository-wide warning-clean claims

Repository-wide warning claims should use the Sprint 30 authoritative warning
docs and workflow:

- [Compile Hygiene Playbook](planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md)
- [Rebuild Workflow](planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md)

Interpretation:

- the Apple Clang CMake full-tree inventory remains the authoritative warning
  proof for repository-wide warning claims
- `Makefile all` remains a narrower library-build cross-check, not the
  repository-wide warning authority
- supported build surfaces define the warning-quality bar, not only the
  easiest local command

## Dead-Code Workflow Meaning

The dead-code workflow is separate from the normal lint and test surfaces:

```bash
make deadcode
make deadcode-report
make deadcode-check
```

Interpretation:

- `make deadcode` refreshes raw dead-code evidence
- `make deadcode-report` regenerates the classified report outputs
- `make deadcode-check` is a report-completeness gate, not a zero-findings
  claim
- keep exact emitted report wording and execution sequencing local to the
  `Makefile` and dead-code scripts

How to read the results:

- treat the workflow as conservative evidence rather than full reachability
  proof
- exported installed-header symbols remain manual-review items, not automatic
  deletion candidates
- dead-code noise and secondary static-analysis buckets are supporting context,
  not automatic cleanup authority by themselves

Operational constraint:

- run the `deadcode*` targets serially because they share
  `build/deadcode-cmake` and `build/deadcode/`

### Current residual dispositions

The remaining quality/platform residuals are intentionally narrower than a
generic “platform cleanup” bucket:

- serialized dead-code execution remains the current operational limit because
  the workflow still shares one build/artifact topology
- macOS dead-code remains staged pending fresh measurement rather than
  speculative enablement
- Windows keeps the reviewed CMake subset enforced while the Makefile reviewed
  wrappers and dead-code flow remain staged
- coverage remains a live supplemental signal and should not be treated as an
  unresolved reviewed-baseline residual unless a new contradiction appears

Interpretation:

- keep these residual dispositions explicit across maintained surfaces
- do not imply that staged limits are already solved
- do not widen the repo into platform-expansion work without fresh
  measurement-backed justification

## Packaging and ABI Contract

The maintained packaging surface is intentionally narrower than a full
shared-library product story.

Current authoritative packaging contract:

- the shipped install/export surface is real and maintained
- the maintained release shape is static-first
- downstream `pkg-config` and `find_package(Sparse)` both describe that same
  installed static archive surface
- version metadata is single-sourced from the repo `VERSION` file and
  propagated through the generated install artifacts
- the exported CMake package version file is exact-version only
- current package-version metadata should not be described as a broad
  dynamic-ABI guarantee that the repo does not review

Interpretation:

- improve packaging clarity and install ergonomics without overstating binary
  compatibility promises
- treat any future shared-library or wider ABI claim as a separate product
  contract with its own validation and platform ownership
- keep platform truth explicit: Linux is still the strongest reviewed source of
  truth, macOS remains narrower with supplemental install validation, and
  Windows remains the reviewed CMake subset and install-consumer lane

Focused install/package regression ownership:

- `tests/test_install.sh` is the local Unix-side proof for Make
  install/uninstall plus `pkg-config`
- `tests/test_cmake_install.sh` is the local Unix-side proof for CMake
  install/export plus `find_package(Sparse)`
- macOS CI carries only a narrower supplemental Make install/`pkg-config`
  verification lane
- Windows does not currently claim a separate reviewed install-validation lane;
  it keeps the reviewed CMake subset plus the CMake-first consumer story

## Capability Surface Ownership

Sprint 74 moved the highest-value bounded capability seams without widening the
shipped product claim beyond what the live code and proof now support.

Current maintained interpretation:

- reviewed builds still default to the 32-bit `idx_t` lane
- wider indices are now a bounded compile-time contract through
  `SPARSE_IDX_BITS`, not a hand-edited typedef story
- the strongest touched public dense-scalar seam now routes through
  `sparse_scalar_t`
- current shipped scalar support still remains real-only `double`
- later scalar breadth and later algorithm-family widening remain explicitly
  deferred

Interpretation:

- caller-facing docs should present the width lane as compile-time-selectable,
  but should not imply that the whole repo is already broadly 64-bit-modernized
- caller-facing docs and touched public headers may use `sparse_scalar_t` as
  the dense-scalar owner on the shared matrix-shell helper seam plus the
  iterative/eigs seam
- maintainers should keep the scalar wording explicit: this is bounded public
  preparation for later widening, not proof of complex support or broad
  numeric genericity
- install/export, reviewed-platform, and ABI wording should stay unchanged
  unless a later sprint actually moves those contracts

Current maintained proof ownership after Sprint 84 Day 6:

- `tests/test_sparse_matrix.c` owns the width-contract proof surface:
  - `SPARSE_IDX_BITS`
  - `IDX_MAX`
  - `sparse_idx_bits()`
  - `sparse_scalar_t` on the shared matrix-shell helper seam
  - `sparse_scalar_bits()` on the shared matrix-shell public contract
- `tests/test_iterative.c` owns the iterative public scalar seam:
  - `sparse_scalar_t` matrix-free callback vectors
  - `sparse_scalar_bits()` on the iterative public contract
- `tests/test_eigs.c` owns the eigensolver public scalar seam:
  - `sparse_scalar_t` caller-owned result buffers and option fields
  - `sparse_scalar_bits()` on the eigensolver public contract
- `tests/test_qr.c` owns the QR public scalar seam:
  - `sparse_scalar_t` caller-owned solve buffers
  - `sparse_scalar_t` QR helper output buffers on the widened public header
- `tests/test_chol_csc.c` owns the bounded direct-family maintained external
  differential lane:
  - Cholesky CSC SPD solves checked against an external-process dense reference
    solve
  - fixture-backed SuiteSparse SPD coverage on `nos4` and `bcsstk04`
  - maintained proof stays family-local to the direct-family SPD Cholesky path

Interpretation:

- examples and docs remain support surfaces on this lane
- do not imply that touched capability wording replaces the focused proof
  owners above
- do not reinterpret `bench_chol_csc` or examples as oracle owners for this
  lane
- do not imply that every solver family now has maintained external
  differential proof
- keep `include/sparse_svd.h` and broader capability widening explicitly
  deferred until a later sprint actually changes those contracts

## Configuration Surface Ownership

Epic 6 Phase 1 moved the highest-value analysis/reorder env-var controls onto
the public typed `sparse_analysis_opts_t.reorder_opts` surface.

Current precedence:

1. explicit typed option value
2. legacy compatibility override when the typed field stays unspecified
3. internal default policy

Interpretation:

- caller-facing docs and headers should present the typed path as the preferred
  control surface
- env vars should be described as compatibility overrides, not as the primary
  front door for new callers
- maintainer-facing docs should keep the precedence rule explicit so future
  cleanup does not drift back into contradictory wording

Current public typed analysis/reorder controls include:

- supernodal etree postorder
- ND root bisection mode
- ND root spectral cutoff
- ND coarsening strategy
- ND coarsest bisection strategy
- ND separator-lift strategy
- ND separator-lift weight scheme
- ND coarsening floor-ratio divisor

Current residual deferred configuration queue:

- compatibility-only legacy alias:
  - `SPARSE_ND_SUPERNODAL_POSTORDER`
- internal/default-policy-only analysis-time control:
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- compatibility-first FM policy overrides lowered through one internal owner:
  - `SPARSE_FM_FINEST_STRATEGY`
  - `SPARSE_FM_ENSEMBLE_STRATEGIES`
  - `SPARSE_FM_FINEST_PASSES`
  - `SPARSE_FM_INTERMEDIATE_PASSES`
  - `SPARSE_FM_ANNEALING_SCHEDULE`
  - `SPARSE_FM_THICK_RESTART_PERTURB`
  - `SPARSE_FM_GAIN_NOISE_SCHEDULE`
- explicitly deferred developer-only debug/profile surfaces:
  - `SPARSE_ND_PROFILE`
  - `SPARSE_QG_PROFILE`
  - `SPARSE_HCC_DEBUG`
  - `SPARSE_FM_ENSEMBLE_DEBUG`
  - `SPARSE_FM_THICK_RESTART_DEBUG`
  - `SPARSE_FM_ANNEALING_DEBUG`
  - `SPARSE_FM_GAIN_NOISE_DEBUG`

Interpretation:

- do not silently promote deferred env vars into the public API
- recognized FM compatibility env vars now parse once at the graph
  orchestration boundary and lower into one internal typed FM
  policy/runtime contract
- the refinement subsystem is no longer a second independent FM parser
- that narrowed internal ownership does not by itself create a public typed FM
  option family
- do not imply that the remaining env-var queue is gone; it is now smaller and
  intentionally bounded
- when future sprints move another control, update the typed path, the
  precedence wording, and this residual queue together

Current maintained proof ownership after Sprint 73 Day 12:

- `tests/test_graph.c` owns the graph/FM compatibility and internal-precedence
  proof surface for:
  - FM-family compatibility env behavior
  - `SPARSE_HCC_DEBUG` internal override precedence
- `tests/test_reorder_nd.c` owns the ND typed/default/env and internal-
  precedence proof surface for:
  - typed analysis ND controls overriding compatibility env vars
  - internal/default-policy ND fallback behavior
  - `SPARSE_ND_PROFILE` internal override precedence
- `src/sparse_reorder_amd_qg.c` and `SPARSE_QG_PROFILE` remain explicitly
  deferred support-only context:
  - no new proof owner should be implied for that lane until a later sprint
    actually changes its maintained contract
- examples and benchmarks stay non-owner support surfaces on this lane:
  - `examples/example_analysis.c` remains adoption/teaching context
  - `bench_reorder` and `bench_amd_qg` remain benchmark/reporting context
  - they do not replace the focused proof owners above

## Documentation Ownership Rules

Sprint 48 exists because too much maintainer policy drifted into user-facing
docs. Use these ownership rules going forward.

### `README.md`

`README.md` should stay the user/operator entry point.

It should keep:

- quick-start material
- build/test essentials
- high-level feature map
- concise operator-quality command map
- compact cross-platform quality table
- direct links to deeper docs

It should not become the full maintainer-policy home again.

### `docs/maintainer_guide.md`

This guide should own repository-wide maintainer policy such as:

- reviewed baseline interpretation
- warning authority
- dead-code meaning
- documentation ownership rules
- lifecycle/cancellation maintainer expectations
- stable style/norm reminders that affect multiple docs

### `docs/tutorial.md`

The tutorial should keep user-facing teaching flow and behavioral guidance
needed to use the library.

It should not carry long maintainer-policy blocks when a concise reference to
this guide is enough.

### Public headers

Public headers should keep concise API-local caveats needed at call sites.

They should not expand into long maintainer-policy explanations if the same
policy is already owned here.

### Local benchmark/example READMEs

`benchmarks/README.md` and `examples/README.md` should keep local usage details
and surface-specific notes.

They should not absorb repo-wide quality policy or warning-policy prose.

## Lifecycle and Cancellation Expectations

Maintainers should treat lifecycle and cancellation policy in two layers.

API-local truth:

- stays in the relevant public headers
- stays in focused tutorial prose when it teaches usage

Maintainer interpretation:

- belongs here when the point is policy ownership, documentation placement, or
  cross-surface consistency

Current stable interpretation:

- in-place direct factorization paths can legitimately carry cancellation caveat
  wording in local headers because users need that at the call site
- iterative solvers and eigensolvers generally do not need the same kind of
  input-mutation caveat because they do not factor into `A`
- long repeated lifecycle explanations across README, tutorial, and headers are
  a documentation smell; keep the concise local truth and move the broader
  policy explanation here

Current direct-family interpretation after Sprint 63:

- one-shot LU / Cholesky / LDL^T remain first-class/default peer entry points
- invalid LU pivot/reorder enums and invalid Cholesky reorder/backend enums
  should reject before reorder or factor mutation begins
- stable-pattern repeated direct reuse belongs on the explicit
  `sparse_analyze()` / `sparse_factor_numeric()` / `sparse_factor_solve()` /
  `sparse_refactor_numeric()` lifecycle
- that public repeated-run lifecycle preserves symbolic/permutation setup
  across successful refactors and preserves the previous usable numeric factor
  state on refactor failure
- the large-`n` CSC-backed Cholesky lane now follows that same old-factor-
  preservation rule on both same-pattern non-SPD failure and obvious nnz drift
- the public repeated-run LDL^T lifecycle now also has explicit same-pattern
  parity coverage on the large indefinite KKT lane, including a bounded
  large-`n` CSC-backed property follow-through
- reordered LU and reordered Cholesky one-shot attempts can preserve the
  caller-owned matrix because they factor a temporary reordered working copy
  and publish back only on success
- no-reorder linked-list Cholesky cancellation remains intentionally
  non-bit-identical because the upper triangle is stripped before the first
  emission
- LDL^T keeps the cleanest cancellation story because factor state is owned
  separately from the input matrix

Current maintained proof ownership after Sprint 79 Day 6:

- `tests/test_reorder_nd.c` owns the shared ND compatibility/default-policy
  convergence proof surface
- `tests/test_chol_csc.c` owns the family-local large-`n` analysis-backed
  Cholesky CSC handoff proof surface
- `tests/test_chol_csc.c` also owns the family-local Cholesky CSC publish-back
  ownership proof surface:
  - a writeback-produced shell is factored, solve-ready, and carries the
    published reorder permutation payload
- `tests/test_integration.c` owns the public one-shot vs explicit repeated-run
  Cholesky parity and failure-preservation contract
- `tests/test_integration.c` also owns the public repeated-run LDL^T lifecycle
  oracle surface:
  - same-pattern indefinite KKT reuse remains aligned with the one-shot LDL^T
    lane
  - the large-`n` same-pattern LDL^T path above the CSC threshold remains
    aligned with the one-shot CSC-backed LDL^T lane
- `tests/test_integration.c` also owns the matrix-shell reset boundary:
  - `sparse_reset_perms()` invalidates stale reordered one-shot solve
    compatibility and recovers a plain matrix shell
- `tests/test_fuzz.c` owns the bounded seeded generative follow-through for the
  large-`n` CSC-backed lifecycle parity lanes:
  - Cholesky repeated-run lifecycle parity
  - LDL^T repeated-run lifecycle parity
- example surfaces stay example-side:
  - `examples/example_analysis.c` teaches the repeated-run lifecycle
  - it does not replace the regression owners above
- benchmark surfaces stay benchmark-side:
  - `bench_refactor` / `bench_refactor_csc` prove retained repeated-run direct
    workflow/performance behavior
  - `bench_refactor_csc --indefinite-kkt` is the bounded benchmark-side LDL^T
    repeated-run throughput/proof surface
  - `bench_chol_csc` proves the maintained backend/path measurement surface
  - they do not replace the family-local, public oracle, or property ownership
    above

Current platform-confidence interpretation after Sprint 68 Day 11:

- Linux and macOS still exercise the full `test_fuzz` binary in their direct
  `make test` / reviewed local paths, so the bounded seeded generative
  lifecycle property lanes are part of those proof surfaces
- Windows still excludes `test_fuzz` from the reviewed CMake subset, so that
  property lane must not be implied as reviewed Windows evidence
- this is a narrow confidence-boundary note only; it does not reopen the
  broader staged Windows exclusions or claim new platform parity

Current deferred direct-usability queue:

- no-reorder linked-list Cholesky bit-identical cancellation restoration
- broader CSC progress-callback parity beyond the landed bounded Cholesky
  orchestration checkpoints, plus any later LDL^T callback follow-through
- any broader LDL^T / QR wording follow-through only if a new contradiction
  appears
- broader direct-family docs/examples simplification outside the bounded Sprint
  62 surfaces

## Backend-Aware Performance Surface Ownership

Sprint 64's first backend-aware landing is intentionally narrower than a
general backend framework.

Current stable interpretation after Sprint 64 Day 12:

- the first backend-aware lane is local to CSC supernodal Cholesky
- the default shipped dense-kernel descriptor for that lane remains
  `builtin`
- the bounded direct-family backend-aware surface now extends one layer wider
  after Sprint 82 Day 9:
  - Cholesky CSC owns the first optional dense-kernel runtime seam
  - LDL^T CSC now also owns a bounded optional dense-factor runtime seam
  - both still preserve the builtin self-contained path as the default product
    route
- `bench_chol_csc` is the maintained benchmark-side proof surface for:
  - linked-list baseline timing
  - CSC scalar timing
  - CSC supernodal timing
  - active dense-kernel descriptor identity
  - active supernodal panel-solve capability identity
- `tests/test_ldlt.c` is the maintained family-local proof surface for the
  bounded LDL^T backend/runtime follow-through:
  - builtin env-selection proof
  - optional Accelerate env-selection proof
  - solver-visible forced-CSC correctness through the widened selector seam
- the benchmark path fields should stay read as bounded proof signals:
  - `csc_scalar_path = scalar`
  - `csc_supernodal_path = supernodal`
  - `csc_supernodal_dense_kernel = builtin` on the default build
  - `csc_supernodal_panel_solver = batched_panel` on the default build
- the Sprint 75 Day 10 callback / cancel semantics remain test-owned in
  `tests/test_integration.c`; do not reinterpret `bench_chol_csc` as the
  owner of public progress/cancel truth
- `SPARSE_ERR_BACKEND_CONTRACT` is a real public error code, but its meaning is
  intentionally narrow:
  - the caller contract was valid
  - the selected internal backend-owned helper/callback contract failed
  - do not collapse this back into `SPARSE_ERR_BADARG`
  - do not over-document it as a generic user-tuning failure mode

Current deferred backend/performance queue:

- any later QR / SVD backend layering only if a later sprint justifies it
- optional build-option or pluggable-kernel widening only if the self-contained
  default build and fallback truthfulness stay explicit
- broader benchmark-governance consolidation outside the bounded Sprint 64
  proof refresh
- any later LDL^T widening beyond the bounded Day 9 dense-factor seam only if a
  later sprint justifies more than the current family-local runtime parity

Interpretation:

- keep backend-aware path claims local to the surfaces that actually prove
  them
- prefer benchmark-side measurability and header-local truth over broad README
  architecture marketing
- treat the default self-contained path as authoritative until a later sprint
  lands and validates a wider backend story

## Benchmark Governance Ownership

Current stable interpretation after Sprint 65 Day 9:

- canonical maintained performance surface:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- regression-sensitive runtime lane:
  - `bench_scaling`
  - `bench_fillin`
  - `bench_colamd`
  - `bench_reorder --skip-factor`
  - bounded adjacent lane:
    - `bench_amd_qg`
- exploratory or broader comparison lane:
  - `bench_main`
  - `bench_convergence`
  - `bench_svd`
  - `bench_bicgstab`
  - `bench_eigs`
  - broader `bench_reorder`

Canonical output ownership:

- all four canonical maintained surfaces should expose stable row identity with:
  - `benchmark`
  - `category`
  - `matrix`
  - `scenario`
- direct canonical surfaces keep their path/backend-specific proof fields:
  - `speedup_refactor`
  - `csc_scalar_path`
  - `csc_supernodal_path`
  - `csc_supernodal_dense_kernel`
  - `csc_supernodal_panel_solver`
- iterative/eigensolver canonical surfaces keep their repeated-run proof fields:
  - one-shot timing
  - reuse timing
  - speedup
  - last-run convergence/residual agreement fields

Current threshold-free reporting surface:

- `make bench-canonical-report`
  - writes one CSV per canonical maintained benchmark under:
    - `build/bench-reports/canonical/`
  - accepts `BENCH_CANONICAL_REPORT_LABEL=<label>` as the bounded comparison
    label override
  - writes `manifest.txt` with:
    - exact fixture/command mapping
    - explicit artifact inventory
    - generated timestamp
    - bounded report label from `BENCH_CANONICAL_REPORT_LABEL`
    - git commit / branch when locally available
  - writes `index.tsv` with one structured row per emitted canonical artifact
  - is acceptable for local before/after comparison or CI artifact capture
  - is intentionally not a timing threshold gate
  - should stay limited to the canonical maintained surface unless a later
    sprint proves a wider report remains cheap and stable

Ownership split:

- benchmark binaries own the emitted fields and their semantics
- `benchmarks/README.md` owns the benchmark-local schema explanation
- `README.md` owns only the compact top-level canonical-surface summary
- this maintainer guide owns the authoritative canonical / runtime /
  exploratory classification

Interpretation:

- do not widen the canonical maintained performance surface casually
- do not turn the runtime lane into threshold-heavy pseudo-governance
- do not reinterpret `bench-canonical-report` as a pass/fail portability claim
- do not let exploratory benchmark breadth blur the smaller claim-bearing
  maintained surface

## Stable Repo Norms

### Non-default option examples

Use designated initializers in README/tutorial/header/example snippets when
teaching non-default option behavior.

Reason:

- evolving option structs stay clearer and less brittle when examples name the
  non-default fields explicitly

### Historical evidence vs live test truth

Do not keep retired targets, old measurements, or dormant experiment evidence
as commented-out active-suite scaffolding.

Put that material in:

- `docs/planning/`

Live non-default test semantics stay with:

- `RUN_TEST_SLOW(...)`
- `RUN_TEST_EXPERIMENTAL(...)`
- `SKIP_TEST(...)`

in:

- `tests/test_framework.h`

### Tree-mutating local modes

Some local modes intentionally rebuild the tree in an alternate configuration,
for example:

- `make sanitize`
- `make asan`
- `make sanitize-all`
- `make tsan`
- `make omp`
- `make coverage`
- `make coverage-lcov`
- `make coverage-gcovr`

When returning to the normal direct or reviewed path, reset with:

```bash
make clean
```

## Cross-Reference Guidance

When editing docs, prefer this pattern:

1. keep local truth where the user needs it
2. keep maintainer-only policy here
3. link rather than repeat when the repeated text is not locally necessary

Good examples:

- README linking here for maintainer policy
- tutorial linking here for policy interpretation while keeping user-facing
  behavior guidance local
- headers keeping short caveats while avoiding long repeated repo-policy blocks

Bad examples:

- restating the full reviewed-baseline contract in multiple user-facing docs
- duplicating dead-code interpretation in README, scripts, and guide prose
- using README as both quick-start and full maintainer handbook
