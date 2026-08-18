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

## Support Surface Ownership

Current support surfaces should keep a clear owner split:

- `README.md` owns the project front door, first local build path, compact
  workflow chooser, and links to deeper support surfaces.
- `INSTALL.md` owns operational setup, staged installs, installed-consumer
  detail, and local install-surface validation.
- `benchmarks/README.md` owns benchmark command groups, CSV schema,
  report-artifact meaning, and measurement caveats.
- `examples/README.md` owns executable example selection and example-local
  usage notes.
- `docs/tutorial.md` owns the longer learning path after the README.
- public headers own API-local call-site contracts.
- tests own regression, oracle, and property guarantees.
- `docs/planning/**` owns historical sprint provenance.
- this guide owns the maintainer interpretation of those boundaries.

Support cross-link rules:

- link to `INSTALL.md` when the reader needs install, package, downstream
  consumer, or install-validation detail
- link to `benchmarks/README.md` when the reader needs benchmark command
  syntax, CSV fields, or measurement interpretation
- link to this guide when the reader needs reviewed-platform interpretation,
  proof ownership, warning authority, or documentation-placement policy
- link to planning artifacts only when historical context explains a current
  limitation, compatibility decision, or validation boundary

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

Normalized report-index interpretation:

- `python3 scripts/normalize_report_index.py --family deadcode --check`
  indexes `build/deadcode/report.tsv` when it exists and otherwise emits
  deterministic `not_generated` rows
- `python3 scripts/normalize_report_index.py --family deadcode
  --check-freshness` keeps dead-code rows advisory unless a caller explicitly
  requires generated dead-code evidence
- do not read a fresh dead-code row as a zero-dead-code guarantee; the rows are
  local static-analysis classifications with bucket/disposition context

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
- `BUILD_SHARED_LIBS=ON` is a configure-time rejection under the static-first
  contract, not a supported shared-library mode
- current package-version metadata should not be described as a broad
  dynamic-ABI guarantee that the repo does not review

Interpretation:

- improve packaging clarity and install ergonomics without overstating binary
  compatibility promises
- treat any future shared-library or wider ABI claim as a separate product
  contract with its own validation and platform ownership
- keep platform truth explicit: Linux is still the strongest reviewed source of
  truth and now includes a reviewed static-first package-contract lane; macOS
  now carries reviewed static-first Make install/`pkg-config` and CMake
  install/export proof for the maintained static archive package contract; and
  Windows remains CMake-first with reviewed CTest coverage plus reviewed CMake
  install/downstream validation for the maintained static-first package
  surface

Focused install/package regression ownership:

- `tests/test_install.sh` is the local Unix-side proof for Make
  install/uninstall plus `pkg-config`; it checks exact installed header count,
  static archive/no-shared-artifact install shape, `.pc` prefix/libdir/include
  variables by filesystem identity, installed include and link flags by
  filesystem identity plus expected library flags, exact package version
  resolution, no `Libs.private` stanza for the current self-contained link
  surface, static archive `.pc` description, absence of unsupported
  package/ABI claims in `sparse.pc`, semantic output checks for two
  compile/link/run consumers, and uninstall cleanup
- `tests/test_cmake_install.sh` is the local Unix-side proof for CMake
  install/export plus `find_package(Sparse)`; it checks exact installed
  header count, static imported-target metadata, installed-prefix include and
  archive locations, absence of shared imported metadata, absence of
  unsupported loader/static-shared selector metadata, absence of
  source/build-tree path leaks, static archive `.pc` description, absence of
  unsupported package/ABI wording, exact-version package configure/build/run
  behavior, mismatched-version rejection, and installed consumer
  configure/build/run
- `scripts/static_package_deferral_check.sh` is the local package-contract
  guard that checks `BUILD_SHARED_LIBS=ON` rejection, the explicit static
  CMake target, exact shared deferral blocker wording, absence of unsupported
  shared ABI metadata/selectors, and deferred support wording
- Linux CI carries a reviewed static-first package-contract lane that runs the
  Make install/`pkg-config` proof, CMake install/export proof, and static
  deferral guard
- macOS CI carries reviewed static-first Make install/`pkg-config` and CMake
  install/export proof lanes for the maintained static archive package
  contract; these lanes preserve no shared-library, dynamic ABI,
  runtime-loader, package-manager, static/shared selector, or broad macOS
  platform parity claims
- Windows CI carries reviewed CMake install/downstream validation for the
  maintained static-first package surface; this lane checks installed static
  `.lib`, headers, CMake package metadata, metadata-only `sparse.pc`
  inspection, generated and maintained installed CMake consumers,
  exact-version behavior,
  mismatched-version rejection, absence of DLL/shared imported metadata, and
  absence of unsupported loader/static-shared selector metadata
- Windows `sparse.pc` inspection remains metadata-only and must not be cited as
  Windows `pkg-config` command execution
- Windows still does not claim Makefile parity, `pkg-config` execution parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, or broad Windows parity

Normalized report-index interpretation:

- `python3 scripts/normalize_report_index.py --family package --check`
  expands the package contract into source-controlled proof-owner rows for
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, `sparse.pc.in`,
  `cmake/SparseConfig.cmake.in`, and
  `scripts/static_package_deferral_check.sh`
- package proof-owner rows use `freshness_status=source_controlled`; they
  prove ownership and scope of maintained checks/templates, not that an
  install validation command was just run
- if an install-run result must be cited, run the relevant install validation
  script and keep its generated output separate from the source-controlled
  proof-owner row

Sprint 112 package/platform proof snapshot:

- the selected package tier remains static-first; shared-library packaging and
  dynamic ABI compatibility remain explicit non-claims
- local Make install proof passed through static archive install, no shared
  artifacts, 19 installed headers, `sparse.pc` field validation, exact
  pkg-config version resolution, no `Libs.private` stanza for the current
  self-contained link surface, pkg-config compile/link/run consumers, and
  uninstall cleanup
- local CMake install/export proof passed through static archive install,
  19 installed headers, `SparseConfig.cmake`, `SparseConfigVersion.cmake`,
  `SparseTargets.cmake`, static imported target metadata, installed-prefix
  include/archive paths, no source/build-tree package path leaks,
  exact-version package behavior, mismatched-version rejection, pkg-config
  version reporting, and installed CMake consumer configure/build/run
- Sprint 133 changed `BUILD_SHARED_LIBS=ON` from warning-only behavior to
  configure-time rejection so shared-library requests remain explicit
  deferrals
- Linux remains the strongest reviewed source of truth; Sprint 134 promoted the
  static-first package-contract proof stack to a reviewed Linux CI lane
- macOS keeps the reviewed Apple Clang lane plus supplemental Homebrew GCC, and
  Sprint 144 promotes static-first Make install/`pkg-config` and CMake
  install/export proof to reviewed macOS package lanes for the maintained
  static archive package contract
- Windows keeps the reviewed MSVC CMake-first subset with 59 registered CTest
  tests; Sprint 148 promotes `test_threads`, `test_sprint4_integration`, and
  `test_fuzz` into that CMake subset through portable test-only thread and
  temp-file helpers
- Sprint 149 promotes reviewed Windows CMake install/downstream validation for
  the maintained static-first package surface, including installed static
  `.lib`, headers, CMake package metadata, metadata-only `sparse.pc`
  inspection, generated and maintained installed CMake consumers,
  exact-version behavior,
  mismatched-version rejection, no DLL/shared imported metadata, and no
  unsupported loader/static-shared selector metadata
- Sprint 162 hardens the retained package non-claim boundary: Windows
  `sparse.pc` inspection remains metadata-only, while Windows Makefile
  install/uninstall parity and Windows `pkg-config` command execution parity
  remain unsupported unless a future product decision adds provider-specific
  proof.
- do not infer shared-library support, ABI stability, package-manager support,
  runtime-loader behavior, Windows Makefile parity, Windows `pkg-config`
  execution parity, or broader macOS platform parity from package evidence

## Capability Surface Ownership

Sprint 74 moved the highest-value bounded capability seams without widening the
shipped product claim beyond what the live code and proof now support.

Current maintained interpretation:

- reviewed builds still default to the 32-bit `idx_t` lane
- wider indices are now a bounded compile-time contract through
  `SPARSE_IDX_BITS`, not a hand-edited typedef story
- the strongest touched public dense-scalar seam now routes through
  `sparse_scalar_t`, and the shared matrix-shell storage/build owner now
  matches that public seam
- current shipped scalar support still remains real-only `double`
- later scalar breadth and later algorithm-family widening remain explicitly
  deferred

Interpretation:

- caller-facing docs should present the width lane as compile-time-selectable,
  but should not imply that the whole repo is already broadly 64-bit-modernized
- caller-facing docs and touched public headers may use `sparse_scalar_t` as
  the dense-scalar owner on the shared matrix-shell helper seam and
  storage/build path plus the iterative/eigs/QR public seams
- maintainers should keep the scalar wording explicit: this is bounded public
  preparation for later widening, not proof of complex support or broad
  numeric genericity
- install/export, reviewed-platform, and ABI wording should stay unchanged
  unless a later sprint actually moves those contracts

Current maintained proof ownership after the Sprint 94 Day 10 baseline:

- `tests/test_sparse_matrix.c` owns the width-contract proof surface:
  - `SPARSE_IDX_BITS`
  - `IDX_MAX`
  - `sparse_idx_bits()`
  - `sparse_scalar_t` on the shared matrix-shell helper seam plus the
    touched storage/build owner
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
- `tests/test_sparse_io.c` owns the touched matrix-shell load-path width and
  parse-rejection proof:
  - malformed Matrix Market dimension and coordinate rejection on the touched
    width-aware consumer seam
- `tests/test_chol_csc.c` owns the bounded direct-family maintained external
  differential lane:
  - Cholesky CSC SPD solves checked against an external-process dense reference
    solve
  - fixture-backed SuiteSparse SPD coverage on `nos4` and `bcsstk04`
  - maintained proof stays family-local to the direct-family SPD Cholesky path
- `tests/test_ldlt_csc.c` and `tests/ldlt_external_dense_reference.py` own the
  bounded LDLT CSC maintained external differential lane:
  - deterministic indefinite KKT solves on `kkt5`, `kkt10`, and
    `ldlt_kkt_scaled_10`
  - external-process dense reference solutions emitted by fixture key
  - maintained proof stays family-local to LDLT CSC solve correctness for these
    deterministic fixtures
- `tests/test_sparse_lu.c` and `tests/lu_external_dense_reference.py` own the
  bounded linked-list LU maintained external differential lane:
  - deterministic nonsymmetric solve comparison on `lu_nonsym_square_5`
  - deterministic singular expected-failure coverage on `lu_singular_square_4`
  - maintained proof stays family-local to linked-list LU solve correctness and
    singular detection for these deterministic fixtures

Interpretation:

- examples and docs remain support surfaces on this lane
- do not imply that touched capability wording replaces the focused proof
  owners above
- do not reinterpret `bench_chol_csc`, `bench_ldlt_csc`, or examples as oracle
  owners for this lane
- do not imply that every solver family now has maintained external
  differential proof
- do not present the LDLT CSC lane as broad indefinite ecosystem parity,
  external factorization parity, or proof of pivot/CSC-layout internals
- do not present the linked-list LU lane as LU CSR external coverage, direct
  compressed-format LU API coverage, or broad nonsymmetric ecosystem parity
- keep `include/sparse_svd.h` and broader capability widening explicitly
  deferred until a later sprint actually changes those contracts

## Sprint 102 Direct Solver Trust Boundary Snapshot

Sprint 102 widened direct-solver oracle evidence in two bounded lanes and
refreshed the direct-solver public guidance boundary.

| Family / lane | Public guidance level | Maintained evidence owner | Trust boundary | Non-claims |
|---|---|---|---|---|
| Cholesky CSC SPD | Use for SPD systems; non-SPD reports `SPARSE_ERR_NOT_SPD` | `tests/test_chol_csc.c` plus `tests/chol_external_dense_reference.py` | named SPD Matrix Market fixtures checked against an external-process dense reference | no broad non-SPD recovery claim; no full backend parity claim from examples or benchmarks |
| LDLT CSC indefinite | Use for symmetric indefinite systems where LDL^T is the natural model | `tests/test_ldlt_csc.c` plus `tests/ldlt_external_dense_reference.py` | deterministic KKT fixtures `kkt5`, `kkt10`, and `ldlt_kkt_scaled_10` checked against an external-process dense reference | no broad indefinite ecosystem parity; no external factorization-layout or pivot-internals proof |
| Linked-list LU | Use for general square systems; singular systems report `SPARSE_ERR_SINGULAR` | `tests/test_sparse_lu.c` plus `tests/lu_external_dense_reference.py` | deterministic nonsymmetric `lu_nonsym_square_5` solve and singular `lu_singular_square_4` expected failure | no LU CSR external oracle coverage; no direct CSR/CSC public LU solve API claim; no broad nonsymmetric ecosystem parity |
| QR | Use for rectangular or rank-deficient least-squares workflows | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_qr_corpus.c`, `tests/test_colamd.c`, `tests/qr_external_dense_reference.py`, `make report-index-oracle-freshness`, and `make report-index-comparison-freshness` | internal invariants, rank, residual, public scalar boundary coverage, bounded external least-squares fixtures `qr_overdetermined_incompatible_4x2` and `qr_overdetermined_compatible_5x3`, bounded rank-only fixture `qr_rankdef_duplicate_5x4_rank_only`, bounded threshold-rank fixtures `qr_rank_threshold_diag4_family`, `qr_rank_threshold_diag4_scaled_family`, `qr_rank_threshold_duplicate_5x4_perturbed_family`, and `qr_rank_threshold_dependent_row_4x3_perturbed_family`, bounded residual-only fixtures `qr_rankdef_duplicate_5x4_residual_only` and `qr_rankdef_dependent_row_4x3_residual_only`, bounded nullspace projector fixtures `qr_rankdef_duplicate_5x4_nullspace_projector`, `qr_rank1_4x3_nullspace_projector`, `qr_rankdef_dependent_row_4x3_nullspace_projector`, and `qr_rankdef_wide_3x5_nullspace_subspace`, Sprint 139 corpus fixture `qr_rank_deficient_6x4_nullspace_v1` proving rank `3`, nullity `1`, and solver-produced nullspace residual `<= 1e-10`, bounded exact minimum-norm fixtures `qr_underdetermined_minnorm_2x4`, `qr_minnorm_3x6_exact_values`, and `qr_minnorm_5x10_exact_values`, bounded owner-local minimum-norm lanes for COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-SVD-pseudoinverse cross-check, and `west0067` submatrix behavior, bounded economy projector fixture `qr_economy_projector_5x3`, and selected generated comparisons for `qr_underdetermined_minnorm_2x4` and `qr_overdetermined_compatible_5x3` against the selected source-controlled dense reference helper | no broad QR, LAPACK, NumPy, or SciPy parity; no global rank-threshold policy; no raw Q-basis, Q-sign/orientation, broad rank-deficient solve, nullspace, minimum-norm, economy-mode, sparse-mode, reorder, SVD-pseudoinverse-as-global-oracle, broad SuiteSparse corpus, broad hosted CI, platform, performance, package/ABI, or state-of-the-art claim |
| SVD | Use for singular-value, pseudoinverse, and low-rank workflows | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, `tests/test_svd_partial_corpus.c`, `tests/test_svd_partial_shared_helpers.h`, `tests/svd_external_dense_reference.py`, `make report-index-oracle-freshness`, and `make report-index-comparison-freshness` | internal reconstruction, rank, condition, pseudoinverse, low-rank, bounded external singular-value fixtures `svd_rect_fullrank_6x4`, `svd_rankdef_duplicate_5x4`, `svd_wide_fullrank_4x6`, `partial_svd_diag6_k2`, `partial_svd_tall_diag_8x5_k3`, and `partial_svd_nonsym_rect10x8_k3`, bounded partial-SVD vector-residual fixtures `partial_svd_vector_residual_diag6_k2`, `partial_svd_vector_residual_tall8x5_k3`, and `partial_svd_vector_residual_nonsym_rect10x8_k3`, bounded partial-SVD rank-deficient range-projector fixture `partial_svd_rankdef_diag6x4_k2_range_projector`, bounded partial-SVD dense low-rank Frobenius fixture `partial_svd_lowrank_diag6x4_k2_frobenius_optimality`, bounded partial-SVD max-iteration fail-closed fixture `partial_svd_max_iter_fail_closed_diag6_k2`, Sprint 140/Sprint 151 corpus fixtures `partial_svd_clustered_repeated_diag8x6_k3_v1`, `partial_svd_rankdef_diag6x4_k2_range_projector_v1`, `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`, and `partial_svd_fail_closed_diag6_k2_v1` proving generated top-k values, rank, selected subspace projectors, triplet residuals, orthogonality, sparse low-rank shape/nnz/selected-value/Frobenius behavior, tight-budget fail-closed behavior, no partial arrays on tight-budget failure, and recovery, plus the selected generated comparison for `partial_svd_diag6_k2` proving fixture-local diagonal top-k singular-value agreement, residual, orthogonality, and diagonal projector diagnostics against the source-controlled dense SVD reference helper | no LAPACK, NumPy, SciPy, or broad SVD parity; no broad partial-SVD correctness; no raw singular-vector identity; no vector sign/orientation identity; no broad vector/subspace, rectangular, nonsymmetric, repeated-spectrum, rank-deficient null-space, pseudoinverse/minimum-norm, sparse-output/drop-tolerance optimality, convergence-rate, partial-result, broad hosted CI, performance, platform, package, ABI, or state-of-the-art claim |

Interpretation:

- README and tutorial wording may mention direct-solver selection and failure
  behavior, but must keep external-oracle confidence tied to named test owners.
- Later comparison work should reuse this table as a wording input before
  making public comparative claims.
- Sprints 121-125 add bounded QR, SVD, and partial-SVD evidence lanes. Sprint
  140 adds the clustered/repeated generated partial-SVD corpus lane, and Sprint
  151 adds rank-deficient rectangular projector, sparse low-rank output, and
  fail-closed recovery generated partial-SVD corpus lanes. These lanes prove
  only the named fixtures and owner-local behaviors listed above.
- Any future external oracle lane needs a family-local boundary artifact before
  implementation, then a validation artifact before public wording changes.

### Selected Oracle Freshness Gate

Use the selected oracle freshness gate when you need the maintained
Sprint 152 QR + partial-SVD generated report family to be current:

```sh
make report-index-oracle-freshness
```

The target builds the static library if needed, runs
`python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`,
and then runs
`python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`.

Expected selected oracle output:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`

The required gate expects `52` generated oracle rows: `3`
generated-reference rows, `23` `solver_family=qr` rows, and `26`
`solver_family=partial_svd` rows. It also expects the selected QR and
partial-SVD fixture-key set to be present. It fails missing, stale, failing,
partial, missing-solver-family, or missing-fixture-key selected oracle output
with diagnostics that name the artifact or manifest path and the regeneration
command.

Generated oracle/report artifacts stay under ignored `build/` paths. Do not
commit them as Sprint 152 proof. Sprint 159 mirrors this selected gate in the
reviewed Linux hosted report-freshness lane and uploads split oracle artifacts
for reviewer inspection. The generated row metadata remains local-only and
fixture-local; the hosted lane proves only that the selected gate ran and
passed on the reviewed Linux CI surface. It is not release evidence, package
proof, ABI proof, broad platform proof, performance proof, external-library
parity, broad QR correctness, broad partial-SVD correctness, or
state-of-the-art evidence.

### Selected Comparison Freshness Gate

Use the selected comparison freshness gate when you need the selected QR and
partial-SVD comparison report families to be current:

```sh
make report-index-comparison-freshness
```

The target builds the static library if needed, runs
`python3 scripts/run_external_comparison.py --target qr-minnorm`, runs
`python3 scripts/run_external_comparison.py --target qr-compatible-ls`, runs
`python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2`,
and then runs
`python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`.

Expected selected comparison output:

- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/summary.md`
- `build/comparison/qr_minnorm/manifest.tsv`
- `build/comparison/qr_compatible_ls/project_observations.tsv`
- `build/comparison/qr_compatible_ls/baseline_observations.tsv`
- `build/comparison/qr_compatible_ls/dependency_status.tsv`
- `build/comparison/qr_compatible_ls/study.tsv`
- `build/comparison/qr_compatible_ls/summary.md`
- `build/comparison/qr_compatible_ls/manifest.tsv`
- `build/comparison/partial_svd_diag6_k2/project_observations.tsv`
- `build/comparison/partial_svd_diag6_k2/baseline_observations.tsv`
- `build/comparison/partial_svd_diag6_k2/dependency_status.tsv`
- `build/comparison/partial_svd_diag6_k2/study.tsv`
- `build/comparison/partial_svd_diag6_k2/summary.md`
- `build/comparison/partial_svd_diag6_k2/manifest.tsv`

The required comparison freshness gate expects three source-controlled
contract rows plus 22 generated selected rows split across
`qr_underdetermined_minnorm_2x4` and
`qr_overdetermined_compatible_5x3`, with six rows each:

- `project_status`
- `baseline_status`
- `residual_norm`
- `solution_norm`
- `solution_values`
- `project_vs_baseline_max_abs_delta`

The selected `partial_svd_diag6_k2` family contributes ten rows:

- `project_status`
- `baseline_status`
- `singular_value_0`
- `singular_value_1`
- `singular_values_max_abs_delta`
- `residual_norm`
- `u_orthogonality`
- `v_orthogonality`
- `u_projector_diag`
- `v_projector_diag`

Interpret the generated rows as fixture-local QR minimum-norm, QR compatible
least-squares, and partial-SVD diagonal top-k comparison evidence only. The
generated rows remain `local_only`. `skip` and `defer` rows are visible
non-proof states, and optional NumPy or SciPy absence cannot create pass
evidence. The comparison families do not claim broad QR, broad SVD or
partial-SVD correctness, raw QR basis identity, raw singular-vector identity,
vector sign/orientation identity, LAPACK, NumPy, SciPy, SuiteSparse, Eigen,
release, broad platform support, package-manager behavior, shared-library ABI,
performance, or state-of-the-art proof.

### QR Corpus Maintenance

The Sprint 139/Sprint 150 QR corpus lane is maintained as a fixture-local
confidence path, not as a broad QR parity claim. It covers the
`qr_rank_deficient_6x4_nullspace_v1` seed plus the Sprint 150
rank-deficient rectangular and underdetermined minimum-norm fixture families.
The selected Sprint 150 fixture keys are:

- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`
- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

Together with the seed fixture, the QR corpus lane proves only named
rank/nullity, nullspace residual/subspace, and underdetermined minimum-norm
status/residual/norm/value rows.

Use these commands to regenerate and interpret the lane from a clean local
checkout:

```sh
python3 scripts/validate_corpus_schema.py
make build/test_qr_corpus && ./build/test_qr_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

The focused C proof should report `14` passing `test_qr_corpus` tests. The
opt-in oracle command should write
`build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv` with three
generated-reference rows plus `23` `solver_family=qr` rows for the seed and
Sprint 150 QR fixtures. The report index under
`build/corpus-reports/index.tsv` should show those QR rows as `pass`; optional
external-data rows belong in `build/corpus-reports/skips.tsv` and remain
skip/defer policy evidence only. The manifest under
`build/corpus-reports/manifest.txt` should record the command, row count,
solver families, the selected six QR fixture keys, `solver_qr_row_count=23`,
and `partial_svd_row_count=0` for a QR-only run.

Use the QR-only oracle command for focused QR debugging. Use
`make report-index-oracle-freshness` for the selected combined freshness gate;
the QR-only run does not satisfy the Sprint 152 selected row-count policy by
itself.

Treat a QR corpus report as stale or non-interpretable when any of these are
true:

- `tests/test_qr_corpus.c`, `tests/test_qr_helpers.h`, corpus fixture rows,
  expected-result rows, generator metadata, schema files, or
  `scripts/run_corpus_oracle.py` changed after the report was generated.
- The command, commit, branch, compiler, configuration, support tier, or
  generated path recorded in the manifest does not match the report being
  reviewed.
- The QR oracle file is missing the `23` `solver_family=qr` rows, reports a
  solver QR row count other than `23`, omits `qr` from the solver families,
  omits any selected QR fixture key, or emits any non-pass comparison status
  for the maintained QR rows.
- Optional SuiteSparse or external-data skip/defer rows are being treated as
  QR pass evidence.

Support tier remains `local_only` for this QR lane until a later sprint
promotes reviewed hosted-platform evidence. Generated oracle/report files stay
ignored build artifacts; `scripts/run_corpus_oracle.py` clears stale generated
oracle/report outputs before writing the current run so normalization reads the
current local lane only. Source-controlled confidence lives in the fixture
metadata, expected rows, `test_qr_corpus`, and the reproducible command.

Remaining QR residuals after Sprint 139:

- Global rank-threshold policy remains open because it needs tolerance-family
  design across scales and perturbations, not a single nullspace fixture.
- Broad rank-deficient least-squares, residual-only solve, and minimum-norm
  behavior remain open because they need solve-side fixtures and separate
  oracle semantics.
- COLAMD/reordered QR behavior remains open because ordering semantics and
  fill behavior are distinct from the nullspace residual lane.
- SuiteSparse and external-library parity remain open until optional-data
  provenance, reviewed platform runs, and external-reference boundaries are
  promoted.
- Raw Q-basis/sign/orientation parity is intentionally not closed; Sprint 139
  compares residual/subspace-safe behavior instead.
- Partial-SVD clustered/repeated, rank-deficient rectangular, sparse low-rank
  output, and fail-closed recovery follow-through are now covered by
  `tests/test_svd_partial_corpus.c` and the selected local oracle freshness
  gate;
  broader repeated-spectrum, raw vector identity, broad sparse-output
  optimality, external-library parity, performance, platform/package/ABI, and
  state-of-the-art claims remain out of scope.

### Partial-SVD Corpus Maintenance

The Sprint 140/Sprint 151 partial-SVD corpus lane is maintained as a
fixture-local confidence path, not as broad partial-SVD parity or product
quality proof. The selected fixture keys are:

- `partial_svd_clustered_repeated_diag8x6_k3_v1`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`
- `partial_svd_fail_closed_diag6_k2_v1`

Together, these fixtures prove only named top-k value, rank, selected
subspace-projector, triplet-residual, orthogonality, sparse-output
shape/nnz/selected-value/Frobenius, fail-closed, no-partial-array, and recovery
rows.

Use these commands to regenerate and interpret the lane from a clean local
checkout:

```sh
python3 scripts/validate_corpus_schema.py
make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
```

The focused C proof should report `10` passing `test_svd_partial_corpus` tests.
The opt-in oracle command writes `build/corpus/oracle/corpus.oracle.tsv` with
three QR generated-reference rows plus `26` `solver_family=partial_svd` rows
for the maintained partial-SVD fixtures. The manifest should record
`partial_svd_row_count=26`, `solver_families=partial_svd,unknown`, the command,
the current commit/branch, and all four selected partial-SVD fixture keys. The
normalized corpus/oracle index should currently contain `105` rows when the
generated local oracle output is present.

Use the partial-SVD-only oracle command for focused partial-SVD debugging. Use
`make report-index-oracle-freshness` for the selected combined freshness gate;
the partial-SVD-only run does not satisfy the Sprint 152 selected row-count
policy by itself.

Treat a partial-SVD corpus report as stale or non-interpretable when any of
these are true:

- `tests/test_svd_partial_corpus.c`,
  `tests/test_svd_partial_shared_helpers.h`, corpus fixture rows,
  expected-result rows, generator metadata, schema files,
  `scripts/validate_corpus_schema.py`, `scripts/run_corpus_oracle.py`, or
  `scripts/normalize_report_index.py` changed after the report was generated.
- The command, commit, branch, platform, compiler, configuration, support tier,
  or generated path recorded in the oracle rows or manifest does not match the
  report being reviewed.
- The oracle output lacks the expected `8`, `7`, `6`, and `5` generated rows
  for the four selected fixtures, reports a partial-SVD row count other than
  `26`, omits `partial_svd` from solver families, omits any selected
  partial-SVD fixture key, or emits any non-pass comparison status for the
  maintained partial-SVD rows.
- Normalized oracle freshness reports stale rows. Default freshness may warn,
  but `--strict-generated --check-freshness` or
  `--require-generated oracle --check-freshness` should fail stale strict
  oracle evidence when a current partial-SVD report is required.

Support tier remains `local_only` for this partial-SVD lane until a later
sprint promotes reviewed hosted-platform evidence. Generated oracle/report
files stay ignored build artifacts. Source-controlled confidence lives in the
fixture metadata, generator metadata, expected rows, focused C proof-owner
tests, normalized report-index tests, and reproducible commands.

## Sprint 103 Iterative, Spectral, and SVD Evidence Boundary Snapshot

Sprint 103 added bounded comparison evidence for iterative, eigensolver, and
SVD workflows. These lanes improve residual, orthogonality, rank, and
convergence-profile confidence for named fixtures. They do not establish broad
external package parity.

| Family / lane | Public guidance level | Maintained evidence owner | Evidence boundary | Residual / quality interpretation | Non-claims |
|---|---|---|---|---|---|
| BiCGSTAB nonsymmetric convergence | Use as a Krylov option for nonsymmetric systems where residual convergence is checked by the caller | `tests/test_bicgstab.c` | deterministic known-solution solve checked against LU, internal GMRES(30)+ILU comparison on `steam1`, and one declared non-convergence budget boundary | relative residuals are fixture-local solve-quality checks; the GMRES comparison is an internal consistency cross-check, not an external oracle | no PETSc, SciPy, Trilinos, or broad nonsymmetric parity claim; no portable iteration-count or performance claim |
| LOBPCG closed-form and preconditioned fixtures | Use for symmetric eigenvalue workflows that can tolerate fixture-local Ritz residual interpretation | `tests/test_eigs_lobpcg.c` | closed-form Laplacian eigenvalues with vector orthogonality plus `bcsstk04` LDLT-versus-IC(0) residual/orthogonality comparison | Ritz residuals and `result.residual_norm` are local quality criteria for the requested eigenpairs; preconditioner deltas are descriptive except where the fixture asserts a threshold | no ARPACK, SciPy, PETSc, Trilinos, or broad eigensolver parity claim; no portable preconditioner superiority claim |
| Thick-restart exact diagonal fixture | Use as bounded regression evidence for restarted symmetric eigenvalue extraction | `tests/test_eigs_thick_restart.c` | exact diagonal eigenvalue fixture with Ritz residual, orthogonality, and bounded peak-basis checks | residuals are checked against exact diagonal eigenpairs; grow-`m` agreement elsewhere remains internal comparison evidence | no ARPACK parity claim; no broad memory, restart, or performance claim beyond the named bounded-basis fixture |
| SVD deterministic rank and full-UV fixture | Use for singular-value, reconstruction, orthogonality, and rank-threshold confidence on deterministic fixtures | `tests/test_svd.c` | exact diagonal singular values `{9, 5, 2, 1e-9, 0, 0}`, full-mode reconstruction residual, U/Vt orthogonality, and explicit rank thresholds | reconstruction residual and orthogonality are separately computed test metrics; rank evidence is tied to the declared tolerances `1e-10` and `1e-8` | no LAPACK, NumPy, SciPy, or broad SVD parity claim; external helper evidence remains limited to named Sprint 121-123 fixtures |

Evidence-type interpretation:

- external dense-reference evidence remains limited to named direct-solver
  lanes plus bounded SVD, QR, and partial-SVD fixture lanes in Sprints 121-123
- deterministic fixture evidence means expected values are constructed inside
  the project and checked directly by the owning test
- internal consistency evidence means one project solver or configuration is
  compared with another project solver or configuration
- residual and orthogonality evidence is a named-fixture quality criterion, not
  a substitute for external package parity
- absent in this snapshot means Sprint 103 did not add an external
  helper-backed comparison lane for that family; later Sprint 121-123 lanes are
  tracked in the Sprint 102 table and Sprint 123 artifacts

Documentation wording rules:

- prefer "bounded", "named fixture", "deterministic fixture", and "internal
  consistency cross-check" when describing Sprint 103 evidence
- avoid "external parity", "ecosystem parity", "state of the art proof", or
  broad package-comparison wording unless a future sprint adds and validates a
  maintained helper-backed lane
- treat iterative solver iteration counts as fixture-local diagnostics unless
  a test artifact defines an explicit threshold
- use `result.residual_norm` carefully: iterative solvers report solve
  residuals, eigensolvers report Ritz residual summaries, and SVD reconstruction
  residuals are test-computed metrics rather than a public SVD result field

## Sprint 98 Assurance Topology Snapshot

Sprint 98 widened assurance evidence in two bounded lanes and audited coverage
and workflow topology without widening those surfaces.

| Evidence class | Owner | Validation command | Interpretation |
|---|---|---|---|
| LDLT CSC external correctness | `tests/test_ldlt_csc.c` plus `tests/ldlt_external_dense_reference.py` | `make build/test_ldlt_csc && ./build/test_ldlt_csc` | bounded deterministic KKT solve comparison on `kkt5` and `kkt10` |
| Reorder/fill calibration | `make bench-reorder-sprint86` / `bench_reorder --sprint86-slice --skip-factor` | `make bench-reorder-sprint86` | bounded two-fixture artifact; `nnz_L` is the fill field and `reorder_ms` is local timing context |
| Coverage topology | `make coverage`, `make coverage-lcov`, `make coverage-gcovr`, and Linux supplemental coverage workflow | no Sprint 98 validation command added | audited but not widened; coverage remains tree-mutating and supplemental |
| Workflow topology | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | no Sprint 98 workflow lane added | audited but not widened; reviewed, supplemental, and staged platform claims stay unchanged |

Interpretation:

- this snapshot is a maintainer map, not a new public claim surface
- do not move proof ownership out of the family-local test owners without a
  separate extraction boundary
- do not treat the Sprint 98 runtime/fill artifact as canonical reporting or a
  portable timing gate
- do not imply that audited coverage or workflows gained new reviewed scope

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

## OpenMP and Runtime-Control Model

OpenMP support is intentionally narrower than the broader `SPARSE_*`
compatibility environment-variable story.

Current interpretation:

- serial builds remain the default product path
- `SPARSE_OPENMP` is a compile-time build option, not a runtime policy object
- the library does not expose a public thread-pool, per-call thread limit, or
  `sparse_set_num_threads` API
- OpenMP team size, affinity, and nested-parallelism behavior remain owned by
  the OpenMP runtime (`OMP_NUM_THREADS`, vendor runtime settings, and caller
  process configuration)
- `SPARSE_MUTEX` is a separate matrix-mutation safety build option and should
  not be described as OpenMP runtime control
- graph/reorder `_Thread_local` overrides are internal scope mechanisms; they
  protect concurrent calls but do not form a public thread-control surface

Current OpenMP implementation owners:

- `src/sparse_matrix.c` owns row-parallel linked-list SpMV and block SpMV
- `src/sparse_eigs.c` owns inner-axis MGS reorthogonalization for Lanczos
  paths, gated by `SPARSE_EIGS_OMP_REORTH_MIN_N`
- solver, SVD, and graph paths may reach OpenMP indirectly through SpMV or
  eigensolver calls; do not add outer OpenMP regions without a fresh
  nested-parallelism and oversubscription validation plan

Validation interpretation:

- docs-only runtime wording changes need docs hygiene
- any `.c` or `.h` OpenMP/runtime cleanup, including comments beside pragma
  owners, should run `make format && make lint && make test`
- behavior changes to OpenMP scheduling or thresholds also need focused OpenMP
  validation and eigensolver/SVD owner tests where relevant
- do not silently translate `SPARSE_*` compatibility env vars into OpenMP
  thread counts
- do not promote thread-local internal override scopes into public runtime
  controls in user-facing docs

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

When cleaning public headers, treat the change as API-surface work even if the
intended edit is comment-only:

- keep function purpose, shape assumptions, ownership/lifetime rules, output
  buffer sizes, NULL/error returns, mutation guarantees, option defaults,
  result-field semantics, backend behavior, and required non-claims at the
  call site;
- shorten sprint history, benchmark narratives, report/CI ownership detail,
  and tutorial-scale examples when a link to README, examples, cookbook,
  solver-selection, benchmarks, install docs, or this guide is clearer;
- do not change declarations, signatures, typedefs, enum values, struct field
  order, macros, include guards, installed header names, or exported names as
  part of documentation cleanup;
- after public header edits, compare declaration-like diff hunks before and
  after the change, scan claim wording, run `git diff --check`, and run
  `make format && make lint && make test`;
- after public-header or API-comment cleanup, check whether
  `docs/api_reference.md` needs a header table or ownership update and whether
  `docs/api/html/` should be refreshed or explicitly treated as stale/partial.

### API reference and generated Doxygen HTML

`docs/api_reference.md` is the user-facing API reference entry point. It should
stay compact and should route exact declarations back to the public headers
under `include/`.

`docs/api/html/` is generated Doxygen output from the configured input set in
`Doxyfile`. The maintained Sprint 158 policy keeps this tree local-only and
ignored rather than committed or hosted.

Use this command to refresh and validate the local generated API view:

```bash
make docs-check
```

Interpretation:

- `make docs-check` runs Doxygen and then checks generated page coverage for
  checked-in public headers under `include/`;
- the generated HTML is current only for the branch and checkout where the
  command just passed;
- generated `sparse_version.h` remains an installed-header policy row derived
  from `VERSION` and `include/sparse_version.h.in`, not an expected Doxygen
  page under the current input set;
- local generated output under `docs/api/html/` is not source-controlled,
  hosted, or release evidence.

Treat generated API HTML as stale or partial when public header comments changed
after the last successful `make docs-check`, new checked-in public headers do
not have generated pages, Doxygen warnings appear without triage, or generated
installed headers are expected to appear even though they are outside the
configured Doxygen input set.

API reference guidance may say that public headers own exact declarations and
call-site contracts and that `make docs-check` generates and validates local
Doxygen HTML. It must not imply dynamic ABI compatibility, shared-library
support, package-manager distribution, broad Windows Makefile or Windows
`pkg-config` parity, external-library parity, portable runtime guarantees,
hosted documentation publication, source-controlled generated HTML, or
completeness beyond the configured Doxygen input set.

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
- `tests/test_direct_csc_dispatch.c` owns the cross-threshold Cholesky CSC
  dispatch and forced backend parity proof surface
- `tests/test_direct_csc_regression.c` owns the retained direct-family CSC
  regression bundle:
  - threshold lock
  - Kuu scalar CSC regression
  - row-adjacency structural checks
  - supernodal LDL^T scalar/batched parity
- `src/sparse_ldlt_csc_rowadj.c` owns the private LDL^T CSC row-adjacency
  helper seam:
  - append growth and argument checks remain covered directly in
    `tests/test_ldlt_csc.c`
  - row-adjacency slot swapping is covered directly in `tests/test_ldlt_csc.c`
    and indirectly through symmetric-swap/native-elimination proof paths
  - SuiteSparse row-adjacency structural correctness remains covered in
    `tests/test_direct_csc_regression.c`
- `tests/test_ldlt_backend_dispatch.c` owns the public LDL^T backend selector
  and AUTO/forced dispatch proof surface
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

Sprint 106 maintainability ownership additions:

- `src/sparse_qr_householder.c` owns the private QR Householder kernel seam:
  - Householder vector construction and application live here rather than in
    the broad QR factorization owner
  - sparse-mode column extraction and column-sliced Householder application
    also live here because they are part of the same QR-local transformation
    responsibility
  - declarations remain private in `src/sparse_qr_internal.h`; do not promote
    these helpers to public headers without a separate API-design review
- `src/sparse_lu_csr_struct.c` owns LU CSR structural storage mechanics:
  - row storage growth and insertion helpers should grow here instead of
    adding more structural code to `src/sparse_lu_csr.c`
  - numeric elimination, solve, and factor orchestration stay in
    `src/sparse_lu_csr.c`
  - declarations remain private in `src/sparse_lu_csr_internal.h`
- the LDLT CSC row-adjacency seam remains in `src/sparse_ldlt_csc_rowadj.c`;
  future row-adjacency allocation, append, growth, or swap mechanics should
  stay with that owner unless the change is a numeric elimination concern
- test fixture/helper ownership is now deliberately split:
  - `tests/test_graph_fixtures.h` owns reusable graph/reorder synthetic graph
    builders, partition counters, cut helpers, and partition invariant checks
  - `tests/test_direct_solver_helpers.h` owns direct-solver assertion and
    residual helpers that are not family-specific public API checks
  - `tests/test_integration_fixtures.h` owns integration progress-callback
    counters and matrix fixtures used across direct, QR, iterative, and
    eigensolver workflow tests
- keep test helpers header-only unless there is a measured compile-time,
  ownership, or reuse reason to create a compiled test support target; a
  compiled helper would require explicit Make/CMake registration and reviewed
  CTest-surface reconciliation
- do not grow giant proof owners with reusable setup code when a Sprint 106
  helper already owns the fixture family; keep call sites readable by using
  helper names that include the family or workflow intent
- large source extraction should keep the same three-surface follow-through:
  `Makefile` `LIB_SRCS`, `CMakeLists.txt` `add_library(...)`, and
  `build-metadata/library_sources.txt`; run
  `python3 scripts/check_library_sources.py` after any new library source
  owner is added

Current platform-confidence interpretation:

- Linux and macOS still exercise the full `test_fuzz` binary in their direct
  `make test` / reviewed local paths, so the bounded seeded generative
  lifecycle property lanes are part of those proof surfaces
- Windows reviewed CMake now includes `test_fuzz`, so its deterministic
  parser/property coverage is part of the reviewed Windows CMake evidence once
  hosted MSVC configure/build/execute proof is green
- Windows reviewed CMake also includes `test_threads` and
  `test_sprint4_integration` through `tests/test_thread_helpers.h`, which keeps
  POSIX pthread behavior on POSIX builds and uses a Windows backend on MSVC
- this is a narrow CTest confidence-boundary note only; it does not claim
  Windows Makefile parity, Windows `pkg-config` execution parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, or broad Windows parity

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
  - `ldlt_dense_backend_request`
  - `ldlt_dense_backend_selected`
  - `ldlt_dense_backend_fallback`
  - `csc_scalar_path`
  - `csc_supernodal_path`
  - `csc_supernodal_dense_kernel`
  - `csc_supernodal_panel_solver`
- iterative/eigensolver canonical surfaces keep their repeated-run proof fields:
  - one-shot timing
  - reuse timing
  - speedup
  - last-run convergence/residual agreement fields
- bounded branch-local runtime lanes should keep the smallest context needed to
  make emitted rows interpretable across reruns:
  - `bench_reorder` now stamps each row with:
    - `reorder_path`
    - `fixture_slice`
    - `nd_base_threshold`
  - the Sprint 98 reorder/fill artifact uses `make bench-reorder-sprint86`
    as a bounded two-fixture calibration slice, with `nnz_L` as the primary
    fill field and `reorder_ms` as local timing context only
  - read those as bounded local runtime-evidence context, not as broad
    benchmark-governance widening

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
    - platform, compiler, runner context, build flags, CPU model, build mode,
      and `OMP_NUM_THREADS` for local or hosted comparison context
  - writes `index.tsv` with one structured row per emitted canonical artifact
    and the same platform/compiler/runner/build/thread context
  - appends methodology fields for `report_family`, `status`,
    `support_tier`, `claim_boundary`, `fixture_or_workload`, `matrix_size`,
    `repeat_semantics`, `warmup`, `variance`, `baseline`, `threshold`,
    `backend_context`, and `methodology_notes`
  - is acceptable for local before/after comparison or CI artifact capture
  - is intentionally not a timing threshold gate
  - uses `status=measurement`, `support_tier=local_only`,
    `claim_boundary=local_threshold_free`, `baseline=n/a`, and
    `threshold=n/a` for unselected canonical rows; do not reinterpret those
    rows as pass/fail evidence
  - records `warmup=not_recorded` and `variance=not_recorded` until a later
    sprint adds explicit warmup or statistical methodology
  - should stay limited to the canonical maintained surface unless a later
    sprint proves a wider report remains cheap and stable
- `make bench-canonical-report-freshness`
  - regenerates the canonical report bundle and checks only the selected
    `bench_refactor_csc` row for `nos4.mtx --repeat 1`
  - validates selected artifact presence, `index.tsv` schema, selected row
    identity, required methodology metadata, threshold-free baseline/threshold
    values, `methodology_notes`, and `manifest.txt` agreement
  - is mirrored by the reviewed Linux hosted selected-performance freshness
    job, which runs the checker in hosted mode with `hosted_selected` and
    `hosted_selected_threshold_free` metadata on the selected row only
  - does not compare timing values, define a regression threshold, promote the
    other canonical rows, or claim portable performance, external-library
    parity, package/ABI support, broad platform support, release proof, or
    state-of-the-art performance

Current bounded local sentinel bundle:

- `make performance-sentinels`
  - writes structured output under:
    - `build/bench-reports/sentinels/`
  - records branch, commit, platform, compiler, build mode,
    `OMP_NUM_THREADS`, `SPARSE_CHOL_DENSE_BACKEND`, and
    `SPARSE_LDLT_DENSE_BACKEND`
  - records per-row support tier, claim boundary, artifact, backend
    request/selection/fallback, dense-kernel descriptor, and panel-solver
    descriptor where applicable
  - appends `baseline_provenance`, `repeat_semantics`, `warmup`, `variance`,
    and `methodology_notes` for publication review
  - uses `n/a` for backend fields on rows such as S5 that do not own a dense
    backend seam
  - treats S5 as the existing hard `wall-check` threshold gate
  - treats S2 Cholesky CSC rows as threshold-free report context only
  - treats S3 LDLT KKT rows as threshold-free backend report context only
  - treats S5 status as meaningful only with the recorded baseline, threshold,
    fixture, command, baseline provenance, and machine context
  - treats S2/S3 `status=report` rows as backend-context rows, not passing
    evidence
  - should not add new hard timing thresholds without a fresh local-baseline
    or same-worktree comparison design
  - should not be described as portable performance evidence

Runtime/backend control boundary:

- public typed controls are limited to caller-meaningful backend or analysis
  decisions: Cholesky backend, LDLT backend, symmetric eigensolver backend,
  and analysis/reorder options
- legacy environment variables are compatibility or maintainer controls when a
  typed field exists; explicit typed values win when both are present
- `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND` are dense-helper
  diagnostics/report controls, not public solver-backend API
- `SPARSE_SVD_LOWRANK_OUTER`, FM strategy/debug/profile variables, OpenMP
  runtime context, package/link settings, and test/benchmark opt-ins remain
  maintainer-only or build/report controls unless a future sprint promotes a
  specific surface with tests and documentation
- generated sentinel backend fields preserve request/selection/fallback
  context; they do not imply optional-backend availability, ABI support, or
  portable timing behavior

Report-index handoff:

- keep canonical report rows threshold-free even though the generated index
  now records platform, compiler, build mode, and `OMP_NUM_THREADS`
- preserve canonical row methodology fields in downstream summaries:
  `support_tier`, `claim_boundary`, `repeat_semantics`, `warmup`, `variance`,
  `baseline`, `threshold`, and `methodology_notes`
- preserve sentinel row `support_tier` and `claim_boundary` fields so reviewed
  thresholded rows, reviewed threshold-free rows, skips, and report-only rows
  remain distinguishable
- preserve sentinel row `baseline_provenance`, `repeat_semantics`, `warmup`,
  `variance`, and `methodology_notes`; S5 baseline provenance is part of the
  local-gate interpretation
- preserve backend `n/a`, `unknown`, selected, and fallback fields instead of
  inferring builtin or optional-backend availability from missing data
- keep generated benchmark, sentinel, and normalized-index artifacts under
  ignored `build/` paths unless a future sprint explicitly promotes a stable
  checked-in example with its own freshness and claim-boundary contract
- do not promote supplemental guardrail rows or benchmark-local rows into
  reviewed recurring evidence without a separate owner, runtime budget, and
  claim-boundary decision
- do not cite package/install, package-manager, shared-library, dynamic ABI,
  or runtime-loader proof as performance evidence

Current large-matrix structural guardrail bundle:

- `make large-matrix-guardrails`
  - writes structured output under:
    - `build/bench-reports/large-matrix-guardrails/`
  - records branch, commit, platform, compiler, timestamp, and whether
    supplemental mode was enabled
  - default reviewed lanes:
    - `G1`: `build/test_reorder_amd_qg`
    - `G2`: `build/test_reorder_nd`
    - `G3`: `build/test_graph`
    - `G4`: `build/bench_reorder --sprint86-slice --skip-factor`
  - default supplemental lanes:
    - `S1`: `build/bench_reorder --skip-factor`
    - `S2`: `build/bench_amd_qg --skip-bitset`
  - keeps `S1` and `S2` as explicit `skip` rows unless
    `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1` is set
  - treats `G1` through `G3` as structural test guardrails, not benchmark
    timing claims
  - treats `G4` as bounded CSV-shape and fill-report evidence for
    `bcsstk14` and `Pres_Poisson`
  - should not add new hard timing, max-RSS, or full numeric-factor thresholds
    without a fresh baseline and machine-class design

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
- do not reinterpret `performance-sentinels` as a broader timing proof; its
  only hard timing gate is the existing wall-check lane
- do not reinterpret the Sprint 98 reorder/fill artifact as a replacement for
  the canonical maintained performance surface or as a portable timing claim
- do not reinterpret `large-matrix-guardrails` as broad large-matrix
  performance proof; its reviewed lanes are structural and bounded, and its
  supplemental lanes are opt-in reports
- do not let exploratory benchmark breadth blur the smaller claim-bearing
  maintained surface

## Normalized Report Index Workflow

Use the normalized report index when you need one cross-family view of
source-controlled report metadata and generated local report artifacts:

```sh
python3 scripts/normalize_report_index.py \
  --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py --check
python3 scripts/normalize_report_index.py --check-freshness
```

Generated output belongs under ignored `build/report-index/`. Do not commit
the generated TSV unless a later sprint explicitly promotes a stable checked-in
example.

For Sprint 163 performance rows, the normalized index is a navigation surface:
it should keep methodology fields visible in `configuration`, but it is not
hosted CI proof, package proof, ABI proof, broad platform proof, backend
superiority proof, OpenMP speedup evidence, or state-of-the-art evidence.

Common focused checks:

```sh
make report-index-oracle-freshness
make report-index-comparison-freshness
make bench-canonical-report-freshness
python3 scripts/normalize_report_index.py --family oracle --check-freshness
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --check-freshness
python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness
```

Read diagnostics as:

```text
freshness: <severity>: <row_id>: <state>: <reason>
```

Interpretation:

- `error` means a selected strict or required row failed freshness or a
  hard-gate row reports failure; for selected oracle rows this includes
  missing artifacts, stale source commits, failing comparison rows, selected
  row-count mismatches, missing solver families, and missing fixture keys
- for selected required oracle/comparison rows generated at the current
  source commit, `fresh` diagnostics mean the selected gate has current
  generated output; stale, missing, failing, skipped, deferred, duplicate,
  unexpected, or incomplete selected rows must not be treated as pass evidence
- `warning` means a strict generated family is absent or stale but was not
  explicitly required
- `advisory` means the row is local measurement, quality, documentation, or
  source-controlled context
- `skip` means optional data or prerequisites are unavailable by policy
- `defer` means the row is intentionally handed off, currently including
  runtime/backend governance for Sprint 142

The normalized index is not release proof by itself. It preserves row meaning,
artifact paths, freshness context, support tier, claim scope, and non-claim
boundaries so maintainers can decide which underlying generator or validation
command must be rerun.

For Sprint 152 selected oracle freshness, prefer
`make report-index-oracle-freshness` over hand-running the two underlying
commands. The Makefile target regenerates current local oracle output and runs
the required oracle freshness gate. Generated report-index and oracle outputs
remain ignored local artifacts. Sprint 159 adds reviewed Linux hosted execution
and split artifact upload for the selected oracle gate only; broad report-index
freshness and unselected generated families remain local/advisory.

For selected QR and partial-SVD comparison freshness, prefer
`make report-index-comparison-freshness` over hand-running the underlying
commands. The Makefile target regenerates current local comparison output for
all selected targets and runs the required comparison freshness gate. Generated
comparison and report-index outputs remain ignored local artifacts by default.
The reviewed Linux hosted report-freshness lane promotes only this selected
comparison gate and its uploaded selected artifacts; it does not promote broad
report-index freshness or unselected comparison families. Optional NumPy/SciPy
defers remain context, not pass evidence.

For selected canonical performance freshness, prefer
`make bench-canonical-report-freshness` over hand-running the report script.
The Makefile target regenerates the canonical bundle and runs the selected
freshness checker for only `bench_refactor_csc` on `nos4.mtx --repeat 1`.
Generated benchmark outputs remain ignored local artifacts by default. The
reviewed Linux hosted selected-performance lane promotes only the selected row
freshness and uploaded canonical bundle metadata; it does not promote raw
timing values, unselected canonical rows, broad benchmark publication,
external-library parity, package/ABI support, broad platform proof, release
proof, or state-of-the-art performance.

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
