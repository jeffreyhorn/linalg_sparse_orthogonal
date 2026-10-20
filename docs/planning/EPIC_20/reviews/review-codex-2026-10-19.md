# Epic 20 Code Review (2026-10-19)

**Scope:** current `planning/epic-20` branch at merged `master` commit
`d3e905ea`; public headers, core kernels, tests, build/install paths, CI,
benchmarks, user documentation, and Epic 17-19 residuals. This is a source
review and local validation, not a new hosted-platform, release, or external
solver result. Findings are ordered by impact. File references are to this
snapshot. The companion [closure list](todo-codex-2026-10-19.md) separates
work funded in Epic 20 from longer-horizon decisions.

## Findings

### High: Numerical norms can overflow before the true result does

[`vec_norm2`](../../../../src/sparse_vector.c) accumulates `v[i] * v[i]` in
`double` and then takes `sqrt`. A finite vector with a representable Euclidean
norm can therefore report infinity (for example, one component near `1e200`).
The function is used to compute right-hand-side and residual norms in
[`src/sparse_iterative.c`](../../../../src/sparse_iterative.c), so this is not
merely cosmetic output: it can distort relative-residual calculations and
termination decisions. `vec_norminf` in the same file also ignores a NaN
component because `fabs(NaN) > mx` is false. The `west0067` ILUT test at
[`tests/test_ilu.c:1152`](../../../../tests/test_ilu.c) printed
`ILUT solve norm=inf` during this review's passing `make test`, but only
asserts `znorm > 0`; this observation does **not** establish that the solution
entries are non-finite, because the norm implementation alone can overflow.

**Close with:** a scaled norm (such as LAPACK's [LASSQ method](https://netlib.org/lapack/explore-html/d8/d76/group__lassq.html)),
explicit non-finite handling policy, extreme-scale/NaN tests, and an ILUT
residual/finite-output assertion that distinguishes a bad solve from bad norm
arithmetic. Check the iterative convergence paths against those fixtures.

### High: No accepted independent solver comparison yet

The optional [CHOLMOD pilot](../../../../scripts/run_cholmod_pilot.py) checks
for `tests/corpus/schemas/external_baseline_evidence_v1.json` and
`scripts/validate_external_baseline_evidence.py` before executing a solve;
both are absent in this snapshot. The [Sprint 215 attempt](../../EPIC_19/SPRINT_215/artifacts/day10-selected-pilot-outcome.md)
stopped at preflight. The [Epic 19 retrospective](../../EPIC_19/EPIC_19_RETROSPECTIVE.md)
reports zero validator-accepted external cells. Existing comparison rows use
project-controlled helpers; they are useful regressions but not independent
CHOLMOD, UMFPACK, or SuperLU parity evidence. An external result also needs
matrix rights, actual loaded-library provenance, raw outputs, and numerical
gates before publication. SuiteSparse's [official package inventory](https://github.com/DrTimothyAldenDavis/SuiteSparse)
shows the breadth of solver families against which a general state-of-the-art
claim would have to be scoped.

**Close with:** the v1 schema/validator first, then one executed, two-sided
CHOLMOD correctness cell on a pinned SPD fixture and an approved, narrowly
worded evidence bundle. One cell is not broad parity.

### High: The sparse-matrix multiply output path has quadratic row flush

[`sparse_matmul`](../../../../src/sparse_matrix.c) uses a Gustavson-style dense
row accumulator, but flushes each sorted touched column through
`sparse_insert`. The implementation comment at `src/sparse_matrix.c:847`
itself identifies `O(nnz_row^2)` worst-case row flushing because
[`sparse_insert`](../../../../src/sparse_matrix.c) starts a sorted row scan for
each element; insert also links columns through a second scan. This makes
wide-result rows especially expensive and reinforces the linked-list
representation's pointer/locality cost. CSR/CSC entry points and compressed
direct backends are already present, so a bounded compressed output/builder
change is preferable to promising a full storage rewrite.

**Close with:** profile a pinned matrix set, build output without repeated
head scans while preserving public semantics and deterministic order, and
validate numerical equality, memory behavior, allocation-failure cleanup,
and retained before/after samples. Do not claim portable speed from one host.

### Medium: Installed index-width contract is only a header convention

[`include/sparse_types.h:20`](../../../../include/sparse_types.h) exposes
`idx_t` as 32 or 64 bits under `SPARSE_IDX_BITS`, and warns consumers to
rebuild with the same width. The default is 32. Yet the reviewed
[`CMakeLists.txt`](../../../../CMakeLists.txt), `Makefile`, and hosted workflows
do not provide a named width selection/installed-target propagation or a
64-bit consumer lane. A consumer can compile against the public header with
`-DSPARSE_IDX_BITS=64` while linking a default-width archive; C symbol names
do not encode the width. The header's warning is valuable but does not
enforce the package contract. A full 64-bit correctness claim remains
unearned until an actual 64-bit build and downstream test pass.

**Close with:** select width at build configuration, propagate it through
installed CMake/pkg-config metadata, add an ABI/width guard and both positive
and mismatched-width installed-consumer tests, and run a bounded 64-bit test
lane. Keep shared-library ABI as a separate product decision.

### Medium: Selected performance freshness is not a regression policy

[`benchmarks/README.md:271`](../../../../benchmarks/README.md) accurately
records the canonical selected row as `--repeat 1`, no warmup, no variance,
`baseline=n/a`, and `threshold=n/a`. Linux/macOS workflows retain selected
freshness artifacts, but a one-sample row cannot separate noise from a real
change or support a speed/parity claim. Other local thresholded sentinels do
not automatically validate this canonical row. This is an evidence gap, not
misleading documentation: the README explicitly warns against overclaiming.

**Close with:** one named stable runner/workload, pre-registered warmup and
repeat policy, retained raw samples, robust dispersion and threshold rule,
versioned baseline, and negative drift/metadata fixtures. Keep conclusions
specific to that runner and workload.

### Medium: Selected Windows freshness remains unpromoted

The [Windows workflow](../../../../.github/workflows/windows-ci.yml) has
bounded Cholesky and QR comparison jobs. The [Epic 19 residual queue](../../EPIC_19/SPRINT_216/RESIDUAL_QUEUE.md)
records why they do not yet authorize selected Windows freshness: Cholesky's
historical artifact needs current metadata/support-tier/non-claim alignment;
QR needs an inspected hosted six-file artifact before promotion. This is not
a claim that Windows CMake/MSVC build or installed static consumption is
absent; [INSTALL.md](../../../../INSTALL.md) records those maintained paths.

**Close with:** separately inspect exact-commit hosted artifacts and promote
each target only when six files, manifest, generator metadata, docs, and
negative guards agree. A failed or unavailable hosted run leaves that target
deferred, not implicitly supported.

### Medium: Workflow-policy parser has a large custom grammar surface

[`scripts/check_api_docs_local_only.sh`](../../../../scripts/check_api_docs_local_only.sh)
is 2,085 lines and contains custom handling for YAML anchors, quoted scalar
folding, flow forms, shell commands, and PowerShell. Prior Sprint 213/PR 236
review cycles exercised many encoded and multiline bypasses. The current
guard and regression fixtures are valuable, but the policy's correctness
depends on maintaining multiple handwritten normalization passes as workflow
syntax evolves. This is a maintainability and policy-enforcement risk, not a
claim that the present guard is currently bypassable.

**Close with:** replace ad hoc YAML interpretation with a supported
structured parse or deliberately fail-closed restricted workflow grammar;
retain shell-aware command analysis, step boundaries, and every existing
allowed/forbidden fixture as differential tests. Preserve the local-only
generated-HTML policy unless publication is separately approved.

### Medium: Static install exists, but release/distribution does not

[`CMakeLists.txt:13`](../../../../CMakeLists.txt) deliberately rejects shared
builds and exports a static target. Unix Make/pkg-config, CMake consumers,
and a Windows CMake/MSVC subset have install tests. The
[Sprint 214 checklist](../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md)
still lacks an approved candidate with immutable source archive, matching
independent hashes, candidate-specific hosted jobs, sign-off, and rollback.
The [residual queue](../../EPIC_19/SPRINT_216/RESIDUAL_QUEUE.md) also keeps
public package-provider and shared/dynamic ABI paths deferred. This limits
usability for users who need a normal versioned distribution, even though
source installation works.

**Close with:** validate one exact static source-release candidate end to end
and document the installed first-use route. Do not imply public provider,
binary package, shared ABI, or published release without their distinct
approval gates.

### Medium: Test volume masks family-level and extreme-scale holes

The repository has broad C suites, fuzz/integration tests, sanitizers,
platform CI, and an aggregate [80% line-coverage gate](../../../../Makefile)
(`Makefile:941-950`). That is a strength. But an aggregate percentage does not
prove difficult branches of every numerical family, while selected tests may
be skipped for fixture-dependent reasons (this local `make test` ended with
one skip in `test_reorder_nd`). The `west0067` assertion described above is a
concrete example of a passing test without a useful finite/residual gate.
Current external-comparison evidence is intentionally local. The absent
proof is representative family-level conformance across structures, scales,
failures, and independent baselines, not a raw test-count problem.

**Close with:** prioritize mutation/exponent tests for shared arithmetic and
one solver family at a time; record per-family branch/fixture coverage,
oracles, seeds, tolerances, skipped reasons, and retained failures. An
all-family guarantee is beyond this epic.

### Low: First-use documentation is accurate but still heavy to navigate

[`README.md`](../../../../README.md) has a useful Start Here path and explicit
non-claims, but is 1,150 lines; `INSTALL.md` is 637 lines and the cookbook
383. The API reference is locally generated, not published. The detailed
support/readiness tables protect correctness, yet the amount of inline
historical evidence makes the shortest external-user route harder to scan.
This is a usability/documentation burden, not missing installation docs.

**Close with:** test a clean-prefix static consumer from the front-door
instructions, remove obsolete duplicated status prose without dropping claim
boundaries, and validate links and public-header/API coverage. Keep generated
HTML local until an explicit publication decision.

## Dimension Summary

| Dimension | Current strength | Limiting gap |
| --- | --- | --- |
| Efficiency | CSR/CSC interop and selected compressed direct backends exist. | Matmul output still repeats linked-list scans; controlled kernel-level measurements are limited. |
| Maintainability | Strict Make/CMake/CI gates and scoped regression guards exist. | The workflow-policy parser and several source/test/report files remain unusually large and costly to review. |
| Usability | Runnable examples, solver selection, and static installed consumers exist. | Width mismatch is not enforced in package metadata; release/provider and shared routes are absent. |
| Documentation | Support tiers and non-claims are unusually explicit; all 18 checked-in public headers have generated local reference pages. | The front door is long, generated HTML is local-only, and current versus historical evidence takes effort to separate. |
| Coherence | Static-first and selected-evidence policies are internally consistent. | Build/header width selection and some Windows generator/manifest claims do not yet move as one contract. |
| Test coverage | `make test`, sanitizer/CI lanes, corpus fixtures and an aggregate coverage gate provide a real baseline. | Extreme-scale norms, independent external cells, family-level conformance and selected hosted promotions remain incomplete. |

## Overall Assessment

The project is a substantial real-double, static-first C sparse library with
multiple direct, iterative, eigensolver and SVD paths, CSR/CSC interop,
installed consumers, platform CI, sanitizers, corpus tests, and unusually
careful claim boundaries. It is coherent as a **bounded source-install
library**. Its weakest current dimensions are extreme-scale numerical
diagnostics, linked-list-heavy output construction, independent comparison
evidence, controlled timing evidence, and distribution/ABI maturity.

**State-of-the-art decision: not demonstrated.** It would be unsound to infer
comparable correctness, robustness, memory efficiency, or speed from the
algorithm inventory alone. SuiteSparse includes distinct [CHOLMOD, UMFPACK,
SPQR and other packages](https://github.com/DrTimothyAldenDavis/SuiteSparse),
while [SciPy exposes SuperLU factor/solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html)
and [PETSc distinguishes solver and preconditioner selection](https://petsc.org/release/manual/ksp/).
Those references establish comparison dimensions, **not** a measured ranking
of this project. A credible ranking needs pre-registered representative
matrix families and sizes, exact independent versions/builds, two-sided
correctness and failure analysis, controlled paired timing/memory results,
and named platform coverage. One tiny accepted CHOLMOD cell, if achieved,
would be a useful first result but still would not establish a broad claim.

## Validation And Limits

- `make test` passed locally on 2026-10-19. The output included the passing
  `west0067` ILUT case printing `norm=inf` and one skipped `test_reorder_nd`
  case; the latter did not fail the suite.
- Only Epic 20 planning documents were added. `make docs-check` passed. No
  fresh `make lint`, sanitizer, coverage, CTest, Windows/macOS hosted,
  CHOLMOD, or benchmark timing run was performed for this document.
- Historical sprint artifacts and CI configuration were read as provenance,
  not reclassified as fresh execution evidence.
