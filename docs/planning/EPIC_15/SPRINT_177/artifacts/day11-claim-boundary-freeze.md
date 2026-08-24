# Sprint 177 Day 11: Claim Boundary Freeze

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Requested sprint path:** `docs/planning/EPIC_15/SPRINT_177/`
**Status:** Complete

## Purpose

Freeze current public claim boundaries before the Epic 16 implementation
sprints begin. Later sprints may update public wording only after their
acceptance gates pass and their evidence is reconciled into the matrix.

## Reviewed Claim Surfaces

The freeze reviewed these public and maintainer-facing surfaces:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `docs/solver_selection.md`
- `docs/tutorial.md`
- `docs/cookbook.md`
- `benchmarks/README.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- Sprint 177 evidence/status matrix and acceptance gates

## Current Frozen Claim Boundaries

| Surface | Current allowed claim | Protected non-claims |
| --- | --- | --- |
| Package/install | Static-first install/export is reviewed on named Linux, macOS, and Windows lanes, with Windows CMake install/downstream validation and metadata-only `sparse.pc` inspection. | No package-manager support, shared-library packaging, dynamic ABI, runtime-loader behavior, Windows Makefile parity, or Windows pkg-config execution parity. |
| Package-manager support | Provider support is deferred and guarded by package-manager deferral checks. | No registry, tap, recipe, binary package, upgrade behavior, provider availability, or package-manager parity claim. |
| Shared-library and ABI | Static-first-only posture is the product decision; `BUILD_SHARED_LIBS=ON` rejects. | No shared-library support, symbol visibility policy, import/export ABI, dynamic ABI compatibility, loader metadata, installed shared consumer, or runtime-loader validation. |
| Generated API HTML | Doxygen HTML is local-only generated output validated by `make api-docs-freshness`. | No hosted API publication, source-controlled generated HTML, artifact-published generated HTML, release evidence, or completeness beyond configured inputs. |
| Selected oracle freshness | Selected QR/partial-SVD oracle freshness is local and mirrored by reviewed Linux hosted report-freshness evidence. | No macOS selected oracle freshness, Windows report freshness, broad oracle proof, broad report-index freshness, package/ABI support, release proof, or state-of-the-art claim. |
| Selected comparison freshness | Selected QR, partial-SVD, and LU comparison freshness is local and mirrored by reviewed Linux/macOS hosted selected comparison lanes. | No unselected comparison, broad external-library parity, Windows report freshness, performance claim, package/ABI support, release proof, or state-of-the-art claim. |
| Selected performance | One selected Linux hosted and local canonical benchmark freshness path exists for `bench_refactor_csc` on `nos4.mtx --repeat 1`. | No raw timing gate, portable performance, superiority, backend parity, OpenMP speedup, external-library parity, broad platform proof, release proof, or state-of-the-art performance. |
| Allocation-failure proof | Focused local CG/GMRES/MINRES repeated-run handle allocation-failure proof exists. | No broad allocation-failure guarantee across direct solvers, eigensolvers, matrix construction, package/install flows, generated-report tooling, or unrelated allocation paths. |
| Windows support | Windows is CMake-first with reviewed CTest coverage and CMake install/downstream validation for static-first package surface. | No Windows report freshness, Makefile parity, pkg-config execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |
| Solver evidence | Solver docs may cite selected fixture-local tests, selected oracle rows, and selected comparison rows. | No broad solver correctness, external-library parity, platform/package/ABI support, performance, or state-of-the-art claims beyond named fixtures and gates. |
| Maintainer report index | Normalized report index is navigation and freshness context. | No release proof, package proof, ABI proof, broad platform proof, broad generated-report parity, broad performance proof, or state-of-the-art evidence from index rows alone. |

## Candidate Wording Updates After Evidence Lands

| Future sprint | Wording may change only if | Candidate surfaces |
| --- | --- | --- |
| Sprint 178 | One additional allocation-heavy subsystem gains deterministic cleanup proof and focused validation. | README allocation-failure bullet, maintainer allocation-failure section, quality command list. |
| Sprint 179 | Generated API status decision is implemented and guarded. | README command list, API reference entry point, maintainer generated API section, tutorial/cookbook navigation. |
| Sprint 180 | One provider proof lands or a stronger deferral is recorded and guarded. | README, INSTALL support split, maintainer package section, package metadata comments. |
| Sprint 181 | Selected-target manifest owns selected report target metadata and guards read or validate it. | Maintainer report workflow, README report commands, benchmark report-index handoff, workflow comments. |
| Sprint 182 | One Windows-safe report freshness lane lands or a formal deferral guard lands. | README CI bullet, INSTALL supported platforms, maintainer platform/report sections, Windows workflow comments. |
| Sprint 183 | One additional bounded comparison family lands with manifest rows and freshness checks. | Solver-selection docs, README, maintainer solver evidence table, report-index docs. |
| Sprint 184 | One public header family is cleaned with declaration-preserving validation. | API reference, header docs, tutorial/cookbook examples, maintainer API docs section. |
| Sprint 185 | One large review surface is reduced without behavior change. | Maintainer contribution notes or selected cluster maintenance note. |
| Sprint 186 | Evidence reconciliation confirms which target gates passed. | README, INSTALL, maintainer guide, benchmark docs, report docs, Epic 16 retrospective. |

## Protected Non-Claim Phrases

These phrases or their exact meaning must remain visible unless a later sprint
explicitly earns evidence and updates the matrix:

- no state-of-the-art claim
- no broad external-library parity
- no portable performance claim
- no package-manager support
- no shared-library support
- no dynamic ABI support
- no runtime-loader behavior
- no broad Windows parity
- no Windows Makefile parity
- no Windows pkg-config execution parity
- no Windows report freshness
- no broad generated-report freshness
- no broad allocation-failure guarantee
- no generated API hosted publication unless selected and proven
- no source-controlled generated API HTML unless selected and proven

## Claim Update Rules

1. Do not update public claims from planned sprint work alone.
2. Update positive wording only after the target acceptance gate passes.
3. Keep adjacent non-claims in the same paragraph, table row, or nearby
   maintainer note as the positive claim.
4. If a workflow lane is the proof source, name the exact workflow job scope.
5. If evidence is local-only, say local-only and name the command.
6. If a row is source-controlled metadata, do not treat it as proof that a
   generator or validation command just ran.
7. If a sprint defers a capability, strengthen the deferral wording and guard
   instead of softening the non-claim.
8. If C/header files change while updating wording, run the full C quality
   gates from Day 10.

## Current Consistency Assessment

| Area | Status | Notes |
| --- | --- | --- |
| README | Consistent | CI, report, performance, allocation-failure, generated API, package, and ABI wording preserve current boundaries. |
| INSTALL | Consistent | Static-first, platform support, package-manager, shared-library, ABI, runtime-loader, and Windows limitations remain explicit. |
| Maintainer guide | Consistent | Detailed interpretation sections match the Day 6 evidence matrix. |
| Solver-selection/tutorial/cookbook | Consistent | Solver evidence remains fixture-local and rejects parity/performance/package/ABI overclaims. |
| Benchmark docs | Consistent | Performance rows remain freshness/methodology context rather than portable performance claims. |
| Workflows | Consistent | Workflow comments describe selected hosted evidence and adjacent non-claims. |

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Current public wording is consistent with current evidence | Complete | Freeze table maps current claims to Day 6 matrix boundaries. |
| Future claim updates are tied to specific sprint gates | Complete | Candidate update table requires Sprint 178-186 gate completion before wording changes. |
| Unsupported surfaces remain explicit non-claims | Complete | Protected non-claim phrases preserve package, ABI, platform, report, performance, allocation, and state-of-the-art boundaries. |
