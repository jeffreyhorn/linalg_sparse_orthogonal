# Day 13 Final Residual Queue And Closeout Prep

## Purpose

Day 13 consolidates residuals from Sprint 157-165 retrospectives and Sprint
166 reconciliation artifacts into an actionable final queue. Each residual is
recorded with an owner, blocker, prerequisite, promotion gate, and recommended
next-epic priority.

This artifact also prepares PR description bullets, review-risk notes, a
validation summary, and the Day 14 closeout checklist.

## Residual Queue

| Priority | Residual | Owner | Current blocker | Prerequisite | Promotion gate |
| --- | --- | --- | --- | --- | --- |
| P0 | Sprint 166 PR-hosted CI confirmation. | Sprint 166 closeout / PR reviewer. | Hosted Linux/macOS/Windows workflow results do not exist until the branch is pushed and PR CI runs. | Branch pushed with Sprint 166 changes. | PR checks pass, or failures are reconciled with evidence and fixes before merge. |
| P1 | Hosted performance publication proof. | Future performance-publication sprint. | Sprint 163 rows are local generated benchmark/sentinel evidence only. | Select hosted runner, compiler, build flags, commands, artifact retention, row-state semantics, and expected report files. | Reviewed hosted lane runs selected benchmark/sentinel publication checks and uploads methodology-bound artifacts without superiority wording. |
| P1 | Shared-library ABI product design. | Future package/ABI sprint. | Current product support is static-first and explicitly rejects shared-library requests. | Define export/import macros, symbol visibility, SONAME/install-name/DLL/import-library behavior, ABI versioning policy, installed shared consumers, and runtime-loader validation. | Shared-library builds install and pass platform-specific downstream consumers with ABI/loader docs and CI proof, or the static-only decision is reaffirmed with guards. |
| P1 | Package-manager distribution readiness. | Future package/distribution sprint. | No package-manager provenance, update/upgrade policy, support matrix, or provider-specific CI proof exists. | Choose target managers and layouts such as Homebrew, apt, dnf, pacman, vcpkg, Conan, or an explicit subset. | Selected package-manager recipes install, compile/link/run downstream consumers, validate metadata/version behavior, and publish support-tier docs. |
| P1 | Broader public-header cleanup batch. | Future API/documentation sprint. | Sprint 164 cleaned only `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h`. | Select next headers and capture normalized declaration baseline. | Declaration-preserving cleanup lands for selected headers, generated docs pass, public docs stay coherent, and `make format && make lint && make test` passes if headers change. |
| P2 | Additional bounded QR/SVD/partial-SVD comparison families. | Future comparison/corpus sprint. | Epic 14 comparison evidence covers selected fixture-local families only. | Select one family, fixtures, dense reference, metrics, tolerances, row IDs, and non-claims. | Descriptor-backed comparison generator, focused tests, normalized selected freshness rows, docs, and hosted/local evidence classification all pass. |
| P2 | macOS/Windows generated report-freshness parity. | Future platform/report sprint. | Reviewed report-freshness promotion is Linux hosted only. | Decide whether cross-platform generated report freshness is worth the runtime and dependency cost. | macOS and/or Windows hosted report-freshness lanes pass selected oracle/comparison commands with artifact upload and support-tier wording. |
| P2 | Generated API HTML hosted publication. | Future documentation-publication sprint. | Sprint 158 intentionally chose ignored/local-only generated HTML. | Decide publication target, artifact retention or committed-output policy, freshness cadence, and warning ownership. | Hosted or committed generated HTML is reproducible, warning-clean, coverage-checked, linked from public docs, and no longer described as local-only. |
| P2 | Generated `sparse_version.h` API-doc representation. | Future API-docs sprint. | `sparse_version.h` is generated at install/build time and is not a checked-in public-header input page. | Decide whether generated installed-header docs are needed and how to avoid stale generated output. | Doxygen/source-header policy explicitly covers generated version header pages or keeps them excluded with tests and docs. |
| P2 | Statistical benchmark methodology. | Future benchmark-methodology sprint. | Sprint 163 added methodology fields, but selected canonical rows still lack a full statistical methodology for superiority claims. | Define repeats, warmup, variance, hardware isolation, baseline provenance, outlier policy, thresholds, and reporting format. | Benchmarks produce statistically interpretable rows and docs still reject unsupported superiority claims unless evidence warrants promotion. |
| P2 | Backend superiority or OpenMP speedup proof. | Future runtime/performance sprint. | S2/S3 rows are threshold-free backend context and S5 is a local wall-check gate. | Define selected workload family, platform matrix, thread policy, thresholds, and comparison baseline. | Backend/speedup claims pass methodology-bound local and hosted validation with explicit caveats. |
| P2 | Windows Makefile install/uninstall parity. | Future Windows package sprint. | Sprint 162 retained Windows Makefile parity as a non-claim. | Select shell/toolchain model, install layout, path handling, uninstall semantics, and downstream consumer proof. | Windows Makefile install/uninstall tests pass in hosted CI and docs distinguish supported Makefile behavior from CMake package behavior. |
| P2 | Windows `pkg-config` command execution parity. | Future Windows package sprint. | Sprint 162 retained Windows `pkg-config` execution parity as a non-claim. | Select provider, path conventions, exact expected `--exists`, `--cflags`, `--libs`, `--modversion`, and downstream consumer behavior. | Hosted Windows `pkg-config` lane executes commands, compiles/links/runs downstream consumers, and rejects unsupported package/ABI wording. |
| P3 | Broad report-index freshness for all generated families. | Future report-governance sprint. | Epic 14 freshness gates apply to selected generated rows, not every report family. | Inventory all report families and choose which should be required, advisory, optional, or deferred. | Normalizer fails closed only for selected required rows and docs make unselected/advisory semantics explicit. |
| P3 | Optional NumPy/SciPy promoted pass evidence. | Future comparison-dependency sprint. | Optional dependency rows are currently context, not selected proof. | Select dependency policy, version/provenance capture, fallback behavior, and CI installation method. | Optional rows are promoted only if dependencies are pinned, available in CI, and failures are meaningful rather than environmental. |
| P3 | Table-wide README/API index reshaping and tutorial expansion. | Future adoption sprint. | Epic 14 touched selected docs only and did not redesign every user-facing navigation table or option/result field. | Select high-value adoption workflows and public API surfaces. | Docs provide coherent first-use, solver-selection, API-reference, cookbook, install, and troubleshooting paths without unsupported claims. |
| P3 | Maintained declaration-preservation helper. | Future tooling sprint. | Sprint 164 used declaration-preservation evidence, but no reusable maintained helper target was selected. | Choose parser/normalizer approach and output location. | A make/script target captures normalized public declarations, supports selected-header subsets, and is documented for review workflows. |
| P3 | Broad external-library parity and state-of-the-art positioning. | Future competitive-validation epic. | Current evidence is fixture-local and methodology-bound; it does not compare broadly against mature sparse linear algebra ecosystems. | Select libraries, versions, fixtures, metrics, tolerances, platforms, compilers, package provenance, failure semantics, and performance methodology. | Claims name exact comparison scope, pass reviewed validation, and avoid generalizing beyond measured fixtures and platforms. |

## Residuals Not To Present As Epic 14 Completion

- Hosted generated API HTML publication.
- Committed `docs/api/html/` freshness or release evidence.
- Broad generated report freshness for every generated family.
- Unselected generated oracle/comparison families as pass evidence.
- macOS/Windows report-freshness parity.
- Broad QR, SVD, or partial-SVD correctness.
- Raw QR basis identity, QR sign/orientation/order identity, raw singular
  vector identity, phase identity, or repeated-spectrum ordering.
- External-library or ecosystem parity.
- Hosted performance proof, portable performance, backend superiority, OpenMP
  speedup portability, or state-of-the-art performance evidence.
- Windows Makefile install/uninstall parity.
- Windows `pkg-config` command execution parity.
- Package-manager distribution.
- Shared-library support.
- Dynamic ABI compatibility.
- Runtime-loader behavior.
- Static/shared package selector UX.
- Broad Windows package or platform parity.
- Broad public-header cleanup beyond the selected Sprint 164 header batch.
- Sprint 166 PR-hosted CI success before PR workflows run and are reconciled.

## PR Description Draft

Suggested PR summary bullets:

- Add Sprint 166 final-validation plan, working notes, and daily evidence
  artifacts through final reconciliation and retrospective drafting.
- Reconcile Epic 14 generated API, hosted generated evidence, QR comparison,
  partial-SVD comparison, Windows package, performance, public-header, and
  static-first package outcomes against the project plan.
- Update reviewed Linux hosted comparison workflow wording, summary, and
  artifact upload paths so selected comparison evidence covers QR min-norm,
  QR compatible least-squares, and partial-SVD diag6 k2 families.
- Tighten public documentation wording so selected comparison rows are local
  generated evidence by default and reviewed Linux hosted evidence only after
  the hosted report-freshness lane runs.
- Tighten package wording around Windows `pkg-config` command execution parity
  while preserving static-first package and ABI non-claims.
- Publish the Epic 14 retrospective draft, final residual queue, review risks,
  and Day 14 closeout checklist.

Suggested validation bullets:

- `make format`
- `make lint`
- `make test`
- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_normalize_report_index.py`
- `python3 tests/test_run_external_comparison.py`
- `python3 -m py_compile scripts/normalize_report_index.py scripts/run_external_comparison.py scripts/run_corpus_oracle.py`
- `make docs-check`
- `make report-index-oracle-freshness`
- `make report-index-comparison-freshness`
- `python3 scripts/normalize_report_index.py --check`
- `python3 scripts/normalize_report_index.py --family package --check`
- `python3 scripts/normalize_report_index.py --family package --check-freshness`
- `bash scripts/static_package_deferral_check.sh`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- `make bench-canonical-report`
- `make performance-sentinels`
- targeted public/workflow claim scans
- `git diff --check`

## Review-Risk Notes

| Risk | Review focus |
| --- | --- |
| Hosted comparison evidence could be overread as broad external parity. | Check `.github/workflows/ci.yml`, README, maintainer guide, solver-selection docs, corpus docs, and report-index schema wording for selected-family and fixture-local boundaries. |
| Sprint 159 historical hosted scope could conflict with Sprint 160/161 selected-family growth. | Review Sprint 166 Day 7 and Day 10-11 reconciliation artifacts; current selected-comparison hosted claims should cite Day 7. |
| Local generated rows could be mistaken for hosted or release proof. | Check docs distinguish local generated output, source-controlled metadata, advisory rows, selected required rows, and hosted artifact evidence. |
| Package wording could imply shared-library, dynamic ABI, runtime-loader, package-manager, or broad Windows support. | Review `INSTALL.md`, `README.md`, `docs/maintainer_guide.md`, `sparse.pc.in`, CMake comments, and static package guard results. |
| Performance report wording could imply superiority. | Confirm benchmark/sentinel rows remain local, methodology-bound, and non-superiority unless future hosted methodology exists. |
| Public-header cleanup could be treated as complete for all headers. | Keep Sprint 164 claims limited to `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h`; broader header cleanup remains residual. |
| Day 12 retrospective draft could be mistaken for final Epic retrospective. | Day 14 must either create `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md` or explicitly hand it off. |
| Branch-level hosted CI remains unproven until PR workflows run. | PR description and final closeout should say local validation passed and hosted evidence is pending until CI completes. |

## Day 14 Closeout Checklist

- Update Sprint 166 working notes with final changed files, validation
  commands, and known residuals.
- Decide whether to publish `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md`
  on this branch from the Day 12 draft and Day 13 residual queue.
- Verify Sprint 166 plan, working notes, daily artifacts, retrospective draft,
  project-plan reconciliation, residual queue, and public-doc changes are
  internally consistent.
- Run final documentation and touched-surface validation:
  - `git diff --check`;
  - targeted stale hosted-comparison wording scan;
  - targeted unsupported package/ABI/platform claim scan;
  - additional touched-surface checks if Day 14 edits public docs, workflows,
    scripts, `.c`, or `.h` files.
- Record final validation summary.
- Record PR description bullets and review-risk notes.
- Record next-epic handoff.
- Confirm no generated build, docs, report, install, coverage, or cache output
  is staged for commit.

## Completion Check

- Residuals are actionable and evidence-bounded.
- Each future work item has a promotion gate.
- Completed Epic 14 claims remain separate from residual work.
- PR closeout material is ready for Day 14.

## Validation

- Documentation/planning artifact only for Day 13.
- No `.c` or `.h` files were modified for this Day 13 residual queue.
- `git diff --check` passed after the artifact and working-notes update.
