# Sprint 175 Working Notes

## Sprint Goal

Promote one selected report freshness path beyond Linux or formally close the
cross-platform deferral with precise blockers.

## Source Artifact Note

The Sprint 175 request referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`,
but the active merged Sprint 175 planning source is
`docs/planning/EPIC_15/PROJECT_PLAN.md`, section
"Sprint 175: Cross-Platform Report Freshness Promotion".

## Branch Baseline

- Branch: `sprint-175`
- Starting point: current `master` after PR #193 merge.
- Sprint 173 status: complete and merged, with guarded local-only generated
  API HTML freshness under `make api-docs-freshness`.
- Sprint 174 status: complete and merged, with selected QR, partial-SVD, and
  LU generated comparison freshness under
  `make report-index-comparison-freshness`.
- Sprint 175 plan status: day-by-day plan exists at
  `docs/planning/EPIC_15/SPRINT_175/PLAN.md`.

## Prior Evidence Carried Forward

| Input | Source | Sprint 175 use |
| --- | --- | --- |
| Local generated API freshness | `docs/planning/EPIC_15/SPRINT_173/` | Keep generated Doxygen HTML local-only and separate from report freshness promotion. |
| Selected comparison freshness | `docs/planning/EPIC_15/SPRINT_174/` | Treat QR, partial-SVD, and LU comparison reports as selected local generated artifacts unless a reviewed promotion lane is explicitly implemented. |
| Oracle freshness | `make report-index-oracle-freshness` | Existing selected oracle report freshness gate, mirrored by reviewed Linux hosted CI only for selected artifacts. |
| Comparison freshness | `make report-index-comparison-freshness` | Existing selected comparison report freshness gate for QR minnorm, QR compatible least-squares, partial-SVD diagonal top-k, and LU nonsymmetric square solve. |
| Performance freshness | `make bench-canonical-report-freshness` | Existing selected benchmark report freshness, mirrored by reviewed Linux hosted selected-performance CI. |
| Report family ownership | `tests/corpus/manifests/report_families.tsv` | Source-controlled owner, command, artifact, support tier, and non-claim record for generated reports. |
| Report index normalization | `scripts/normalize_report_index.py` | Source-controlled selected row and generated artifact enforcement. |
| Static package/shared ABI deferral | `scripts/static_package_deferral_check.sh` | Guard support wording if Sprint 175 touches package, ABI, runtime-loader, or platform claims. |
| Package-manager deferral | `scripts/package_manager_deferral_check.sh` | Guard provider/package-manager non-claims if Sprint 175 touches adoption wording. |

## Current Generated Report Freshness Commands

| Command | Current meaning | Current platform/publishing boundary |
| --- | --- | --- |
| `make report-index-oracle-freshness` | Regenerates selected local oracle outputs and checks selected oracle freshness. | Local generated artifacts; selected lane is mirrored by reviewed Linux hosted CI only. |
| `make report-index-comparison-freshness` | Regenerates selected QR, partial-SVD, and LU comparison outputs and checks selected comparison freshness. | Local generated artifacts; selected lane is mirrored by reviewed Linux hosted report-freshness CI only. |
| `make bench-canonical-report-freshness` | Checks selected canonical benchmark report freshness. | Local selected row plus reviewed Linux hosted selected-performance lane. |
| `make api-docs-freshness` | Regenerates local Doxygen HTML, checks API page coverage, and enforces local-only staging. | Guarded local-only generated API HTML; not hosted or artifact-published. |

## Current Report Family Manifest Surface

The report-family manifest records these relevant generated/local families:

- `oracle/generated_reference`: `make report-index-oracle-freshness`,
  `build/corpus/oracle/*.tsv`, `local_only`;
- `oracle/solver_backed`: `make report-index-oracle-freshness`,
  `build/corpus/oracle/*.tsv`, `local_only`;
- `coverage/src`: `make coverage`, `coverage/coverage-src.info`,
  `local_only`;
- `report_index/missing_generated`:
  `python3 scripts/normalize_report_index.py`,
  `build/report-index/normalized-index.tsv`, `local_only`;
- `comparison/qr_minnorm`:
  `python3 scripts/run_external_comparison.py --target qr-minnorm`,
  `build/comparison/qr_minnorm/study.tsv`, `local_only`;
- `comparison/qr_compatible_ls`:
  `python3 scripts/run_external_comparison.py --target qr-compatible-ls`,
  `build/comparison/qr_compatible_ls/study.tsv`, `local_only`;
- `comparison/partial_svd_diag6_k2`:
  `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2`,
  `build/comparison/partial_svd_diag6_k2/study.tsv`, `local_only`;
- `comparison/lu_nonsym_square_5`:
  `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5`,
  `build/comparison/lu_nonsym_square_5/study.tsv`, `local_only`;
- `ci/reviewed_lanes`: GitHub Actions hosted CI lane definitions,
  `reviewed_cross_platform`, but not local report freshness proof by itself.

## Current Platform Boundary

- Linux is the reviewed hosted source of truth for selected report freshness
  lanes already promoted to hosted CI.
- macOS has reviewed static-first package/install evidence, but selected
  generated report freshness is not yet promoted as a macOS report-freshness
  lane.
- Windows has reviewed CMake-first build/test and CMake install/downstream
  evidence, but selected generated report freshness is not yet promoted as a
  Windows report-freshness lane.
- Generated comparison, oracle, API, coverage, missing-report, and benchmark
  outputs remain ignored local artifacts unless a specific reviewed hosted lane
  has been implemented and documented.

## Retained Claim Non-Claims

Sprint 175 starts with no support claim for:

- broad cross-platform report freshness;
- hosted publication of all generated reports;
- hosted publication of generated API HTML;
- macOS selected report freshness beyond explicitly reviewed lanes;
- Windows selected report freshness beyond explicitly reviewed lanes;
- Windows Makefile parity;
- Windows `pkg-config` command execution parity;
- package-manager provider availability;
- shared-library ABI support;
- runtime-loader behavior;
- release evidence;
- performance superiority beyond selected benchmark rows;
- external-library ecosystem parity;
- state-of-the-art sparse linear algebra coverage.

## Sprint 175 Stop Conditions

Stop and revise before proceeding if a change:

- promotes a platform support tier before selecting a specific lane;
- treats local generated output as hosted publication evidence;
- broadens reviewed Linux hosted report freshness into macOS or Windows
  support without an implemented lane;
- modifies selected report rows or artifacts without updating tests and
  manifest ownership;
- adds report-family rows without freshness or deferral enforcement;
- stages generated output under `build/`, `coverage/`, or `docs/api/`;
- weakens package-manager, static package, shared-library ABI, platform,
  performance, release, or state-of-the-art non-claims;
- changes `.c` or `.h` files without running `make format && make lint &&
  make test`.

## Working Assumptions

- Day 1 is planning and intake only.
- If only planning files change on a given day, `git diff --check` is
  sufficient for that day.
- If scripts, Make targets, workflows, report manifests, report-index rows,
  docs, or generated-output rules change later, run focused freshness,
  report-index, claim-scan, and deferral-guard checks.
- If `.c` or `.h` files change, run the full C quality gate.
- Sprint 175 should close one lane completely or leave a formal enforceable
  deferral rather than partially broadening multiple platforms.

## Daily Log

### Day 1: Sprint Intake And Report Freshness Boundary

- Re-read the active Sprint 175 section of
  `docs/planning/EPIC_15/PROJECT_PLAN.md`.
- Confirmed the prompt path points at an older Epic 12 planning file, while
  the active Sprint 175 section lives in Epic 15.
- Reviewed Sprint 173 generated API HTML retrospective and preserved its
  guarded local-only generated-output boundary.
- Reviewed Sprint 174 comparison retrospective and preserved its selected
  local QR, partial-SVD, and LU comparison freshness boundary.
- Inventoried current generated report freshness commands:
  `make report-index-oracle-freshness`,
  `make report-index-comparison-freshness`,
  `make bench-canonical-report-freshness`, and
  `make api-docs-freshness`.
- Inventoried current report-family manifest rows for oracle, coverage,
  report-index missing-generated, comparison, and CI reviewed lanes.
- Recorded retained platform, hosted, package, ABI, performance, release, and
  state-of-the-art non-claims before promotion selection.
- Created Sprint 175 artifact directory structure.
- Day 1 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day1-freshness-intake.md`.

### Day 2: Generated Report Family Inventory

- Reviewed the Day 2 plan scope: generated report commands, output staging,
  manifest mapping, platform assumptions, and current support-tier notes.
- Inventoried maintained report freshness commands:
  `make report-index-oracle-freshness`,
  `make report-index-comparison-freshness`,
  `make bench-canonical-report-freshness`, and
  `make api-docs-freshness`.
- Inventoried related local/advisory report commands for coverage, deadcode,
  performance sentinels, large-matrix guardrails, and normalized report-index
  output.
- Cross-referenced `tests/corpus/manifests/report_families.tsv` rows for
  oracle, benchmark, sentinel, guardrail, deadcode, coverage, package, CI,
  documentation, report-index, runtime-backend, and comparison families.
- Classified generated outputs under `build/`, `coverage/`, and `docs/api/`
  as ignored local artifacts unless an explicit reviewed hosted lane uploads
  selected artifacts.
- Reviewed `.github/workflows/ci.yml` generated-report freshness lanes and
  recorded that the Linux hosted selected comparison summary/upload inventory
  still lists only `qr-minnorm`, `qr-compatible-ls`, and
  `partial-svd-diag6-k2`, while the local Sprint 174 freshness target now also
  includes `lu-nonsym-square-5`.
- Identified platform assumptions for oracle, comparison, canonical benchmark,
  generated API, coverage, deadcode, and normalized report-index paths.
- Identified candidate lanes for later selection: Linux hosted comparison LU
  reconciliation, macOS selected comparison freshness, Windows selected
  comparison freshness deferral/promotion design, macOS/Windows oracle
  freshness, and generated API freshness as a non-report alternative.
- Day 2 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day2-generated-report-inventory.md`.

### Day 3: Platform Gap Matrix

- Built a cross-platform report freshness matrix for selected oracle,
  comparison, canonical benchmark, generated API, coverage, deadcode,
  sentinel/guardrail, and normalized report-index paths.
- Classified Linux selected oracle and selected comparison freshness as
  reviewed hosted only for the explicitly maintained selected lanes.
- Recorded that Linux hosted selected comparison freshness now has a concrete
  Sprint 174 mismatch: `make report-index-comparison-freshness` includes
  `lu-nonsym-square-5`, but the hosted summary/upload inventory still lists
  only `qr-minnorm`, `qr-compatible-ls`, and `partial-svd-diag6-k2`.
- Classified macOS selected report freshness as staged/local-only, despite
  existing reviewed macOS static-first package/install evidence.
- Classified Windows selected report freshness as blocked/staged because the
  maintained Windows support model is CMake-first while selected report
  freshness commands still carry Make/POSIX shell, temporary probe,
  executable, and generated-output assumptions.
- Separated local generation support from hosted publication support for
  commands, ignored generated artifacts, CI lane definitions, manifest rows,
  and normalized report-index output.
- Defined blocker classes: shell, path, compiler, dependency, executable,
  newline/encoding, temp directory, generated-output staging, CI
  permission/artifact, and claim wording.
- Ranked candidate lanes for Day 4 decision. The strongest concrete closure is
  Linux hosted selected comparison LU reconciliation; the strongest beyond
  Linux promotion candidate is macOS selected comparison freshness; the most
  defensible formal deferral is Windows selected comparison freshness with
  precise blockers.
- Day 3 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day3-platform-gap-matrix.md`.

### Day 4: Promotion Or Deferral Decision

- Reviewed the Day 3 platform matrix, blocker taxonomy, and ranked candidate
  lanes.
- Selected exactly one Sprint 175 path for complete closure: macOS selected
  comparison freshness promotion.
- Defined the selected lane as hosted macOS execution of
  `make report-index-comparison-freshness` for four selected comparison
  targets: `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, and
  `lu-nonsym-square-5`.
- Defined the post-implementation support tier as reviewed macOS selected
  comparison freshness, bounded to selected fixture-local generated
  comparison artifacts only.
- Recorded required implementation work: macOS workflow lane, all-four-target
  summary, all-four-target artifact upload, path/execution audit, maintained
  docs, and local/hosted validation.
- Recorded that the existing Linux hosted selected comparison lane must be
  reconciled for the Sprint 174 LU addition while implementing the selected
  macOS lane.
- Rejected Linux-only LU reconciliation as the primary Sprint 175 lane because
  it does not promote report freshness beyond Linux.
- Deferred Windows selected comparison promotion/deferral until macOS selected
  comparison freshness is either implemented or blocked.
- Preserved non-claims for broad macOS parity, Windows report freshness,
  hosted publication of all generated reports, hosted generated API HTML,
  broad report-index freshness, unselected comparison families, package,
  ABI, runtime-loader, release, performance, external-library parity, and
  state-of-the-art claims.
- Day 4 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day4-promotion-decision.md`.

### Day 5: Path And Execution Assumption Audit

- Traced the selected macOS comparison freshness lane from
  `make report-index-comparison-freshness` through all four
  `scripts/run_external_comparison.py --target ...` calls and the final
  `scripts/normalize_report_index.py --family comparison --require-generated
  comparison --check-freshness` check.
- Audited `scripts/run_external_comparison.py` path handling: repository root
  resolution, `Path`-based output directories, temporary project probe
  directories, include paths, static library path, manifest relative paths,
  and generated output paths are macOS-compatible.
- Audited shell/executable assumptions: Make invokes `python3`, runner invokes
  helper scripts via `sys.executable`, project probes compile with `CC` or
  default `cc`, and macOS does not require `.exe` handling.
- Audited temporary probe behavior: source and binary live under
  `tempfile.mkdtemp(prefix="sparse-comparison-")`, compile and run commands
  are argv-list subprocess calls, and temp cleanup uses `shutil.rmtree` unless
  `--keep-temp` is used.
- Audited text output: TSV writers use UTF-8, `newline=""`, tab delimiters,
  and LF line terminators; helper output is parsed with `splitlines()`;
  summaries and manifests are text-only.
- Confirmed generated comparison outputs must remain ignored local files under
  `build/comparison/*` unless uploaded by a reviewed hosted lane.
- Identified minimal normalization required for Day 6/Day 7: add a macOS
  workflow job, summarize all four selected targets, upload all six generated
  files for each selected target directory, and reconcile the existing Linux
  hosted selected comparison inventory to include `lu-nonsym-square-5`.
- Day 5 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day5-path-execution-audit.md`.

### Day 6: Normalization Design

- Converted the Day 5 path and execution audit into a concrete implementation
  design for reviewed macOS selected comparison freshness.
- Designed a new `.github/workflows/macos-ci.yml` job named macOS reviewed
  selected comparison freshness that runs
  `make report-index-comparison-freshness`, summarizes selected outputs, and
  uploads selected generated comparison artifacts.
- Defined the exact four-target hosted summary inventory:
  `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, and
  `lu-nonsym-square-5`.
- Defined expected selected generated row counts: 6, 6, 10, and 6 rows,
  respectively, for 28 total generated selected comparison rows.
- Designed artifact upload behavior for all six generated files from each of
  the four selected comparison target directories.
- Designed Linux hosted selected comparison reconciliation so the existing
  Linux summary/upload inventory includes the Sprint 174 LU target and matches
  the new macOS inventory.
- Defined documentation follow-through for README, maintainer guide, corpus
  docs, and benchmark/report-index handoff docs while preserving package, ABI,
  Windows, performance, release, and state-of-the-art non-claims.
- Recorded that no Python runner or Make target path normalization is required
  before the macOS workflow implementation.
- Day 6 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day6-normalization-design.md`.

### Day 7: Normalization Implementation

- Implemented the selected Sprint 175 promotion lane by adding a
  `selected-comparison-freshness` job to `.github/workflows/macos-ci.yml`.
- The new macOS job runs `make report-index-comparison-freshness`, summarizes
  all four selected generated comparison targets, and uploads selected
  comparison artifacts with `if-no-files-found: error`.
- Added explicit macOS workflow summary assertions for selected row counts:
  `6` rows for `qr-minnorm`, `6` rows for `qr-compatible-ls`, `10` rows for
  `partial-svd-diag6-k2`, and `6` rows for `lu-nonsym-square-5`.
- Added macOS manifest provenance assertions for `source_commit`,
  `source_branch`, and `platform`.
- Reconciled the existing Linux reviewed hosted selected comparison freshness
  job so its summary and artifact upload inventory includes
  `lu-nonsym-square-5`.
- Added the same exact row-count and manifest provenance assertions to the
  Linux selected comparison summary.
- Normalized the reconciled Linux hosted artifact name to
  `sprint175-linux-selected-comparison-freshness`.
- Confirmed generated comparison outputs remain ignored local files under
  `build/comparison/*`; hosted publication is via workflow artifacts only.
- Validation passed:
  `make report-index-comparison-freshness`,
  `python3 tests/test_run_external_comparison.py`,
  `python3 tests/test_normalize_report_index.py`,
  `python3 scripts/run_external_comparison.py --self-check`,
  `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`,
  workflow selected comparison inventory check, and `git diff --check`.
- Preserved non-claims for Windows report freshness, broad macOS parity,
  hosted publication of all generated reports, generated API HTML hosting,
  unselected comparison families, package-manager support, shared-library ABI,
  runtime-loader behavior, release evidence, performance superiority, and
  state-of-the-art status.
- Day 7 changed workflow and planning artifacts only. No `.c` or `.h` files
  were modified, so the full C quality gate is not required for this day.
- Created `artifacts/day7-normalization-implementation.md`.

### Day 8: Gate Integration

- Identified the Day 8 integration point as the Linux/macOS GitHub Actions
  selected comparison freshness workflow pair.
- Added `tests/test_selected_comparison_workflow.py` as a source-controlled
  workflow guard for the selected comparison hosted lanes.
- The guard verifies that both Linux and macOS workflow lanes run
  `make report-index-comparison-freshness`.
- The guard verifies all four selected targets, expected row counts, uploaded
  generated files, `if-no-files-found: error`, and fail-closed summary
  assertions for row counts and manifest provenance.
- The guard verifies the macOS selected comparison freshness lane preserves
  explicit non-claims for Windows report freshness, external-library parity,
  package/ABI support, performance superiority, and state-of-the-art status.
- Confirmed no Make target, report-family manifest, or normalized report-index
  row-set change was required on Day 8.
- Validation passed:
  `python3 tests/test_selected_comparison_workflow.py`, workflow selected
  comparison inventory check,
  `python3 -m py_compile tests/test_selected_comparison_workflow.py`,
  `bash scripts/package_manager_deferral_check.sh`,
  `bash scripts/static_package_deferral_check.sh`, and `git diff --check`.
- Preserved generated-output staging: `build/comparison/*` remains ignored
  local output and hosted Linux/macOS evidence is workflow-artifact-only.
- Day 8 changed workflow-test and planning artifacts only. No `.c` or `.h`
  files were modified, so the full C quality gate is not required for this
  day.
- Created `artifacts/day8-gate-integration.md`.

### Day 9: Documentation Tier Update

- Updated `README.md` so selected comparison freshness is documented as local
  plus reviewed Linux/macOS hosted selected-artifact evidence.
- Updated `README.md` platform support wording so macOS selected comparison
  freshness is visible without broad macOS parity or Windows report freshness
  claims.
- Updated `docs/maintainer_guide.md` support-tier and report freshness
  sections to document reviewed macOS selected comparison freshness.
- Updated `tests/corpus/README.md` to separate Linux selected oracle hosting
  from Linux/macOS selected comparison hosting.
- Updated `benchmarks/README.md` report handoff wording to mention reviewed
  Linux/macOS hosted selected-artifact lanes for selected comparison
  freshness only.
- Validation passed:
  `python3 tests/test_selected_comparison_workflow.py`,
  `bash scripts/package_manager_deferral_check.sh`,
  `bash scripts/static_package_deferral_check.sh`, and `git diff --check`.
- Preserved non-claims for selected oracle freshness on macOS, Windows report
  freshness, broad report-index freshness, unselected generated families,
  package-manager support, shared-library ABI, runtime-loader behavior,
  release evidence, performance superiority, external-library parity, and
  state-of-the-art status.
- Day 9 changed documentation and planning artifacts only. No `.c` or `.h`
  files were modified, so the full C quality gate is not required for this
  day.
- Created `artifacts/day9-documentation-tier-update.md`.

### Day 10: Report Index And Manifest Reconciliation

- Reviewed `scripts/normalize_report_index.py` selected comparison row ids and
  artifact paths; no script change was required because all four selected
  comparison families were already represented and strict freshness already
  fails closed.
- Reconciled `tests/corpus/manifests/report_families.tsv` so the CI
  reviewed-lanes row names Linux selected oracle/comparison freshness and
  macOS selected comparison freshness.
- Kept selected comparison manifest rows as `generated_local`, `local_only`,
  `generated_compare_inputs` metadata rather than incorrectly promoting
  generated TSV rows to hosted support tier.
- Updated selected comparison manifest non-claims from broad `no hosted CI
  proof` wording to `no hosted CI proof from generated-local row metadata`.
- Added explicit selected comparison manifest non-claims for no broad platform
  portability proof and no Windows report freshness.
- Added
  `test_selected_comparison_manifest_support_tiers_remain_bounded` to
  `tests/test_normalize_report_index.py`.
- Validation passed:
  `python3 tests/test_normalize_report_index.py`,
  `python3 tests/test_selected_comparison_workflow.py`,
  `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`,
  `bash scripts/package_manager_deferral_check.sh`,
  `bash scripts/static_package_deferral_check.sh`, and `git diff --check`.
- Day 10 changed manifest, Python test, documentation, and planning artifacts
  only. No `.c` or `.h` files were modified, so the full C quality gate is not
  required for this day.
- Created `artifacts/day10-report-index-reconciliation.md`.

### Day 11: Cross-Platform Claim Review

- Scanned maintained docs, workflows, report-family manifests, and focused
  tests for stale selected comparison, platform, hosted, package, ABI,
  performance, release, and state-of-the-art wording.
- Confirmed selected comparison freshness is consistently bounded to local
  generation plus reviewed Linux/macOS hosted selected-artifact evidence.
- Confirmed selected oracle freshness remains Linux-hosted only; Linux-only
  oracle wording in README and the oracle generated-reference manifest row is
  intentional, not stale selected-comparison wording.
- Confirmed selected comparison generated rows remain `generated_local`,
  `local_only`, and `generated_compare_inputs`; hosted evidence lives in
  workflow artifacts and CI lane metadata.
- Confirmed Windows report freshness, selected oracle freshness on macOS,
  hosted publication of all generated reports, hosted generated API HTML,
  broad report-index freshness, unselected comparison families,
  package-manager support, shared-library ABI, runtime-loader behavior,
  release evidence, performance superiority, external-library parity, and
  state-of-the-art status remain non-claims.
- No additional stale wording fixes were required on Day 11.
- Validation passed:
  targeted stale wording scans,
  `python3 tests/test_selected_comparison_workflow.py`,
  `python3 tests/test_normalize_report_index.py`,
  `bash scripts/package_manager_deferral_check.sh`,
  `bash scripts/static_package_deferral_check.sh`, and `git diff --check`.
- Day 11 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day11-cross-platform-claim-review.md`.

### Day 12: Integrated Validation

- Ran the selected Sprint 175 freshness lane:
  `make report-index-comparison-freshness`.
- Confirmed the selected freshness lane regenerated all four selected
  comparison families: `qr-minnorm`, `qr-compatible-ls`,
  `partial-svd-diag6-k2`, and `lu-nonsym-square-5`.
- Confirmed the selected comparison freshness gate reported
  `normalize-report-index: freshness ok (32 rows)`.
- Ran focused validation:
  `python3 tests/test_run_external_comparison.py`,
  `python3 tests/test_normalize_report_index.py`,
  `python3 tests/test_selected_comparison_workflow.py`,
  `python3 scripts/run_external_comparison.py --self-check`,
  `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`,
  `bash scripts/package_manager_deferral_check.sh`,
  `bash scripts/static_package_deferral_check.sh`, and `git diff --check`.
- Confirmed generated selected comparison outputs exist under
  `build/comparison/*` and remain ignored by Git; hosted Linux/macOS evidence
  remains workflow-artifact-only.
- Preserved remaining deferrals for Windows report freshness, selected oracle
  freshness on macOS, hosted publication of all generated reports, hosted
  generated API HTML, broad report-index freshness, unselected comparison
  families, package-manager support, shared-library ABI, runtime-loader
  behavior, release evidence, performance superiority, external-library
  parity, and state-of-the-art status.
- Day 12 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day12-integrated-validation.md`.

### Day 13: Maintenance And Handoff Review

- Reviewed Sprint 175 artifacts and working notes from Day 1 through Day 12.
- Confirmed the selected lane is maintainable through one shared Make target,
  workflow summary/upload assertions, `tests/test_selected_comparison_workflow.py`,
  and report-index/manifest guards in `tests/test_normalize_report_index.py`.
- Identified fragile assumptions: duplicated Linux/macOS workflow summary
  Python, explicit artifact upload path lists, generated-local support tiers
  that must not be confused with hosted workflow evidence, intentional
  Linux-only selected oracle wording, and deferred Windows report freshness.
- Recorded manual review points for future selected comparison freshness
  changes: runner targets, normalized row ids, workflows, workflow guard,
  report-index guard, report-family manifest, and support-tier docs.
- Recorded future automation opportunities: factor workflow summary logic,
  generate artifact path lists from target inventory, expose selected target
  metadata for tests/workflows, and design Windows freshness separately.
- Prepared Day 14/Sprint 176 handoff boundaries: final item reconciliation,
  final focused validation, generated-output staging check, and preservation
  of selected comparison, selected oracle, Windows, broad report-index, and
  unselected-family support tiers.
- Validation passed:
  targeted stale wording scans,
  `python3 tests/test_selected_comparison_workflow.py`,
  `python3 tests/test_normalize_report_index.py`,
  `bash scripts/package_manager_deferral_check.sh`,
  `bash scripts/static_package_deferral_check.sh`, and `git diff --check`.
- Day 13 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day13-maintenance-review.md`.

### Day 14: Sprint Closeout And Final Freshness Record

- Reconciled Sprint 175 project-plan items 175.1 through 175.6 against the
  completed artifacts, workflow changes, documentation updates, manifest
  reconciliation, and validation records.
- Confirmed the platform gap matrix, selected macOS comparison promotion,
  path/execution audit, normalization design, CI/workflow guard, documentation
  tier updates, report-family manifest updates, and final validation evidence
  are complete.
- Ran the selected freshness target:
  `make report-index-comparison-freshness`.
- Confirmed the selected freshness target regenerated all four selected
  comparison families: `qr-minnorm`, `qr-compatible-ls`,
  `partial-svd-diag6-k2`, and `lu-nonsym-square-5`.
- Confirmed the selected comparison freshness gate reported
  `normalize-report-index: freshness ok (32 rows)`.
- Ran focused validation:
  `python3 tests/test_run_external_comparison.py`,
  `python3 tests/test_normalize_report_index.py`,
  `python3 tests/test_selected_comparison_workflow.py`,
  `python3 scripts/run_external_comparison.py --self-check`,
  `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`,
  `bash scripts/package_manager_deferral_check.sh`,
  `bash scripts/static_package_deferral_check.sh`, and `git diff --check`.
- Confirmed generated selected comparison outputs exist under
  `build/comparison/*` and remain ignored by Git; hosted Linux/macOS selected
  comparison evidence remains workflow-artifact-only.
- Preserved remaining deferrals for Windows report freshness, selected oracle
  freshness on macOS, hosted publication of all generated reports, hosted
  generated API HTML, broad report-index freshness, unselected comparison
  families, package-manager support, shared-library ABI, runtime-loader
  behavior, release evidence, performance superiority, external-library
  parity, and state-of-the-art status.
- Prepared Sprint 176 handoff notes: start from bounded local plus
  Linux/macOS selected comparison evidence, do not infer unsupported platform
  or hosted-publication claims, and consider factoring duplicated workflow
  summary/artifact logic before expanding another lane.
- Day 14 changed planning artifacts only. No `.c` or `.h` files were modified,
  so the full C quality gate is not required for this day.
- Created `artifacts/day14-sprint-closeout.md`.
