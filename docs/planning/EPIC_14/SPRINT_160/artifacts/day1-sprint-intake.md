# Day 1 Sprint Intake And Handoff Review

## Scope

Sprint 160 closes one additional QR comparison family end to end. The sprint
starts from the Sprint 159 hosted report-freshness surface and must avoid
broadening QR, external-library parity, platform, package, performance, ABI, or
state-of-the-art claims.

The current source section is
`docs/planning/EPIC_14/PROJECT_PLAN.md`, `Sprint 160: QR Comparison Family
Closure`. The prompt cited an older Epic 12 line range, but the current
project-plan owner for Sprint 160 is Epic 14.

## Branch Baseline

| Field | Value |
| --- | --- |
| Branch | `sprint-160` |
| Starting commit | `cd92502465ca21c96fc81ac5b268bba715a56a88` |
| Starting commit summary | `cd925024 Merge pull request #177 from jeffreyhorn/sprint-159` |
| Baseline source | current `master` after merged PR #177 |

## Sprint 159 Handoff Reviewed

| Handoff item | Sprint 160 posture |
| --- | --- |
| Hosted report freshness | Linux reviewed hosted evidence exists for selected oracle and selected QR minimum-norm comparison gates only. |
| Comparison artifacts | Current comparison artifact group is `sprint159-comparison-qr-minnorm` with 7-day retention and selected rows plus dependency context. |
| Normalizer semantics | Selected current generated rows report `fresh`; invalid selected rows fail. |
| Optional dependencies | NumPy/SciPy defers remain context and cannot create pass evidence. |
| Broad report index | Still advisory/local and not a hosted proof surface. |
| Platform scope | Linux reviewed hosted execution only; no macOS or Windows report-index parity. |

The Sprint 159 recommended first step is to choose one additional QR comparison
family, preferably an overdetermined compatible QR least-squares fixture with
residual and solution checks against the source-controlled dense helper. Day 2
will make that target decision explicitly.

## Current QR Comparison Surface Inventory

| Area | Concrete owner | Current behavior |
| --- | --- | --- |
| Make gate | `Makefile`, `report-index-comparison-freshness` | Builds the library, regenerates selected QR minimum-norm comparison output, then runs strict selected comparison freshness. |
| Generator | `scripts/run_external_comparison.py` | Supports only `--target qr-minnorm`; writes `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`. |
| Output directory | `build/comparison/qr_minnorm/` | Current generated comparison artifact root. |
| Dense helper | `tests/qr_external_dense_reference.py` | Source-controlled reference helper with compatible, incompatible, rank-deficient residual, and minimum-norm QR fixture builders. |
| Selected row IDs | `scripts/normalize_report_index.py` and `tests/test_normalize_report_index.py` | Fixed row set for `qr_underdetermined_minnorm_2x4`: project status, baseline status, residual norm, solution norm, solution values, and max absolute delta. |
| Report metadata | `tests/corpus/manifests/report_families.tsv` | `comparison/qr_minnorm` describes the current local generated comparison family and non-claims. |
| QR C proof owners | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_qr_corpus.c`, `tests/test_qr_helpers.h` | Own solver behavior, corpus proof, and QR helper coverage. |
| Public docs | `README.md`, `docs/maintainer_guide.md`, `docs/solver_selection.md`, `tests/corpus/README.md` | Describe one selected QR minimum-norm comparison and maintain broad non-claims. |
| Hosted CI | `.github/workflows/ci.yml` | Reviewed Linux hosted report-freshness lane runs selected oracle/comparison gates and uploads selected artifacts. |

## Day 2 Candidate Register

| Candidate | Why it is plausible | Initial risk |
| --- | --- | --- |
| `qr_overdetermined_compatible_5x3` | Existing dense helper builder, exact compatible least-squares solution, likely stable residual and solution metrics. | Needs row IDs and tolerances before any generator changes. |
| `qr_overdetermined_incompatible_4x2` | Existing dense helper builder and meaningful least-squares residual. | Nonzero residual semantics need careful metric wording. |
| `qr_rankdef_duplicate_5x4_residual_only` | Existing dense helper builder and rank-deficient residual coverage. | Could imply broad rank-deficient solve if wording is loose. |
| `qr_rankdef_dependent_row_4x3_residual_only` | Existing dense helper builder and dependent-row residual coverage. | Could drift into rank-threshold or basis claims. |

Day 2 should prefer the simplest family that can be closed with stable,
fixture-local comparison rows and no broad parity implications.

## Non-Goals Locked Before Target Selection

- No broad QR parity with LAPACK, NumPy, SciPy, SuiteSparse, Eigen, or any
  other external-library ecosystem.
- No raw QR basis identity, sign/orientation identity, or basis ordering claim.
- No global rank-threshold policy.
- No broad rank-deficient solve, nullspace, minimum-norm, or least-squares
  correctness claim beyond named fixture rows.
- No optional NumPy/SciPy pass evidence.
- No macOS or Windows report-index parity.
- No package-manager, package metadata, ABI, shared-library, dynamic-linking,
  performance, release, or state-of-the-art proof.
- No hosted CI edits until target, metrics, row IDs, artifact paths, runtime,
  and row-state semantics are explicitly documented.

## Completion Check

- Sprint 160 scope is tied to the Epic 14 project plan.
- Sprint 159 hosted freshness handoff was reviewed and summarized.
- QR comparison, corpus, oracle, and report-index owners are inventoried.
- Broad unsupported claims are blocked before Day 2 target selection.

## Day 2 Handoff

Day 2 should select one QR fixture family, document rejected alternatives, and
map the selected family to exact fixture owner, generated row IDs, metrics,
tolerances, artifact paths, support tier, claim scope, and non-claim wording.
