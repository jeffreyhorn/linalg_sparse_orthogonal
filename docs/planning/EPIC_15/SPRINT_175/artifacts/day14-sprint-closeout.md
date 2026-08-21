# Day 14: Sprint Closeout And Final Freshness Record

## Purpose

Close Sprint 175 by reconciling project-plan items 175.1 through 175.6,
recording the final validation results, confirming generated-output staging,
and preparing the handoff boundary for Sprint 176.

## Final Outcome

Sprint 175 promoted a reviewed macOS selected comparison freshness lane while
preserving the local generated-output boundary for selected comparison TSV and
Markdown artifacts. The sprint also reconciled the Linux selected comparison
lane to include the LU nonsymmetric square target, added workflow-level tests
for Linux and macOS selected comparison artifact publication, and updated the
documentation and report-family manifest to distinguish generated-local rows
from hosted workflow evidence.

The selected comparison lane now has:

- local freshness through `make report-index-comparison-freshness`;
- reviewed Linux hosted selected comparison artifact evidence;
- reviewed macOS hosted selected comparison artifact evidence;
- workflow tests that enforce target inventory, row counts, summary wording,
  artifact paths, and fail-closed upload behavior;
- report-index and manifest tests that keep generated-local support tiers
  bounded.

Generated comparison outputs remain under `build/comparison/*` and are ignored
by Git. Hosted evidence is workflow-artifact-only.

## Project-Plan Reconciliation

| Item | Result | Evidence |
| --- | --- | --- |
| 175.1 Platform Gap Matrix | Complete | `day2-generated-report-inventory.md` and `day3-platform-gap-matrix.md` classify generated reports by Linux, macOS, Windows, local-only, hosted, artifact-only, and blocker status. |
| 175.2 Promotion Candidate | Complete | `day4-promotion-decision.md` selects macOS selected comparison freshness as the single promotion lane and records deferred alternatives. |
| 175.3 CI or Deferral Work | Complete | `.github/workflows/macos-ci.yml` adds reviewed macOS selected comparison freshness; `.github/workflows/ci.yml` reconciles Linux selected comparison freshness; `tests/test_selected_comparison_workflow.py` enforces the selected lane. |
| 175.4 Path Normalization | Complete | `day5-path-execution-audit.md`, `day6-normalization-design.md`, and `day7-normalization-implementation.md` show the selected path is portable through repository-relative Make/Python invocation, POSIX shell on Linux/macOS, explicit artifact paths, and no staged generated output. |
| 175.5 Tier Documentation | Complete | `README.md`, `docs/maintainer_guide.md`, `benchmarks/README.md`, `tests/corpus/README.md`, and `tests/corpus/manifests/report_families.tsv` reflect Linux/macOS selected comparison evidence without broad platform, Windows, or hosted-publication claims. |
| 175.6 Verification | Complete | Day 12 and Day 14 validation passed the selected freshness target, external comparison tests, workflow guard, manifest/report-index guard, package/static deferral guards, and whitespace hygiene. |

## Deliverable Reconciliation

| Deliverable | Status |
| --- | --- |
| Cross-platform report freshness matrix | Complete in `day3-platform-gap-matrix.md`. |
| Selected promotion or formal deferral | Complete in `day4-promotion-decision.md`; selected macOS comparison promotion. |
| Path/execution audit and normalization design | Complete in Day 5 and Day 6 artifacts. |
| CI or enforceable deferral work | Complete through Linux/macOS selected comparison workflows and `tests/test_selected_comparison_workflow.py`. |
| Platform-tier documentation | Complete across README, maintainer guide, benchmark docs, corpus docs, and manifest metadata. |
| Final validation and closeout record | Complete in this Day 14 artifact. |

## Final Validation

The final Day 14 validation passed:

| Check | Result |
| --- | --- |
| `make report-index-comparison-freshness` | Passed; regenerated four selected comparison families and reported `normalize-report-index: freshness ok (32 rows)`. |
| `python3 tests/test_run_external_comparison.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 scripts/run_external_comparison.py --self-check` | Passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | Passed with 32 fresh selected comparison rows. |
| `bash scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `git diff --check` | Passed. |

No `.c` or `.h` files were modified during Day 14, so the full C quality gate
(`make format && make lint && make test`) was not required for this day.

## Generated-Output Staging Check

The selected freshness target regenerated 24 files under these ignored
directories:

- `build/comparison/qr_minnorm/`
- `build/comparison/qr_compatible_ls/`
- `build/comparison/partial_svd_diag6_k2/`
- `build/comparison/lu_nonsym_square_5/`

`git status --short --ignored build/comparison` reports `!! build/`, confirming
the generated outputs remain ignored and are not source-controlled sprint
artifacts.

## Remaining Deferrals And Non-Claims

Sprint 175 intentionally does not claim:

- Windows report freshness;
- selected oracle freshness on macOS;
- hosted publication of all generated reports;
- hosted publication of generated API HTML;
- broad report-index freshness across every generated family;
- freshness for unselected comparison families;
- package-manager provider availability;
- shared-library ABI support;
- runtime-loader behavior;
- release evidence;
- performance superiority;
- external-library ecosystem parity;
- state-of-the-art sparse linear algebra status.

## Sprint 176 Handoff

Sprint 176 should start from the bounded selected comparison evidence now in
place: local Make freshness plus reviewed Linux/macOS selected comparison
workflow artifacts. It should not infer macOS selected oracle freshness,
Windows report freshness, broad hosted report publication, or package/ABI
readiness from Sprint 175.

If Sprint 176 expands report freshness further, the next maintainable step is
to factor the duplicated Linux/macOS workflow summary logic and derive artifact
path lists from the selected target inventory before adding another platform
or report family.
