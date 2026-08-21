# Day 13: Maintenance And Handoff Review

## Purpose

Review Sprint 175 artifacts and working notes through Day 12, confirm the
selected macOS comparison freshness lane is maintainable, identify fragile
assumptions, and prepare Day 14/Sprint 176 handoff boundaries.

## Maintained Lane Summary

Sprint 175 selected and implemented reviewed macOS selected comparison
freshness.

The maintained path is:

```sh
make report-index-comparison-freshness
```

The selected hosted Linux and macOS workflow lanes run the same Make target,
summarize all four selected comparison targets, and upload selected generated
artifacts as workflow artifacts only.

Selected targets:

| Target | Directory | Expected generated rows |
| --- | --- | ---: |
| `qr-minnorm` | `build/comparison/qr_minnorm` | 6 |
| `qr-compatible-ls` | `build/comparison/qr_compatible_ls` | 6 |
| `partial-svd-diag6-k2` | `build/comparison/partial_svd_diag6_k2` | 10 |
| `lu-nonsym-square-5` | `build/comparison/lu_nonsym_square_5` | 6 |

The normalized freshness gate still validates 32 comparison rows: four
source-controlled contract rows plus 28 generated selected rows.

## Maintainability Review

### What Is Maintainable

- The selected freshness command is a single Make target shared by local,
  Linux hosted, and macOS hosted validation.
- `scripts/run_external_comparison.py` owns target definitions, output
  directories, generated row schemas, and manifest generation.
- `scripts/normalize_report_index.py` owns selected comparison row identity,
  artifact diagnostics, and strict freshness behavior.
- `tests/test_selected_comparison_workflow.py` guards Linux/macOS workflow
  target inventories, row counts, artifact upload paths, fail-closed summary
  checks, and macOS non-claim wording.
- `tests/test_normalize_report_index.py` guards selected comparison row
  identity, artifact paths, freshness failure modes, and report-family
  manifest support-tier boundaries.
- Documentation now describes the selected comparison lane as local plus
  reviewed Linux/macOS hosted selected-artifact evidence.

### Fragile Assumptions

- The Linux and macOS workflow summary scripts duplicate target inventory,
  expected row counts, and manifest checks. The workflow guard catches drift,
  but future maintainers must update both workflow scripts when adding or
  removing selected comparison targets.
- Workflow artifact upload paths enumerate every generated file explicitly.
  This is fail-closed and reviewable, but path additions require updates in
  both workflows and in `tests/test_selected_comparison_workflow.py`.
- Generated comparison row support tiers must remain `local_only` even when
  hosted workflow artifacts exist. Hosted evidence belongs to workflow
  artifacts and CI lane metadata, not generated-local TSV row support tiers.
- Linux-only selected oracle wording is intentional and should not be
  mechanically converted to macOS until a separate oracle-hosting lane exists.
- Windows report freshness remains unpromoted. Future Windows work should be a
  separate CMake/PowerShell-native or otherwise Windows-safe design, not an
  inference from the macOS lane.

## Manual Review Points

Before changing selected comparison freshness, review:

1. `scripts/run_external_comparison.py` target definitions.
2. `scripts/normalize_report_index.py` selected comparison row ids and
   artifact diagnostics.
3. `.github/workflows/ci.yml` selected comparison summary and artifact upload.
4. `.github/workflows/macos-ci.yml` selected comparison summary and artifact
   upload.
5. `tests/test_selected_comparison_workflow.py` workflow guard.
6. `tests/test_normalize_report_index.py` report-index and manifest guards.
7. `tests/corpus/manifests/report_families.tsv` support tiers and non-claims.
8. `README.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md`, and
   `benchmarks/README.md` support-tier wording.

## Future Automation Opportunities

- Factor the duplicated Linux/macOS workflow summary logic into a small
  checked-in helper script.
- Generate workflow artifact path lists from the selected comparison target
  inventory instead of maintaining explicit workflow lists by hand.
- Add a manifest-level helper that emits selected target names, directories,
  expected row counts, and required artifact filenames for tests and workflow
  scripts.
- Add a workflow syntax validation tool if the repository standardizes on one.
- Consider a future Windows report freshness design only after selecting a
  Windows-safe execution model and artifact policy.

## Sprint 176 Handoff

Day 14 should close Sprint 175 by:

- reconciling project-plan items against the final artifacts;
- rerunning final focused validation;
- confirming generated `build/comparison/*` outputs remain ignored;
- confirming no `.c` or `.h` files were modified;
- preserving the support-tier boundary:
  - selected comparison freshness: local plus reviewed Linux/macOS hosted
    selected-artifact evidence;
  - selected oracle freshness: Linux hosted only;
  - Windows report freshness: not promoted;
  - broad report-index freshness and unselected families: not promoted.

Sprint 176/Epic 15 closeout should use Sprint 175 evidence as a bounded
cross-platform report freshness promotion, not as a broad platform, package,
performance, release, external-library parity, or state-of-the-art claim.

## Validation Results

| Check | Result |
| --- | --- |
| targeted stale selected-comparison Linux-only scan | Passed with intentional oracle-only exception. |
| targeted selected oracle/macOS and Windows report freshness scan | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `git diff --check` | Passed. |

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 13.
