# Day 9: Documentation Tier Update

## Purpose

Update maintained user and maintainer documentation so the Sprint 175 support
tier matches the implemented selected comparison freshness lane: local
generation plus reviewed Linux and macOS hosted selected-artifact evidence.

## Updated Documentation

### README

Updated `README.md` to state that:

- `make report-index-comparison-freshness` is mirrored by reviewed
  Linux/macOS hosted CI for selected QR, partial-SVD, and LU comparison
  freshness;
- Linux still owns selected oracle hosted freshness;
- macOS now owns selected comparison hosted freshness only;
- Windows report freshness remains unsupported;
- broad report-index freshness and unselected generated families remain
  unpromoted.

### Maintainer Guide

Updated `docs/maintainer_guide.md` to record:

- macOS reviewed hosted selected comparison freshness as selected generated
  artifact evidence;
- selected comparison freshness is mirrored by Linux and macOS hosted lanes;
- selected oracle freshness on macOS remains unpromoted;
- Windows report freshness, broad report-index freshness, package/ABI support,
  performance, release readiness, external-library parity, and state-of-the-art
  status remain non-claims.

### Corpus Documentation

Updated `tests/corpus/README.md` to separate hosted evidence tiers:

- reviewed Linux hosted evidence covers selected oracle and selected
  comparison freshness;
- reviewed macOS hosted evidence covers selected comparison freshness only;
- generated corpus/report rows remain fixture-local.

### Benchmark/Report Handoff

Updated `benchmarks/README.md` so the generated report handoff table describes
`make report-index-comparison-freshness` as mirrored by reviewed Linux/macOS
hosted selected-artifact lanes only.

## Support-Tier Result

The maintained support tier after Day 9 is:

| Surface | Tier |
| --- | --- |
| Selected comparison freshness, local | Maintained Make target for four selected families. |
| Selected comparison freshness, Linux hosted | Reviewed selected-artifact workflow evidence. |
| Selected comparison freshness, macOS hosted | Reviewed selected-artifact workflow evidence. |
| Selected oracle freshness, Linux hosted | Reviewed selected-artifact workflow evidence. |
| Selected oracle freshness, macOS hosted | Not promoted. |
| Windows report freshness | Not promoted. |
| Broad report-index freshness | Not promoted. |
| Unselected generated families | Local-only, supplemental, or advisory unless separately promoted. |

## Preserved Non-Claims

Day 9 keeps these boundaries explicit:

- no Windows report freshness;
- no broad platform parity;
- no hosted publication of all generated reports;
- no selected oracle freshness on macOS;
- no unselected comparison family freshness;
- no package-manager support;
- no shared-library ABI support;
- no runtime-loader support;
- no release evidence;
- no performance superiority;
- no external-library parity;
- no state-of-the-art sparse linear algebra claim.

## Validation Results

Because Day 9 changed documentation and claim wording only, the validation
scope was documentation/claim focused:

| Check | Result |
| --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `git diff --check` | Passed. |

No `.c` or `.h` files were modified, so the full C quality gate is not
required for Day 9.
