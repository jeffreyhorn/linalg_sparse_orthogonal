# Sprint 163 Day 5 Schema Gap Analysis

## Purpose

Day 5 compares the current selected report scripts, report-family metadata,
normalizer behavior, documentation, and validation commands against the Day 4
methodology contract. This artifact defines the exact implementation work for
Days 6 and 7 without widening Sprint 163 beyond the Day 3 selected surface.

## Reviewed Sources

- `scripts/bench_canonical_report.sh`
- `scripts/performance_sentinels.sh`
- `scripts/wall_check.sh`
- `scripts/normalize_report_index.py`
- `tests/corpus/manifests/report_families.tsv`
- `Makefile`
- `README.md`
- `benchmarks/README.md`

## Current Coverage Summary

| Area | Current State | Contract Fit |
| --- | --- | --- |
| Canonical report generation | `scripts/bench_canonical_report.sh` emits one CSV per selected canonical benchmark plus `index.tsv` and `manifest.txt`. | Strong base, but `index.tsv` needs explicit methodology fields for publication. |
| Canonical provenance | `index.tsv` records surface, category, label, UTC time, commit, branch, platform, compiler, build mode, thread setting, artifact, relative path, and command. | Meets core provenance requirements. |
| Canonical caveats | `manifest.txt` states threshold-free local/CI snapshot and not portable claims. | Good but weaker than Day 4 public caveat wording. |
| Sentinel report generation | `scripts/performance_sentinels.sh` emits `sentinels.tsv`, `manifest.txt`, wall-check output, and threshold-free raw CSVs. | Strong fit for S5/S2/S3 split. |
| Sentinel row classification | `sentinels.tsv` records `support_tier` and `claim_boundary`; S5 uses `local_wall_gate`, S2/S3 use `local_threshold_free`. | Meets gate versus threshold-free distinction. |
| Sentinel provenance | Manifest records UTC time, commit, branch, platform, compiler, build mode, threads, and backend env requests. Rows record build mode, threads, command, fixture, metric, artifact, backend fields. | Rows inherit some provenance from manifest; acceptable if docs make inheritance explicit, but row-level baseline provenance is missing. |
| Normalized report index | `scripts/normalize_report_index.py` maps benchmark rows to advisory status and separates sentinel hard gates from sentinel advisory measurements by `claim_boundary`. | Preserves selected row meaning and non-claims. |
| Report-family manifest | `tests/corpus/manifests/report_families.tsv` already defines benchmark and sentinel families with non-claims. | Fits Day 4 advisory/local-only boundary, though benchmark family wording can be made methodology-bound. |
| Public docs | `README.md` and `benchmarks/README.md` repeatedly state benchmark rows are local evidence, not portable guarantees. | Mostly aligned; Day 8+ docs should adopt the Day 4 caveat wording and selected-row language. |

## Methodology Field Gap Table

| Contract Field | Canonical `index.tsv` | Canonical `manifest.txt` | Sentinel `sentinels.tsv` | Sentinel `manifest.txt` | Gap / Action |
| --- | --- | --- | --- | --- | --- |
| report family | `surface=canonical` | `surface=canonical` | `report_family=sentinel` | script title | Add explicit `report_family=benchmark` or document `surface=canonical` mapping for canonical rows. |
| row/artifact identity | `artifact`, `relative_path` | artifact list | `sentinel_id`, metric, artifact | artifact list | Covered. |
| category/support tier | `category=measurement` | `category=measurement` | `support_tier` | not row-level | Add canonical `support_tier=local_only` or `reviewed_threshold_free`; preserve sentinel. |
| claim boundary | missing | notes only | `claim_boundary` | notes | Add canonical `claim_boundary=local_threshold_free`; preserve sentinel. |
| row state/status | missing | notes only | `status` | notes | Add canonical `status=measurement` or `report`; preserve sentinel. |
| command | `command` | command mapping | `command` | command list | Covered. |
| artifact path | `relative_path` plus report directory | artifact list | `artifact` | artifact list | Covered; canonical should make raw artifact local-only explicit. |
| generated UTC | `generated_at_utc` | `generated_at_utc` | inherited from manifest only | `generated_at_utc` | Accept inherited sentinel provenance; no row-level duplication required unless Day 6 chooses to add it. |
| git commit / branch | row fields | manifest fields | inherited from manifest only | manifest fields | Accept inherited sentinel provenance; normalizer already reads manifest. |
| platform / compiler | row fields | manifest fields | inherited from manifest only | manifest fields | Accept inherited sentinel provenance; docs should state inheritance. |
| build mode / threads | row fields | manifest fields | row fields | manifest fields | Covered. |
| fixture/workload | canonical command only | command mapping | `matrix_or_fixture` | command list | Add or document canonical workload mapping; sentinel covered. |
| matrix size | raw CSV may include benchmark-specific fields | missing | raw CSV may include fields; `matrix_or_fixture` only in TSV | missing | Record as missing/not recorded unless source rows emit it; do not block Day 6 if caveated. |
| repeat count | command text for `--repeat 1` rows only | command mapping | command text for S2/S3 | command list | Add explicit `repeat_count` or `repeat_semantics` fields for selected rows. |
| warmup | missing | missing | missing | missing | Add explicit `warmup=not_recorded` or `warmup_semantics=not_recorded`. |
| variance | missing | missing | missing | missing | Add explicit `variance=not_recorded` or `variance_semantics=not_recorded`. |
| baseline | missing | not applicable | `baseline` | raw wall-check notes | Canonical threshold-free rows should emit `baseline=n/a`; S5 needs baseline provenance. |
| threshold | missing | not applicable | `threshold` | notes | Canonical threshold-free rows should emit `threshold=n/a`; sentinel covered. |
| backend/runtime context | build mode and threads only | build mode and threads | backend request/selected/fallback, dense kernel, panel solver | env requests | Covered for sentinel; canonical should emit `backend_context=n/a` or use row-specific caveat. |
| local-only caveat | notes only | notes | notes field | notes | Add stronger Day 4 caveat text to manifests and docs. |

## Script And Schema Change List

### `scripts/bench_canonical_report.sh`

Required or strongly recommended:

- Add explicit row-class fields to `index.tsv`:
  - `report_family`
  - `status`
  - `support_tier`
  - `claim_boundary`
  - `baseline`
  - `threshold`
  - `repeat_count` or `repeat_semantics`
  - `warmup`
  - `variance`
  - `methodology_notes`
- Preserve existing fields and column order compatibility where practical by
  appending new fields instead of renaming current fields.
- Add Day 4 caveat wording to `manifest.txt`.
- Add a single place in the script for canonical methodology constants so
  future row additions do not drift.

### `scripts/performance_sentinels.sh`

Required or strongly recommended:

- Add `baseline_provenance` to `sentinels.tsv`, with S5 pointing at
  `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt` and S2/S3 using
  `n/a`.
- Add explicit `repeat_count` or `repeat_semantics`, `warmup`, and `variance`
  fields.
- Consider adding `methodology_notes` while preserving existing `notes`.
- Strengthen manifest caveats with Day 4 wording.
- Preserve nonzero exit behavior when S5 fails.

### `scripts/normalize_report_index.py`

Focused review only unless Day 6/7 script changes break parsing:

- Confirm benchmark generated rows keep new canonical fields visible in
  `configuration` or future normalized fields.
- Confirm sentinel rows continue to separate S5 by `claim_boundary`.
- Confirm S2/S3 remain advisory when `status=report`.
- No broad schema redesign is needed for Day 6.

### `tests/corpus/manifests/report_families.tsv`

Possible documentation-level refinement:

- Update benchmark and sentinel family wording only if script field additions
  create clearer methodology-bound semantics.
- Do not change family meanings in a way that turns advisory benchmark rows
  into release proof.

### `README.md` And `benchmarks/README.md`

Required during documentation alignment:

- Add selected-row language for the Sprint 163 publication surface.
- Use Day 4 public caveat wording or equivalent.
- State that canonical rows are threshold-free, S5 is the only selected hard
  local timing gate, and S2/S3 are threshold-free backend-context rows.
- Keep package, ABI, runtime-loader, platform, external-library, and
  state-of-the-art non-claims intact.

## Focused Test And Self-Check Map

| Change Area | Required Check |
| --- | --- |
| Canonical script field additions | `make bench-canonical-report`; inspect `build/bench-reports/canonical/index.tsv` header and `manifest.txt`. |
| Sentinel script field additions | `make performance-sentinels`; inspect `build/bench-reports/sentinels/sentinels.tsv` header, S5/S2/S3 rows, and `manifest.txt`. |
| Normalizer compatibility | After generated reports exist, run `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv`. |
| Documentation caveats | `rg -n "portable performance|state-of-the-art|S5|threshold-free|bench-canonical-report|performance-sentinels" README.md benchmarks/README.md docs/maintainer_guide.md`. |
| Shell script syntax | `bash -n scripts/bench_canonical_report.sh scripts/performance_sentinels.sh`. |
| Documentation-only changes | `git diff --check`; no C/header gate required if no `*.c` or `*.h` files change. |

## Unsupported-Wording Risk Register

| Risk | Where It Could Appear | Required Guard |
| --- | --- | --- |
| Canonical rows described as release benchmarks | README, benchmark docs, manifests | Use "threshold-free local measurement" and "methodology-bound local artifact". |
| S5 described as portable timing guarantee | sentinel manifest, docs, PR description | Always pair S5 status with baseline, threshold, fixture, command, and local wall-check context. |
| S2/S3 described as passing or failing | sentinel docs or normalized index text | Use "threshold-free local backend-context rows". |
| Backend selected/fallback hidden | sentinel report/index summaries | Preserve backend request, selected backend, fallback, dense-kernel, and panel-solver context. |
| Package proof reused as performance proof | README, retrospective, PR description | Keep Sprint 162 package proof explicitly separate from Sprint 163 performance evidence. |
| OpenMP context read as speedup proof | benchmark/sentinel manifests | Treat build mode and thread count as methodology context only. |
| Normalized index read as release proof | report-index docs | Keep normalized index as navigation/freshness aid. |

## Day 6 Implementation Scope

Day 6 should start with `scripts/bench_canonical_report.sh` because canonical
rows have the largest methodology-field gap and the lowest risk of changing
gate behavior. The first implementation should append fields, strengthen
manifest caveats, and verify with `bash -n` plus `make bench-canonical-report`
if runtime is acceptable.

Sentinel changes can follow once canonical schema changes are stable, with care
to preserve the existing S5 failure behavior.

## Completion Check

- Report enhancement work is source-backed.
- Script and documentation edits are scoped to selected rows and fields.
- Required validation commands are known before implementation begins.
