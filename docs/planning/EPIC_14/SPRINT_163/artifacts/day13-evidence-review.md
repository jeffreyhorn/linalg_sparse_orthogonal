# Sprint 163 Day 13 Evidence Review

## Purpose

Day 13 traces each Sprint 163 performance-publication statement back to the
selected command, generated row, documentation surface, and validation artifact.
The review also confirms that retained non-claims remain explicit boundaries
rather than implied positive proof.

## Positive Claim-To-Evidence Trace

| Claim | Evidence | Boundary |
| --- | --- | --- |
| Canonical benchmark reports publish a methodology-bound local measurement snapshot. | `scripts/bench_canonical_report.sh`, `make bench-canonical-report`, `build/bench-reports/canonical/index.tsv`, Day 11 and Day 12 row checks, `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`. | Rows are `status=measurement`, `support_tier=local_only`, `claim_boundary=local_threshold_free`, `baseline=n/a`, and `threshold=n/a`; they are not portable performance proof. |
| The S5 sentinel remains a hard local wall-check timing gate. | `scripts/performance_sentinels.sh`, `scripts/wall_check.sh`, `docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt`, `make performance-sentinels`, Day 11 and Day 12 S5 row checks. | S5 rows are local wall gates only; they require baseline provenance and local runner context and do not create broad platform or hosted proof. |
| S2 and S3 sentinel rows publish backend-context observations without pass/fail meaning. | `scripts/performance_sentinels.sh`, `make performance-sentinels`, `build/bench-reports/sentinels/sentinels.tsv`, Day 11 and Day 12 S2/S3 row checks, benchmark and maintainer docs. | Rows are `status=report` and `claim_boundary=local_threshold_free`; they are not backend superiority claims. |
| The normalized report index preserves benchmark and sentinel methodology fields for navigation. | `scripts/normalize_report_index.py`, `python3 tests/test_normalize_report_index.py`, `python3 scripts/normalize_report_index.py --family benchmark --family sentinel --output build/report-index/normalized-index.tsv`, report-index schema notes. | Normalized rows are advisory report-index metadata; they are not release, hosted, package, ABI, platform, or performance proof by themselves. |
| Public and maintainer documentation present generated outputs as local-only evidence. | `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, `tests/corpus/schemas/report_index_fields.md`, Day 10 documentation review, Day 12 unsupported-claim scan. | Documentation separates local performance reports from package, ABI, runtime-loader, package-manager, external-library, OpenMP, backend-superiority, and state-of-the-art claims. |

## Retained Non-Claim Trace

| Non-Claim | Guard Or Documentation Surface |
| --- | --- |
| Portable performance guarantee | `README.md`, `benchmarks/README.md`, `scripts/bench_canonical_report.sh`, `docs/maintainer_guide.md`, and report-index schema wording keep generated rows local-only. |
| State-of-the-art evidence | Public, benchmark, maintainer, and schema docs name this as unsupported by Sprint 163 generated rows. |
| Backend superiority | S2/S3 sentinel rows are `status=report`, threshold-free backend context; maintainer docs and README reject superiority proof. |
| Hosted CI proof | Day 12 hosted-only checklist and public docs require a hosted lane before citing hosted evidence. |
| Package, package-manager, ABI, shared-library, and runtime-loader proof | `bash scripts/static_package_deferral_check.sh` passed on Day 12; docs keep package/install evidence separate from performance evidence. |
| OpenMP speedup proof | README and maintainer docs list OpenMP speedup as outside the selected Sprint 163 evidence surface. |
| External-library parity | README and maintainer docs keep comparison/parity proof outside the generated local report rows. |
| Release proof | Report-index schema and maintainer docs describe normalized rows as navigation metadata, not release proof. |

## Wording Review

The Day 13 sensitive-phrase scan covered:

```sh
rg -n "(portable performance guarantee|state-of-the-art evidence|backend superiority evidence|hosted CI proof|package proof|ABI proof|runtime-loader proof|OpenMP speedup evidence|release proof|external library parity)" \
  README.md benchmarks/README.md docs/maintainer_guide.md \
  tests/corpus/schemas/report_index_fields.md \
  scripts/bench_canonical_report.sh scripts/performance_sentinels.sh
```

The hits are boundary statements, exclusions, or package/performance separation
notes. No hit turns local generated rows into portable performance,
state-of-the-art, hosted, package, ABI, runtime-loader, OpenMP speedup, external
library parity, or backend superiority proof.

## Diff Review

- `scripts/bench_canonical_report.sh` appends methodology fields while
  preserving existing leading columns and generated artifact locations.
- `scripts/performance_sentinels.sh` appends baseline, repeat, warmup,
  variance, and methodology notes while preserving S5 as the only hard local
  wall gate.
- `scripts/normalize_report_index.py` carries the added fields into normalized
  row configuration without changing the source row meaning.
- `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, and
  `tests/corpus/schemas/report_index_fields.md` align on local-only,
  threshold-free, and non-superiority wording.
- Generated outputs remain ignored under `build/`; no generated report or
  normalized index was committed.

## Sprint 164 API-Header Handoff

Sprint 163 did not change public C headers or exported API behavior. Sprint 164
should keep API-header and reference-publication work separate from performance
publication proof:

- audit public headers and generated API docs for any performance, backend,
  package, ABI, or platform language;
- if API docs mention benchmarks, cite only methodology-bound local report rows
  and preserve the local-only caveat;
- keep package/install confidence, shared-library ABI decisions, runtime-loader
  support, and package-manager distribution claims out of performance evidence;
- preserve the distinction between S5 hard local wall gate rows and S2/S3
  threshold-free backend-context rows;
- ensure API/header docs do not imply state-of-the-art, broad platform,
  external-library parity, OpenMP speedup, or backend superiority proof.

## Validation Basis

Day 13 relies on the Day 11 and Day 12 executed checks:

- shell syntax for selected report scripts passed;
- `make bench-canonical-report` passed;
- `make performance-sentinels` passed;
- `python3 tests/test_normalize_report_index.py` passed;
- selected benchmark/sentinel normalization wrote `26` rows;
- corpus schema validation passed;
- static package deferral guard passed;
- whitespace checks passed;
- no `.c` or `.h` files changed, so the full C quality gate was not required.

## Completion Check

- Positive performance-publication statements are traceable to selected local
  commands, generated rows, docs, and validation artifacts.
- Unsupported positive claims remain explicit non-claims.
- Generated outputs remain local and uncommitted.
- Sprint 164 API-header handoff is ready.
