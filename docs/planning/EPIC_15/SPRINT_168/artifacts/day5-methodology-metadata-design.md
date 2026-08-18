# Sprint 168 Day 5: Methodology Metadata Design

## Purpose

Day 5 defines the methodology metadata contract for the selected hosted
performance lane. The design starts from the current canonical report metadata
and the Day 4 dry-run gaps, then specifies which fields belong in the selected
CSV, `index.tsv`, `manifest.txt`, and hosted CI environment.

The selected lane remains threshold-free. Metadata supports interpretation and
freshness; it does not create a timing threshold, portable performance
guarantee, backend superiority claim, or state-of-the-art claim.

## Selected Lane

| Field | Selected value |
| --- | --- |
| Benchmark binary | `build/bench_refactor_csc` |
| Command | `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` |
| Generator | `make bench-canonical-report` or a focused selected-lane wrapper around the same command |
| Fixture | `tests/data/suitesparse/nos4.mtx` |
| Scenario | `chol_spd` |
| Repeat semantics | `configured_repeat_1` |
| Selected CSV | `build/bench-reports/canonical/bench_refactor_csc.csv` |
| Metadata artifacts | `build/bench-reports/canonical/index.tsv`, `build/bench-reports/canonical/manifest.txt` |
| Hosted lane assumption | Linux GitHub Actions lane selected on Day 9 |

## Current Fields Compared With Required Fields

| Required field | Current owner | Current value source | Day 5 decision |
| --- | --- | --- | --- |
| Compiler | `index.tsv`, `manifest.txt` | `${CC:-cc} --version | head -n 1` | Keep; hosted CI should set `CC` explicitly if the workflow needs a stable compiler identity. |
| Build flags | Missing | n/a | Add to metadata, preferably from `CFLAGS` or explicit `SPARSE_CANONICAL_BUILD_FLAGS`. |
| CPU | Missing | n/a | Add as best-effort metadata using a hosted/local override such as `SPARSE_CANONICAL_CPU_MODEL`; allow `unknown`. |
| OS/platform | `index.tsv`, `manifest.txt` | `uname -a` | Keep; add optional runner label/image fields for hosted interpretation. |
| Runner | Missing | n/a | Add `runner_context`, supplied by CI as a stable label such as `github-actions-ubuntu`. |
| Thread settings | `index.tsv`, `manifest.txt` | `OMP_NUM_THREADS` or `unset` | Keep; hosted CI should set or deliberately leave unset and record the result. |
| Backend/build mode | `index.tsv`, `manifest.txt` | `SPARSE_CANONICAL_BUILD_MODE` or binary detection | Keep; hosted CI should set `SPARSE_CANONICAL_BUILD_MODE=serial` unless OpenMP is intentionally selected. |
| Repeat count | `index.tsv` command/repeat fields and CSV command | `--repeat 1` / `configured_repeat_1` | Keep; selected freshness should require `configured_repeat_1`. |
| Warmup state | `index.tsv`, `manifest.txt` | `not_recorded` | Keep as explicit `not_recorded` for Sprint 168; Sprint 169 may harden warmup policy. |
| Variance state | `index.tsv`, `manifest.txt` | `not_recorded` | Keep as explicit `not_recorded` for Sprint 168; do not infer statistical confidence. |
| Timestamp | `index.tsv`, `manifest.txt` | UTC timestamp | Keep; freshness should check presence/format, not exact value. |
| Branch | `index.tsv`, `manifest.txt` | `git rev-parse --abbrev-ref HEAD` | Keep for context; hosted detached refs should be allowed. |
| Commit | `index.tsv`, `manifest.txt` | `git rev-parse --short HEAD` | Keep for traceability. |
| Command | `index.tsv`, `manifest.txt` | Fixed command string | Keep; selected freshness should require the exact selected command. |
| Fixture | `index.tsv`, CSV row | `nos4.mtx` | Keep; selected freshness should require `nos4.mtx`. |
| Threshold policy | `index.tsv`, `manifest.txt` | `baseline=n/a`, `threshold=n/a`, `claim_boundary=local_threshold_free` | Keep threshold-free policy; hosted claim boundary should say selected hosted threshold-free. |
| Claim boundary | `index.tsv`, `manifest.txt` | `local_threshold_free` | Add or override to a hosted selected value when CI owns the lane. |

## Metadata Field Specification

| Field | Artifact owner | Required for selected hosted lane | Deterministic formatting | Unknown-value behavior |
| --- | --- | --- | --- | --- |
| `surface` | `index.tsv` | yes | literal `canonical` or selected hosted surface value | never unknown |
| `category` | `index.tsv` | yes | literal `measurement` | never unknown |
| `report_label` | `index.tsv`, `manifest.txt` | yes | no tabs/newlines; CI label should be stable | default `unlabeled` allowed locally, but hosted lane should set a label |
| `generated_at_utc` | `index.tsv`, `manifest.txt` | yes | ISO-like UTC `YYYY-MM-DDTHH:MM:SSZ` | never blank |
| `git_commit` | `index.tsv`, `manifest.txt` | yes | short or full git hash | `unknown` only outside git |
| `git_branch` | `index.tsv`, `manifest.txt` | yes | git branch or `detached` | `unknown` only outside git |
| `platform` | `index.tsv`, `manifest.txt` | yes | single-line `uname -a` output | `unknown` if unavailable |
| `runner_context` | `index.tsv`, `manifest.txt` | yes for hosted | no tabs/newlines; stable CI label | `local` for local runs, `unknown` only if not supplied and not local |
| `compiler` | `index.tsv`, `manifest.txt` | yes | first line of compiler version | `unknown` if unavailable |
| `build_flags` | `index.tsv`, `manifest.txt` | yes for hosted | no tabs/newlines; exact flags or `default_make_flags` | `not_recorded` only for legacy/local rows |
| `cpu_model` | `index.tsv`, `manifest.txt` | yes for hosted | no tabs/newlines; CI override preferred | `unknown` allowed because hosted runners vary |
| `build_mode` | `index.tsv`, `manifest.txt` | yes | `serial`, `openmp`, or explicit override | never blank |
| `omp_num_threads` | `index.tsv`, `manifest.txt` | yes | integer string or `unset` | `unset` if not set |
| `artifact` | `index.tsv` | yes | `bench_refactor_csc` for selected row | never unknown |
| `relative_path` | `index.tsv` | yes | `bench_refactor_csc.csv` for selected row | never unknown |
| `command` | `index.tsv`, `manifest.txt` | yes | exact selected command without binary prefix in current script | never unknown |
| `report_family` | `index.tsv`, `manifest.txt` | yes | `benchmark` | never unknown |
| `status` | `index.tsv`, `manifest.txt` | yes | `measurement` | never unknown |
| `support_tier` | `index.tsv`, `manifest.txt` | yes | `local_only` locally; hosted selected lane should use `hosted_selected` or equivalent | never blank |
| `claim_boundary` | `index.tsv`, `manifest.txt` | yes | `local_threshold_free` locally; hosted lane should use `hosted_selected_threshold_free` or equivalent | never blank |
| `fixture_or_workload` | `index.tsv` | yes | `nos4.mtx` | never unknown for selected row |
| `matrix_size` | `index.tsv`, `manifest.txt` | preferred | Current script says `not_recorded`; selected row can derive `100x100` or keep `not_recorded` until Sprint 169 | `not_recorded` allowed in Sprint 168 |
| `repeat_semantics` | `index.tsv` | yes | `configured_repeat_1` | never unknown for selected row |
| `warmup` | `index.tsv`, `manifest.txt` | yes | `not_recorded` in Sprint 168 | keep explicit; do not infer warmup |
| `variance` | `index.tsv`, `manifest.txt` | yes | `not_recorded` in Sprint 168 | keep explicit; do not infer confidence |
| `baseline` | `index.tsv`, `manifest.txt` | yes | `n/a` | never unknown |
| `threshold` | `index.tsv`, `manifest.txt` | yes | `n/a` | never unknown |
| `backend_context` | `index.tsv`, `manifest.txt` | preferred | `n/a`, `serial`, or selected backend context | `n/a` allowed for selected SPD lane |
| `methodology_notes` | `index.tsv`, `manifest.txt` | yes | semicolon-separated no-control-character tokens | must include non-portable-performance note |

## Artifact Ownership Map

| Artifact | Owns | Must not own |
| --- | --- | --- |
| `bench_refactor_csc.csv` | Benchmark row identity, measured timing columns, speedup value, residuals, matrix/scenario dimensions. | Hosted support tier, runner context, claim boundary, publication status, or freshness policy. |
| `index.tsv` | Machine-readable artifact identity, command, fixture, support tier, claim boundary, environment, build metadata, repeat/warmup/variance policy, and methodology notes. | Timing pass/fail thresholds or portable comparisons. |
| `manifest.txt` | Human-readable bundle context, environment, selected command mapping, artifact inventory, and non-claim interpretation notes. | Machine-only freshness semantics that are absent from `index.tsv`. |
| CI workflow | Hosted runner, compiler/toolchain, build flags, report label, artifact upload, and hosted evidence classification. | Broad performance, backend, platform, or external-library claims. |
| `benchmarks/README.md` | Public interpretation of local versus hosted selected performance evidence. | Raw timing conclusions or superiority claims. |
| `README.md` | Short high-level pointer to selected hosted performance evidence after it exists. | Detailed benchmark methodology or broad performance positioning. |

## Local Versus Hosted Metadata

| Field group | Local canonical report | Hosted selected lane |
| --- | --- | --- |
| Support tier | `local_only` | `hosted_selected` or an equivalent selected-lane value. |
| Claim boundary | `local_threshold_free` | `hosted_selected_threshold_free` or an equivalent threshold-free hosted boundary. |
| Report label | Optional; defaults to `unlabeled`. | Required; should name Sprint 168 hosted performance lane. |
| Runner context | Not currently present; can be `local`. | Required; should identify GitHub Actions and runner image/label. |
| Build flags | Not currently present. | Required or explicitly `default_make_flags`. |
| CPU model | Not currently present. | Best effort; allow `unknown` because hosted CPU assignment can vary. |
| Warmup/variance | `not_recorded`. | Keep `not_recorded` for Sprint 168 unless policy changes; Sprint 169 may improve this. |
| Timing values | Generated CSV values. | Generated CSV values; not freshness-comparable and not thresholded. |
| Artifact status | Ignored local generated output. | Uploaded hosted artifact after CI wiring; not committed generated output. |

## Implementation Recommendation For Day 6

Prefer a small extension to `scripts/bench_canonical_report.sh` rather than a
new independent benchmark runner:

1. Keep the existing four-artifact canonical bundle behavior for local users.
2. Add optional metadata overrides for:
   - support tier;
   - claim boundary;
   - runner context;
   - build flags;
   - CPU model;
   - methodology notes.
3. Preserve default local values so existing local canonical reports remain
   compatible.
4. Use environment variables for hosted CI because the Make target already
   passes label/build-mode values this way.
5. Extend `index.tsv` and `manifest.txt` with new fields, while keeping
   generated CSV timing rows unchanged.

Candidate environment variables:

| Variable | Purpose | Local default |
| --- | --- | --- |
| `SPARSE_CANONICAL_SUPPORT_TIER` | Override `support_tier` for hosted selected lane. | `local_only` |
| `SPARSE_CANONICAL_CLAIM_BOUNDARY` | Override `claim_boundary`. | `local_threshold_free` |
| `SPARSE_CANONICAL_RUNNER_CONTEXT` | Record local or hosted runner context. | `local` |
| `SPARSE_CANONICAL_BUILD_FLAGS` | Record build flags or CI policy. | `not_recorded` |
| `SPARSE_CANONICAL_CPU_MODEL` | Record CPU model or hosted runner CPU note. | `unknown` |
| `SPARSE_CANONICAL_METHODOLOGY_NOTES` | Override or append methodology notes. | `threshold_free_local_measurement;not_portable_performance_claim` |

## Freshness Design Inputs For Day 7

The selected performance freshness check should require:

- `bench_refactor_csc.csv` exists;
- `index.tsv` exists;
- `manifest.txt` exists;
- `index.tsv` contains exactly one selected `bench_refactor_csc` row for
  `nos4.mtx` and `configured_repeat_1`;
- selected row has required non-empty metadata fields;
- selected row has threshold-free `baseline=n/a` and `threshold=n/a`;
- selected row has a support tier and claim boundary matching local or hosted
  invocation expectations;
- selected row methodology notes include `not_portable_performance_claim`;
- freshness does not compare raw timing or speedup values.

## Non-Claim Guard

The metadata design must preserve these non-claims:

- no portable performance guarantee;
- no backend superiority claim;
- no external-library performance parity;
- no state-of-the-art performance claim;
- no broad platform parity;
- no release benchmark proof;
- no timing threshold or regression gate for the selected hosted publication
  row.

## Validation Notes

Day 5 changed only Sprint 168 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every required methodology field has an owner. | Complete | Metadata field specification maps fields to CSV, `index.tsv`, `manifest.txt`, CI, or docs owners. |
| Missing metadata behavior is explicit. | Complete | Unknown-value behavior table defines `unknown`, `unset`, `not_recorded`, and `n/a` cases. |
| Report metadata does not imply performance superiority. | Complete | Threshold-free fields, non-claim guard, and artifact ownership map reject timing gates and superiority claims. |
