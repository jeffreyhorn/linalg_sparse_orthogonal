# Sprint 152 Day 5 Generator Stabilization Design

## Purpose

Day 5 turns the selected oracle freshness policy into concrete stabilization
requirements for commands, paths, metadata, stale-output handling, and failure
messages. This is a design artifact; implementation belongs to Day 6 and the
freshness-gate test work belongs to Days 7-8.

## Current Generator Behavior

`scripts/run_corpus_oracle.py` currently:

- accepts `--root`, `--oracle-dir`, `--report-dir`, `--include-solver-qr`,
  `--include-partial-svd`, and `--solver-library`;
- records the invoked command using `shlex.join(sys.argv)`;
- writes generated oracle TSV output to `build/corpus/oracle/`;
- writes report rows, skip rows, and a manifest to `build/corpus-reports/`;
- resets stale `*.tsv` oracle files and stale `index.tsv`, `skips.tsv`, and
  `manifest.txt` report files before writing the current run;
- writes manifest fields for generated timestamp, source commit, source
  branch, platform, compiler set, configuration set, oracle row count, solver
  families, solver QR row count, partial-SVD row count, command, fixture keys,
  support tier, claim boundary, and non-claims.

`scripts/normalize_report_index.py` currently:

- discovers generated oracle rows from `build/corpus/oracle/*.tsv`;
- normalizes row-level command, source commit, source branch, timestamp,
  platform, compiler, configuration, support tier, artifact path, claim scope,
  and non-claim fields;
- emits `not_generated` family rows when generated artifacts are absent;
- supports `--require-generated oracle` and `--check-freshness`.

## Canonical Command Design

Sprint 152 should stabilize around one canonical local command for the selected
oracle family:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
```

Accepted development variants may include:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/run_corpus_oracle.py --oracle-dir <tmp>/corpus/oracle --report-dir <tmp>/corpus-reports --include-solver-qr --include-partial-svd
```

Policy:

- The canonical command is the promoted closeout command for selected oracle
  freshness.
- Temporary output-directory variants are test fixtures and must not weaken
  production command diagnostics.
- QR-only and partial-SVD-only commands remain supported for focused local
  debugging but the combined command owns Sprint 152 selected row-count checks.
- Command comparison should allow path-normalized Python invocation only after
  Day 6/Day 8 code proves it; until then diagnostics should report the raw
  recorded command and the canonical command to rerun.

## Selected Output Path Design

| Output | Canonical Path | Role |
| --- | --- | --- |
| oracle TSV | `build/corpus/oracle/corpus.oracle.tsv` | Combined generated oracle rows for selected QR and partial-SVD families. |
| report index | `build/corpus-reports/index.tsv` | Human-readable generated report rows from the oracle command. |
| skip report | `build/corpus-reports/skips.tsv` | Optional-data skip/defer rows; never pass evidence. |
| manifest | `build/corpus-reports/manifest.txt` | Command, commit, branch, platform, compiler/configuration, row-count, fixture-key, support-tier, and non-claim metadata. |
| normalized index | `build/report-index/normalized-index.tsv` | Optional output from `scripts/normalize_report_index.py`; not committed. |

Path normalization requirements:

- normalized report rows should preserve repository-relative paths for default
  generated outputs;
- temporary test paths should remain path-specific in tests so stale or
  duplicate artifacts cannot collapse into the same row ID;
- generated outputs under `build/` remain ignored and must not be committed.

## Metadata Normalization Design

Selected oracle freshness checks should consume both row-level and
manifest-level metadata.

### Row-Level Required Fields

| Field | Stabilization Requirement |
| --- | --- |
| `oracle_row_id` | Stable native row ID; must be unique within the generated artifact. |
| `fixture_key` | Must be one of the selected QR or partial-SVD fixture keys for selected row-count policy. |
| `solver_family` | `unknown`, `qr`, or `partial_svd`; selected policy checks `qr` and `partial_svd` counts. |
| `operation` | Preserved for diagnostics; not a freshness input by itself. |
| `comparison_kind` | Preserved for diagnostics and row-meaning integrity. |
| `command` | Must record the generation command. |
| `source_commit` | Must equal current `HEAD` for strict/required freshness. |
| `source_branch` | Preserved for maintainer interpretation; branch mismatch should warn, not fail by default. |
| `generated_at_utc` | Must be present; timestamp age is not a failure criterion in Sprint 152. |
| `platform` | Must be present; platform mismatch is context, not failure, unless CI policy later selects it. |
| `compiler` | Must be present or `not_applicable`; preserved for row interpretation. |
| `configuration` | Must include solver family, fixture key, operation/comparison details, support details, hashes/tolerances where available. |
| `support_tier` | Must remain `local_only` for selected generated-local oracle rows. |
| `comparison_status` | Selected generated rows must not report unexpected `fail`. |
| `claim_scope` | Must remain fixture-local. |
| `non_claims` | Must preserve broad non-claims. |

### Manifest-Level Required Fields

| Field | Stabilization Requirement |
| --- | --- |
| `command` | Records the generating command; diagnostics should compare to canonical command. |
| `source_commit` | Must equal current `HEAD` for strict/required freshness. |
| `source_branch` | Preserved for context. |
| `platform` | Preserved for context. |
| `compiler` | Preserved for context. |
| `configuration` | Preserved for context and command-family review. |
| `oracle_row_count` | Combined command expected count is `52`: 3 generated-reference rows, 23 QR solver-backed rows, and 26 partial-SVD solver-backed rows. |
| `solver_families` | Combined command should include `partial_svd,qr,unknown`. |
| `solver_qr_row_count` | Combined command expected count is `23`. |
| `partial_svd_row_count` | Combined command expected count is `26`. |
| `fixture_keys` | Combined command should include the QR seed and selected Sprint 150 QR fixtures plus the four maintained partial-SVD fixtures. |
| `support_tier` | Must remain `local_only`. |
| `claim_boundary` | Must remain fixture-local corpus/oracle evidence only. |
| `non_claims` | Must reject broad QR, partial-SVD, external-library, performance, and state-of-the-art claims. |

## Selected Row-Count Policy

The combined canonical command should produce:

- total oracle rows: `52`;
- generated-reference rows: `3`;
- solver-backed QR rows: `23`;
- solver-backed partial-SVD rows: `26`;
- solver families: `partial_svd`, `qr`, and `unknown`.

The selected fixture-key set should include:

- `qr_rank_deficient_6x4_nullspace_v1`;
- `qr_rankdef_duplicate_5x4_v1`;
- `qr_rankdef_dependent_row_4x3_v1`;
- `qr_underdetermined_minnorm_2x4`;
- `qr_minnorm_3x6_exact_values`;
- `qr_minnorm_5x10_exact_values`;
- `partial_svd_clustered_repeated_diag8x6_k3_v1`;
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1`;
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`;
- `partial_svd_fail_closed_diag6_k2_v1`.

Focused QR-only and partial-SVD-only commands may keep their existing row-count
interpretation, but they should not satisfy the combined command's strict
row-count policy.

## Stale-Output Cleanup Design

Current cleanup behavior is correct for selected oracle outputs:

- remove stale `build/corpus/oracle/*.tsv`;
- remove stale `build/corpus-reports/index.tsv`;
- remove stale `build/corpus-reports/skips.tsv`;
- remove stale `build/corpus-reports/manifest.txt`;
- write the current oracle TSV, report index, skip report, and manifest.

Day 6 should preserve this behavior and add tests or documentation only where
needed. The cleanup policy should not remove unrelated build outputs outside
the selected oracle/report directories.

## Failure Message Design

Day 6/Day 8 diagnostics should make failures actionable.

| Failure | Message Should Name | Remediation |
| --- | --- | --- |
| Missing required oracle artifact | `oracle` family, expected artifact pattern, selected command | Run `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`. |
| Stale source commit | row ID, artifact path, recorded commit, current commit | Regenerate oracle output with the canonical command. |
| Unexpected row count | expected total/solver count, observed total/solver count, artifact path | Regenerate and inspect `build/corpus-reports/manifest.txt`. |
| Missing solver family | expected family, observed families, artifact path | Regenerate combined oracle command. |
| Missing fixture key | missing fixture key, manifest path, selected command | Regenerate combined oracle command and inspect selected fixture metadata. |
| Oracle comparison failure | oracle row ID, fixture key, comparison status, failure class | Run focused proof-owner test and regenerate oracle output. |
| Unchecked generated row under strict policy | row ID, freshness status, selected policy | Implement or rerun strict comparison path; do not downgrade silently. |

## Day 6 Implementation Checklist

- Preserve `reset_generated_outputs()` behavior.
- Add canonical selected oracle command constants or policy helpers in the
  normalizer if implementation needs them.
- Add row-count and fixture-key policy helpers for selected combined oracle
  output.
- Improve required-family and freshness diagnostics to name the regeneration
  command.
- Keep QR-only and partial-SVD-only commands supported but distinguish them
  from combined strict policy.
- Add focused tests only after Day 7/Day 8 gate-design work defines exact
  test cases.
- Do not commit generated `build/` or `coverage/` outputs.

## Non-Claims

This stabilization design does not promote generated reports to hosted CI,
release artifacts, package proof, ABI proof, platform proof, portable
performance proof, external-library parity, or state-of-the-art proof.
