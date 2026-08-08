# Day 10 Freshness Gate Design

## Purpose

Day 10 defines the stale-report gate that will validate normalized report
index rows without converting local measurements into release proof. The gate
must be mechanical and reproducible, but strict only where freshness can be
asserted honestly from source-controlled inputs and generated row metadata.

This is a design artifact. Day 11 owns implementation.

## Freshness Inputs

| Input | Source | Used for |
| --- | --- | --- |
| Normalized row fields | `scripts/normalize_report_index.py` output | Primary gate input. |
| `source_commit` | Generated row or current Git revision | Detect generated rows from a different commit. |
| `source_branch` | Generated row or current Git branch | Informational mismatch context; not a hard gate by itself. |
| `generator_command` | Contract or generated row command | Detect missing required command options or wrong producer. |
| `generated_at_utc` | Generated row timestamp | Human freshness context; age-only failures are deferred unless a later owner sets a budget. |
| `platform` | Generated row platform | Detect platform-scoped generated rows; avoid treating local rows as hosted parity. |
| `compiler` | Generated row compiler | Preserve build context; warn on mismatch where family is advisory. |
| `configuration` | Generated row config text | Compare support tier, backend, optional-data, build mode, and hash context when present. |
| `artifact_path` | Normalized artifact path | Detect missing or generated-local artifact references. |
| `freshness_status` | Normalized row freshness state | Drives the state/severity mapping. |
| Contract freshness policy | `tests/corpus/manifests/report_families.tsv` | Decides whether a missing/stale row is error, warning, advisory, skip, or defer. |
| Optional-data state | `tests/corpus/manifests/optional_data.tsv` and normalized rows | Keeps unavailable optional data as skip/defer evidence. |
| Generator and expected-row hashes | Corpus generator metadata and oracle configuration when present | Strong freshness input for corpus/oracle rows. |

## Freshness States

| State | Meaning | Typical source |
| --- | --- | --- |
| `source_controlled` | Row is governed by committed metadata and schema validation. | Corpus manifests, package proof-owner rows, CI lane definitions, documentation advisories. |
| `generated_present_unchecked` | A generated artifact or native generated row exists, but stale comparison has not yet been applied. | Oracle, benchmark, sentinel, guardrail, coverage, and dead-code rows. |
| `fresh` | Generated row matches current required inputs for a strict family. | Day 11 computed result for selected rows. |
| `stale` | Generated row exists but does not match required current inputs. | Day 11 computed result for selected rows. |
| `not_generated` | Expected local generated report is absent. | Missing generated families. |
| `optional_data_skip` | Optional data is disabled, unavailable, or intentionally skipped. | Optional-data rows. |
| `deferred` | Row is acknowledged but policy or implementation belongs to another sprint. | Runtime/backend governance. |
| `unsupported` | Platform, backend, or package lane is explicitly out of scope. | Future generated rows or source-controlled platform metadata. |

## Severity Model

| Severity | Exit behavior | Use |
| --- | --- | --- |
| `error` | Nonzero in `--check` mode. | Strict generated rows that are required and stale/missing; malformed normalized rows; missing source-controlled proof-owner paths. |
| `warning` | Zero by default, visible diagnostic. | Generated rows from advisory families that are stale or missing but selected for review. |
| `advisory` | Zero with row-level context only. | Local measurement, coverage, dead-code, and documentation rows unless explicitly required. |
| `skip` | Zero with skip reason. | Optional-data unavailable or intentionally disabled rows. |
| `defer` | Zero with handoff reason. | Runtime/backend policy rows and other planned follow-up. |
| `unsupported` | Zero unless a caller requires the unsupported family. | Explicitly unsupported platform/package/backend rows. |

## Family Behavior Matrix

| Family | Policy | Missing behavior | Stale behavior | Default severity |
| --- | --- | --- | --- | --- |
| Corpus metadata | `source_controlled` | Error if source-controlled manifest is missing. | Covered by schema validation and Git diff review. | Error for malformed metadata. |
| Expected-result rows | `source_controlled` | Warning if no expected files exist; error for malformed selected rows. | Covered by schema validation and Git diff review. | Error for malformed metadata. |
| Oracle generated-reference rows | `generated_compare_inputs` | Warning by default; error when `--require-generated oracle` is set. | Error only when required; otherwise warning. | Warning. |
| Oracle solver-backed rows | `generated_compare_inputs` | Warning by default; error when `--require-generated oracle` is set. | Error only when required; otherwise warning. | Warning. |
| Canonical benchmark rows | `generated_local_advisory` | Advisory `not_generated`. | Advisory; never release proof. | Advisory. |
| Sentinel hard-gate rows | `generated_compare_inputs` | Warning by default; error only if caller requires `sentinel`. | Error only when required or when a generated hard-gate row reports `fail`. | Warning/error depending on selected mode. |
| Sentinel advisory rows | `generated_local_advisory` | Advisory `not_generated`. | Advisory. | Advisory. |
| Large-matrix guardrails | `generated_compare_inputs` | Warning by default; error only if caller requires `guardrail`. | Error only when required or when reviewed generated row reports `fail`. | Warning/error depending on selected mode. |
| Coverage rows | `generated_local_advisory` | Advisory by default; error when `--require-generated coverage` is set. | Advisory. | Advisory. |
| Dead-code rows | `generated_local_advisory` | Advisory by default; error when `--require-generated deadcode` is set. | Advisory. | Advisory. |
| Package/install proof-owner rows | `source_controlled` | Error if required source-controlled owner file is missing. | Source-control/schema freshness only. | Error for missing owner paths. |
| CI lane definitions | `hosted_ci_external` | Warning if workflow definitions are missing. | Hosted logs are external; no local stale claim. | Warning. |
| Documentation advisories | `source_controlled` | Warning if expected docs are missing. | Source-control freshness only. | Warning. |
| Report-index missing rows | `generated_local_advisory` | Advisory unless required by caller. | Not applicable. | Advisory. |
| Runtime/backend governance | `deferred_governance` | Defer. | Defer. | Defer. |

## Strictness Rules

1. A generated row may be `fresh` only if the gate can compare all required
   inputs for that family.
2. `source_commit` mismatch is stale for strict generated families and a
   warning/advisory for local measurement families.
3. Missing generated reports are never pass evidence.
4. Generated local benchmark, coverage, and dead-code rows cannot become
   release proof through freshness alone.
5. Optional-data skips remain skip evidence unless the optional data is
   explicitly configured and a generated row exists.
6. Runtime/backend governance remains deferred to Sprint 142.
7. Package proof-owner rows are source-controlled owner metadata; install-run
   result logs are not synthesized.

## CLI Design

Day 11 should add freshness validation to `scripts/normalize_report_index.py`
or a sibling helper with this shape:

```sh
python3 scripts/normalize_report_index.py --check-freshness
python3 scripts/normalize_report_index.py --check-freshness --family oracle
python3 scripts/normalize_report_index.py --check-freshness --require-generated oracle
python3 scripts/normalize_report_index.py --check-freshness --strict-generated
python3 scripts/normalize_report_index.py --check-freshness --advisory-ok
```

Proposed behavior:

| Option | Meaning |
| --- | --- |
| `--check-freshness` | Emit diagnostics and exit according to the freshness severity model. |
| `--require-generated <family>` | Already present; missing rows for the family become errors. Day 11 should reuse it. |
| `--strict-generated` | Treat stale or missing `generated_compare_inputs` rows as errors. |
| `--advisory-ok` | Keep advisory family diagnostics zero-exit even when selected with broad filters. |
| `--family <name>` | Restrict both normalized output and freshness diagnostics. |

The default `--check` should continue to validate output construction. The
freshness gate should be explicit until Day 13 decides whether to add a Make
target.

## Diagnostic Format

Use deterministic one-line diagnostics:

```text
freshness: <severity>: <row_id>: <state>: <reason>
```

Examples:

```text
freshness: warning: report_missing_oracle_generated_reference_observed_oracle_comparison_v1: not_generated: local generated oracle report is absent
freshness: error: oracle_qr_rank_deficient_6x4_nullspace_v1_build_corpus_oracle_corpus_oracle_tsv_v1: stale: source_commit does not match current HEAD
freshness: advisory: benchmark_bench_refactor_csc_build_bench_reports_canonical_index_tsv_v1: generated_present_unchecked: local measurement freshness is advisory
freshness: defer: report_contract_runtime_backend_governance_deferred_governance_v1: deferred: runtime/backend governance belongs to Sprint 142
```

## Test Plan

| Test | Expected behavior |
| --- | --- |
| Source-controlled package proof-owner rows | `source_controlled` rows pass if paths exist; missing source owner path is an error in a temp fixture. |
| Missing oracle without `--require-generated` | Warning diagnostic; zero exit. |
| Missing oracle with `--require-generated oracle` | Error diagnostic; nonzero exit. |
| Missing coverage without requirement | Advisory diagnostic; zero exit. |
| Missing coverage with `--require-generated coverage` | Error diagnostic; nonzero exit. |
| Generated oracle row with mismatched `source_commit` | Stale warning by default; error under `--strict-generated` or required oracle. |
| Generated benchmark row with mismatched `source_commit` | Advisory stale diagnostic; zero exit. |
| Sentinel hard-gate generated row with `status=fail` | Error if selected for freshness check, because the row itself is a failed hard gate. |
| Optional-data skip row | Skip diagnostic; zero exit. |
| Runtime/backend deferred row | Defer diagnostic; zero exit. |
| Unsupported row | Unsupported diagnostic; zero exit unless required by caller. |
| Duplicate row IDs | Existing output validation error before freshness evaluation. |

## CI And Local Behavior

| Context | Recommended command | Expected strictness |
| --- | --- | --- |
| Local inventory | `python3 scripts/normalize_report_index.py --check` | Validate row construction only. |
| Local freshness review | `python3 scripts/normalize_report_index.py --check-freshness` | Warn/advisory by default; no broad generated requirements. |
| Corpus/oracle owner review | `python3 scripts/normalize_report_index.py --check-freshness --family oracle --require-generated oracle` | Missing/stale oracle rows fail. |
| Runtime report review | `python3 scripts/normalize_report_index.py --check-freshness --family benchmark --family sentinel --family guardrail` | Hard-gate failures error; advisory measurement rows warn/advisory. |
| Quality/package review | `python3 scripts/normalize_report_index.py --check-freshness --family coverage --family deadcode --family package` | Package proof-owner paths strict; coverage/dead-code advisory unless required. |
| Future CI target | Candidate `make report-index-check` after Day 11 implementation and Day 13 validation. | Start non-invasive; strict generated families only when explicitly required. |

## Non-Claim Wording

Freshness diagnostics should use this framing:

- Fresh local benchmark rows are local measurement context, not portable
  performance proof.
- Fresh coverage rows are local tool output, not coverage completeness or
  product quality proof.
- Fresh dead-code rows are local classification output, not a zero-dead-code
  guarantee.
- Fresh package proof-owner rows prove source-controlled ownership only, not a
  generated install-run result unless an install validation command was
  actually run and indexed.
- Fresh oracle rows are fixture-local to the named command, commit, platform,
  compiler, configuration, support tier, and artifact.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Stale detection is mechanical and reproducible. | Complete | Inputs, states, comparison rules, diagnostics, and test cases are specified. |
| Advisory measurements are never treated as broad release proof. | Complete | Benchmark, coverage, dead-code, and sentinel advisory rows stay warning/advisory by default. |
| Gate behavior is strict only where report freshness can be asserted honestly. | Complete | Strictness is limited to malformed source-controlled rows, required generated families, hard-gate failures, and explicit strict-generated mode. |
