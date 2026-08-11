# Sprint 152 Day 1 Generated Report Baseline

## Purpose

Day 1 establishes the generated report freshness baseline before Sprint 152
selects families for promotion. The key boundary is that generated-local rows
can aid local review, but they do not become broad release proof unless a later
policy explicitly promotes them with matching validation.

## Source Section

Sprint 152 is defined in `docs/planning/EPIC_13/PROJECT_PLAN.md` as
"Generated Report Freshness Publication." The user-provided Epic 12 line range
points to an older Sprint 142 section, so this sprint follows the Epic 13
project-plan section and writes to `docs/planning/EPIC_13/SPRINT_152/`.

## Prior Sprint Inputs

| Source | Relevant Sprint 152 Input |
| --- | --- |
| Sprint 141 report-index architecture | Source-controlled report-family contract, normalized row fields, freshness state vocabulary, missing-generated rows, and advisory/required generated behavior. |
| Sprint 150 QR corpus/report closeout | QR generated-local oracle rows, stale-output cleanup before current report writes, and advisory `generated_present_unchecked` freshness warnings. |
| Sprint 151 partial-SVD closeout | Four maintained partial-SVD fixtures, `26` generated-local partial-SVD oracle rows, combined oracle output of `29` generated rows, and explicit Sprint 152 freshness-policy residual. |

## Current Report-Family Contracts

| Family | Subfamily | Origin | Freshness Policy | Generator Command | Artifact Pattern | Day 1 Posture |
| --- | --- | --- | --- | --- | --- | --- |
| `corpus` | `fixtures` | source-controlled | `source_controlled` | `python3 scripts/validate_corpus_schema.py` | `tests/corpus/manifests/fixtures.tsv` | Baseline source-controlled metadata; not generated proof. |
| `corpus` | `generators` | source-controlled | `source_controlled` | `python3 scripts/validate_corpus_schema.py` | `tests/corpus/manifests/generators.tsv` | Baseline source-controlled generator metadata. |
| `corpus` | `optional_data` | source-controlled | `optional_data_skip` | `python3 scripts/validate_corpus_schema.py` | `tests/corpus/manifests/optional_data.tsv` | Skip/defer policy only; never pass evidence. |
| `corpus` | `expected` | source-controlled | `source_controlled` | `python3 scripts/validate_corpus_schema.py` | `tests/corpus/expected/*.tsv` | Expected rows define targets before observed evidence. |
| `oracle` | `generated_reference` | generated-local | `generated_compare_inputs` | `python3 scripts/run_corpus_oracle.py` | `build/corpus/oracle/*.tsv` | Candidate for freshness-policy work; currently generated-present unchecked. |
| `oracle` | `solver_backed` | generated-local | `generated_compare_inputs` | `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | `build/corpus/oracle/*.tsv` | High-value candidate because Sprints 150-151 use it for QR and partial-SVD local evidence. |
| `benchmark` | `canonical` | generated-local | `generated_local_advisory` | `make bench-canonical-report` | `build/bench-reports/canonical/index.tsv` | Candidate only if command/runtime stability is acceptable. |
| `sentinel` | `runtime` | generated-local | `generated_compare_inputs` | `make performance-sentinels` | `build/bench-reports/sentinels/sentinels.tsv` | Candidate with care; hard-gate semantics must remain local and bounded. |
| `sentinel` | `advisory` | generated-local | `generated_local_advisory` | `make performance-sentinels` | `build/bench-reports/sentinels/*.tsv` | Advisory candidate; not a portable performance claim. |
| `guardrail` | `large_matrix` | generated-local | `generated_compare_inputs` | `make large-matrix-guardrails` | `build/bench-reports/large-matrix-guardrails/index.tsv` | Candidate if row meanings and runtime cost are stable. |
| `deadcode` | `report` | generated-local | `generated_local_advisory` | `make deadcode-report` | `build/deadcode/report.tsv` | CI uploads artifacts, but rows remain triage evidence. |
| `coverage` | `src` | generated-local | `generated_local_advisory` | `make coverage` | `coverage/coverage-src.info` | CI uploads HTML artifacts; not behavioral completeness proof. |
| `package` | `static_install` | source-controlled | `source_controlled` | `bash tests/test_install.sh` | `tests/test_install.sh` | Proof-owner row only; install logs are workflow/local evidence. |
| `ci` | `reviewed_lanes` | source-controlled | `hosted_ci_external` | GitHub Actions | `.github/workflows/*.yml` | Lane-definition row only; hosted logs remain external. |
| `documentation` | `report_guidance` | documentation | `source_controlled` | manual review | `README.md;docs/maintainer_guide.md;benchmarks/README.md;INSTALL.md` | Interpretation anchor only. |
| `report_index` | `missing_generated` | generated-local | `generated_local_advisory` | `python3 scripts/normalize_report_index.py` | `build/report-index/normalized-index.tsv` | Makes missing local reports explicit without manufacturing pass evidence. |
| `runtime_backend` | `governance` | documentation | `source_controlled` | manual review | `docs/maintainer_guide.md;benchmarks/README.md` | Policy row only; sentinel measurements remain under sentinel families. |

## Producer Inventory

| Producer | Current Role | Generated Outputs |
| --- | --- | --- |
| `scripts/run_corpus_oracle.py` | Generates corpus oracle rows for generated-reference, QR solver-backed, and partial-SVD solver-backed families. | `build/corpus/oracle/*.tsv`, `build/corpus-reports/index.tsv`, `build/corpus-reports/skips.tsv`, `build/corpus-reports/manifest.txt` |
| `scripts/normalize_report_index.py` | Normalizes source-controlled and generated-local report families and evaluates freshness/advisory status. | `build/report-index/normalized-index.tsv` by default |
| `scripts/bench_canonical_report.sh` | Produces canonical benchmark report rows. | `build/bench-reports/canonical/index.tsv`, manifest files |
| `scripts/performance_sentinels.sh` | Produces bounded local performance sentinel rows. | `build/bench-reports/sentinels/*.tsv`, manifest files |
| `scripts/large_matrix_guardrails.sh` | Produces large-matrix guardrail rows. | `build/bench-reports/large-matrix-guardrails/index.tsv`, manifest files |
| `scripts/deadcode_workflow.sh` and `scripts/deadcode_report.py` | Produce dead-code workflow evidence and classified report rows. | `build/deadcode/report.tsv`, `build/deadcode/report.md`, supporting raw evidence |
| `make coverage` | Produces coverage reports via lcov or gcovr. | `coverage/coverage-src.info`, `coverage/html/` |

## Workflow Surface Inventory

| Workflow | Generated/Freshness-Relevant Surface | Day 1 Boundary |
| --- | --- | --- |
| `.github/workflows/ci.yml` | Linux package install proof, dead-code artifact upload, coverage artifact upload, supplemental runtime lanes. | Hosted logs/artifacts remain external evidence; source-controlled report freshness does not own them by default. |
| `.github/workflows/macos-ci.yml` | macOS reviewed static-first install and pkg-config proof. | Package/platform proof is workflow evidence, not local generated report freshness. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake/CTest/install validation lanes. | CMake-first platform proof remains external hosted evidence with explicit non-claims. |

## Current Freshness Semantics

| State Or Option | Current Meaning |
| --- | --- |
| `source_controlled` | Governed by Git state and schema validation. |
| `not_generated` | Selected generated family has no local generated artifact; advisory unless the family is required. |
| `generated_present_unchecked` | Generated row exists and is indexed, but strict comparison with command/source inputs is not yet promoted. |
| `deferred` | Governance acknowledged but intentionally not closed. |
| `optional_data_skip` | Optional data is absent, disabled, or deferred; it must not count as pass evidence. |
| `--require-generated <family>` | Missing generated rows for that report family fail in `--check` mode. |
| `--check-freshness` | Evaluates freshness status and prints advisory/warning/error output. |
| `--strict-generated` | Treats unchecked/stale generated rows more strictly when freshness checking is requested. |

## Claim Boundaries

- Generated-local oracle rows may support only the named fixture, command,
  commit, branch, platform, compiler, configuration, and support tier recorded
  in the row or manifest.
- Benchmark and sentinel rows are local measurement/regression aids and do not
  prove portable performance, throughput guarantees, or algorithmic
  superiority.
- Coverage rows are supplemental tool output and do not prove behavioral
  completeness or product quality.
- Dead-code rows are maintainer triage evidence and do not prove zero dead
  code or removal readiness.
- Package and CI rows identify proof owners or hosted lanes; they do not make
  local report-index freshness a package/platform guarantee.

## Stop Conditions

- Missing generated rows are counted as pass evidence.
- `generated_present_unchecked` rows are promoted without policy, owner, stable
  command, stable path, and validation.
- Optional-data skip/defer rows are cited as proof.
- Hosted CI logs are treated as source-controlled generated report artifacts.
- Generated `build/` or `coverage/` outputs are committed as release proof.
- Benchmark, sentinel, guardrail, dead-code, coverage, package, CI, or
  documentation rows are flattened into generic pass/fail evidence without
  preserving their row meaning and non-claims.
- A selected family becomes required before its failure message tells a
  maintainer how to regenerate or resolve the row.

## Day 2 Handoff

Day 2 should score generated families by claim value, determinism, runtime
cost, local/CI suitability, artifact stability, failure clarity, and claim
boundary risk. The highest-value candidates are likely the `oracle` families
from Sprints 150-151, followed by benchmark/sentinel/guardrail families if
their command/runtime stability is sufficient.
