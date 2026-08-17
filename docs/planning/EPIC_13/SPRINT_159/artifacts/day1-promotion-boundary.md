# Day 1 Promotion Boundary

## Scope

Day 1 establishes Sprint 159 intake, artifact structure, candidate freshness
commands, and hosted-promotion boundaries. The sprint target is narrow:
promote selected oracle and comparison generated freshness rows to reviewed
hosted evidence only when those rows have explicit command, runtime, artifact,
normalizer, support-tier, and claim boundaries.

## Source Inputs

| Input | Day 1 use |
| --- | --- |
| `docs/planning/EPIC_14/PROJECT_PLAN.md` Sprint 159 | Authoritative Sprint 159 scope and estimate. |
| `docs/planning/EPIC_14/SPRINT_157/artifacts/day9-evidence-contract-templates.md` | Hosted generated report evidence contract. |
| `docs/planning/EPIC_14/SPRINT_158/artifacts/day14-closeout-handoff.md` | Confirms generated API HTML remains local-only and hands hosted reports to Sprint 159. |
| `Makefile` | Owns current oracle and comparison freshness targets. |
| `scripts/run_corpus_oracle.py` | Owns local oracle row generation. |
| `scripts/run_external_comparison.py` | Owns local comparison row generation. |
| `scripts/normalize_report_index.py` | Owns report-index normalization and freshness diagnostics. |
| `tests/corpus/manifests/report_families.tsv` | Owns report-family row meanings, support tiers, artifact patterns, claim scopes, and non-claims. |
| `.github/workflows/*.yml` | Future hosted execution and artifact-publication owner. |
| `docs/maintainer_guide.md`, `tests/corpus/README.md`, `README.md`, `docs/solver_selection.md` | Documentation owners for support-tier and claim wording. |

## Prompt Path Resolution

The user prompt referenced
`docs/planning/EPIC_12/PROJECT_PLAN.md` lines 96-130, but that range contains
Sprint 139. The matching Sprint 159 section is in
`docs/planning/EPIC_14/PROJECT_PLAN.md` lines 96-130. Sprint 159 artifacts are
kept in the requested path:

```text
docs/planning/EPIC_13/SPRINT_159/
```

## Branch Baseline

| Field | Value |
| --- | --- |
| Branch | `sprint-159` |
| Starting commit | `b53810ba514b030a0cbe6153cd92e9760a51b5b3` |
| Starting commit summary | `b53810ba Merge pull request #176 from jeffreyhorn/sprint-158` |
| Upstream state | Created from current `master` after PR #176 merge. |

## Candidate Freshness Commands

| Candidate | Command | Current output | Current interpretation |
| --- | --- | --- | --- |
| Selected oracle freshness gate | `make report-index-oracle-freshness` | `build/corpus/oracle/corpus.oracle.tsv`, `build/corpus-reports/index.tsv`, `build/corpus-reports/skips.tsv`, `build/corpus-reports/manifest.txt` | Local-only generated freshness for selected QR and partial-SVD oracle rows. |
| Oracle generator | `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | ignored `build/corpus/` and `build/corpus-reports/` outputs | Generator command; not pass evidence without required freshness normalization. |
| Oracle normalizer check | `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | normalized diagnostics under ignored `build/report-index/` when output is requested | Strict local selected oracle freshness check. |
| Selected comparison freshness gate | `make report-index-comparison-freshness` | `build/comparison/qr_minnorm/project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` | Local-only generated freshness for one QR minimum-norm comparison family. |
| Comparison generator | `python3 scripts/run_external_comparison.py --target qr-minnorm` | ignored `build/comparison/qr_minnorm/` outputs | Generator command; not broad external-library parity. |
| Comparison normalizer check | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | normalized diagnostics under ignored `build/report-index/` when output is requested | Strict local selected comparison freshness check. |
| Cross-family report-index check | `python3 scripts/normalize_report_index.py --check` and `--check-freshness` | normalized report-index diagnostics | Useful for Day 9/10 semantics, but too broad for automatic hosted promotion without family selection. |

## Candidate Report Families

| Report family | Current row owner | Current support tier | Day 1 disposition |
| --- | --- | --- | --- |
| `oracle/generated_reference` | Corpus maintainer | local-only generated | Candidate only if selected row counts, runtime, and artifacts remain bounded. |
| `oracle/solver_backed` | Solver owner | local-only generated | Candidate for QR/partial-SVD fixture-local hosted evidence; not broad solver correctness. |
| `comparison/qr_minnorm` | Report maintainer | local-only generated | Candidate for hosted promotion because it is a single fixture-local QR comparison lane. |
| `corpus/*` source metadata | Corpus maintainer | advisory/local-only source-controlled metadata | Not pass evidence; remains context for selected row meaning. |
| `benchmark`, `sentinel`, `guardrail`, `coverage`, `deadcode`, `package`, `runtime_backend`, `documentation`, `ci` | Family-specific owners | advisory, supplemental, reviewed, or local-only depending on row | Not selected by Day 1. Keep out of Sprint 159 hosted promotion unless a later day explicitly reclassifies with evidence. |

## Hosted Promotion Boundary

For a row family to become reviewed hosted evidence, Sprint 159 must produce:

1. a selected family list tied to a claim surface;
2. a command list that can run in hosted CI;
3. runtime measurements and hosted timeout expectations;
4. artifact names, paths, retention, and deterministic summaries;
5. normalizer semantics for stale, missing, skipped, failing, and valid rows;
6. docs and metadata wording that names exactly what is reviewed hosted
   evidence;
7. preserved non-claims for everything outside the selected rows.

Until all seven conditions are met, generated outputs stay local-only or
advisory.

## Non-Goals And Guardrails

| Non-goal | Guardrail |
| --- | --- |
| Broad QR or partial-SVD correctness | Only named fixture families can be promoted. |
| Broad external-library parity | Comparison rows stay fixture-local and dependency-aware. |
| Platform or package proof | Report freshness cannot prove Windows parity, install behavior, static packages, shared libraries, or ABI compatibility. |
| Performance proof | Oracle/comparison freshness outputs are correctness/report freshness evidence, not timing superiority. |
| Generated API HTML publication | Sprint 158 kept API HTML local-only; Sprint 159 does not change that policy. |
| Advisory-family promotion | Benchmark, sentinel, coverage, dead-code, large-matrix, package, runtime-backend, and documentation rows remain out of scope unless explicitly selected later. |
| Source-controlled metadata as pass evidence | `report_families.tsv` defines row meaning; it does not prove freshness. |

## Day 2 Handoff

Day 2 should classify candidate families into:

- reviewed-hosted candidate;
- supplemental-hosted;
- advisory-local;
- deferred.

The classification should start from these likely candidates:

| Candidate | Initial Day 2 question |
| --- | --- |
| `oracle/generated_reference` | Are all generated-reference rows claim-bearing enough for hosted promotion, or are they support context for solver-backed rows? |
| `oracle/solver_backed` QR rows | Which QR oracle rows map to current public/maintainer claims and can fit hosted runtime? |
| `oracle/solver_backed` partial-SVD rows | Which partial-SVD oracle rows map to current public/maintainer claims and can fit hosted runtime? |
| `comparison/qr_minnorm` | Should the single QR minimum-norm comparison become reviewed hosted evidence, or stay local-only until Sprint 160 expands QR comparison work? |
| Broad report-index `--check-freshness` | Should broad report-index freshness remain local/advisory while selected `oracle` and `comparison` rows are promoted? |

## Completion Check

- Sprint 159 scope is tied to the authoritative Epic 14 project-plan section.
- Artifact directory and working notes are created.
- Candidate oracle, comparison, report-index, QR, and partial-SVD freshness
  commands are inventoried.
- Hosted-promotion boundaries and non-goals are explicit.
- Day 2 has concrete classification inputs.

