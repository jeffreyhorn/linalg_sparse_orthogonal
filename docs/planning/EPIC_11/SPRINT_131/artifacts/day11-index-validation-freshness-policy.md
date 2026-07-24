# Sprint 131 Day 11 - Index Validation and Freshness Policy

## Purpose

Day 11 validates the first accepted index path and defines recurring freshness,
drift, missing-input, optional-data, and failure behavior without requiring CI
changes or stronger release guarantees.

This is a documentation-only policy artifact. It does not change
`scripts/large_matrix_guardrails.sh`, Makefile targets, benchmarks, coverage,
dead-code workflow, tests, CI, generated report schemas, or source code.

## First Index Path

| Field | Decision |
| --- | --- |
| Report family | `large-matrix-guardrails` |
| Generation command | `make large-matrix-guardrails` |
| Supplemental command | `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails` |
| Primary index | `build/bench-reports/large-matrix-guardrails/index.tsv` |
| Companion manifest | `build/bench-reports/large-matrix-guardrails/manifest.txt` |
| Support tier | Reviewed structural guardrail rows plus supplemental opt-in report rows |
| Claim boundary | Structural test and bounded CSV-shape guardrail evidence only |

## Validation Results

Day 10 ran `make large-matrix-guardrails` successfully. Day 11 inspected the
resulting index, manifest, row count, artifacts, and checkout freshness anchors.

| Check | Result | Evidence |
| --- | --- | --- |
| Generation command available | Pass | `make large-matrix-guardrails` completed successfully on Day 10. |
| Index exists | Pass | `build/bench-reports/large-matrix-guardrails/index.tsv` exists. |
| Manifest exists | Pass | `build/bench-reports/large-matrix-guardrails/manifest.txt` exists. |
| Index schema | Pass | Header has six tab-separated fields: `lane_id`, `status`, `category`, `command`, `artifact`, `notes`. |
| Row count | Pass | Index has six data rows plus one header row. |
| Reviewed rows | Pass | `G1`, `G2`, `G3`, and `G4` are `pass` and `reviewed`. |
| Supplemental rows | Pass | `S1` and `S2` are `skip` and `supplemental` in default mode. |
| Artifact list | Pass | Reviewed artifacts present: `test_graph.txt`, `test_reorder_nd.txt`, `test_reorder_amd_qg.txt`, and `bench_reorder_sprint86.csv`. |
| Manifest branch | Pass | Manifest branch is `sprint-131`, matching the current checkout. |
| Manifest commit | Pass | Manifest commit is `2e3125a2`, matching the current checkout. |
| Supplemental mode | Pass | Manifest records `supplemental=0`, matching default reviewed mode. |

## Validation Command Log

Executed validation commands:

```bash
make large-matrix-guardrails
awk -F '\t' 'NR==1 {print "header_fields=" NF; print $0; next} {print $1 ":" $2 ":" $3 ":" $5}' build/bench-reports/large-matrix-guardrails/index.tsv
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
find build/bench-reports/large-matrix-guardrails -maxdepth 1 -type f -print | sort
```

Documentation hygiene commands for this artifact:

```bash
git diff --check
if rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_131; then exit 1; fi
```

No `.c` or `.h` files were modified, so the full `make format && make lint &&
make test` quality gate is not required by Sprint 131 validation policy.

## Freshness Policy

Freshness means "the artifact can be traced to the current report command and
checkout context." It does not mean CI coverage, release readiness, broad
large-matrix support, portable performance, memory bounds, or numerical
correctness beyond the guardrail lanes.

| Freshness source | Required behavior |
| --- | --- |
| `generated_at_utc` | Record historical generation time. A timestamp alone is not a pass/fail signal unless a caller defines a freshness window. |
| `git_commit` | Compare to `git rev-parse --short HEAD` when a report is used as current evidence. Mismatch means stale relative to checkout. |
| `git_branch` | Compare to `git rev-parse --abbrev-ref HEAD` for local traceability. Branch mismatch means stale or copied evidence. |
| `platform` | Display as run context. Platform mismatch does not invalidate historical evidence but prevents portable claims. |
| `compiler` | Display as run context. Compiler mismatch prevents compiler-portability claims. |
| `supplemental` | `0` means default reviewed mode; `1` means supplemental lanes were attempted. |
| `index.tsv` and `manifest.txt` location | Must live in the same report directory and be produced by the same generation command. |

Freshness labels:

| Label | Meaning | Allowed use |
| --- | --- | --- |
| `current` | Manifest commit and branch match checkout; expected artifacts exist. | Current large-matrix guardrail evidence for the bounded lanes. |
| `historical` | Artifact exists but freshness window, branch, or commit is not current. | Planning or trend context only. |
| `stale` | Commit or branch mismatch relative to current checkout. | Must regenerate before using as current evidence. |
| `missing` | Index, manifest, or required artifact is absent. | No current evidence; show regeneration command. |
| `invalid` | Schema, row count, lane IDs, or manifest contract is malformed. | Report contract failure; owner triage required. |

## Drift Detection Responsibilities

| Drift type | Detection owner | Required action |
| --- | --- | --- |
| Lane ID drift | `large-matrix-guardrails` owner | Preserve `G1`-`G4`, `S1`, and `S2` semantics or record a migration row/deferral. |
| Schema drift | `report-index-owner` | Update report-index documentation and validation policy before consumers depend on new fields. |
| Command drift | Makefile and script owners | Keep `index.tsv` command fields aligned with actual binaries and flags. |
| Artifact drift | `large-matrix-guardrails` owner | Ensure every non-`n/a` artifact listed in `index.tsv` exists in the report directory. |
| Manifest drift | `report-index-owner` | Preserve freshness anchors or document replacement fields. |
| Claim drift | Maintainer-guide owner | Prevent structural guardrail rows from becoming scalability, timing, memory, coverage, or corpus parity claims. |
| Curated artifact drift | Sprint planning owner | Treat planning artifacts as traceability docs, not generated current evidence. |

## Missing-Input and Optional-Data Behavior

| Condition | Expected behavior | Validation posture |
| --- | --- | --- |
| Required test or benchmark binary missing | Script exits before producing a valid reviewed report. | Setup failure; rebuild with `make large-matrix-guardrails`. |
| Required reviewed lane fails | Script exits nonzero and report generation is incomplete. | Guardrail failure; triage owning test or benchmark lane. |
| Reviewed CSV header changes | CSV validator exits nonzero. | Report contract failure; update schema docs only with explicit owner decision. |
| Reviewed CSV row count or fixture set changes | CSV validator exits nonzero. | Guardrail failure or intentional schema change requiring documentation. |
| Supplemental mode disabled | `S1` and `S2` appear as `skip` rows with opt-in notes. | Expected default state. |
| Supplemental mode enabled | `S1` and `S2` appear as `report` rows and artifacts are listed in the manifest. | Supplemental evidence only; not a reviewed timing or scalability gate. |
| Future lane unsupported | Add explicit `deferred`, `unsupported`, or documented deferral before relying on it. | Do not silently omit a previously documented lane. |
| Optional corpus missing inside supplemental report | Supplemental failure or skip depending on owner design. | Must remain supplemental unless promoted by a future gate. |

## Generated and Curated Row Policy

| Row type | Source | Freshness requirement | Drift requirement |
| --- | --- | --- | --- |
| Generated guardrail index row | `index.tsv` | Current manifest and matching artifact directory. | Stable lane ID and stable category semantics. |
| Generated guardrail manifest row | `manifest.txt` | Current commit/branch when used as current evidence. | Freshness fields remain readable. |
| Generated dead-code report row | Future use of `report.tsv` | Current report command and `deadcode-check` pass. | Bucket semantics must remain classified before indexing. |
| Generated coverage report row | Future coverage index | Current coverage command, backend, threshold, source filter, and tree-mutating status. | Must keep supplemental coverage boundary from Day 8. |
| Curated planning row | Sprint artifact | Path exists and artifact explicitly names claim boundary. | Manual owner review on sprint closeout. |

## Residual Validation Queue

| Residual | Blocker | Dependency | Future owner |
| --- | --- | --- | --- |
| Automated stale-report scanner | No common report metadata contract across coverage, dead-code, benchmark, and guardrail outputs. | Day 12 ownership map and future normalized schema decision. | `report-index-owner` |
| Supplemental-mode validation | Default Day 10 run did not opt into supplemental lanes to avoid expanding report cost. | Runtime/support-tier decision before recurring supplemental validation. | `large-matrix-guardrails` |
| Missing-input destructive validation | Artificially removing binaries would mutate build state and duplicate script coverage. | Future script-level tests if report tooling gains its own test harness. | `large-matrix-guardrails` |
| Cross-report freshness windows | No release policy currently defines maximum artifact age. | Maintainer-guide decision if freshness windows become release criteria. | Maintainer-guide owner. |
| Dead-code report freshness integration | `report.tsv` lacks manifest-style branch/commit fields. | Future dead-code index design. | `deadcode-workflow` |
| Coverage report freshness integration | Coverage reports are tree-mutating and backend-specific. | Future coverage index design using Day 8 fields. | `coverage-workflow` |

## Day 12 Handoff

Day 12 should convert these validation owners into the broader Sprint 131 owner
map:

- `large-matrix-guardrails` owns lane IDs, artifact presence, supplemental
  mode, and guardrail command drift;
- `report-index-owner` owns schema, freshness metadata, stale-report policy,
  and cross-report normalization decisions;
- `coverage-workflow` owns coverage report freshness only when a coverage
  index exists;
- `deadcode-workflow` owns classified report freshness only when dead-code
  rows are indexed beyond `report.tsv`;
- maintainer-guide ownership controls claim drift.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implemented or deferred index path has validated behavior. | Complete | Day 10 regenerated the index; Day 11 inspected schema, row count, lane status/category, artifacts, and manifest freshness anchors. |
| Freshness does not imply stronger CI or release guarantees than supported. | Complete | Freshness policy limits `current`, `historical`, `stale`, `missing`, and `invalid` labels to report evidence scope. |
| Validation commands and gaps are reproducible. | Complete | Command log and residual validation queue record checked commands, dry-run policy, blockers, dependencies, and owners. |
