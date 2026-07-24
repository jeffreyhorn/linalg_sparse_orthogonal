# Sprint 131 Day 7 - Report Index Design

## Purpose

Day 7 selects the first report/index artifact candidate and defines its
source inputs, output location, schema, sorting, stable row identity,
regeneration command, stale-output behavior, missing-input behavior, and
implementation checklist.

This is a design artifact. It does not change `scripts/large_matrix_guardrails.sh`,
benchmark semantics, Makefile targets, CI policy, generated outputs, or public
claims.

## Selected First Index Candidate

| Field | Decision |
| --- | --- |
| Candidate | Large-matrix guardrail report index. |
| Existing source | `make large-matrix-guardrails` through `scripts/large_matrix_guardrails.sh`. |
| Existing index output | `build/bench-reports/large-matrix-guardrails/index.tsv`. |
| Existing manifest output | `build/bench-reports/large-matrix-guardrails/manifest.txt`. |
| Reason selected | The existing index already has stable lane IDs, reviewed/supplemental categories, explicit skip rows, deterministic source commands, artifact names, and notes. |
| Day 7 posture | Use as first generated/index artifact design candidate; implementation or explicit deferral remains Day 10 work. |

## Source-To-Output Design

| Input | Current owner | Current role | Required design interpretation |
| --- | --- | --- | --- |
| `build/test_reorder_amd_qg` | `tests/test_reorder_amd_qg.c`, Makefile target | Reviewed structural qg-AMD guardrail lane. | Index row `G1`; failure is guardrail failure, not benchmark timing failure. |
| `build/test_reorder_nd` | `tests/test_reorder_nd.c`, Makefile target | Reviewed ND generated-family and named-matrix structural lane. | Index row `G2`; explicit skips inside artifact remain visible in test output. |
| `build/test_graph` | `tests/test_graph.c`, Makefile target | Reviewed graph partition/separator structural lane. | Index row `G3`; graph structural evidence, not numerical solve evidence. |
| `build/bench_reorder --sprint86-slice --skip-factor` | `benchmarks/bench_reorder.c` | Reviewed bounded CSV-shape and fill-report lane for `bcsstk14` and `Pres_Poisson`. | Index row `G4`; CSV shape/fill rows only, not portable timing. |
| `build/bench_reorder --skip-factor` | `benchmarks/bench_reorder.c` | Supplemental threshold-free full named-matrix reorder report. | Index row `S1`; `skip` unless `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1`. |
| `build/bench_amd_qg --skip-bitset` | `benchmarks/bench_amd_qg.c` | Supplemental qg-AMD/generated-banded report. | Index row `S2`; `skip` unless `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1`. |

## Output Location

| Output | Path | Role |
| --- | --- | --- |
| Primary index | `build/bench-reports/large-matrix-guardrails/index.tsv` | Machine-readable lane index. |
| Manifest | `build/bench-reports/large-matrix-guardrails/manifest.txt` | Human-readable run context and artifact inventory. |
| Reviewed logs | `test_graph.txt`, `test_reorder_nd.txt`, `test_reorder_amd_qg.txt` | Raw reviewed lane output. |
| Reviewed CSV artifact | `bench_reorder_sprint86.csv` | Bounded CSV-shape/fill report for `bcsstk14` and `Pres_Poisson`. |
| Supplemental CSV artifacts | `bench_reorder_all.csv`, `bench_amd_qg_skip_bitset.csv` | Opt-in threshold-free supplemental reports. |

## Current Schema And Proposed Normalized Schema

### Current `index.tsv`

| Field | Meaning |
| --- | --- |
| `lane_id` | Stable row identity: `G1`-`G4`, `S1`, `S2`. |
| `status` | `pass`, `report`, or `skip`. |
| `category` | `reviewed` or `supplemental`. |
| `command` | Binary command run or skipped. |
| `artifact` | Output artifact basename or `n/a`. |
| `notes` | Human-readable lane interpretation. |

### Normalized Future Schema

| Field | Source | Required for Day 10 implementation? |
| --- | --- | --- |
| `report_key` | Constant `large-matrix-guardrails`. | Yes if schema is widened. |
| `lane_id` | Current index field. | Yes. |
| `row_type` | Derived from lane: `artifact` for `G1`-`G4`, `skip` for skipped `S1`/`S2`, `artifact` for supplemental report rows. | Yes if schema is widened. |
| `status` | Current index field. | Yes. |
| `support_tier` | Current `category`, normalized to support-tier vocabulary. | Yes. |
| `source_command` | Current `command`. | Yes. |
| `artifact_path` | Current `artifact`, relative to report directory. | Yes. |
| `input_corpus` | Derived from lane notes/commands: generated graph families, `bcsstk14`, `Pres_Poisson`, or full named-matrix slice. | Optional for first implementation; required for cross-report corpus index. |
| `output_class` | Derived: `test-log` or `csv`. | Optional for first implementation; recommended. |
| `failure_meaning` | Derived from lane status/category. | Optional for first implementation; required for claim-gate index. |
| `claim_boundary` | Derived from lane notes and manifest notes. | Optional for first implementation; recommended. |

## Sorting And Stable Row Identity

Rows should remain sorted by lane family and lane number:

1. `G1`
2. `G2`
3. `G3`
4. `G4`
5. `S1`
6. `S2`

The lane IDs are stable row identities. A future lane must not reuse an
existing lane ID with different semantics. A removed lane should remain as a
`deferred` or `unsupported` row for one transition artifact if report
consumers need a migration path.

## Regeneration And Freshness Policy

| Policy item | Decision |
| --- | --- |
| Regeneration command | `make large-matrix-guardrails`. |
| Supplemental regeneration command | `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails`. |
| Freshness anchors | `generated_at_utc`, `git_commit`, `git_branch`, `platform`, `compiler`, and `supplemental` in `manifest.txt`. |
| Freshness check | Index and manifest must be produced by the same run and live in the same report directory. |
| Stale output | Stale or missing output means no current large-matrix guardrail report evidence; it is not a solver correctness failure by itself. |
| Reviewed failure | A failing `G1`-`G4` lane is a guardrail failure. |
| Supplemental skip | `S1`/`S2` skip is expected unless supplemental mode is enabled. |

## Missing, Optional, And Unsupported Behavior

| Condition | Required behavior |
| --- | --- |
| Required binary missing | Script exits with usage/setup failure before writing a valid reviewed report. |
| Reviewed test lane fails | Script exits nonzero and report generation is incomplete; treat as guardrail failure. |
| Reviewed CSV shape validation fails | Script exits nonzero; treat as report contract failure, not timing failure. |
| Supplemental mode disabled | `S1` and `S2` appear as `skip` rows with explicit opt-in note. |
| Supplemental mode enabled | `S1` and `S2` appear as `report` rows and artifacts are listed in manifest. |
| Supplemental report fails | Script exits nonzero; future index should preserve whether failure happened in supplemental mode. |
| Unsupported future lane | Add `status=deferred` or document deferral before implementation; do not silently omit. |

## Implementation Checklist

If Day 10 implements a schema change instead of deferring, likely touched files
are:

| File | Change type | Validation |
| --- | --- | --- |
| `scripts/large_matrix_guardrails.sh` | Add normalized fields or companion normalized index. | `make large-matrix-guardrails`; supplemental mode if supplemental rows change. |
| `benchmarks/README.md` | Document new schema if user-facing benchmark docs need it. | `git diff --check` and markdown whitespace scan. |
| `docs/maintainer_guide.md` | Document maintainer interpretation only if schema semantics change. | Evidence-to-claim and non-claim scan plus docs hygiene. |
| Sprint 131 artifact | Record implementation or deferral decision. | Docs hygiene. |

If no script changes are made, Day 10 should publish an explicit deferral with
the blocker: current `index.tsv` is adequate for guardrail lanes, while
cross-report normalized schema should wait until coverage and dead-code
architecture are complete.

## Day 7 Design Decision

Day 7 selects the large-matrix guardrail index as the first index candidate,
but does not implement a schema change yet. The design recommends Day 10
either:

- add a companion normalized index only after Day 8 coverage and Day 9
  dead-code architecture confirm common fields; or
- explicitly defer implementation and accept the existing guardrail
  `index.tsv` as the first generated report/index artifact for Sprint 131.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| First index candidate has a clear implementation path or blocker. | Complete | Large-matrix guardrail index selected; implementation checklist and deferral blocker are defined. |
| Source inputs and output paths are deterministic. | Complete | Source-to-output table and output-location table name commands, owners, and artifact paths. |
| Missing or optional inputs have explicit behavior. | Complete | Missing, optional, unsupported, stale, reviewed failure, and supplemental skip behavior are defined. |

## Day 8 Handoff

Day 8 should design coverage architecture before any cross-report normalized
index is implemented. Coverage reports are tree-mutating and have different
freshness/failure semantics from large-matrix guardrails, so their reviewed
versus supplemental split must be clear before Day 10 chooses implementation
or deferral.

