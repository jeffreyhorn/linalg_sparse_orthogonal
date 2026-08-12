# Sprint 152 Day 3 Generated Family Selection

## Purpose

Day 3 selects the generated report families Sprint 152 will close and records
the policy class, claim scope, non-claims, implementation map, and rollback
rules before freshness policy design begins.

## Selection Summary

Sprint 152 will close generated freshness publication for the corpus oracle
families and the supporting missing-generated report-index behavior.

| Family | Subfamily | Day 3 Policy Class | Selected | Reason |
| --- | --- | --- | --- | --- |
| `oracle` | `solver_backed` | Required locally after generation; strict freshness candidate | Yes | Directly supports Sprint 150 QR and Sprint 151 partial-SVD maintained corpus evidence. |
| `oracle` | `generated_reference` | Required locally after generation; strict freshness candidate | Yes | Shares the same corpus oracle command/path surface and records generated-reference expected-row comparisons. |
| `report_index` | `missing_generated` | Supporting required-family failure surface | Yes | Ensures missing generated reports are explicit and cannot be silently omitted. |
| `corpus` | `fixtures`/`generators`/`expected` | Source-controlled prerequisite | Yes, prerequisite only | Defines eligible fixtures and expected rows before generated evidence exists. |
| `benchmark` | `canonical` | Advisory/deferred | No | Runtime cost and local-machine variance make required freshness too risky for Sprint 152. |
| `sentinel` | `runtime` | Deferred required; advisory only if touched | No | Hard-gate semantics need a narrower runtime policy before promotion. |
| `sentinel` | `advisory` | Advisory | No | Row meaning is advisory and should not become required. |
| `guardrail` | `large_matrix` | Deferred | No | Runtime/platform cost and broad scalability overclaim risk are higher than the oracle freshness residual. |
| `deadcode` | `report` | Advisory | No | Triage artifact, not release or removal-ready proof. |
| `coverage` | `src` | Advisory | No | Tree-mutating supplemental coverage output, not behavioral completeness proof. |
| `package` | `static_install` | Source-controlled proof owner | No generated freshness work | Package proof remains command/workflow evidence, not local generated report freshness. |
| `ci` | `reviewed_lanes` | Hosted external lane definition | No generated freshness work | Hosted logs remain external evidence. |
| `documentation` | `report_guidance` | Source-controlled guidance | Update only if wording changes | Documentation explains evidence boundaries but is not executable proof. |
| `runtime_backend` | `governance` | Source-controlled policy row | No generated freshness work | Generated runtime measurements belong under sentinel families. |

## Selected Claim Scope

### Oracle Solver-Backed Rows

Selected scope:

- QR generated-local solver-backed rows from
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`;
- partial-SVD generated-local solver-backed rows from
  `python3 scripts/run_corpus_oracle.py --include-partial-svd`;
- combined local oracle output from
  `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`;
- command, source commit, source branch, generated timestamp, platform,
  compiler, configuration, support tier, artifact path, fixture key, solver
  family, comparison status, claim scope, and non-claim fields.

Selected freshness claim:

The normalized report index can require the oracle family locally and can fail
when selected oracle generated rows are missing, stale relative to current
`HEAD`, or inconsistent with the selected command/path/row-count policy.

### Oracle Generated-Reference Rows

Selected scope:

- generated-reference rows emitted by the corpus oracle command for maintained
  expected rows;
- the same metadata, artifact, and freshness fields used by solver-backed
  oracle rows.

Selected freshness claim:

Generated-reference rows should not silently disappear when the oracle family
is selected and required.

### Report-Index Missing-Generated Rows

Selected scope:

- missing generated rows emitted by `scripts/normalize_report_index.py` for
  selected report families;
- required-family errors that name the missing family and regeneration path.

Selected freshness claim:

Missing selected generated report families should be visible and actionable,
not silently absent.

## Non-Claims

The selected Sprint 152 generated freshness work does not claim:

- broad QR correctness;
- broad partial-SVD correctness;
- raw singular-vector, QR basis, sign, orientation, phase, or arbitrary basis
  ordering parity;
- external-library parity;
- hosted CI proof for generated-local oracle rows;
- package-manager availability;
- shared-library ABI support;
- broad platform support;
- portable performance;
- benchmark superiority;
- coverage completeness;
- zero dead code;
- state-of-the-art status.

## Implementation Map

| Day | Implementation Owner |
| --- | --- |
| Day 4 | Define selected oracle and missing-generated freshness policy semantics: missing, present, fresh, stale, strict, required, advisory, and deferred. |
| Day 5 | Design command/path/metadata stabilization for selected oracle outputs and missing-generated diagnostics. |
| Day 6 | Implement selected command/path/metadata/failure-message stabilization if needed. |
| Day 7 | Design report-index tests for required oracle rows, strict freshness, stale oracle metadata, and missing generated oracle reports. |
| Day 8 | Implement freshness checks and tests for selected oracle and missing-generated policy. |
| Day 9 | Decide that selected oracle freshness remains local-only unless a later day explicitly promotes a hosted CI lane. |
| Day 10 | Implement any selected CI/local command alignment while preserving local-only generated report boundaries. |
| Day 11 | Update corpus, report schema, and maintainer guidance for selected oracle freshness policy. |
| Day 12 | Regenerate oracle reports and validate normalization/freshness behavior together. |
| Day 13 | Run required quality gates and record residual generated families. |
| Day 14 | Close with Sprint 153 ABI/package handoff. |

## Rollback Rules

Rollback or defer selected generated freshness promotion if any of these occur:

- oracle generation is nondeterministic for selected row counts or row IDs;
- stale generated files can contaminate current report-index output;
- required-family behavior cannot name the regeneration command;
- strict freshness produces false failures for current local regenerated rows;
- selected oracle rows require generated `build/` artifacts to be committed;
- hosted CI behavior becomes necessary before local policy is stable;
- docs or report rows imply platform, package, ABI, performance, or
  state-of-the-art claims;
- optional skip/defer rows become pass or freshness evidence;
- validation commands fail and cannot be resolved within Sprint 152.

## Day 4 Handoff

Day 4 should define the exact freshness state model for selected oracle rows:

- when missing oracle output is advisory versus required failure;
- when `generated_present_unchecked` becomes fresh, warning, or error;
- how strict freshness treats matching and mismatched `source_commit`;
- how selected row-count and command/path metadata should be represented;
- how diagnostics should direct maintainers to regenerate oracle reports.
