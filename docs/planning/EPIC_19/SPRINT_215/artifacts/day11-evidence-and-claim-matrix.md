# Sprint 215 Day 11: Evidence And Claim Matrix

## Item 215.4 Outcome Audit

**Disposition: blocked before execution**, not executed and not a failed
numerical comparison. The selected cell is one real-float64 SPD factorization
and solve on the SHA-256-pinned `tests/data/symmetric_4.mtx` fixture, with
`x_ref=[1,1,1,1]` and `b=[13,14,17,16]`. The project side is no-reorder
linked-list Cholesky; the external side is SuiteSparse `v7.12.2` (commit
`42151688813c45846a597edcb601435a0e38f3dd`), CHOLMOD 5.3.4 with
natural ordering and simplicial factorization. This is not the selected
canonical `bench_refactor_csc` / `nos4.mtx --repeat 1` row.

The [Day 3 selection](day3-baseline-selection-decision.md) and
[methodology inputs](day3-methodology-inputs.md) freeze that identity.
The [Day 4 numerical contract](day4-correctness-and-tolerance-contract.md)
requires matching input, successful phases, finite outputs, independent
residual/oracle limits, and cross-solver consistency before a correctness
pass. The [Day 5 methodology](day5-external-comparison-methodology.md)
specifies a separate optional timing protocol; the pilot remains
correctness-only. The [Day 8 decision](day8-pilot-feasibility-decision.md)
authorized a gated attempt, not an observed result.

The [Day 10 outcome](day10-selected-pilot-outcome.md) records that the v1
schema and semantic validator are missing. The fail-closed wrapper exited 2
with `schema_validation_unavailable` before invoking either solver. Its local
`attempt.json` has `provisional_status=blocked`, `sides={}`,
`timing=not_run`, and no project/baseline command or accepted `evidence.json`.
The fixture hash, discovered CHOLMOD package/version and candidate dylib hash
matched preflight expectations, but actual loaded binary/dependencies,
numerical behavior and timing were not observed. The checked-in
[Day 7 JSON](day7-external-evidence-blueprint.example.json) is a planning
example, not the generated attempt or a validated observation. Item 215.4
therefore closes for this sprint as a **reproducible named blocker with an
opt-in executable fallback**, while execution and publication remain open.

## Claim Matrix

| Claim scope | Status on Day 11 | Approved wording now | Evidence required to change status |
| --- | --- | --- | --- |
| Blueprint-only | Earned as a design claim | "Sprint 215 specifies a version-pinned, one-cell external CHOLMOD correctness comparison and separate validation/timing requirements." | None for describing the design; never call the blueprint an observed result. |
| Selected local pilot | Blocked, not observed | "The selected macOS x86-64 CHOLMOD pilot preflight stopped before either solver ran because the v1 validator/schema was unavailable." | Implement and pass v1 validation; capture full cell/provenance and raw logs; run both sides; satisfy Day 4 independent numerical gates. |
| Cross-matrix | Unearned | "Only `symmetric_4.mtx` was selected; no external cross-matrix comparison was run." | Freeze additional attributed/hash-pinned fixtures, input conversions and per-cell numerical gates; execute and validate each. |
| Cross-solver | Unearned | "CHOLMOD is the sole selected external baseline; SciPy/SuperLU and other families are deferred." | Version/acquisition and dependency pins, operation-matched fixtures, solver-specific gates, and independent validated runs for each family. |
| Hosted | Unearned for external baseline | "Existing hosted selected freshness concerns internal benchmark and source-controlled-helper rows, not this CHOLMOD cell." | Reviewed hosted workflow, installed baseline/runtime identity, retained raw evidence and validated external results for the same cell. |
| Cross-platform | Unearned | "The selected pilot was local macOS x86-64 preparation only; no external result or Linux/macOS comparison exists." | Separate complete validated runs with matched operation/input and explicit platform/toolchain/dependency provenance; timing comparability needs more than two hosted runs. |
| State-of-the-art | Unearned | "No state-of-the-art ranking or parity conclusion follows from the blueprint or blocked preflight." | Representative matrix and solver coverage, fair matched methods/hardware, correctness gates, predeclared timing/statistics, reproducible artifacts, and publication review. |

Even a future single-cell `correctness_observed` record would support only
that cell's numerical result. It would not retroactively earn the last five
rows or a portable-performance statement. A separately validated local
timing record would remain descriptive for its recorded host and operation,
not an algorithmic or cross-platform speed ranking.

## Wording Firewall

Approved: "The selected canonical benchmark checks internal artifact and
methodology freshness without a timing threshold." Approved: "Sprint 215
selected a CHOLMOD comparison design, but the local pilot was blocked before
solver execution." Approved: "Current selected comparison rows use
source-controlled dense reference helpers, not a versioned third-party
CHOLMOD result."

Reject: "The library matches CHOLMOD," "CHOLMOD parity is proven," "the
pilot confirms a speedup," "Linux and macOS timings agree," "the selected
benchmark establishes external performance," and "state-of-the-art sparse
solvers are matched or exceeded." None follows from a blueprint, package
discovery, compiled helper, blocked attempt or internal freshness check.

## Surface Audit And Remaining Gates

| Surface | Day 11 reading or action |
| --- | --- |
| [Benchmark guide](../../../../../benchmarks/README.md#report-index-handoff) | Updated: distinguishes internal selected freshness, source-controlled-helper comparisons and the blocked external cell. |
| [Maintainer guide](../../../../../docs/maintainer_guide.md#benchmark-governance-ownership) | Updated: names the exact external cell, blocker, validator/numerical/publication gates, and separate report ownership. |
| [README](../../../../../README.md) | Existing selected benchmark and report-index caveats must not be read as CHOLMOD evidence. Day 12 should add a short route to the external claim boundary if public wording is expanded. |
| [Solver selection](../../../../../docs/solver_selection.md) | The Cholesky row's selected comparison is a source-controlled dense helper on `cholesky_spd_tridiag_5`, not the Sprint 215 `symmetric_4.mtx`/CHOLMOD cell. Do not replace its family-selection guidance with an external parity assertion. |
| [Report index](../../../../../scripts/normalize_report_index.py) and [selected manifest](../../../../../tests/corpus/manifests/selected_report_targets.tsv) | No Sprint 215 external row or target was added. Keep blocked `attempt.json` out of normalized/selected generated artifacts. |

Before a retry, implement the [Day 7 validator design](day7-schema-validation-design.md)
and its positive/negative fixtures, verify complete cell-ID encoding and
actual loaded CHOLMOD/transitive-library, build, ABI, module, ordering and
thread provenance, then retain both raw solver outputs and apply the frozen
Day 4 gates. Optional timing additionally needs the Day 5 paired sampling
and host comparability checks. Publication separately needs fixture
attribution and CHOLMOD module-license review. A changed matrix, package,
binary, backend, precision or operation gets a new cell identity, not a
reinterpretation of this blocked attempt.
