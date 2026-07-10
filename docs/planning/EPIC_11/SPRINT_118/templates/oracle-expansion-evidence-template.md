# <Sprint/Day> Oracle Expansion Evidence

## Scope

| Field | Value |
|---|---|
| Sprint/day | `<Sprint N Day N>` |
| Artifact owner | `<owner>` |
| Solver or behavior family | `<direct / iterative / eigensolver / SVD / QR / rank / corpus>` |
| Touched surfaces | `<files, fixtures, docs, tests>` |
| Explicitly out of scope | `<surfaces not touched>` |

## Baseline

| Baseline item | Current value |
|---|---|
| Existing proof owner | `<test/artifact>` |
| Existing oracle/reference | `<none / dense / external / cross-solver>` |
| Current product truth references | `<Sprint 118 Day 8 or other evidence>` |
| Current non-claims | `<non-claims preserved before work>` |

## Proof Values

| Proof value | Protected behavior | Evidence before change |
|---|---|---|
| `<value>` | `<behavior>` | `<test/artifact/doc>` |

## Fixture Taxonomy

| Fixture | Symmetry | Definiteness | Rank | Conditioning/scaling | Sparsity pattern | Expected behavior |
|---|---|---|---|---|---|---|
| `<fixture>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<pass/fail/skip>` |

## Matrix, RHS, Or Eigenpair Construction

- Matrix source:
- Dimensions:
- Nonzeros or sparsity:
- RHS construction:
- Eigenpair or singular-vector construction:
- Ordering/reorder/backend/runtime settings:

## Oracle Or Reference Source

| Oracle/reference | Invocation | Trust boundary | Skip/error handling |
|---|---|---|---|
| `<oracle>` | `<command/API>` | `<boundary>` | `<behavior>` |

## Tolerance And Acceptance Model

| Metric | Tolerance | Rationale |
|---|---:|---|
| Residual | `<tol>` | `<reason>` |
| Reconstruction | `<tol>` | `<reason>` |
| Orthogonality | `<tol>` | `<reason>` |
| Convergence | `<tol/limit>` | `<reason>` |

## Correctness Metrics

| Metric | Expected value | Observed value | Status |
|---|---:|---:|---|
| `<metric>` | `<expected>` | `<observed>` | `<status>` |

## Convergence Metrics

Use for iterative and eigensolver lanes.

| Metric | Expected value | Observed value | Status |
|---|---:|---:|---|
| Iterations | `<expected>` | `<observed>` | `<status>` |
| Converged flag | `<expected>` | `<observed>` | `<status>` |

## Unsupported Or Expected-Failure Cases

| Case | Disposition | Reason |
|---|---|---|
| `<case>` | `<unsupported / expected failure / skipped>` | `<reason>` |

## Validation Commands

| Command | Required because | Reviewed/supplemental/local | Result |
|---|---|---|---|
| `<command>` | `<reason>` | `<classification>` | `<pending/pass/fail>` |

Required trigger check:

- If any `.c` or `.h` file changed, run `make format && make lint && make test`.
- Keep correctness, convergence, and timing evidence separate.
- Record external-reference skip/error/success states distinctly.

## Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | `<none / update / fence>` | `<action>` |
| Solver-selection docs | `<none / update / fence>` | `<action>` |
| Examples/tutorial | `<none / update / fence>` | `<action>` |
| Benchmark/performance wording | `<none / update / fence>` | `<action>` |

## Non-Claims Preserved

- `<non-claim>`
- `<non-claim>`

## Residual Handoff

| Residual | Next owner | Evidence link |
|---|---|---|
| `<residual>` | `<sprint/day/future epic>` | `<artifact>` |

## Completion Check

| Criterion | Status |
|---|---|
| Fixture taxonomy is recorded. | `<status>` |
| Oracle or reference trust boundary is recorded. | `<status>` |
| Tolerances are explicit. | `<status>` |
| Unsupported cases are explicit. | `<status>` |
| Validation commands are recorded. | `<status>` |
| Drift and non-claims are recorded. | `<status>` |
| Residual handoff is recorded. | `<status>` |
