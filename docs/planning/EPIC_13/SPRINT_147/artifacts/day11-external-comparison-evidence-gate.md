# Sprint 147 Day 11 External Comparison Evidence Gate

## Purpose

Day 11 defines the evidence gate for Sprint 154's first narrow external
comparison study. The comparison must be based on a maintained QR or
partial-SVD corpus family from Sprints 150-151, must name external dependencies
and versions, and must preserve non-claims for broad ecosystem parity,
portable performance, and state-of-the-art status.

This gate is a planning contract. It does not create comparison evidence by
itself.

## Candidate Comparison Targets

| Candidate | Prerequisites | Strength | Risk | Default Decision |
| --- | --- | --- | --- | --- |
| QR maintained corpus family | Sprint 150 fixture rows, proof-owner tests, oracle/report rows, and QR comparison semantics. | Strong if rank-deficient rectangular, minimum-norm, or reorder/COLAMD families are fully maintained. | Raw QR basis orientation, rank-threshold policy, and minimum-norm wording can overclaim if not fenced. | Preferred if Sprint 150 lands at least two closed fixture families. |
| Partial-SVD maintained corpus family | Sprint 151 fixture rows, proof-owner tests, oracle/report rows, and subspace-safe comparison semantics. | Strong if repeated/clustered spectra and rank-deficient rectangular cases are fully maintained. | Raw singular-vector identity and broad repeated-spectrum claims are unsafe. | Preferred if Sprint 151 lands the richer maintained family set. |
| Existing direct-solver dense-reference lanes | Existing Cholesky/LDLT/LU bounded external helper patterns. | Mature pattern for invoking external dense references and handling skips. | Does not address Epic 13's selected QR/partial-SVD gap unless reused only as pattern. | Pattern input, not the default Sprint 154 target. |
| Benchmark or sentinel rows | Sprint 152 generated-report policy. | Useful local measurement context. | Easy to confuse with portable performance or state-of-the-art proof. | Not selected for the first comparison claim. |

Sprint 154 should choose exactly one primary target unless both QR and
partial-SVD have already landed complete maintained-family evidence and the
second target can be closed without weakening validation.

## Dependency And Optional-Data Policy

External comparison requires:

- named external library or tool;
- exact version and installation method;
- license/terms note if the dependency or data is redistributed or cached;
- platform support statement;
- skip/defer behavior when the dependency is unavailable;
- fixture set from source-controlled corpus metadata;
- no optional-data pass claim unless optional-data availability and terms are
  reviewed;
- no generated report freshness claim unless Day 9/Sprint 152 freshness policy
  selects the comparison family.

Suggested dependency policy:

| Field | Requirement |
| --- | --- |
| Dependency name | Human-readable package/library/tool name. |
| Dependency version | Exact version string captured at runtime. |
| Installation method | Package manager, virtual environment, system path, source build, or explicit local path. |
| Invocation | Exact command, script, or API used by the harness. |
| Availability status | `available`, `skip`, `defer`, or `unsupported`. |
| Skip/defer reason | Required when unavailable, optional data is disabled, or platform support is absent. |
| License/terms | Required for optional external data or redistributed fixtures. |

Unavailable dependencies produce skip/defer evidence only. They do not create
pass evidence or parity claims.

## Comparison Row Schema

Sprint 154 should add a generated comparison report family only if report
integration is part of the selected implementation. A comparison row should
include:

| Field | Meaning |
| --- | --- |
| `comparison_row_id` | Stable row ID for one fixture, external library, metric, and configuration. |
| `comparison_family` | `qr`, `partial_svd`, or another explicitly selected family. |
| `fixture_key` | Source-controlled corpus fixture key. |
| `project_command` | Exact command/API path used for this project. |
| `external_library` | External dependency name. |
| `external_version` | Exact external version string. |
| `external_command` | Exact external command/API path. |
| `metric` | Residual, rank, nullity, singular value, subspace distance, status, or diagnostic. |
| `expected_or_reference` | External result, deterministic expected value, or bounded reference condition. |
| `observed_project` | Project result. |
| `tolerance_kind` | Exact, absolute, relative, mixed, projector, status-only, or not applicable. |
| `tolerance_value` | Numeric or structured tolerance. |
| `comparison_status` | `pass`, `fail`, `skip`, `defer`, `unsupported`, or `xfail`. |
| `failure_class` | Required for non-pass rows. |
| `platform` | OS, architecture, runner, and hardware context if relevant. |
| `compiler` | Project compiler and version. |
| `configuration` | Build flags, solver options, optional-data state, and dependency path. |
| `source_commit` | Project commit SHA. |
| `generated_at_utc` | Timestamp. |
| `claim_scope` | One-sentence narrow claim the row may support if passing. |
| `non_claims` | Explicit boundaries the row must preserve. |

If this schema is implemented as normalized report-index rows, it must also
include or map to the Day 9 freshness fields.

## Metric Rules

| Target | Allowed Metrics | Required Boundary |
| --- | --- | --- |
| QR | Rank, nullity, normalized residual, least-squares residual, solution norm for selected minimum-norm fixtures, status, and ordering diagnostics if selected. | No raw Q-basis equality, no global rank-threshold policy, no broad minimum-norm or SuiteSparse parity claim. |
| Partial-SVD | Singular values, projector/subspace distance, triplet residuals, orthogonality residuals, convergence status, fail-closed diagnostics. | No raw singular-vector identity, no broad repeated-spectrum or convergence-rate claim. |
| Performance | Local wall time only if explicitly selected as advisory context. | No portable performance, throughput, memory, or algorithmic superiority claim. |
| Platform | Hosted platform status only when run in CI and recorded. | No cross-platform parity from one local run. |

## Narrow Claim Wording Rules

Acceptable wording patterns:

- "For the named `<fixture_family>` fixtures, the project result matched
  `<external_library> <version>` on `<metric>` within `<tolerance>` under
  `<platform/compiler/configuration>`."
- "The first external comparison study covers `<fixture_keys>` and supports
  only the recorded metrics and tolerances."
- "Skipped optional-data rows indicate unavailable comparison inputs and are
  not pass evidence."
- "The comparison is local/generated unless the artifact records hosted run
  metadata."

Required qualifiers:

- named external library and version;
- fixture keys or fixture family;
- metric and tolerance;
- platform, compiler, and configuration;
- support tier;
- caveat that the result is not broad ecosystem parity.

## Rejected Wording

Do not use:

- "state-of-the-art sparse linear algebra library";
- "matches LAPACK/SciPy/SuiteSparse/PETSc/Trilinos/ARPACK";
- "external-library parity" without a named, versioned, bounded fixture set;
- "portable performance" or "faster than";
- "drop-in replacement";
- "broad QR correctness" or "broad partial-SVD correctness";
- "raw basis parity" or "raw singular-vector parity";
- "cross-platform comparison proof" from a single local run;
- "hosted comparison proof" without recorded hosted logs and artifacts.

## Validation Requirements

Minimum local checks for Sprint 154:

```sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --no-generated --check
```

If the comparison target is QR:

```sh
make build/test_qr_corpus && ./build/test_qr_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

If the comparison target is partial-SVD:

```sh
make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus
python3 scripts/run_corpus_oracle.py --include-partial-svd
```

If generated comparison rows are integrated:

```sh
python3 scripts/normalize_report_index.py --check-freshness
```

The exact comparison harness command must be recorded by Sprint 154 after the
target and dependency are selected.

Any `.c` or `.h` changes require:

```sh
make format && make lint && make test
```

## Sprint 154 Handoff Requirements

Sprint 154 must begin by filling these fields:

| Field | Required Value |
| --- | --- |
| Primary target | QR or partial-SVD. |
| Fixture keys | Source-controlled maintained fixtures from Sprint 150 or 151. |
| External dependency | Library/tool name, version, installation method, and invocation. |
| Metrics | Exact metrics and tolerances. |
| Harness owner | Script/test/report owner files. |
| Skip/defer policy | Dependency unavailable, optional-data unavailable, platform unsupported, or inconclusive result. |
| Report integration | Whether generated comparison rows are indexed and whether they are required-generated. |
| Claim wording | One narrow statement plus explicit non-claims. |
| Validation package | Local commands, C quality gate if applicable, and hosted run metadata if used. |

## Stop Conditions

- The comparison target is not backed by maintained source-controlled fixtures.
- External dependency version or invocation is unknown.
- Optional external data is used without availability and terms review.
- Raw QR basis or raw singular-vector identity is required for pass status.
- Skip, defer, unsupported, or advisory rows are treated as pass evidence.
- Local timing is used as portable performance evidence.
- A narrow comparison is worded as ecosystem parity or state-of-the-art proof.
- Hosted proof is claimed without run IDs, commit SHA, job names, conclusions,
  and artifact policy.
- Required C quality gates fail after `.c` or `.h` changes.

## Day 12 Handoff

Day 12 should convert all Sprint 147 gates into a quality surface map. The map
must specify which surfaces require the full C gate, which require corpus,
report, package, CI, or comparison-specific checks, and when to stop for failed
or unclear evidence.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 154 has a bounded comparison target. | Complete | Candidate target table defaults to one QR or partial-SVD maintained corpus family after Sprints 150-151. |
| Comparison evidence cannot be widened into broad parity. | Complete | Row schema, metric rules, wording rules, rejected wording, and stop conditions fence broad ecosystem parity. |
| State-of-the-art remains rejected without direct evidence. | Complete | State-of-the-art wording remains rejected unless Sprint 154 produces named, versioned, bounded comparison evidence and Sprint 156 approves only a narrow claim. |
