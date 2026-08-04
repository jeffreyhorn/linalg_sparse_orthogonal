# Sprint 137 Day 8 - Corpus & Oracle Evidence Templates

## Purpose

Day 8 defines reusable evidence templates for the Sprint 138 maintained
numerical corpus and oracle lane. These templates give Sprint 138 enough
structure to implement the corpus without redefining row semantics.

The templates are intentionally narrow. They support fixture-local evidence,
deterministic generated matrices, explicit optional-data handling, and oracle
rows that can feed later QR, partial-SVD, and report-index work without
claiming broad external-library parity or state-of-the-art status.

## Scope

Selected Day 7 target:

- Sprint 138 builds a maintained numerical corpus/oracle contract with one
  durable deterministic fixture lane and explicit skip/defer semantics.

These templates cover:

- corpus fixture metadata;
- deterministic generated-matrix metadata;
- optional external-data skip/defer metadata;
- oracle row fields for expected result, observed result, tolerance, support
  tier, command, fixture key, and source commit;
- failure interpretation for oracle mismatches, unsupported fixtures, and
  unavailable external data.

## Corpus Fixture Metadata Template

Every maintained corpus fixture should have one manifest row. The row defines
what the fixture is allowed to prove and what it is not allowed to imply.

| Field | Required | Meaning |
| --- | --- | --- |
| `fixture_key` | Yes | Stable unique key used by tests, reports, oracle rows, and docs. |
| `fixture_family` | Yes | Corpus family such as `qr_rank_deficient`, `svd_clustered`, `least_squares`, `spd`, `indefinite`, or `runtime_sentinel`. |
| `storage_kind` | Yes | `inline`, `generated`, `matrix_market`, or `optional_external`. |
| `matrix_path` | Conditional | Repository path or optional-data path for stored matrix fixtures. Empty for generated-only fixtures. |
| `generator_key` | Conditional | Stable generator key for deterministic generated fixtures. Required when `storage_kind` is `generated`. |
| `rows` | Yes | Matrix row count. |
| `cols` | Yes | Matrix column count. |
| `nnz` | Yes | Expected structural nonzero count after loading/generation. |
| `symmetry` | Yes | `none`, `symmetric`, `hermitian_not_applicable`, or `structural_symmetric`. |
| `definiteness` | Yes | `spd`, `semidefinite`, `indefinite`, `singular`, `rectangular`, or `unknown`. |
| `rank_status` | Yes | `full_rank`, `rank_deficient`, `numerically_rank_deficient`, or `unknown`. |
| `expected_rank` | Conditional | Expected rank when rank participates in the claim. |
| `nullity` | Conditional | Expected nullity when nullspace/subspace behavior participates in the claim. |
| `conditioning_class` | Yes | `well_conditioned`, `moderate`, `ill_conditioned`, `near_singular`, or `not_applicable`. |
| `scale_class` | Yes | `unit`, `scaled`, `mixed_scale`, or `not_applicable`. |
| `sparsity_class` | Yes | `diagonal`, `banded`, `block`, `graph_laplacian`, `random_sparse`, `structured_sparse`, or `other`. |
| `rhs_policy` | Yes | `none`, `single_rhs`, `multi_rhs`, `generated_rhs`, or `stored_rhs`. |
| `expected_behavior` | Yes | `success`, `diagnostic_failure`, `unsupported`, `non_convergence`, or `skip`. |
| `claim_scope` | Yes | One-sentence fixture-local claim boundary. |
| `non_claims` | Yes | Semicolon-separated boundaries, such as no external parity, no broad corpus coverage, or no portable performance claim. |
| `support_tier` | Yes | `reviewed_linux`, `reviewed_cross_platform`, `supplemental_macos`, `supplemental_windows`, `local_only`, or `optional_data`. |
| `validation_command` | Yes | Command expected to exercise the fixture or report row. |
| `owner` | Yes | Primary owner workstream. |
| `introduced_in` | Yes | Sprint or commit where the fixture becomes maintained. |

### Fixture Manifest Example

```text
fixture_key: qr_rank_deficient_6x4_nullspace_v1
fixture_family: qr_rank_deficient
storage_kind: generated
matrix_path:
generator_key: qr_rank_deficient_nullspace_generator_v1
rows: 6
cols: 4
nnz: 14
symmetry: none
definiteness: rectangular
rank_status: rank_deficient
expected_rank: 3
nullity: 1
conditioning_class: moderate
scale_class: unit
sparsity_class: structured_sparse
rhs_policy: generated_rhs
expected_behavior: success
claim_scope: Proves fixture-local QR rank/nullity and subspace residual behavior.
non_claims: no raw-basis parity; no global minimum-norm guarantee; no SuiteSparse parity
support_tier: reviewed_linux
validation_command: make test TEST_FILTER=qr_rank_deficient
owner: QR owner
introduced_in: Sprint 139
```

## Deterministic Generated-Matrix Metadata Template

Generated fixtures must be reproducible from metadata alone. Randomized
generation is permitted only when the seed and algorithm version are stable.

| Field | Required | Meaning |
| --- | --- | --- |
| `generator_key` | Yes | Stable generator identifier referenced by fixture rows. |
| `generator_version` | Yes | Integer or semantic version that changes when generated values change. |
| `algorithm` | Yes | Short name of deterministic construction, such as `diagonal_plus_rank_one`, `seeded_sparse_blocks`, or `clustered_spectrum`. |
| `seed` | Conditional | Required for seeded generators. |
| `parameters` | Yes | Stable key/value parameters needed to reproduce the matrix. |
| `expected_structure_hash` | Yes | Hash of row/column index structure after generation. |
| `expected_value_hash` | Yes | Hash of numeric values after generation and canonical serialization. |
| `canonical_format` | Yes | Format used to compute hashes, such as sorted CSR triplets with fixed precision. |
| `floating_policy` | Yes | Precision, rounding, tolerance, and platform-sensitive behavior notes. |
| `regeneration_command` | Yes | Command that regenerates or verifies the fixture. |
| `change_policy` | Yes | Required review action when generator output changes. |

### Generated-Matrix Example

```text
generator_key: svd_clustered_spectrum_generator_v1
generator_version: 1
algorithm: clustered_spectrum
seed: 137140
parameters: rows=8; cols=6; cluster_center=1.0; cluster_gap=1e-8; rank=5
expected_structure_hash: TBD_BY_SPRINT_138
expected_value_hash: TBD_BY_SPRINT_138
canonical_format: sorted_csr_triplets_fixed_17g
floating_policy: compare singular values with fixture-local tolerance; compare subspaces with projector metric
regeneration_command: TBD_BY_SPRINT_138
change_policy: update generator version, fixture hash, oracle rows, and docs together
```

## Optional Data Skip/Defer Template

Optional external data must never be interpreted as pass evidence when it is
unavailable. A skipped optional row is evidence that the skip policy worked,
not evidence that the numerical behavior passed.

| Field | Required | Meaning |
| --- | --- | --- |
| `optional_data_key` | Yes | Stable key for the optional data source or subset. |
| `source_name` | Yes | Human-readable source name. |
| `source_url_or_reference` | Yes | URL, citation, or local acquisition note. |
| `license_or_terms` | Yes | License or usage terms required before use. |
| `expected_location` | Yes | Path or environment variable used by tests/scripts. |
| `availability_state` | Yes | `available`, `unavailable`, `disabled`, `unsupported_platform`, `license_missing`, or `network_unavailable`. |
| `skip_reason` | Conditional | Required unless `availability_state` is `available`. |
| `defer_reason` | Conditional | Required when the row is intentionally deferred instead of skipped. |
| `fixture_keys` | Yes | Fixture keys that depend on this optional data. |
| `validation_command` | Yes | Command that reports available/unavailable/deferred state. |
| `pass_interpretation` | Yes | Meaning when the optional data is available and the numerical check passes. |
| `skip_interpretation` | Yes | Meaning when the optional data is unavailable or disabled. |
| `claim_boundary` | Yes | Explicit statement that skip/defer does not prove solver behavior. |

### Optional Data Example

```text
optional_data_key: suitesparse_rank_deficient_qr_subset_v1
source_name: SuiteSparse Matrix Collection rank-deficient QR subset
source_url_or_reference: TBD_BY_SPRINT_138
license_or_terms: TBD_BY_SPRINT_138
expected_location: SPARSE_CORPUS_OPTIONAL_DIR/suitesparse_rank_deficient_qr
availability_state: unavailable
skip_reason: optional data directory not configured
defer_reason:
fixture_keys: qr_rank_deficient_external_*
validation_command: TBD_BY_SPRINT_138
pass_interpretation: fixture-local QR check passed for configured external data
skip_interpretation: optional data was not configured; no QR behavior was proven
claim_boundary: no SuiteSparse parity or broad corpus coverage claim
```

## Oracle Row Template

Oracle rows connect fixtures to expected and observed results. They should be
machine-readable where possible, but the field meanings must stay clear even
before Sprint 138 chooses the concrete storage format.

| Field | Required | Meaning |
| --- | --- | --- |
| `oracle_row_id` | Yes | Stable row identifier. |
| `fixture_key` | Yes | Fixture key from the corpus manifest. |
| `solver_family` | Yes | `qr`, `partial_svd`, `lu`, `ldlt`, `iterative`, `eigs`, `runtime`, or another maintained family. |
| `operation` | Yes | Operation being checked, such as `factor`, `solve`, `rank`, `nullspace_subspace`, `singular_values`, or `convergence_budget`. |
| `command` | Yes | Exact command that produced the observed row. |
| `source_commit` | Yes | Git commit used for the observed row. |
| `platform` | Yes | OS and architecture. |
| `compiler` | Yes | Compiler and version where relevant. |
| `configuration` | Yes | Build flags, optional modes, backend state, and data availability relevant to the row. |
| `support_tier` | Yes | Reviewed, supplemental, optional, local-only, or staged status. |
| `expected_result_kind` | Yes | `value`, `residual_norm`, `rank`, `nullity`, `subspace_distance`, `status`, `diagnostic`, or `performance_local`. |
| `expected_result` | Yes | Expected value, range, status, or structured result. |
| `observed_result` | Yes | Observed value, range, status, or structured result. |
| `tolerance_kind` | Yes | `absolute`, `relative`, `mixed`, `projector`, `status_only`, or `not_applicable`. |
| `tolerance_value` | Conditional | Required unless `tolerance_kind` is `status_only` or `not_applicable`. |
| `comparison_status` | Yes | `pass`, `fail`, `skip`, `defer`, `unsupported`, or `xfail`. |
| `failure_class` | Conditional | Required when `comparison_status` is not `pass`. |
| `skip_or_defer_reason` | Conditional | Required for `skip` or `defer`. |
| `claim_scope` | Yes | Fixture-local statement the row supports when it passes. |
| `non_claims` | Yes | Boundaries preserved by the row. |
| `generated_at` | Yes | Timestamp or build metadata for generated reports. |

### Oracle Row Example

```text
oracle_row_id: qr_rank_deficient_6x4_nullspace_v1_projector
fixture_key: qr_rank_deficient_6x4_nullspace_v1
solver_family: qr
operation: nullspace_subspace
command: make test TEST_FILTER=qr_rank_deficient
source_commit: TBD_BY_SPRINT_139
platform: linux-x86_64
compiler: TBD_BY_SPRINT_139
configuration: static_default; optional_data=disabled
support_tier: reviewed_linux
expected_result_kind: subspace_distance
expected_result: projector_distance <= 1e-10
observed_result: TBD_BY_SPRINT_139
tolerance_kind: projector
tolerance_value: 1e-10
comparison_status: TBD_BY_SPRINT_139
failure_class:
skip_or_defer_reason:
claim_scope: fixture-local QR nullspace/subspace residual behavior
non_claims: no raw-basis parity; no broad QR parity; no global minimum-norm guarantee
generated_at: TBD_BY_SPRINT_139
```

## Failure Interpretation Rules

| Status or failure class | Meaning | Required handling |
| --- | --- | --- |
| `pass` | Observed result satisfied the expected result under the row tolerance and support tier. | May support only the row's `claim_scope`; preserve `non_claims`. |
| `fail_oracle_mismatch` | Observed numerical result did not satisfy the expected oracle value, residual, rank, nullity, status, or subspace metric. | Treat as a failing validation result; do not update reports to pass without implementation or tolerance review. |
| `fail_generator_mismatch` | Regenerated structure or values did not match expected hashes. | Treat as corpus integrity failure; update generator version only with reviewed fixture/oracle changes. |
| `fail_report_stale` | Oracle or report row was produced from stale commit, command, configuration, platform, or data metadata. | Treat as stale evidence, not numerical failure; regenerate before using the row for claims. |
| `skip_optional_unavailable` | Optional external data was unavailable, disabled, not licensed, unsupported, or not configured. | Record skip as policy evidence only; it is not solver pass evidence. |
| `defer_not_implemented` | Fixture or oracle row is defined but intentionally not implemented yet. | Keep out of pass counts; list owner and prerequisite. |
| `unsupported_platform` | Fixture, data source, generator, or validation command is outside the current platform support tier. | Keep support-tier boundary explicit; do not promote platform claims. |
| `xfail_known_residual` | Known residual is expected to fail until a selected sprint closes it. | Require owner, issue/residual reference, and removal condition. |

## Minimum Sprint 138 Implementation Contract

Sprint 138 should not be considered complete until it implements the following
minimum corpus/oracle contract:

1. A maintained manifest path with rows that satisfy the corpus fixture
   metadata template.
2. At least one deterministic generated-matrix fixture lane with stable
   generator metadata and reproducibility checks.
3. Explicit optional-data handling where unavailable data produces `skip` or
   `defer`, never `pass`.
4. At least one oracle row path with expected result, observed result,
   tolerance, support tier, command, fixture key, source commit, and
   comparison status.
5. Failure interpretation docs for oracle mismatches, generator mismatches,
   stale reports, optional-data skips, unsupported platforms, and known
   residual xfails.
6. A validation command that exercises the maintained corpus/oracle lane.
7. Public or maintainer documentation stating fixture-local claim boundaries.

## Claim Boundaries

Corpus/oracle rows may claim fixture-local evidence only. Even after Sprint 138
implements these templates, the following claims remain blocked unless later
sprints add independent evidence:

- broad SuiteSparse Matrix Collection coverage;
- external-library parity;
- broad rank-deficient QR correctness;
- broad partial-SVD convergence or ordering guarantees;
- package, ABI, or platform support expansion;
- portable performance or scalability;
- unqualified state-of-the-art status.

## Day 8 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 138 can implement the corpus lane without redefining row semantics. | Complete | Fixture, generated-matrix, optional-data, oracle-row, failure-interpretation, and minimum implementation templates define required fields and meanings. |
| Skipped optional data cannot be mistaken for pass evidence. | Complete | Optional-data template and failure rules require `skip_optional_unavailable` to remain policy evidence only. |
| Oracle rows preserve fixture-local claim boundaries. | Complete | Oracle row template requires `claim_scope` and `non_claims`, and the claim-boundary section blocks broad parity and state-of-the-art claims. |
