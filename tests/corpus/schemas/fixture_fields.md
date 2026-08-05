# Corpus Fixture Field Schema

This file records the Day 5 maintained corpus skeleton fields. The field
meanings come from the Sprint 137 Day 8 evidence templates and the Sprint 138
Day 3 taxonomy review.

## Fixture Manifest

`tests/corpus/manifests/fixtures.tsv` contains one row per maintained fixture.

| Field | Required | Meaning |
| --- | --- | --- |
| `fixture_key` | Yes | Stable unique key used by tests, reports, oracle rows, and docs. |
| `fixture_family` | Yes | Corpus family such as `qr_rank_deficient`, `svd_clustered`, `least_squares`, `spd`, `indefinite`, or `runtime_sentinel`. |
| `storage_kind` | Yes | `inline`, `generated`, `matrix_market`, or `optional_external`. |
| `matrix_path` | Conditional | Repository path or optional-data path for stored matrix fixtures; empty for generated-only fixtures. |
| `generator_key` | Conditional | Stable generator key for deterministic generated fixtures. |
| `rows` | Yes | Matrix row count. |
| `cols` | Yes | Matrix column count. |
| `nnz` | Yes | Nonzero count expected from fixture metadata. |
| `symmetry` | Yes | `none`, `symmetric`, `structural_symmetric`, or `hermitian_not_applicable`. |
| `definiteness` | Yes | `spd`, `semidefinite`, `indefinite`, `singular`, `rectangular`, or `unknown`. |
| `rank_status` | Yes | `full_rank`, `rank_deficient`, `numerically_rank_deficient`, or `unknown`. |
| `expected_rank` | Conditional | Expected rank when rank participates in the claim. |
| `nullity` | Conditional | Expected nullity when nullspace/subspace behavior participates in the claim. |
| `conditioning_class` | Yes | `well_conditioned`, `moderate`, `ill_conditioned`, `near_singular`, or `not_applicable`. |
| `scale_class` | Yes | `unit`, `scaled`, `mixed_scale`, or `not_applicable`. |
| `sparsity_class` | Yes | `diagonal`, `banded`, `block`, `graph_laplacian`, `random_sparse`, `structured_sparse`, or `other`. |
| `rhs_policy` | Yes | `none`, `single_rhs`, `multi_rhs`, `generated_rhs`, or `stored_rhs`. |
| `expected_behavior` | Yes | `success`, `diagnostic_failure`, `unsupported`, `non_convergence`, or `skip`. |
| `claim_scope` | Yes | Fixture-local statement the row may support when validated. |
| `non_claims` | Yes | Semicolon-separated claims this row does not support. |
| `support_tier` | Yes | `reviewed_linux`, `reviewed_cross_platform`, `supplemental_macos`, `supplemental_windows`, `local_only`, or `optional_data`. |
| `validation_command` | Yes | Maintained command expected to exercise the fixture or report skip/defer state. |
| `owner` | Yes | Responsible owner or sprint. |
| `introduced_in` | Yes | Sprint or commit where the fixture becomes maintained. |

`TBD_*` values are allowed only in Day 5 skeleton rows. They are not pass
evidence and must be replaced before the fixture can be promoted as a
validated corpus lane.

## Generator Manifest

`tests/corpus/manifests/generators.tsv` contains deterministic generator
contracts.

| Field | Required | Meaning |
| --- | --- | --- |
| `generator_key` | Yes | Stable generator identifier referenced by fixture rows. |
| `generator_version` | Yes | Integer version incremented when generated structure, values, or semantics change. |
| `algorithm` | Yes | Short deterministic algorithm description. |
| `seed` | Yes | Use `none` for non-random deterministic generators. |
| `parameters` | Yes | Machine-readable parameter string with stable key order. |
| `expected_structure_hash` | Yes | Hash of canonical generated structure output. |
| `expected_value_hash` | Yes | Hash of canonical generated value output. |
| `canonical_format` | Yes | Matrix/vector serialization form used by hashes. |
| `floating_policy` | Yes | Tolerance, exact-value, or comparison policy. |
| `regeneration_command` | Yes | Maintained command that regenerates or verifies the fixture. |
| `change_policy` | Yes | Required update set when the generator changes. |

## Optional Data Manifest

`tests/corpus/manifests/optional_data.tsv` contains optional external-data
skip/defer policy rows. Actual optional data lives outside the repository
under `SPARSE_CORPUS_OPTIONAL_DATA_DIR`.

| Field | Required | Meaning |
| --- | --- | --- |
| `optional_data_key` | Yes | Stable key for the optional data source or subset. |
| `source_name` | Yes | External source name. |
| `source_url_or_reference` | Yes | Source URL or citation/reference. |
| `license_or_terms` | Yes | License, terms, or review status. |
| `expected_location` | Yes | Relative location under `SPARSE_CORPUS_OPTIONAL_DATA_DIR`. |
| `availability_state` | Yes | `available`, `unavailable`, `disabled`, or `deferred`. |
| `skip_reason` | Conditional | Required unless `availability_state=available`. |
| `defer_reason` | Conditional | Required when the row is intentionally deferred instead of skipped. |
| `fixture_keys` | Yes | Fixture keys that depend on the optional data. |
| `validation_command` | Yes | Command that reports available/unavailable/deferred state. |
| `pass_interpretation` | Yes | Meaning when the optional data is available and the numerical check passes. |
| `skip_interpretation` | Yes | Meaning when the optional data is unavailable or disabled. |
| `claim_boundary` | Yes | Explicit statement that skip/defer does not prove solver behavior. |

Unavailable, disabled, or deferred optional data must not be counted as
numerical pass evidence. The maintained validator requires skip interpretation
wording that avoids pass evidence and claim-boundary wording that preserves
external-parity non-claims.
