# Sprint 138 Day 4 - Corpus Storage Layout Design

## Purpose

Day 4 defines the maintained repository paths, optional-data boundary,
generated-matrix metadata paths, expected-result paths, oracle/report output
paths, and row naming rules for the Sprint 138 corpus/oracle lane.

This is a design artifact. It does not create directories, manifests,
fixtures, generators, oracle rows, scripts, Makefile targets, or generated
reports. Day 5 owns the first repository layout implementation.

## Layout Principles

| Principle | Decision |
| --- | --- |
| Maintained metadata in source control | Corpus manifests, generator metadata, optional-data policy rows, expected-result baselines, and layout documentation live under a maintained source path. |
| Generated outputs out of source control | Report outputs, observed oracle rows, logs, and regenerated artifacts live under `build/` unless a later sprint explicitly promotes a stable artifact. |
| Optional data cannot look bundled | Optional external matrices are referenced by metadata and environment variables only; they do not live under the maintained corpus path. |
| Existing fixtures stay readable | Existing `tests/data` Matrix Market files remain usable by current tests; the maintained corpus layer records row meaning separately. |
| Report-index compatibility | Corpus/oracle outputs carry enough row identity, command, commit, support tier, status, claim scope, non-claims, and freshness metadata for Sprint 141. |
| One lane first | The layout supports all row types but should be implemented first for `qr_rank_deficient_6x4_nullspace_v1`. |

## Maintained Source Layout

Day 5 should create this source-controlled layout:

| Path | Row or artifact type | Source-control policy |
| --- | --- | --- |
| `tests/corpus/README.md` | Human-readable corpus ownership, row meaning, generated-output, and optional-data policy. | Commit. |
| `tests/corpus/manifests/fixtures.tsv` | Maintained corpus fixture rows. | Commit. |
| `tests/corpus/manifests/generators.tsv` | Deterministic generated-matrix metadata rows. | Commit. |
| `tests/corpus/manifests/optional_data.tsv` | Optional external-data skip/defer policy rows. | Commit. |
| `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` | Expected values for the first durable QR fixture lane. | Commit when stable. |
| `tests/corpus/expected/README.md` | Expected-result row ownership and update policy. | Commit. |
| `tests/corpus/schemas/fixture_fields.md` | Manifest field definitions copied from the accepted Sprint 137/138 contract. | Commit. |
| `tests/corpus/schemas/oracle_fields.md` | Oracle row field definitions and comparison status values. | Commit after Day 6 finalizes schema. |
| `tests/corpus/schemas/report_fields.md` | Report-index compatibility notes for corpus/oracle rows. | Commit after Day 10/13 when report command exists. |
| `tests/corpus/fixtures/` | Optional future source-controlled Matrix Market fixtures promoted into the maintained corpus. | Commit only reviewed small deterministic fixtures. |

The first generated QR lane does not require a stored matrix file because its
fixture row uses `storage_kind=generated` and a generator metadata row. If a
future maintained Matrix Market fixture is promoted under
`tests/corpus/fixtures/`, Day 5 or the promotion sprint must also update
`.gitignore` so reviewed corpus `.mtx` files are not ignored accidentally.

## Generated Output Layout

Generated outputs should live under `build/` and remain ignored:

| Path | Artifact type | Source-control policy |
| --- | --- | --- |
| `build/corpus/generated/` | Regenerated matrices, dense expansions, hashes, or temporary fixture material. | Do not commit. |
| `build/corpus/oracle/` | Observed oracle rows, comparison output, and validation logs. | Do not commit. |
| `build/corpus-reports/` | Maintained corpus/oracle command reports that can feed Sprint 141 normalization. | Do not commit. |
| `build/corpus-reports/index.tsv` | Generated report index for the latest local run. | Do not commit unless a later sprint explicitly promotes a stable checked-in example. |
| `build/corpus-reports/manifest.txt` | Human-readable local run manifest. | Do not commit. |

The generated-output naming follows the existing `build/bench-reports/...`
pattern while keeping corpus outputs separate from benchmark and performance
sentinel outputs.

## Fixture Manifest Path

`tests/corpus/manifests/fixtures.tsv` is the authoritative maintained fixture
manifest.

| Column | Required | Notes |
| --- | --- | --- |
| `fixture_key` | Yes | Stable unique key used by tests, reports, oracle rows, and docs. |
| `fixture_family` | Yes | Example: `qr_rank_deficient`, `svd_clustered`, `least_squares`, `spd`, `indefinite`, `runtime_sentinel`. |
| `storage_kind` | Yes | `inline`, `generated`, `matrix_market`, or `optional_external`. |
| `matrix_path` | Conditional | Repository path for committed data, optional-data relative path for external data, or empty for generated-only fixtures. |
| `generator_key` | Conditional | Required when `storage_kind=generated`. |
| `rows` | Yes | Matrix row count. |
| `cols` | Yes | Matrix column count. |
| `nnz` | Yes | Nonzero count expected from fixture metadata. |
| `symmetry` | Yes | Accepted Sprint 138 taxonomy value. |
| `definiteness` | Yes | Accepted Sprint 138 taxonomy value. |
| `rank_status` | Yes | Accepted Sprint 138 taxonomy value. |
| `expected_rank` | Conditional | Required when rank participates in the claim. |
| `nullity` | Conditional | Required when nullspace/subspace behavior participates in the claim. |
| `conditioning_class` | Yes | Accepted Sprint 138 taxonomy value. |
| `scale_class` | Yes | Accepted Sprint 138 taxonomy value. |
| `sparsity_class` | Yes | Accepted Sprint 138 taxonomy value. |
| `rhs_policy` | Yes | Accepted Sprint 138 taxonomy value. |
| `expected_behavior` | Yes | `success`, `diagnostic_failure`, `unsupported`, `non_convergence`, or `skip`. |
| `claim_scope` | Yes | Fixture-local claim boundary. |
| `non_claims` | Yes | Semicolon-separated non-claims. |
| `support_tier` | Yes | Initial rows should use `local_only` unless reviewed platform evidence exists. |
| `validation_command` | Yes | Maintained command expected to exercise the fixture or report skip/defer state. |
| `owner` | Yes | Responsible owner or sprint. |
| `introduced_in` | Yes | Sprint or commit where the row became maintained. |

## Generator Manifest Path

`tests/corpus/manifests/generators.tsv` records deterministic generator
contracts.

| Column | Required | Notes |
| --- | --- | --- |
| `generator_key` | Yes | Stable generator identifier referenced by fixture rows. |
| `generator_version` | Yes | Increment when structure, value generation, or semantics change. |
| `algorithm` | Yes | Short deterministic algorithm description. |
| `seed` | Yes | Use `none` for non-random deterministic generators. |
| `parameters` | Yes | Machine-readable parameter string; keep stable key order. |
| `expected_structure_hash` | Yes | Hash of canonical structure output. |
| `expected_value_hash` | Yes | Hash of canonical value output. |
| `canonical_format` | Yes | Matrix/vector serialization form used by the hash. |
| `floating_policy` | Yes | Tolerance, exact-value, or comparison policy. |
| `regeneration_command` | Yes | Maintained command that regenerates or verifies the fixture. |
| `change_policy` | Yes | Required update set when the generator changes. |

The first generator key is
`qr_rank_deficient_6x4_nullspace_generator_v1`.

## Optional External-Data Policy

Optional external data must not live under `tests/corpus/` or be silently
counted as bundled coverage. The maintained policy row lives in
`tests/corpus/manifests/optional_data.tsv`; the actual optional data root is
configured by environment variable.

| Setting | Decision |
| --- | --- |
| Environment variable | `SPARSE_CORPUS_OPTIONAL_DATA_DIR` |
| Expected local layout | `$SPARSE_CORPUS_OPTIONAL_DATA_DIR/<optional_data_key>/...` |
| Source-control policy | Optional external matrices, archives, extracted data, and downloaded data are not committed. |
| Missing data status | `skip_optional_unavailable` for unavailable or unconfigured data. |
| Intentionally postponed status | `defer_not_implemented` when the data policy exists but the fixture lane is not implemented. |
| Pass interpretation | Fixture-local numerical behavior only for configured optional data. |
| Skip interpretation | Skip-policy evidence only; no solver behavior was proven. |
| Claim boundary | No SuiteSparse parity, no external-library parity, and no broad corpus completeness claim. |

`tests/data/suitesparse/` remains the legacy bundled test-data location for
current tests. Day 4 does not reclassify those files as optional external
corpus rows. Any future promotion must create explicit corpus manifest rows and
non-claims before report indexes consume them.

## Expected-Result Layout

Expected-result files live under `tests/corpus/expected/` and are committed
only when they are stable, small, deterministic, and tied to a manifest row.

| Path pattern | Meaning |
| --- | --- |
| `tests/corpus/expected/<fixture_key>.tsv` | Fixture-local expected results for one maintained fixture. |
| `tests/corpus/expected/<fixture_key>.<operation>.tsv` | Optional split when one fixture needs multiple operation-specific expected result files. |
| `tests/corpus/expected/README.md` | Update rules, tolerance policy pointers, and non-claim reminders. |

For the first lane, the expected-result row should include rank, nullity,
projector/subspace residual tolerance, reconstruction/residual tolerance if
used, and non-claims. Raw QR basis vectors should not be the primary expected
artifact.

## Oracle and Report Output Paths

The maintained command introduced later in Sprint 138 should write observed
oracle rows and report metadata to ignored build paths:

| Path | Meaning |
| --- | --- |
| `build/corpus/oracle/<fixture_key>.oracle.tsv` | Observed oracle comparison rows for one fixture. |
| `build/corpus/oracle/<fixture_key>.log` | Human-readable validation log for one fixture. |
| `build/corpus-reports/index.tsv` | Corpus/report row index compatible with Sprint 141 normalization. |
| `build/corpus-reports/manifest.txt` | Local run manifest with command, commit, branch, platform, compiler, and configuration. |
| `build/corpus-reports/skips.tsv` | Current skip/defer rows for optional data or intentionally unimplemented rows. |

Oracle output rows should include the Day 8/Sprint 138 fields finalized on
Day 6: fixture key, solver family, operation, command, source commit,
platform, compiler, configuration, support tier, expected result, observed
result, tolerance kind/value, comparison status, failure class, skip/defer
reason, claim scope, non-claims, and generated timestamp.

## Naming Rules

| Identifier | Rule | Example |
| --- | --- | --- |
| Fixture key | Lowercase snake case: `<family>_<descriptor>_v<version>`. Include dimensions when they matter. | `qr_rank_deficient_6x4_nullspace_v1` |
| Fixture family | Lowercase snake case solver or matrix family. | `qr_rank_deficient` |
| Generator key | `<fixture_key>`-aligned generator key ending in `_generator_v<version>`. | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| Optional data key | Source or subset plus purpose and version. | `suitesparse_rank_deficient_qr_subset_v1` |
| Oracle row ID | `<fixture_key>_<comparison_kind>` when unambiguous; include operation as `<fixture_key>_<operation>_<comparison_kind>` only when needed to disambiguate rows. | `qr_rank_deficient_6x4_nullspace_v1_projector_residual` |
| Report row ID | `corpus_<row_family>_<row_subject>_v<version>`. | `corpus_oracle_qr_rank_deficient_6x4_nullspace_v1_v1` |
| Expected file | `<fixture_key>.tsv` unless operation-specific split is required. | `qr_rank_deficient_6x4_nullspace_v1.tsv` |

Identifiers must be stable once referenced by tests, reports, and docs. A
semantic change to generated structure, expected values, oracle meaning, or
claim scope should create a new versioned key or a documented generator
version bump with expected-result and report updates.

## Day 5 Implementation Checklist

| Task | Day 4 design requirement |
| --- | --- |
| Create source layout | Add `tests/corpus/`, `tests/corpus/manifests/`, `tests/corpus/expected/`, `tests/corpus/schemas/`, and optional future `tests/corpus/fixtures/`. |
| Add skeleton manifests | Add `fixtures.tsv`, `generators.tsv`, and `optional_data.tsv` with headers matching this design. |
| Add first-lane placeholders | Add rows for `qr_rank_deficient_6x4_nullspace_v1` and `qr_rank_deficient_6x4_nullspace_generator_v1` only if they can remain honest placeholders before generator/oracle implementation. |
| Add layout docs | Add `tests/corpus/README.md` and `tests/corpus/expected/README.md`. |
| Check ignored outputs | Confirm `build/` covers generated corpus outputs and update `.gitignore` only if committed corpus fixtures need exceptions. |
| Preserve non-claims | Ensure skeleton rows and docs do not imply broad corpus completeness, SuiteSparse parity, QR closure, SVD closure, report release proof, platform parity, performance, package support, or state-of-the-art status. |

## Day 4 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| All corpus row types have maintained paths. | Complete | Source layout covers fixture rows, generator rows, optional-data rows, expected results, schemas, and future promoted fixtures. |
| Optional-data paths cannot be mistaken for bundled fixtures. | Complete | Optional data uses `SPARSE_CORPUS_OPTIONAL_DATA_DIR` and metadata rows only; source-controlled corpus paths exclude optional external data payloads. |
| Report output paths can be normalized later without changing row meaning. | Complete | Oracle/report output paths include row identity, command, commit, support tier, status, claim scope, non-claims, and freshness-compatible metadata. |
