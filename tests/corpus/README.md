# Maintained Numerical Corpus

This directory contains source-controlled metadata for maintained numerical
corpus fixtures. It records what each fixture is allowed to prove, how
generated fixtures are reproduced, how optional external data is skipped or
deferred, and where expected results live.

The corpus metadata is fixture-local evidence only. It does not claim broad
corpus completeness, SuiteSparse parity, external-library parity, broad QR or
SVD correctness, package/platform support, portable performance, coverage
completeness, or state-of-the-art status.

## Layout

| Path | Purpose |
| --- | --- |
| `manifests/fixtures.tsv` | Maintained fixture rows. |
| `manifests/generators.tsv` | Deterministic generated-matrix metadata rows. |
| `manifests/optional_data.tsv` | Optional external-data skip/defer policy rows. |
| `manifests/report_families.tsv` | Report-family contract rows for normalized index generation. |
| `manifests/selected_report_targets.tsv` | Selected oracle, comparison, and performance target metadata for report/workflow guards. |
| `expected/` | Small committed expected-result rows for maintained fixtures. |
| `schemas/fixture_fields.md` | Fixture, generator, and optional-data field definitions. |
| `schemas/oracle_fields.md` | Observed oracle row field definitions and status semantics. |
| `schemas/report_index_fields.md` | Normalized report-index contract field definitions. |
| `fixtures/` | Future promoted source-controlled matrix fixtures. |
| `../../scripts/validate_corpus_schema.py` | Lightweight schema check for maintained corpus TSV skeletons. |
| `../../scripts/run_corpus_oracle.py` | Local oracle/report emission command for maintained corpus rows. |

Generated matrices, observed oracle rows, logs, report indexes, and local run
manifests belong under ignored `build/corpus/` or `build/corpus-reports/`, not
under this source-controlled directory.

## First Lane

Sprint 138 reserves the first durable fixture lane for:

- fixture key: `qr_rank_deficient_6x4_nullspace_v1`
- fixture family: `qr_rank_deficient`
- generator key: `qr_rank_deficient_6x4_nullspace_generator_v1`

The Day 5 rows began as placeholders for layout validation. Day 9 added
deterministic generator hashes and first-lane expected results. Day 10 added
the maintained oracle/report command that emits observed rows under ignored
`build/` paths. Passing generated rows are local, fixture-scoped evidence for
this first lane only; they are not broad QR, least-squares, minimum-norm, or
package/platform evidence.

Sprint 150 expands the maintained QR corpus family with selected
rank-deficient rectangular and underdetermined minimum-norm fixtures. The
expanded family remains fixture-local and `local_only`; it does not promote
broad QR correctness, raw-basis identity, external-library parity, platform,
package, ABI, performance, or state-of-the-art claims.

## Ownership

| Surface | Owner | Update rule |
| --- | --- | --- |
| `manifests/fixtures.tsv` | Corpus maintainer, with solver-owner review for numerical semantics. | Defines fixture identity, family, support tier, claim scope, and validation command. New rows must keep claim scope fixture-local until reviewed evidence exists. |
| `manifests/generators.tsv` | Corpus maintainer. | Defines deterministic generator metadata, canonical format, hashes, seed policy, and regeneration command. Hash changes require an explicit generator or fixture revision. |
| `manifests/optional_data.tsv` | Corpus maintainer. | Defines external-data availability, skip/defer policy, and non-claim wording. Optional payloads stay outside the repository. |
| `manifests/report_families.tsv` | Report maintainer, with corpus, benchmark, package, CI, and documentation owners for family-specific semantics. | Defines report-family row meanings, support tiers, freshness policies, commands, artifact patterns, claim scopes, and non-claim boundaries for normalized indexing. |
| `manifests/selected_report_targets.tsv` | Report maintainer, with corpus, benchmark, and CI-owner review for target-specific semantics. | Defines selected report target identity, commands, expected rows, required files, workflow artifacts, support tiers, claim scopes, and non-claims for selected report/workflow guards. |
| `expected/*.tsv` | Corpus maintainer for schema; solver owner for expected numerical meaning. | Source-controlled target rows are prerequisites for observed evidence, not observed pass evidence by themselves. |
| `schemas/*.md` | Corpus and report maintainers. | Defines row semantics. Field or status changes need migration notes and validator updates. |
| `../../scripts/validate_corpus_schema.py` | Corpus maintainer. | Enforces TSV shape, required references, selected enums, first-lane generator hashes, and false-pass guardrails. |
| `../../scripts/run_corpus_oracle.py` | Corpus maintainer. | Emits local observed oracle rows, skip/defer rows, report index, and run manifest under ignored `build/` paths. |
| `build/corpus/`, `build/corpus-reports/` | Generated output owner is the local runner. | Do not commit generated oracle rows, report indexes, skip rows, or manifests unless a later sprint explicitly promotes them. |

## Row Interpretation

Fixture manifest rows define eligible evidence lanes. They do not prove that a
solver passed.

Generator rows define reproducibility metadata and hash expectations. They do
not prove solver correctness.

Expected-result rows define target comparisons for a fixture. They do not
become observed evidence until an oracle command emits matching generated
rows.

Observed oracle rows under `build/corpus/oracle/` are generated local evidence.
Only rows with `comparison_status=pass` and a fixture-local claim scope count
as pass evidence, and only for the named fixture, operation, command, commit,
platform, compiler, configuration, tolerance, and support tier.

Optional-data skip or defer rows are policy evidence only. They must not be
counted as solver pass evidence, SuiteSparse parity, external corpus parity, or
reviewed platform coverage.

Report index rows aggregate generated output locations and row status. They
preserve row meaning; they are not release proof, broad coverage proof, or
state-of-the-art evidence.

Report-family contract rows in `manifests/report_families.tsv` are the
source-controlled vocabulary for normalized report indexing. They define how a
family should be discovered and interpreted, but they do not prove that any
generated report exists, is fresh, or passed.

Selected report target rows in `manifests/selected_report_targets.tsv` are the
source-controlled authority for the selected oracle, comparison, and
performance targets that report and workflow guards consume. They narrow
existing report-family semantics to named selected targets; they do not widen
family-level claims or turn unselected report families into selected proof.
The selected performance target is the threshold-free
`SRT-BENCH-REFACTOR-CSC-NOS4` benchmark row for `bench_refactor_csc` on
`tests/data/suitesparse/nos4.mtx --repeat 1`; it records methodology and
freshness evidence only, with `baseline=n/a`, `threshold=n/a`, and no portable
performance, release benchmark, algorithmic superiority, platform parity,
package/ABI, or state-of-the-art claim.
The manifest is positive selected-target authority, not a general deferral
registry. Sprint 190 wires one bounded Windows selected Cholesky comparison
freshness workflow for `cholesky-spd-tridiag-5`, but the source manifest does
not list `windows` until selected Cholesky metadata, support tier, and claim
wording are reviewed together. The Sprint 182 deferral remains active for all
other Windows report freshness. A broader Windows promotion must add exact
workflow file, job, artifact, platform, support-tier, claim-scope, and
non-claim metadata rather than drifting an existing selected row silently. The
hosted Windows PowerShell validation lane owns selected workflow snippet
parsing and structural guards only; it is not selected report freshness or
selected artifact publication evidence.

Sprint 186 closeout keeps selected targets and residuals separate: selected
rows are positive proof inputs for their named target, while broad Windows
report freshness, Windows selected oracle or benchmark freshness, unavailable
local PowerShell validation, missing optional dependencies, and absent local
generated reports remain deferral, skip, warning, or residual context. Do not
reinterpret those states as pass evidence.

## Stale Reports

Generated oracle rows and report indexes are fresh only for the recorded
commit, branch, command, platform, compiler, configuration, generator hashes,
expected-result rows, optional-data state, and support tier.

Regenerate `build/corpus/` and `build/corpus-reports/` outputs whenever any of
these inputs change:

- source commit or branch
- corpus manifest row
- expected-result row
- generator algorithm, parameters, canonical text, or hash
- oracle command or validator behavior
- optional-data configuration or availability
- compiler, platform, or build configuration
- support tier, claim scope, tolerance, or non-claim wording

Sprint 141 report-index work should keep using the current report fields for
commit, command, platform, compiler, configuration, support tier, status,
claim scope, generated path, and non-claims. Sprint 141 may normalize freshness
checks, but it should not reinterpret skip/defer rows as pass evidence.

Use the normalized index after generating corpus/oracle reports when you need
cross-family row discovery or freshness diagnostics:

```sh
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

`--check` validates normalized row construction. `--check-freshness` emits
diagnostics in the form
`freshness: <severity>: <row_id>: <state>: <reason>`. Missing oracle rows warn
by default and become errors only when `--require-generated oracle` is used.
Source-controlled fixture, generator, optional-data, and expected-result rows
remain advisory or skip/defer policy evidence until a generated oracle row
records observed status. The reviewed Linux hosted lane mirrors the selected
required oracle gate and the selected QR, partial-SVD, LU, and Cholesky
comparison freshness gate. The reviewed macOS hosted lane mirrors only the
selected QR, partial-SVD, LU, and Cholesky comparison freshness gate. Those
hosted lanes cover only their selected generated rows and split artifacts.
Sprint 190 adds one bounded Windows hosted workflow path for
`cholesky-spd-tridiag-5` with target-specific freshness checking. These lanes
do not prove broad report-index freshness, selected oracle freshness on macOS,
selected oracle freshness on Windows, selected benchmark freshness on Windows,
broad Windows report freshness, or all local-only families. Windows CMake
build/test, static install/downstream validation, and PowerShell validation do
not imply generated report freshness.

## Selected QR, Partial-SVD, LU, And Cholesky Comparison Freshness

The selected comparison freshness gate is:

```sh
make report-index-comparison-freshness
```

It regenerates six fixture-local comparison families before strict
normalization:

| Target | Fixture | Comparison meaning | Artifact |
| --- | --- | --- | --- |
| `qr-minnorm` | `qr_underdetermined_minnorm_2x4` | minimum-norm solve against the source-controlled dense QR reference helper | `build/comparison/qr_minnorm/study.tsv` |
| `qr-compatible-ls` | `qr_overdetermined_compatible_5x3` | compatible least-squares solve against the source-controlled dense QR reference helper | `build/comparison/qr_compatible_ls/study.tsv` |
| `qr-incompatible-ls` | `qr_overdetermined_incompatible_4x2` | incompatible least-squares solve against the source-controlled dense QR reference helper | `build/comparison/qr_incompatible_ls/study.tsv` |
| `partial-svd-diag6-k2` | `partial_svd_diag6_k2` | partial-SVD diagonal top-k comparison against the source-controlled dense SVD reference helper | `build/comparison/partial_svd_diag6_k2/study.tsv` |
| `lu-nonsym-square-5` | `lu_nonsym_square_5` | linked-list LU square solve against the source-controlled dense LU reference helper | `build/comparison/lu_nonsym_square_5/study.tsv` |
| `cholesky-spd-tridiag-5` | `cholesky_spd_tridiag_5` | Cholesky SPD tridiagonal solve against the source-controlled dense Cholesky reference helper | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |

Each selected QR comparison family contributes six generated rows:
`project_status`, `baseline_status`, `residual_norm`, `solution_norm`,
`solution_values`, and `project_vs_baseline_max_abs_delta`.

The selected `lu_nonsym_square_5` family contributes the same six generated
rows as the QR comparison families.

The selected `cholesky_spd_tridiag_5` family contributes the same six
generated rows as the QR and LU solve comparison families.

The selected `partial_svd_diag6_k2` family contributes ten generated rows:
`project_status`, `baseline_status`, `singular_value_0`,
`singular_value_1`, `singular_values_max_abs_delta`, `residual_norm`,
`u_orthogonality`, `v_orthogonality`, `u_projector_diag`, and
`v_projector_diag`.

These rows are local generated evidence for the named fixtures by default. The
reviewed Linux and macOS hosted report-freshness lanes promote only this
selected comparison gate and their uploaded selected artifacts after hosted CI
passes. They do not prove broad QR, LU, nonsymmetric solve, Cholesky, SPD, SVD,
or partial-SVD correctness; broad least-squares parity; LU CSR parity;
Cholesky reordering parity; CSC-vs-linked-list parity; fill superiority; raw
QR basis identity; raw singular-vector identity; vector sign/orientation
identity; global rank-threshold behavior; broad rank-deficient solve behavior;
external-library parity; broad Windows report freshness; Windows selected
oracle or benchmark freshness; broad platform support; package/ABI support;
performance; release readiness; or state-of-the-art status. Optional
NumPy/SciPy dependency rows are deferred context only and never pass evidence.

## Sprint 139/Sprint 150 QR Lane

Sprint 139 uses `qr_rank_deficient_6x4_nullspace_v1` as the first QR fixture
closure lane. Sprint 150 keeps that seed and adds two rank-deficient
rectangular fixtures plus three underdetermined minimum-norm fixtures.

Seed fixture facts:

- generator key: `qr_rank_deficient_6x4_nullspace_generator_v1`
- shape: 6 rows by 4 columns
- nonzeros: 14
- rank: 3
- nullity: 1
- null vector direction: `[-1, -1, 0, 1]`
- rank row ID: `qr_rank_deficient_6x4_nullspace_v1_rank`
- nullity row ID: `qr_rank_deficient_6x4_nullspace_v1_nullity`
- residual row ID: `qr_rank_deficient_6x4_nullspace_v1_projector_residual`
- normalized null-vector residual tolerance: `1e-10`

Sprint 150 selected fixture facts:

| Fixture key | Family | Proof rows |
| --- | --- | --- |
| `qr_rankdef_duplicate_5x4_v1` | rank-deficient rectangular | rank, nullity, nullspace residual, projector/subspace distance |
| `qr_rankdef_dependent_row_4x3_v1` | rank-deficient rectangular | rank, nullity, nullspace residual, projector/subspace distance |
| `qr_underdetermined_minnorm_2x4` | underdetermined minimum-norm | status, residual, solution norm, exact solution values |
| `qr_minnorm_3x6_exact_values` | underdetermined minimum-norm | status, residual, solution norm, exact solution values |
| `qr_minnorm_5x10_exact_values` | underdetermined minimum-norm | status, residual, solution norm, exact solution values |

QR validation compares normalized residuals and projector/subspace distances
rather than raw QR basis vector equality because valid bases may differ by
sign, scale normalization, or equivalent subspace basis. The source-controlled
proof owner is [`../../tests/test_qr_corpus.c`](../../tests/test_qr_corpus.c),
and the opt-in solver-backed oracle/report path is:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

Before interpreting the QR lane, run:

```sh
python3 scripts/validate_corpus_schema.py
make build/test_qr_corpus && ./build/test_qr_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

Expected generated outputs for the opt-in QR lane:

- `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv` contains
  `26` observed rows: three generated-reference rows with
  `solver_family=unknown` and `23` solver-backed QR rows with
  `solver_family=qr`.
- The solver-backed QR rows cover the seed fixture plus the five Sprint 150
  fixtures listed above, and should all report `comparison_status=pass`.
- `build/corpus-reports/index.tsv` indexes the QR rows and their fixture-local
  claim scope.
- `build/corpus-reports/skips.tsv` contains optional-data skip/defer rows, not
  QR pass evidence.
- `build/corpus-reports/manifest.txt` records the command, row count, solver
  families, support tier, selected fixture keys, `solver_qr_row_count=23`, and
  `partial_svd_row_count=0` for a QR-only run.

Stale or unsupported QR report signals:

- the manifest command does not include `--include-solver-qr`;
- the report predates changes to corpus manifests, expected rows, schemas,
  `scripts/run_corpus_oracle.py`, `tests/test_qr_corpus.c`, or
  `tests/test_qr_helpers.h`;
- the oracle output lacks `23` `solver_family=qr` rows or lists a solver QR
  row count other than `23`;
- the manifest omits any selected QR fixture key;
- any maintained QR comparison row is not `pass`;
- optional-data skip/defer rows are cited as solver pass evidence.

Keep optional-data skip rows separate from QR pass evidence. Do not promote
support beyond `local_only` until a reviewed hosted lane records matching
generated evidence.

## Partial-SVD Corpus Lane

Sprint 140 and Sprint 151 maintain generated partial-SVD fixture lanes:

| Fixture key | Family | Generator key | Generated oracle rows |
| --- | --- | --- | ---: |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | `partial_svd_clustered_repeated` | `partial_svd_clustered_repeated_diag8x6_generator_v1` | 8 |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | `partial_svd_rankdef_rectangular` | `partial_svd_rankdef_diag6x4_k2_range_projector_generator_v1` | 7 |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1` | `partial_svd_lowrank_sparse_output` | `partial_svd_lowrank_rect5x7_k3_sparse_output_generator_v1` | 6 |
| `partial_svd_fail_closed_diag6_k2_v1` | `partial_svd_fail_closed` | `partial_svd_fail_closed_diag6_k2_generator_v1` | 5 |

The source-controlled proof owner is `tests/test_svd_partial_corpus.c`.

Run the local partial-SVD corpus lane with:

```sh
python3 scripts/run_corpus_oracle.py --include-partial-svd
```

Before interpreting the partial-SVD lane, run:

```sh
python3 scripts/validate_corpus_schema.py
make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus
python3 scripts/run_corpus_oracle.py --include-partial-svd
```

Expected generated outputs for the opt-in partial-SVD lane:

- `build/corpus/oracle/corpus.oracle.tsv` contains the QR generated-reference
  rows plus `26` partial-SVD generated-local rows for the four maintained
  fixtures above.
- `build/corpus-reports/index.tsv` indexes the partial-SVD rows and their
  fixture-local claim scope.
- `build/corpus-reports/manifest.txt` records the command, row count, solver
  families, support tier, and `partial_svd_row_count=26`.

The lane proves only fixture-local top-k singular values, rank, left/right
selected-subspace projectors, triplet residuals, orthogonality, sparse
low-rank shape/nnz/selected values/Frobenius behavior, default-budget success,
tight-budget fail-closed behavior, no partial arrays on tight-budget failure,
and default-budget recovery after failure for the generated fixtures above. It
does not claim broad partial-SVD correctness, raw singular-vector identity,
broad repeated-spectrum coverage, broad rank-deficient behavior, broad
sparse-output optimality, external-library parity, performance, package/ABI
support, platform parity, partial-result guarantees, or state-of-the-art
behavior.

Stale or unsupported partial-SVD report signals:

- the manifest command does not include `--include-partial-svd`;
- the report predates changes to corpus manifests, expected rows, schemas,
  `scripts/run_corpus_oracle.py`, `tests/test_svd_partial_corpus.c`, or
  `tests/test_svd_partial_shared_helpers.h`;
- the oracle output lacks the expected `8`, `7`, `6`, and `5` generated rows
  for the four maintained partial-SVD fixtures or lists a partial-SVD row count
  other than `26`;
- any maintained partial-SVD expected row is not reflected in the local oracle
  output or normalized report index;
- optional-data skip/defer rows are cited as partial-SVD pass evidence.

## Residual Register

- Reviewed hosted-platform promotion for corpus/oracle rows.
- Reviewed hosted-platform promotion for the Sprint 139 solver-backed QR rows.
- Global QR rank-threshold policy across scales and perturbations.
- Broad rank-deficient QR solve, residual-only least-squares, and minimum-norm
  behavior.
- COLAMD/reordered QR behavior with ordering-specific semantics.
- Broad SuiteSparse, LAPACK, NumPy, SciPy, platform, performance, corpus
  completeness, and state-of-the-art claims.
- Raw QR basis/sign/orientation parity; Sprint 139 compares normalized
  residual behavior instead.
- Reviewed hosted-platform promotion for the maintained partial-SVD corpus
  rows.
- Broad partial-SVD behavior beyond the maintained clustered/repeated,
  rank-deficient projector, sparse low-rank output, and fail-closed recovery
  fixtures.
- Report freshness normalization and stale-report diagnostics in Sprint 141.
- Optional external data availability, provenance, and reviewed pass policy.
- Public documentation/adoption wording that references corpus evidence
  without broadening claims.

## Validation

Run this structural check after editing corpus TSV files:

```sh
python3 scripts/validate_corpus_schema.py
```

The validator checks TSV widths, required fields, basic enum values,
fixture-to-generator references, deterministic first-lane generator hashes,
expected-result fixture references, and that placeholder expected-result rows
are not pass evidence.

Run this local corpus/oracle command to validate the first deterministic lane
and emit generated rows:

```sh
python3 scripts/run_corpus_oracle.py
```

Run this opt-in command when interpreting the Sprint 139 solver-backed QR lane:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

The command writes observed oracle rows under `build/corpus/oracle/` and a
report index plus optional-data skip rows under `build/corpus-reports/`. Those
outputs are generated local evidence and are not committed.

Default validation does not require optional external data. Disabled,
unavailable, deferred, or unsupported optional data is reported as skip/defer
policy evidence only and never as solver pass evidence.

Run the normalized freshness check when reviewing generated corpus evidence:

```sh
python3 scripts/normalize_report_index.py --family oracle --check-freshness
```

Use `--require-generated oracle` only when the current review actually requires
local generated oracle artifacts to exist and match the selected freshness
policy. Hosted CI runs the selected oracle gate as reviewed Linux evidence and
runs the selected QR, partial-SVD, LU, and Cholesky comparison gate as reviewed
Linux and macOS selected-artifact evidence, but generated corpus/report rows
remain fixture-local and do not imply broad solver, broad Windows report
freshness, Windows selected oracle or benchmark freshness, broad platform,
package, performance, external-parity, or state-of-the-art claims.

## Optional Data

Optional external data is configured outside the repository with
`SPARSE_CORPUS_OPTIONAL_DATA_DIR`. Optional matrices, archives, extracted
datasets, and downloaded data must not be committed here.

Unavailable optional data is skip-policy evidence only. It is not solver pass
evidence.
