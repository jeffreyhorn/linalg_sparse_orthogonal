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
| `expected/` | Small committed expected-result rows for maintained fixtures. |
| `schemas/fixture_fields.md` | Fixture, generator, and optional-data field definitions. |
| `schemas/oracle_fields.md` | Observed oracle row field definitions and status semantics. |
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

## Ownership

| Surface | Owner | Update rule |
| --- | --- | --- |
| `manifests/fixtures.tsv` | Corpus maintainer, with solver-owner review for numerical semantics. | Defines fixture identity, family, support tier, claim scope, and validation command. New rows must keep claim scope fixture-local until reviewed evidence exists. |
| `manifests/generators.tsv` | Corpus maintainer. | Defines deterministic generator metadata, canonical format, hashes, seed policy, and regeneration command. Hash changes require an explicit generator or fixture revision. |
| `manifests/optional_data.tsv` | Corpus maintainer. | Defines external-data availability, skip/defer policy, and non-claim wording. Optional payloads stay outside the repository. |
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

## Sprint 139 QR Lane

Sprint 139 uses `qr_rank_deficient_6x4_nullspace_v1` as the first QR fixture
closure lane.

Required fixture facts:

- generator key: `qr_rank_deficient_6x4_nullspace_generator_v1`
- shape: 6 rows by 4 columns
- nonzeros: 14
- rank: 3
- nullity: 1
- null vector direction: `[-1, -1, 0, 1]`
- rank row ID: `qr_rank_deficient_6x4_nullspace_v1_rank`
- nullity row ID: `qr_rank_deficient_6x4_nullspace_v1_nullity`
- residual row ID: `qr_rank_deficient_6x4_nullspace_v1_projector_residual`
- initial normalized null-vector residual tolerance: `1e-10`

QR validation for this first lane compares normalized residuals rather than
raw QR basis vector equality because valid bases may differ by sign, scale
normalization, or equivalent subspace basis. The source-controlled proof owner
is [`../../tests/test_qr_corpus.c`](../../tests/test_qr_corpus.c), and the
opt-in solver-backed oracle/report path is:

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
  six observed rows: three generated-reference rows with
  `solver_family=unknown` and three solver-backed QR rows with
  `solver_family=qr`.
- The solver-backed QR rows cover rank, nullity, and normalized nullspace
  residual, and should all report `comparison_status=pass`.
- `build/corpus-reports/index.tsv` indexes the QR rows and their fixture-local
  claim scope.
- `build/corpus-reports/skips.tsv` contains optional-data skip/defer rows, not
  QR pass evidence.
- `build/corpus-reports/manifest.txt` records the command, row count, solver
  families, support tier, and `solver_qr_row_count=3`.

Stale or unsupported QR report signals:

- the manifest command does not include `--include-solver-qr`;
- the report predates changes to corpus manifests, expected rows, schemas,
  `scripts/run_corpus_oracle.py`, `tests/test_qr_corpus.c`, or
  `tests/test_qr_helpers.h`;
- the oracle output lacks three `solver_family=qr` rows or lists a solver QR
  row count other than `3`;
- any Sprint 139 QR comparison row is not `pass`;
- optional-data skip/defer rows are cited as solver pass evidence.

Keep optional-data skip rows separate from QR pass evidence. Do not promote
support beyond `local_only` until a reviewed hosted lane records matching
generated evidence.

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
- Partial-SVD clustered and repeated singular-value fixture lanes in Sprint
  140.
- Partial-SVD rank-deficient range-projector follow-through in Sprint 140.
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

## Optional Data

Optional external data is configured outside the repository with
`SPARSE_CORPUS_OPTIONAL_DATA_DIR`. Optional matrices, archives, extracted
datasets, and downloaded data must not be committed here.

Unavailable optional data is skip-policy evidence only. It is not solver pass
evidence.
