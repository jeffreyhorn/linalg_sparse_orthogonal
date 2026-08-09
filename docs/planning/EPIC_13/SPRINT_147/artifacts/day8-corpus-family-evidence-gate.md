# Sprint 147 Day 8 Corpus-Family Evidence Gate

## Purpose

Day 8 defines the evidence gate for Sprint 150 QR corpus expansion and Sprint
151 partial-SVD corpus expansion. The gate requires maintained fixture rows,
generator rows, expected-result rows, proof-owner tests, oracle/report rows,
validation commands, and bounded documentation before any broader corpus-family
claim is promoted.

The gate preserves the Day 6 claim boundaries: corpus-family claims may cover
only named fixtures, metrics, tolerances, commands, commit/platform context,
and support tier. They do not create broad QR, SVD, partial-SVD,
external-library parity, performance, platform, package, ABI, or
state-of-the-art claims.

## Source-Controlled Row Requirements

| Row Type | Required For Promotion | Owner |
| --- | --- | --- |
| Fixture row | Stable `fixture_key`, `fixture_family`, shape, `nnz`, rank/nullity when relevant, conditioning/scale/sparsity class, RHS policy, expected behavior, claim scope, non-claims, support tier, validation command, owner, and provenance. | Corpus maintainer with solver-owner review. |
| Generator row | Deterministic generator key/version, algorithm, seed, parameters, structure hash, value hash, canonical format, floating policy, regeneration command, and change policy. | Corpus maintainer. |
| Expected-result rows | One source-controlled expected file per fixture with row IDs, operation, comparison kind, expected result kind/value, tolerance kind/value, claim scope, non-claims, and `ready_for_oracle` status. | Solver owner for numerical meaning; corpus maintainer for schema. |
| Optional-data row | Availability, license/terms, expected external location, skip/defer reason, fixture keys, validation command, pass interpretation, skip interpretation, and claim boundary. | Corpus maintainer. |
| Report-family row | Row meaning, origin, status, support tier, freshness policy, generator command, artifact pattern, claim scope, non-claims, owner, and provenance when a new report family/subfamily is introduced. | Report maintainer. |

Source-controlled rows are prerequisites. They are not observed pass evidence
until a maintained proof command emits passing generated-local or hosted rows.

## QR Corpus-Family Gate

Sprint 150 should select two or three bounded QR fixture families that can
close completely. Preferred candidates from the Epic 13 plan are:

| Candidate Family | Required Semantics | Promotion Boundary |
| --- | --- | --- |
| Rank-deficient rectangular solve | Rank, nullity, residual, returned status, and solve-side behavior for named rectangular fixtures. | No broad rank-deficient QR correctness and no global rank-threshold policy. |
| Underdetermined minimum-norm | Residual, solution norm, rank/nullity where relevant, and explicit minimum-norm comparison metric for named fixtures. | No global minimum-norm guarantee beyond the named fixtures and tolerances. |
| Reorder/COLAMD-influenced QR | Permutation/fill or ordering-specific behavior, residual, and status for named fixtures. | No broad reorder optimality, COLAMD parity, or performance claim. |

Allowed QR comparison kinds:

| Semantics | Preferred Row Fields | Notes |
| --- | --- | --- |
| Rank/nullity | `operation=rank_info`, `comparison_kind=rank` or `nullity`, `tolerance_kind=exact` | Use exact expected values only when fixture construction supports them. |
| Residual | `comparison_kind=residual_norm`, `tolerance_kind=absolute`, `relative`, or `mixed` | State the norm formula, normalization, RHS policy, and tolerance. |
| Nullspace/subspace | `comparison_kind=residual_norm` or `subspace_distance` | Prefer residual/projector checks. Never require raw QR basis identity, sign, orientation, or column order. |
| Minimum-norm | `comparison_kind=value` or `residual_norm` with structured expected result | Define residual and norm comparison separately; do not use SVD as a global oracle unless the row says it is a bounded cross-check. |
| Reorder/COLAMD | `comparison_kind=value`, `diagnostic`, `residual_norm`, or `status` | Define permutation/fill semantics and keep performance wording out of the claim. |

QR promotion requires:

- source-controlled fixture, generator, and expected rows;
- a focused C proof owner, preferably extending `tests/test_qr_corpus.c` or a
  new focused corpus proof file instead of expanding `tests/test_qr.c`;
- oracle/report integration in `scripts/run_corpus_oracle.py` when generated
  rows are needed;
- `tests/corpus/README.md`, solver-selection docs, and maintainer guidance
  updated with fixture-family boundaries;
- no public wording that turns selected fixtures into broad QR parity.

## Partial-SVD Corpus-Family Gate

Sprint 151 should select bounded partial-SVD fixture families that can close
completely. Preferred candidates from the Epic 13 plan are:

| Candidate Family | Required Semantics | Promotion Boundary |
| --- | --- | --- |
| Repeated or clustered spectra | Singular values, left/right projector distances, triplet residuals, orthogonality, status, and top-k ambiguity handling. | No raw singular-vector identity and no broad repeated-spectrum guarantee. |
| Rank-deficient rectangular matrices | Rank, range/null-space-safe projector checks, residuals, and status for named fixtures. | No broad rank-deficient null-space or pseudoinverse claim. |
| Sparse low-rank output | Low-rank approximation metric, sparse-output/drop semantics if selected, and residual/quality checks. | No broad low-rank optimality or drop-tolerance guarantee unless exactly tested. |
| Convergence and fail-closed behavior | Default-budget success, tight-budget `SPARSE_ERR_NOT_CONVERGED`, no partial arrays on failure if promised, and recovery behavior. | No convergence-rate, portable iteration-count, or partial-result guarantee. |

Allowed partial-SVD comparison kinds:

| Semantics | Preferred Row Fields | Notes |
| --- | --- | --- |
| Singular values | `operation=singular_values`, `comparison_kind=value`, `tolerance_kind=absolute`, `relative`, or `mixed` | Define ordering and tolerance; repeated values require subspace-safe interpretation. |
| Subspaces | `operation=singular_subspace`, `comparison_kind=subspace_distance`, `tolerance_kind=projector` | Compare projectors or subspace distance, not raw singular vectors. |
| Triplet residuals | `operation=vector_residuals`, `comparison_kind=residual_norm` | Check `A*v ~= sigma*u` and `A^T*u ~= sigma*v` with named tolerance. |
| Orthogonality | `operation=orthogonality`, `comparison_kind=residual_norm` | State U/V orthogonality metric and tolerance. |
| Convergence budget | `operation=convergence_budget`, `comparison_kind=status`, `tolerance_kind=status_only` | Distinguish default success, tight-budget failure, and recovery. |
| Failure diagnostics | `operation=diagnostic`, `comparison_kind=diagnostic`, `tolerance_kind=not_applicable` | Use for no-partial-array or malformed-output checks. |

Partial-SVD promotion requires:

- source-controlled fixture, generator, and expected rows;
- a focused C proof owner, preferably extending `tests/test_svd_partial_corpus.c`
  and reusable helpers in `tests/test_svd_partial_shared_helpers.h`;
- oracle/report integration in `scripts/run_corpus_oracle.py` when generated
  rows are needed;
- documentation that names the exact fixture families and non-claims;
- no raw singular-vector identity, broad SVD/partial-SVD correctness,
  repeated-spectrum generality, convergence-rate, partial-result, performance,
  external-library parity, package/ABI, platform, or state-of-the-art claim.

## Oracle And Report Row Requirements

Generated oracle rows must live under ignored `build/corpus/oracle/` paths.
Report indexes and run manifests must live under ignored `build/corpus-reports/`
or `build/report-index/` paths unless a later sprint explicitly adopts a
source-controlled snapshot policy.

Each generated row must record:

- `oracle_row_id`;
- `fixture_key`;
- `solver_family`;
- operation and comparison kind;
- exact command;
- source commit and branch;
- generated timestamp;
- platform, compiler, and configuration;
- support tier;
- expected and observed result;
- tolerance kind and value;
- comparison status and failure class when relevant;
- skip/defer reason when relevant;
- claim scope and non-claims.

Pass rows may support only their fixture-local or selected fixture-family
claim scope. Skip, defer, unsupported, and xfail rows never count as pass
evidence.

## Optional Data Gate

Optional external data can participate only when:

- license and redistribution terms are reviewed;
- expected external location is documented;
- availability is explicit;
- skip/defer semantics are source-controlled;
- pass interpretation is fixture-local;
- skip interpretation states that unavailable data is not solver pass evidence;
- claim boundary excludes external-library parity and broad corpus
  completeness unless a later reviewed gate explicitly earns them.

Disabled optional data remains policy evidence only.

## Validation Checklist

Run after any corpus metadata, expected row, schema, oracle, or proof-owner
change:

```sh
python3 scripts/validate_corpus_schema.py
```

Run for QR corpus-family promotion:

```sh
make build/test_qr_corpus && ./build/test_qr_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
```

Run for partial-SVD corpus-family promotion:

```sh
make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus
python3 scripts/run_corpus_oracle.py --include-partial-svd
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --check-freshness
```

Run combined solver-family oracle/report checks when both QR and partial-SVD
families changed:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

Run the full C quality gate when any `.c` or `.h` file changes:

```sh
make format && make lint && make test
```

## Promotion Rules

1. A corpus-family claim needs at least two maintained fixtures in that family
   unless the sprint explicitly documents why one fixture closes the selected
   gap.
2. Every promoted fixture needs a source-controlled expected-result file.
3. Every generated fixture needs generator hashes and a regeneration command.
4. Every solver-backed claim needs a compiled proof owner.
5. Every generated-local claim needs oracle/report rows with command, commit,
   platform, compiler, configuration, support tier, claim scope, and
   non-claims.
6. Source-controlled metadata may prove corpus ownership, not solver pass.
7. Generated local rows remain local evidence unless a later hosted artifact
   policy promotes them.
8. Public and support docs must name the fixture-family boundary and preserve
   broad non-claims.

## Stop Conditions

- A proposed QR row requires raw QR basis equality, sign, orientation, or
  column order.
- A proposed partial-SVD row requires raw singular-vector identity for repeated
  or clustered spectra.
- A source-controlled expected row is cited as observed pass evidence.
- Optional-data skip/defer rows are counted as solver pass evidence.
- A fixture-family claim omits tolerances, support tier, proof owner, or
  validation command.
- Oracle/report rows omit command, commit, platform, compiler, configuration,
  support tier, claim scope, or non-claims.
- Documentation widens selected fixture-family proof into broad solver,
  external-library, platform, package, performance, or state-of-the-art claims.
- Required C quality gates fail after `.c` or `.h` changes.

## Sprint 150 And 151 Handoff

Sprint 150 should use this gate to choose QR fixture families and keep the
proof owner focused. Sprint 151 should reuse the same row and report structure
while applying partial-SVD-specific subspace-safe comparisons and
fail-closed/budget semantics.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 150 and Sprint 151 have reusable corpus gates. | Complete | QR and partial-SVD gates define row requirements, comparison semantics, proof owners, oracle/report rows, and validation commands. |
| Raw-basis and raw-vector identity claims are excluded. | Complete | QR and partial-SVD stop conditions reject raw basis/vector identity and require residual/projector/subspace-safe checks. |
| Generated rows remain correctly classified. | Complete | Oracle/report requirements and promotion rules classify generated rows as local evidence with command, commit, platform, compiler, configuration, support tier, claim scope, and non-claims. |
