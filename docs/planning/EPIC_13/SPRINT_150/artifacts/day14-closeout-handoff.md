# Sprint 150 Day 14: Closeout And Sprint 151 Handoff

## Purpose

Finalize Sprint 150 artifacts, record validation status and residuals, and
prepare a clean Sprint 151 handoff for partial-SVD maintained corpus family
expansion.

## Sprint 150 Closure Summary

Sprint 150 closed a bounded maintained QR corpus family beyond the single
Sprint 139 fixture. The closed family now includes:

- `qr_rank_deficient_6x4_nullspace_v1`
- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`
- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

The sprint added or aligned:

- source-controlled fixture and generator metadata;
- source-controlled expected-result TSV files;
- deterministic generator validation for the selected QR fixtures;
- QR oracle semantics for rank, nullity, nullspace residual, projector
  distance, minimum-norm status, residual, solution norm, and exact values;
- focused proof-owner coverage in `tests/test_qr_corpus.c`;
- local QR oracle/report generation and stale generated-output cleanup;
- normalized report-index interpretation for QR generated-local rows;
- README, algorithm, cookbook, maintainer, corpus, and oracle-schema
  documentation for the bounded QR claim.

## Validation Status

Day 13 ran the full integrated validation lane after Sprint 150 C changes:

```sh
python3 scripts/validate_corpus_schema.py
make build/test_qr_corpus && ./build/test_qr_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
python3 -m py_compile scripts/run_corpus_oracle.py scripts/validate_corpus_schema.py scripts/normalize_report_index.py
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check
make format && make lint && make test
git diff --check
```

All commands passed. The focused QR proof-owner test reported 14 tests, 0
failures, 0 skips, and 258 assertions.

Day 14 reran closeout checks:

```sh
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check
git diff --check
```

These checks passed. The oracle freshness command still reports expected
`generated_present_unchecked` warnings for generated-local rows and exits
successfully with freshness ok for 28 rows.

## Report And Claim Boundary

The maintained QR oracle command remains:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

The generated-local QR report surface remains bounded to:

- `oracle_row_count=26`
- `solver_qr_row_count=23`
- `partial_svd_row_count=0`
- `support_tier=local_only`
- six selected QR fixture keys

No generated `build/` oracle or report output is source-controlled. Generated
rows remain local evidence tied to the recorded command, commit, branch,
platform, compiler, configuration, support tier, and artifact path.

Sprint 150 does not claim:

- broad QR correctness;
- raw QR basis or raw nullspace basis identity;
- sign, orientation, scale, or column-order parity;
- global rank-threshold policy;
- broad rank-deficient solve behavior;
- broad minimum-norm or least-squares behavior;
- SVD-pseudoinverse global-oracle behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art status.

The Day 14 stale-reference scan found old `solver_qr_row_count=3` wording only
inside historical Sprint 150 baseline or validation artifacts that explicitly
describe the old count as historical. Current README, algorithm, cookbook,
maintainer, corpus, and oracle-schema guidance is aligned to the six-fixture
family and `solver_qr_row_count=23`.

## Residuals

Closed in Sprint 150:

- QR rank-deficient rectangular fixture metadata, expected rows, oracle rows,
  proof-owner tests, reports, and docs for two added fixtures plus the Sprint
  139 seed fixture.
- QR underdetermined minimum-norm fixture metadata, expected rows, oracle rows,
  proof-owner tests, reports, and docs for three selected fixtures.
- Generated-local report cleanup so stale ignored oracle/report outputs do not
  contaminate report-index normalization.

Deferred from Sprint 150:

- Reorder/COLAMD QR corpus promotion remains deferred because the evidence
  mixes residual/status, permutation, fill, optional SuiteSparse, and
  performance-adjacent semantics. It should be handled as a separate bounded
  product decision rather than folded into the Sprint 150 family.
- Broad rank-threshold policy remains unclaimed. Future work must define
  fixture-local or family-local rank semantics before promoting broader rank
  behavior.
- Strict generated-row freshness comparison remains pending. Current report
  freshness records generated-local rows as present but unchecked, with
  advisory warnings by design.
- No downstream package or platform proof was added by Sprint 150. The sprint
  validates QR corpus behavior, not packaging, ABI, Windows parity, or
  installed-consumer behavior.

## Sprint 151 Handoff

Sprint 151 should use the Sprint 150 QR corpus pattern for partial-SVD
maintained corpus expansion:

1. Select a small set of partial-SVD fixture families that can be fully closed,
   rather than partially expanding every candidate family.
2. Start from the Sprint 140 partial-SVD fixture-local closure and identify
   candidate families for repeated spectra, rank-deficient rectangular
   matrices, sparse low-rank output, and convergence/fail-closed behavior.
3. Define comparison semantics before writing rows. Use singular-value,
   projector, vector residual, ordering, tolerance, sparse-output, and
   convergence rules that avoid raw-vector identity claims.
4. Add source-controlled fixture, generator, expected, support-tier,
   claim-scope, and non-claim rows only after the comparison contract is clear.
5. Add focused partial-SVD corpus proof-owner tests instead of expanding a
   broad monolithic SVD lane.
6. Extend oracle/report generation and normalization after proof owners and
   expected rows are stable.
7. Align documentation with bounded fixture-local partial-SVD claims and keep
   platform, package, ABI, performance, and state-of-the-art claims out of
   scope unless separately proven.

Sprint 151 should reuse these Sprint 150 implementation lessons:

- reset generated-local oracle/report outputs before each oracle run;
- treat generated-local rows as local evidence, not source-controlled release
  artifacts;
- record stale-report warnings without turning them into broad correctness
  claims;
- keep current documentation searches separate from historical planning
  artifacts;
- run full C gates when `.c` or `.h` proof-owner changes are made.

## Retrospective Inputs

For the Sprint 150 retrospective:

- completed QR families: rank-deficient rectangular and underdetermined
  minimum-norm;
- primary proof owner: `tests/test_qr_corpus.c`;
- primary oracle/report owner: `scripts/run_corpus_oracle.py`;
- primary schema owner: `scripts/validate_corpus_schema.py`;
- documentation surfaces: `README.md`, `docs/algorithm.md`,
  `docs/cookbook.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md`,
  and `tests/corpus/schemas/oracle_fields.md`;
- validation anchor: Day 13 full integrated validation and Day 14 closeout
  checks.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Sprint 150 QR family closure is ready for retrospective. | Complete | Six-fixture QR corpus family, proof-owner tests, oracle/report rows, docs, and validation are recorded. |
| Residuals are explicit and assigned to later sprint candidates. | Complete | Reorder/COLAMD QR, strict generated-row freshness, and broader rank-threshold policy are deferred explicitly. |
| Branch is clean except for intentional Sprint 150 changes. | Complete | `git status --short` shows only Sprint 150 modifications and untracked Sprint 150/expected corpus files. |
| Downstream consumer proof boundary is clear. | Complete | Sprint 150 records no package, ABI, platform, or installed-consumer proof claim. |
| Sprint 151 partial-SVD handoff is prepared. | Complete | Handoff maps Sprint 151 to partial-SVD family selection, comparison contracts, metadata, proof owners, reports, docs, and validation. |
