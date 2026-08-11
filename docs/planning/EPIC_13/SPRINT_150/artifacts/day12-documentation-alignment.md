# Sprint 150 Day 12: Documentation Alignment

## Purpose

Align corpus documentation, solver-selection wording, README guidance,
cookbook/tutorial-style guidance, and maintainer guidance with the Sprint 150
QR corpus family boundary.

## Documentation Updates

Updated user-facing QR guidance in `README.md`:

- replaced the one-fixture QR corpus description with the maintained
  Sprint 139/Sprint 150 QR family;
- named the six fixture-local QR corpus rows;
- recorded that a current local QR oracle run reports `23` solver-backed QR
  rows;
- preserved non-claims for raw QR basis parity, broad rank-threshold policy,
  broad rank-deficient solve behavior, broad minimum-norm behavior,
  external-library parity, platform, performance, and state-of-the-art
  evidence.

Updated workflow guidance in `docs/cookbook.md`:

- changed the QR solver-selection note from the single Sprint 139 fixture to
  the bounded Sprint 139/Sprint 150 family;
- kept the wording fixture-local and non-comparative.

Updated algorithm documentation in `docs/algorithm.md`:

- expanded maintained QR corpus coverage from the 6x4 seed to the six-fixture
  maintained family;
- named the proof owner, oracle command, `23` solver-backed local rows, and
  residual/subspace-safe interpretation;
- kept raw basis identity and broad QR claims out of scope.

Updated maintainer guidance in `docs/maintainer_guide.md`:

- renamed the QR corpus section from Sprint 139-only maintenance to
  Sprint 139/Sprint 150 QR corpus maintenance;
- listed the selected Sprint 150 fixture keys;
- updated focused proof expectations to `14` passing `test_qr_corpus` tests;
- updated generated-local report interpretation to `26` oracle rows,
  `23` solver-backed QR rows, six fixture keys, and
  `partial_svd_row_count=0` for QR-only runs;
- documented that `scripts/run_corpus_oracle.py` clears stale generated
  oracle/report outputs before writing the current run.

Updated corpus documentation in `tests/corpus/README.md` and
`tests/corpus/schemas/oracle_fields.md`:

- added the Sprint 150 rank-deficient rectangular and underdetermined
  minimum-norm fixture families;
- documented rank/nullity/nullspace residual/subspace and
  status/residual/norm/value expected-row families;
- updated generated-local QR output expectations from `3` to `23`
  solver-backed QR rows;
- made the maintained QR command explicit:
  `python3 scripts/run_corpus_oracle.py --include-solver-qr`.

## Documentation Boundary

The aligned documentation supports only:

- named fixture-local rank/nullity evidence;
- named fixture-local nullspace residual and projector/subspace evidence;
- named fixture-local underdetermined minimum-norm status, residual, solution
  norm, and exact-value evidence;
- local generated report interpretation tied to command, commit, branch,
  platform, compiler, configuration, support tier, and artifact path.

The documentation still rejects:

- broad QR correctness;
- raw QR basis or raw nullspace basis identity;
- sign, orientation, scale, or column-order parity;
- global rank-threshold policy;
- broad rank-deficient solve behavior;
- broad minimum-norm or least-squares behavior;
- SVD-pseudoinverse global-oracle behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art claims.

## Stale-Claim Search Results

Focused stale-row-count search across current user/maintainer/corpus docs found
no remaining current-doc references to:

- `solver_qr_row_count=3`
- three solver-backed QR rows as the current maintained QR corpus lane
- a Sprint 139-only QR corpus maintenance section
- four passing `test_qr_corpus` tests as the current proof expectation

The only remaining `solver_qr_row_count=3` and three-row hits are historical
Sprint 139 planning artifacts under `docs/planning/EPIC_12/SPRINT_139/` and
the Sprint 150 Day 1 baseline artifact. Those remain historical records rather
than current guidance.

Focused non-claim searches confirmed current docs retain explicit wording for
broad QR, raw-basis, external-library, platform/package/ABI, performance, and
state-of-the-art non-claims.

## Validation

Day 12 validation commands:

```sh
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
git diff --check
```

Trailing-whitespace scans were also run over the touched documentation and
Sprint 150 planning files.

No `.c` or `.h` files were modified on Day 12, so the full C quality gate was
not required for this day.

## Day 13 Handoff

Day 13 should run integrated schema, focused QR proof-owner, oracle/report, and
documentation checks. Because Sprint 150 has earlier C changes, Day 13 should
decide whether to rerun the full C gate as part of integrated validation even
if Day 12 itself was documentation-only.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Docs name the selected QR family scope accurately. | Complete | README, cookbook, algorithm docs, maintainer guide, corpus README, and oracle schema docs name the six-fixture Sprint 139/Sprint 150 QR family. |
| Raw-basis identity and broad QR claims remain non-claims. | Complete | Updated docs retain explicit raw-basis, broad QR, external-library, platform, package, ABI, performance, and state-of-the-art non-claims. |
| Maintainer guidance identifies proof owners and future update rules. | Complete | Maintainer and corpus docs identify `tests/test_qr_corpus.c`, `scripts/run_corpus_oracle.py --include-solver-qr`, generated-local report counts, stale-report rules, and source-controlled owners. |
