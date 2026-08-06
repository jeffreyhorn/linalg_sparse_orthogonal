# Sprint 138 Retrospective

**Sprint:** 138 - Maintained Numerical Corpus Architecture
**Duration:** 14 days (Days 1-14 landed on branch `sprint-138`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 138 day-by-day plan, working notes, artifact directory,
      and closeout artifact.
- [x] Re-read Sprint 137 corpus/oracle evidence contracts, report freshness
      templates, public claim freeze, quality surface map, and Sprint 138
      readiness handoff.
- [x] Designed the maintained fixture taxonomy and selected the first durable
      QR rank-deficient fixture lane.
- [x] Designed and implemented the maintained corpus layout under
      `tests/corpus/`, including fixture, generator, optional-data,
      expected-result, schema, README, and future fixture paths.
- [x] Defined durable observed oracle row fields, comparison kinds, tolerance
      kinds, statuses, failure classes, support-tier fields, claim scopes, and
      non-claim fields.
- [x] Added `scripts/validate_corpus_schema.py` to validate corpus TSV shape,
      required references, selected enums, generator hashes, expected rows,
      and false-pass guardrails.
- [x] Added the first deterministic generated fixture lane:
      `qr_rank_deficient_6x4_nullspace_v1`.
- [x] Added `scripts/run_corpus_oracle.py` to emit local observed oracle rows,
      report indexes, skip/defer rows, and run manifests under ignored
      `build/` paths.
- [x] Added optional external-data skip/defer semantics without converting
      unavailable optional data into pass evidence.
- [x] Published maintainer documentation for corpus ownership, row
      interpretation, stale-report assumptions, and Sprint 139 QR handoff.
- [x] Published final Sprint 138 deliverable checklist, validation summary,
      residual register, and Sprint 139 QR readiness criteria.
- [x] Ran Sprint 138 final validation:
  - `python3 -B scripts/validate_corpus_schema.py`;
  - `env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B
    scripts/run_corpus_oracle.py`;
  - generated report split check;
  - `git diff --check`;
  - trailing-whitespace scan under Sprint 138 docs, `tests/corpus`, and
    touched scripts;
  - focused Markdown local link/path validation under `docs/planning/EPIC_12`;
  - corpus TSV width consistency check;
  - changed/untracked `.c` and `.h` scan.
- [x] No `.c` or `.h` files changed, so the full
      `make format && make lint && make test` gate was not required.

## What Went Well

1. **The sprint converted evidence templates into maintained repository
   structure.** Sprint 137's corpus/oracle contracts became concrete
   `tests/corpus/` manifests, schemas, expected-result rows, validators,
   documentation, and generated-output boundaries.

2. **The first lane is deterministic and reproducible.** The selected
   `qr_rank_deficient_6x4_nullspace_v1` lane records dimensions, nonzeros,
   rank, nullity, null-vector direction, generator metadata, canonical text
   policy, and SHA-256 structure/value hashes.

3. **Observed evidence is separated from source-controlled targets.**
   Expected-result rows live in source control, while observed oracle rows,
   report indexes, skip/defer rows, and run manifests are generated under
   ignored `build/` paths.

4. **Skip/defer behavior is explicit.** Optional external data defaults to a
   disabled skip/defer path. The oracle/report command records skip rows, and
   validation confirms optional rows are not counted as solver pass evidence.

5. **The Sprint 139 QR handoff is actionable.** The retrospective and Day 13
   and Day 14 artifacts give Sprint 139 the fixture key, generator key, rank,
   nullity, null-vector direction, oracle row IDs, normalized residual
   tolerance, validation commands, and claim boundaries.

## What Didn't Go Well

1. **The first implementation lane still uses reference-side oracle evidence.**
   Sprint 138 validates the corpus/oracle architecture and generated reference
   rows. It does not close solver-backed QR behavior against the fixture; that
   remains Sprint 139 work.

2. **Report freshness is documented but not enforced.** Generated report
   fields carry commit, command, platform, compiler, configuration, status,
   support tier, claim scope, and generated path context, but stale-report
   normalization and diagnostics remain Sprint 141 work.

3. **The corpus is intentionally narrow.** A single deterministic QR lane is
   the right bootstrap, but it does not prove broad corpus completeness,
   SuiteSparse parity, partial-SVD behavior, external-data behavior, platform
   parity, or state-of-the-art status.

4. **The sprint added Python tooling without promoting it into the global
   quality gate.** The focused script checks passed, but the new corpus
   commands are still maintained as local commands rather than integrated into
   every CI lane.

## Final Metrics

### Validation

| Metric | Sprint 138 close state |
|---|---|
| tracked `.c`/`.h` changes | 0 |
| `python3 -B scripts/validate_corpus_schema.py` | passed |
| `env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py` | passed |
| generated oracle pass rows | 3 |
| generated optional-data skip rows | 1 |
| generated optional-data pass rows | 0 |
| `git diff --check` | passed |
| Sprint 138 trailing-whitespace scan | passed |
| focused Epic 12 Markdown local link/path validation | passed |
| corpus TSV width consistency check | passed |
| full C quality gate | not required; no `.c`/`.h` changes |

### Corpus Package

| Metric | Sprint 138 close state |
|---|---:|
| source-controlled fixture rows | 1 |
| source-controlled generator rows | 1 |
| source-controlled optional-data rows | 1 |
| source-controlled first-lane expected-result rows | 3 |
| observed oracle rows generated locally | 3 |
| report-index rows generated locally | 4 |
| generated run manifests | 1 |
| daily artifacts under `SPRINT_138/artifacts/` | 14 |
| final retrospective files | 1 |

## Sprint 139 Readiness

Sprint 139 should consume the first QR lane exactly as defined:

| Field | Value |
| --- | --- |
| fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| generator key | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| shape | 6 rows by 4 columns |
| nonzeros | 14 |
| expected rank | 3 |
| expected nullity | 1 |
| null vector direction | `[-1, -1, 0, 1]` |
| rank row ID | `qr_rank_deficient_6x4_nullspace_v1_rank` |
| nullity row ID | `qr_rank_deficient_6x4_nullspace_v1_nullity` |
| residual row ID | `qr_rank_deficient_6x4_nullspace_v1_projector_residual` |
| initial normalized null-vector residual tolerance | `1e-10` |

QR closure for this first lane should compare the normalized residual of the
fixed null-vector direction. Later solver-backed QR work may add projector or
two-way projection-distance rows, but it should not require raw QR basis
equality. Any support-tier promotion beyond local must be backed by reviewed
generated evidence for that lane.

## Residual Deferred Debt

Most important carry-forward work:

- close solver-backed QR behavior against
  `qr_rank_deficient_6x4_nullspace_v1` in Sprint 139;
- promote corpus/oracle evidence beyond local only after reviewed hosted-lane
  proof exists;
- add partial-SVD clustered and repeated singular-value fixture lanes in
  Sprint 140;
- normalize report freshness and stale-report diagnostics in Sprint 141;
- define optional external-data availability, provenance, and reviewed pass
  policy before any external-data pass claim;
- keep public adoption wording tied to generated evidence without widening
  claim scope.

Still consciously constrained rather than silently solved:

- no broad corpus completeness claim;
- no broad QR correctness claim;
- no raw QR basis parity claim;
- no global least-squares or minimum-norm claim;
- no broad partial-SVD correctness claim;
- no SuiteSparse or external corpus parity claim;
- no package, ABI, shared-library, loader, or package-manager support claim;
- no platform parity claim;
- no portable performance, coverage-completeness, release-readiness, or
  state-of-the-art claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-corpus-contract-setup.md](./artifacts/day1-scope-corpus-contract-setup.md)
- [day2-fixture-taxonomy-draft.md](./artifacts/day2-fixture-taxonomy-draft.md)
- [day3-taxonomy-review-claim-boundaries.md](./artifacts/day3-taxonomy-review-claim-boundaries.md)
- [day4-corpus-storage-layout-design.md](./artifacts/day4-corpus-storage-layout-design.md)
- [day5-corpus-storage-layout-implementation.md](./artifacts/day5-corpus-storage-layout-implementation.md)
- [day6-oracle-row-schema-design.md](./artifacts/day6-oracle-row-schema-design.md)
- [day7-oracle-schema-implementation.md](./artifacts/day7-oracle-schema-implementation.md)
- [day8-deterministic-fixture-lane-design.md](./artifacts/day8-deterministic-fixture-lane-design.md)
- [day9-first-corpus-lane-implementation.md](./artifacts/day9-first-corpus-lane-implementation.md)
- [day10-maintained-oracle-report-command.md](./artifacts/day10-maintained-oracle-report-command.md)
- [day11-optional-data-skip-semantics.md](./artifacts/day11-optional-data-skip-semantics.md)
- [day12-focused-validation-quality-gates.md](./artifacts/day12-focused-validation-quality-gates.md)
- [day13-documentation-sprint139-handoff.md](./artifacts/day13-documentation-sprint139-handoff.md)
- [day14-closeout-validation-summary.md](./artifacts/day14-closeout-validation-summary.md)

## Closeout

Sprint 138 is complete. It closes the maintained numerical corpus
architecture sprint with a source-controlled corpus skeleton, deterministic
first QR lane, oracle row schema, validation helper, maintained oracle/report
command, optional-data skip/defer semantics, final validation evidence,
residual register, and Sprint 139 QR handoff. It does not change C source or
widen public claims beyond fixture-local corpus/oracle evidence.
