# Sprint 138 Day 13: Documentation & Sprint 139 Handoff

## Purpose

Day 13 turns the implemented corpus/oracle lane into maintainer-facing
documentation and a Sprint 139 handoff. The goal is to let maintainers extend
the corpus without redefining row semantics, counting generated skip/defer rows
as solver passes, or widening fixture-local claims.

## Maintainer Documentation Updates

Updated `tests/corpus/README.md` to document:

- corpus layout and generated-output boundaries
- ownership for fixture manifests, generator manifests, optional-data rows,
  expected-result rows, schema docs, validation scripts, oracle/report scripts,
  and ignored generated report paths
- interpretation rules for manifest rows, generator rows, expected-result
  rows, observed oracle rows, optional-data skip/defer rows, and report-index
  rows
- stale-report assumptions and the conditions that require regeneration
- Sprint 139 QR fixture prerequisites
- remaining corpus/oracle residuals

The README now also removes the outdated statement that the first lane still
needs the Day 10 oracle/report command. The command exists, emits generated
rows under `build/`, and produces fixture-local evidence only.

## Row Ownership

| Surface | Owner | Day 13 rule |
| --- | --- | --- |
| `tests/corpus/manifests/fixtures.tsv` | Corpus maintainer, with solver-owner review for numerical semantics. | Fixture rows define eligible evidence lanes, not observed pass evidence. |
| `tests/corpus/manifests/generators.tsv` | Corpus maintainer. | Generator rows define deterministic reproduction metadata and hash expectations. |
| `tests/corpus/manifests/optional_data.tsv` | Corpus maintainer. | Optional external-data rows define skip/defer policy and must not be counted as solver pass evidence. |
| `tests/corpus/expected/*.tsv` | Corpus maintainer for schema; solver owner for numerical meaning. | Expected-result rows define target comparisons and remain prerequisites until generated observed rows exist. |
| `tests/corpus/schemas/*.md` | Corpus and report maintainers. | Row semantics live here; semantic changes require matching validator and migration updates. |
| `scripts/validate_corpus_schema.py` | Corpus maintainer. | Enforces row shape, references, selected enums, hashes, and false-pass guardrails. |
| `scripts/run_corpus_oracle.py` | Corpus maintainer. | Emits ignored local oracle rows, skip/defer rows, report indexes, and run manifests. |
| `build/corpus/`, `build/corpus-reports/` | Local runner. | Generated outputs stay uncommitted unless a later sprint explicitly promotes them. |

## Row Interpretation Guidance

- Fixture manifest rows define what a fixture is allowed to test.
- Generator rows prove reproducibility metadata, not solver correctness.
- Expected-result rows define targets, not observed results.
- Observed oracle rows are pass evidence only when
  `comparison_status=pass` and only for the exact fixture, command, commit,
  platform, compiler, configuration, support tier, tolerance, and claim scope
  recorded in the row.
- Optional-data skip/defer rows are policy evidence only.
- Report-index rows preserve row meaning and generated-output location; they
  do not prove release readiness, broad correctness, coverage completeness, or
  state-of-the-art status.

## Stale Report Assumptions

Generated oracle rows and report indexes become stale when any of these inputs
change:

- source commit or branch
- fixture, generator, optional-data, or expected-result row
- generator algorithm, parameters, canonical text, or hash
- validator or oracle command behavior
- optional-data configuration or availability
- compiler, platform, or build configuration
- support tier, claim scope, tolerance, or non-claim wording

Sprint 141 should normalize freshness checks using the current report index
and manifest fields for commit, command, platform, compiler, configuration,
support tier, status, claim scope, generated path, and non-claims. It should
preserve the current skip/defer boundary.

## Sprint 139 QR Handoff Requirements

Sprint 139 should use the first lane:

- fixture key: `qr_rank_deficient_6x4_nullspace_v1`
- generator key: `qr_rank_deficient_6x4_nullspace_generator_v1`
- shape: 6 rows by 4 columns
- nonzeros: 14
- expected rank: 3
- expected nullity: 1
- null vector direction: `[-1, -1, 0, 1]`
- oracle rows:
  - `qr_rank_deficient_6x4_nullspace_v1_rank`
  - `qr_rank_deficient_6x4_nullspace_v1_nullity`
  - `qr_rank_deficient_6x4_nullspace_v1_projector_residual`
- initial normalized null-vector residual tolerance: `1e-10`

QR closure for this first lane should compare the normalized residual of the
fixed null-vector direction. Later solver-backed QR work may add projector or
two-way projection-distance rows, but it should not require raw QR basis
equality because sign, scaling, and equivalent basis choices can differ while
representing the same subspace.

Minimum Sprint 139 validation before claiming fixture closure:

```sh
python3 scripts/validate_corpus_schema.py
python3 scripts/run_corpus_oracle.py
```

Hosted or reviewed support-tier promotion requires reviewed generated evidence
from that lane. The current first-lane evidence remains local unless a later
run records and reviews a broader support tier.

## Residual Register

| Residual | Owner sprint | Boundary |
| --- | --- | --- |
| Reviewed hosted-platform corpus/oracle promotion | Sprint 139 or later | Local generated rows are not hosted evidence. |
| Solver-backed QR fixture closure | Sprint 139 | Sprint 138 provides fixture facts and oracle semantics only. |
| Partial-SVD clustered/repeated singular-value lanes | Sprint 140 | No SVD correctness claim is earned by this QR lane. |
| Report freshness normalization and stale diagnostics | Sprint 141 | Day 13 documents assumptions but does not implement freshness enforcement. |
| Optional external-data pass policy | Later Epic 12 sprint | Optional rows remain disabled, skipped, or deferred by default. |
| Public adoption wording tied to corpus evidence | Later Epic 12 sprint | Public claims must remain fixture-local unless backed by reviewed evidence. |

## Non-Claims

Day 13 does not claim broad QR correctness, raw QR basis parity, global
least-squares or minimum-norm behavior, broad partial-SVD correctness,
SuiteSparse parity, external corpus parity, package/platform support, portable
performance, coverage completeness, release readiness, or state-of-the-art
status.

## Completion Criteria

- Maintainers can update corpus rows without redefining row semantics.
- Sprint 139 has clear QR fixture and oracle prerequisites.
- Documentation preserves fixture-local claim boundaries and skip/defer
  semantics.
