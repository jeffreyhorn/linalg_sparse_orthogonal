# Sprint 150 Day 10: Report Integration Design

## Purpose

Design the QR oracle/report generation and normalized report-index updates for
the Sprint 150 selected fixture families. Day 10 is a design day: it defines
the source-controlled ownership and implementation rules for Day 11 without
checking generated `build/` outputs into the repository.

## Existing Report Surface

The maintained report contracts already provide the required families in
`tests/corpus/manifests/report_families.tsv`:

| Report Family | Subfamily | Row Meaning | Artifact Pattern | Current Interpretation |
| --- | --- | --- | --- | --- |
| `corpus` | `fixtures` | `fixture_metadata` | `tests/corpus/manifests/fixtures.tsv` | Source-controlled fixture identity and eligibility metadata. |
| `corpus` | `generators` | `generator_metadata` | `tests/corpus/manifests/generators.tsv` | Source-controlled generator commands and deterministic hashes. |
| `corpus` | `expected` | `expected_result` | `tests/corpus/expected/*.tsv` | Source-controlled expected rows, tolerances, claim scopes, and non-claims. |
| `oracle` | `generated_reference` | `observed_oracle_comparison` | `build/corpus/oracle/*.tsv` | Generated-local non-solver reference comparison rows. |
| `oracle` | `solver_backed` | `solver_backed_fixture_proof` | `build/corpus/oracle/*.tsv` | Generated-local solver-backed rows for named fixtures and commands only. |
| `report_index` | `missing_generated` | `not_generated` | `build/report-index/normalized-index.tsv` | Advisory rows for absent generated-local reports. |

The existing contracts are sufficient for Sprint 150. Day 11 should not add a
new report family unless a validation failure proves the current
`corpus`/`oracle` split cannot express the QR rows.

## Generated Output Design

Day 11 should preserve the existing command as the maintained local generation
entry point:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

The command should generate local-only rows under:

- `build/corpus/oracle/*.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`

The selected Sprint 150 QR rows should remain solver-backed rows with:

- `solver_family=qr`
- `support_tier=local_only`
- `command=scripts/run_corpus_oracle.py --include-solver-qr`
- `proof_owner=runtime_qr_probe` in the configuration field
- fixture-local `structure_hash` and `value_hash`
- fixture-local claim scopes and non-claims carried from expected rows

The generated report surface should expose the following fixture families:

| Fixture | Operation Family | Expected Solver-Backed Rows |
| --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1` | `rankdef_nullspace` seed | rank, nullity, nullspace residual |
| `qr_rankdef_duplicate_5x4_v1` | `rankdef_nullspace` | rank, nullity, nullspace residual, nullspace subspace |
| `qr_rankdef_dependent_row_4x3_v1` | `rankdef_nullspace` | rank, nullity, nullspace residual, nullspace subspace |
| `qr_underdetermined_minnorm_2x4` | `minnorm_solve` | status, residual, solution norm, solution values |
| `qr_minnorm_3x6_exact_values` | `minnorm_solve` | status, residual, solution norm, solution values |
| `qr_minnorm_5x10_exact_values` | `minnorm_solve` | status, residual, solution norm, solution values |

Expected total Sprint 150 QR solver-backed rows after local generation: `23`.

## Report Index Design

Day 11 should validate that normalized report-index generation produces:

- source-controlled `corpus_fixture_*` rows for all selected QR fixtures;
- source-controlled `corpus_generator_*` rows for all selected QR generators;
- source-controlled `corpus_expected_*` rows for all selected QR expected
  results;
- generated-local `oracle_*` rows for the `23` solver-backed QR observations;
- no generated-local QR row with `support_tier` broader than `local_only`;
- no generated-local QR row with a broad QR, external-library, platform,
  package, ABI, performance, or state-of-the-art claim.

The normalized row IDs are produced by `scripts/normalize_report_index.py`.
Day 11 should not hand-author normalized index output. The implementation
should instead update source-controlled manifests or generator behavior only
when the generated rows are missing, ambiguous, or incorrectly scoped.

## Freshness Rules

Generated QR oracle/report rows are local evidence, not source-controlled pass
evidence. The source-controlled owners are:

- fixture rows in `tests/corpus/manifests/fixtures.tsv`;
- generator rows in `tests/corpus/manifests/generators.tsv`;
- expected rows in `tests/corpus/expected/*.tsv`;
- report-family contracts in `tests/corpus/manifests/report_families.tsv`;
- proof-owner tests in `tests/test_qr_corpus.c`;
- generation and normalization scripts in `scripts/run_corpus_oracle.py` and
  `scripts/normalize_report_index.py`.

Freshness expectations:

| Surface | Freshness Interpretation |
| --- | --- |
| Source-controlled corpus rows | Governed by schema validation and Git review. |
| `build/corpus/oracle/*.tsv` | Fresh only for the local command, commit, branch, platform, compiler, and configuration recorded in the row. |
| Normalized generated-local oracle rows | Advisory unless produced during the same validation run. |
| Missing generated rows | Explicit advisory gaps, not pass evidence. |
| Hosted CI logs | Not part of Sprint 150 QR report freshness unless a future reviewed lane adds them. |

The current freshness checker may report advisory
`generated_present_unchecked` warnings for generated-local oracle rows. Day 11
should treat those warnings as acceptable only when the command exits `0` and
the artifact records the warnings explicitly.

## Non-Claim Wording

Report and index rows for Sprint 150 QR must retain these non-claims:

- no broad QR correctness;
- no raw QR basis or raw nullspace basis identity;
- no sign, orientation, scale, or column-order parity;
- no global rank-threshold policy;
- no broad rank-deficient solve claim;
- no broad minimum-norm or least-squares behavior;
- no SVD-pseudoinverse global-oracle claim;
- no inconsistent-system behavior claim;
- no external-library parity;
- no platform, package, ABI, performance, or state-of-the-art claim.

Rank-deficient rows should use the rank/nullity/nullspace/subspace-specific
wording from the expected rows. Minimum-norm rows should use the
underdetermined minimum-norm wording from the expected rows and must not imply
rank-deficient minimum-norm recovery or broad pseudoinverse parity.

## Documentation Update Map

Day 12 should use the Day 11 implementation evidence to update these surfaces:

| Surface | Required Reference |
| --- | --- |
| `README.md` | Name the selected QR corpus family only if Day 11 report/index validation passes. |
| `docs/maintainer_guide.md` | Update the QR oracle command interpretation, row counts, fixture list, and generated-local freshness boundary. |
| corpus documentation or sprint artifacts | Link fixture rows, expected rows, proof-owner tests, oracle command, and report-index checks. |
| solver-selection documentation | Keep QR claims bounded to named fixtures and selected operations only. |
| Sprint 150 retrospective | Record generated-local evidence, non-claims, and Day 11 validation results. |

Day 12 should not cite generated-local rows as release or hosted-platform proof.

## Day 11 Implementation Checklist

1. Run `python3 scripts/run_corpus_oracle.py --include-solver-qr`.
2. Confirm the generated manifest reports the selected six QR fixture keys and
   `solver_qr_row_count=23`.
3. Run
   `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`.
4. Run
   `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check`.
5. Inspect generated-local warnings and record them without broadening claims.
6. Update source-controlled report contracts or generator behavior only if the
   selected QR rows are missing, incorrectly grouped, or incorrectly scoped.
7. Write the Day 11 implementation artifact with exact command outcomes.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Report integration has source-controlled owners. | Complete | Owners are fixture, generator, expected, report-family, proof-owner test, oracle script, and normalization script files. |
| Report rows do not imply freshness without generated evidence. | Complete | Design keeps generated rows under `build/`, requires recorded command/commit/platform/compiler/configuration, and treats missing rows as advisory gaps. |
| Normalized index changes are planned before implementation. | Complete | Day 11 checklist defines the exact normalization checks and update conditions. |
