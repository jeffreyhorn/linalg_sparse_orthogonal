# Day 7 Corpus And Oracle Index Integration

## Purpose

Day 7 connects maintained corpus and oracle rows to the normalized report
index. The Day 6 generator could emit contract rows and generated-artifact
presence rows. Day 7 adds native row preservation for corpus manifests,
expected-result rows, optional-data policy, and generated oracle TSV rows.

Generated oracle outputs remain ignored local evidence. The normalized index
can read them when present, but it does not commit them or promote them into
broad solver, platform, package, performance, or external-library claims.

## Implemented Surfaces

| Surface | Change | Purpose |
| --- | --- | --- |
| `scripts/normalize_report_index.py` | Added native corpus and oracle row emitters. | Preserves fixture keys, generator keys, expected oracle IDs, optional-data skip/defer rows, QR generated-reference rows, and partial-SVD generated rows. |
| `tests/test_normalize_report_index.py` | Added corpus/oracle integration checks. | Verifies source-controlled corpus rows and generated QR/partial-SVD oracle rows survive normalization with claim boundaries. |
| `docs/planning/EPIC_12/SPRINT_141/artifacts/day7-corpus-oracle-index-integration.md` | Added this integration artifact. | Records Day 7 behavior, validation, and handoff. |
| `docs/planning/EPIC_12/SPRINT_141/WORKING_NOTES.md` | Updated Day 7 notes. | Keeps sprint evidence current. |

## Native Row Mapping

| Input | Normalized row behavior |
| --- | --- |
| `tests/corpus/manifests/fixtures.tsv` | Emits `corpus_fixture_<fixture_key>_v1` rows with fixture key, generator key, shape, rank status, support tier, claim scope, and non-claims. |
| `tests/corpus/manifests/generators.tsv` | Emits `corpus_generator_<generator_key>_v1` rows with generator version, algorithm, seed, canonical format, change policy, and regeneration command. |
| `tests/corpus/manifests/optional_data.tsv` | Emits `corpus_optional_<optional_data_key>_v1` rows with `skip`, `defer`, or advisory status and optional-data claim boundaries. |
| `tests/corpus/expected/*.tsv` | Emits `corpus_expected_<oracle_row_id>_v1` rows preserving fixture key, operation, comparison kind, expected result kind, tolerance, claim scope, and non-claims. |
| `build/corpus/oracle/*.tsv` | Emits `oracle_<oracle_row_id>_<artifact>_v1` rows when generated oracle reports are present. Native oracle row IDs remain in `native_row_id`. |

## QR And Partial-SVD Preservation

The focused test path runs:

```sh
python3 scripts/run_corpus_oracle.py \
  --include-partial-svd \
  --oracle-dir <temp-build>/corpus/oracle \
  --report-dir <temp-build>/corpus-reports

python3 scripts/normalize_report_index.py \
  --build-root <temp-build> \
  --family oracle \
  --output <temp>/oracle-index.tsv
```

The normalized output preserves:

- QR generated-reference row IDs such as
  `qr_rank_deficient_6x4_nullspace_v1_rank`;
- partial-SVD generated-reference row IDs such as
  `partial_svd_clustered_repeated_diag8x6_k3_v1_singular_values`;
- `comparison_status=pass` as normalized `status=pass` only for generated
  oracle rows that already reported pass;
- fixture keys, solver family, operation, comparison kind, command, source
  revision, platform, compiler, support tier, claim scope, and non-claims;
- `freshness_status=generated_present_unchecked` until Sprint 141 Day 10/11
  stale-report gates implement freshness comparison.

## Skip And Defer Preservation

Optional-data rows are normalized from source-controlled policy rather than
generated pass output. Disabled optional external data becomes `status=skip`
with the source-controlled skip reason and claim boundary. Deferred optional
or runtime/backend governance rows remain `status=defer`.

## Duplicate Row Handling

Generated oracle rows can appear in multiple local ignored artifacts, such as
a QR-only oracle output and a combined corpus oracle output. Day 7 keeps
`native_row_id` equal to the original oracle row ID and includes the artifact
path in the normalized `row_id` so local overlapping generated files do not
collide.

## Validation Evidence

Commands run:

```sh
python3 -m py_compile scripts/validate_corpus_schema.py scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --no-generated --check
python3 scripts/normalize_report_index.py --check
python3 scripts/validate_corpus_schema.py
```

Results:

- focused generator tests passed;
- source-controlled smoke check reported `42` normalized rows;
- default generated-artifact discovery check reported `54` normalized rows in
  the current local worktree;
- corpus schema validation passed.

## Claim Boundaries Preserved

- Source-controlled fixture, generator, optional-data, and expected-result rows
  remain advisory or skip/defer policy rows unless an observed generated row
  supplies pass/fail status.
- QR rows remain fixture-local to the maintained rank-deficient QR fixture.
- Partial-SVD rows remain fixture-local to the generated clustered/repeated
  8x6 diagonal fixture.
- Optional-data skip rows are not solver pass evidence.
- Generated oracle rows are local evidence tied to command, source revision,
  platform, compiler, configuration, support tier, artifact path, and native
  row ID.
- Freshness comparison remains deferred to Sprint 141 Day 10/11.

## Day 8 Handoff

Day 8 should extend the same pattern to benchmark, performance sentinel, and
large-matrix guardrail outputs. Those rows need local-measurement boundaries
and runtime/backend defers to remain explicit.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 138-140 rows are represented without changing their claim scope. | Complete | Native corpus fixture, generator, expected, optional-data, QR, and partial-SVD rows preserve claim scope and non-claims in normalized output. |
| Generated oracle outputs are not committed as release proof. | Complete | Tests generate oracle TSVs only under temporary build roots; local `build/` output remains ignored. |
| Corpus validation and normalized index checks agree on row identity. | Complete | `python3 scripts/validate_corpus_schema.py`, generator tests, `--no-generated --check`, and default `--check` all passed. |
