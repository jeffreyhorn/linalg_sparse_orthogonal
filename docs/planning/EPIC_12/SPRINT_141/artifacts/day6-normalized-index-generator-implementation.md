# Day 6 Normalized Index Generator Implementation

## Purpose

Day 6 implements the first normalized report index generator. The generator
reads the Day 5 report-family contract, emits deterministic TSV rows, makes
missing generated reports explicit, and keeps unsupported or deferred areas as
non-pass evidence.

This implementation is intentionally standalone. It does not add Makefile
integration, freshness failure gates, or deep per-family native row parsing.
Those remain scheduled for later Sprint 141 days.

## Implemented Surfaces

| Surface | Change | Purpose |
| --- | --- | --- |
| `scripts/normalize_report_index.py` | Added standalone generator CLI. | Emits normalized report index rows from `tests/corpus/manifests/report_families.tsv`. |
| `tests/test_normalize_report_index.py` | Added focused Python tests. | Verifies stable output, family filtering, missing generated rows, required-generated failure behavior, generated-artifact presence, and deferred rows. |
| `docs/planning/EPIC_12/SPRINT_141/artifacts/day6-normalized-index-generator-implementation.md` | Added this implementation artifact. | Records Day 6 behavior, validation, and Day 7 handoff. |
| `docs/planning/EPIC_12/SPRINT_141/WORKING_NOTES.md` | Updated Day 6 notes. | Keeps sprint evidence current. |

## CLI

Default local use:

```sh
python3 scripts/normalize_report_index.py \
  --output build/report-index/normalized-index.tsv
```

Deterministic source-controlled smoke path:

```sh
python3 scripts/normalize_report_index.py \
  --no-generated \
  --output build/report-index/normalized-index.tsv
```

Supported options:

| Option | Behavior |
| --- | --- |
| `--corpus-root <path>` | Reads report-family contracts from an alternate corpus root. |
| `--build-root <path>` | Discovers generated `build/` artifacts under an alternate root. |
| `--output <path>` | Writes the normalized TSV to the requested path. |
| `--family <name>` | Restricts output to one report family; repeatable. |
| `--include-generated` | Includes generated artifact presence rows when artifacts exist. This is the default. |
| `--no-generated` | Emits contract rows and explicit `not_generated` rows without reading generated artifacts. |
| `--require-generated <family>` | Makes missing generated artifacts for the named family a nonzero check result. |
| `--check` | Validates and prints row count or missing-required diagnostics without rewriting output. |
| `--format tsv` | Pins the initial output format. |

## Output Contract

The generator emits the Day 4 normalized field order:

```text
row_id
report_family
subfamily
native_row_id
row_origin
row_meaning
status
status_reason
support_tier
claim_scope
non_claims
generator_command
source_commit
source_branch
generated_at_utc
platform
compiler
configuration
artifact_path
freshness_status
freshness_reason
skip_or_defer_reason
```

Rows are sorted by `report_family`, `subfamily`, `row_origin`,
`row_meaning`, `native_row_id`, `artifact_path`, and `row_id`.

## Row Behavior

| Situation | Generator behavior |
| --- | --- |
| Source-controlled contract row | Emits one `report_contract_*_v1` row with the contract status and non-claim boundaries. |
| Generated artifact pattern with no local artifact | Emits one `report_missing_*_v1` row with `freshness_status=not_generated`. |
| Generated artifact pattern with a local artifact | Emits one `report_artifact_*_v1` row with `freshness_status=generated_present_unchecked`. |
| `--no-generated` | Emits `not_generated` rows for generated artifact patterns even if local files exist. |
| `--require-generated <family>` with missing artifacts | Returns nonzero in `--check` mode and prints the missing family. |
| Deferred runtime/backend governance | Preserves `status=defer`, `freshness_status=deferred`, and the Sprint 142 handoff reason. |

The generator never synthesizes `status=pass` for source-controlled contract
rows or missing generated reports.

## Focused Tests

`tests/test_normalize_report_index.py` covers:

- current-repository `--no-generated` output;
- deterministic row sorting;
- source-controlled contract rows not using `status=pass`;
- family filtering for `oracle`;
- required-generated failure when `oracle` generated rows are absent;
- alternate temporary `--corpus-root` and `--build-root`;
- generated-artifact presence rows for a temporary benchmark report;
- deferred runtime/backend row presence.

## Validation Evidence

Commands run:

```sh
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --no-generated --output build/report-index/normalized-index.tsv
python3 scripts/validate_corpus_schema.py
```

Results:

- focused generator tests passed;
- smoke run wrote `26` rows to ignored `build/report-index/normalized-index.tsv`;
- corpus schema validation still passed with the Day 5 contract rows.

## Claim Boundaries Preserved

- Missing generated benchmark, sentinel, guardrail, dead-code, coverage,
  oracle, and report-index artifacts are explicit `not_generated` rows, not
  silent success.
- Generated artifact presence is recorded as
  `generated_present_unchecked`; Day 10/11 own stale/fresh comparison gates.
- Runtime/backend governance remains deferred to Sprint 142.
- Package/install rows remain source-controlled proof-owner metadata.
- CI rows remain source-controlled lane definitions; hosted logs are not
  committed or reinterpreted.

## Day 7 Handoff

Day 7 should deepen corpus/oracle integration by mapping native corpus
fixture, expected-result, generated-reference, and solver-backed oracle rows
onto the normalized index. The Day 6 generator already provides the CLI,
contract loading, sorting, filtering, missing-generated handling, and output
writer needed for that work.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The generator produces stable output from the current repository state. | Complete | `python3 scripts/normalize_report_index.py --no-generated --output build/report-index/normalized-index.tsv` wrote 26 deterministic rows. |
| Unsupported report families are represented as defer/skip rows, not fabricated proof. | Complete | Runtime/backend governance emits `status=defer`; absent generated reports emit `freshness_status=not_generated`; source-controlled contract rows cannot become pass evidence. |
| Tests cover normal, missing, and deferred report-family paths. | Complete | `tests/test_normalize_report_index.py` covers current-repo output, required missing generated rows, generated artifact presence, family filtering, sorting, and deferred governance. |
