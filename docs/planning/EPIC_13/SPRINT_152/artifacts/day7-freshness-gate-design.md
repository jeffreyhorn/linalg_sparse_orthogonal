# Sprint 152 Day 7 Freshness Gate Design

## Purpose

Day 7 defines the report-index freshness gate behavior for the selected
generated families before Day 8 expands executable coverage. The design keeps
selected oracle freshness local-only and fixture-local; it does not broaden
generated rows into release, CI, package, ABI, platform, performance, or
state-of-the-art proof.

## Selected Gate Scope

The only selected generated families for Sprint 152 gates are:

| Family | Subfamily | Gate Role | Required Mode | Strict Mode | Claim Boundary |
| --- | --- | --- | --- | --- | --- |
| `oracle` | `generated_reference` | Required local generated corpus reference rows | `--require-generated oracle --check-freshness` must fail when missing | `--strict-generated` must fail stale or failing rows | Fixture-local generated oracle evidence only |
| `oracle` | `solver_backed` | Required local QR and partial-SVD generated proof rows | `--require-generated oracle --check-freshness` must fail when partial or stale | `--strict-generated` must fail stale, failing, or incomplete selected rows | Fixture-local generated solver-backed evidence only |
| `report_index` | `missing_generated` | Supporting missing-row visibility | Source-controlled/advisory unless selected generated family is required | Does not become a generated proof row | Index navigation and governance only |

Non-selected generated families stay advisory, source-controlled, optional, or
deferred unless a later sprint explicitly selects them.

## Required Assertions

When `--family oracle --require-generated oracle --check-freshness` is used,
the gate must assert:

- generated oracle artifacts exist under the selected artifact pattern
  `build/corpus/oracle/*.tsv`;
- selected generated oracle rows are present;
- current source commit matches `HEAD` for generated oracle rows;
- no selected generated oracle row reports `comparison_status=fail`;
- selected combined oracle output contains `52` generated rows;
- row counts by solver family are exactly:
  - `unknown`: `3`;
  - `qr`: `23`;
  - `partial_svd`: `26`;
- selected solver families include `unknown`, `qr`, and `partial_svd`;
- selected fixture keys include the Sprint 150 QR and Sprint 151 partial-SVD
  maintained family set;
- failures name the row or aggregate mismatch, artifact path or manifest path,
  observed versus expected value, and regeneration command.

Required oracle freshness may still emit `warning` diagnostics for
`generated_present_unchecked` rows until Day 8 or later upgrades strict
comparison semantics. Warnings must not fail the required gate unless they are
stale, missing, failing, or selected-family mismatches.

## Strict Assertions

When `--strict-generated --check-freshness` is used for selected oracle rows,
the gate must:

- apply the same selected row-count, solver-family, and fixture-key assertions;
- fail stale selected oracle rows;
- fail selected oracle rows with `status=fail`;
- fail incomplete selected oracle output even if some oracle rows exist;
- preserve advisory handling for explicitly advisory non-selected families when
  `--advisory-ok` is supplied.

Strict generated mode is not a hosted CI claim by itself. It is a local
freshness comparison mode unless Day 9-10 deliberately promote a selected lane.

## Advisory And Deferred Assertions

The following rows must not become required just because oracle is selected:

| Family | Expected Gate Behavior |
| --- | --- |
| `benchmark` | Advisory generated-local freshness; stale rows warn/error only under strict local policy unless advisory is accepted. |
| `sentinel` | Hard-gate and guardrail failures remain errors, but advisory sentinel measurements do not become selected oracle freshness proof. |
| `guardrail` | Guardrail lane failures remain errors; supplemental guardrails do not become oracle proof. |
| `coverage` | Missing coverage reports remain advisory unless explicitly required. |
| `deadcode` | Dead-code rows remain advisory/summarizing evidence, not zero-dead-code proof. |
| `package` | Source-controlled package contract rows remain source-controlled and do not become generated freshness rows. |
| `ci` | Hosted workflow logs remain external evidence, not local generated report freshness artifacts. |
| `documentation` | Source-controlled documentation rows remain Git-reviewed guidance. |
| `runtime_backend` | Source-controlled governance rows remain schema/Git evidence. |

## CLI Behavior Design

### Required Oracle Gate

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

Expected behavior:

- succeeds only when the selected combined oracle family exists, is current,
  and matches the selected row-count/fixture-key policy;
- prints diagnostics for every oracle row plus aggregate selected-family
  diagnostics when mismatches exist;
- exits `1` on missing, stale, failing, incomplete, or mismatched selected
  oracle output.

### Strict Oracle Gate

```sh
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --strict-generated --check-freshness
```

Expected behavior:

- fails stale or incomplete selected oracle output;
- keeps `generated_present_unchecked` warnings visible until strict comparison
  semantics are fully upgraded;
- must not downgrade selected oracle aggregate mismatches to advisory.

### Advisory Family Check

```sh
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 scripts/normalize_report_index.py --family coverage --check-freshness
```

Expected behavior:

- missing or stale advisory generated rows do not fail by default;
- `--require-generated <family>` may still fail missing required rows for an
  explicitly requested family;
- advisory output must preserve non-claims and row meaning.

## Test Matrix For Day 8

| Case | Setup | Command | Expected Result |
| --- | --- | --- | --- |
| Missing required oracle | No oracle artifact or `--no-generated` | `--family oracle --require-generated oracle --check-freshness` | Exit `1`; message names `oracle`, artifact pattern, canonical command. |
| Complete selected oracle | 52 synthetic or generated rows with current commit | `--family oracle --require-generated oracle --check-freshness` | Exit `0`; no selected row-count/fixture-key errors. |
| Partial selected oracle | Any selected row removed or QR/partial-SVD-only output | `--family oracle --require-generated oracle --check-freshness` | Exit `1`; `oracle_selected_row_count` mismatch. |
| Missing solver family | Remove all `qr` or `partial_svd` rows | Required oracle gate | Exit `1`; `oracle_selected_solver_families` mismatch. |
| Missing fixture key | Remove rows for one selected fixture while preserving total count | Required oracle gate | Exit `1`; `oracle_selected_fixture_keys` mismatch. |
| Stale oracle row | Generated row has old `source_commit` | Required or strict oracle gate | Exit `1`; message includes recorded/current commit and artifact. |
| Advisory stale oracle | Stale oracle without required/strict | `--family oracle --check-freshness` | Exit `0`; warning is visible. |
| Oracle comparison failure | Generated row maps to `status=fail` | Required or strict oracle gate | Exit `1`; message includes fixture key and artifact. |
| Advisory family missing | Missing coverage or benchmark output | `--family coverage --check-freshness` | Exit `0`; advisory diagnostic. |
| Explicit advisory required | Missing coverage with `--require-generated coverage` | `--family coverage --require-generated coverage --check-freshness` | Exit `1`; family-specific missing message. |
| Source-controlled compatibility | Package, documentation, runtime-backend rows | `--check-freshness` | Exit `0`; source-controlled advisory diagnostics. |
| Ignored artifact compatibility | Generated `build/` files exist but are untracked | Required oracle gate | Gate reads artifacts but does not require committing them. |

## Failure Message Contract

Failures should be stable enough for maintainers to act without reading
implementation code:

- missing oracle output: family, expected artifact pattern, canonical command;
- stale source commit: row ID, recorded commit, current commit, artifact path,
  canonical command;
- row-count mismatch: expected total/counts, observed total/counts, artifact
  pattern, canonical command;
- missing solver family: missing family, observed families, artifact pattern,
  canonical command;
- missing fixture key: missing key, manifest path, canonical command;
- oracle comparison failure: row ID, fixture key, artifact path, canonical
  command;
- advisory/deferred rows: reason and non-claim boundary.

## Compatibility Boundaries

- Historical source-controlled rows remain valid if their schemas and
  non-claims are intact.
- Generated `build/` and `coverage/` artifacts remain ignored local outputs.
- Temporary output paths used by tests are acceptable if the gate receives the
  matching `--build-root`.
- QR-only and partial-SVD-only oracle commands remain debugging tools but must
  not satisfy selected combined oracle required freshness.
- Hosted CI logs and uploaded artifacts remain external evidence until the
  Day 9-10 CI policy explicitly changes that posture.

## Day 8 Implementation Checklist

- Add or strengthen tests for missing required oracle output.
- Add or strengthen tests for stale required oracle rows.
- Add or strengthen tests for oracle comparison failure rows.
- Add a missing-solver-family case.
- Add a missing-fixture-key case that proves total row count alone is
  insufficient.
- Add advisory/deferred compatibility tests for at least coverage and one
  source-controlled family.
- Re-run focused Python/report validation and `git diff --check`.

## Non-Claims

This gate design does not claim broad QR correctness, broad partial-SVD
correctness, external-library parity, hosted CI proof, release artifact proof,
package-manager availability, shared-library ABI support, broad platform
support, portable performance, benchmark superiority, complete coverage, zero
dead code, or state-of-the-art sparse linear algebra status.
