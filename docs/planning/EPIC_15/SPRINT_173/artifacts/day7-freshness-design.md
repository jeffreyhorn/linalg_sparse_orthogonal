# Day 7: Freshness Gate Design

## Purpose

Design freshness checks that prove the selected generated API HTML status is
accurate for Sprint 173's guarded local-only publication path.

## Selected Freshness Model

Sprint 173 should use **command-regenerated local freshness** for generated API
HTML.

Under this model, generated API HTML is current only for the active checkout
where the selected local command has just passed. The selected local command is
the Day 6 aggregate proof:

```bash
make api-docs-validate
```

That command proves:

1. Doxygen generated local HTML from the current checkout.
2. Generated reference/source pages exist for the configured checked-in public
   header input set.
3. `docs/api/` remains ignored.
4. generated API HTML is not tracked.
5. generated API HTML is not staged.
6. generated API HTML is not visible as non-ignored untracked output.

This freshness model deliberately avoids treating ignored generated HTML as
source-controlled evidence.

## Source Inputs That Invalidate Freshness

| Input | Why it invalidates generated API freshness | Required response |
| --- | --- | --- |
| `include/*.h` | Public declarations/comments are Doxygen input. | Rerun selected local freshness command. |
| added/removed checked-in top-level public header | Page coverage expectations change. | Rerun selected local freshness command. |
| `Doxyfile` | Doxygen input/output/warning behavior can change. | Rerun selected local freshness command. |
| `Makefile` docs targets | Generated-doc command behavior can change. | Rerun selected local freshness command and review target semantics. |
| `scripts/check_api_docs_coverage.py` | Page coverage semantics can change. | Rerun selected local freshness command. |
| `scripts/check_api_docs_local_only.sh` | Local-only staging semantics can change. | Rerun selected local freshness command. |
| `.gitignore` | Generated-output tracking policy can change. | Rerun local-only guard and review publication decision. |
| `docs/api_reference.md` | User-facing generated-doc interpretation can change. | Run docs/claim review; run freshness command if command wording changes. |
| `docs/maintainer_guide.md` | Maintainer generated-doc policy can change. | Run docs/claim review; run deferral guards if package/ABI wording changes. |
| `README.md` | User command/navigation surface can change. | Run docs/claim review; run freshness command if command wording changes. |
| Day 4 publication decision record | Claim boundaries can change. | Reconcile against generated-doc guard behavior. |

`include/sparse_version.h.in` remains outside current Doxygen page expectations
unless a future decision changes the generated installed-header policy.

## Freshness Check Design

### Selected Local Gate

The selected local gate is:

```bash
make api-docs-validate
```

It should remain the authoritative local generated API proof for Sprint 173.

### Recommended Day 8 Alias

Add a Make alias:

```make
api-docs-freshness: api-docs-validate
```

Rationale:

- the project already has explicit freshness targets such as
  `report-index-oracle-freshness`, `report-index-comparison-freshness`, and
  `bench-canonical-report-freshness`;
- `api-docs-freshness` gives maintainers a plain freshness command without
  changing `docs-check` semantics;
- the alias can reuse the Day 6 aggregate proof and avoid duplicate logic.

Day 7 does not recommend a new Python metadata checker yet. Regeneration before
coverage is stronger and simpler for local-only output than comparing stored
hash metadata for ignored generated files.

## Persisted Metadata Decision

Do not add persisted generated API freshness metadata in Sprint 173 unless Day
8 finds a concrete stale-output failure that cannot be handled by regeneration.

Reasons:

- generated HTML is ignored and local-only;
- source-controlled metadata about ignored generated output risks being read as
  release evidence;
- `make api-docs-validate` regenerates before checking, so it avoids stale
  local HTML by construction;
- the remaining value is command discoverability and staging enforcement, not
  a separate artifact manifest.

## Generated Output Exclusion Rules

| Unselected path | Exclusion rule |
| --- | --- |
| Hosted generated API HTML | No hosted URL, pages deployment, or hosted freshness claim may be added under Sprint 173. |
| Committed generated API HTML | `docs/api/` must remain ignored and no `docs/api/` files may be tracked or staged. |
| CI artifact-only generated API HTML | No artifact upload or artifact-retention claim may be added under Sprint 173. |
| Generated installed-header Doxygen pages | `sparse_version.h` remains install/version owned and is not an expected page. |

## Failure Semantics

| Failure | Meaning | Maintainer action |
| --- | --- | --- |
| Doxygen fails | local generated API HTML cannot be considered current | fix Doxygen/config/header issue and rerun gate |
| page coverage fails | generated HTML is incomplete for configured public headers | regenerate/fix coverage expectations |
| `docs/api/` not ignored | local-only tracking policy was weakened | restore ignore policy or write a new publication decision |
| generated API file tracked | committed-output path appeared without decision | remove tracked generated file or create new decision |
| generated API file staged | accidental generated-output staging | unstage generated file |
| non-ignored untracked generated file | ignore policy does not cover output | update ignore policy or generation path |
| docs claim hosted/committed/artifact output | unsupported publication claim | revise docs or create a new decision with evidence |

## CI And Local Integration

### Local

Use:

```bash
make api-docs-validate
```

After Day 8, prefer:

```bash
make api-docs-freshness
```

if the alias is implemented.

### CI

Sprint 173 does not add a hosted or artifact publication lane. A future CI
integration may run the selected freshness target as a check, but that would
still not publish generated API HTML unless a later decision selects it.

## Documentation Integration

Day 9/Day 10 should update docs only if needed to name the selected freshness
command. The wording should preserve:

- `docs/api_reference.md` and checked-in public headers as source-controlled
  API truth;
- generated HTML as local-only;
- current generated HTML only after the selected local freshness command
  passes;
- no hosted, committed, artifact-only, package, ABI, platform, performance,
  external-parity, or state-of-the-art claim.

## Completion Check

Day 7 completion criteria are met:

- freshness checks match the Day 4 local-only publication decision;
- stale generated API docs cannot be represented as current when maintainers
  use the selected regenerated local gate;
- unselected generated-output paths remain rejected or ignored.

No `.c` or `.h` files changed on Day 7, so the full C quality gate is not
required for this day.
