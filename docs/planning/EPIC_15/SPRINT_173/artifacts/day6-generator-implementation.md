# Day 6: Generator Command Implementation

## Purpose

Implement the selected local-only enforcement path for generated API HTML while
preserving existing generation behavior.

## Implemented Changes

Day 6 added:

- `scripts/check_api_docs_local_only.sh`;
- Make target `api-docs-local-only`;
- Make target `api-docs-validate`.

Day 6 did not change:

- `Doxyfile`;
- `.gitignore`;
- checked-in public headers;
- user-facing documentation;
- generated output tracking policy;
- hosted, committed, or CI artifact publication behavior.

## Guard Behavior

`scripts/check_api_docs_local_only.sh` verifies:

1. `docs/api` is ignored.
2. `docs/api/html` is ignored.
3. `docs/api/html/index.html` is ignored.
4. no generated API files under `docs/api/` are tracked.
5. no generated API files under `docs/api/` are staged.
6. no generated API files under `docs/api/` are visible as non-ignored
   untracked files.

Failure messages point maintainers back to the selected local-only policy and
state that committed generated output requires a future publication decision.

## Make Target Behavior

| Target | Behavior |
| --- | --- |
| `make docs` | Unchanged: runs `doxygen Doxyfile`. |
| `make api-docs-coverage` | Unchanged: runs `python3 scripts/check_api_docs_coverage.py`. |
| `make docs-check` | Unchanged: runs `docs` and `api-docs-coverage`. |
| `make api-docs-local-only` | New: runs `bash scripts/check_api_docs_local_only.sh`. |
| `make api-docs-validate` | New: runs `docs-check` and `api-docs-local-only`. |

This keeps generation, page coverage, and local-only repository-boundary checks
separate while providing one aggregate local validation command.

## Local-Only Guard Proof

Day 6 ran:

```bash
make api-docs-local-only
```

Result:

```text
api-docs-local-only: docs/api ignore rule ok
api-docs-local-only: docs/api/html ignore rule ok
api-docs-local-only: docs/api/html/index.html ignore rule ok
api-docs-local-only: no tracked generated API files ok
api-docs-local-only: no staged generated API files ok
api-docs-local-only: no non-ignored generated API files ok
api-docs-local-only: passed
```

## Aggregate Generated API Validation Proof

Day 6 ran:

```bash
make api-docs-validate
```

Result:

```text
Generating API documentation with Doxygen...
doxygen Doxyfile
Documentation generated in docs/api/html/
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
api-docs-local-only: docs/api ignore rule ok
api-docs-local-only: docs/api/html ignore rule ok
api-docs-local-only: docs/api/html/index.html ignore rule ok
api-docs-local-only: no tracked generated API files ok
api-docs-local-only: no staged generated API files ok
api-docs-local-only: no non-ignored generated API files ok
api-docs-local-only: passed
```

## Generated-Output Staging Evidence

The guard proves:

- `docs/api/` remains ignored;
- generated API HTML is not tracked;
- generated API HTML is not staged;
- generated API HTML is not visible as non-ignored untracked output.

This matches the Day 4 selected guarded local-only publication decision.

## Claim Boundaries Preserved

Day 6 does not claim:

- hosted generated API HTML;
- committed generated API HTML;
- CI artifact-only generated API HTML;
- generated API HTML as release evidence;
- generated installed-header Doxygen coverage;
- package-manager provider support;
- shared-library support;
- dynamic ABI stability;
- runtime-loader behavior;
- broad platform parity;
- portable performance guarantees;
- external-library parity;
- state-of-the-art sparse linear algebra coverage.

## Day 7 Handoff

Day 7 should design the freshness gate on top of the implemented command
surface:

- `make docs-check` proves local generation plus page coverage;
- `make api-docs-local-only` proves ignored/tracked/staged boundaries;
- `make api-docs-validate` proves both together;
- any additional freshness work should decide whether command success is
  sufficient for local-only freshness or whether source-input metadata is
  needed.

## Completion Check

Day 6 completion criteria are met:

- selected generation path is executable and explicitly enforced;
- generated output appears only where the decision allows it;
- failures are visible to maintainers through focused guard messages.

No `.c` or `.h` files changed on Day 6, so the full C quality gate is not
required for this day.
