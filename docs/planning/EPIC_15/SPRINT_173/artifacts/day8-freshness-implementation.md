# Day 8: Freshness Gate Implementation

## Purpose

Implement the generated API HTML freshness check selected by the Day 7 design
and record passing plus fail-mode evidence.

## Implemented Change

Day 8 added a Make alias:

```make
api-docs-freshness: api-docs-validate
```

This gives maintainers a freshness-named command while reusing the Day 6
aggregate generated API validation path.

## Target Behavior

| Target | Behavior after Day 8 |
| --- | --- |
| `make docs` | unchanged: run Doxygen and write ignored local HTML under `docs/api/html/` |
| `make api-docs-coverage` | unchanged: verify generated reference/source pages for checked-in public headers |
| `make api-docs-local-only` | unchanged: verify `docs/api/` remains ignored, untracked, unstaged, and not visible as non-ignored untracked output |
| `make docs-check` | unchanged: run `docs` plus `api-docs-coverage` |
| `make api-docs-validate` | unchanged: run `docs-check` plus `api-docs-local-only` |
| `make api-docs-freshness` | new alias: run `api-docs-validate` |

## Passing Freshness Evidence

Day 8 ran:

```bash
make api-docs-freshness
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

## Fail-Mode Proof

Day 8 ran an isolated proof in a temporary Git repository with generated
Doxygen-like output under `docs/api/html/` but without a `docs/api/` ignore
rule.

Command shape:

```bash
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/api_docs_local_only_fail.XXXXXX")
mkdir -p "$tmpdir/scripts" "$tmpdir/docs/api/html"
cp scripts/check_api_docs_local_only.sh "$tmpdir/scripts/"
cd "$tmpdir"
git init -q
touch docs/api/html/index.html
bash scripts/check_api_docs_local_only.sh
```

Observed failure:

```text
api-docs-local-only: FAIL: docs/api is not ignored; generated API HTML must remain local-only unless a future publication decision selects committed output
```

This proves the guard rejects a missing-ignore local-only boundary before
generated API HTML can be mistaken for a supported committed-output path.

## Generated-Output Staging Result

The passing freshness target proves:

- `docs/api/` is ignored;
- `docs/api/html/` is ignored;
- generated API HTML is not tracked;
- generated API HTML is not staged;
- generated API HTML is not visible as non-ignored untracked output.

Generated HTML remains local-only and ignored.

## Claim Boundaries Preserved

Day 8 does not claim:

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

## Day 9 Handoff

Day 9 should design navigation updates around this selected command:

```bash
make api-docs-freshness
```

Navigation wording should still route source-controlled API truth to
`docs/api_reference.md` and checked-in public headers. Generated HTML should be
described as local-only and current only after the local freshness target
passes.

## Completion Check

Day 8 completion criteria are met:

- selected generated API status is mechanically checkable;
- check failures are actionable;
- no unselected generated output is staged unintentionally.

No `.c` or `.h` files changed on Day 8, so the full C quality gate is not
required for this day.
