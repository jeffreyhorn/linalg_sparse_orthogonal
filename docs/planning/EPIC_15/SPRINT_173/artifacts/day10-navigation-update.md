# Day 10: Documentation Navigation Update

## Purpose

Update README and docs navigation so the selected local generated API
freshness command is discoverable without widening generated API publication
claims.

## Updated Files

| File | Update |
| --- | --- |
| `README.md` | Added `make api-docs-freshness` to the command list as the selected local Doxygen freshness plus local-only staging guard. |
| `docs/api_reference.md` | Added the selected freshness proof and changed generated HTML freshness wording from `make docs-check` to `make api-docs-freshness`. |
| `docs/maintainer_guide.md` | Updated maintainer command guidance to use `make api-docs-freshness` and explain that it runs `docs-check` plus local-only generated-output staging enforcement. |

## Navigation Result

The supported API reference hierarchy remains:

1. README routes exact API declaration users to `docs/api_reference.md`.
2. `docs/api_reference.md` routes exact declarations and contracts to
   checked-in public headers under `include/`.
3. `make docs-check` remains the local Doxygen generation plus page coverage
   layer.
4. `make api-docs-freshness` is the selected local generated API freshness
   proof, combining generation, page coverage, and local-only staging
   enforcement.
5. `docs/api/html/` remains ignored local generated output, not a supported
   hosted, committed, or artifact-published API documentation surface.

## Validation

Day 10 ran:

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

Day 10 also ran:

```bash
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
```

Both deferral guards passed.

## Claim Scan

Day 10 ran:

```bash
rg -n "hosted|committed|source-controlled|artifact|release evidence|package-manager|shared-library|dynamic ABI|runtime-loader|platform parity|portable performance|external-library parity|state-of-the-art" README.md docs/api_reference.md docs/maintainer_guide.md
```

The scan returned expected matches. The touched generated API docs use these
terms as non-claim boundaries:

- `docs/api_reference.md` states generated HTML is not hosted or
  source-controlled;
- `docs/api_reference.md` states the API reference does not imply dynamic ABI,
  shared-library, package-manager, platform, external-library, portable
  performance, or state-of-the-art coverage;
- `docs/maintainer_guide.md` states local generated output is not
  source-controlled, hosted, artifact-published, or release evidence;
- `docs/maintainer_guide.md` preserves non-claims for ABI, package,
  runtime-loader, hosted documentation publication, source-controlled generated
  HTML, and artifact-published generated HTML.

Other matches in README and the maintainer guide are pre-existing bounded
evidence or deferral language outside the generated API navigation update.

## Non-Claims Preserved

Day 10 did not add support for:

- hosted generated API HTML;
- committed generated API HTML;
- CI artifact-only generated API HTML;
- generated API HTML as release evidence;
- generated installed-header Doxygen coverage;
- package-manager provider support;
- shared-library support;
- dynamic ABI stability;
- runtime-loader behavior;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- portable performance guarantees;
- external-library parity;
- state-of-the-art sparse linear algebra coverage.

## Completion Check

Day 10 completion criteria are met:

- supported API reference location is discoverable;
- documentation matches the Day 4 local-only publication decision;
- unsupported claim boundaries remain intact.

No `.c` or `.h` files changed on Day 10, so the full C quality gate is not
required for this day.
