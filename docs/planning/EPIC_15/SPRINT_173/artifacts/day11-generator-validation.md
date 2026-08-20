# Day 11: Integrated Generator Validation

## Purpose

Run the generated API freshness gate, local-only staging checks, docs claim
scan, and package/ABI deferral guards together after the Day 10 navigation
updates.

## Commands Run

Day 11 ran:

```bash
make api-docs-freshness
make api-docs-local-only
rg -n "hosted|committed|source-controlled|artifact|release evidence|package-manager|shared-library|dynamic ABI|runtime-loader|platform parity|portable performance|external-library parity|state-of-the-art" README.md docs/api_reference.md docs/maintainer_guide.md
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
```

## Generated API Freshness Result

`make api-docs-freshness` passed:

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

## Local-Only Staging Result

`make api-docs-local-only` passed:

```text
api-docs-local-only: docs/api ignore rule ok
api-docs-local-only: docs/api/html ignore rule ok
api-docs-local-only: docs/api/html/index.html ignore rule ok
api-docs-local-only: no tracked generated API files ok
api-docs-local-only: no staged generated API files ok
api-docs-local-only: no non-ignored generated API files ok
api-docs-local-only: passed
```

This confirms:

- `docs/api/` remains ignored;
- generated API HTML is not tracked;
- generated API HTML is not staged;
- generated API HTML is not visible as non-ignored untracked output.

## Claim Scan Result

The targeted scan returned matches in README, `docs/api_reference.md`, and
`docs/maintainer_guide.md`. The matches were reviewed as expected:

- generated API docs explicitly say generated HTML is not hosted,
  source-controlled, artifact-published, or release evidence;
- API reference non-claim wording preserves dynamic ABI, shared-library,
  package-manager, broad platform, external-library, portable performance, and
  state-of-the-art boundaries;
- README and maintainer-guide matches outside the generated API section are
  pre-existing bounded evidence, report freshness, package/ABI deferral, or
  planning/report artifact references.

No generated API publication overclaim was found.

## Deferral Guard Results

`bash scripts/static_package_deferral_check.sh` passed:

```text
static-package-deferral-check: Sprint 170 product decision record ok
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: Makefile static archive contract ok
static-package-deferral-check: static install metadata ok
static-package-deferral-check: no shared export/ABI metadata found ok
static-package-deferral-check: package metadata has no static/shared selector ok
static-package-deferral-check: support wording remains deferred ok
static-package-deferral-check: Windows package non-claim wording ok
static-package-deferral-check: Windows workflow has no unselected package execution ok
static-package-deferral-check: passed
```

`bash scripts/package_manager_deferral_check.sh` passed:

```text
package-manager-deferral-check: deferral record ok
package-manager-deferral-check: provider recipe absence ok
package-manager-deferral-check: package metadata neutrality ok
package-manager-deferral-check: package-manager public non-claims ok
package-manager-deferral-check: passed
```

## Validation Conclusion

The selected generated API status is proven locally:

- `make api-docs-freshness` is executable and passing;
- generated API output remains local-only and ignored;
- documentation points users to the source-controlled API reference and names
  the selected freshness command;
- package-manager and static-package/shared ABI deferral boundaries remain
  intact.

## Completion Check

Day 11 completion criteria are met:

- selected generated API status is proven locally;
- no unintended generated output is staged;
- no validation failure requires user input.

No `.c` or `.h` files changed on Day 11, so the full C quality gate is not
required for this day.
