# Day 13: Integrated Claim Review

## Purpose

Reconcile Sprint 173 generated API publication, navigation, freshness gates,
and inherited non-claims before closeout.

## Inputs Reviewed

Day 13 reviewed the Sprint 173 artifact chain:

- Day 4 publication decision: selected guarded local-only generated API HTML;
- Day 6 implementation: local-only guard script and validation target;
- Day 8 freshness implementation: `api-docs-freshness` target;
- Day 10 navigation update: README, API reference, and maintainer guidance;
- Day 11 validation record: freshness and deferral checks;
- Day 12 owner map and residual list.

## Reconciled Supported Claim

Sprint 173 supports this generated API claim:

> The repository provides a local generated Doxygen API view that is current
> only after `make api-docs-freshness` passes in the active checkout.

That command proves the selected local path by:

- regenerating Doxygen HTML from the configured checked-in public-header input
  set;
- checking generated reference/source pages for the 18 checked-in public
  headers;
- proving `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` remain
  ignored local output;
- proving no generated API HTML under `docs/api/` is tracked, staged, or
  visible as non-ignored untracked output.

The source-controlled API truth remains:

- checked-in public headers under `include/`;
- `docs/api_reference.md`;
- maintainer policy in `docs/maintainer_guide.md`.

## Selected And Unselected Publication Modes

| Mode | Sprint 173 status | Evidence |
| --- | --- | --- |
| Guarded local-only generated HTML | Selected | `make api-docs-freshness` and `scripts/check_api_docs_local_only.sh`. |
| Hosted generated HTML | Not selected | No hosted URL, deployment workflow, retention policy, or support wording. |
| Committed generated HTML | Not selected | `docs/api/` remains ignored and untracked. |
| CI artifact-only generated HTML | Not selected | No artifact upload or retention policy. |
| Release evidence generated HTML | Not selected | Generated API HTML is local documentation output, not release proof. |

## Claim Scan Results

Day 13 ran a generated API claim scan over:

```text
README.md
docs/api_reference.md
docs/maintainer_guide.md
scripts/check_api_docs_local_only.sh
Makefile
```

The generated API matches were reviewed as expected:

- README lists `make api-docs-freshness` as the selected local Doxygen
  freshness plus local-only staging guard.
- `docs/api_reference.md` says generated HTML is local-only, ignored, and
  current only after `make api-docs-freshness` has just passed.
- `docs/maintainer_guide.md` says generated API HTML is not source-controlled,
  hosted, artifact-published, or release evidence.
- `scripts/check_api_docs_local_only.sh` fails if generated API HTML becomes
  tracked, staged, or non-ignored without a future committed-output decision.
- Makefile references unrelated report local-only targets as pre-existing
  bounded report evidence, not generated API publication claims.

## Deferral Guard Results

Day 13 reran deferral guards because generated API publication wording touches
adoption, package, and support-claim boundaries:

- `bash scripts/static_package_deferral_check.sh` passed.
- `bash scripts/package_manager_deferral_check.sh` passed.

These checks confirm the generated API work does not reopen:

- package-manager provider support;
- shared-library support;
- dynamic ABI stability;
- runtime-loader behavior;
- Windows package execution parity;
- unsupported package selector wording.

## Local Output State

`git status --ignored --short docs/api` reports:

```text
!! docs/api/
```

This is the intended Sprint 173 state: generated API HTML exists only as
ignored local output.

## Residuals

| Residual | Handoff |
| --- | --- |
| Hosted generated API HTML | Future work must define URL ownership, deployment permissions, retention, freshness semantics, and claim wording before implementation. |
| CI artifact-only generated API HTML | Future work must define artifact upload, retention, reviewer access, and whether it is evidence or only convenience output. |
| Committed generated API HTML | Future work must reverse the local-only decision with an explicit generated-output review policy before staging `docs/api/`. |
| CI docs-check lane | Future work may run `make api-docs-freshness` in CI as a check without changing publication status. |
| `sparse_version.h` Doxygen page | Remains owned by install/version validation unless Doxygen input policy changes. |

## Sprint 174 Handoff

If future work continues the selected local-only path:

- use `make api-docs-freshness` as the generated API freshness gate;
- do not stage files under `docs/api/`;
- keep API declarations anchored in checked-in public headers and
  `docs/api_reference.md`;
- rerun static-package and package-manager deferral guards when public wording
  touches support, packaging, ABI, or adoption surfaces.

If future work wants publication rather than local generation, create a new
decision record first. Publication work should not start by removing the
`docs/api/` ignore rule or uploading generated HTML without the support-tier
and retention policy.

## Validation

Day 13 validation passed:

- `make api-docs-freshness`
- `make api-docs-local-only`
- generated API claim scan over README/API/maintainer/Makefile/script surfaces
- `bash scripts/static_package_deferral_check.sh`
- `bash scripts/package_manager_deferral_check.sh`
- `git diff --check`

No `.c` or `.h` files changed on Day 13, so the full C quality gate is not
required for this day.

## Completion Check

Day 13 completion criteria are met:

- generated API publication claim boundary is internally coherent;
- docs navigation and freshness gates match the selected guarded local-only
  path;
- hosted, committed, artifact-only, release-evidence, package, ABI, platform,
  performance, external-parity, and state-of-the-art claims remain unselected.
