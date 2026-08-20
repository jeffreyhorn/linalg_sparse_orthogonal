# Day 14: Sprint Closeout And Sprint 174 Handoff

## Purpose

Close Sprint 173 by reconciling planned work against completed artifacts,
recording final validation, and handing off the generated API publication
boundary for future work.

## Project-Plan Item Reconciliation

| Item | Status | Evidence |
| --- | --- | --- |
| 173.1 Publication Decision | Complete | Day 4 selected guarded local-only generated API HTML and rejected hosted, committed, and artifact-only publication modes for Sprint 173. |
| 173.2 Generator Audit | Complete | Days 2 and 5 inventoried `Doxyfile`, `make docs`, `make docs-check`, generated output paths, ignored output state, and coverage behavior. |
| 173.3 Publication Implementation | Complete | Day 6 added `scripts/check_api_docs_local_only.sh`, `make api-docs-local-only`, and `make api-docs-validate`; Day 8 added `make api-docs-freshness`. |
| 173.4 Freshness Gate | Complete | `make api-docs-freshness` regenerates Doxygen, checks generated page coverage, and enforces local-only generated-output boundaries. |
| 173.5 Navigation Update | Complete | Day 10 updated README, `docs/api_reference.md`, and `docs/maintainer_guide.md` to point users and maintainers at `make api-docs-freshness`. |
| 173.6 Verification | Complete | Days 11, 13, and 14 reran freshness, local-only, claim-scan, and package/static deferral checks. |

## Final Supported State

Sprint 173 closes with a guarded local-only generated API HTML path.

The supported command is:

```bash
make api-docs-freshness
```

That target proves:

- local Doxygen generation completed;
- generated reference/source pages exist for the 18 checked-in public headers;
- `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` remain ignored;
- no generated API HTML under `docs/api/` is tracked, staged, or visible as
  non-ignored untracked output.

The source-controlled API reference remains:

- public headers under `include/`;
- `docs/api_reference.md`;
- generated API maintenance policy in `docs/maintainer_guide.md`.

## Publication Boundary

Sprint 173 does not claim:

- hosted generated API HTML publication;
- committed generated API HTML;
- CI artifact-only generated API HTML;
- generated API HTML as release evidence;
- generated installed-header Doxygen coverage for `sparse_version.h`;
- package-manager provider availability;
- shared-library support;
- dynamic ABI stability;
- runtime-loader behavior;
- Windows Makefile or `pkg-config` parity;
- broad platform parity;
- portable performance guarantees;
- external-library parity;
- state-of-the-art sparse linear algebra coverage.

## Final Validation

Day 14 validation passed:

```text
make api-docs-freshness
make api-docs-local-only
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
generated API claim scan over README/API/maintainer/Makefile/script surfaces
git diff --check
```

`make api-docs-freshness` reported:

```text
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
api-docs-local-only: passed
```

`git status --ignored --short docs/api` reported:

```text
!! docs/api/
```

This is the intended local-only generated-output state.

## Files Changed By Sprint 173

Sprint 173 changed the maintained generated API surface:

- `Makefile`
- `README.md`
- `docs/api_reference.md`
- `docs/maintainer_guide.md`
- `scripts/check_api_docs_local_only.sh`
- `docs/planning/EPIC_15/SPRINT_173/PLAN.md`
- `docs/planning/EPIC_15/SPRINT_173/WORKING_NOTES.md`
- daily Sprint 173 artifacts under
  `docs/planning/EPIC_15/SPRINT_173/artifacts/`

No `.c` or `.h` files changed during Day 14, so the full C quality gate is not
required for this closeout day.

## Sprint 174 Handoff

Future work should treat generated API HTML as local-only unless it first
creates and lands a new publication decision.

If continuing local-only:

- use `make api-docs-freshness` before relying on local generated HTML;
- do not stage generated output under `docs/api/`;
- keep exact declarations anchored in public headers and `docs/api_reference.md`;
- use `docs/maintainer_guide.md` as the generated API maintenance policy.

If promoting generated API HTML to hosted, committed, or artifact-only output:

- define the publication mode before changing `.gitignore`;
- define ownership for URL, artifact retention, or generated-output review
  churn;
- add CI workflow behavior only after the claim boundary is explicit;
- rerun static-package and package-manager deferral guards if public wording
  touches package, ABI, runtime, platform, or adoption surfaces.

## Closeout Assessment

Sprint 173 completed its planned goal: generated API HTML now has a maintained
local freshness command, a local-only enforcement guard, public navigation, and
bounded non-claim wording. Generated HTML remains useful for local inspection
without becoming a hosted, committed, artifact, package, ABI, platform,
performance, external-parity, or state-of-the-art claim.
