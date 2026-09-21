# Sprint 207 Day 11 Maintainer Package Docs

## Purpose

Day 11 updated maintainer-facing package documentation so future package claim
changes route through the Sprint 207 continued-deferral decision instead of
mistaking Sprint 198 local Homebrew proof for broader provider support.

## Maintainer Surfaces Updated

| Surface | Day 11 update |
| --- | --- |
| `docs/maintainer_guide.md` | Adds Sprint 207 to package/Homebrew evidence ownership and records exact evidence required before changing package-provider claims. |
| `packaging/homebrew/README.md` | Adds a maintainer claim-change checklist for reopening public Homebrew tap/source formula or Homebrew/core readiness paths. |
| `docs/planning/EPIC_19/SPRINT_207/WORKING_NOTES.md` | Marks item 207.5 complete for user and maintainer docs and records validation. |

## Current Maintainer Routing

| Decision point | Required maintainer action |
| --- | --- |
| Package wording clarity | Keep source install as the user-facing path and Homebrew as developer-mode local proof only. |
| Public Homebrew tap/source formula | Add stable archive/checksum provenance, provider formula ownership, non-local proof, cleanup proof, docs, residuals, and guard coverage before changing wording. |
| Homebrew/core readiness | Add all public tap/source formula evidence plus Homebrew/core-style formula audit evidence, release discipline, and submission/maintenance ownership. |
| Bottles, Linuxbrew, binary packages, release packages, or other providers | Add a separate product decision, provider-specific proof, artifact policy, docs, and guard coverage for that exact tier. |
| Shared-library packages or dynamic ABI behavior | Add a separate package/ABI product decision and validation stack before changing the static-first boundary. |

## Supersession Notes

- Sprint 198 remains the local Homebrew proof evidence owner.
- Sprint 207 is the current package-provider decision owner.
- Sprint 207 does not invalidate the Sprint 198 proof; it narrows its public
  interpretation by selecting continued package-provider deferral with stronger
  guards.
- Historical Homebrew proof evidence must not be cited as public tap,
  Homebrew/core, bottle, Linuxbrew, binary package, release package, or broad
  package-manager evidence.

## Validation

| Command | Result |
| --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `make docs-check` | Passed. |
| `python3 tests/test_package_manager_deferral_guard.py` | Passed. |
| `python3 -m py_compile tests/test_package_manager_deferral_guard.py` | Passed. |
| `bash -n scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed, including embedded local Homebrew proof boundary checks. |
| `brew list --formula | rg '^sparse-lu-ortho-local$' || true` | No installed proof formula remained. |
| `brew tap | rg '^sparse-lu-ortho/local-proof-' || true` | No temporary proof tap remained. |
| `find packaging/homebrew -maxdepth 3 ...` | No generated archive, log, formula, bottle, or `Formula/` output was present. |

## Completion Criteria

- Item 207.5 has maintainer-facing documentation implementation.
- Maintainers have exact evidence requirements for future package claim
  changes.
- Sprint 198 local package evidence is retained without being mistaken for
  broader current package support.
