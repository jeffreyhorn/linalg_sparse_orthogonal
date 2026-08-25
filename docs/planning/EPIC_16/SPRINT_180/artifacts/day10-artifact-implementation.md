# Sprint 180 Day 10: Artifact Implementation

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Add the selected Homebrew local formula proof material designed on Day 9 while
preserving the Sprint 171 public package-manager deferral posture. Day 10 adds
a source-controlled template and local provider notes only; it does not add a
live tap, a generated formula, a proof script, public support wording, or guard
narrowing.

## Implemented Artifacts

| Path | Status | Purpose |
| --- | --- | --- |
| `packaging/homebrew/sparse-lu-ortho.rb.in` | Added | Homebrew local formula template rendered later by the proof script into a temporary local formula. |
| `packaging/homebrew/README.md` | Added | Explains local-proof scope, generated-output hygiene, and unsupported Homebrew/package-manager boundaries. |

## Template Behavior

The template is intentionally not a live Homebrew/core formula. It contains
placeholders that the future proof script must replace before running any
provider command:

| Placeholder | Source |
| --- | --- |
| `__SPARSE_HOMEBREW_HOMEPAGE__` | Local-proof homepage chosen by proof implementation. |
| `__SPARSE_FORMULA_URL__` | Temporary `file://` source archive URL. |
| `__SPARSE_FORMULA_SHA256__` | SHA-256 of the temporary source archive. |
| `__SPARSE_VERSION__` | Current checkout `VERSION`. |
| `__SPARSE_HOMEBREW_LICENSE__` | Accurate local-proof license metadata, or proof stop if unavailable. |

The formula install block:

- configures the current source with CMake;
- installs into Homebrew's formula `prefix`;
- sets `SPARSE_OPENMP=OFF` and `SPARSE_MUTEX=OFF`;
- keeps `CMAKE_INSTALL_LIBDIR=lib`;
- checks `libsparse_lu_ortho.a` exists;
- fails if shared-library artifacts appear.

The formula `test do` block:

- writes a downstream CMake consumer in `testpath`;
- calls `find_package(Sparse __SPARSE_VERSION__ EXACT REQUIRED)`;
- links against `Sparse::sparse_lu_ortho`;
- runs a small sparse-matrix consumer;
- checks installed static archive, CMake package config, and `sparse.pc`;
- fails if shared-library artifacts appear.

## Static-First Boundary Notes

Day 10 keeps the selected provider artifact aligned with the static-first
package contract:

| Boundary | Implementation |
| --- | --- |
| Static archive only | Template checks `lib/libsparse_lu_ortho.a` and rejects shared artifacts. |
| No optional feature expansion | Template passes `SPARSE_OPENMP=OFF` and `SPARSE_MUTEX=OFF`. |
| CMake package consumer | Template test uses `find_package(Sparse)` and `Sparse::sparse_lu_ortho`. |
| Provider-neutral metadata | No changes to `sparse.pc.in` or `cmake/SparseConfig.cmake.in`. |
| Generated-output hygiene | README states rendered formula, tap, archive, cache, logs, bottle, build, and install outputs are not committed. |
| Public non-claim posture | No README, INSTALL, maintainer guide, package metadata, or workflow wording changed on Day 10. |

## Focused Checks

Day 10 focused validation covers artifact presence, template syntax, required
placeholders, unsupported live tap absence, and existing static/package-manager
guards.

| Check | Result |
| --- | --- |
| `test -f packaging/homebrew/sparse-lu-ortho.rb.in` | Passed |
| `test -f packaging/homebrew/README.md` | Passed |
| `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` | Passed |
| Required placeholders present in template | Passed |
| No committed `*/Formula/*` path | Passed |
| `bash scripts/package_manager_deferral_check.sh` | Passed |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `git diff --check` | Passed |

## Residual Risks

| Risk | Follow-up |
| --- | --- |
| License metadata remains unresolved. | Day 11-12 proof design/implementation must fail safely if accurate local-proof license metadata cannot be injected. |
| Template is not yet rendered or installed. | Day 12 proof script must generate the source archive, render the formula, and run local `brew` checks. |
| Current package-manager guard still describes formal deferral. | Day 13 must narrow the guard only after proof behavior exists. |
| Public docs still preserve Sprint 171 deferral. | Day 13 updates wording only to the proven local Homebrew proof level. |
| Homebrew formula class/file naming is proof-script dependent. | The proof script should render the template to a temporary formula path that matches `SparseLuOrthoLocal`. |

## Day 10 Decision

The selected Day 10 artifact exists in source control as a Homebrew local-proof
template, not as a live provider package. The implementation proceeds to Day
11 proof-script design.

No stop condition was triggered on Day 10 because unresolved source URL,
SHA-256, version, and license values are explicit placeholders that the future
proof script must inject or fail on before invoking `brew`.

## Day 10 Deliverables

- source-controlled Homebrew local formula template
- static-first boundary notes
- focused artifact checks
- focused validation summary
- Day 10 implementation notes
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day10-artifact-implementation.md`

## Validation

Day 10 added provider prototype template material and planning notes. It did
not modify `.c`, `.h`, package metadata templates, workflows, guards, or public
user-facing package-manager docs.

Validation commands:

```sh
test -f packaging/homebrew/sparse-lu-ortho.rb.in
test -f packaging/homebrew/README.md
ruby -c packaging/homebrew/sparse-lu-ortho.rb.in
grep -Fq "__SPARSE_FORMULA_URL__" packaging/homebrew/sparse-lu-ortho.rb.in
grep -Fq "__SPARSE_FORMULA_SHA256__" packaging/homebrew/sparse-lu-ortho.rb.in
grep -Fq "__SPARSE_VERSION__" packaging/homebrew/sparse-lu-ortho.rb.in
grep -Fq "__SPARSE_HOMEBREW_LICENSE__" packaging/homebrew/sparse-lu-ortho.rb.in
find . -path ./docs/planning -prune -o -path ./.git -prune -o -path "*/Formula/*" -print
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected artifact exists in source control. | Complete | `packaging/homebrew/sparse-lu-ortho.rb.in` and `packaging/homebrew/README.md` were added. |
| Artifact wording does not promote unsupported provider status. | Complete | README and template comments keep local-proof-only boundaries and reject broader claims. |
| Focused validation passes or failures are recorded with blockers. | Complete | Focused checks and residual risks are listed above. |
