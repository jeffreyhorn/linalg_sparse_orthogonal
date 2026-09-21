# Sprint 207 Day 8: Proof Regression And Cleanup

## Purpose

Complete the selected continued-deferral proof behavior by adding focused
regression coverage for the package guard and verifying cleanup after the
selected local Homebrew proof path.

## Changed Surface

| File | Change |
| --- | --- |
| `scripts/package_manager_deferral_check.sh` | Moved Sprint 207 decision and forbidden-provider-claim checks before the embedded Homebrew proof so unsupported wording fails before expensive proof work. |
| `tests/test_package_manager_deferral_guard.py` | Added focused regression fixtures for unsupported public provider claims. |
| `docs/planning/EPIC_19/SPRINT_207/WORKING_NOTES.md` | Recorded Day 8 implementation and validation evidence. |
| `docs/planning/EPIC_19/SPRINT_207/artifacts/day8-proof-regression-cleanup.md` | Added this implementation record. |

No C source, public header, formula template, proof script, install metadata,
or CMake package metadata changed.

## Regression Coverage

Added `tests/test_package_manager_deferral_guard.py`.

The fixture creates a minimal repository with the package guard script and the
planning/public documentation markers needed to reach the Sprint 207
provider-claim checks. It then injects unsupported positive provider wording
and verifies the guard fails clearly before local Homebrew proof execution is
needed.

Covered mutations:

| Regression | Injected wording | Expected behavior |
| --- | --- | --- |
| Homebrew/core readiness | `Homebrew/core readiness is supported.` | Guard fails with unsupported provider claim diagnostic. |
| Public tap support | `public tap support is available.` | Guard fails with unsupported provider claim diagnostic. |
| Binary package support | `binary package support is provided.` | Guard fails with unsupported provider claim diagnostic. |

## Guard Call-Order Change

The package guard now checks Sprint 207 provider decision and forbidden
provider claims before running the selected Homebrew local proof boundary.

This preserves normal validation behavior while improving failure mode:

- unsupported public provider wording fails quickly;
- expensive local Homebrew proof is still run when wording and decision checks
  pass;
- local proof success still means local static source formula proof only.

## Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_package_manager_deferral_guard.py` | Passed | New regression fixtures fail clearly for unsupported provider wording. |
| `bash -n scripts/package_manager_deferral_check.sh` | Passed | Shell guard syntax is valid. |
| `python3 -m py_compile tests/test_package_manager_deferral_guard.py` | Passed | Python regression test syntax is valid. |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Deferral record, Sprint 198 proof record, provider recipe absence, Sprint 207 decision, forbidden provider claims, selected Homebrew local proof boundary, package metadata neutrality, and public non-claims all passed. |
| Installed formula scan | Passed | No `sparse-lu-ortho-local` formula remained installed after validation. |
| Temporary proof tap scan | Passed | No `sparse-lu-ortho/local-proof-*` tap remained after validation. |
| Generated Homebrew output scan | Passed | No generated `.tar.gz`, `.tgz`, `.zip`, `.log`, `.rb`, `.bottle.*`, or `Formula/` output appeared under `packaging/homebrew`. |

## Cleanup Policy Evidence

Day 8 did not change the proof script cleanup implementation. It verified that
the selected proof path, when invoked through the package guard, still leaves:

- no installed `sparse-lu-ortho-local` formula;
- no `sparse-lu-ortho/local-proof-*` temporary tap;
- no generated Homebrew proof artifacts under `packaging/homebrew`.

## Claim Boundary

Day 8 completes implementation for the selected continued-deferral path. It
does not promote:

- public Homebrew tap support;
- Homebrew/core readiness or acceptance;
- bottles;
- Linuxbrew;
- vcpkg, Conan, pkgsrc, distro/system packages, or binary packages;
- shared-library package support;
- dynamic ABI compatibility;
- broad package-manager distribution.

## Day 8 Completion Criteria

| Criterion | Status |
| --- | --- |
| Item 207.3 is implementation-complete for the selected path. | Complete. |
| Proof cleanup behavior is guarded or documented. | Complete; package guard and Day 8 cleanup scans verify no installed formula, temp tap, or generated Homebrew outputs remain. |
| Any environment blocker prevents support promotion rather than weakening the decision boundary. | Complete; proof unavailable semantics remain local-proof-only and unsupported provider wording fails guard checks. |

## Day 8 Outcome

The selected continued-deferral implementation is complete. Package-provider
promotion remains unclaimed, the local proof remains intact, unsupported
provider wording now has focused regression coverage, and cleanup evidence is
recorded for the selected proof path.
