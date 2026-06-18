# Sprint 77 Day 4 Artifact: First Packaging/Platform Boundary

Date: 2026-06-17
Branch: sprint-77

## Purpose

Freeze the first Sprint 77 packaging/platform fence so the next design pass
starts from one bounded release/install productization lane rather than from a
mixed install, workflow, export, CI, and platform-proof backlog.

## Main Result

Sprint 77 now has one explicit first landing boundary:

- required first landing:
  - `INSTALL.md`
- support only if the first landing forces it:
  - `docs/maintainer_guide.md`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `README.md`
- explicitly deferred:
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - broader CI/workflow contract surfaces
  - broad ABI/version claim widening
  - shared-library or dynamic-ABI marketing

## Why This Is the Right First Fence

The operator-facing install/export contract remains the best first landing
because it already has the strongest bounded product shape:

- one explicit operator-facing install guide:
  - `INSTALL.md`
- one already-real static-first release contract
- one already-real downstream consumer story:
  - `pkg-config`
  - `find_package(Sparse)`
- one existing install-proof split:
  - local Unix-side proof scripts
  - narrower reviewed platform lanes

That gives Sprint 77 the strongest combination of:

- downstream leverage
- low compatibility risk
- manageable proof cost
- bounded payoff without widening the product claim surface

## Support Surface Reading

The support surfaces are bounded rather than assumed:

- `docs/maintainer_guide.md`
  - move only if the first batch makes the reviewed-versus-supplemental
    package/platform reading clearer in a way the policy surface should capture
- `CMakeLists.txt`
  - move only if the first batch truly needs a narrower export/install wording
    or metadata follow-through to keep the package surface coherent
- `tests/test_install.sh`
  - move only if the first batch changes the local Make install/uninstall proof
    reading
- `tests/test_cmake_install.sh`
  - move only if the first batch changes the local CMake install/export proof
    reading
- `README.md`
  - move only if the compact front-door package/platform summary becomes
    inaccurate after the first batch

## Explicit Deferred Set

The Day 4 deferred set is now fixed:

- platform-proof lane:
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- broader workflow/contract reconciliation:
  - other CI/workflow surfaces beyond direct support follow-through
- broad product-claim widening:
  - shared-library maturity
  - dynamic-ABI guarantees
  - broader reviewed install-validation parity
  - broader reviewed Windows Makefile parity
- unrelated solver/backend/capability work

## Non-Goal Fence

The first Sprint 77 batch explicitly does not include:

- broad packaging-system rewrites
- fake shared-library or dynamic-ABI maturity claims
- broader reviewed cross-platform claims than maintained evidence supports
- CI-lane expansion disguised as install/productization cleanup
- capability, backend, or benchmark-governance reopening

## Day 5 Implication

The Day 5 design pass should therefore start from:

- exact first implementation center:
  - `INSTALL.md`
- support only if truly forced:
  - `docs/maintainer_guide.md`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `README.md`
- explicitly not next:
  - CI workflow edits
  - platform-proof widening
  - ABI promise expansion
  - shared-library marketing
