# Sprint 77 Day 10 Artifact: Workflow & Contract Reconciliation Design

Date: 2026-06-17
Branch: sprint-77

## Purpose

Define the bounded workflow/docs/policy reconciliation batch from the landed
Day 6 and Day 9 state, and decide whether any maintained command or contract
surface actually needs follow-through.

## Main Result

Sprint 77 now has one explicit Day 11 follow-through contract:

- required next surface:
  - `docs/maintainer_guide.md`
- support only if wording truly forces it:
  - `README.md`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Why This Is The Right Day 11 Center

The strongest remaining question is not the install guide or workflow layer
anymore.

It is whether the authoritative maintainer-policy surface should now say the
workflow-level proof split more directly after the Day 9 landing.

That makes `docs/maintainer_guide.md` the strongest next surface because it is
where the repo already explains:

- static-first package truth
- reviewed-versus-supplemental platform reading
- local install-proof ownership
- bounded non-claims around reviewed install-validation parity

## Support-Surface Reading

The support surfaces are now explicitly conditional:

- `README.md`
  - support only if the compact package/platform summary becomes inaccurate
- `CMakeLists.txt`
  - support only if the policy wording would otherwise drift from export
    mechanics
- `tests/test_install.sh`
  - support only if Day 11 changes the interpretation of the local Make proof
- `tests/test_cmake_install.sh`
  - support only if Day 11 changes the interpretation of the local CMake proof

## Preserved Reviewed-vs-Supplemental Split

The Day 11 batch must preserve:

- reviewed lanes:
  - Linux strongest reviewed truth
  - macOS reviewed Apple Clang lane
  - Windows reviewed CMake consumer subset
- supplemental proof:
  - macOS Make install/`pkg-config` confidence path
  - Homebrew GCC second-compiler path
- local install/package regression proof:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Day 11 Intended Shape

The intended Day 11 shape is:

- update maintainer-policy wording only where the Day 9 workflow clarification
  made the proof split easier to say directly
- keep front-door, export, and script surfaces untouched unless a specific
  contradiction appears
- preserve the no-widening fence around reviewed platform claims

## Exit State

Sprint 77 now has one explicit workflow/contract reconciliation design:

- maintainer-policy follow-through first
- all other support surfaces conditional
- no reopening of the Day 6 or Day 9 lanes by default
