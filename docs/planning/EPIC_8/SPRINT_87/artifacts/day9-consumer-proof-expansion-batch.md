# Sprint 87 Day 9: Consumer-Proof Expansion Batch

## Purpose

Strengthen the maintained static-first Make/pkg-config consumer story without
reopening package semantics or widening workflow/platform claims.

## Main Result

Sprint 87's second implementation landing stayed inside the Day 8 fence:

- required implementation center:
  - `tests/test_install.sh`
- directly forced support follow-through actually needed:
  - none
- not needed in the batch:
  - `examples/cmake_example/CMakeLists.txt`
  - `tests/test_cmake_install.sh` logic changes
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

## Landed Surface

The kept consumer-proof win is explicit:

- the Make/pkg-config lane no longer proves only a tiny ad hoc consumer
- `tests/test_install.sh` still proves the basic installed compile/link/run
  path
- it now also proves the maintained `examples/cmake_example/main.c` source can
  compile and run through the installed pkg-config metadata

That makes the static-first downstream story stronger without changing package
semantics.

## Strongest Clarification

The useful Day 9 clarification is now explicit:

- this batch did not reopen CMake package semantics or version behavior
- it did not widen platform or ABI claims
- it improved only the bounded local consumer evidence on the maintained
  Unix-side Make/pkg-config lane

## Validation

The landed batch passed:

- `bash tests/test_install.sh`
  - `13` passed, `0` failed
  - including the new maintained example source pkg-config compile/run proof
- `bash tests/test_cmake_install.sh`
  - `15` passed, `0` failed

Because no `*.c` or `*.h` files changed, `make format`, `make lint`, and
`make test` were not required for this batch.

## Exit State

- Sprint 87 now has one landed bounded consumer-proof expansion batch.
- The maintained static-first consumer story is stronger on the Make/pkg-config
  lane than it was at sprint start.
- Workflow/platform follow-through remains the next later lane rather than part
  of this consumer batch.
