# Sprint 87 Day 8: Consumer-Proof Expansion Design

## Purpose

Define the bounded local install/export and downstream-consumer proof package
Sprint 87 should land next so the maintained static-first consumer story is
stronger than it was at sprint start.

## Main Result

Sprint 87 now has one explicit second implementation contract:

- required Day 9 center:
  - `tests/test_install.sh`
- directly forced support-only follow-through if the consumer batch truly
  needs it:
  - `examples/cmake_example/CMakeLists.txt`
  - `tests/test_cmake_install.sh`
- strongest support-only wording if the proof contract truly changes the
  maintained rerun story:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- lower-value non-touch surfaces:
  - `CMakeLists.txt`
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

## Exact Day 9 Center

The exact Day 9 implementation center is now fixed to one bounded expansion
inside `tests/test_install.sh`, not another package-contract batch and not a
workflow-first widening pass.

The decisive Day 8 reason is explicit:

- `tests/test_cmake_install.sh` already proves installed CMake consumer and
  exact-version behavior end to end
- `tests/test_install.sh` proves the Make/pkg-config lane, but it still owns
  the narrowest maintained downstream consumer reading
- the highest-value next seam is therefore to make that static-first
  pkg-config consumer proof richer without widening platform or ABI claims

## Best Consumer Lane

The strongest bounded Day 9 consumer lane is now fixed to:

- keep `tests/test_install.sh` as the implementation owner
- strengthen the installed Make/pkg-config consumer story through one more
  explicit downstream-consumer proof seam
- reuse the retained local compile/link/run lane rather than inventing a new
  workflow or broad package surface
- touch `examples/cmake_example/CMakeLists.txt` only if the Day 9 proof package
  truly benefits from reusing the maintained example consumer shape
- touch `tests/test_cmake_install.sh` only if the Day 9 batch exposes a real
  shared helper or contract seam that should stay synchronized

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `examples/cmake_example/CMakeLists.txt`
- `tests/test_cmake_install.sh`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

Current reading:

- `examples/cmake_example/CMakeLists.txt` should stay untouched unless the
  maintained local consumer batch can reuse it directly without turning Day 9
  into a CMake-first consumer rewrite
- `tests/test_cmake_install.sh` should stay untouched unless the Day 9 proof
  package truly changes a shared downstream-consumer contract
- docs wording should stay deferred unless the landed batch changes the
  maintained rerun contract, not just the local proof depth

## Preserved Fence

The bounded Day 8 fence is explicit:

- no second immediate product-matrix or export-semantics batch
- no workflow/platform widening folded into Day 9
- no shared-library or broad ABI claim widening
- no generic install-script rewrite detached from the consumer-proof seam
- no drift from maintained local proof into benchmark or reviewed-test
  ownership

## Exit State

- Sprint 87 now has one exact second implementation contract.
- Day 9 can stay bounded to `tests/test_install.sh` and strengthen the
  maintained static-first consumer story without reopening package semantics or
  workflow claims.
- Workflow/platform follow-through remains explicitly later.
