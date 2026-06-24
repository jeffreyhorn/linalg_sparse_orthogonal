# Sprint 87 Day 7: Post-Landing Package Audit and Rerank

## Purpose

Re-rank the remaining Sprint 87 package and consumer contradiction map after
the Day 6 packaging/export landing.

## Main Result

The Day 6 landing closed the strongest first Sprint 87 contradiction:

- `CMakeLists.txt` no longer stands out as the clear next landing center
- the repo now has one real bounded package-version/export truth seam landed
- a second immediate product-contract batch is not the highest-value next move

The strongest remaining Sprint 87 seam is now maintained consumer-proof
expansion.

## Exact Next Center

The exact Day 8 design center is now fixed to:

- `tests/test_install.sh`

The key post-Day-6 package reading is explicit:

- the installed CMake package contract is now sharper and more truthful than
  at sprint start
- `tests/test_cmake_install.sh` already owns the exact-version CMake consumer
  contract end to end
- the thinnest remaining maintained consumer surface is the Make/pkg-config
  install path and the local compiled consumer it proves

That means the remaining contradiction is no longer primarily package-version
truth. It is the narrower maintained local consumer story that still needs to
be widened deliberately and boundedly.

## Post-Day-6 Hotspot Context

Post-Day-6 live hotspot map:

- `tests/test_install.sh` = `172` lines
- `tests/test_cmake_install.sh` = `192` lines
- `examples/cmake_example/CMakeLists.txt` = `10` lines
- `.github/workflows/macos-ci.yml` = `117` lines
- `.github/workflows/windows-ci.yml` = `63` lines
- `README.md` = `1051` lines
- `INSTALL.md` = `266` lines
- `docs/maintainer_guide.md` = `727` lines
- `CMakeLists.txt` = `416` lines

The useful distinction is no longer raw size alone. It is that the first
product-contract contradiction has already been reduced in code, while the
maintained consumer proof remains the next thinnest and highest-value lane.

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `tests/test_cmake_install.sh`
- `examples/cmake_example/CMakeLists.txt`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

Current reading:

- `tests/test_cmake_install.sh` remains a retained adjacent proof owner but
  does not become the next landing center unless the Day 8 design truly forces
  shared package-proof movement
- `examples/cmake_example/CMakeLists.txt` remains the retained downstream CMake
  consumer surface, not the best next batch owner by itself
- the macOS and Windows workflows remain later evidence owners and should stay
  deferred until the maintained local consumer contract is stronger
- package/docs wording is already truthful enough to remain deferred unless the
  next consumer-proof batch changes the maintained rerun contract

## Preserved Non-Touch Map

The useful Day 7 clarification is explicit now:

- no second immediate product-matrix or export-semantics batch as the next
  center
- no early workflow widening before the local maintained consumer proof is
  strengthened
- no support-surface churn detached from a real consumer-proof seam
- no shared-library or broad ABI claim widening

## Strongest Clarification

Sprint 87's next contradiction center is no longer “do more package-contract
work because the first batch succeeded.”

It is also not “jump straight to workflow widening because package wording is
sharper now.”

It is the remaining maintained local consumer-proof gap on the static-first
install/pkg-config lane.

That fixes the ordering:

- next seam = consumer-proof expansion
- later seam = workflow/platform follow-through
- later seam = broader support-surface alignment
- later only if newly justified = another package/export contract batch

## Validation

This was a docs-only rerank day, so no build/test rerun was required.

The rerank was grounded in direct rereads of:

- `CMakeLists.txt`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `examples/cmake_example/CMakeLists.txt`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

## Exit State

- Sprint 87 now has one explicit post-Day-6 rerank.
- Day 8 can stay bounded to one consumer-proof design lane centered on
  `tests/test_install.sh`.
- Workflow/platform follow-through and broader support-surface alignment remain
  clearly separated from the real next implementation move.
