# Sprint 39 Day 4 Artifact: Final Cross-Platform Audit

## Purpose

Confirm that the Sprint 36 cross-platform parity contract still matches the
current README and CI workflows so Sprint 39 can treat the remaining
platform-specific limits as explicit closeout items rather than hidden drift.

## Day 4 Bottom Line

Sprint 39 Day 4 did **not** find a new cross-platform regression queue. The
current platform model is still the Sprint 36 model:

- Linux = strongest enforced reviewed baseline
- macOS = enforced Apple Clang reviewed/supporting baseline, dead-code staged,
  Homebrew GCC supplemental
- Windows = enforced reviewed CMake subset, local Makefile reviewed-wrapper
  parity staged, dead-code excluded

## Current Enforced / Staged / Excluded Map

### Linux

Current enforced reviewed baseline:

- `make quality-review-compile`
- `make quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`

Current supplemental signals:

- direct runtime/testing outside the reviewed wrapper abstraction
- `bench-fast`
- TSan
- coverage

Sprint 39 interpretation:

- Linux remains the strongest overall enforced baseline

### macOS

Current enforced Apple Clang path:

- `make quality-review-compile`
- `make quality-review-cmake`
- `make wall-check`
- `make sanitize`

Current staged/supplemental status:

- dead-code remains staged
- Homebrew GCC direct build/test leg remains supplemental
- install/pkg-config validation remains supplemental

Sprint 39 interpretation:

- the current contract is already honest
- no new macOS parity expansion is implied by the present workflow/docs state

### Windows

Current enforced reviewed subset:

- reviewed CMake configure
- reviewed CMake build
- `ctest -N`
- full `ctest`

Current staged/excluded status:

- local Makefile reviewed-wrapper parity remains staged
- dead-code remains staged/excluded rather than enforced
- named excluded tests remain explicit:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`

Sprint 39 interpretation:

- Windows still truthfully exposes a narrower reviewed subset
- current docs/workflows do not overclaim local Makefile or dead-code parity

## Shared-Baseline Interpretation

The strongest shared cross-platform reviewed baseline remains:

- reviewed CMake parity

The strongest overall platform baseline remains:

- Linux reviewed wrapper + dead-code contract

Sprint 39 should preserve both distinctions in final closeout language.

## Day 7 Likely Implementation Shape

Unless later validation surfaces a real platform regression, the expected final
cross-platform batch is narrow:

1. preserve the current enforced/staged/excluded wording in final closeout docs
2. keep reviewed CMake parity framed as the strongest shared reviewed baseline
3. avoid fake claims of universal local Makefile reviewed-wrapper parity
4. avoid fake claims of universal dead-code parity

## Immediate Guidance For Later Sprint 39 Work

- Treat the residual platform queue as a documentation and closeout contract
  problem unless a stronger rerun proves otherwise.
- Keep macOS dead-code, Windows local reviewed-wrapper parity, and Windows
  dead-code described as intentionally non-universal.
- Use the existing workflow files and README contract as the authoritative
  statement of current platform truth.
