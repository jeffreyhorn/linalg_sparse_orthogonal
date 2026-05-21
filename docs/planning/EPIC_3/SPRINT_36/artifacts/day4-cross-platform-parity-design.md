# Sprint 36 Day 4: Cross-Platform Parity Design

## Scope

Define the reviewed cross-platform quality contract for Sprint 36 before
changing CI or helper scripts.

This design needs to answer one central question clearly:

What does "parity" mean in this repo right now, given that Linux already
enforces the reviewed wrapper contract while macOS and Windows still validate
different but real platform-specific surfaces?

## Main Decision

Sprint 36 parity means **explicit reviewed-path interpretation by platform**,
not an immediate requirement that every platform execute the exact same
commands.

That is the correct design for the current repo state because:

- Linux already enforces the reviewed wrapper contract
- macOS already has substantial real build/test/sanitize/install coverage
- Windows already has a real CMake configure/build/ctest path
- dead-code portability and several Win32/macOS-specific exclusions remain
  staged rather than closed

## Reviewed Contract By Platform

### Linux

Linux remains the enforced source-of-truth baseline:

- reviewed compile-quality path:
  - `make quality-review-compile`
- reviewed CMake parity path:
  - `make quality-review-cmake`
- enforced dead-code path:
  - `make deadcode-report`
  - `make deadcode-check`

Interpretation:

- Linux continues to define the strongest practical reviewed contract in Sprint
  36

### macOS

macOS parity should be expressed as:

- enforced current real coverage:
  - build
  - test
  - wall-check
  - Apple Clang sanitize
  - Homebrew GCC matrix leg
  - install/pkg-config validation
- staged reviewed-wrapper alignment:
  - reviewed Makefile entrypoint wording
  - reviewed CMake parity wording
- staged dead-code parity:
  - explicit status/reporting, not fake enforcement

Interpretation:

- Sprint 36 should align macOS workflow naming and entrypoint intent with the
  reviewed contract without deleting macOS-specific value

### Windows

Windows parity should be expressed as:

- enforced current real coverage:
  - CMake configure
  - CMake build
  - CMake `ctest`
- staged reviewed-wrapper alignment:
  - reviewed CMake parity wording
  - test-count parity framing
- staged dead-code parity:
  - explicit staged/unavailable status
- explicit excluded surfaces:
  - pthread-based tests
  - POSIX-tempfile fuzz test
  - POSIX-bound benchmark set

Interpretation:

- Sprint 36 should make the Windows contract more explicit before it tries to
  broaden it

## Enforced vs Staged vs Excluded Model

Sprint 36 will classify platform quality surfaces into three buckets:

### Enforced

Commands or paths that already run as the maintained CI contract for that
platform.

### Staged

Reviewed surfaces that are relevant to that platform but are not yet fully
enforced there. These must be named explicitly so the repo does not overclaim
parity.

### Excluded

Surfaces intentionally not covered on that platform because they depend on:

- unsupported runtime/tooling
- POSIX-specific APIs
- currently unported build/test assumptions

This model is load-bearing because it avoids two failure modes:

- pretending all platforms are already equivalent
- treating every platform difference as technical debt that must be erased in
  Sprint 36

## Dead-Code Design Decision

Sprint 36 will **not** pretend dead-code is a portable all-platform quality
gate yet.

Why:

- compile-db coverage gap still exists
- execution still relies on shared build/artifact paths
- `xunused` setup is materially different from ordinary compiler/test setup

Chosen contract:

- Linux: dead-code remains enforced
- macOS: dead-code is staged/unavailable unless Sprint 36 explicitly adds a
  safe supported path
- Windows: dead-code is staged/unavailable unless Sprint 36 explicitly adds a
  safe supported path

Interpretation:

- Sprint 36 should improve truthfulness of dead-code parity reporting, not
  overclaim portability that the repo has not actually shipped

## Source-Of-Truth Baseline

The reviewed local/CMake contract from Sprint 34/Sprint 35 remains the anchor:

- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`
- `53` registered CTest tests

Later platform-specific reporting should point back to this baseline when it
describes:

- which platform surfaces are fully aligned
- which are partial/staged
- which remain excluded

## Implementation Order For Days 5-10

### Day 5: macOS workflow alignment

Primary file:

- `.github/workflows/macos-ci.yml`

Focus:

- reviewed-path wording
- entrypoint alignment
- preserve wall-check, sanitize, and install/pkg-config value

### Day 6: Windows workflow alignment

Primary file:

- `.github/workflows/windows-ci.yml`

Focus:

- reviewed CMake parity wording
- explicit staged/excluded surface framing

### Day 7: portability audit

Primary files:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`

Focus:

- shell/tool assumptions
- path/model assumptions
- stage-vs-enforce truthfulness

### Day 8: portability fixes

Primary surfaces:

- reviewed wrapper support logic
- dead-code helper portability/reporting logic

### Day 9: CI expectation refinement

Primary files:

- all three workflow YAML files

Focus:

- consistent naming
- explicit platform contract expression

### Day 10: parity report

Primary output:

- compact enforced/staged/excluded platform map

## Bottom Line

The Day 4 design makes Sprint 36 concrete:

- Linux stays the enforced reviewed baseline
- macOS and Windows are aligned toward that contract truthfully rather than
  performatively
- dead-code remains a special staged surface outside fake all-platform parity
- the next implementation days now have a fixed sequence instead of an ad hoc
  "improve portability" loop
