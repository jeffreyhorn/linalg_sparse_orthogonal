# Sprint 66 Day 4: Platform and Dead-Code Residual Recheck

Date: 2026-06-12
Branch: `sprint-66`

## Purpose

Reassess the live macOS, Windows, and dead-code residual queue against the
current reviewed truthfulness contract before Sprint 66 moves from audit into
design and implementation.

## Current Cross-Platform Contract

The repo already carries an intentionally asymmetric but explicit platform
contract:

- Linux enforces:
  - reviewed Makefile compile-quality path
  - reviewed CMake parity path
  - dead-code report and completeness path
- macOS enforces:
  - reviewed Apple Clang quality path
  - reviewed CMake parity path
  - supplemental install + `pkg-config` verification
  - dead-code remains staged
- Windows enforces:
  - reviewed CMake configure/build/`ctest -N`/full `ctest`
  - Makefile reviewed wrappers remain staged
  - dead-code remains staged

This means Sprint 66 is not starting from a vague "some platform work is left."
It is starting from a repo that already distinguishes enforced, staged, and
supplemental lanes on purpose.

## Ranked Residuals

### 1. Strongest real residual: staged-lane interpretation still needs tighter contract ownership

The strongest remaining platform-quality gap is not a missing CI job by itself.
It is that the staged lanes still need sharper interpretation across:

- `README.md`
- `docs/maintainer_guide.md`
- `INSTALL.md`
- workflow comments and job names

That is the most valuable Day 4 narrowing because it keeps Sprint 66 focused on
truthful productization instead of broad platform expansion.

### 2. Windows reviewed-wrapper parity is weaker than the epic-level wording implied

Windows already routes to the reviewed CMake subset consistently across the
repo. The Makefile reviewed wrappers remain staged, but that is not a direct
contradiction because:

- the Makefile is Unix-oriented
- `INSTALL.md` already tells Windows users to use the CMake workflow
- the Windows workflow already enforces only the reviewed CMake subset

So the Windows wrapper gap is real only as a staged limit, not as the strongest
Sprint 66 bug.

### 3. macOS dead-code remains staged-by-design pending fresh measurement

macOS still does not ship a maintained dead-code path. The repo instead keeps:

- Apple Clang reviewed compile/test/sanitize surfaces
- Homebrew GCC supplemental coverage
- supplemental install/`pkg-config` verification
- dead-code explicitly staged

That makes macOS dead-code a real residual, but still a lower-priority one than
the packaging/install/release contract itself.

### 4. Windows dead-code is tied to the Linux-centered dead-code execution model

The active dead-code workflow still depends on:

- CMake compile database generation
- `bash`
- `python3`
- `cppcheck`
- `xunused`
- one serialized shared artifact topology

That makes Windows dead-code more than a missing workflow toggle. It is tied to
the current execution model of the dead-code system, which is still centered on
the Linux reviewed lane.

### 5. Serialized dead-code execution remains the clearest active operational limit

The `deadcode*` targets still share:

- `build/deadcode-cmake`
- `build/deadcode/`

The docs, Makefile, and Linux dead-code workflow all still reflect that serial
execution requirement. This remains a real operational limit, but it is not by
itself the best first Sprint 66 implementation target unless a later change can
improve truthfulness without widening into a broad dead-code redesign.

## First Target Set

The highest-value Sprint 66 platform/dead-code follow-through set is now:

- docs/workflow/contract reconciliation around the staged platform lanes
- install/package/platform truth alignment where the packaging batch changes the
  released story
- bounded residual wording cleanup where current platform/dead-code language is
  still too generic

## Explicit Deferred Set

The following should remain outside the first Sprint 66 platform lane unless
fresh measurement proves otherwise:

- Windows Makefile reviewed-wrapper parity
- Windows dead-code enforcement
- macOS dead-code enforcement
- broad dead-code topology redesign
- fake cross-platform closure beyond reviewed evidence

## Measured Hotspots

Measured Day 4 hotspot sizes for the main platform/dead-code truth surfaces:

- `README.md` = `1000`
- `docs/maintainer_guide.md` = `511`
- `INSTALL.md` = `206`
- `Makefile` = `897`
- `.github/workflows/ci.yml` = `221`
- `.github/workflows/macos-ci.yml` = `111`
- `.github/workflows/windows-ci.yml` = `57`

## Exit State

Sprint 66 Day 4 closes with:

- one sharper platform/dead-code residual map
- one explicit split between real Sprint 66 platform-quality work and deferred
  platform-expansion work
- one fixed interpretation that the strongest remaining platform job is
  contract reconciliation around staged lanes, not immediate fake parity
