# Sprint 59 Day 3 - quality/platform residual audit

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Reduce the remaining Sprint 59 quality/platform queue to concrete residual
classes before any follow-through batch lands:

- serialized dead-code execution
- macOS dead-code staging
- Windows reviewed-wrapper parity
- Windows dead-code exclusion
- coverage calibration

## Re-audited surfaces

- `README.md`
- `docs/maintainer_guide.md`
- `Makefile`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- Epic 5 review/todo notes

## Main findings

### 1. The residual queue is smaller than the raw project-plan list suggests

The live repo already has more quality/platform convergence than the Sprint 59
headline implies:

- Linux enforced reviewed paths are live in CI.
- Windows enforced reviewed CMake subset is live in CI.
- macOS enforced Apple Clang reviewed path is live in CI.
- coverage is already a live supplemental CI signal.

Conclusion:

- Sprint 59 is not recovering a broken or missing platform story.
- It is reducing the remaining staged/excluded limits to a smaller set of
  defensible residuals.

### 2. Coverage calibration is no longer an active residual

Coverage is now already in a steady-state posture:

- `Makefile` enforces `COV_THRESHOLD = 80`
- Linux CI runs `make coverage`
- README and maintainer-facing wording agree that coverage is a supplemental
  signal, not part of the reviewed baseline

Conclusion:

- coverage calibration should drop out of the active Sprint 59 implementation
  queue
- classification:
  - no longer justified by the current repo state

### 3. Serialized dead-code execution remains explicit and still justified

The live dead-code surfaces still agree that execution remains serialized:

- `Makefile` marks the `deadcode*` targets `.NOTPARALLEL`
- `docs/maintainer_guide.md` says to run them serially because they share
  `build/deadcode-cmake` and `build/deadcode/`
- Linux CI keeps dead-code in one serial job for the same reason

Conclusion:

- this is not a hidden truthfulness problem
- it remains an operational limit that is already documented honestly
- classification:
  - already acceptable as deferred residual

### 4. macOS dead-code staging is the strongest remaining measurement seam

The macOS story is now narrow and concrete:

- `macos-ci.yml` enforces the reviewed compile/CMake path plus
  `wall-check`/`sanitize`
- README still stages:
  - `make deadcode-report`
  - `make deadcode-check`
- no maintained macOS CI job currently runs the dead-code workflow

Conclusion:

- this remains a real residual
- but the repo does not yet provide fresh evidence that it is either ready or
  clearly impossible on macOS
- classification:
  - needs measurement before any change

### 5. Windows reviewed-wrapper parity and dead-code remain real but honestly bounded residuals

The Windows story is also narrower than the inherited review summary alone:

- `windows-ci.yml` enforces the reviewed CMake subset:
  - configure
  - build
  - `ctest -N`
  - full `ctest`
- README keeps Windows Makefile reviewed wrappers and dead-code staged
- the workflow itself prints the current excluded tests explicitly

Conclusion:

- Windows is not missing from CI
- the remaining queue is the staged local reviewed-wrapper path and the
  dead-code exclusion, not reviewed CMake parity itself
- classification:
  - already acceptable as deferred residual

## Residual classes

### Already acceptable as deferred residual

- serialized dead-code execution
- Windows reviewed-wrapper parity
- Windows dead-code exclusion

### Needs measurement before any change

- macOS dead-code staging

### Needs bounded follow-through now

- cross-surface residual-disposition wording reconciliation

### No longer justified by the current repo state

- coverage calibration as a standalone active residual

## Ranked follow-through order

1. macOS dead-code staging measurement
2. bounded residual-disposition reconciliation across maintained
   quality/platform surfaces
3. preserve serialized dead-code execution as an explicit deferred limit
4. preserve Windows reviewed-wrapper parity as an explicit deferred limit
5. preserve Windows dead-code exclusion as an explicit deferred limit
6. remove coverage calibration from the active residual queue

## Proposed first implementation boundary

The cleanest Day 4 target is not a broad platform expansion batch. It is a
bounded residual-disposition reconciliation batch across:

- `README.md`
- `docs/maintainer_guide.md`
- `Makefile`
- workflow comments only where they materially disagree

The likely non-goal fence is also explicit:

- no forced full macOS dead-code convergence without measurement
- no forced Windows Makefile/dead-code parity expansion for cosmetic symmetry

## Conclusion

Day 3 closes with a ranked, defensible residual map:

- coverage calibration drops out as an active Sprint 59 item
- serialized dead-code execution remains an explicit and acceptable deferred
  limit
- macOS dead-code staging is the strongest remaining measurement seam
- Windows reviewed-wrapper parity and dead-code remain honestly bounded staged
  residuals
- the best first follow-through target is bounded residual-disposition
  reconciliation, not broad platform ambition
