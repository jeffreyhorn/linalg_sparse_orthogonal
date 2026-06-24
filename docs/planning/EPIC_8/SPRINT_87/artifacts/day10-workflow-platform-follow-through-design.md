# Sprint 87 Day 10: Workflow / Platform Follow-Through Design

## Purpose

Define the bounded cross-platform quality convergence package Sprint 87 can
truthfully maintain after the stronger Day 9 local consumer proof.

## Main Result

Sprint 87 now has one explicit third implementation contract:

- required Day 11 center:
  - `.github/workflows/macos-ci.yml`
- directly forced support-only follow-through if the workflow batch truly
  changes maintained package-flow wording:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- strongest adjacent support-only workflow surfaces only if the macOS batch
  exposes a real shared contract seam:
  - `.github/workflows/ci.yml`
  - `.github/workflows/windows-ci.yml`
- lower-value non-touch surfaces:
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`

## Exact Day 11 Center

The exact Day 11 implementation center is now fixed to one bounded macOS
workflow follow-through seam, not a generic CI expansion and not a Windows
scope widening batch.

The decisive Day 10 reason is explicit:

- Day 9 strengthened the maintained local Make/pkg-config consumer proof in
  `tests/test_install.sh`
- the current macOS supplemental install job still runs a thinner manual
  install plus `pkg-config` check than the maintained local proof owner
- Windows already states a narrower reviewed CMake-first consumer story and
  does not claim a separate reviewed install-validation lane
- the highest-value next seam is therefore to align the macOS supplemental job
  more closely to the maintained local proof it is meant to support

## Best Workflow Lane

The strongest bounded Day 11 workflow lane is now fixed to:

- keep `.github/workflows/macos-ci.yml` as the implementation owner
- strengthen the macOS supplemental package job by reusing the maintained local
  Make/pkg-config proof surface rather than a thinner hand-rolled subset
- keep the lane explicitly supplemental rather than rebranding it as reviewed
  install/export parity
- leave Windows wording and scope untouched unless the macOS batch reveals a
  real shared contract phrase that must stay synchronized

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/windows-ci.yml`

Current reading:

- docs wording should stay untouched unless the Day 11 workflow batch truly
  changes the maintained CI rerun contract, not merely the internal
  implementation of the macOS supplemental lane
- `.github/workflows/windows-ci.yml` should stay untouched unless the macOS
  update exposes one shared wording seam about reviewed versus supplemental
  install evidence
- `.github/workflows/ci.yml` remains a support-only orchestrator surface, not
  the best next batch owner

## Preserved Fence

The bounded Day 10 fence is explicit:

- no broadened Windows install-validation claim
- no new shared-library or ABI promise
- no generic CI fan-out detached from the maintained package proof
- no conversion of supplemental macOS package evidence into reviewed parity
- no support-surface churn unless the landed workflow change truly alters the
  maintained contract reading

## Exit State

- Sprint 87 now has one exact third implementation contract.
- Day 11 can stay bounded to `.github/workflows/macos-ci.yml` and tighten the
  supplemental macOS package lane around the maintained local proof without
  widening broader platform claims.
- Windows scope and broader support-surface alignment remain explicitly later
  unless a real shared wording seam is forced.
