# Sprint 70 Day 9: Validation and Platform Contract I

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Reduce the broad Epic 7 validation/platform truthfulness concern to one ranked
contract-preservation map so later implementation work knows exactly which
reviewed, supplemental, staged, and excluded claims must not drift.

## Surfaces Audited

- `Makefile`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `scripts/bench_canonical_report.sh`

## Ranked Truthfulness Sensitivities

### 1. Strongest platform sensitivity: reviewed-subset asymmetry

The current reviewed-platform contract is intentionally asymmetric:

- Linux is the strongest reviewed source of truth
- macOS is narrower and mixes reviewed plus supplemental lanes
- Windows is a reviewed CMake subset only, with explicit staged exclusions

This remains the highest-risk truthfulness seam because future work could
easily over-read overall repository maturity into fake cross-platform parity
claims.

### 2. Strongest install/package sensitivity: local proof scripts versus reviewed CI interpretation

The install/package proof split remains:

- local Unix-side regression scripts:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- supplemental platform evidence:
  - macOS Make install/`pkg-config`
- reviewed platform reading:
  - Windows CMake-consumer lane only
  - no separate reviewed install-validation lane on Linux or Windows

This is a major drift risk because the local scripts are strong maintained
proof surfaces, but they do not by themselves widen the reviewed CI contract.

### 3. Strongest benchmark/reporting sensitivity: threshold-free canonical report reading

The canonical maintained benchmark/report surface remains intentionally narrow:

- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`
- four canonical proof benchmarks only

The key contract is unchanged:

- artifact-friendly snapshot
- not a pass/fail timing gate
- not portable performance proof by itself

This remains a high-risk wording seam for later Epic 7 work because capability
or backend improvements could easily tempt broader performance claims than the
current governance model supports.

### 4. Strongest operational quality residual: staged/serialized dead-code contract

The dead-code contract remains:

- Linux enforced
- macOS staged pending fresh measurement
- Windows staged

with the explicit operational limit that the workflow remains serialized around
shared dead-code build/artifact paths.

This is still a real residual, but it is already truthfully bounded and should
not be treated as if it were solved cross-platform.

## Lower-Risk Contract Areas

The following surfaces are comparatively lower-risk right now:

- static-first packaging shape
  - already coherent across README, INSTALL, maintainer guide, and local proof
    scripts
- canonical benchmark-set definition
  - already explicit and bounded
- local reviewed-wrapper naming
  - already coherent across Makefile and docs

These still matter, but they are not the strongest present drift points.

## Exit State

Sprint 70 Day 9 closes with one ranked contract-preservation map:

1. strongest platform truthfulness sensitivity:
   - reviewed-subset asymmetry across Linux / macOS / Windows
2. strongest install/package proof sensitivity:
   - local regression scripts versus reviewed CI-lane interpretation
3. strongest benchmark/reporting sensitivity:
   - threshold-free canonical-report reading
4. strongest operational quality residual:
   - staged/serialized dead-code contract

That gives Day 10 one exact job:

- freeze the validation, package, benchmark, and platform non-goal fence for
  later Epic 7 implementation without widening any current reviewed claims
