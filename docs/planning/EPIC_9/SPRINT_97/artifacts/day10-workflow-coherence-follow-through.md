# Sprint 97 Day 10: Workflow Coherence Follow-Through

## Purpose

Day 10 reconciles local command names, CI workflow labels, expected counts, and
package-proof language after the Day 5 source-list guard and the Day 8-9
static-first package proof updates.

## Reviewed Surfaces

Workflow and command surfaces reviewed:

- `Makefile`
- `README.md`
- `INSTALL.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- Sprint 97 Day 5-9 artifacts

## Update Batch

Day 10 updates one stale front-door command-map line in `README.md`.

Before:

```text
make quality-review-compile  # reviewed format-check + lint wrapper
```

After:

```text
make quality-review-compile  # reviewed format-check + source-list-check + lint wrapper
```

This matches the current `Makefile` target, which now runs:

1. `format-check`
2. `source-list-check`
3. `lint`

## Workflow Coherence Findings

### Linux

`.github/workflows/ci.yml` remains coherent:

- `lint` job name: Linux enforced reviewed Makefile compile-quality path
- command: `make quality-review-compile`
- inherited Day 5 guard: `source-list-check`

No workflow YAML edit is needed because the job already calls the target whose
phase list changed.

### macOS

`.github/workflows/macos-ci.yml` remains coherent:

- Apple Clang leg runs `make quality-review-compile`
- Apple Clang leg runs `make quality-review-cmake`
- supplemental Make install/`pkg-config` job remains static-first confidence,
  not reviewed install/export parity

No workflow YAML edit is needed because the workflow labels remain accurate and
the Day 9 no-shared-artifact proof is inherited through `tests/test_install.sh`.

### Windows

`.github/workflows/windows-ci.yml` remains coherent:

- reviewed scope is still CMake-first consumer proof
- expected CTest count remains `51`
- staged exclusions remain `test_threads`, `test_sprint4_integration`, and
  `test_fuzz`
- no reviewed Makefile parity or separate reviewed install-validation lane is
  implied

No workflow YAML edit is needed because Days 5-9 did not change Windows CMake
test registration or reviewed Windows package ownership.

### Local Commands

Local command wording now aligns:

- `README.md` describes `quality-review-compile` as
  `format-check + source-list-check + lint`
- `INSTALL.md` describes the same phase list
- `Makefile` prints the same phase list
- `source-list-check` remains separately runnable
- `quality-review-cmake-compile` retains its CTest-count parity assertion

## Preserved Differences

Intentional differences after Day 10:

- Linux remains the strongest reviewed source of truth.
- macOS install/`pkg-config` proof remains supplemental static-first
  confidence.
- Windows remains reviewed CMake-first consumer proof only.
- Install/export scripts remain local package proof surfaces, with macOS
  inheriting only the supplemental Make install/`pkg-config` path.
- Shared-library packaging remains deferred and has no workflow lane.

## Validation

Focused validation for Day 10:

```sh
python3 scripts/check_library_sources.py
make source-list-check
git diff --check
rg -n "[ \t]+$" README.md docs/planning/EPIC_9/SPRINT_97
```

Observed results:

- direct source-list checker: `source-list-check: PASS (42 library sources)`
- Make target: `source-list-check: PASS (42 library sources)`
- `git diff --check`: passed
- trailing-whitespace scan: passed with no matches
- README, INSTALL, and Makefile agree on the
  `quality-review-compile` phase list
- no workflow YAML change is needed
- Windows expected CTest count remains 51

No `.c` or `.h` files are modified by this workflow-coherence update, so the
full `make format && make lint && make test` chain is not required.

## Day 10 Result

Local command help, public workflow guidance, and CI workflow labels now match
the build/package story landed in Days 5-9. The remaining platform differences
are explicit proof-boundary choices, not contradictory workflow claims.
