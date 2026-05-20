# Sprint 36 Day 12: Platform-Focused Validation

## Scope

Run the practical local validation paths that Sprint 36 changed or re-framed,
re-check the Day 10 parity-report claims against live command behavior, and pin
the exact Day 13 full-sweep command set before closeout.

Files changed:

- `docs/planning/EPIC_3/SPRINT_36/WORKING_NOTES.md`
- `docs/planning/EPIC_3/SPRINT_36/artifacts/day12-platform-focused-validation.md`

## Commands Run

```bash
ruby -e 'require "yaml"; %w[.github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml].each { |p| YAML.load_file(p); puts "yaml_ok #{p}" }'
make wall-check
make quality-review-compile
make quality-review-cmake-compile
make deadcode-report
make deadcode-check
make sanitize
```

## Main Results

### 1. The maintained reviewed paths still behave exactly as Sprint 36 claims

The local reviewed-quality entry points that Sprint 36 aligned all passed:

- `make quality-review-compile`
- `make quality-review-cmake-compile`
- `make deadcode-report`
- `make deadcode-check`

This confirms that the repo still has a live practical baseline for the
platform contract described earlier in the sprint:

- reviewed Makefile compile-quality path works
- reviewed CMake parity path works
- dead-code reporting/checking still works on the Linux-style local setup

### 2. The CMake parity contract remains exact and auditable

`make quality-review-cmake-compile` still reported:

- `ctest -N`: `53`
- Makefile tests: `53`
- CMake tests: `53`
- `PASS: test counts match`

Interpretation:

- the reviewed CMake parity contract is still the strongest honest
  cross-platform reviewed baseline
- Sprint 36 did not drift the active-suite parity story while aligning CI and
  reporting

### 3. The staged dead-code story remains truthful

The dead-code paths still work locally in the supported serialized form:

- `make deadcode-report`
- `make deadcode-check`

The important point for Sprint 36 is not just success. It is that the current
status is still represented honestly:

- locally available and passing on the Linux-style toolchain path
- staged on macOS
- excluded on Windows

Nothing in Day 12 suggested that Sprint 36 accidentally overclaimed broader
dead-code portability than the repo actually has.

### 4. The macOS-side enforced helper path still has live local evidence

Two Apple-Clang-adjacent helper validations also passed:

- `make wall-check`
- `make sanitize`

`make wall-check` ended with `wall-check: PASS`.

`make sanitize` completed the full UBSan/ASan-oriented test pass and ended with
`All tests passed.` The long-tail suites that matter most for Sprint 36’s
platform claims also completed cleanly, including:

- `test_reorder_nd`
- `test_chol_csc`
- `test_ldlt_csc`
- `test_sprint18_integration`
- `test_sprint19_integration`
- `test_sprint20_integration`
- `test_eigs`
- `test_eigs_lobpcg`
- `test_graph`
- `test_framework_optin`

Interpretation:

- the Apple-Clang-oriented reviewed/supporting contract still has real local
  validation behind it
- Sprint 36’s wording around enforced Apple Clang paths and supplemental helper
  flows remains grounded in working commands, not stale assumptions

### 5. Workflow YAML still parses cleanly after the wording/alignment passes

All three CI workflow files loaded successfully through Ruby YAML parsing:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

Interpretation:

- the Day 5/6/9/11 workflow-alignment work remains structurally valid
- Day 13 can focus on the full sweep instead of reopening workflow syntax drift

## Day 12 Interpretation

- Day 12 succeeded because the Sprint 36 changes were mostly contract-alignment
  work around already-real command paths.
- The high-signal result is that the parity report is still operationally
  truthful when checked against live commands:
  - reviewed Makefile compile path works
  - reviewed CMake parity path works
  - dead-code report/check still works in its supported serialized model
  - macOS-side helper/support commands still have live local evidence
- No new portability or parity queue reopened. Sprint 36 can move into the Day
  13 full validation sweep without carrying unresolved contract drift.

## Day 13 Planned Full Sweep

```bash
make format
make lint
make test
make quality-review-compile
make quality-review
make quality-review-cmake-compile
make quality-review-cmake
```

## Conclusion

Day 12 verified that Sprint 36’s platform-parity story is not just documented;
it still matches the maintained local command behavior. The sprint is now in a
clean state for the final full validation sweep.
