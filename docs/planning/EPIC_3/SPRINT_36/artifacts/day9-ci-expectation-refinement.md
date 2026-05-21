# Sprint 36 Day 9: CI Expectation Refinement

## Scope

Refine the platform CI contract so Linux, macOS, and Windows all state their
reviewed-quality expectations explicitly and consistently.

Files changed:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`

## Main Result

Sprint 36 now has a consistent enforced/staged/supplemental vocabulary across:

- the Linux workflow
- the macOS workflow
- the Windows workflow
- the main operator-facing README

This does not broaden platform enforcement beyond what is actually implemented.
It makes the current contract visible and operationally truthful.

## Changes

### 1. Linux workflow now states its stronger role directly

Updated `.github/workflows/ci.yml` to say explicitly that Linux is:

- the enforced reviewed Makefile compile-quality baseline
- the enforced reviewed CMake parity baseline
- the enforced dead-code report/check baseline

It also now labels Linux-only extra signals as supplemental:

- direct runtime + `bench-fast`
- TSan
- coverage

### 2. macOS workflow now uses the same contract language

Updated `.github/workflows/macos-ci.yml` to classify its surfaces more clearly:

- Apple Clang reviewed leg = enforced
- Homebrew GCC leg = supplemental
- dead-code = staged
- install/pkg-config = supplemental

### 3. Windows workflow now uses the same contract language

Updated `.github/workflows/windows-ci.yml` to classify its surfaces more
clearly:

- reviewed CMake subset = enforced
- reviewed Makefile wrappers = staged
- dead-code = staged
- excluded tests remain named explicitly:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`

### 4. README now mirrors the same platform contract

Added a `Cross-Platform CI Contract` section to `README.md` so maintainers do
not need to reconstruct the current parity state from three separate workflow
files.

The section captures:

- enforced paths by platform
- staged paths by platform
- supplemental or excluded paths by platform

## Validation

Validated the workflow YAML directly:

```bash
ruby -e 'require "yaml"; %w[.github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml].each { |p| YAML.load_file(p); puts "yaml_ok #{p}" }'
```

Result:

- `yaml_ok .github/workflows/ci.yml`
- `yaml_ok .github/workflows/macos-ci.yml`
- `yaml_ok .github/workflows/windows-ci.yml`

Also re-read the new README section against the live workflow contracts.

## Conclusion

Day 9 closes the remaining expectation gap from the Day 4 design:

- Linux is explicitly the strongest enforced baseline
- macOS and Windows explicitly document their narrower enforced scopes
- staged and supplemental surfaces are named rather than implied

That leaves Day 10 ready to produce a compact parity report from a stable,
already-aligned contract.
