# Sprint 36 Day 11: Final Parity Consistency Pass

## Scope

Close the remaining small naming and reporting mismatches surfaced by the Day 10
parity report.

Files changed:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`

## Main Result

The enforced/staged/supplemental contract is now consistent across:

- workflow comments
- workflow job names
- workflow step names
- the README parity section
- the Day 10 parity report

This was a consistency pass only. It did not broaden any platform’s actual
quality enforcement scope.

## Changes

### 1. Workflow step names now reflect the real contract

Adjusted step labels so the CI output itself uses the same vocabulary as the
parity report:

- Linux:
  - reviewed Makefile / CMake / dead-code = `enforced`
  - runtime / ASan / UBSan / bench-fast / TSan / coverage = `supplemental`
- macOS:
  - Apple Clang reviewed / wall-check / sanitize = `enforced`
  - GCC direct build/test and install/pkg-config = `supplemental`
- Windows:
  - reviewed CMake configure/build/`ctest -N`/`ctest` = `enforced`

### 2. README now states the Windows distinction explicitly

The README now says directly that Windows currently enforces:

- configure
- build
- `ctest -N`
- full `ctest`

And that this is **not yet** equivalent to claiming full local Makefile
reviewed-wrapper parity on Windows.

## Validation

Validated workflow YAML:

```bash
ruby -e 'require "yaml"; %w[.github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml].each { |p| YAML.load_file(p); puts "yaml_ok #{p}" }'
```

Result:

- `yaml_ok .github/workflows/ci.yml`
- `yaml_ok .github/workflows/macos-ci.yml`
- `yaml_ok .github/workflows/windows-ci.yml`

## Conclusion

Day 11 closed the remaining contract-language drift without reopening broader
platform work. Sprint 36 is now in a clean state for the validation days:

- same platform model in workflows
- same platform model in README
- same platform model in the parity report
