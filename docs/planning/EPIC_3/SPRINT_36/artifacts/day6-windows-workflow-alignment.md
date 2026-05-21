# Sprint 36 Day 6: Windows Workflow Alignment

## Scope

Implement the first Windows parity batch by making the current Win32/MSVC
workflow express its reviewed CMake subset explicitly, while preserving the
staged exclusion model identified on Day 3.

The point of Day 6 is not to force Windows into the full Linux reviewed
wrapper contract immediately. It is to make the existing Windows contract
truthful, named, and auditable.

## Files Changed

- `.github/workflows/windows-ci.yml`

## Main Result

The Windows workflow now frames itself as a reviewed CMake parity subset:

- reviewed CMake configure path
- reviewed CMake build path
- `ctest -N` visibility into the registered Windows suite
- full `ctest` execution

It also now asserts the current staged Windows suite size directly:

- `EXPECTED_WINDOWS_CTEST_COUNT=50`

## What Changed

### 1. Workflow framing was updated

The top-of-file commentary now states directly that Windows is currently:

- CMake-first
- a reviewed CMake parity subset
- not yet the full Linux reviewed-wrapper contract

It also now names the staged exclusions directly:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

### 2. Step naming now reflects the real contract

The workflow steps now read as:

- reviewed CMake configure path
- reviewed CMake build path
- reviewed Windows CTest surface (`ctest -N`)
- reviewed CMake execution path (`ctest`)

This closes the Day 3 gap where the workflow was technically real but still
looked like a generic draft build/test path.

### 3. The staged Windows test-count surface is now explicit

The workflow now:

- runs `ctest -N`
- extracts the `Total Tests:` line
- asserts that the current Windows suite count remains `50`
- prints the named staged exclusions

This gives Windows a real parity/reporting signal instead of only a final
pass/fail `ctest` result.

## Validation

Day 6 changed workflow YAML only, so the required `make format && make lint &&
make test` gate for `*.c` / `*.h` changes did not apply.

Validation run:

- `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/windows-ci.yml"); puts "yaml_ok"'`

Result:

- `yaml_ok`

Additional sanity checks:

- inspected the resulting workflow file
- inspected the resulting diff

## Contract Delta

### Before Day 6

Windows CI expressed:

- CMake configure
- CMake build
- CMake `ctest`

### After Day 6

Windows CI expresses:

- reviewed CMake configure path
- reviewed CMake build path
- reviewed Windows CTest surface (`ctest -N`)
- full reviewed CMake execution path (`ctest`)
- staged Windows suite count expectation:
  - `50`
- staged exclusion list:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`

## What Did Not Change

Day 6 intentionally did **not** add:

- `make quality-review-compile`
- `make quality-review`
- `make deadcode-report`
- `make deadcode-check`

Windows remains a reviewed CMake subset plus explicit staged exclusions, in
line with the Day 4 design.

## Bottom Line

Day 6 closed the biggest Windows reporting/contract gap without overreaching:

- the workflow now states what Windows actually validates
- the staged suite count is explicit and checked
- the current exclusions are surfaced directly
- Windows is still truthfully staged relative to the stronger Linux reviewed
  contract
