# Sprint 36 Day 5: macOS Workflow Alignment

## Scope

Implement the first macOS parity batch by aligning the Apple Clang CI leg with
the reviewed Sprint 34 contract while preserving the extra macOS-specific value
already present in the workflow.

The point of Day 5 is not to flatten the macOS matrix into the Linux workflow.
It is to make the reviewed contract explicit where it belongs and keep the
useful platform-specific coverage that does not need to disappear.

## Files Changed

- `.github/workflows/macos-ci.yml`

## Main Result

The Apple Clang leg now acts as the reviewed macOS baseline:

- installs reviewed-path tools:
  - Homebrew `llvm`
  - Homebrew `cppcheck`
- adds Homebrew LLVM `bin` to `PATH`
- runs:
  - `make quality-review-compile`
  - `make quality-review-cmake`
  - `make wall-check`
  - `make sanitize`

The Homebrew GCC leg remains intentionally direct coverage:

- `make`
- `make test`
- `make wall-check`

## What Changed

### 1. Reviewed-path tool setup was added for Apple Clang

The reviewed compile-quality path depends on:

- `clang-format`
- `clang-tidy`
- `cppcheck`

On macOS, the workflow now installs:

- `llvm`
- `cppcheck`

and then exports the Homebrew LLVM `bin` directory so the reviewed Makefile
path can resolve `clang-format` and `clang-tidy` consistently.

### 2. Apple Clang now uses reviewed wrapper entrypoints

The workflow now makes the reviewed contract explicit on the Apple Clang leg by
running:

- `make quality-review-compile`
- `make quality-review-cmake`

This closes the Day 2 gap where macOS CI had real coverage but still expressed
older direct build/test entrypoints rather than the reviewed wrapper layer.

### 3. Homebrew GCC remains a second-compiler signal, not fake parity

The Homebrew GCC leg was kept as direct coverage instead of being forced
through the reviewed wrapper contract.

That is intentional:

- Apple Clang is the reviewed macOS baseline
- Homebrew GCC remains additional compiler diversity coverage

This preserves the value of the macOS matrix without overstating what each leg
is supposed to mean.

### 4. macOS-only value was preserved

The workflow still retains:

- `wall-check`
- Apple Clang sanitize
- Homebrew GCC matrix leg
- install/pkg-config validation job

These remain useful platform-specific signals and are not parity failures.

## Validation

Day 5 changed workflow YAML only, so the required `make format && make lint &&
make test` gate for `*.c` / `*.h` changes did not apply.

Validation run:

- `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/macos-ci.yml"); puts "yaml_ok"'`

Result:

- `yaml_ok`

Additional sanity checks:

- inspected the resulting workflow file
- inspected the resulting diff

## Contract Delta

### Before Day 5

Apple Clang CI expressed:

- direct `make`
- direct `make test`
- `make wall-check`
- `make sanitize`

### After Day 5

Apple Clang CI expresses:

- reviewed compile-quality path
  - `make quality-review-compile`
- reviewed CMake parity path
  - `make quality-review-cmake`
- macOS regression path
  - `make wall-check`
- sanitizer path
  - `make sanitize`

## What Did Not Change

Day 5 intentionally did **not** add:

- `make deadcode-report`
- `make deadcode-check`

Dead-code remains staged on macOS pending later portability/reporting work, in
line with the Day 4 contract.

## Bottom Line

Day 5 closed the biggest macOS parity gap without overreaching:

- Apple Clang is now the explicit reviewed macOS baseline
- Homebrew GCC remains valuable second-compiler coverage
- wall-check, sanitize, and install/pkg-config validation are preserved
- dead-code remains staged rather than falsely claimed as portable
