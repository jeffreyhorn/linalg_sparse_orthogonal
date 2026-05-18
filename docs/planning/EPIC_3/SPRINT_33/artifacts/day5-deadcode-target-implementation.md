# Sprint 33 Day 5 Deadcode Target Implementation

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Implement the first runnable Sprint 33 dead-code workflow so maintainers can refresh a dedicated compilation database, run the agreed raw analysis tools from one command, and preserve the known compile-db coverage gap in stable local artifacts for later reporting work.

## What Shipped

Day 5 added two operator-facing pieces:

1. `make deadcode-compile-db`
2. `make deadcode`

Implementation split:

- `Makefile`
  - thin operator targets
  - dedicated dead-code build path:
    - `build/deadcode-cmake`
  - dedicated local artifact root:
    - `build/deadcode`
- `scripts/deadcode_workflow.sh`
  - prerequisite checks
  - compile-database existence check
  - coverage-gap note generation
  - raw `cppcheck` capture
  - raw `xunused` capture

## Actual Day 5 Invocation Contract

The Day 4 command sketch needed two practical refinements during implementation.

### `cppcheck`

Raw Day 4 sketch:

```sh
cppcheck --enable=all --quiet src/
```

Actual Day 5 command:

```sh
cppcheck \
  --enable=all \
  --quiet \
  --std=c11 \
  --suppress=missingIncludeSystem \
  -Iinclude \
  -Ibuild/include \
  -Isrc \
  src/
```

Reason:

- the bare command produced too much include-resolution noise to be useful as a raw artifact on this repo
- adding the repo include paths preserved the “raw evidence” intent while making the output materially more reviewable

### `xunused`

Raw Day 4 sketch:

```sh
xunused build/deadcode-cmake/compile_commands.json
```

Actual Day 5 macOS behavior:

```sh
LLVM_PREFIX="$(brew --prefix llvm@18)"
xunused \
  --extra-arg-before=-isysroot \
  --extra-arg-before="$(xcrun --sdk macosx --show-sdk-path)" \
  --extra-arg-before=-resource-dir="$("$LLVM_PREFIX/bin/clang" -print-resource-dir)" \
  build/deadcode-cmake/compile_commands.json
```

Reason:

- local macOS parsing failed without an SDK sysroot
- after adding the sysroot, builtin-header parsing still failed on `stdatomic.h`
- adding the LLVM resource directory resolved the remaining parser issue

The helper script applies the Darwin-specific extra args automatically.

## Validation

End-to-end Day 5 validation:

1. `bash -n scripts/deadcode_workflow.sh`
2. `make deadcode-compile-db`
3. `make deadcode`

Results:

- workflow exit status: success
- dedicated compile database refreshed successfully
- raw artifacts written successfully

Generated files:

- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`
- `build/deadcode/coverage-notes.txt`

Final artifact sizes from the successful rerun:

- `cppcheck.txt`: `920` lines
- `xunused.txt`: `107` lines
- `coverage-notes.txt`: `17` lines

## Coverage-Gap Output

The Day 5 workflow now emits the known compilation-database gap explicitly instead of hiding it in sprint notes.

Captured in `build/deadcode/coverage-notes.txt`:

- missing benchmark:
  - `bench_svd`
- missing examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

That preserves the critical Day 3 policy caveat:

- absence from `xunused` output is not proof of absence from the full Makefile tooling surface

## Initial Raw Findings

### `xunused`

Current compile-db-backed `xunused` warnings:

- `chol_csc_dump_supernodes`
- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

Interpretation:

- `xunused` is currently the narrower, more specific dead-code signal
- even so, the list is not auto-delete-ready because several candidates are exposed through installed public headers and therefore need a separate public-surface review bucket

### `cppcheck`

Top recurring raw IDs:

- `constVariablePointer`: `106`
- `staticFunction`: `90`
- `unusedFunction`: `80`
- `normalCheckLevelMaxBranches`: `23`

Interpretation:

- `cppcheck` is useful evidence input
- it is much broader than dead-code detection alone
- Sprint 33 still needs a reporting/classification layer before the output is suitable for cleanup decisions or enforcement semantics

## Day 5 Conclusion

Day 5 successfully delivered the first reusable dead-code workflow layer:

- reproducible compile-db refresh
- reproducible raw artifact generation
- explicit coverage-gap disclosure
- working local `xunused` integration on macOS

The main follow-on requirement is now clear: Day 6 and Day 7 should focus on turning these raw artifacts into a categorized report rather than adding more ad hoc scanner plumbing.
