# Sprint 97 Day 5: Source-List Reduction Batch 1

## Purpose

Day 5 lands the first bounded build-topology reduction batch from the Day 4
boundary freeze. The batch adds a library-source manifest, a parity checker,
and a Make reviewed-path hook without changing library behavior or package
claims.

## Implemented Files

New files:

- `build-metadata/library_sources.txt`
- `scripts/check_library_sources.py`

Updated files:

- `Makefile`
- `docs/planning/EPIC_9/SPRINT_97/WORKING_NOTES.md`
- `docs/planning/EPIC_9/SPRINT_97/artifacts/day5-source-list-reduction-batch1.md`

## Manifest

`build-metadata/library_sources.txt` now lists the 42 library source files in
the current reviewed Makefile order.

Manifest rules implemented by convention and checker validation:

- one source path per non-empty line
- comments start with `#`
- paths are repo-relative
- entries use forward slashes
- entries are library implementation sources under `src/*.c`

The manifest is intentionally not generated. It is the durable
source-membership reference for the first convergence batch.

## Checker

`scripts/check_library_sources.py` checks source membership and order across:

- `build-metadata/library_sources.txt`
- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`

Checker behavior:

- uses Python 3 standard library only
- does not shell out to Make or CMake
- normalizes `$(SRCDIR)/...` Make paths to `src/...`
- parses the current CMake `add_library` block directly
- reports duplicates
- reports missing, extra, or order-mismatched entries
- exits non-zero on drift or parser failure
- prints a concise pass line on success

Successful direct output:

```text
source-list-check: PASS (42 library sources)
```

## Makefile Integration

Added:

```make
.PHONY: source-list-check
source-list-check:
	@python3 scripts/check_library_sources.py
```

Updated `quality-review-compile` so the reviewed compile-quality path now runs:

1. `format-check`
2. `source-list-check`
3. `lint`

This keeps source-list drift visible before the expensive lint/static-analysis
phase.

## Preserved Surfaces

The Day 5 batch deliberately did not change:

- Make `LIB_SRCS` membership or order
- CMake `add_library(...)` membership or order
- test registration
- benchmark registration
- example registration
- install/export behavior
- package wording
- platform workflow messages
- Windows expected CTest count
- `.c` or `.h` files

## Validation

Required Day 5 validation passed:

```sh
python3 scripts/check_library_sources.py
make source-list-check
make quality-review-compile
```

Observed results:

- direct checker: `source-list-check: PASS (42 library sources)`
- Make target: `source-list-check: PASS (42 library sources)`
- reviewed compile-quality path:
  `quality-review-compile: passed (format-check + source-list-check + lint)`

No `.c` or `.h` files were modified, so the full
`make format && make lint && make test` chain was not required by the Day 4
validation plan.

## Residual Queue

Carry forward to Day 6:

- decide whether `source-list-check` should also be wired into another local
  reviewed path such as `quality-review-cmake-compile`
- decide whether CMake configure-time parity checking is worth the extra
  dependency and configuration complexity
- decide whether the checker should grow a self-test fixture or stay narrow
  and parser-driven
- keep test registration centralization deferred until platform exclusions and
  CTest count assertions can be preserved explicitly
- keep benchmark and example registration residual unless a later manifest
  pattern proves clearly worth reusing

## Day 5 Result

Sprint 97 now has an enforced source-list parity guard. The first build
topology reduction does not remove all duplicate source-list text, but it
reduces silent drift risk while preserving reviewed Make and CMake readability.
