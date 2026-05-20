# Sprint 36 Day 13: Full Validation Sweep

## Scope

Run the full Sprint 36 validation set from the Day 12 plan, capture the measured
end state, and re-confirm that the reviewed local/CMake baseline still holds
after the parity and portability work.

Files changed:

- `docs/planning/EPIC_3/SPRINT_36/WORKING_NOTES.md`
- `docs/planning/EPIC_3/SPRINT_36/artifacts/day13-full-validation-sweep.md`

## Commands Run

```bash
/usr/bin/time -p make format
/usr/bin/time -p make lint
/usr/bin/time -p make test
/usr/bin/time -p make quality-review-compile
/usr/bin/time -p make quality-review
/usr/bin/time -p make quality-review-cmake-compile
/usr/bin/time -p make quality-review-cmake
```

## Main Results

### 1. The full maintained command set passed

After one clean-build reset, the complete Sprint 36 Day 13 validation set
passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`

Measured wall times:

- `make format`: `real 3.31`
- `make lint`: `real 303.32`
- `make test`: `real 264.17`
- `make quality-review-compile`: `real 696.45`
- `make quality-review`: `real 487.59`
- `make quality-review-cmake-compile`: `real 93.11`
- `make quality-review-cmake`: `real 817.17`

Interpretation:

- Sprint 36 closed with the entire maintained local reviewed-command set still
  green
- the new parity/reporting work did not regress either the direct commands or
  the wrapper entry points

### 2. The only Day 13 hiccup was a stale sanitizer build tree, not a repo regression

The first `make lint` attempt failed before the authoritative rerun because the
prior Day 12 `make sanitize` pass had left a UBSan-instrumented
`build/libsparse_lu_ortho.a` in place, and the benchmark link step then tried
to reuse it without the UBSan runtime.

The fix was:

```bash
make clean
```

Then the Day 13 sweep was rerun from a clean baseline and passed fully.

Interpretation:

- this was not a source regression in Sprint 36
- it is an operational note worth carrying into closeout:
  - the authoritative direct/wrapper validation sweep should start from a clean
    `build/` tree if a sanitizer path ran immediately before it

### 3. The reviewed CMake parity contract still matches exactly

`make quality-review-cmake-compile` again reported:

- `ctest -N`: `53`
- Makefile tests: `53`
- CMake tests: `53`
- `PASS: test counts match`

`make quality-review-cmake` then completed full `ctest` successfully:

- `53 / 53` passed
- `Total Test time (real) = 703.03 sec`

Interpretation:

- the reviewed CMake path remains the strongest honest cross-platform reviewed
  baseline
- the Sprint 36 CI/workflow wording changes did not drift the underlying active
  suite

### 4. The reviewed wrapper contract remains live, not just documented

The wrapper commands both completed successfully:

- `quality-review-compile: passed (format-check + lint)`
- `quality-review: passed (format-check + lint + test + deadcode-check)`
- `quality-review-cmake-compile: passed (configure + clean rebuild + ctest -N + test-count parity)`
- `quality-review-cmake: passed (configure + clean rebuild + ctest -N + ctest)`

Interpretation:

- Sprint 34’s reviewed wrapper contract is still intact after Sprint 36
- Sprint 36 improved platform truthfulness without hollowing out the maintained
  operator entry points

### 5. Key inherited invariants remain intact

The Day 13 end state still preserves the important inherited invariants:

- active CTest registry remains `53`
- Makefile/CMake test-count parity remains `53` vs `53`
- `deadcode-check` still passes inside the reviewed local wrapper path
- the Sprint 32 opt-in/self-check coverage remains part of the suite via
  `test_framework_optin`

Interpretation:

- Sprint 36 did not regress the validated baseline it inherited from Sprints
  32, 34, and 35

## Day 13 Interpretation

- Day 13 succeeded because Sprint 36 stayed narrow: it aligned platform
  contract expression and portability assumptions without reopening feature or
  warning-debt work.
- The one operational lesson from the day is specific and bounded:
  sanitizer-driven `build/` artifacts can contaminate a later direct `make lint`
  run unless the tree is cleaned first.
- Outside that cleanup note, the measured end state is clean and ready for Day
  14 closeout.

## Conclusion

Sprint 36’s full validation sweep passed across the direct commands, the
reviewed local wrappers, and the reviewed CMake parity wrappers. The branch is
in a validated state for closeout, with one operational note about cleaning the
`build/` tree after sanitizer runs.
