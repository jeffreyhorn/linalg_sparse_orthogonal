# Sprint 29 Day 8 — Item 5 close (Windows CI) + Item 10b (macOS-15+ tsan)

## Item 5 — Windows CI

### Decision: MSVC + CMake on `windows-latest`, build + ctest only

The Day-7 draft `.github/workflows/windows-ci.yml` runs the CMake
project on `windows-latest` with Visual Studio 17 2022 (x64),
builds Release config, and runs ctest with `--output-on-failure`.
No sanitizer / coverage / clang-format / clang-tidy steps on the
Windows job — those stay Linux-only (the existing `.github/workflows/
ci.yml` lint + sanitize + coverage + tsan jobs cover them).  Rationale:
the matrix is "build + test compiles + runs" only, mirroring how the
existing Linux job's `build-and-test` is structured.

### Day-8 close — portability fixes landed

**1. `clock_gettime(CLOCK_MONOTONIC)` → C11 `timespec_get` fallback**

The Sprint 29 Day 6/7 progress-timing helpers (`s29_now_s` in
`sparse_lu.c` / `sparse_cholesky.c` / `sparse_ldlt.c`; `s29_qr_now_s`
in `sparse_qr.c`; `s29_iter_now_s` in `sparse_iterative.c`;
`s29_eigs_now_s` in `sparse_eigs.c`) all used POSIX
`clock_gettime(CLOCK_MONOTONIC, ...)`.  MSVC's `<time.h>` doesn't
expose `clock_gettime`, so the Day-7 first CI run would have failed
on `cl.exe` with an undeclared-identifier error.

Day-8 fix: mirror the existing Sprint-24 pattern from
`src/sparse_reorder_amd_qg.c::qg_prof_now_ns`:

```c
static double s29_now_s(void) {
    struct timespec ts;
#ifdef _WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
```

C11's `timespec_get(&ts, TIME_UTC)` is available on MSVC and gives
wall-clock seconds with `ts.tv_nsec` precision.  It's not strictly
monotonic (it tracks the wall clock, not a monotonic counter), but
the only consumer is `progress.elapsed_s` for progress callbacks —
where even a non-monotonic source is acceptable since callbacks
should not be sensitive to small backward jumps.

Applied to all 6 Day 6/7 files.  Each file also got the
`_POSIX_C_SOURCE 199309L` feature-test macro at the top, mirroring
the Sprint-24 pattern.

**2. CMakeLists.txt warning flags gated on `NOT MSVC`**

Day 7 already did this: `add_compile_options(-Wall -Wextra ...)` only
runs on gcc/clang.  MSVC path uses `/W3` + `_CRT_SECURE_NO_WARNINGS`.

**3. Sprint 28 PR #36 inheritance: tf_setenv / tf_unsetenv compat**

The `tests/test_framework.h` `tf_setenv` / `tf_unsetenv` macros
already route through `_putenv_s` on `_WIN32` (lines 193-201).
Sprint 28 PR #36 lit this up + added `_POSIX_C_SOURCE 200809L`
guards to the 3 supernodal-postorder test files.  Sprint 29 added
no new env-mutating tests, so no further test-file guards needed.

**4. Other POSIX headers**

Swept `src/` for `<unistd.h>` / `<sys/time.h>` / `<sys/resource.h>` /
`<sys/wait.h>` / `<sys/mman.h>` / `fork(` / `mmap(` etc.: zero hits.
The only POSIX-header includes in the repo are
`tests/test_fuzz.c` and `benchmarks/bench_amd_qg.c`, neither of which
runs in the Windows CI matrix (`tests/test_fuzz.c` is already gated
`if(NOT WIN32 AND NOT MSVC)` in `CMakeLists.txt` line 151;
`benchmarks/bench_amd_qg.c` is built only via `bench-build` which the
Windows workflow doesn't invoke).

### Anticipated remaining Day-8 risks (red-on-first-push)

The Day-7 draft workflow's first push may still fail on:

- **`__attribute__` usage**: MSVC doesn't recognise gcc's
  `__attribute__((cleanup))` / `__attribute__((unused))` / etc.
  A grep of `src/` found no `__attribute__` usage; should be safe.
- **`_Thread_local`**: C11 keyword; supported by MSVC 2022 but may
  emit warnings.
- **clang-format style**: the format-check step is on the Linux
  `lint` job, NOT the Windows job — so style mismatches across
  platforms don't affect Windows CI.
- **Test data file paths**: tests load `tests/data/*.mtx` with
  forward-slash paths.  Windows `fopen` accepts forward slashes,
  so this should work; verify on first push.
- **`int` vs `idx_t` (int32_t)**: MSVC may warn on implicit
  conversions; `_CRT_SECURE_NO_WARNINGS` doesn't silence all of
  these.  The `/W3` warning level was chosen to be permissive on
  this.

If any of these surface, the fix lands in this branch + push iterates.

## Item 10b — macOS-15+ tsan handling

### Decision: Option (i) — Linux-CI tsan job (already in place)

Per Sprint 28 retrospective: macOS 15.7's tsan hangs in
`__tsan::CheckAndProtect → get_dyld_hdr` during dyld initialisation.
Local `make tsan` on macOS 15+ is blocked.  Sprint 29 Item 10b had
two options:

- **(i) Linux-CI tsan job**: run `make tsan` on `ubuntu-latest` —
  tsan works there without the dyld bug.
- **(ii) macOS-version-gated `make tsan`**: detect macOS 15+ at
  make-time + emit a non-zero-exit "blocked" message.

**Pick (i).**  The existing `.github/workflows/ci.yml::tsan` job
(lines 59-124) already runs `make tsan` on `ubuntu-latest`, plus an
OpenMP-clang-libomp variant for the Sprint 21 Day 5/6 parallel
Lanczos MGS kernel.  Sprint 29 inherits this — no new workflow file
needed for Item 10b.

Rejection rationale for option (ii): macOS-version-gated `make tsan`
would emit a "tsan blocked on macOS 15+" error locally, but the
intent (catch thread-safety regressions in CI) is already covered by
the Linux job.  Adding a local-only error message duplicates the
intent at a less-discoverable surface.

The macOS local `make tsan` remains broken on macOS 15+ (Sprint 28
inheritance).  Developers running local TSan on macOS 14- still
work.  The recommendation in `Makefile` / docs is to use the Linux
CI tsan job as the source of truth on macOS-15+ workstations; no
further action needed.

## Quality gates (Day 8 local)

- `make clean && make`: clean compile, no warnings
- `make format`: clean
- `make lint`: 77/77 files, exit 0
- `make test`: 2068 + 20 integration tests PASS
- `make wall-check`: Pres_Poisson ND within ceiling

The actual Windows CI green check happens after Day 8 pushes to the
sprint-29 branch.  This decision doc captures the local portability-
fix surface; the GitHub Actions iteration loop closes any first-push
issues that surface beyond what was anticipated above.

## References

- `docs/planning/EPIC_2/SPRINT_29/PLAN.md` Day 8 section.
- `docs/planning/EPIC_2/SPRINT_28/RETROSPECTIVE.md` "What didn't go
  well" — macOS 15+ tsan dyld hang.
- `.github/workflows/windows-ci.yml` — Day-7 draft + Day-8 fixes.
- `.github/workflows/ci.yml` lines 59-124 — existing tsan-on-Linux
  job that satisfies Item 10b option (i).
- `src/sparse_reorder_amd_qg.c:1-11, 226-236` — Sprint 24 Day 1
  reference pattern for the `clock_gettime` / `_WIN32` /
  `timespec_get` portability shim.
