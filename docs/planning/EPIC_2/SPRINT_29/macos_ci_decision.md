# Sprint 29 Day 9 — Item 6: macOS CI Job (Apple Clang + Homebrew GCC)

## Decision

**Two-job matrix on `macos-latest`: Apple Clang (`cc`) + Homebrew GCC
(`gcc-15`).**  Plus a separate `install-and-pkgconfig` job that
verifies the `make install` + `pkg-config` flow on a temp prefix.

The matrix exercises:
- The default Apple Clang build path that mirrors developer
  experience on macOS workstations.
- The Homebrew GCC alternative that catches any gcc-specific warning
  or feature-detect issue (since Linux CI also uses gcc, this is the
  bridge between the two main developer toolchains).

Apple Clang job additionally runs `make sanitize` (ASan + UBSan) so
the macOS sanitize signal stays green alongside the Linux sanitize
job in `.github/workflows/ci.yml`.

## OpenMP linkage decision: serial path only for now

The Makefile already handles both OpenMP linkage variants:
- Apple Clang: `-Xpreprocessor -fopenmp -I$(LIBOMP_FLAG_PREFIX)/include`
  + `-L$(LIBOMP_FLAG_PREFIX)/lib -lomp` (libomp via Homebrew).
- Homebrew GCC: `-fopenmp` directly (links libgomp).

`make omp` builds with `-DSPARSE_OPENMP`.  This workflow runs the
**serial** path (no `-DSPARSE_OPENMP`) on both matrix entries to
keep the matrix simple.  OpenMP-on-macOS coverage lives in:
- Local developer `make omp` runs (well-exercised in Sprint 21).
- Linux CI's OpenMP tsan job (`.github/workflows/ci.yml:97-124`).

If a future sprint needs explicit OpenMP-on-macOS-CI coverage, the
matrix expands with a third entry (`compiler: apple-clang-omp` /
`extra_flag: omp`).

## TSan deferred to Linux CI per Item 10b

The macOS workflow does NOT run TSan.  Sprint 28 retrospective +
Sprint 29 Day 8 Item 10b document the `__tsan::CheckAndProtect →
get_dyld_hdr` hang on macOS 15+; thread-safety regressions are
caught by `.github/workflows/ci.yml::tsan` on `ubuntu-latest`.

## Homebrew GCC version pin: `gcc-15`

Homebrew's `gcc` formula ships the latest GCC major (currently 15
as of 2026-04).  The unversioned symlink `gcc-15` matches the
formula's binary.  If Homebrew bumps to gcc-16 mid-sprint, the
workflow needs a one-line update + the floor must stay at gcc-13 to
exercise the C11 `_Thread_local` + `_Static_assert` paths.

This pinning is intentional: floating to "latest" via
`brew install gcc && CC=$(ls /opt/homebrew/bin/gcc-* | head -1)`
would break reproducibility across CI runs spanning major-version
bumps.  Pin + bump-on-version-update keeps the signal stable.

## Coverage + install verification

- `make coverage` already works on macOS (`lcov`, `genhtml`, `bc`
  all available via Homebrew; `llvm-cov` available via `xcrun`).
  The Linux CI `coverage` job is canonical; Day 9 doesn't add a
  duplicate macOS coverage job (would double CI runtime for no
  new signal — same gcc-style instrumentation either way).
- `make install` + `pkg-config` verified locally on macOS:
  ```
  $ make install PREFIX=/tmp/sparse-test-prefix
  Installed:
    library  → /tmp/sparse-test-prefix/lib/libsparse_lu_ortho.a
    headers  → /tmp/sparse-test-prefix/include/sparse/
    pkg-config → /tmp/sparse-test-prefix/lib/pkgconfig/sparse.pc
  $ PKG_CONFIG_PATH=/tmp/sparse-test-prefix/lib/pkgconfig \
      pkg-config --modversion sparse
  2.2.0
  $ PKG_CONFIG_PATH=/tmp/sparse-test-prefix/lib/pkgconfig \
      pkg-config --cflags --libs sparse
  -I/tmp/sparse-test-prefix/include -L/tmp/sparse-test-prefix/lib \
      -lsparse_lu_ortho -lm
  $ make uninstall PREFIX=/tmp/sparse-test-prefix
  Done.
  ```
  The new `install-and-pkgconfig` job in macos-ci.yml runs the
  same verification on `macos-latest` so regressions get caught
  in CI.

## Platform-gate sweep — zero hits

Searched `src/` + `include/` + `tests/` for:
- `__APPLE__`
- `__GLIBC__`
- `__linux__`
- `TARGET_OS_*`

Zero hits.  The code uses portable C11 + POSIX (`_POSIX_C_SOURCE
199309L` feature-test for `clock_gettime`) with no per-platform
branches.  Homebrew GCC and Apple Clang should both compile this
cleanly.

## Anticipated remaining Day-9 risks (red-on-first-push)

- **`-Wconversion` strictness differences**: Homebrew GCC may emit
  `-Wconversion` warnings that Apple Clang silently accepts (or
  vice versa).  Both compilers are configured `-Werror`-equivalent
  via `-Wpedantic`; if any new warnings surface, fix at the call
  site (typically `(idx_t)` casts).
- **Homebrew GCC bottle drift**: if `brew install gcc` pulls a
  pre-built bottle that doesn't include `<omp.h>` correctly, the
  serial-only Day-9 workflow is unaffected, but local `make omp`
  on the same machine would need investigation.

## Quality gates (Day 9 local)

- `make` (Apple Clang default): clean compile, no warnings
- `make test`: 2068 assertions PASS, 20 integration tests PASS
- `make wall-check`: Pres_Poisson ND within ceiling
- `make install` + `pkg-config` + `make uninstall`: round-trip
  verified

The actual GitHub Actions macOS run happens after Day 9 pushes to
the sprint-29 branch.

## References

- `docs/planning/EPIC_2/SPRINT_29/PLAN.md` Day 9 section.
- `.github/workflows/macos-ci.yml` — Day-9 workflow.
- `Makefile:16-35` — existing OpenMP linkage matrix logic.
- `Makefile:521-537` — existing install target.
- `Makefile:540-550` — existing uninstall target.
