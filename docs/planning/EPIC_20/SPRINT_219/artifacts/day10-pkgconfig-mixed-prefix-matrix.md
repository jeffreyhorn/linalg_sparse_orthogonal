# Day 10: pkg-config And Mixed-Prefix Consumer Matrix

**Date:** 2026-10-24

**Scope:** Sprint 219 item 219.4, final 10 of 24 hours
**Host:** macOS, native static-library installs

## Four-Route Identity And Consumer Results

`tests/test_pkgconfig_idx_width_install.sh` builds four independent packages
in temporary prefixes. Each package passes the static install validator,
which checks the generated header, pkg-config metadata, and queried archive
width together. The script confines discovery to that prefix with
`PKG_CONFIG_LIBDIR`, verifies the exact package prefix, version, include
path, width definition, and link flags, and compiles the maintained example
with both the advertised pkg-config link flags and its own archive by
absolute path. The example checks the archive width before the first matrix
operation and completes an LU solve in both modes.

| Install route | Width | Header/metadata/archive identity | pkg-config consumer | CMake imported-target consumer |
| --- | --- | --- | --- | --- |
| Make | 32 | Pass | Width 32, version `2.2.0`, solve `OK` | Not exported by Make install |
| Make | 64 | Pass | Width 64, version `2.2.0`, solve `OK` | Not exported by Make install |
| CMake | 32 | Pass | Width 32, version `2.2.0`, solve `OK` | Width 32, solve `OK` (Day 9) |
| CMake | 64 | Pass | Width 64, version `2.2.0`, solve `OK` | Width 64, solve `OK` (Day 9) |

The script obtains the expected version from `VERSION`; the table records
the installed value during this run. Day 9's separate CMake matrix used
`find_package(Sparse <version> EXACT REQUIRED CONFIG)` and also checked
relocation, stale cache, and explicit-width rejection.

## Mixed-Prefix Rejections

For each of the four base packages, the foreign package comes from the
*other* install route and the *opposite* width. All five negatives passed:

| Mutation | Rejection and identity proof |
| --- | --- |
| `PKG_CONFIG_PATH` points at foreign package while `PKG_CONFIG_LIBDIR` points at own package | Prefix check detects that pkg-config selected the foreign prefix before compilation. |
| Foreign `-I` precedes the own package's pkg-config flags | Public header rejects its conflict with `SPARSE_PACKAGE_IDX_BITS` at compile time. |
| Foreign archive linked by absolute path | The executable reports `consumer=N header=N archive=M` and exits before version, solution, or `OK` output. |
| Foreign library directory precedes own directory in `-L` lookup | The same pre-use diagnostic proves the linker selected the wrong-width archive; no solver call follows. |
| Nonexistent absolute archive path | Link fails, no executable is produced, and the diagnostic names the missing path. |

Each route has an additional positive link using the validated absolute
archive, so a same-named host installation cannot satisfy the overall
fixture even if it could satisfy the normal pkg-config `-l` link. The
negative archive and missing-archive links also use absolute paths; the
search-order case deliberately uses `-L` and proves which archive was
selected by its runtime width. Checking
the missing path rather than linker-specific wording keeps that assertion
portable across ld64, GNU ld, and lld styles. The existing
`tests/test_install.sh` additionally covers their known missing-library
diagnostics in its own fixture.

## Verification And Boundary

`bash tests/test_pkgconfig_idx_width_install.sh` passed **40/40**: four
installations, four static identity checks, four metadata checks, four
pkg-config-flag consumers, four absolute-archive consumers, and five
negative cases for each route. `bash -n` passed on the final script. This
is static, same-version, native package evidence; it does not claim a
shared-library ABI or automatic protection for consumers
that omit the maintained pre-use check. The 24-hour positive/negative
consumer allocation in item 219.4 is closed. Day 11 owns selected 64-bit
functional behavior beyond these installed-consumer tests.
