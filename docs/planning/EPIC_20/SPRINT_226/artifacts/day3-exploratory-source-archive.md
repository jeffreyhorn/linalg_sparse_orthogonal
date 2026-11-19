# Sprint 226 Day 3: Exploratory Source Archive and Extraction

## Decision and scope

**The release-candidate archive gate remains blocked.** [Day 2](day2-static-candidate-definition.md) found no approved changelog/version decision, committed Sprint 226 inputs, or named release and rollback approvers. To exercise the proposed archive path without claiming a frozen candidate, this day built an **exploratory**, local-only archive from the exact committed base `c42d6715fad7e895bae4941a006df3563f1afc38` (`VERSION=2.2.0`). The archive is in ignored `build/` and is not a release asset. No tag or approval was created. A later selected candidate SHA requires a fresh archive and all dependent checks.

## Archive method and environment

The method follows the [Sprint 214 source-archive procedure](../../../EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md), substituting the base commit for an unapproved release tag. The output is deliberately named `EXPLORATORY`:

```sh
set -o pipefail
LC_ALL=C TZ=UTC git archive --format=tar \
  --prefix=sparse-lu-ortho-2.2.0/ \
  c42d6715fad7e895bae4941a006df3563f1afc38 |
  gzip -n > build/sprint226/day3/sparse-lu-ortho-2.2.0-src-EXPLORATORY.tar.gz
```

`git archive` selects only tracked files at the commit, in Git's tree order. Its tar member timestamps are the commit timestamp; `gzip -n` omits the gzip header name/timestamp. `LC_ALL=C` and `TZ=UTC` were set for the archive command. No `SOURCE_DATE_EPOCH` override was used; the commit timestamp was `1795050112`. Determinism is a *method hypothesis* until Day 4 independently rebuilds and compares bytes.

| Provenance | Observed value |
| --- | --- |
| Host | macOS 15.7.9, x86_64 |
| Git | 2.51.2 |
| gzip | Apple gzip 457.140.3 |
| tar inspection/extraction | bsdtar 3.5.3 / libarchive 3.7.4 |
| CMake | 4.4.4; compiler AppleClang 17.0.0.17000604 |
| Archive file | `build/sprint226/day3/sparse-lu-ortho-2.2.0-src-EXPLORATORY.tar.gz` (ignored local output) |
| Archive size | 17,755,923 bytes |
| Archive SHA-256 | `cdfceff5ef4fdf405f8bf3b15e6a7fe608d5f4566345ea3fd2f8d6ec0e27be8f` |
| Member listing | `build/sprint226/day3/manifest.txt`; SHA-256 `f09d10fe131ef1ab0fdedcd6ebd4feff055b5e6380207dcad7b7971b5462c468` |
| Command result | Archive pipeline exit 0; no stderr diagnostic. The tar stream was consumed by gzip. |

The ignored archive, full member listing, extracted tree, CMake tree, `configure.log`, and `build.log` remain local under `build/sprint226/day3/`. The hashes and commands here let a reviewer identify those bytes if retained; this document alone is not an independent archived artifact or public provenance attestation.

## Member and source inspection

| Check | Result |
| --- | --- |
| Members | 5,053 total: 4,557 regular tracked files, 496 directory entries, no links or special entries. |
| Git tree comparison | All 4,557 tracked file paths from `git ls-tree -r --name-only` appear once under `sparse-lu-ortho-2.2.0/`; no missing or extra file paths. |
| Metadata | All members have one mtime, `1795050112`, and uid/gid `0/0` with names `root/root`. |
| Path safety | No absolute/traversal path or unexpected top-level prefix found. |
| Forbidden local output | No `.git/`, `build/`, `cmake-build/`, `CMakeFiles/`, `CMakeCache.txt`, generated `docs/api/html/`, generated `include/sparse_version.h`, object/archive/shared-library output, or Python cache. The source template `include/sparse_version.h.in` is correctly present. |
| Minimum release inputs | `LICENSE`, `VERSION`, `README.md`, `INSTALL.md`, `Makefile`, `CMakeLists.txt`, `Doxyfile`, and the tracked source/build/test/docs paths are present. |
| Blocking missing input | `CHANGELOG.md` is absent in the base tree and archive. This is a release checklist failure, not an archive-tool failure. |

The archive has no approved full-content review. Matching path lists do not prove license completeness, release-note accuracy, or source-to-installed-file equivalence.

## Extracted-tree smoke test

The archive was extracted beneath `build/sprint226/day3/sparse-lu-ortho-2.2.0/`. The configure and clean static-library build used only that extracted source tree:

```sh
cmake -S build/sprint226/day3/sparse-lu-ortho-2.2.0 \
  -B build/sprint226/day3/cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  > build/sprint226/day3/configure.log 2>&1
cmake --build build/sprint226/day3/cmake --target sparse_lu_ortho \
  --parallel 2 --clean-first > build/sprint226/day3/build.log 2>&1
```

Both commands exited 0. CMake configured with AppleClang 17 and built `libsparse_lu_ortho.a`; its SHA-256 is `85b9f278ee5aa26c8f13605ba3e57cfb51b904e3f2753f803d119197fe241921`. The retained clean build log has 17 compiler warnings and zero `error:` diagnostics. These warnings do not make the build fail, but a release owner must decide whether any violate the candidate quality gate. This smoke test did **not** run the full test suite, install, downstream consumer, or hosted CI.

## Gate status and Day 4 handoff

| Gate | Status |
| --- | --- |
| Exploratory base archive construction and extraction | Passed locally for the stated base SHA. |
| Candidate freeze/changelog/approvers | **Blocked**, unchanged from Day 2. |
| Independent byte-for-byte reproducibility | **Not run**; Day 4 must use an independent clean checkout and compare hashes/manifests. |
| Exact-candidate archive and source-to-install proof | **Not started**; requires a selected, reviewed candidate SHA. |

Day 4 can independently test the archive procedure against the same **exploratory** base while the candidate remains blocked. Any later committed changelog or source edit changes the candidate SHA and invalidates this archive for release purposes. The first release-candidate archive must be rebuilt from the finally approved snapshot.
