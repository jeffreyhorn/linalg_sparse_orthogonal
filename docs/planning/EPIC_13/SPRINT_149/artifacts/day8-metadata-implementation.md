# Sprint 149 Day 8: Package Metadata Implementation

## Purpose

Implement stronger Windows package metadata assertions in the reviewed CMake
install/downstream lane, completing the Day 7 metadata design without widening
Windows support claims.

## Files Changed

| File | Change |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Added explicit installed version-header and stricter `sparse.pc` metadata checks. |
| `docs/planning/EPIC_13/SPRINT_149/WORKING_NOTES.md` | Recorded Day 8 implementation and handoff. |
| `docs/planning/EPIC_13/SPRINT_149/artifacts/day8-metadata-implementation.md` | Published this implementation artifact. |

No `.c` or `.h` files were changed.

## Added Metadata Checks

| Check | Implementation |
| --- | --- |
| Installed version header | Requires `include/sparse/sparse_version.h` under the install prefix. |
| `sparse.pc` package name | Requires `Name: sparse`. |
| `sparse.pc` package version | Requires `Version: $version`, where `$version` is read from repository `VERSION`. |
| `sparse.pc` compile metadata | Requires `Cflags: -I${includedir}`. |
| `sparse.pc` link metadata | Requires `Libs: -L${libdir} -lsparse_lu_ortho -lm` with optional trailing whitespace. |

## Preserved Metadata Checks

Day 8 preserved the Day 6 checks for:

- installed static `.lib` presence;
- absence of installed DLLs;
- fixed 19-header count;
- required CMake package files, including `SparseTargets-release.cmake`;
- positive `STATIC IMPORTED` CMake imported target;
- install-prefix include metadata;
- installed static `.lib` imported-location metadata;
- source/build path leak rejection;
- absence of shared/module imported targets and shared imported locations;
- `sparse.pc` static archive description;
- `sparse.pc` unsupported wording and `Libs.private` rejection.

## Before And After Coverage

| Coverage Area | Before Day 8 | After Day 8 |
| --- | --- | --- |
| Version header | Counted only through aggregate header count. | Explicit `sparse_version.h` presence check. |
| `sparse.pc` package identity | File presence and static description only. | Explicit `Name: sparse` check. |
| `sparse.pc` version metadata | Version behavior proved through CMake package consumers, not text. | Exact `Version: $version` text check added. |
| `sparse.pc` compile metadata | Not directly checked on Windows. | `Cflags: -I${includedir}` text check added. |
| `sparse.pc` link metadata | Unsupported wording was rejected, but static link line was not required. | Static archive `Libs` line check added. |

## Support Boundary

The Windows lane still treats `sparse.pc` as installed package metadata only.
Day 8 does not add or imply:

- Windows `pkg-config --exists`;
- Windows `pkg-config --cflags`;
- Windows `pkg-config --libs`;
- Windows `pkg-config --static`;
- Windows `pkg-config --modversion`;
- Windows `pkg-config` downstream compile/link/run.

Those remain Unix-side proof surfaces unless a separate future evidence lane is
created.

## Failure Ordering

The workflow checks package metadata before downstream consumers:

1. static library and shared-artifact shape;
2. header and version-header shape;
3. CMake package file and target metadata;
4. `sparse.pc` metadata;
5. maintained installed CMake example;
6. exact-version generated CMake consumer;
7. mismatch-version generated CMake consumer.

This ordering keeps package-export defects separate from downstream
configure/build/run defects in hosted Windows logs.

## Hosted Evidence Requirement

The new checks are reviewed only after the hosted Windows job passes:

`Windows reviewed CMake install/downstream validation path`

If hosted Windows reports line-ending, whitespace, or generated-target text
differences in `sparse.pc` or CMake package files, treat that as a Day 8
follow-up fix before closeout.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Windows install proof checks all required installed package artifacts. | Complete | Workflow checks static `.lib`, headers, version header, CMake package files, `SparseTargets-release.cmake`, and `sparse.pc`. |
| Shared-library artifacts and unsupported wording fail explicitly. | Complete | DLL/shared imported metadata and `sparse.pc` unsupported wording checks remain active. |
| Local static review shows path handling is robust. | Complete | Checks use `Join-Path`, text reads with `-Raw`, CRLF-tolerant regexes, and separate source/build path leak variants. |
