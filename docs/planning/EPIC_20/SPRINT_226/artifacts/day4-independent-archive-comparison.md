# Sprint 226 Day 4: Independent Exploratory Archive Comparison

## Decision

**The exploratory archive method reproduced byte for byte on this macOS host.** A second clean, independent local clone of `c42d6715fad7e895bae4941a006df3563f1afc38` produced exactly the same compressed tar bytes and member metadata as the [Day 3 exploratory archive](day3-exploratory-source-archive.md). The extracted files also matched every Git blob and executable bit in that commit.

This is **not** a reproducible release-candidate result. [Day 2](day2-static-candidate-definition.md) left candidate selection blocked by the missing changelog/version decision, uncommitted Sprint 226 inputs, and unassigned release/rollback approvers. A finally selected candidate must be independently built and compared again at its own exact SHA. No cross-OS reproducibility is inferred from two builds on one macOS host.

## Independent checkout and commands

The first archive was produced from the original repository on Day 3. The second used a separate Git object store and clean checkout created under ignored `build/`:

```sh
mkdir -p build/sprint226/day4
git clone --no-local --no-hardlinks . build/sprint226/day4/checkout \
  > build/sprint226/day4/clone.log 2>&1
git -C build/sprint226/day4/checkout status --porcelain=v1
git -C build/sprint226/day4/checkout rev-parse HEAD
git -C build/sprint226/day4/checkout rev-parse 'HEAD^{tree}'

set -o pipefail
LC_ALL=C TZ=UTC git -C build/sprint226/day4/checkout archive \
  --format=tar --prefix=sparse-lu-ortho-2.2.0/ \
  c42d6715fad7e895bae4941a006df3563f1afc38 |
  gzip -n > build/sprint226/day4/sparse-lu-ortho-2.2.0-src-EXPLORATORY-2.tar.gz
```

The clone command and archive pipeline exited 0. `git status --porcelain=v1` printed nothing. The independent clone resolved to the expected commit and tree `934a16b09ada33e12f726ec944fc675bd7d244f4`. The same macOS 15.7.9 x86_64 environment used Git 2.51.2 and Apple gzip 457.140.3; `LC_ALL=C` and `TZ=UTC` were set for archive creation, with no `SOURCE_DATE_EPOCH` override. The clone log is retained locally; the archive command emitted no stderr diagnostic. The first worktree's untracked Sprint 226 files and ignored build output did not enter either archive.

## Byte, manifest, metadata, and source comparison

| Check | First archive | Independent archive | Result |
| --- | --- | --- | --- |
| SHA-256 | `cdfceff5ef4fdf405f8bf3b15e6a7fe608d5f4566345ea3fd2f8d6ec0e27be8f` | `cdfceff5ef4fdf405f8bf3b15e6a7fe608d5f4566345ea3fd2f8d6ec0e27be8f` | Exact match; byte comparison true. |
| Size | 17,755,923 bytes | 17,755,923 bytes | Match. |
| Ordered member listing SHA-256 | `f09d10fe131ef1ab0fdedcd6ebd4feff055b5e6380207dcad7b7971b5462c468` | `f09d10fe131ef1ab0fdedcd6ebd4feff055b5e6380207dcad7b7971b5462c468` | Exact match. |
| Member metadata | 5,053 members: 4,557 files, 496 directories | Same names, types, modes, owner IDs/names, mtimes, and sizes | Exact match. |
| Forbidden generated content | None in Day 3 listing | None in independent listing | Pass for `.git/`, `build/`, generated API HTML/header, object/library output, and caches. |
| Changelog | Missing | Missing | **Release blocker**, not an archive mismatch. |

The ordered second listing is retained as `build/sprint226/day4/manifest.txt`; a machine-readable local summary is `build/sprint226/day4/comparison-receipt.json`. Those paths are ignored build evidence, not committed release assets. The archive comparison used SHA-256 and a full byte equality check, so there was no mismatch to diagnose.

For the extracted source verification, the Day 3 extracted tree was compared with the independent clone's `git ls-tree -r -z HEAD`: each regular file's contents were rehashed as a Git blob, and each executable bit was checked against the tree mode. Result: **4,557 tracked files, 4,557 extracted files, zero missing, zero extra, zero wrong blob hashes, and zero wrong executable bits.** Because the compressed archives are byte-identical, the same tar payload is represented by both comparisons. This validates exact-base source contents; it does not prove the future installed files or a release approval.

## Gate ledger and handoff

| Gate | Day 4 state | Next action |
| --- | --- | --- |
| Exploratory same-host byte reproducibility | **Pass** for commit `c42d6715fad7e895bae4941a006df3563f1afc38` and the recorded tools only. | Rebuild both archives at the selected candidate SHA after freeze. |
| Full tracked-source content/mode match | **Pass** for all 4,557 base files. | Review the full candidate manifest and source/license scope with owners. |
| Changelog, candidate freeze, approvers | **Blocked**, unchanged. | Resolve version/changelog and owner assignments; commit intended inputs, then review and select a clean candidate. |
| Candidate archive, installed-file map, hosted proof | **Not established**. | Day 5 may map the exploratory archive's install outputs but cannot promote them as candidate proof. |

The original and independent archive bytes, manifest files, clone log, and comparison receipt remain under ignored `build/sprint226/day3/` and `build/sprint226/day4/`. Their local retention is useful for the sprint audit but is not an immutable published provenance bundle. Any source or packaged documentation change requires a new candidate ID and two new archive builds.
