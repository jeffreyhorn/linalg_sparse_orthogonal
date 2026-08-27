# Sprint 182 Windows Report Freshness Deferral Decision

**Sprint:** 182 - Windows Report Freshness Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Status:** Accepted for Sprint 182 implementation

## Decision

Windows report freshness remains formally deferred.

Windows CI continues to prove the reviewed CMake/MSVC build and test path plus
the static-first CMake install/downstream path. It does not prove selected
oracle, comparison, benchmark, or broad generated report freshness on Windows.

## Supported Windows Claims

The supported Windows surface remains:

- reviewed CMake configure and build on the pinned hosted Windows runner;
- reviewed `ctest -N` registration count and full `ctest` execution;
- reviewed static-first CMake install/downstream validation;
- installed static `.lib`, headers, CMake package metadata, exact-version
  behavior, mismatch-version rejection, and metadata-only `sparse.pc`
  inspection;
- rejection of unsupported shared-library, loader, and static/shared selector
  metadata in the reviewed Windows package lane.

These are CMake/package validation claims, not generated report freshness
claims.

## Unsupported Claims

Sprint 182 keeps these Windows claims unsupported:

- selected oracle freshness;
- selected comparison freshness;
- selected benchmark freshness;
- broad generated report freshness;
- Makefile parity;
- Bash/POSIX report generation support;
- `pkg-config` command execution parity;
- package-manager support;
- shared-library, dynamic ABI, DLL/import-library, or runtime-loader support;
- broad Windows platform parity;
- portable performance, performance superiority, or state-of-the-art status
  from benchmark artifacts.

## Accepted Evidence

Days 1-5 found that Python TSV data formats are not the primary blocker.
Generated comparison and oracle files use repo-relative paths, stable tab
delimiters, and LF line endings. The blockers are command/runtime and claim
ownership issues.

## Blockers

The formal deferral is based on these exact blockers:

- no reviewed Windows Makefile parity for current selected freshness wrappers;
- no Windows-safe CMake/MSVC project probe path for selected comparison or
  oracle generators;
- no reviewed Windows `.lib`/MSVC link model for generated comparison/oracle
  probes;
- no Windows `.exe`-aware temporary probe execution path;
- no Windows-native canonical benchmark report generator;
- no selected Windows workflow artifact name or exact Windows upload scope;
- selected target manifest rows do not list `windows` in `workflow_platforms`;
- existing docs and workflow comments correctly preserve Windows report
  freshness as a non-claim.

## Rejected Alternatives

| Alternative | Reason |
| --- | --- |
| Promote selected comparison now | The artifact shape is favorable, but project probes still assume `cc`, Unix `.a`, `-lm`, extensionless executables, and fallback `make`. |
| Promote selected oracle now | It carries the same probe/link blockers and adds a broader 52-row selected surface plus selected glob upload handling. |
| Promote selected benchmark now | The generator is Bash-based, depends on Unix metadata commands, runs benchmark binaries, and carries performance-adjacent claim risk. |

## Guard Contract

The active guard contract is:

- `.github/workflows/windows-ci.yml` must not run selected oracle,
  comparison, or benchmark freshness commands;
- `.github/workflows/windows-ci.yml` must not upload selected freshness
  artifact names used by Linux or macOS;
- selected target manifest rows do not list `windows` as a selected freshness
  platform;
- any future Windows promotion must be manifest-backed, exact, and guarded by
  an allowlist rather than broad workflow scans.

## Revisit Criteria

A future sprint may reconsider Windows selected report freshness only after it
has all of these gates:

- direct Windows-safe command that does not rely on Makefile or Bash behavior;
- CMake/MSVC project probe build/link support;
- reviewed `.lib` linkage without Unix `-lm`;
- `.exe`-aware temporary executable handling;
- exact Python executable proof on the hosted Windows runner;
- exact selected artifact upload scope with `if-no-files-found: error`;
- selected target manifest metadata for Windows workflow file, job, artifact,
  platform, support tier, claim scope, and non-claims;
- workflow guard allowlist for exactly one selected Windows path;
- README, INSTALL, maintainer guide, workflow comments, and report-index docs
  aligned to the proven Windows scope.

## Pass/Fail Contract

This deferral passes when Windows report freshness stays unpromoted,
guard-owned, and documented while current Windows CMake/package claims remain
intact.

It fails if workflow, manifest, or docs changes imply selected Windows report
freshness without a reviewed Windows lane and manifest-backed evidence.
