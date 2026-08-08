# Sprint 143 Day 10 CI And Package Report Alignment

## Purpose

Align Linux, macOS, Windows, and package-report wording with the selected
static-first package contract and the Day 9 downstream proof shape.

## Changes Implemented

| Surface | Change | Reason |
| --- | --- | --- |
| `.github/workflows/ci.yml` | Clarified that the Linux reviewed package lane covers static archive install/export/downstream proof, static `pkg-config`/CMake metadata, and unsupported shared-artifact checks. | Keeps the reviewed Linux package contract aligned with the Day 7-9 implementation. |
| `.github/workflows/macos-ci.yml` | Clarified that supplemental macOS package lanes include downstream consumer and unsupported shared-artifact proof but do not become reviewed macOS parity. | Preserves Sprint 144 platform-promotion boundaries. |
| `.github/workflows/windows-ci.yml` | Clarified the supplemental Windows lane as CMake-first static package confidence only, not Makefile, `pkg-config`, shared-library, or dynamic ABI support. | Avoids overclaiming Windows support from a supplemental static CMake lane. |
| `.github/workflows/windows-ci.yml` | Added installed `sparse.pc` presence, static `.pc` description, unsupported package/ABI wording, and no shared imported CMake metadata checks. | Mirrors the selected static-first metadata proof shape on the Windows supplemental lane where feasible. |
| `.github/workflows/windows-ci.yml` | Strengthened installed example output checks to require version text, solution output, and `OK`. | Aligns Windows supplemental downstream proof with Day 9 local CMake proof. |
| `.github/workflows/windows-ci.yml` | Builds and runs the exact-version consumer after exact-version configure succeeds. | Aligns Windows supplemental exact-version proof with Day 9 local CMake proof. |

## Package Report Decision

No package report row semantics changed on Day 10. The package report rows
remain source-controlled proof-owner metadata, not evidence that a fresh
install command was run by the report script.

The existing package rows still point at the correct proof owners:

- `tests/test_install.sh`;
- `tests/test_cmake_install.sh`;
- `sparse.pc.in`;
- `cmake/SparseConfig.cmake.in`;
- `scripts/static_package_deferral_check.sh`.

## Platform Boundaries Preserved

- Linux remains the reviewed static-first package-contract source of truth.
- macOS install/package jobs remain supplemental confidence paths.
- Windows install/downstream remains supplemental CMake-first static package
  confidence.
- Sprint 143 does not promote macOS or Windows reviewed install/export parity.
- No CI wording claims shared-library support, dynamic ABI compatibility,
  runtime-loader behavior, package-manager support, or broad platform parity.

## Focused Validation

Focused checks run for this batch:

```sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
rg -n "shared-library support|dynamic ABI support|package-manager|runtime-loader|reviewed install-validation|pkg-config parity" .github/workflows
ruby -e 'require "yaml"; ARGV.each { |f| YAML.load_file(f); puts "#{f} ok" }' .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_12/SPRINT_143 .github/workflows
```

Results:

| Check | Result |
| --- | --- |
| Package report index check | Passed: 6 rows |
| Package report freshness check | Passed: 6 source-controlled advisory rows |
| Workflow non-claim scan | Passed; matches are explicit non-claims and support-tier boundaries |
| Workflow YAML parse | Passed for Linux, macOS, and Windows workflows |
| `git diff --check` | Passed |
| Trailing-whitespace scan | Passed |

Local `pwsh` was unavailable, so the Windows supplemental PowerShell block was
reviewed structurally and will be executed by the Windows CI lane.

## Day 11 Input

Day 11 should align public and maintainer docs with the final static-first
package implementation:

1. README and INSTALL should describe the static `.pc` metadata and stronger
   downstream proof without implying package-manager or shared-library support.
2. Maintainer guide should reflect the current pass counts and proof ownership.
3. Runtime/backend sentinel non-claims and Sprint 144 platform boundaries must
   remain visible.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| CI wording matches the selected package contract. | Complete | Workflow comments now describe static archive install/export/downstream proof and non-claims. |
| Linux/macOS/Windows comments do not imply unsupported platform parity. | Complete | macOS and Windows package lanes remain explicitly supplemental. |
| Package report rows preserve honest proof-owner semantics. | Complete | No row-semantic changes were made; report checks remain required validation. |
