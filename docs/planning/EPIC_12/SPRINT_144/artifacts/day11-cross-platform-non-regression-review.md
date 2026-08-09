# Sprint 144 Day 11 Cross-Platform Non-Regression Review

## Purpose

Confirm that the selected macOS reviewed static-first install/export promotion
did not weaken Linux ownership, accidentally promote Windows, alter package
mechanics, or create unsupported platform/package claims.

## Diff Review Summary

| Surface | Day 11 review |
| --- | --- |
| `.github/workflows/macos-ci.yml` | Selected macOS package jobs are promoted to reviewed static-first proof; commands are unchanged. |
| `.github/workflows/ci.yml` | No diff; Linux remains the enforced source-of-truth reviewed baseline and reviewed static-first package contract owner. |
| `.github/workflows/windows-ci.yml` | No diff; Windows remains reviewed CMake subset plus supplemental CMake install/downstream confidence. |
| `README.md` | CI summary now separates Linux, macOS reviewed static-first package proof, supplemental Homebrew GCC, and Windows supplemental install/downstream confidence. |
| `INSTALL.md` | Supported-platform and install-validation wording now reflects macOS reviewed static-first package proof. |
| `docs/maintainer_guide.md` | Maintainer support-tier guidance now reflects current macOS status and preserves non-claims. |
| `tests/corpus/manifests/report_families.tsv` | CI row now identifies Linux source-of-truth, macOS reviewed static-first install/export proof, and Windows reviewed CMake subset lanes; hosted logs remain external evidence. |

## Cross-Platform Support-Tier Review

| Platform lane | Day 11 status | Non-regression result |
| --- | --- | --- |
| Linux reviewed source of truth | Still owns reviewed Makefile compile-quality, reviewed CMake parity, dead-code, and reviewed static-first package contract. | Unchanged and preserved. |
| macOS Apple Clang reviewed path | Still owns compile-quality, CMake parity, wall-check, and sanitizer. | Unchanged and preserved. |
| macOS static-first package proof | Promoted to reviewed Make install/`pkg-config` and CMake install/export proof. | Intended selected-lane change. |
| macOS Homebrew GCC | Still supplemental second-compiler coverage. | Unchanged and preserved. |
| Windows MSVC CMake subset | Still reviewed with `EXPECTED_WINDOWS_CTEST_COUNT=56`. | Unchanged and preserved. |
| Windows CMake install/downstream | Still supplemental CMake-first confidence. | Unchanged and preserved. |
| Windows staged tests | `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain staged due pthread/POSIX blockers. | Unchanged and preserved. |

## Static-First Package Guard Confirmation

`bash scripts/static_package_deferral_check.sh` passed.

Confirmed boundaries:

- `BUILD_SHARED_LIBS=ON` rejection still passes;
- CMake target remains explicitly static;
- static install metadata remains present;
- unsupported shared export/ABI metadata remains absent;
- package metadata has no static/shared selector;
- support wording still keeps shared/ABI support deferred.

## Package And Report Non-Regression

| Check | Result |
| --- | --- |
| `python3 scripts/normalize_report_index.py --family package --check` | Passed: 6 rows |
| `python3 scripts/normalize_report_index.py --family ci --check` | Passed: 1 row |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Passed: 6 source-controlled advisory rows |
| `python3 scripts/normalize_report_index.py --family ci --check-freshness` | Passed: 1 source-controlled advisory row |

Interpretation:

- package rows remain proof-owner metadata, not fresh hosted-run evidence;
- CI row identifies hosted lane definitions whose logs remain external;
- report rows do not manufacture macOS hosted proof.

## Claim Boundary Review

Unsupported-claim scans found only explicit non-claims for:

- shared-library packaging/support;
- dynamic ABI compatibility;
- runtime-loader behavior/compatibility;
- package-manager support;
- static/shared selectors;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- broader macOS platform parity.

No documentation or workflow wording was found that promotes those unsupported
claims.

## Workflow Syntax Review

Ruby YAML parse passed for:

- `.github/workflows/ci.yml`;
- `.github/workflows/macos-ci.yml`;
- `.github/workflows/windows-ci.yml`.

## Commands Run

```bash
rg -n "Linux.*source-of-truth|Linux.*reviewed static-first package|Linux remains the strongest reviewed" .github/workflows/ci.yml README.md INSTALL.md docs/maintainer_guide.md
rg -n "Windows.*supplemental CMake install/downstream|Windows reviewed scope remains CMake-first|Windows staged exclusions remain|test_threads|test_sprint4_integration|test_fuzz" .github/workflows/windows-ci.yml README.md INSTALL.md docs/maintainer_guide.md
! rg -n "macOS.*supplemental.*install|supplemental.*macOS.*install|macOS supplemental package confidence|not claim full reviewed|do not become reviewed macOS install/export|macOS full install/export parity|static-first install/export confidence" README.md INSTALL.md docs/maintainer_guide.md .github/workflows/macos-ci.yml
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family ci --check
python3 scripts/normalize_report_index.py --family package --check-freshness
python3 scripts/normalize_report_index.py --family ci --check-freshness
ruby -e 'require "yaml"; ARGV.each { |p| YAML.load_file(p) }' .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml
bash scripts/static_package_deferral_check.sh
rg -n "shared-library packaging|dynamic ABI compatibility|runtime-loader compatibility|package-manager support|static/shared selectors|broader macOS platform parity|Windows Makefile parity|Windows.*pkg-config parity|Windows reviewed install-validation" README.md INSTALL.md docs/maintainer_guide.md .github/workflows/*.yml
git diff --check
```

## Day 12 Handoff

Day 12 should run the formal quality gate execution for changed surfaces:

1. Re-run workflow YAML and support-tier scans.
2. Re-run package/CI report normalization and freshness checks.
3. Re-run static package deferral guard.
4. Re-run install/export scripts if the gate needs a fresh complete proof
   record.
5. Confirm whether `.c` or `.h` files changed and run `make format`,
   `make lint`, and `make test` only if required.

## Day 11 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Non-selected lanes remain staged or unsupported unless explicitly promoted. | Complete | Linux unchanged; Windows supplemental/staged boundaries preserved; macOS Homebrew GCC remains supplemental. |
| Sprint 143 static-first package contract remains intact. | Complete | Static package deferral guard passed and package rows remain static-first proof-owner metadata. |
| Selected-lane changes do not create undocumented platform claims. | Complete | Unsupported-claim scan found only explicit non-claims and selected-lane documentation names the macOS scope. |
