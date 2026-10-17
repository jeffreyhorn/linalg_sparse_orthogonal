# Sprint 214 Day 14: Decision Closeout Review

## Final Disposition

Sprint 214 selects complete ABI/release design with stronger static-first
deferral. It did not implement a shared target, approve a dynamic ABI, or
produce a release. The current static Make and CMake package and out-of-tree
consumers passed locally on Day 13. That evidence is not a branch-specific
hosted run or a tagged release-candidate result.

| Item | Final status | Evidence and claim limit |
| --- | --- | --- |
| 214.1 ABI/Release Scope Decision | Complete | Day 2 compares a narrow Linux shared proof with design/deferral and selects the latter. Sprint 170 static-first and Sprint 207 provider deferrals remain in force. |
| 214.2 ABI Policy Design | Complete as future policy | Days 3-5 inventory public/export risks and specify a provisional eight-function subset, visibility/allowlist, ownership/layout/index/scalar, ABI epoch/break, and ELF/Mach-O/PE loader gates. No subset is approved as a shipped ABI and no cross-version guarantee exists. |
| 214.3 Build/Install Proof Or Guard | Complete for selected deferral | Days 6-8 reject unsupported CMake shared/link-mode selectors, inspect static installed artifact/metadata contracts, add 14 mutation fixtures, and register Linux/macOS/Windows package checks. No shared artifact or hosted branch result follows. |
| 214.4 Downstream Consumer Evidence | Complete for selected static scope | Days 9-10 execute installed Make/`pkg-config` and CMake `find_package(Sparse)` consumers and negative missing-archive/version checks. Shared consumer and wrong/missing dynamic-loader cases remain specified future tests, not executed proof. |
| 214.5 Release Checklist | Complete as gate design | Days 11-12 define changelog, clean commit, tag/signing, reproducible source archive, manifest, provenance, platform CI, approval and rollback gates in `../RELEASE_READINESS_CHECKLIST.md`. Required candidate evidence is absent, so release readiness is blocked. |
| 214.6 Validation And Closeout | Complete for local selected path | Day 13 integrated static install/consumer, fixture, guard, reviewed compile, and 60/60 CTest gates passed. Day 14 reconciles this result and the residual queue; hosted candidate, shared, and release gates remain unrun. |

## Validation Boundary

The Day 13 transcript is `day13-integrated-validation.md`: Make install
25/25; CMake install 30/30; static fixture suite 14/14; static, generated
API, support-doc, and provider guards; reviewed compile quality; reviewed
CMake quality with 60/60 CTest; Windows workflow-owner regression; and Bash
syntax checks all passed locally. No `*.c` or `*.h` changed in Sprint 214, so
the separate `make format && make lint && make test` command is not required.
The reviewed compile target exercised format/lint, and reviewed CMake ran
CTest, but neither is represented as a `make test` run.

Day 14 recheck results are recorded below after the final documentation
edits. No branch-specific hosted Linux/macOS/Windows results, PowerShell
runtime, shared loader/consumer proof, archive reproducibility, release
provenance, tag, or approval is claimed.

## Residuals And Retrospective Handoff

`../RESIDUAL_QUEUE.md` lists the candidate-specific static release gates,
shared product decision, dynamic ABI/compatibility tests, per-platform
loader/consumer tests, hosted evidence, and separate distribution decisions.
The Epic 19 project-plan snapshot points to that queue. Future work should
not turn the Day 3 provisional symbols, existing package version, or local
Homebrew proof into a dynamic ABI, public provider, or release claim.

The retrospective can treat this as a completed *decision and guarded
deferral* sprint. It should evaluate the cost and utility of the stronger
static contract tests separately from the unimplemented shared/release
product paths, carry the prioritized queue forward, and preserve the exact
non-claims. Sprint 215 can proceed without assuming any shared or release
promotion.

## Day 14 Final Recheck

| Command | Result | Scope |
| --- | --- | --- |
| `bash tests/test_install.sh` | PASS: 25/25 | Fresh Make static install, installed `pkg-config` consumers, missing-archive rejection, and uninstall. |
| `bash tests/test_cmake_install.sh` | PASS: 30/30, 0 skipped | Fresh CMake static install/export, installed `find_package(Sparse)` consumers, version mismatch and missing-archive diagnostics. |
| `python3 tests/test_static_install_contract.py` | PASS: 14 tests | Static artifact and metadata positive/mutation fixtures. |
| `bash scripts/static_package_deferral_check.sh` | PASS | Unsupported shared/link-mode selectors, static package metadata, and public non-claims. |
| `make docs-check` | PASS | Local Doxygen coverage for 18 checked-in public headers. |
| `make api-docs-local-only` | PASS | Local-only workflow/ignore checks and the complete regression suite; generated HTML remains ignored. |
| `make support-docs-guard` | PASS | Public support wording. |
| `python3 tests/test_package_manager_deferral_guard.py` | PASS | Provider-claim regression fixtures. |
| `make format-check` | PASS | C/header formatting unchanged. |
| `make source-list-check` | PASS: 49 library sources | Reviewed source inventory unchanged. |

Day 14 changed only Markdown. The full `make quality-review-compile`,
`make quality-review-cmake` (60/60 CTest), and
`make package-manager-deferral-guard` passed on Day 13 after the last
build/test/workflow/public-claim changes. They were not rerun for Day 14's
planning/checklist edits; the affected static install and docs/claim checks
above were rerun. No C/header quality trigger applies. Final whitespace and
worktree status checks passed: `git diff --check` was clean, a trailing-
whitespace scan of the new Sprint 214 Markdown found no matches, and
`git diff --name-only -- '*.c' '*.h'` was empty. The `sprint-214` worktree
remains uncommitted, with prior sprint build/test/workflow edits preserved.
