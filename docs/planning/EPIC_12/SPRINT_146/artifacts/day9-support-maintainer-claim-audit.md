# Day 9 Support And Maintainer Claim Audit

## Scope

Day 9 audits support and maintainer surfaces for claim coherence after the
public Day 8 audit. The audit checks that report rows, benchmark guidance,
package/ABI guards, CI comments, install validation scripts, and Sprint 146
planning artifacts preserve the same evidence boundaries as public docs.

Audited surfaces:

- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- `tests/corpus/schemas/report_index_fields.md`
- `tests/corpus/manifests/report_families.tsv`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `scripts/static_package_deferral_check.sh`
- Sprint 146 artifacts through Day 8

## Support Claim Inventory

| Support Claim | Owner Surface | Evidence | Audit Result |
| --- | --- | --- | --- |
| Source-controlled report-family rows define row meaning, freshness policy, support tier, owner, claim scope, and non-claims. | `tests/corpus/manifests/report_families.tsv`; `tests/corpus/schemas/report_index_fields.md` | Day 5 schema and report checks passed. | Supported; rows remain advisory/metadata unless generated evidence or hosted logs exist. |
| Generated oracle, benchmark, sentinel, guardrail, dead-code, and coverage rows are local/generated evidence, not source-controlled pass proof. | report schema docs; report-family rows; benchmark docs; Day 5 log | Day 5 local oracle/report refresh generated ignored files only. | Supported; no generated freshness or release proof claim found. |
| Benchmark and sentinel outputs are local measurement context. | `benchmarks/README.md`; `tests/corpus/manifests/report_families.tsv`; `docs/maintainer_guide.md` | Sprint 142 sentinel governance; Day 3 support inventory. | Supported; portable performance, benchmark superiority, and state-of-the-art claims remain rejected. |
| Dead-code rows classify local report output. | `tests/corpus/manifests/report_families.tsv`; `docs/maintainer_guide.md`; Linux CI dead-code lane | Day 6 latest master dead-code job succeeded. | Supported; no zero-dead-code guarantee claim found. |
| Coverage rows are supplemental local/hosted artifacts. | `tests/corpus/manifests/report_families.tsv`; `.github/workflows/ci.yml`; `docs/maintainer_guide.md` | Day 6 latest master coverage job succeeded. | Supported; no coverage-completeness claim found. |
| Static-first package contract is executable and guarded. | `scripts/static_package_deferral_check.sh`; `tests/test_install.sh`; `tests/test_cmake_install.sh`; workflows; maintainer guide | Day 5 package checks passed; Day 6 latest master hosted package lanes succeeded. | Supported; shared-library, dynamic ABI, loader, package-manager, and selector non-claims remain explicit. |
| CI rows identify reviewed hosted lanes but do not replace hosted logs. | report-family CI row; workflow comments; Day 6 intake | `gh` found latest master hosted runs; no `sprint-146` hosted run exists yet. | Supported; branch-specific hosted Sprint 146 status remains pending. |
| Windows support remains CMake-first with staged exclusions. | `.github/workflows/windows-ci.yml`; `INSTALL.md`; maintainer guide; Day 7 reconciliation | Latest master Windows run succeeded with expected `56` CTest count. | Supported; Windows Makefile, `pkg-config`, reviewed install-validation parity, and staged pthread/POSIX tests remain unpromoted. |

## Report And Benchmark Non-Claim Summary

| Surface | Preserved Non-Claims |
| --- | --- |
| Report schema | Source-controlled contract rows are advisory or deferred; they are not pass evidence. |
| Report-family manifest | Generated local report rows do not create hosted CI proof, broad corpus completeness, external-library parity, platform portability, release benchmark proof, coverage completeness, or zero-dead-code claims. |
| Benchmark docs | Benchmark, sentinel, and guardrail rows are local measurement or structural context; they do not prove portable performance or state-of-the-art behavior. |
| Normalized report index | Missing generated rows expose absent local reports; they do not manufacture pass evidence or freshness proof. |
| Day 5 validation log | Local generated corpus/oracle rows are ignored `build/` artifacts and remain reproducibility evidence only. |

## Package And ABI Support Boundary

The package/ABI support boundary remains coherent across scripts, workflows,
docs, and report rows:

- `scripts/static_package_deferral_check.sh` guards `BUILD_SHARED_LIBS=ON`
  rejection, explicit static CMake target shape, static install metadata, no
  public export/import macro, no shared ABI metadata, no package selector, and
  deferred support wording.
- `tests/test_install.sh` proves local Make install/`pkg-config` behavior for
  the static archive package surface.
- `tests/test_cmake_install.sh` proves local CMake install/export,
  downstream `find_package(Sparse)`, exact-version behavior, mismatch-version
  rejection, static imported metadata, and no shared imported metadata.
- Linux CI carries a reviewed static-first package-contract lane.
- macOS CI carries reviewed static-first Make install/`pkg-config` and CMake
  install/export lanes.
- Windows CI carries supplemental CMake install/downstream confidence, not a
  reviewed Windows install-validation parity claim.

The following remain explicit non-claims:

- shared-library support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager distribution;
- static/shared package selector support;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- broad platform parity.

## Support-Surface Fix Or Defer List

| Finding | Action |
| --- | --- |
| Report schema and report-family rows preserve source-controlled versus generated-local boundaries. | No Day 9 fix needed. |
| Benchmark/sentinel guidance preserves local-only and no-portable-performance boundaries. | No Day 9 fix needed. |
| Package scripts and workflow comments preserve static-first and shared-library deferral boundaries. | No Day 9 fix needed. |
| Windows workflow comments and maintainer guidance preserve staged blocker wording and `56` CTest expectation. | No Day 9 fix needed. |
| Maintainer guide uses "parity" in reviewed CMake or bounded internal-fixture contexts and pairs broad parity with non-claims. | No Day 9 fix needed; preserve Day 8 public boundary. |
| Sprint 146 still lacks branch-specific hosted CI evidence. | Defer to PR/branch CI; do not claim hosted Sprint 146 pass yet. |
| Generated benchmark, coverage, dead-code, and sentinel reports were not refreshed on Day 5. | Keep freshness claims out of closeout unless a later day explicitly regenerates and checks those families. |

## Final Support Claim Audit Notes

No support or maintainer wording issue required a Day 9 source fix. The support
surfaces agree with public docs and the Day 2-7 evidence inventories:

- report rows preserve row meaning and freshness semantics;
- source-controlled report rows do not imply generated pass evidence;
- benchmark and sentinel rows remain local measurement context;
- package/ABI wording remains static-first;
- CI comments preserve reviewed, supplemental, staged, hosted-only, and
  deferred distinctions;
- Windows staged blockers remain visible;
- no state-of-the-art, broad parity, portable performance, shared-library ABI,
  package-manager, or generated-report freshness claim is promoted.

## Day 10 Handoff

Day 10 should consolidate residuals from Sprints 137-145 and Days 2-9 into a
prioritized residual queue. The queue should preserve the same support
boundaries recorded here and give each residual an owner, blocker,
prerequisite evidence, and promotion gate.
