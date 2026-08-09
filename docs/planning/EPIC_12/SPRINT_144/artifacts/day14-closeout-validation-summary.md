# Sprint 144 Day 14 Closeout Validation Summary

## Purpose

Finalize Sprint 144 artifacts, validation evidence, selected-lane status,
Sprint 145 adoption handoff, and retrospective inputs.

## Closeout Decision

Sprint 144 is complete. The selected lane, **macOS reviewed static-first
install/export proof**, is closed for the maintained static archive package
contract.

The promoted macOS proof is scoped to hosted `macos-latest` execution of:

- `bash tests/test_install.sh`;
- `bash tests/test_cmake_install.sh`;
- `bash scripts/static_package_deferral_check.sh`.

The promotion changes support-tier ownership and CI wording around already
maintained static-first proof commands. It does not add new C/header behavior,
new package formats, shared-library support, ABI compatibility guarantees,
runtime-loader guarantees, package-manager support, or broad platform parity.

## Deliverable Traceability

| Sprint 144 item | Closeout status | Primary evidence |
| --- | --- | --- |
| Item 1: Platform Lane Selection | Complete | Day 1 intake and Day 2 lane scoring selected macOS reviewed static-first install/export proof. |
| Item 2: Source/Script Portability Fixes | Complete | Day 3-5 showed no source portability defect was required for the selected lane; Day 5 updated workflow proof ownership. |
| Item 3: CI Promotion Implementation | Complete | Day 6-7 designed and reviewed the macOS workflow promotion while preserving Linux and Windows boundaries. |
| Item 4: Package/Report Integration | Complete | Day 8 updated `tests/corpus/manifests/report_families.tsv` CI row semantics and validated package/CI report indexes. |
| Item 5: Documentation Alignment | Complete | Day 9 aligned `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` with the earned macOS reviewed static-first proof. |
| Item 6: Validation | Complete | Day 10-12 recorded focused install/export proof, report checks, workflow syntax checks, claim scans, and the formal quality gate. |
| Item 7: Closeout | Complete | Day 13-14 published promotion evidence, residual non-claims, adoption handoff, and closeout notes. |

## Final Artifact Package

Sprint 144 produced:

- `PLAN.md`;
- `WORKING_NOTES.md`;
- Day 1 platform promotion intake;
- Day 2 platform lane selection;
- Day 3 blocker reproduction evidence baseline;
- Day 4 portability design;
- Day 5 source and script fix batch;
- Day 6 CI promotion design;
- Day 7 CI promotion implementation;
- Day 8 package and report integration;
- Day 9 documentation support-tier alignment;
- Day 10 selected-lane validation pass;
- Day 11 cross-platform non-regression review;
- Day 12 quality gate execution;
- Day 13 promotion evidence and residual non-claims;
- Day 14 closeout validation summary.

## Changed Surface Summary

| Surface | Status |
| --- | --- |
| `.github/workflows/macos-ci.yml` | Updated to treat macOS Make install/`pkg-config` and CMake install/export package proof as reviewed static-first proof. |
| `tests/corpus/manifests/report_families.tsv` | Updated CI report row semantics to include macOS reviewed static-first install/export proof. |
| `README.md` | Updated platform support summary for macOS reviewed static-first proof and preserved non-claims. |
| `INSTALL.md` | Updated validation story, supported-platform table, and interpretation guidance. |
| `docs/maintainer_guide.md` | Updated maintainer support-tier guidance and release/check expectations. |
| Sprint 144 planning artifacts | Added day-by-day evidence, closeout summary, residual non-claims, and adoption handoff. |

No `.c` or `.h` files changed during Sprint 144.

## Final Validation Summary

| Check | Result |
| --- | --- |
| Ruby workflow YAML parse for Linux, macOS, and Windows workflows | Passed |
| `python3 scripts/validate_corpus_schema.py` | Passed |
| `python3 scripts/normalize_report_index.py --family package --check` | Passed: 6 rows |
| `python3 scripts/normalize_report_index.py --family ci --check` | Passed: 1 row |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Passed: 6 source-controlled advisory rows |
| `python3 scripts/normalize_report_index.py --family ci --check-freshness` | Passed: 1 source-controlled advisory row |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| stale macOS supplemental install/export wording scan | Passed |
| Linux source-of-truth preservation scan | Passed |
| Windows reviewed/supplemental/staged preservation scan | Passed |
| unsupported package/platform claim scan | Passed; matches are explicit non-claims |
| `git diff --check` | Passed |

Day 12 remains the full focused local install/export proof record:

- `bash tests/test_install.sh`: passed, 23 passed and 0 failed;
- `bash tests/test_cmake_install.sh`: passed, 26 passed, 0 failed, 0 skipped.

The full C quality gate `make format && make lint && make test` was not
required because no `.c` or `.h` files changed.

## Hosted-CI Proof Boundary

Local validation proves the commands, scripts, report rows, workflow syntax,
and support-tier wording available in this checkout. The final external proof
owner for the reviewed macOS lane remains PR CI on GitHub-hosted
`macos-latest`.

If hosted macOS CI fails, the failure should be treated as a Sprint 144
selected-lane blocker and fixed before merge. If hosted CI passes, Sprint 145
can treat the macOS static-first install/export support tier as reviewed for
the maintained static archive package contract.

## Preserved Non-Claims

Sprint 144 still does not claim:

- shared-library build/install/export support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager availability;
- static/shared package selector support;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- Windows staged test closure;
- broader macOS platform parity beyond reviewed static-first install/export
  proof;
- portable performance parity;
- state-of-the-art sparse linear algebra status from platform support work.

## Sprint 145 Adoption Handoff

Sprint 145 can start from this adoption-facing platform contract:

- Linux remains the strongest reviewed source-of-truth baseline for Make,
  CMake, dead-code, package-contract, and broader local quality proof.
- macOS now carries reviewed static-first Make install/`pkg-config` and CMake
  install/export proof for the maintained static archive package contract.
- Windows remains CMake-first: reviewed MSVC CMake subset proof plus
  supplemental CMake install/downstream confidence.
- Windows staged `test_threads`, `test_sprint4_integration`, and `test_fuzz`
  remain future portability work due pthread/POSIX blockers.
- Adoption guidance should prefer static-first install language and avoid
  shared-library, ABI, loader, package-manager, Windows Makefile,
  Windows `pkg-config`, Windows install-validation parity, and broad platform
  parity claims.

## Retrospective Inputs

What worked:

- Selecting a single narrow platform lane avoided partial promotion across
  Linux, macOS, and Windows at once.
- Keeping the proof commands unchanged reduced implementation risk.
- Report, docs, workflow comments, and planning artifacts now tell the same
  support-tier story.

What remains risky:

- The reviewed macOS claim still depends on hosted `macos-latest` CI for final
  external proof.
- Windows install/downstream remains supplemental and CMake-first only.
- Static-first packaging remains a product decision, not shared-library or ABI
  maturity.

Follow-through candidates:

- Sprint 145 adoption docs should surface the static-first platform contract
  without implying package-manager or shared-library availability.
- A later Windows/platform sprint can choose either Windows install-validation
  parity or staged pthread/POSIX test closure as a single selected lane.
- A future package/ABI epic can revisit shared libraries only with explicit ABI,
  symbol, loader, and downstream proof.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected platform lane is closed or rejected with source-level proof. | Complete | macOS reviewed static-first install/export proof is closed for existing static-first commands; no source portability defect was required. |
| Support-tier docs, CI comments, report rows, and artifacts agree. | Complete | Changed surfaces summary, validation scans, and Day 8-13 artifacts align on the same scope. |
| Sprint 145 can start from a clear adoption-facing platform contract. | Complete | Sprint 145 handoff names Linux, macOS, Windows, staged blockers, and unsupported adoption claims. |
