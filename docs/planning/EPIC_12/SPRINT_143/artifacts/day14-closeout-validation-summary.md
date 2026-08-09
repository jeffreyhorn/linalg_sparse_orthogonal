# Sprint 143 Day 14 Closeout Validation Summary

## Purpose

Finalize Sprint 143 artifacts, validation evidence, working notes, and the
Sprint 144 platform-promotion handoff.

## Deliverable Traceability

| Sprint 143 item | Closeout status | Primary evidence |
| --- | --- | --- |
| Item 1: ABI Feasibility Audit | Complete | Day 1 intake, Day 2 public header/symbol audit, Day 3 install/export metadata audit, Day 4 platform/loader risk audit |
| Item 2: Product Decision | Complete | Day 5 package/ABI decision selected stricter static-first-only support |
| Item 3: Implementation Batch | Complete | Day 7 static-first metadata batch and Day 8 install proof diagnostics |
| Item 4: Downstream Consumer Proof | Complete | Day 9 downstream consumer proof and Day 12 focused validation |
| Item 5: CI/Packaging Alignment | Complete | Day 10 CI and package-report alignment |
| Item 6: Documentation Alignment | Complete | Day 11 README, INSTALL, and maintainer guide alignment |
| Item 7: Validation and Closeout | Complete | Day 12 focused validation, Day 13 quality/claim closure, Day 14 closeout |

## Final Artifact Package

Sprint 143 produced:

- `PLAN.md`;
- `WORKING_NOTES.md`;
- Day 1 package/ABI intake;
- Day 2 public header and symbol audit;
- Day 3 install/export metadata audit;
- Day 4 platform and loader risk audit;
- Day 5 package/ABI product decision;
- Day 6 static-first implementation design;
- Day 7 package batch 1 static-first metadata;
- Day 8 package batch 2 install proof diagnostics;
- Day 9 downstream consumer proof;
- Day 10 CI and package report alignment;
- Day 11 documentation alignment;
- Day 12 focused package validation;
- Day 13 quality gate and claim closure;
- Day 14 closeout validation summary.

## Final Validation

| Check | Result |
| --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `python3 scripts/normalize_report_index.py --family package --check` | Passed: 6 rows |
| `python3 scripts/normalize_report_index.py --family package --check-freshness` | Passed: 6 source-controlled advisory rows |
| Workflow YAML parse | Passed for Linux, macOS, and Windows workflows |
| Final package/docs/workflow/artifact claim-boundary scan | Passed; matches are explicit non-claims, support-tier boundaries, sprint planning text, or unrelated bounded algorithm notes |
| `git diff --check` | Passed |
| Final trailing-whitespace scan | Passed |
| Generated-output hygiene | Passed; only Sprint 143 planning files are untracked, and `build/` remains ignored |

Day 13 remains the current full focused install proof record:

- `bash tests/test_install.sh`: 23 passed, 0 failed;
- `bash tests/test_cmake_install.sh`: 26 passed, 0 failed, 0 skipped.

No `.c` or `.h` files changed during Sprint 143, so the full C gate
`make format && make lint && make test` was not required.

## Final Implemented Package Behavior

Sprint 143 leaves the project with a stricter static-first package contract:

- CMake rejects `BUILD_SHARED_LIBS=ON`;
- `sparse_lu_ortho` remains an explicit `STATIC` target;
- CMake install metadata installs the archive without shared/runtime
  destinations;
- installed `sparse.pc` metadata describes static archive package metadata;
- `pkg-config` metadata has no `Libs.private` or static/shared selector;
- install proofs reject `.so`, `.so.*`, `.dylib`, and `.dll` artifacts;
- CMake package proof rejects shared/module imported targets and shared
  imported locations;
- Make/`pkg-config` consumers compile, link, and run;
- CMake installed consumers configure, build, and run;
- exact-version CMake consumers configure, build, and run;
- package report rows remain source-controlled proof-owner metadata.

## Preserved Non-Claims

Sprint 143 still does not claim:

- shared-library build/install/export support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager availability;
- static/shared CMake or `pkg-config` selector support;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- macOS reviewed install/export parity;
- portable performance from package proof;
- state-of-the-art status from package/ABI work.

## Sprint 144 Platform-Promotion Handoff

Sprint 144 should treat platform promotion as a separate product/support-tier
decision, using Sprint 143 static-first package semantics as input.

Recommended Sprint 144 questions:

1. Should macOS static-first install/export confidence be promoted from
   supplemental to reviewed parity, and what hosted-runner repetition is
   required?
2. Should Windows CMake install/downstream confidence be promoted from
   supplemental to reviewed install-validation parity, and what exact static
   scope should be reviewed?
3. Should Windows package proof remain CMake-first only, or should Makefile or
   `pkg-config` parity remain explicit non-goals?
4. Which staged Windows blockers remain source-level blockers rather than
   package-contract blockers?
5. What failure ownership and retry policy is required before a supplemental
   package lane becomes reviewed?

## Closeout Decision

Sprint 143 is complete. It selected the static-first package path, implemented
the selected mechanics, strengthened downstream proof, aligned CI and docs,
published current validation evidence, and routed unearned package/platform
claims forward.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 143 deliverables are present and traceable to Items 1-7. | Complete | Deliverable traceability table maps each item to artifacts. |
| Validation evidence is current and reproducible. | Complete | Final validation table and Day 13 install proof counts name exact commands and results. |
| Remaining package/platform work is explicitly routed forward. | Complete | Sprint 144 handoff lists platform-promotion questions and preserved non-claims. |
