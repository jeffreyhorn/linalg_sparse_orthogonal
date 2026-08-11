# Sprint 153 ABI And Package Handoff

## Purpose

Sprint 153 is the shared-library ABI product-decision sprint. Sprint 152 hands
off a clarified generated-report freshness boundary so ABI and package work can
use the report index without mistaking local oracle freshness for package,
loader, shared-library, or release evidence.

## Sprint 153 Starting Scope

From `docs/planning/EPIC_13/PROJECT_PLAN.md`, Sprint 153 is:

**Sprint 153: Shared-Library ABI Product Decision**

Goal: make a product-level shared-library ABI decision and either implement the
first supported shared surface or publish a stronger static-first deferral with
exact blockers.

Planned Sprint 153 work includes:

- ABI surface audit
- platform loader audit
- shared-library product decision
- selected implementation or stronger static-first deferral
- downstream package proof
- documentation alignment
- validation and closeout

## Inputs From Sprint 152

Sprint 152 provides:

- selected generated oracle freshness target:
  `make report-index-oracle-freshness`
- selected local oracle row-count policy:
  `52` total rows, `23` QR rows, `26` partial-SVD rows, and `3`
  generated-reference rows
- report-family metadata distinguishing generated-local, source-controlled,
  advisory, hosted-external, optional, and local-only rows
- documentation that selected local oracle rows are not package, ABI, hosted CI,
  platform, performance, or release proof
- focused report-index tests for required, strict, advisory, stale, missing,
  failing, partial, missing-solver-family, and missing-fixture-key selected
  oracle cases

## ABI Work Must Not Infer

Sprint 153 should not infer any of the following from Sprint 152 generated
oracle freshness:

- shared-library ABI support
- dynamic-loader support
- symbol visibility stability
- CMake shared target correctness
- `.so`, `.dylib`, `.dll`, or import-library support
- package-manager availability
- Windows Makefile parity
- Windows pkg-config execution parity
- hosted CI package proof
- portable platform support
- release artifact readiness

## Recommended Sprint 153 Opening Checks

Before changing ABI/package behavior, Sprint 153 should run or inspect:

```sh
make report-index-oracle-freshness
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family package --family ci --family runtime_backend --check-freshness
bash tests/test_install.sh
cmake -S . -B build-cmake-handoff -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake-handoff
```

If `.c` or `.h` files change in Sprint 153, run the full C quality gate required
by project practice:

```sh
make format && make lint && make test
```

## Decision Inputs To Produce In Sprint 153

Sprint 153 should produce an explicit decision record answering:

- Is shared-library support implemented now, or is static-first deferral
  strengthened?
- Which public headers, structs, macros, symbols, allocator behaviors, callback
  contracts, and version metadata are ABI-governed?
- Which platforms are supported or explicitly deferred?
- What loader proofs exist for Linux, macOS, and Windows?
- What downstream CMake and pkg-config proofs exist?
- Which claims remain static-first only?

## Handoff Risk Register

| Risk | Mitigation |
| --- | --- |
| Local oracle freshness is cited as package or ABI proof. | Keep `make report-index-oracle-freshness` scoped to fixture-local generated evidence only. |
| Source-controlled package rows are mistaken for generated freshness. | Use the report-family `row_origin` and `freshness_policy` fields before citing a row. |
| Shared-library work expands claims without loader tests. | Require product-decision evidence before docs claim shared support. |
| Static-first deferral remains vague. | If shared support is deferred, list exact blockers and maintained rejection tests. |
| Platform-specific evidence is flattened into broad platform support. | Preserve Linux/macOS/Windows proof boundaries in report rows and docs. |

## Non-Claims

This handoff does not claim shared-library support, shared ABI stability,
dynamic-loader support, package-manager availability, broad platform support,
or release artifact readiness.
