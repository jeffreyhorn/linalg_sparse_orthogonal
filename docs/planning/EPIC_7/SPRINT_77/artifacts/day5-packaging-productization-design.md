# Sprint 77 Day 5 Artifact: Packaging/Productization Design

Date: 2026-06-17
Branch: sprint-77

## Purpose

Define the bounded implementation contract for the first Sprint 77
release/install improvement batch before edits begin, with one explicit owner
for package-facing clarity and one explicit non-touch fence around unsupported
platform, ABI, or shared-library widening.

## Main Result

Sprint 77 now has one explicit first implementation contract:

- required implementation center:
  - `INSTALL.md`
- support only if the first batch truly forces it:
  - `docs/maintainer_guide.md`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `README.md`

## Ownership Split

The first-batch ownership split is now fixed:

- product-facing install and consumer-guidance owner:
  - `INSTALL.md`
- authoritative policy and truthfulness owner:
  - `docs/maintainer_guide.md`
- concrete export and metadata owner:
  - `CMakeLists.txt`
- local install-proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- compact front-door summary:
  - `README.md`

## Why This Is the Right First Design

The first batch should improve release/install clarity first, not package
mechanics first.

That means the first landing should make it easier for a downstream reader to
understand, in one operator-facing place:

- the static-first package shape
- what `pkg-config` and `find_package(Sparse)` actually promise
- what gets installed and how to verify it
- which platform claims are reviewed versus supplemental

This lane ranks above export-metadata edits because:

- the export surface already exists and is real
- the local proof scripts already exist and are real
- the strongest remaining gap is contract readability and truthfulness, not
  missing package machinery

## Preserved Compatibility Checklist

The Day 6 batch must preserve:

- the maintained static-first release shape
- the current `pkg-config` and `find_package(Sparse)` consumer story
- the current bounded ABI/version reading
- Linux as the strongest reviewed truth
- macOS as narrower reviewed plus supplemental install proof
- Windows as reviewed CMake subset and CMake-first consumer lane

## First-Batch Non-Touch Set

The first Sprint 77 implementation batch explicitly does not include:

- shared-library or dynamic-ABI marketing
- broad ABI/version promise widening
- CI workflow edits
- platform-proof expansion
- export metadata churn unless the product-facing wording would otherwise
  become inconsistent
- unrelated solver, capability, backend, or benchmark-governance work

## Day 6 Implication

Day 6 should therefore start from:

- exact first implementation center:
  - `INSTALL.md`
- support only if truly forced:
  - `docs/maintainer_guide.md`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `README.md`

The intended shape is a bounded productization batch:

- clearer operator-facing install and export contract
- clearer proof-owner reading
- clearer reviewed-versus-supplemental platform reading
- no widened product claim beyond maintained evidence
