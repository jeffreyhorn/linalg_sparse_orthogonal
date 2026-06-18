# Sprint 77 Day 7 Artifact: Post-Landing Audit and Rerank

Date: 2026-06-17
Branch: sprint-77

## Purpose

Re-audit the release and platform surface after the Day 6 landing so Sprint 77
targets the strongest remaining bounded seam rather than repeating the
install-guide cleanup.

## Main Result

The Day 6 landing closed the strongest first package contradiction:

- `INSTALL.md` no longer reads like the strongest remaining Sprint 77 seam
- a second operator-facing install-guide batch is not the highest-value next
  move

The strongest remaining seam has now shifted to platform-proof asymmetry and
how it is interpreted across:

- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

## Why The Rerank Shifted

The Day 6 cleanup removed the densest operator-facing ambiguity:

- the static-first package shape now reads more directly
- the downstream consumer story now reads more directly
- the local-versus-reviewed proof split now reads more directly

That means the next highest-value contradiction is no longer "what is the
package contract?" It is now "how narrow are the macOS and Windows proof lanes,
and how should that narrowness be read?"

## Updated Seam Ranking

- required next batch:
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- strongest support-only follow-through:
  - `docs/maintainer_guide.md`
- lower-value support-only follow-through:
  - `README.md`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Interpretation

The useful Day 7 clarification is now explicit:

- the next batch should focus on narrowing the highest-value macOS/Windows
  proof-reading gap
- the strongest remaining risk is not the static-first package contract itself
- it is how readers reconcile:
  - macOS supplemental Make install verification
  - Windows reviewed CMake-only consumer proof
  - the absence of a broader reviewed install-validation parity claim

## Day 8 Implication

The Day 8 design pass should therefore start from:

- exact next design center:
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- strongest support-only follow-through:
  - `docs/maintainer_guide.md`
- lower-value support-only context:
  - `README.md`
  - `CMakeLists.txt`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Exit State

Sprint 77 now has one explicit post-Day-6 rerank:

- do not repeat the install-guide batch
- move next to the bounded macOS/Windows proof-asymmetry seam
- keep support follow-through narrow unless the proof batch truly forces it
