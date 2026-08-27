# Sprint 182 Day 10: Documentation Alignment

## Purpose

Update user-facing and maintainer-facing documentation so Windows report
freshness is clearly represented as a formal Sprint 182 deferral, while the
reviewed Windows CMake build/test and static install/downstream claims remain
intact.

## Documentation Decision

The Day 10 documentation path follows the formal deferral decision instead of a
Windows freshness promotion. Documentation should therefore state that Windows
report freshness is deferred and should not imply generated report freshness
from Windows CMake, CTest, install, downstream consumer, or static package
metadata validation.

## Updated Surfaces

| Surface | Update |
| --- | --- |
| `README.md` | Added the formal Sprint 182 deferral record to the CI and report-index wording and listed the promotion prerequisites for Windows report freshness. |
| `docs/maintainer_guide.md` | Added maintainer guidance that Windows must stay out of selected target `workflow_platforms` while deferral is active and that promotion requires generation, upload, manifest, and guard updates together. |
| `.github/workflows/windows-ci.yml` | Updated workflow comments to exclude generated report freshness explicitly from the reviewed Windows lane. |

## Unchanged Surfaces

| Surface | Reason |
| --- | --- |
| `INSTALL.md` | The package/install support contract did not change. Windows remains reviewed for the CMake static-first install/downstream route, and the deferral concerns generated report freshness only. |
| `benchmarks/README.md` | Selected performance freshness remains Linux hosted for `bench_refactor_csc`; Sprint 182 did not promote or newly defer a Windows benchmark freshness lane beyond the general Windows report freshness decision. |

## Claim Boundary

Windows currently supports reviewed CMake configure, build, `ctest`, static
install, installed CMake package metadata, metadata-only `sparse.pc`
inspection, and installed downstream consumer validation. It does not support
generated report freshness, Makefile parity, package-manager support,
shared-library support, dynamic ABI support, runtime-loader behavior,
pkg-config execution parity, broad Windows parity, portable performance
claims, or broad external-library parity.

## Future Promotion Contract

A future Windows report freshness promotion must add all of these together:

- Windows-safe generated report command or generator path;
- exact selected target manifest `workflow_platforms` and related workflow
  metadata;
- exact selected artifact upload scope with fail-closed missing-file behavior;
- workflow guard allowlist updates;
- public and maintainer documentation updates.

## Validation

Planned Day 10 validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

## Completion Criteria

- Users can tell from the README that Windows report freshness is deferred.
- Maintainers can tell from the guide and workflow comments that Windows CI
  must not drift into selected report freshness without manifest-backed
  promotion.
- Docs preserve package, platform, performance, external-library, ABI, and
  release non-claims.
