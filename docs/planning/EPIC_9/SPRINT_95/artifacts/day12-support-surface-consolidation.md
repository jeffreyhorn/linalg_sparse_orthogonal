# Sprint 95 Day 12: Support Surface Consolidation

## Purpose

Day 12 reconciles install, benchmark, and maintainer support surfaces with the
Day 3 audience ownership model. The cleanup keeps current operational guidance
in permanent docs and leaves sprint chronology in planning artifacts.

## Consolidation Batch

Updated support surfaces:

- `INSTALL.md`
  - collapsed duplicate front-door routing into the opening scope statement
  - renamed the install routing section to `Support Split`
  - kept install ownership on operational setup, installed-consumer detail, and
    install-surface validation
  - preserved local validation scripts as Unix-oriented install/export checks
- `benchmarks/README.md`
  - added an explicit benchmark ownership paragraph near the top
  - clarified that the compile-only gate catches benchmark drift but does not
    own repository-wide reviewed-baseline or maintainer-policy claims
  - kept the live `--sprint86-slice` and `bench-reorder-sprint86` names while
    describing their current meaning as a bounded ND rerun slice
- `docs/maintainer_guide.md`
  - added `Support Surface Ownership`
  - recorded the current support owner split
  - added cross-link rules for install, benchmark, maintainer-policy, and
    historical planning references

## Ownership Notes

### Install

`INSTALL.md` remains the owner for:

- prerequisites and platform setup
- Make and CMake install flows
- static package shape
- downstream `pkg-config` and `find_package(Sparse)` use
- install validation scripts

It now avoids repeating the README adoption path beyond short routing text.

### Benchmarks

`benchmarks/README.md` remains the owner for:

- benchmark command groups
- benchmark target mapping
- CSV fields and report artifacts
- measurement caveats

It now points policy and reviewed-baseline interpretation back to the
maintainer guide instead of restating that policy inline.

### Maintainer Guide

`docs/maintainer_guide.md` remains the owner for:

- reviewed-platform interpretation
- proof ownership
- warning authority
- documentation-placement policy
- support-surface boundary interpretation

The new ownership section gives future cleanup work one stable support map.

## Support Cross-Link Map

| Reader need | Owning surface | Notes |
|---|---|---|
| First local build and solver route | `README.md` | Keep as compact front door. |
| Executable usage example | `examples/README.md` | Link here after README adoption. |
| Install, package, or downstream consumer detail | `INSTALL.md` | Includes validation scripts. |
| Benchmark command or CSV/report meaning | `benchmarks/README.md` | Does not own policy interpretation. |
| Reviewed-platform or proof-owner meaning | `docs/maintainer_guide.md` | Policy home for maintainers. |
| API-local caveat | `include/*.h` | Update source comments, not generated docs. |
| Historical provenance | `docs/planning/**` | Link only when history explains current behavior. |

## Historical Content Kept Out Of Support Surfaces

The following should remain planning-history content unless a current behavior
depends on it:

- sprint-by-sprint development chronology
- retired benchmark targets or old timing evidence
- old proof-owner names after product-oriented owners have landed
- development-closeout explanations that do not affect current install,
  benchmark, or maintainer workflows

The live `bench-reorder-sprint86` target and `--sprint86-slice` flag remain
documented because they are active commands. Permanent docs now describe them as
current bounded ND rerun support surfaces rather than as sprint narrative.

## Validation

- Day 12 changed documentation only.
- No `.c` or `.h` files were modified for Day 12.
- Full `make format && make lint && make test` is not required for this
  docs-only support consolidation.
- `git diff --check` passed.
- Trailing-whitespace scans passed on touched docs and the Day 12 planning
  artifact.
- Focused sprint-era wording scans found only active benchmark command names
  and maintainer-guide policy/provenance references.
