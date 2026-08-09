# Sprint 145 Day 11 Cross-Surface Coherence

## Purpose

Day 11 checked that the adoption-facing surfaces tell one consistent story
after the README, INSTALL, examples, cookbook, solver-selection, and public
header cleanup from Days 5-10.

Reviewed surfaces:

- `README.md`
- `INSTALL.md`
- `examples/README.md`
- `docs/cookbook.md`
- `docs/solver_selection.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`
- selected public headers from Day 10
- `tests/corpus/manifests/report_families.tsv`
- `tests/corpus/schemas/report_index_fields.md`
- Sprint 139-144 retrospective/artifact evidence

## Coherence Finding

The front-door adoption route is now consistent:

1. README owns the short local build and first-solve route.
2. Examples own runnable build-tree entry points and example-local diagnostics.
3. Cookbook owns data-first CSR, CSC, and Matrix Market recipes.
4. Solver-selection owns problem-shape routing and diagnostics escalation.
5. INSTALL owns static-first install/downstream setup and platform support.
6. Benchmarks own local measurement/report interpretation.
7. Maintainer guide owns support-tier and proof-owner interpretation.
8. Public headers own API-local contracts and point deeper readers to examples
   or solver-selection guidance.

## Coherence Fix

One stale report-family row was repaired.

| Surface | Issue | Fix |
| --- | --- | --- |
| `tests/corpus/manifests/report_families.tsv` | The `runtime_backend` governance row still described a Sprint 142 defer row even though Sprint 142 closed the runtime/backend governance contract. | Promoted the row to `runtime_backend_governance_policy`, `status=advisory`, `freshness_policy=source_controlled`, and pointed it at maintainer/benchmark policy owners while keeping generated sentinel measurements under the `sentinel` family. |
| `scripts/validate_corpus_schema.py` | The schema only accepted `deferred_governance` for this family. | Added `runtime_backend_governance_policy` as an allowed row meaning. |
| `tests/corpus/schemas/report_index_fields.md` | Runtime/backend schema guidance only described deferred rows. | Added source-controlled policy-row guidance and kept sentinel measurements separate. |
| `tests/test_normalize_report_index.py` | Normalizer expectations still looked for a deferred runtime/backend contract row. | Updated the expected row id and freshness diagnostic to the source-controlled advisory path. |

## Support-Tier Consistency

- Linux remains the strongest reviewed source of truth and includes reviewed
  static-first package-contract proof.
- macOS carries reviewed static-first Make install/`pkg-config` and CMake
  install/export proof for the maintained static archive package contract.
- Windows remains CMake-first: reviewed MSVC CMake subset plus supplemental
  CMake install/downstream confidence.
- Windows Makefile parity, Windows `pkg-config` parity, Windows reviewed
  install-validation parity, and staged pthread/POSIX tests remain non-claims.
- Package rows identify proof owners and static-first scope; they do not
  become fresh install-run evidence unless the install commands are run.

## Advanced-Link Routing

| Need | Public route confirmed |
| --- | --- |
| First build and solve | `README.md#start-here` -> `examples/README.md#start-here` |
| CSR/CSC/Matrix Market data | `README.md` -> `docs/cookbook.md#first-use-ladder` |
| Solver choice | README workflow table -> `docs/solver_selection.md#choose-the-smallest-workflow` |
| Diagnostics escalation | examples/cookbook -> `docs/solver_selection.md#diagnostics-handoff` |
| Install/downstream consumers | README -> `INSTALL.md#start-here` |
| Platform support interpretation | `INSTALL.md#supported-platforms` -> `docs/maintainer_guide.md` |
| Benchmarks and report indexes | README/solver-selection -> `benchmarks/README.md` |
| API-local ownership and errors | docs/examples -> selected public headers under `include/` |

## Claim Boundary Review

The coherence pass preserved these boundaries:

- QR evidence remains fixture-local and does not become broad QR or
  external-library parity.
- Partial-SVD evidence remains fixture-local and does not become broad
  repeated-spectrum, performance, or partial-result proof.
- Runtime/backend sentinels remain local measurement/governance context, not
  portable performance or backend portability proof.
- Report indexes remain navigation/freshness aids, not release proof.
- Static-first package support remains distinct from shared-library,
  dynamic-ABI, runtime-loader, and package-manager support.
- Platform tiers remain explicit and do not claim Windows install-validation
  parity or broad macOS platform parity.

## Validation

| Check | Result |
| --- | --- |
| `python3 scripts/validate_corpus_schema.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness` | Passed with one source-controlled advisory row. |
| stale runtime/backend defer wording scan | Passed; only sentinel `no backend policy closure` non-claim remains. |
| unsupported-claim scan across public docs, headers, maintainer guide, benchmark guide, and report schemas | Passed; matches are explicit non-claims, support boundaries, or bounded evidence. |
| advanced-link routing scan | Passed; adoption surfaces link to the expected owners. |

## Completion Criteria

- Public adoption surfaces now tell one consistent story.
- Stale runtime/backend report-family wording was repaired.
- Support-tier, package, platform, report, QR, partial-SVD, and
  runtime/backend claim boundaries remain intact.
- Coherence fixes are documented and validated.

## Day 12 Handoff

Day 12 should run the formal validation gate across documentation/reference
checks, report-index checks, install/downstream checks selected by the sprint
plan, and any C/header gates still required by the cumulative Sprint 145 diff.
