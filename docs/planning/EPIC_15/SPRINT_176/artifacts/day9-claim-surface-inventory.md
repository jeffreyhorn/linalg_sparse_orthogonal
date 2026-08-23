# Day 9: Claim Surface Inventory

## Purpose

Day 9 inventories public, maintainer, and planning claim surfaces before the
Sprint 176 claim recalibration pass. The goal is to separate earned evidence
from local-only, hosted-only, advisory, supplemental, and unsupported wording
before editing any public claims on Day 10.

## Reviewed Surfaces

| Surface | Role | Day 9 finding |
| --- | --- | --- |
| `README.md` | Short adoption front door, capability summary, CI/platform summary, benchmark/report boundary, installation summary, API lists. | Major claim boundaries are visible. Day 10 should add the Sprint 176 allocation-failure proof to the high-level quality/evidence story only if it remains explicitly family-local. |
| `INSTALL.md` | Operational install and static-first package contract. | Static-first package, Windows CMake-first, package-manager deferral, shared-library deferral, and dynamic ABI non-claims remain explicit. No Sprint 176 allocation-failure wording needed here unless package wording is touched. |
| `docs/maintainer_guide.md` | Maintainer evidence ownership, support tiers, report freshness, package/ABI policy, generated-output policy. | Already names the Sprint 176 iterative allocation-failure owner and focused gates. Day 10 should ensure final claim wording points back to this owner rather than restating broad guarantees elsewhere. |
| `docs/api_reference.md` | Public API index and generated HTML boundary. | Correctly keeps exact contracts in public headers and says generated HTML is local-only. No stale Sprint 176 claim found. |
| `docs/tutorial.md` and `docs/cookbook.md` | User workflow guidance. | Should stay workflow-oriented. If allocation-failure wording is added, it should remain narrow and point to repeated-run iterative handles only. |
| `docs/solver_selection.md` | Solver-family decision guide. | Should not absorb allocation-failure proof as solver-selection confidence unless it is clearly limited to CG/GMRES/MINRES repeated-run handles. |
| `benchmarks/README.md` | Benchmark command, report, and measurement interpretation. | Benchmark rows remain local measurement context, not allocation-failure, correctness, release, or state-of-the-art evidence. |
| `examples/README.md` | First-use examples and diagnostics handoff. | Examples remain adoption aids, not proof owners. No Sprint 176 claim should be added here unless phrased as a usage note. |
| `tests/corpus` manifests and schemas | Report-family source-controlled proof-owner rows and generated report metadata. | Generated report rows remain local-only or selected hosted evidence depending on lane. They should not be cited as allocation-failure proof. |
| `.github/workflows/*.yml` | Reviewed hosted evidence lanes. | Hosted lanes cover selected quality, package, report, and performance surfaces. Sprint 176 allocation-failure proof is source-controlled and locally reachable through Make/CMake labels unless a later day promotes it to a hosted lane. |
| `scripts/*_deferral_check.sh` and freshness scripts | Executable guards for package, ABI, package-manager, report, and performance boundaries. | Static/shared/package-manager deferrals remain separate from allocation-failure evidence. Do not overload these guards with Sprint 176 cleanup claims. |
| `docs/planning/EPIC_15/SPRINT_167` through `SPRINT_175` artifacts and retrospectives | Epic 15 evidence and residual history. | Prior sprints closed hosted performance freshness, package/provider deferrals, public-header coherence, API HTML local-only policy, bounded comparison families, and selected cross-platform report freshness. None close broad allocation-failure coverage. |
| `docs/planning/EPIC_15/SPRINT_176` Day 1-8 artifacts | Current sprint evidence trail. | Supports exactly one new proof: iterative repeated-run handle allocation-failure cleanup for selected CG/GMRES/MINRES prepare/growth paths. |

## Evidence-To-Claim Map

| Claim area | Earned wording | Evidence owner | Boundary |
| --- | --- | --- | --- |
| Iterative allocation-failure cleanup | Maintained deterministic proof exists for selected repeated-run iterative handle owner and workspace growth failures. | `tests/test_iterative.c`, `tests/test_iterative_handle_helpers.h`, `make iterative-allocation-failure-gate`, `ctest -L allocation_failure`, Day 5-8 Sprint 176 artifacts. | CG, GMRES, and MINRES repeated-run handle prepare/growth paths only. |
| Invalid iterative prepare calls | Invalid prepare arguments do not publish private handle state and remain safe to clean up. | `test_iter_handle_invalid_prepare_calls_do_not_publish_state`; Day 6 artifact. | Public repeated-run iterative handle preparation only. |
| Public iterative cleanup invariant | `sparse_iter_handle_free()` is safe on NULL, zeroed, and already-freed handles. | `include/sparse_iterative.h`; focused and full Day 8 validation. | Iterative repeated-run public handle only, not every public object type. |
| QR corpus and selected comparison | Fixture-local QR rank/nullspace/minimum-norm evidence and selected comparison freshness exist. | `tests/test_qr_corpus.c`, selected oracle/comparison gates, Sprint 139/150/159/174/175 artifacts. | Named fixtures and selected hosted lanes only. No broad QR parity. |
| Partial-SVD corpus and selected comparison | Fixture-local partial-SVD clustered/repeated, rank-deficient, sparse-output, fail-closed, and selected diagonal comparison evidence exists. | `tests/test_svd_partial_corpus.c`, selected oracle/comparison gates, Sprint 140/151/159/174/175 artifacts. | Named fixtures only. No broad partial-SVD correctness or external-library parity. |
| Linked-list LU selected comparison | Fixture-local linked-list LU nonsymmetric square-solve comparison exists. | `tests/lu_external_dense_reference.py`, `make report-index-comparison-freshness`, Sprint 174/175 artifacts. | `lu_nonsym_square_5` only. No LU CSR or broad nonsymmetric solve claim. |
| Selected hosted report freshness | Linux hosts selected oracle and selected comparison freshness; macOS hosts selected comparison freshness. | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, selected workflow guard tests, Sprint 159/175 artifacts. | Selected rows/artifacts only. No broad report-index or Windows report freshness. |
| Selected hosted performance freshness | Linux hosts selected `bench_refactor_csc` threshold-free canonical row freshness. | `make bench-canonical-report-freshness`, hosted selected-performance lane, Sprint 168/169 artifacts. | Artifact/methodology freshness only. No timing superiority or portable performance claim. |
| Static-first package contract | Source install, Make install/`pkg-config` on Unix, CMake install/export, and Windows CMake downstream validation are maintained. | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`, CI package lanes, Sprint 170/171/162 artifacts. | Static archive package surface only. |
| Package-manager support | Not currently provided. | `scripts/package_manager_deferral_check.sh`, `INSTALL.md`, Sprint 171 artifacts. | Source install remains the maintained path until a provider-specific proof exists. |
| Shared-library and dynamic ABI | Deferred and explicitly rejected when requested through `BUILD_SHARED_LIBS=ON`. | Sprint 170 decision record, static package deferral guard, CMake configure rejection. | No shared library, dynamic ABI, runtime-loader, or symbol-visibility product claim. |
| API HTML | Local generated Doxygen view can be refreshed and checked. | `make api-docs-freshness`, `docs/api_reference.md`, Sprint 173 artifacts. | Local-only generated output. No hosted/source-controlled HTML publication claim. |

## Claim Classification

### Earned Claims

- The project has one maintained deterministic allocation-failure proof for
  selected iterative repeated-run handle prepare/growth paths.
- The public iterative handle cleanup invariant is documented in the header,
  README, and maintainer guide.
- Selected QR, partial-SVD, and linked-list LU comparison families have
  fixture-local evidence and selected freshness gates.
- Static-first install/export/package metadata is maintained and reviewed on
  the platform lanes named in install and maintainer docs.

### Local-Only Claims

- Generated benchmark, oracle, comparison, report-index, API HTML, coverage,
  and dead-code outputs are local unless a reviewed hosted lane explicitly
  promotes a selected subset.
- `make iterative-allocation-failure-gate` is currently a maintained local
  focused gate and CTest-label surface. It is not yet a separate hosted CI
  promotion claim.
- Benchmark timing rows are branch-local measurements and must keep their
  manifest/methodology context.

### Hosted-Only Claims

- Linux selected oracle/comparison freshness and selected performance
  freshness are hosted for the named rows/artifacts only.
- macOS selected comparison freshness is hosted for selected comparison
  artifacts only.
- Windows reviewed evidence remains CMake-first and package/downstream scoped.

### Advisory Or Supplemental Claims

- Benchmark exploratory rows, broad report-index rows, coverage, dead-code
  reports, and example outputs are useful context but not release or broad
  correctness proof by themselves.
- Environment/backend controls remain diagnostic and report-context surfaces,
  not new public typed API, ABI, package, or platform guarantees.

### Unsupported Or Explicit Non-Claims

- broad allocation-failure cleanup coverage across all solvers and allocation
  paths;
- broad state-of-the-art sparse linear algebra status;
- portable performance superiority;
- broad external-library parity;
- shared-library support, dynamic ABI compatibility, and runtime-loader
  behavior;
- package-manager provider availability;
- broad platform parity;
- Windows Makefile parity or Windows `pkg-config` command execution parity;
- Windows report freshness;
- hosted publication of all generated reports or generated API HTML;
- release evidence.

## Stale, Ambiguous, Or Overbroad Wording Candidates

| Surface | Candidate issue | Day 10 action |
| --- | --- | --- |
| `README.md` quality/capability summary | The README now documents repeated-run allocation-failure scope in the lifecycle section, but the top-level quality evidence list does not yet mention the new proof. | Add a narrow bullet only if it says "selected iterative repeated-run handle allocation-failure proof" and names the local gate. |
| `README.md` command list | `make iterative-allocation-failure-gate` is absent from the command map. | Consider adding it near focused report/quality commands, explicitly as a focused local gate. |
| `docs/maintainer_guide.md` allocation-failure owner | Already precise, but may need final closeout cross-reference to the Day 9/Day 10 claim inventory. | Add or adjust only if it improves discoverability without widening the claim. |
| `docs/api_reference.md` workflow guide | It does not mention the new iterative handle cleanup invariant. | Usually leave unchanged because headers own exact contract; add only if Day 10 needs a compact pointer. |
| `docs/tutorial.md` / `docs/cookbook.md` | No allocation-failure wording found. | Leave unchanged unless a repeated-run iterative workflow section is already being touched. |
| `benchmarks/README.md` | No allocation-failure wording found. | Leave unchanged; benchmarks are not proof owners for this lane. |
| `INSTALL.md` | No allocation-failure wording found. | Leave unchanged; install/package docs are unrelated proof surfaces. |
| Planning closeout | Day 1 retained non-claims need a Day 10 update after public wording edits. | Convert selected allocation-failure proof from a gap to an earned-but-bounded claim while keeping broad allocation-failure as a non-claim. |

## Day 10 Recalibration Checklist

1. Update only the claim surfaces that need the Sprint 176 proof discoverable
   to users or maintainers.
2. Keep the phrase family-local or equivalent beside every allocation-failure
   proof mention.
3. Name the exact focused gate:
   `make iterative-allocation-failure-gate`.
4. Name the exact family scope: CG, GMRES, and MINRES repeated-run handle
   prepare/growth paths.
5. Preserve the broad allocation-failure non-claim for direct solvers,
   eigensolvers, matrix construction, package/install flows, generated-report
   tooling, and unrelated allocation paths.
6. Do not cite benchmarks, examples, package lanes, report-index rows, or
   hosted performance/report lanes as allocation-failure proof.
7. If package, ABI, package-manager, platform, or generated-report wording is
   touched, run the corresponding existing guard rather than relying on this
   inventory.
8. After public documentation edits, run `git diff --check`; run broader gates
   only if affected surfaces require them.

## Validation

Day 9 changed planning artifacts only. No `.c` or `.h` files were modified for
this day, so the full C quality gate is not required.

Validation command:

```sh
git diff --check
```

Result: passed.
