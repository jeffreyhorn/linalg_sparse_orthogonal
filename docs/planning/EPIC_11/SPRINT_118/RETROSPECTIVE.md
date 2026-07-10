# Sprint 118 Retrospective

**Sprint:** 118 - Epic 11 Baseline, Residual Conversion & Product Truth Freeze
**Duration:** 14 days
**Status:** Complete

## Definition of Done Checklist

- [x] Created Sprint 118 day-by-day plan, working notes, artifact directory,
      and template directory.
- [x] Inventoried Sprint 118 inputs from Epic 10 closeout, Sprint 117
      artifacts, the Epic 11 review, the Epic 11 todo, and the Epic 11 project
      plan.
- [x] Built the reviewed and supplemental validation-surface inventory.
- [x] Ran documentation hygiene and the strongest local reviewed baseline:
      `make quality-review-full`.
- [x] Froze CI-tier, platform, install, package, and support-boundary truth.
- [x] Converted the Epic 10 residual queue into Epic 11 owners, dependencies,
      proof gates, and future-epic/non-claim dispositions.
- [x] Designed and completed the current product truth map.
- [x] Recorded baseline claims, Epic 11 candidate claims, explicit non-claims,
      and evidence cross-references.
- [x] Collected current source/test hotspot metrics.
- [x] Interpreted hotspot metrics into Sprint 119-123 owner handoff guidance.
- [x] Designed and published refreshed evidence templates for source movement,
      oracle expansion, performance sentinels, package/ABI decisions, and
      adoption cleanup.
- [x] Audited public/support claims against the Day 8 truth map.
- [x] Published the Sprint 119-127 closeout and handoff package.
- [x] Created this retrospective.
- [x] Ran focused documentation hygiene after finalizing the retrospective.

## What Went Well

1. **Sprint 118 established a clean Epic 11 baseline before implementation.**
   The sprint converted Epic 10 closeout evidence into concrete Sprint 119-127
   owners without pulling implementation work into the baseline sprint.

2. **Validation evidence was stronger than the touched-surface minimum.**
   Sprint 118 changed planning documentation only, but Day 3 still ran
   `make quality-review-full`. The Makefile reviewed path, CMake reviewed
   parity path, CTest registration parity, and full CTest run all passed.

3. **Product truth was frozen before future claims could drift.**
   Day 8 separated baseline truth, candidate Epic 11 claims, and explicit
   non-claims across compressed-first workflows, mutable-shell compatibility,
   solver families, Matrix Market, graph/reorder, package/platform,
   benchmarks, validation, adoption, and public claim boundaries.

4. **Residual debt is now dependency ordered.**
   Day 6 assigned residuals to Sprints 119-127 and documented dependencies:
   source-boundary proof before oracle/test splits; oracle and corpus decisions
   before report/performance interpretation; package/ABI decisions before
   platform install parity; and platform/package truth before adoption wording.

5. **Hotspot work was evidence-ranked instead of broadly mandated.**
   Day 9 collected current metrics, and Day 10 separated high-risk owners from
   large-but-coherent owners. Future source movement and giant-test splits now
   have prerequisites instead of a vague refactor instruction.

6. **Future implementation sprints received reusable evidence templates.**
   Day 12 published templates for source movement, oracle expansion,
   performance sentinels, package/ABI decisions, adoption cleanup, and usage
   notes. Each template preserves proof, validation, drift, non-claim, and
   handoff fields.

7. **The public-claim audit found no immediate unsupported public wording.**
   Day 13 found that public/support surfaces already preserve key fences
   around static-first packaging, tiered platforms, benchmark locality,
   bounded Matrix Market support, selected solver evidence, and broad
   ecosystem/state-of-the-art non-claims.

## What Did Not Go Well

1. **The baseline sprint produced a large documentation package.**
   The artifact set is useful, but there is still a lot for Sprint 119-127
   owners to read. The Day 12 templates and Day 14 handoff mitigate this by
   giving each future sprint a shorter starting path.

2. **The README product identity remains only partially modernized.**
   Public docs support compressed-first workflows, but the README still opens
   with the orthogonal linked-list identity. This is not incorrect because the
   mutable shell remains supported, but Sprint 126 should make the
   compressed-first product story clearer.

3. **No fresh supplemental package, benchmark, sanitizer, coverage, or
   platform workflow lanes were run after Day 3.**
   This was correct for a documentation-only sprint, but future sprints that
   touch those surfaces must regenerate the relevant evidence rather than cite
   Sprint 118 as fresh proof.

4. **The hotspot metrics confirm the same maintainability pressure rather than
   eliminating it.**
   `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_iterative.c`,
   `tests/test_svd.c`, `src/sparse_ldlt_csc.c`, `src/sparse_iterative.c`,
   and `src/sparse_eigs.c` remain large owners. Sprint 118 ranked them and
   gave proof gates, but did not reduce them.

## Final Metrics

### Validation

| Metric | Sprint 118 close state |
|---|---:|
| documentation hygiene | `git diff --check` passed on documentation-only days |
| trailing-whitespace scan | passed over `docs/planning/EPIC_11/SPRINT_118` |
| strongest local reviewed baseline | `make quality-review-full` passed on Day 3 |
| Makefile reviewed path | passed: `format-check`, `lint`, `test`, `deadcode-check` |
| CMake reviewed parity path | passed: configure, clean build, `ctest -N`, count parity, full CTest |
| CMake registered tests | `54` |
| Makefile/CMake test-count parity | `54` vs `54` |
| CTest result | `54 / 54` passed |
| CTest failures | `0` |
| CTest real time | `208.17 sec` |
| changed `.c` files | `0` |
| changed `.h` files | `0` |
| changed Make/CMake/workflow/package/script files | `0` |
| changed benchmark/source/test/include files | `0` |

### Sprint Artifact Package

| Metric | Sprint 118 close state |
|---|---:|
| artifact files under `SPRINT_118/artifacts/` | `14` |
| refreshed template files under `SPRINT_118/templates/` | `6` |
| plan files | `1` |
| working notes files | `1` |
| retrospective files | `1` |

### Hotspot Metrics Captured

| Metric | Sprint 118 observed value |
|---|---:|
| files across `src include tests benchmarks examples docs` | `2435` |
| `src` files | `68` |
| `include` files | `19` |
| `tests` files | `89` |
| `benchmarks` files | `19` |
| `examples` files | `18` |
| `docs` files | `2222` |
| C source files across measured surfaces | `134` |
| headers across measured surfaces | `49` |
| Markdown files across measured surfaces | `1693` |

Largest owners recorded:

| Owner | Lines |
|---|---:|
| `tests/test_ldlt_csc.c` | `3915` |
| `tests/test_integration.c` | `3279` |
| `tests/test_qr.c` | `3234` |
| `tests/test_ldlt.c` | `3006` |
| `tests/test_etree.c` | `2962` |
| `tests/test_iterative.c` | `2924` |
| `tests/test_svd.c` | `2823` |
| `src/sparse_ldlt_csc.c` | `2095` |
| `src/sparse_lu_csr.c` | `1594` |
| `src/sparse_iterative.c` | `1495` |
| `src/sparse_eigs.c` | `1412` |

## Claim And Product-Truth Outcomes

| Area | Outcome |
|---|---|
| Product identity | Compressed-first workflows are supported and preferred for CSR/CSC inputs; mutable shell remains supported compatibility. |
| Direct solvers | LU, Cholesky, LDLT, QR, CSR LU, CSC Cholesky/LDLT, one-shot, and selected repeated lifecycle support are baseline truth with bounded proof breadth. |
| Iterative solvers | CG, GMRES, MINRES, BiCGSTAB, block variants, preconditioners, diagnostics, and selected repeated handles remain supported within documented limits. |
| Eigensolvers | Symmetric eigensolver workflows are supported; source-boundary and external-comparison breadth remain future owner work. |
| SVD/QR/rank | Supported with current regression evidence; broad LAPACK/SciPy parity remains unclaimed. |
| Matrix Market | Load/save support is bounded to documented coordinate variants and unsupported-feature boundaries. |
| Graph/reorder | RCM, AMD, ND, COLAMD-style surfaces, graph partition helpers, and typed options remain supported with bounded caveats. |
| Package/platform | Static-first package support and tiered platform support remain current truth. |
| Benchmarks/performance | Benchmark, report, and local sentinel surfaces remain local measurement context, not portable performance proof. |
| Adoption/docs | Public routes exist and are honest, but scanability and compressed-first identity remain Sprint 126 work. |

## Residual Deferred Debt

Carry-forward work for owner sprints:

- Sprint 119 should perform eigensolver movement feasibility before moving any
  private owner.
- Sprint 119 should move or explicitly defer `s20_select_indices`,
  `s20_lift_ritz_vectors`, shift-invert setup/conversion, and
  `lanczos_iterate_op` only with focused consumer proof.
- Sprint 120 should build direct/iterative oracle ownership and split giant
  tests only where failure localization and CTest membership are preserved.
- Sprint 121 should expand SVD/QR/rank proof helpers and dense/external
  reference evidence without claiming LAPACK/SciPy parity.
- Sprint 122 should create corpus taxonomy, report indexes, and coverage
  interpretation based on risk rather than vanity percentages.
- Sprint 123 should strengthen local performance/backend governance without
  portable speed, vendor-backend, or universal reorder/fill claims.
- Sprint 124 should decide static-first continuation versus shared-library/ABI
  productization and publish package-manager disposition.
- Sprint 125 should handle Linux/macOS/Windows install/export and staged
  platform follow-through only with reviewed/supplemental classification and
  expected-count updates.
- Sprint 126 should improve compressed-first adoption identity, algorithm-doc
  scanability, cookbook examples, links, and public wording.
- Sprint 127 should perform final claim recalibration, strongest reviewed
  validation, earned/unearned claim publication, and residual publication.

Still consciously constrained rather than silently solved:

- no broad state-of-the-art replacement claim;
- no broad ecosystem parity claim;
- no every-family external solver validation claim;
- no portable performance superiority claim;
- no universal reorder/fill superiority claim;
- no shared-library package support or dynamic ABI guarantee;
- no package-manager support;
- no symmetric Linux/macOS/Windows reviewed parity;
- no Windows Makefile, install-validation, thread/fuzz/property, or full CTest
  parity;
- no GPU support;
- no distributed-memory support;
- no broad complex or mixed-precision maturity.

Not carried forward as unresolved Sprint 118 debt:

- baseline validation package;
- CI-tier and platform truth freeze;
- residual queue intake and owner mapping;
- product truth map;
- source/test hotspot metric collection;
- hotspot owner handoff;
- evidence template design and publication;
- public claim drift audit;
- Sprint 119-127 closeout handoff.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-validation-inventory.md](./artifacts/day2-validation-inventory.md)
- [day3-baseline-quality-recheck.md](./artifacts/day3-baseline-quality-recheck.md)
- [day4-ci-tier-platform-truth.md](./artifacts/day4-ci-tier-platform-truth.md)
- [day5-residual-intake.md](./artifacts/day5-residual-intake.md)
- [day6-residual-owner-map.md](./artifacts/day6-residual-owner-map.md)
- [day7-product-truth-map-design.md](./artifacts/day7-product-truth-map-design.md)
- [day8-product-truth-map.md](./artifacts/day8-product-truth-map.md)
- [day9-hotspot-metrics.md](./artifacts/day9-hotspot-metrics.md)
- [day10-hotspot-owner-handoff.md](./artifacts/day10-hotspot-owner-handoff.md)
- [day11-evidence-template-design.md](./artifacts/day11-evidence-template-design.md)
- [day12-evidence-template-refresh.md](./artifacts/day12-evidence-template-refresh.md)
- [day13-public-claim-drift-audit.md](./artifacts/day13-public-claim-drift-audit.md)
- [day14-sprint-closeout-handoff.md](./artifacts/day14-sprint-closeout-handoff.md)
- [source-movement-evidence-template.md](./templates/source-movement-evidence-template.md)
- [oracle-expansion-evidence-template.md](./templates/oracle-expansion-evidence-template.md)
- [performance-sentinel-evidence-template.md](./templates/performance-sentinel-evidence-template.md)
- [package-abi-decision-template.md](./templates/package-abi-decision-template.md)
- [adoption-cleanup-evidence-template.md](./templates/adoption-cleanup-evidence-template.md)
- [template-usage-notes.md](./templates/template-usage-notes.md)
