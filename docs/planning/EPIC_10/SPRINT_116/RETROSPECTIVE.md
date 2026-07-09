# Sprint 116 Retrospective

**Sprint:** 116 - Adoption Surface Residual QA & Claim Guardrails
**Duration:** 14 days (Days 1-14 landed on branch `sprint-116`)
**Status:** Complete

## Definition of Done Checklist

- [x] Created Sprint 116 day-by-day plan, working notes, artifact directory,
      and adoption QA intake.
- [x] Inventoried adoption-facing documents:
  - `README.md`;
  - `INSTALL.md`;
  - `docs/tutorial.md`;
  - `docs/solver_selection.md`;
  - `docs/matrix_market.md`;
  - `docs/algorithm.md`;
  - `benchmarks/README.md`;
  - `examples/README.md`.
- [x] Validated external adoption references:
  - `https://math.nist.gov/MatrixMarket/formats.html`;
  - `https://sparse.tamu.edu/`.
- [x] Audited README quality, CI, install, support-tier, package/platform, ABI,
      package-manager, and benchmark wording.
- [x] Applied the README wording cleanup that replaced ambiguous
      "package-manager detail" with "install-support detail".
- [x] Audited benchmark documentation scanability, live target names, report
      mechanics, and performance-claim boundaries.
- [x] Added benchmark Quick Navigation without changing benchmark commands,
      report semantics, CI lanes, or performance claims.
- [x] Audited `docs/algorithm.md` positioning and added a top note that frames
      it as technical background rather than first-use adoption guidance,
      install/support contract, package/ABI reference, or portable performance
      guarantee.
- [x] Audited performance wording across README, solver-selection,
      benchmark, algorithm, and install docs.
- [x] Replaced broad ILU(0) "3-1000x speedup" wording with
      workload-dependent local-benchmark wording.
- [x] Completed adoption non-claims checklist and final claim-guardrail
      follow-through.
- [x] Confirmed no unsupported package/platform, performance, proof-owner,
      Matrix I/O module, builder API, ABI, shared-library, package-manager, or
      state-of-the-art claim remains in the audited adoption surface.
- [x] Documentation hygiene passed:
  - `git diff --check`;
  - trailing-whitespace scan across touched docs.
- [x] Full C quality gate was not required because Sprint 116 changed only
      documentation and planning artifacts and did not touch `.c` or `.h`
      files.

## What Went Well

1. **The sprint stayed adoption-focused.**
   Sprint 116 avoided absorbing implementation, package-manager recipe, ABI,
   source movement, helper-abstraction, package install lane, or platform
   parity work. That kept the output aligned with the project-plan goal:
   adoption QA and claim guardrails.

2. **External reference QA was concrete.**
   The sprint inventoried adoption-facing external links first, then validated
   the Matrix Market and SuiteSparse URLs with redirects followed. Both
   resolved cleanly, so no stale-link changes were needed.

3. **README support wording became tighter.**
   The README already carried most package/platform guardrails. The one
   ambiguous phrase, "package-manager detail", was replaced with
   "install-support detail" so users are not led to infer package-manager
   support.

4. **Benchmark documentation became easier to scan without changing semantics.**
   The Quick Navigation table improved discoverability while preserving all
   existing benchmark boundaries: threshold-free reports stay reports,
   `wall-check` remains the hard timing gate, and benchmark rows remain local
   measurement artifacts.

5. **Algorithm docs now have an explicit role.**
   `docs/algorithm.md` remains useful technical background, but it no longer
   relies on surrounding docs to define that role. The new positioning note
   prevents historical measurements and implementation detail from being read
   as first-use adoption guidance or support contracts.

6. **Performance wording cleanup was surgical.**
   The only broad performance phrase found during the audit was the ILU(0)
   `3-1000x speedup` table entry. Replacing it with workload-dependent local
   benchmark wording preserved the technical reference while removing a broad
   speed claim.

7. **Validation matched the touched surface.**
   Since the sprint only changed docs, validation used documentation hygiene
   checks and explicit no-code-change confirmation instead of unrelated C
   quality gates.

## What Didn't Go Well

1. **The adoption surface is still spread across many files.**
   Sprint 116 made claim boundaries clearer, but users still move between
   README, INSTALL, solver selection, examples, benchmark docs, Matrix Market
   docs, and algorithm docs for a complete picture.

2. **`docs/algorithm.md` remains large.**
   The top positioning note makes its role explicit, but the document still
   mixes current technical reference, historical measurements, and sprint-era
   evidence. A future split may be worthwhile.

3. **Benchmark report discoverability is improved but still manual.**
   The Quick Navigation table helps readers, but generated benchmark artifacts
   still do not have a richer generated index surfaced into public docs.

4. **Package/platform support remains intentionally narrow.**
   Sprint 116 clarified non-claims; it did not add Linux install CI,
   full macOS install/export parity, Windows install-validation parity,
   shared-library support, dynamic ABI guarantees, or package-manager support.

## Final Metrics

### Validation

| Metric | Sprint 116 close state |
|---|---:|
| documentation hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed |
| retired wording scan | passed |
| full C quality gate | not required; no `.c` or `.h` changes |
| changed source/header files | 0 |
| changed workflow files | 0 |
| changed Make/CMake/package metadata files | 0 |
| changed script files | 0 |
| package-manager recipes added | 0 |
| shared-library build rules added | 0 |
| dynamic ABI claims added | 0 |
| benchmark command/report semantic changes | 0 |

### Adoption QA Decisions

| Surface | Sprint 116 close decision |
|---|---|
| External references | Matrix Market and SuiteSparse URLs resolved with HTTP 200; no doc edits required. |
| README | Compact support wording preserved; package-manager implication removed. |
| Benchmark docs | Quick Navigation added; benchmark semantics unchanged. |
| Algorithm docs | Technical-background role made explicit. |
| Performance wording | Broad ILU(0) speedup phrase downgraded to local benchmark wording. |
| Matrix Market docs | Public load/save surface remains fenced from public module/builder claims. |
| Package/platform docs | Static-first install surface preserved; shared-library, ABI, package-manager, Windows install-validation, and full macOS install/export claims remain fenced. |
| Proof-owner/internal helpers | Not promoted into first-use adoption guidance. |

### Sprint 116 Artifact Package

| Metric | Sprint 116 close state |
|---|---:|
| artifact files under `SPRINT_116/artifacts/` | 14 |
| artifact lines before retrospective | 1051 |
| working notes lines before retrospective | 398 |
| plan lines | 436 |
| retrospective files | 1 |

Notes:

- intake and external-reference artifacts:
  - `day1-adoption-qa-intake.md`
  - `day2-external-reference-inventory.md`
  - `day3-external-reference-qa.md`
- README, benchmark, and algorithm artifacts:
  - `day4-readme-boundary-audit.md`
  - `day5-readme-follow-through.md`
  - `day6-benchmark-scanability-audit.md`
  - `day7-benchmark-follow-through.md`
  - `day8-algorithm-positioning-audit.md`
  - `day9-algorithm-follow-through.md`
- performance, non-claims, and closeout artifacts:
  - `day10-performance-wording-audit.md`
  - `day11-performance-wording-follow-through.md`
  - `day12-adoption-non-claims-checklist.md`
  - `day13-claim-guardrail-follow-through.md`
  - `day14-validation-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- Consider splitting `docs/algorithm.md` into a concise public algorithm
  reference plus a historical measurement appendix if scanability remains a
  recurring concern.
- Consider generated benchmark artifact indexes in a future benchmark sprint
  if report discoverability needs to improve.

Still consciously constrained rather than silently solved:

- no reviewed Linux install CI lane;
- no full reviewed macOS CMake install/export parity;
- no Windows install-validation parity;
- no Windows thread/fuzz/property parity;
- no Windows Makefile parity;
- no macOS coverage reviewed-lane claim;
- no Homebrew GCC reviewed-lane promotion;
- no shared-library package support;
- no dynamic ABI compatibility guarantee;
- no package-manager support;
- no public Matrix I/O module or public builder API;
- no proof-owner/internal-helper public contract expansion;
- no source-list, helper-target, CTest membership, or implementation change.

Not carried forward as unresolved Sprint 116 debt:

- adoption QA intake and scope fence;
- external-reference inventory;
- external-reference network QA;
- README boundary audit and wording follow-through;
- benchmark scanability audit and Quick Navigation follow-through;
- algorithm positioning audit and top-note follow-through;
- performance wording evidence audit and ILU(0) wording cleanup;
- adoption non-claims checklist;
- final claim-guardrail follow-through;
- Sprint 116 validation and handoff.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-adoption-qa-intake.md](./artifacts/day1-adoption-qa-intake.md)
- [day2-external-reference-inventory.md](./artifacts/day2-external-reference-inventory.md)
- [day3-external-reference-qa.md](./artifacts/day3-external-reference-qa.md)
- [day4-readme-boundary-audit.md](./artifacts/day4-readme-boundary-audit.md)
- [day5-readme-follow-through.md](./artifacts/day5-readme-follow-through.md)
- [day6-benchmark-scanability-audit.md](./artifacts/day6-benchmark-scanability-audit.md)
- [day7-benchmark-follow-through.md](./artifacts/day7-benchmark-follow-through.md)
- [day8-algorithm-positioning-audit.md](./artifacts/day8-algorithm-positioning-audit.md)
- [day9-algorithm-follow-through.md](./artifacts/day9-algorithm-follow-through.md)
- [day10-performance-wording-audit.md](./artifacts/day10-performance-wording-audit.md)
- [day11-performance-wording-follow-through.md](./artifacts/day11-performance-wording-follow-through.md)
- [day12-adoption-non-claims-checklist.md](./artifacts/day12-adoption-non-claims-checklist.md)
- [day13-claim-guardrail-follow-through.md](./artifacts/day13-claim-guardrail-follow-through.md)
- [day14-validation-handoff.md](./artifacts/day14-validation-handoff.md)
