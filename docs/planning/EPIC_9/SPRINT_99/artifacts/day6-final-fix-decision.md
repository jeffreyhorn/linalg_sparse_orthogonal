# Sprint 99 Day 6: Final Fix Decision

## Purpose

Day 6 decides whether Sprint 99 needs a last bounded implementation or support
fix batch before Epic 9 closeout. The decision is based on:

- Day 2 contradiction-class re-audit
- Day 4 correctness/runtime evidence
- Day 5 package/usability/workflow evidence

The decision must not promote broad residual work, speculative polish, or
future comparison architecture into a final closeout fix.

## Evidence Reviewed

### Day 2 Candidate Queue

Day 2 did not find a clear implementation blocker. It listed only conditional
candidate classes:

| Candidate class | Trigger required | Day 6 status |
|---|---|---|
| stale public or maintainer wording | Day 3-5 evidence finds live overclaim or contradiction | not triggered |
| workflow/test-count drift | final CTest or workflow-surface check disagrees with documented counts/scope | not triggered |
| install/export proof drift | package proof commands fail or contradict static-first docs | not triggered |
| benchmark/reporting claim drift | selected benchmark/report output contradicts docs or guardrails | not triggered |
| residual queue ambiguity | unresolved items are duplicated, stale, or phrased as accidental claims | carry forward to Day 9 residual finalization |

### Day 4 Evidence

Day 4 produced no final-fix candidates.

Closeout-ready lanes:

- LDLT helper fixtures
- Cholesky CSC external correctness
- LDLT CSC external correctness
- reorder/fill calibration
- canonical benchmark report generation

Residual/non-claims:

- broader solver-family external comparison
- broader LDLT corpus comparison
- portable timing thresholds
- universal reorder/fill superiority
- generated reorder/fill report target

### Day 5 Evidence

Day 5 produced no final-fix candidates.

Closeout-ready lanes:

- Make install/export proof
- CMake install/export and consumer proof
- public and maintainer docs stale-claim scan
- benchmark documentation boundaries
- workflow/platform scope wording
- Windows expected CTest count

Residual/non-claims:

- shared-library package maturity
- dynamic ABI guarantee
- Windows install-validation lane
- Windows Makefile parity
- symmetric platform parity
- portable benchmark/timing claims

## Decision

No final bounded fix batch is selected for Sprint 99.

## Rationale

The evidence package supports closeout without a new implementation/support
batch:

- selected correctness lanes passed
- selected runtime/fill and benchmark reporting lanes passed
- selected package/install/export lanes passed
- stale-claim scans found only negative guardrails, not positive overclaims
- workflow/platform assertions match the current reviewed proof model
- Windows expected CTest count remains consistent with CMake registrations and
  staged exclusions

The unresolved Day 2 classes are real, but they are not blockers:

- large source and proof owners remain active residuals
- broader comparison lanes need architecture before implementation
- broad complex/mixed-precision, shared-library, platform parity, and portable
  timing claims remain deliberate non-claims
- lower-level chronology cleanup remains useful but not closeout-blocking

Starting a Day 7-8 implementation batch would therefore create scope without a
live closeout contradiction.

## Rejected Candidates

| Candidate | Decision | Reason |
|---|---|---|
| broader external solver comparison | reject for Sprint 99 fix batch | residual architecture work, not a broken closeout claim |
| broader LDLT Matrix Market corpus | reject for Sprint 99 fix batch | useful future comparison depth, not required for named KKT lane |
| generated reorder/fill report target | reject for Sprint 99 fix batch | repeated need not yet proven; existing command passes |
| source/test extraction | reject for Sprint 99 fix batch | active residual, but not required to close bounded Epic 9 |
| lower-level chronology cleanup | reject for Sprint 99 fix batch | partially resolved; remaining instances are not public overclaims |
| shared-library packaging | reject as non-claim | static-first package proof passed and scripts assert no shared artifacts |
| Windows install-validation or Makefile parity | reject as non-claim | Windows reviewed scope remains CMake-first subset |
| portable timing thresholds | reject as non-claim | Day 4 evidence is local calibration only |

## Day 7-8 Boundary

Because no final fix batch is selected:

- Day 7 should record no-op implementation evidence and confirm there is no
  new blocker.
- Day 8 should record no-op fix closeout and reconcile residual/final
  validation readiness.
- No source, header, script, build-system, workflow, benchmark, or public-doc
  edit should occur on Days 7-8 unless new evidence appears and this Day 6
  decision is explicitly reopened.

If the decision is reopened, the required boundary must name:

- exact contradiction
- exact touched files
- exact validation commands
- rollback plan
- claim text affected
- why residual/non-claim classification is insufficient

## Validation and Rollback Checklist

### If No Files Change Beyond Planning Artifacts

Run:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_9/SPRINT_99
```

### If Public or Maintainer Docs Change

Run:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_9/SPRINT_99 README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md
rg -n "best-in-class|benchmark supremacy|full platform parity|shared-library-first|dynamic ABI|broad complex|mixed-precision|portable timing|universal .*superiority|all solver" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md include .github/workflows
```

### If Build, CMake, Install, or Workflow Files Change

Run the focused local equivalent plus:

```sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
make quality-review-cmake-compile
```

CI remains the final proof for platform-specific workflow syntax.

### If `.c` or `.h` Files Change

Run:

```sh
make format && make lint && make test
```

Then run focused commands for the touched family.

## Day 6 Conclusion

Sprint 99 should proceed without a final implementation/support fix batch.
Days 7-8 should document that no-op path and preserve the evidence-backed
claim boundaries. Day 9 should finalize the residual queue from the rejected
candidates and deliberate non-claims above.
