# Sprint 98 Working Notes

## Day 1 - Sprint 98 Scope & Assurance Baseline

### Goal

Open Sprint 98 with a current assurance, external comparison, runtime/fill,
coverage, documentation, and workflow-proof baseline from the merged Sprint 97
tree. Day 1 does not widen comparison proof. It identifies the surfaces and
validation rules that Day 2 can audit and rank.

### Actions

- Re-read the Sprint 98 section of
  `docs/planning/EPIC_9/PROJECT_PLAN.md`.
- Re-read the Sprint 98 plan Day 1 scope.
- Re-read prior Sprint 90, Sprint 94, Sprint 96, and Sprint 97 planning
  artifacts that define:
  - comparison and measurement contract
  - capability-surface baseline
  - proof-owner and maintainability closeout
  - build/package/product convergence closeout
- Inventoried current assurance and comparison topology:
  - maintained external correctness tests
  - benchmark and runtime reporting surfaces
  - fill and structural comparison artifacts
  - coverage and proof-owner topology
  - CI workflow lanes
  - local reviewed validation gates
  - public and maintainer documentation claim surfaces
- Separated the live surfaces into:
  - correctness evidence
  - runtime/fill evidence
  - coverage/proof-owner topology
  - documentation claims
  - workflow ownership
- Recorded authoritative inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Day 1 assurance baseline in
  `artifacts/day1-assurance-baseline.md`.

### Findings

- The maintained external correctness lane remains bounded to Cholesky CSC SPD
  solves checked against an external dense reference on `nos4` and `bcsstk04`.
- `tests/test_ldlt_csc.c`, `tests/test_iterative.c`, the eigensolver tests,
  QR, SVD, and ordering/fill tests all contain candidate evidence, but none is
  yet a widened maintained external-differential lane.
- Runtime/fill evidence remains threshold-free and calibration-oriented:
  `make bench-canonical-report`, `make bench-reorder-sprint86`, and
  `make bench-fast` should not be reinterpreted as broad product superiority
  proof.
- Coverage remains supplemental and tree-mutating. Returning from coverage
  modes requires `make clean` before normal reviewed validation paths.
- Workflow proof remains intentionally asymmetric: Linux is strongest, macOS
  is narrower with supplemental confidence paths, and Windows is a reviewed
  CMake-first subset with staged exclusions.
- Day 2 should rank external correctness and runtime/fill lanes by
  maintainability, deterministic reference availability, CI suitability, and
  risk of claim drift.

### Validation

- Day 1 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only baseline pass.
- Follow-up hygiene checks should include `git diff --check` and a
  trailing-whitespace scan on Sprint 98 planning files.

### Day 1 Exit State

Sprint 98 now has a current assurance baseline, authoritative inputs, a
starting comparison-surface candidate queue, and validation expectations for
later docs, scripts, benchmark, test, workflow, and code-touch days. Day 2 can
rerank comparison lanes from current evidence instead of treating all existing
proof, benchmark, and coverage signals as equally claim-bearing.

## Day 2 - Comparison-Surface Rerank

### Goal

Rank the highest-value next correctness, runtime, and fill comparison lanes so
Sprint 98 can design one bounded assurance expansion before implementation.

### Actions

- Re-read the Day 1 assurance baseline and Day 2 plan.
- Inspected existing external correctness and candidate proof surfaces:
  - `tests/test_chol_csc.c`
  - `tests/chol_external_dense_reference.py`
  - `tests/test_ldlt_csc.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_eigs_lobpcg.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_colamd.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_reorder_amd_qg.c`
- Inspected benchmark and runtime/fill reporting surfaces:
  - `Makefile`
  - `benchmarks/README.md`
  - `scripts/bench_canonical_report.sh`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
  - `benchmarks/bench_fillin.c`
  - canonical maintained benchmark sources
- Ranked correctness candidates by user-visible value, reference availability,
  deterministic reproducibility, maintenance cost, CI suitability, and claim
  drift risk.
- Ranked runtime/fill candidates by workload relevance, existing ownership,
  reporting clarity, artifact cost, and risk of misleading timing claims.
- Recorded the ranking in
  `artifacts/day2-comparison-surface-rerank.md`.

### Findings

- LDLT CSC is the strongest first correctness-expansion candidate because it
  is adjacent to the current Cholesky CSC external proof lane and already has
  rich scalar/native/supernodal residual and factor-state checks.
- Iterative solver comparison remains a strong fallback, but convergence and
  preconditioner semantics make it riskier as the first maintained external
  expansion.
- Eigensolver, QR, and SVD comparison lanes have real value but need tighter
  tolerance, fixture, and runtime design before they should carry maintained
  external-proof status.
- Reorder/fill is the strongest runtime/fill candidate because existing
  benchmark and test surfaces already expose bounded fill and timing evidence.
- Canonical benchmark reporting should be preserved as threshold-free local
  calibration; any expansion must not turn it into portable performance
  governance.

### Validation

- Day 2 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only rerank pass.
- Follow-up hygiene checks should include `git diff --check` and a
  trailing-whitespace scan on Sprint 98 planning files.

### Day 2 Exit State

Sprint 98 now has one authoritative comparison ranking. Day 3 should design a
bounded proof/comparison architecture around LDLT CSC external correctness and
reorder/fill runtime evidence, with iterative solver comparison and canonical
report metadata as fallback/support candidates.

## Day 3 - Proof/Comparison Architecture Design

### Goal

Define the bounded architecture for differential proof, runtime/fill
comparison, and proof-owner topology cleanup before any comparison lane is
widened.

### Actions

- Re-read the Day 2 comparison-surface rerank and Day 3 plan.
- Re-inspected LDLT CSC proof seams in `tests/test_ldlt_csc.c`:
  - `ldlt_csc_from_sparse_with_analysis`
  - deterministic KKT fixtures
  - `s20_solve_residual`
  - scalar/native/supernodal factor-state comparisons
  - row-adjacency reference checks
- Re-inspected the existing Cholesky CSC external helper model in
  `tests/chol_external_dense_reference.py` and the harness calls in
  `tests/test_chol_csc.c`.
- Re-inspected reorder/fill benchmark ownership:
  - `make bench-reorder-sprint86`
  - `bench_reorder --sprint86-slice --skip-factor`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
  - canonical benchmark report boundaries
- Defined ownership boundaries for:
  - external correctness helpers
  - C harness assertions
  - deterministic fixtures
  - benchmark output
  - planning artifacts
  - benchmark and maintainer docs
  - workflow assertions
- Recorded the architecture in
  `artifacts/day3-proof-comparison-architecture-design.md`.

### Findings

- LDLT CSC remains the selected correctness expansion, but Day 4 must still
  freeze whether the first maintained fixture is deterministic KKT,
  SPD-as-LDLT fallback, or a small Matrix Market fixture.
- The first LDLT external lane should prefer user-visible solve agreement
  under explicit tolerance rather than implementation-specific factor or pivot
  signatures.
- A helper that has to mirror Bunch-Kaufman pivot internals too closely is too
  coupled for the first external lane.
- Reorder/fill remains the selected runtime/fill lane. It should prioritize
  fill and structural context first, with runtime values treated as
  branch-local calibration.
- Day 3 requires no workflow change. Later workflow changes must label any
  widened lane as reviewed, supplemental, or artifact-only and must avoid
  cross-platform parity language.

### Validation

- Day 3 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only architecture pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 98 planning files.

### Day 3 Exit State

Sprint 98 now has a bounded proof/comparison architecture. Day 4 should freeze
the LDLT CSC correctness boundary before implementation, including fixture
class, helper shape, harness insertion point, tolerances, skip behavior, and
validation commands.

## Day 4 - External Correctness Boundary Freeze

### Goal

Freeze the highest-value maintained external correctness expansion before
implementation so Day 5 can land the LDLT CSC lane without widening scope.

### Actions

- Re-read the Day 3 proof/comparison architecture and Day 4 plan.
- Re-inspected the existing deterministic LDLT CSC KKT fixtures:
  - `build_kkt_5x5`
  - `build_kkt_10x10`
- Re-inspected the existing analysis-aware LDLT CSC path:
  - `s20_two_pass_indefinite_factor`
  - `s20_solve_residual`
  - `ldlt_csc_from_sparse_with_analysis`
  - `ldlt_csc_eliminate_supernodal`
- Re-inspected the existing Cholesky external dense helper contract and skip
  behavior in `tests/test_chol_csc.c` and
  `tests/chol_external_dense_reference.py`.
- Froze the first LDLT external lane around deterministic KKT fixtures rather
  than SPD fallback, random fixtures, or broader Matrix Market corpus entries.
- Froze the new helper shape as `tests/ldlt_external_dense_reference.py`,
  keyed by fixture names `kkt5` and `kkt10`.
- Froze the C harness insertion point and validation commands.
- Recorded the boundary in
  `artifacts/day4-external-correctness-boundary-freeze.md`.

### Findings

- The first external LDLT CSC lane should use deterministic KKT fixtures:
  `kkt5` as the smoke fixture and `kkt10` as the stronger off-block coupling
  fixture.
- The helper should solve dense systems from fixture keys with a small
  deterministic dense solver. It must not mirror Bunch-Kaufman pivot internals
  or project CSC storage.
- The C harness should assert user-visible solve agreement and residual
  strength, not external factor entries, pivot arrays, permutation arrays, or
  CSC structure.
- The initial tolerance is `1e-10` for max solution difference and final
  residual. If implementation evidence requires relaxing above `1e-9`, stop
  and record the reason before proceeding.
- Because Day 5 will modify `tests/test_ldlt_csc.c`, the required validation
  after implementation is `make format && make lint && make test`.

### Validation

- Day 4 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, or test files were
  modified.
- Full `make format && make lint && make test` is not required for this
  docs-only boundary pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 98 planning files.

### Day 4 Exit State

Sprint 98 now has a frozen external correctness boundary. Day 5 should add the
LDLT external dense-reference helper, add the `kkt5` and `kkt10` C harness
tests, run focused helper and `test_ldlt_csc` commands, then run
`make format && make lint && make test` because the implementation will touch a
C test file.

## Day 5 - External Correctness Expansion Batch 1

### Goal

Land the first maintained LDLT CSC external correctness comparison path from
the frozen Day 4 boundary.

### Actions

- Added `tests/ldlt_external_dense_reference.py`.
- Implemented deterministic dense KKT fixture construction in the helper:
  - `kkt5`
  - `kkt10`
- Implemented a small partial-pivoting dense solve in the helper with no
  project C-code dependency.
- Added LDLT external-reference reader and assertion helpers to
  `tests/test_ldlt_csc.c`.
- Added C harness tests:
  - `test_s98_external_dense_reference_kkt_5x5`
  - `test_s98_external_dense_reference_kkt_10x10`
- Registered the new tests near the existing Sprint 20 analysis-aware LDLT CSC
  KKT tests.
- Ran focused helper and LDLT CSC validation.
- Ran the required full validation chain because a C test file changed.
- Recorded implementation notes in
  `artifacts/day5-external-correctness-expansion-batch1.md`.

### Findings

- The external helper emits the expected original-order reference solution for
  `kkt5` exactly.
- The external helper emits the expected original-order reference solution for
  `kkt10` with only round-off-level deviation from integer values.
- The C harness solves the internally pre-permuted LDLT CSC system, maps the
  result back to original fixture order, and compares against the external
  helper without exposing pivot or CSC layout details as the comparison oracle.
- Focused `test_ldlt_csc` observed:
  - `kkt5`: `max|x-x_ref| = 0.000e+00`, `rel_residual = 0.000e+00`
  - `kkt10`: `max|x-x_ref| = 3.553e-15`, `rel_residual = 2.292e-16`
- The implementation preserves the Day 4 claim fence: this is bounded
  deterministic KKT solve evidence, not broad indefinite LDLT ecosystem proof.

### Validation

- Focused helper checks:
  - `python3 tests/ldlt_external_dense_reference.py kkt5`
  - `python3 tests/ldlt_external_dense_reference.py kkt10`
  - `python3 tests/ldlt_external_dense_reference.py nope`
- Focused LDLT CSC check:
  - `make build/test_ldlt_csc && ./build/test_ldlt_csc`
- Required full validation:
  - `make format && make lint && make test`
- Full validation passed and ended with:
  - `All tests passed.`

### Day 5 Exit State

Sprint 98 now has the first working external correctness expansion batch.
Day 6 should tighten any naming/comment/doc ownership needed for the landed
LDLT CSC external lane, then rerun the targeted proof commands and document the
closeout.

## Day 6 - External Correctness Expansion Batch 2

### Goal

Complete the LDLT CSC external correctness lane by reconciling maintained proof
ownership and rerunning the targeted proof commands from the frozen Day 4
boundary.

### Actions

- Re-read the Day 4 boundary and Day 5 implementation notes.
- Kept the Day 5 implementation unchanged because the fixture names, tolerance,
  and failure messages were already aligned with the frozen boundary.
- Updated `docs/maintainer_guide.md` to name the LDLT CSC external dense
  reference lane and its proof-owner boundary.
- Recorded the closeout in
  `artifacts/day6-correctness-expansion-closeout.md`.
- Reran the focused helper and LDLT CSC proof commands.

### Findings

- The maintained LDLT CSC proof owner is now explicit:
  `tests/test_ldlt_csc.c` plus `tests/ldlt_external_dense_reference.py`.
- The lane remains deterministic and fixture-keyed on `kkt5` and `kkt10`.
- Benchmarks, examples, factor entries, pivot arrays, permutations, and CSC
  layout details are not oracle owners for this lane.
- The Day 6 maintainer-guide update prevents the new correctness evidence from
  being read as broad indefinite ecosystem parity.

### Validation

- Focused helper checks:
  - `python3 tests/ldlt_external_dense_reference.py kkt5`
  - `python3 tests/ldlt_external_dense_reference.py kkt10`
- Focused LDLT CSC check:
  - `make build/test_ldlt_csc && ./build/test_ldlt_csc`
- Hygiene:
  - `git diff --check`
  - trailing-whitespace scan on touched Sprint 98 docs, the maintainer guide,
    and the Day 5 LDLT files

Day 6 changed documentation only. The required post-code-change full validation
chain was already run after the Day 5 C test change and passed.

### Day 6 Exit State

Sprint 98 now has a completed first external correctness expansion. Day 7 can
move to the runtime/fill comparison boundary without reopening LDLT CSC
correctness ownership unless focused validation regresses.

## Day 7 - Runtime/Fill Comparison Boundary Freeze

### Goal

Freeze the Sprint 98 runtime/fill comparison workload, metric contract, artifact
shape, and validation commands before implementation.

### Actions

- Re-read the Day 2 runtime/fill ranking and Day 3 proof/comparison
  architecture.
- Inspected the existing benchmark ownership surfaces:
  - `Makefile`
  - `benchmarks/README.md`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
  - `docs/maintainer_guide.md`
- Selected `make bench-reorder-sprint86` as the Sprint 98 runtime/fill
  workload.
- Ran the focused benchmark command to verify the selected surface still emits
  the expected bounded CSV rows.
- Recorded the boundary in
  `artifacts/day7-runtime-fill-boundary-freeze.md`.

### Findings

- The selected lane expands to
  `bench_reorder --sprint86-slice --skip-factor`.
- The workload is the bounded two-fixture slice:
  - `bcsstk14`
  - `Pres_Poisson`
- The selected metrics are the existing `bench_reorder` CSV fields:
  - `matrix`
  - `n`
  - `reorder`
  - `nnz_L`
  - `reorder_ms`
  - `factor_ms`
  - `reorder_path`
  - `fixture_slice`
  - `nd_base_threshold`
- `nnz_L` is the primary fill-quality field. `reorder_ms` is branch-local
  timing context and must not be read as portable performance proof.
- The current benchmark docs and maintainer guide already describe the bounded
  runtime lane and its claim fence, so no adjacent docs needed edits on Day 7.

### Validation

- Focused runtime/fill command:
  - `make bench-reorder-sprint86`
- Observed output shape:
  - `nd_base_threshold=160`
  - `factor=no`
  - `via_analyze=no`
  - `slice=sprint86`
  - rows for `bcsstk14` and `Pres_Poisson`
  - rows for `none`, `rcm`, `amd`, `colamd`, and `nd`
  - `factor_ms=skip`, `reorder_path=direct`, `fixture_slice=sprint86`
- Day 7 changed planning documentation only, so full source validation is not
  required.

### Day 7 Exit State

Sprint 98 now has a frozen runtime/fill boundary. Day 8 should produce the
bounded runtime/fill artifact from `make bench-reorder-sprint86` without
expanding canonical reporting, workflow checks, benchmark code, or timing
thresholds unless the boundary is explicitly reopened.

## Day 8 - Runtime/Fill Comparison Batch 1

### Goal

Produce the initial bounded runtime/fill comparison artifact from the frozen
Day 7 `make bench-reorder-sprint86` workload.

### Actions

- Re-read the Day 7 runtime/fill boundary.
- Reran `make bench-reorder-sprint86`.
- Captured the raw bounded two-fixture CSV output.
- Computed fill reductions against each fixture's `none` row using `nnz_L`.
- Recorded the artifact in
  `artifacts/day8-runtime-fill-comparison-batch1.md`.
- Kept benchmark code, benchmark docs, canonical reporting, workflows, and
  timing thresholds unchanged.

### Findings

- The selected workload still emits the expected stable schema:
  `matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold`.
- Both selected fixtures were present:
  - `bcsstk14`
  - `Pres_Poisson`
- All selected reorder rows were present for both fixtures:
  - `none`
  - `rcm`
  - `amd`
  - `colamd`
  - `nd`
- `nnz_L` remains the primary fill comparison field.
- `reorder_ms` remains branch-local timing context only.
- The run preserved `factor_ms=skip`, `reorder_path=direct`,
  `fixture_slice=sprint86`, and `nd_base_threshold=160`.

### Validation

- Focused runtime/fill command:
  - `make bench-reorder-sprint86`
- Day 8 changed planning documentation only, so full source validation is not
  required.

### Day 8 Exit State

Sprint 98 now has its initial bounded runtime/fill comparison artifact. Day 9
should complete the lane by checking whether any benchmark-doc or maintainer
guardrail is needed, then write the runtime/fill closeout without widening the
selected workload.

## Day 9 - Runtime/Fill Comparison Batch 2

### Goal

Complete the runtime/fill comparison lane by aligning the Day 8 artifact with
maintainer benchmark-governance language and rerunning the focused validation
command.

### Actions

- Re-read the Day 8 runtime/fill artifact and benchmark-governance section in
  `docs/maintainer_guide.md`.
- Added a maintainer-guide guardrail that names the Sprint 98
  `make bench-reorder-sprint86` artifact as a bounded two-fixture calibration
  slice.
- Kept benchmark code, benchmark schema, Makefile targets, workflow files,
  benchmark README content, and canonical reporting unchanged.
- Reran the focused runtime/fill validation command.
- Recorded the closeout in
  `artifacts/day9-runtime-fill-comparison-closeout.md`.

### Findings

- The Day 8 artifact is sufficient as the Sprint 98 runtime/fill evidence
  owner.
- The maintainer guide needed one explicit guardrail so future readers do not
  treat the Sprint 98 artifact as canonical-report expansion or a portable
  timing claim.
- No benchmark schema or command documentation changed, so
  `benchmarks/README.md` did not need an update.
- The validation output preserved the selected two-fixture, five-reorder row
  structure.

### Validation

- Focused runtime/fill command:
  - `make bench-reorder-sprint86`
- Hygiene:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 98 docs and `docs/maintainer_guide.md`

Day 9 changed documentation only. Full source validation is not required.

### Day 9 Exit State

Sprint 98 now has completed bounded runtime/fill evidence and maintainer-facing
claim guardrails. Day 10 can move to coverage-topology audit work.

## Day 10 - Coverage-Topology Audit

### Goal

Audit Sprint 98 proof-owner, comparison-owner, coverage, workflow, and
validation-target topology after the correctness and runtime/fill lanes landed.

### Actions

- Re-read the Day 10 plan.
- Scanned proof-owner references in:
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - Sprint 98 artifacts
  - `.github/workflows/`
  - `Makefile`
  - `tests/test_ldlt_csc.c`
  - `tests/ldlt_external_dense_reference.py`
- Audited the new correctness lane against the existing Cholesky CSC external
  lane.
- Audited the runtime/fill lane against benchmark governance and workflow
  labels.
- Audited coverage references and confirmed Sprint 98 did not change coverage
  targets, thresholds, or workflow behavior.
- Recorded the audit in
  `artifacts/day10-coverage-topology-audit.md`.

### Findings

- The new LDLT CSC external correctness lane is coherent and should stay
  family-local for now:
  - `tests/test_ldlt_csc.c`
  - `tests/ldlt_external_dense_reference.py`
- The runtime/fill lane is coherent and should stay artifact-owned rather than
  becoming canonical reporting:
  - `make bench-reorder-sprint86`
  - Day 8 and Day 9 Sprint 98 artifacts
- Coverage remains supplemental and tree-mutating; no target or threshold
  cleanup is justified.
- Workflow labels already preserve reviewed, supplemental, and staged platform
  distinctions.
- The highest-value Day 11 cleanup is not structural movement. It is a compact
  maintainer-guide topology snapshot that ties the new Sprint 98 evidence lanes
  together and states that coverage/workflows were audited but not widened.

### Validation

- Day 10 changed planning documentation only.
- Required hygiene:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 98 planning docs

### Day 10 Exit State

Sprint 98 now has a coverage/proof-owner topology audit and a bounded Day 11
cleanup target. Day 11 should add the compact maintainer-guide topology map
without moving proof code, changing benchmark behavior, or widening workflow
or coverage claims.

## Day 11 - Coverage-Topology Cleanup Batch

### Goal

Implement the Day 10 cleanup target by adding a compact maintainer-guide
topology snapshot for Sprint 98 assurance evidence.

### Actions

- Re-read the Day 10 topology audit.
- Added `Sprint 98 Assurance Topology Snapshot` to
  `docs/maintainer_guide.md`.
- Mapped the new Sprint 98 evidence classes to owner, validation command, and
  interpretation boundary:
  - LDLT CSC external correctness
  - reorder/fill calibration
  - coverage topology
  - workflow topology
- Recorded the cleanup in
  `artifacts/day11-coverage-topology-cleanup.md`.
- Left test code, benchmark code, Makefile targets, workflow files, coverage
  targets, and public README claims unchanged.

### Findings

- The topology cleanup is documentation-only and reduces discoverability
  fragmentation without moving proof owners.
- The LDLT CSC correctness lane remains family-local to
  `tests/test_ldlt_csc.c` and `tests/ldlt_external_dense_reference.py`.
- The reorder/fill lane remains artifact-owned by
  `make bench-reorder-sprint86` and the Sprint 98 planning artifacts.
- Coverage and workflows were explicitly mapped as audited but not widened.

### Validation

- Day 11 changed documentation only.
- Hygiene:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 98 docs and `docs/maintainer_guide.md`

No focused test or benchmark rerun was required because no proof surface was
renamed, moved, or behaviorally changed.

### Day 11 Exit State

Sprint 98 now has a compact maintainer-guide assurance topology map. Day 12
can review CI and support-surface alignment from that map without reopening
test, benchmark, workflow, or coverage behavior.

## Day 12 - CI and Support-Surface Alignment

### Goal

Reconcile workflows, local gates, public docs, benchmark docs, and maintainer
guidance with the widened but bounded Sprint 98 assurance model.

### Actions

- Re-read the Day 12 plan and Day 11 topology snapshot.
- Reviewed workflow labels and comments in:
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Reviewed public/support docs:
  - `README.md`
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- Confirmed local validation ownership for:
  - LDLT CSC external correctness
  - Sprint 98 reorder/fill calibration
  - coverage targets
  - platform workflow claims
- Recorded the alignment in
  `artifacts/day12-ci-support-surface-alignment.md`.

### Findings

- The workflow comments already preserve the reviewed/supplemental/staged
  split.
- Linux remains the strongest reviewed source of truth, with supplemental
  runtime, benchmark, sanitizer, TSan, and coverage signals.
- macOS remains the enforced Apple Clang reviewed path plus supplemental
  Homebrew GCC and static-first install confidence.
- Windows remains the reviewed CMake-first consumer subset.
- Public docs are already appropriately high-level and do not claim broad LDLT
  external proof, portable timing parity, or widened coverage/workflow scope.
- Benchmark docs already describe the bounded ND rerun slice and canonical
  reporting boundary.
- No workflow, public-doc, benchmark-doc, Makefile, coverage, or code edit was
  needed.

### Validation

- Day 12 changed planning documentation only.
- Hygiene:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 98 docs

### Day 12 Exit State

Sprint 98 support surfaces are aligned with the bounded assurance model. Day 13
should run the practical validation sweep across touched code, helper,
runtime/fill artifact, and documentation hygiene, then convert remaining work
into a residual queue.

## Day 13 - Validation Sweep and Residual Queue

### Goal

Run the strongest practical validation set for Sprint 98 touched surfaces and
convert remaining assurance work into a bounded residual queue.

### Actions

- Reran focused helper checks:
  - `python3 tests/ldlt_external_dense_reference.py kkt5`
  - `python3 tests/ldlt_external_dense_reference.py kkt10`
  - `python3 tests/ldlt_external_dense_reference.py nope`
- Reran focused LDLT CSC validation:
  - `make build/test_ldlt_csc && ./build/test_ldlt_csc`
- Reran focused runtime/fill validation:
  - `make bench-reorder-sprint86`
- Reran the full required quality chain because Sprint 98 includes a C test
  change:
  - `make format && make lint && make test`
- Ran documentation and whitespace hygiene.
- Scanned touched docs and public/support docs for stale or overstated claims.
- Recorded validation results and residual work in
  `artifacts/day13-validation-and-residual-queue.md`.

### Findings

- LDLT helper positive fixtures passed.
- LDLT helper unknown fixture emitted the expected error and exited `1`.
- Focused `test_ldlt_csc` passed:
  - 98 tests passed
  - 0 failed
  - 0 skipped
- Focused runtime/fill output preserved the selected two-fixture,
  five-reorder-row structure.
- Full quality validation passed and ended with `All tests passed.`
- Stale-claim scan found only negative guardrails and boundary language, not
  overstated positive claims.

### Validation

- `python3 tests/ldlt_external_dense_reference.py kkt5`
- `python3 tests/ldlt_external_dense_reference.py kkt10`
- `python3 tests/ldlt_external_dense_reference.py nope`
- `make build/test_ldlt_csc && ./build/test_ldlt_csc`
- `make bench-reorder-sprint86`
- `make format && make lint && make test`
- `git diff --check`
- trailing-whitespace scan on touched Sprint 98/code/doc surfaces
- stale-claim scan across Sprint 98 docs, maintainer guide, README, INSTALL,
  and benchmark docs

### Day 13 Exit State

Sprint 98 is ready for Day 14 closeout from a validation standpoint. Remaining
work is queued as bounded residual items for future external correctness,
runtime/fill, coverage topology, and CI/support alignment work.

## Day 14 - Sprint 98 Closeout and Handoff

### Goal

Close Sprint 98 with project-plan item reconciliation, final validation status,
and a Sprint 99 handoff queue.

### Actions

- Re-read the Sprint 98 project-plan section.
- Reconciled the seven Sprint 98 project-plan items against artifacts:
  - comparison-surface rerank
  - proof/comparison architecture design
  - external correctness expansion
  - runtime/fill comparison batch
  - coverage-topology cleanup
  - CI/support-surface alignment
  - validation and closeout
- Wrote the closeout and Sprint 99 handoff artifact:
  - `artifacts/day14-closeout-and-handoff.md`
- Preserved the Day 13 validation summary as the final full validation record.
- Ran final documentation hygiene.

### Findings

- All seven Sprint 98 project-plan items have explicit closeout artifacts.
- The new LDLT CSC external correctness lane is bounded, validated, and
  documented.
- The reorder/fill runtime artifact is bounded, validated, and documented.
- Coverage and workflow topology were audited but not widened.
- Public docs remain appropriately high-level; maintainer-only proof details
  stay in the maintainer guide and Sprint artifacts.
- Sprint 99 can start from a ranked queue for external correctness,
  runtime/fill, coverage topology, and CI/support alignment.

### Validation

- Day 14 changed documentation only.
- Day 13 already ran the required post-code-change full validation:
  - `make format && make lint && make test`
  - result: `All tests passed.`
- Day 14 hygiene:
  - `git diff --check`
  - trailing-whitespace scan on Sprint 98 docs and `docs/maintainer_guide.md`

### Day 14 Exit State

Sprint 98 is closed from an implementation and validation standpoint. The
branch is ready for retrospective/PR preparation after any requested final
review.
