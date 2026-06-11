# Sprint 63 Working Notes

## Day 1

**Objective:** Turn the Sprint 63 project-plan scope plus the Sprint 62
validated close into a concrete direct-lifecycle implementation starting point
by confirming the preserved reviewed baseline, fixing the strongest live
LU/CSC lifecycle hotspots, and making the sprint workstreams explicit before
design or code work begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 63 project-plan source and the new sprint plan:
   - `sed -n '121,149p' docs/planning/EPIC_6/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_63/PLAN.md`
3. Re-read the strongest inherited Sprint 62 closeout source:
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_62/RETROSPECTIVE.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_62/artifacts/day14-closeout-and-handoff.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Measure the main Sprint 63 public/direct/docs/proof hotspots:
   - `wc -l README.md docs/tutorial.md docs/maintainer_guide.md include/sparse_analysis.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h src/sparse_analysis.c src/sparse_lu.c src/sparse_cholesky.c src/sparse_chol_csc.c src/sparse_ldlt.c src/sparse_ldlt_csc.c tests/test_integration.c tests/test_sparse_lu.c tests/test_chol_csc.c tests/test_ldlt.c tests/test_ldlt_csc.c benchmarks/bench_refactor.c examples/example_analysis.c`

### Day 1 Findings

#### 1. Sprint 63 starts from the Sprint 62 validated close, not from renewed usability or configuration work

Sprint 62 already closed the first Epic 6 direct-usability package:

- one-shot direct wrappers remain first-class/default peer entry points
- the explicit repeated-run direct lifecycle remains the canonical reuse path:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`
- reordered LU and reordered Cholesky now preserve the caller matrix on
  cancel/failure
- the remaining Sprint 62 deferred queue is already narrowed to:
  - no-reorder linked-list Cholesky cancellation restoration
  - broader LDL^T wording follow-through only if needed
  - QR as a comparison/deferred surface
  - deeper direct-lifecycle uniformity and CSC/LU follow-through

Interpretation:

- Sprint 63 is not another one-shot direct-usability sprint
- Sprint 63 is not reopening the Sprint 61 configuration-first debate
- Sprint 63 is the first bounded Epic 6 sprint centered on internal repeated-run
  direct-lifecycle uniformity and CSC/LU follow-through
- the main Day 1 job is to translate that frozen state into an exact LU and
  CSC lifecycle follow-through map

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible through the entire lifecycle-uniformity sprint

The maintained local truth surfaces remain:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the current reviewed wrapper still presents the expected maintained path.

Interpretation:

- Sprint 63 should inherit the exact Sprint 62 truthfulness wording
- later `*.c` / `*.h` landing days should still default to:
  - `make format`
  - `make lint`
  - `make test`
- substantial LU or CSC lifecycle-sensitive days should still treat
  `make quality-review-full` as the stronger default

#### 3. The broad Epic 6 direct-lifecycle claim is already concentrated in LU follow-through and CSC repeated-run seams

The live repo now shows the strongest current Sprint 63 pressure clustered in:

- lifecycle and caller-story wording:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- implementation seams where remaining heterogeneity can still leak through:
  - `src/sparse_analysis.c`
  - `src/sparse_lu.c`
  - `src/sparse_cholesky.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_ldlt_csc.c`
- proof and workflow surfaces that already carry the lifecycle burden:
  - `tests/test_integration.c`
  - `tests/test_sparse_lu.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `benchmarks/bench_refactor.c`
  - `examples/example_analysis.c`

Interpretation:

- Sprint 63 should not pretend direct repeated-run heterogeneity is equally
  spread across every direct family
- the highest-value first batch is LU lifecycle follow-through plus CSC-backed
  repeated-run coherence
- the existing explicit `analysis` / `factors` lifecycle remains the stable
  ownership model rather than something Sprint 63 should blur away

#### 4. Sprint 63 reduces cleanly to seven bounded implementation workstreams

The project-plan scope collapses to:

1. internal path audit
2. LU lifecycle follow-through
3. CSC repeated-run uniformity
4. solve/refactor semantics alignment
5. benchmark/example proof refresh
6. regression expansion
7. validation and closeout

Interpretation:

- the Sprint 63 implementation order is already smaller and clearer than a
  generic “more lifecycle uniformity” description suggests
- the right Day 1 deliverable is not “solve direct lifecycle heterogeneity”
- the right Day 1 deliverable is a bounded implementation map with a fixed
  safety and non-goal fence

#### 5. The strongest likely Sprint 63 touch surfaces are now explicit from the live tree

The highest-value current Sprint 63 surfaces are:

- caller-facing docs and lifecycle story:
  - `README.md` = `983`
  - `docs/tutorial.md` = `464`
  - `docs/maintainer_guide.md` = `391`
- public direct and lifecycle headers:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_lu.h` = `359`
  - `include/sparse_cholesky.h` = `212`
  - `include/sparse_ldlt.h` = `334`
- strongest implementation/lifecycle seams:
  - `src/sparse_analysis.c` = `1020`
  - `src/sparse_lu.c` = `1034`
  - `src/sparse_cholesky.c` = `546`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_ldlt_csc.c` = `2127`
- strongest proof and workflow surfaces likely to matter in Phase 1:
  - `tests/test_integration.c` = `2168`
  - `tests/test_sparse_lu.c` = `908`
  - `tests/test_chol_csc.c` = `4552`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_ldlt_csc.c` = `3680`
  - `benchmarks/bench_refactor.c` = `303`
  - `examples/example_analysis.c` = `210`

Interpretation:

- the early code pressure is concentrated enough to support a bounded first
  landing
- the strongest proof pressure is already split between:
  - public lifecycle regression
  - LU and CSC direct-family repeated-run proof
- example and benchmark follow-through should follow landed lifecycle changes
  rather than lead them

#### 6. Sprint 63 needs an explicit Day 1 non-goal fence before any lifecycle-uniformity design begins

The preserved non-goal fence for Sprint 63 is:

- no reopening the repeated-run workflow fence
- no broad configuration-surface rewrite in the same batch
- no packaging/platform widening disguised as lifecycle work
- no fake family uniformity that breaks real ownership or cancellation
  semantics
- no broad backend/AUTO-policy or benchmark-governance work unless a lifecycle
  landing proves it is actually blocking the direct control plane

Interpretation:

- Sprint 63 should make the existing repeated-run direct story more uniform
  internally
- compatibility behavior should tighten without pretending every direct family
  is now identical
- success is more coherent LU and CSC lifecycle behavior, not erasing the
  distinction between one-shot direct APIs and the explicit repeated-run
  lifecycle

### Day 1 Close

Sprint 63 now starts from one explicit direct-lifecycle implementation
baseline:

- the Sprint 62 direct-usability close remains frozen and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 direct-lifecycle claim has already narrowed to LU
  follow-through, CSC repeated-run seams, and solve/refactor semantics
- the public/implementation/docs/proof hotspots for the first follow-through
  batch are explicit
- the next step is to turn that live lifecycle map into a ranked LU/CSC
  uniformity design before code changes begin

## Day 2

**Objective:** Freeze the validation and truthfulness baseline that Sprint 63
LU and CSC lifecycle implementation work must preserve before the sprint moves
into the deeper internal-path audit.

### Commands Run

1. Confirm branch cleanliness before the Day 2 pass:
   - `git status --short --branch`
2. Re-read the current Sprint 63 notes plus the Day 2 plan slice:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_63/WORKING_NOTES.md`
   - `sed -n '80,150p' docs/planning/EPIC_6/SPRINT_63/PLAN.md`
3. Re-read the strongest inherited Day 2 shape from Sprint 62:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_62/artifacts/day2-validation-baseline-and-touched-surface-recheck.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Re-read the current quality/truthfulness wording:
   - `rg -n "quality-review-full|quality-review-cmake|deadcode|Windows|macOS|Linux|coverage" README.md docs/maintainer_guide.md Makefile .github/workflows`
7. Confirm the Sprint 63 targeted rerun-set presence in the live build tree:
   - `for f in ./build/test_integration ./build/test_sparse_lu ./build/test_cholesky ./build/test_chol_csc ./build/test_ldlt ./build/test_ldlt_csc ./build/test_iterative ./build/test_eigs ./build/test_eigs_lobpcg ./build/example_analysis ./build/example_basic_solve ./build/example_ldlt ./build/example_iterative ./build/example_ic_minres ./build/example_eigs ./build/example_svd_lowrank ./build/bench_refactor ./build/bench_refactor_csc ./build/bench_iterative_reuse ./build/bench_eigs_reuse; do [ -e "$f" ] && echo "$f"; done`

### Day 2 Findings

#### 1. The strongest local reviewed baseline is still `make quality-review-full`

Sprint 63 inherits the same authoritative local validation command as the
Sprint 62 close state:

- `make quality-review-full`

That remains the strongest local reviewed baseline because it preserves both:

- the reviewed Makefile path
- the reviewed CMake parity path

This should remain the top-level local trust anchor unless a later Epic 6
implementation sprint proves that the contract itself must change.

#### 2. The reviewed CMake parity count is still the main numerical truthfulness anchor

The current reviewed CMake inventory remains:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

That count still matters because it is the simplest exact proof that:

- the reviewed CMake path still sees the maintained local full test surface
- Makefile/CMake parity has not drifted silently

#### 3. The current code-day gate versus stronger reviewed baseline split is stable

The maintained split is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial direct-lifecycle or CSC-sensitive work:
  - `make quality-review-full`
- docs-only days:
  - no automatic code-quality gate required
  - use targeted sanity checks instead

This remains consistent with the repo’s current Sprint 62 close discipline and
does not need reinterpretation on Sprint 63 Day 2.

#### 4. The current quality/platform story is coherent across README, maintainer guide, Makefile, and workflows

The main maintained surfaces still agree on the current contract:

- Linux remains the enforced reviewed source-of-truth path
- macOS remains reviewed but narrower, with dead-code still staged
- Windows keeps the reviewed CMake subset enforced while the Makefile reviewed
  wrappers and dead-code flow stay staged
- coverage remains a supplemental signal, not an active reviewed-baseline
  residual
- dead-code remains operationally serialized and separate from `lint` and
  `test`

That means Sprint 63 can proceed from a stable truthfulness contract rather
than needing a wording-reconciliation batch just to start lifecycle
implementation work.

#### 5. The targeted Sprint 63 rerun set is present and aligned to the actual direct lifecycle-risk surface

The confirmed rerun set is:

- direct lifecycle and integration proofs:
  - `./build/test_integration`
- direct solver family proofs:
  - `./build/test_sparse_lu`
  - `./build/test_cholesky`
  - `./build/test_chol_csc`
  - `./build/test_ldlt`
  - `./build/test_ldlt_csc`
- adjacent repeated-run solver proofs:
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_iterative`
  - `./build/example_ic_minres`
  - `./build/example_eigs`
  - `./build/example_svd_lowrank`
- representative workflow benchmarks:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

That is already strong enough to support:

- LU repeated-run follow-through
- CSC lifecycle and dispatch verification
- representative direct example sanity checks after behavior-affecting edits
- adjacent repeated-run regression verification so Sprint 63 does not widen
  solver-support boundaries by accident

### Authoritative Day 2 Validation Boundary

- docs-only days:
  - use targeted sanity checks, not the full code-day gate by default
- bounded `*.c` / `*.h` days:
  - run:
    - `make format`
    - `make lint`
    - `make test`
- substantial LU/CSC lifecycle or semantics-sensitive code days:
  - prefer:
    - `make quality-review-full`
  - and refresh representative proof/benchmark/example surfaces as needed

### Day 2 Exit State

Sprint 63 now has a written validation baseline that matches the live repo:

- strongest local reviewed baseline unchanged
- reviewed CMake parity anchor unchanged
- rerun set fixed from the current build tree around direct lifecycle, LU,
  CSC, and adjacent repeated-run regression surfaces
- docs-only versus code-day versus stronger-review path split fixed explicitly
- no contradiction across the main quality/truthfulness surfaces
