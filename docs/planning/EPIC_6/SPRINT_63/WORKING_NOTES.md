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

Interpretation:

- Sprint 63 already has a concrete validation surface that matches the actual
  direct lifecycle risk
- the integration proof remains the first public lifecycle truth surface
- LU, Cholesky, and CSC-specific tests already provide natural family-local
  follow-through proof homes

### Day 2 Close

Sprint 63 now has one explicit validation contract before lifecycle code
changes begin:

- `make quality-review-full` remains the strongest local reviewed baseline
- the reviewed CMake parity anchor remains exact at `53`
- the maintained quality/platform story is coherent across the live repo
  surfaces
- the targeted Sprint 63 rerun set is fixed and present in `build/`
- the next step is the deeper internal-path audit that ranks LU, CSC, and
  remaining direct-family lifecycle seams before design or code work lands

## Day 3

**Objective:** Reduce the broad Sprint 63 “direct-lifecycle uniformity” claim
to a ranked live seam map by auditing the current LU, Cholesky/CSC, and LDL^T
internal paths before choosing the first bounded implementation target.

### Commands Run

1. Confirm branch cleanliness before the Day 3 audit:
   - `git status --short --branch`
2. Re-read the Day 3 sprint-plan slice plus current Sprint 63 notes:
   - `sed -n '115,210p' docs/planning/EPIC_6/SPRINT_63/PLAN.md`
   - `sed -n '1,360p' docs/planning/EPIC_6/SPRINT_63/WORKING_NOTES.md`
3. Map the main direct lifecycle and CSC-sensitive seams:
   - `rg -n "sparse_analyze|sparse_factor_numeric|sparse_factor_solve|sparse_refactor_numeric|reorder|publish|cancel|working copy|analysis|factors" include/sparse_analysis.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h src/sparse_analysis.c src/sparse_lu.c src/sparse_cholesky.c src/sparse_chol_csc.c src/sparse_ldlt.c src/sparse_ldlt_csc.c tests/test_integration.c tests/test_sparse_lu.c tests/test_chol_csc.c tests/test_ldlt.c tests/test_ldlt_csc.c benchmarks/bench_refactor.c examples/example_analysis.c`
4. Re-read the main public direct/lifecycle headers:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,240p' include/sparse_cholesky.h`
   - `sed -n '1,260p' include/sparse_ldlt.h`
5. Re-read the strongest current internal implementation seams:
   - `sed -n '1,260p' src/sparse_lu.c`
   - `sed -n '1,260p' src/sparse_cholesky.c`
   - `sed -n '1,260p' src/sparse_chol_csc.c`
   - `sed -n '1,260p' src/sparse_ldlt.c`
   - `sed -n '1,260p' src/sparse_ldlt_csc.c`

### Day 3 Findings

#### 1. LU now owns the strongest remaining direct-lifecycle follow-through seam

The live LU surface is materially cleaner than before Sprint 62, but it still
has the strongest remaining lifecycle crossover:

- the public wrapper story is now explicit about fresh-matrix versus explicit
  repeated-run lifecycle use
- reordered one-shot publication already preserves the caller matrix on
  cancel/failure
- but the implementation still contains the strongest wrapper-to-lifecycle
  crossover logic through the default-compatible shared-lifecycle path and
  publish-back behavior

Interpretation:

- LU is the best first Sprint 63 implementation target
- the main remaining LU problem is no longer basic one-shot mutation surprise
- the main remaining LU problem is lifecycle/result/factor-state coherence
  where one-shot behavior and the shared repeated-run machinery still meet

#### 2. Cholesky now owns the strongest CSC repeated-run uniformity seam

The public Cholesky story is also materially cleaner than before Sprint 62:

- reordered one-shot preservation is already hardened
- the public header already describes the shipped reordered-path preservation
  rule
- the remaining pressure is now mostly internal rather than caller-story first

The strongest live asymmetry sits behind the public surface:

- linked-list versus CSC path differences
- CSC conversion, write-back, and backend dispatch behavior
- analysis-aware repeated-run coherence on the CSC side

Interpretation:

- Cholesky is now the strongest second target
- the right Sprint 63 Cholesky work is CSC/lifecycle uniformity follow-through,
  not another broad public-wrapper cleanup pass
- the strongest proof burden after LU remains split between
  `tests/test_integration.c` and `tests/test_chol_csc.c`

#### 3. LDL^T is cleaner than the top-level Sprint 63 headline might suggest

The live LDL^T surfaces are comparatively less urgent:

- the family-local ownership model is already explicit
- the one-shot path is less entangled with the shared lifecycle than LU
- CSC complexity exists, but it is not currently the strongest first
  contradiction in the direct repeated-run story

Interpretation:

- LDL^T remains a later follow-through target, not the best first Sprint 63
  landing
- Sprint 63 should not widen into LDL^T just to create cosmetic family
  symmetry
- LDL^T should only move early if a later design pass exposes a concrete
  contradiction that LU and Cholesky do not already cover

#### 4. The proof burden already has a clear home and does not need a new harness

The current proof split is already strong enough for the next design step:

- `tests/test_integration.c` remains the highest-signal public lifecycle proof
  home
- `tests/test_sparse_lu.c` is the natural family-local LU follow-through proof
  surface if needed later
- `tests/test_chol_csc.c` is the natural CSC repeated-run proof home
- the example and benchmark follow-through surfaces should stay later and
  downstream:
  - `examples/example_analysis.c`
  - `benchmarks/bench_refactor.c`

Interpretation:

- Sprint 63 does not need a new bespoke lifecycle harness
- the right next step is a bounded design pass over the existing public and
  family-local proof homes
- example and benchmark updates should follow landed lifecycle semantics rather
  than lead them

#### 5. The exact Day 4 target is now fixed

The broad Day 1 lifecycle map now reduces to a concrete ranked queue:

1. LU lifecycle follow-through
2. Cholesky CSC repeated-run uniformity
3. LDL^T follow-through only if later needed
4. QR remains a comparison/deferred surface

Interpretation:

- Day 4 should design the LU first landing explicitly
- Day 4 should also fix the Cholesky/CSC second-cut fence
- Sprint 63 should stay out of QR and broad LDL^T work unless the design pass
  proves they are actually blocking the direct control plane

### Day 3 Close

Sprint 63’s broad lifecycle-uniformity claim is now reduced to one ranked live
seam map:

- LU is the strongest first follow-through target
- Cholesky is the strongest second target through CSC repeated-run asymmetry
- LDL^T is cleaner and stays in the later/deferred lane
- the regression burden already has a clear home in the current integration and
  family-local proof surfaces
- the next step is a bounded Day 4 design that turns this ranking into an
  exact first implementation fence

## Day 4

**Objective:** Turn the Day 3 ranking into an explicit lifecycle-uniformity
design and safety contract by separating what Sprint 63 should normalize in
public behavior, what it should harden internally, and what it should leave on
the compatibility/deferred lane.

### Commands Run

1. Confirm branch cleanliness before the Day 4 design pass:
   - `git status --short --branch`
2. Re-read the Day 4-5 sprint-plan slice plus current Sprint 63 notes:
   - `sed -n '120,240p' docs/planning/EPIC_6/SPRINT_63/PLAN.md`
   - `sed -n '1,520p' docs/planning/EPIC_6/SPRINT_63/WORKING_NOTES.md`
3. Re-scan the strongest LU lifecycle crossover seam:
   - `rg -n "s51_lu_opts_can_use_shared_lifecycle|s51_lu_publish_analysis_factor|sparse_factor_solve|sparse_refactor_numeric|publish|working copy|reorder|cancel|analysis|factors" src/sparse_lu.c include/sparse_lu.h tests/test_integration.c tests/test_sparse_lu.c`
4. Re-scan the strongest Cholesky/CSC publication and dispatch seams:
   - `rg -n "working copy|publish|reorder|cancel|analysis|factors|refactor|solve|csc" src/sparse_cholesky.c src/sparse_chol_csc.c include/sparse_cholesky.h tests/test_integration.c tests/test_chol_csc.c`

### Day 4 Findings

#### 1. Sprint 63 should normalize lifecycle interpretation, not redesign the direct API split

The preserved direct workflow fence stays exact:

- one-shot direct wrappers remain first-class/default peer entry points
- the explicit repeated-run direct lifecycle remains:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`

Interpretation:

- Sprint 63 should not hide the public difference between one-shot wrappers and
  the explicit repeated-run lifecycle
- Sprint 63 should instead make the internal ownership and result semantics
  more coherent where the same direct family already crosses between those
  paths

#### 2. LU should land first as factor-state and result-semantics follow-through

The strongest LU seam is now exact:

- default-compatible reordered LU already crosses into the shared lifecycle
  machinery
- reordered one-shot publication already preserves the caller matrix on
  cancel/failure
- the remaining risk is lifecycle/result/factor-state coherence when one-shot
  LU succeeds, fails, or rejects refactor-like re-entry on an already
  reordered/factored matrix

The right Sprint 63 LU ownership split is:

- public behavior:
  - clearer one-shot versus lifecycle positioning where still useful
  - no new top-level direct API
- internal hardening:
  - factor publication semantics
  - rejection/preservation semantics on old factors
  - wrapper/shared-lifecycle coherence for solve/refactor-like outcomes
- proof:
  - `tests/test_integration.c` first
  - `tests/test_sparse_lu.c` only if the family-local burden becomes real

Interpretation:

- LU Day 6-7 work should target factor-state/result semantics first
- it should not widen into broad docs/examples or other direct families

#### 3. Cholesky should land second as CSC repeated-run publication and dispatch follow-through

The strongest Cholesky seam is also now exact:

- reordered one-shot preservation is already hardened
- the remaining mismatch is internal, especially where the CSC-backed path
  carries analysis-aware conversion, write-back, and repeated-run behavior that
  is less uniform than the public lifecycle story

The right Sprint 63 Cholesky ownership split is:

- public behavior:
  - only minimal wording follow-through if the landed internal behavior needs
    it
- internal hardening:
  - CSC dispatch/lifecycle coherence
  - solve/refactor publication discipline
  - repeated-run state retention where CSC and linked-list paths currently
    diverge more than justified
- proof:
  - `tests/test_integration.c` for public lifecycle semantics
  - `tests/test_chol_csc.c` for CSC-family repeated-run proof

Interpretation:

- Cholesky Day 8-10 work should be a CSC follow-through sprint inside Sprint 63
- it should not reopen the already-settled one-shot reordered preservation
  story unless a concrete contradiction appears

#### 4. Compatibility rules are now explicit enough to guide the code batches

The preserved compatibility behavior is:

- one-shot wrappers remain one-shot and caller-owned
- the explicit repeated-run lifecycle remains the only canonical reuse path
- LU and Cholesky reordered one-shot calls keep their Sprint 62 caller-matrix
  preservation guarantees on cancel/failure
- family-local cancellation differences that were intentionally preserved in
  Sprint 62 stay family-local unless Sprint 63 proves a tighter change is both
  safe and high-value

What becomes more uniform in Sprint 63:

- factor-state publication and old-factor preservation semantics
- solve/refactor interpretation across LU and CSC-backed repeated-run direct
  paths
- internal CSC dispatch and state-retention discipline where the public
  lifecycle story already promises a coherent repeated-run path

What stays out of scope:

- no fake family identity across LU, Cholesky, LDL^T, and QR
- no broad cancellation-model rewrite
- no direct-API redesign
- no packaging/platform/configuration spillover

#### 5. The first implementation fence is now fixed

The exact Sprint 63 implementation order is now:

1. LU lifecycle follow-through
2. Cholesky CSC repeated-run uniformity
3. later proof/example/benchmark refresh if the landed semantics justify it
4. LDL^T only if a later contradiction appears
5. QR deferred

The likely ownership lanes are now explicit:

- public/header/doc surface:
  - small, bounded, only if implementation landing needs it
- internal implementation surface:
  - primary ownership for Sprint 63
- proof surface:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - optional bounded `tests/test_sparse_lu.c`

### Day 4 Close

Sprint 63 now has one explicit lifecycle-uniformity design contract before the
first code batch:

- the direct API split remains preserved
- LU is fixed as a factor-state/result-semantics first landing
- Cholesky is fixed as a CSC repeated-run follow-through second landing
- compatibility behavior is explicit about what stays unchanged versus what
  becomes more uniform
- Day 5 can now reduce this contract to an exact touched-file implementation
  fence

## Day 5

**Objective:** Convert the Day 4 lifecycle-uniformity contract into an exact
Day 6-10 touched-file and helper-boundary plan so the first LU and CSC
implementation batches stay narrow instead of drifting across unrelated direct
families or docs surfaces.

### Commands Run

1. Confirm branch cleanliness before the Day 5 landing design:
   - `git status --short --branch`
2. Re-read the Day 5 sprint-plan slice plus current Sprint 63 notes:
   - `sed -n '150,290p' docs/planning/EPIC_6/SPRINT_63/PLAN.md`
   - `sed -n '1,760p' docs/planning/EPIC_6/SPRINT_63/WORKING_NOTES.md`
3. Re-read the exact LU public and implementation seam:
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,260p' src/sparse_lu.c`
4. Re-read the exact Cholesky/CSC public and implementation seam:
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,320p' src/sparse_cholesky.c`
   - `sed -n '1,260p' src/sparse_chol_csc.c`
5. Re-scan the helper and state-publication support lane:
   - `rg -n "sparse_factor_state|matrix_state|publish_factored|replace_reorder_perm|used_csc_path|factor_numeric|refactor_numeric|writeback_to_sparse|from_sparse_with_analysis|with_analysis" src tests include`

### Day 5 Findings

#### 1. The minimum viable Sprint 63 public surface is smaller than the broad sprint scope suggests

The first code landing does not need a wide public-surface sweep.

Required public/header lane:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`

Only if the landed implementation forces follow-through later:

- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`

Interpretation:

- Day 6-10 should stay implementation-first
- public header edits should be small truthfulness adjustments, not another
  adoption-story rewrite
- broader docs/example/benchmark follow-through should remain later and
  conditional on real landed semantics

#### 2. The exact first LU implementation fence is now fixed

Required Day 6 LU lane:

- `src/sparse_lu.c`
- `tests/test_integration.c`

Likely public/header companion:

- `include/sparse_lu.h`

Optional support/helper lane only if the first landing proves it necessary:

- `src/sparse_factor_state_internal.c`
- `src/sparse_matrix_internal.h`
- `src/sparse_matrix_state_internal.h`
- `tests/test_sparse_lu.c`

The intended LU focus is narrow:

- factor publication semantics
- rejection/preservation of old factors on wrapper re-entry
- solve/refactor-style result-state coherence where the one-shot wrapper and
  shared lifecycle already meet

Interpretation:

- Day 6 should start in `src/sparse_lu.c` plus `tests/test_integration.c`
- helper/state files should only move if the current factor-state seam is too
  awkward to harden in place
- `src/sparse_analysis.c` is not part of the first LU landing by default

#### 3. The exact first Cholesky/CSC implementation fence is now fixed

Required Day 7-10 CSC lane:

- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`

Likely public/header companion:

- `include/sparse_cholesky.h`

Optional follow-through only if the CSC/public lifecycle seam proves it
necessary:

- `src/sparse_analysis.c`
- `include/sparse_analysis.h`

The intended CSC focus is also narrow:

- CSC publication/write-back discipline
- CSC dispatch/state-retention coherence
- repeated-run solve/refactor behavior that is already supposed to look stable
  through the explicit public lifecycle

Interpretation:

- Sprint 63 should touch `src/sparse_analysis.c` only if the Cholesky/CSC
  landing proves that the public repeated-run path itself is the real seam
- otherwise the right first reduction is local to `src/sparse_cholesky.c` and
  `src/sparse_chol_csc.c`

#### 4. The proof home split is now exact enough to stop new harness drift

Primary proof surfaces:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`

Secondary/optional proof surface:

- `tests/test_sparse_lu.c`

Not part of the first landing by default:

- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`
- new bespoke lifecycle harness files

Interpretation:

- integration remains the public lifecycle truth surface
- CSC-family proof remains in `tests/test_chol_csc.c`
- LU family-local proof should widen only if the integration burden becomes too
  awkward or too opaque

#### 5. The Day 6-10 implementation boundary is now operational

The exact implementation order is:

1. Day 6:
   - bounded LU lifecycle follow-through in:
     - `src/sparse_lu.c`
     - `tests/test_integration.c`
     - optional small `include/sparse_lu.h` truthfulness follow-through
2. Day 7:
   - first bounded CSC repeated-run uniformity slice in:
     - `src/sparse_cholesky.c`
     - `src/sparse_chol_csc.c`
     - `tests/test_integration.c`
     - `tests/test_chol_csc.c`
3. Day 8:
   - post-landing audit on the remaining LU/CSC queue
4. Day 9-10:
   - second bounded follow-through slice only if the audit exposes a real
     remaining lifecycle contradiction

The explicit non-goal fence is now also exact:

- no `src/sparse_ldlt.c`
- no `src/sparse_ldlt_csc.c`
- no `include/sparse_ldlt.h`
- no `src/sparse_qr.c`
- no `include/sparse_qr.h`
- no broad `README.md` / tutorial rewrite
- no benchmark-governance or packaging/platform work
- no configuration-surface spillover

### Day 5 Close

Sprint 63 now has one exact implementation fence before the first code batch:

- the minimum viable public surface is bounded to LU and Cholesky headers
- the first LU landing is fixed to `src/sparse_lu.c` plus integration proof,
  with helper/state files only optional
- the first Cholesky/CSC landing is fixed to `src/sparse_cholesky.c`,
  `src/sparse_chol_csc.c`, integration proof, and CSC-family proof
- `src/sparse_analysis.c` is explicitly conditional rather than assumed
- LDL^T, QR, docs simplification, packaging, and configuration work stay
  outside the first implementation fence

## Day 6 - 2026-06-10

### Goal

Land the first bounded LU lifecycle follow-through slice inside the Day 5
fence:

- keep the work in `src/sparse_lu.c` plus integration proof
- tighten wrapper/shared-lifecycle rejection semantics where the current seam is
  genuinely under-specified
- preserve Sprint 62 caller-matrix preservation guarantees and avoid widening
  into Cholesky, LDL^T, QR, or broad docs work

### Actions

1. Re-read the live LU one-shot and integration proof seam in:
   - `include/sparse_lu.h`
   - `src/sparse_lu.c`
   - `tests/test_integration.c`
2. Re-read the adjacent factor-state and matrix-state helpers to confirm whether
   Day 6 really needed support-lane widening:
   - `src/sparse_factor_state_internal.c`
   - `src/sparse_matrix_internal.h`
   - `src/sparse_matrix_state_internal.h`
3. Identify the smallest real lifecycle contradiction still left in the LU
   wrapper path.
4. Land the implementation and proof only in the minimum file set that closes
   that contradiction.
5. Run the full required validation gate for `*.c` / `*.h` direct-lifecycle
   work.

### Findings

#### 1. The real Day 6 seam was invalid LU pivot handling, not a broader state helper rewrite

After re-reading the live path, the highest-value remaining LU contradiction was
not reorder publication anymore; Sprint 62 had already closed that. The live
remaining hole was narrower:

- `reorder` was validated explicitly
- `pivot` was not validated explicitly before entering the one-shot factor path
- this left invalid enum values under-specified at the wrapper boundary

That made invalid pivot handling the right Day 6 seam because it sits directly
at the one-shot wrapper/shared-lifecycle contract:

- callers should get a deterministic `SPARSE_ERR_BADARG`
- the matrix should remain unchanged
- a later valid retry should still work

#### 2. The fix stayed inside the planned first landing fence

The landed Day 6 patch only touched:

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`

I did not need to widen into:

- `src/sparse_factor_state_internal.c`
- `src/sparse_matrix_internal.h`
- `src/sparse_matrix_state_internal.h`
- `tests/test_sparse_lu.c`
- `src/sparse_analysis.c`

That kept the LU batch aligned to the Day 5 promise: first harden the wrapper
boundary in place, then widen helper/state files only if the seam proves it is
actually necessary.

#### 3. Invalid pivot values now reject cleanly before mutation

The implementation change in `src/sparse_lu.c` is intentionally small:

- add a local LU-pivot validation helper
- reject invalid pivot values in both:
  - `sparse_lu_factor_inner(...)`
  - `sparse_lu_factor_opts(...)`

Result:

- invalid pivot now returns `SPARSE_ERR_BADARG`
- the rejection happens before reorder/factor mutation work begins
- the wrapper semantics now match the already-explicit invalid-`reorder` lane

This is the right kind of Sprint 63 uniformity work:

- clearer wrapper contract
- no ownership redesign
- no hidden copy behavior
- no compatibility break in valid paths

#### 4. The public header truth surface now says the invalid-pivot contract directly

`include/sparse_lu.h` was updated only enough to keep the public contract
truthful:

- `sparse_lu_factor_opts(...)` now explicitly lists invalid `opts->pivot` as a
  `SPARSE_ERR_BADARG` case
- `sparse_lu_factor(...)` now explicitly lists invalid `pivot` as a
  `SPARSE_ERR_BADARG` case

This keeps the header aligned with the shipped wrapper behavior without turning
Day 6 into a larger docs pass.

#### 5. Integration proof now shows preserved original state and successful retry

The new integration proof is:

- `test_lu_invalid_pivot_opts_preserve_original_matrix_and_allow_retry`

It proves the exact Day 6 contract:

- invalid pivot through `sparse_lu_factor_opts(...)` returns `SPARSE_ERR_BADARG`
- the matrix row/column permutation state stays identity
- representative matrix entries remain unchanged
- no factor is published by the failed call
- a later valid LU one-shot retry still succeeds

This keeps the proof where the public lifecycle story already lives instead of
spreading the same story into another family-local harness too early.

### Validation

Because `*.c` / `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 299.14 sec`

One non-blocking note remains unchanged from the inherited reviewed baseline:

- the reviewed CMake rebuild again emitted the existing
  `bench_eigs_reuse.c` double-promotion warnings
- the full reviewed path still completed cleanly and passed all parity gates

### Day 6 Close

Sprint 63 Day 6 landed one bounded LU lifecycle follow-through slice without
breaking the implementation fence:

- invalid LU pivot values now reject deterministically with `SPARSE_ERR_BADARG`
- the rejection happens before reorder/factor mutation
- the caller matrix stays unchanged on that failure path
- a later valid one-shot retry still succeeds
- the batch stayed inside `src/sparse_lu.c` plus header truthfulness and
  integration proof

## Day 7 - 2026-06-10

### Goal

Land the first bounded Cholesky CSC repeated-run uniformity slice from the
Sprint 63 Day 5 fence, keeping the work local to CSC dispatch coherence,
header truthfulness, and the highest-signal direct proof homes.

### What I changed

I touched only the planned Day 7 surfaces:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`

The landed implementation tightened two concrete Cholesky/CSC seams:

1. invalid backend values are now rejected deterministically with
   `SPARSE_ERR_BADARG`
2. `used_csc_path` is now published immediately after backend selection, before
   later reorder/factor failures can return

That required a small internal dispatch cleanup in `src/sparse_cholesky.c`:

- select and validate the backend once at the wrapper boundary
- thread the resolved `use_csc` decision into both the no-reorder and reordered
  working-copy paths
- remove the later duplicated backend-selection logic from the no-reorder path

The public header was updated only enough to keep the truth surface exact:

- `sparse_cholesky_factor_opts(...)` now explicitly documents invalid
  `opts->backend` as a `SPARSE_ERR_BADARG` case alongside the existing invalid
  reorder/state cases

### Why this was the right Day 7 seam

After the Sprint 62 Cholesky preservation work, the strongest remaining CSC
follow-through hole was narrower than a broad “direct family asymmetry” label
suggested:

- Cholesky did not reject invalid backend enum values explicitly
- Cholesky published `used_csc_path` later than LDLT, after other failures
  could already have returned
- that left wrapper dispatch/result semantics less uniform than the adjacent
  CSC-backed direct family

That made Day 7 a dispatch/result-state batch, not a lifecycle redesign:

- reject invalid backend input early
- publish CSC path telemetry before later reorder/factor failures
- preserve caller-visible direct-lifecycle boundaries and the existing
  cancel/failure model

### Proof added

The public proof expansion landed in `tests/test_integration.c`:

- `test_cholesky_invalid_backend_preserves_original_matrix_and_allows_retry`

It proves:

- invalid backend through `sparse_cholesky_factor_opts(...)` returns
  `SPARSE_ERR_BADARG`
- the caller matrix remains in original identity-permutation state
- no usable factor is published by the failed call
- a later valid reordered CSC retry still succeeds

The family-local CSC proof expansion landed in `tests/test_chol_csc.c`:

- `test_dispatch_invalid_backend_rejected`
- `test_dispatch_csc_reports_selected_path_before_reorder_error`

Those prove:

- invalid backend is rejected explicitly
- `used_csc_path` is still reported as `1` on the selected CSC path even when a
  later invalid reorder argument fails

### Validation

Because `*.c` / `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 311.10 sec`

One non-blocking note remains unchanged from the inherited reviewed baseline:

- the reviewed CMake rebuild again emitted the existing
  `bench_eigs_reuse.c` double-promotion warnings
- the full reviewed path still completed cleanly and passed all parity gates

### Day 7 Close

Sprint 63 Day 7 landed one bounded CSC lifecycle follow-through slice without
widening the sprint:

- invalid Cholesky backend input now rejects deterministically with
  `SPARSE_ERR_BADARG`
- CSC path telemetry is published earlier and more uniformly
- invalid-backend failure preserves the caller matrix and allows a later valid
  retry
- the batch stayed inside `src/sparse_cholesky.c` plus header truthfulness,
  integration proof, and family-local CSC proof

## Day 8 - 2026-06-10

### Goal

Re-rank the remaining Sprint 63 lifecycle queue from the landed Day 6-7 branch
state instead of from the pre-landing audit, then fix the exact Day 9-10
target for the next implementation slice.

### What I reviewed

I re-read the live direct-lifecycle seams across:

- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `tests/test_integration.c`
- `tests/test_sparse_lu.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`
- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

I also rechecked the public truth surfaces in:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`

### Main result

After the Day 6-7 landings, Sprint 63 no longer has a broad wrapper-entry
problem.

The strongest remaining queue is now narrower and more specific:

1. shared public lifecycle solve/refactor semantics
2. large-`n` CSC-backed Cholesky lifecycle failure-path proof
3. docs/example/benchmark follow-through only after the semantics lane is fixed
4. LDL^T remains a lower-priority comparison surface, not the next target

### Why the queue changed

#### 1. LU is no longer the strongest remaining lifecycle seam

The Day 6 batch closed the highest-value LU wrapper contradiction:

- invalid pivot is rejected deterministically
- rejection happens before mutation
- preserved-state retry behavior is explicit and tested

That means LU is no longer the best next implementation target unless a later
solve/refactor semantics pass proves it needs small follow-through.

#### 2. The Day 7 Cholesky CSC dispatch seam is materially reduced

The Day 7 batch closed the highest-value Cholesky CSC wrapper asymmetry:

- invalid backend now rejects explicitly
- CSC dispatch is selected once
- `used_csc_path` is published before later failures

That leaves Cholesky less as a wrapper-entry problem and more as a shared
lifecycle semantics / proof problem.

#### 3. The strongest remaining hole now sits in the shared lifecycle layer

The live code now points to `src/sparse_analysis.c` as the highest-leverage
remaining Sprint 63 seam.

What is already true:

- `sparse_factor_numeric(...)` factors into temporary storage and only replaces
  the caller `factors` object after success
- `sparse_refactor_numeric(...)` validates the existing factor object, factors
  into a temporary, and preserves old factors on error
- `tests/test_integration.c` already proves:
  - zeroed-factor solve rejection
  - mismatched-analysis solve rejection with preserved factors
  - zeroed-factor refactor acceptance
  - mismatched-existing-factor refactor rejection
  - old-factor preservation on refactor failure
  - same-pattern Cholesky public-lifecycle parity against one-shot factorization

What is still uneven:

- the strongest refactor-failure preservation proof is still concentrated in:
  - sub-threshold linked-list Cholesky (`n = 40`)
  - large-`n` LDL^T (`n = 150`) on the indefinite KKT path
- the large-`n` Cholesky public lifecycle path is already proven on success
  (`n = 120`), but its CSC-backed failure/retention semantics are not yet
  pinned with the same strength
- `example_analysis.c` and `bench_refactor.c` correctly describe successful
  same-pattern reuse, but they are not proof surfaces for failure-path
  retention semantics

That makes the remaining Sprint 63 problem a solve/refactor semantics problem,
not another wrapper-entry cleanup.

### Updated rank order

#### 1. Strongest next target: shared direct lifecycle semantics on the CSC-backed Cholesky lane

This is now the best Day 9-10 target because it sits at the intersection of:

- public repeated-run direct lifecycle truthfulness
- CSC-backed direct follow-through
- factor-retention semantics
- already-existing high-signal proof homes

The specific hole is:

- success parity for large-`n` Cholesky public lifecycle is already explicit
- failure-path retention semantics for that same CSC-backed lane are not yet as
  explicit as the linked-list Cholesky and large-`n` LDL^T lanes

#### 2. Secondary target: bounded docs/header/example follow-through

This remains real, but it should only move after the next semantics slice is
landed. The current docs already say the right high-level workflow story; the
remaining issue is precision around what reuse preserves and what failure
retains.

#### 3. Deferred target: LDL^T follow-through

LDL^T is no longer the next best Sprint 63 code target:

- it already has large-`n` public lifecycle same-pattern and failure-preserve
  proof on the KKT path
- its CSC dispatch/result semantics were already tighter than Cholesky before
  Day 7

That leaves LDL^T in the comparison/deferred lane unless the next pass exposes
an actual contradiction.

### Exact Day 9 target

The next design batch should focus on one bounded shared-lifecycle semantics
question:

- how should Sprint 63 pin large-`n` CSC-backed Cholesky
  `factor` / `refactor` / `solve` retention semantics so they read as one
  coherent public direct lifecycle?

The likely touched-file fence is now:

- required:
  - `src/sparse_analysis.c`
  - `tests/test_integration.c`
- likely header truth follow-through only if the landed semantics move it:
  - `include/sparse_analysis.h`
- optional only if family-local proof burden forces it:
  - `tests/test_chol_csc.c`
  - `examples/example_analysis.c`
  - `benchmarks/bench_refactor.c`

### Explicit non-targets after Day 8

The post-landing audit also makes the non-targets clearer:

- no reopening LU one-shot semantics unless the shared lifecycle pass exposes a
  real regression
- no broad LDL^T widening for symmetry
- no benchmark-governance or packaging/platform spillover
- no general docs cleanup while the remaining semantics lane is still moving

### Day 8 Close

Sprint 63 Day 8 reduces the remaining queue to one concrete implementation
question instead of one generic “more lifecycle uniformity” bucket:

- LU wrapper follow-through is no longer the strongest remaining seam
- Cholesky CSC dispatch follow-through is no longer the strongest remaining seam
- the strongest remaining work is now shared direct lifecycle
  solve/refactor semantics on the large-`n` CSC-backed Cholesky lane
- Day 9 can proceed from the landed branch state with an exact touched-file
  fence and a smaller deferred queue

## Day 9 - 2026-06-10

### Goal

Turn the Day 8 rerank into one exact Day 10 code fence for the remaining
shared direct lifecycle semantics lane, without reopening wrapper-entry,
configuration, or broad docs work.

### What I re-read

I re-read the live shared lifecycle path in:

- `src/sparse_analysis.c`
- `tests/test_integration.c`

I also rechecked the adjacent public-story surfaces that explain the repeated-
run contract:

- `include/sparse_analysis.h`
- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

### Main design result

The remaining Sprint 63 semantics queue is now reduced to one exact Day 10
question:

- how should the shared direct lifecycle prove and, if needed, tighten the
  large-`n` CSC-backed Cholesky factor/refactor retention contract?

This is a narrower question than “more direct lifecycle uniformity”:

- `sparse_factor_numeric(...)` already builds a temporary `new_factors` object
  and only swaps it into the caller `factors` object after success
- `sparse_refactor_numeric(...)` already validates existing factors, factors
  into a temporary, and preserves old factors on error
- the missing strength is not the broad mechanism
- the missing strength is the explicit large-`n` CSC-backed Cholesky proof and,
  only if needed, the smallest semantics follow-through that makes that proof
  read as one coherent public contract

### What is already strong enough

The current branch already has the following high-signal proof:

- public lifecycle solve rejects zeroed factors
- public lifecycle solve rejects mismatched analysis and preserves factors
- public lifecycle refactor accepts zeroed factors as a first factorization
- public lifecycle refactor rejects mismatched existing factors
- public lifecycle refactor preserves old factors on failure
- public lifecycle refactor rejects nnz drift and preserves old factors
- public lifecycle same-pattern Cholesky success parity against one-shot
  factorization at `n = 120`, which is already on the CSC side of
  `SPARSE_CSC_THRESHOLD`

That means Day 10 should not try to redesign `sparse_factor_numeric(...)` or
`sparse_refactor_numeric(...)` from scratch.

### What is still not explicit enough

The remaining proof asymmetry is:

- linked-list Cholesky failure-preserve proof exists on the public lifecycle
  path (`n = 40`)
- large-`n` LDL^T failure-preserve proof exists on the public lifecycle path
  (`n = 150`)
- large-`n` Cholesky success parity exists on the public lifecycle path
  (`n = 120`)
- but large-`n` CSC-backed Cholesky failure-preserve semantics are not yet
  pinned with the same explicitness

That is the right Day 10 seam because it is:

- public repeated-run direct lifecycle behavior
- CSC-sensitive
- already close to fully covered
- small enough to land without widening into unrelated direct-family work

### Exact Day 10 target

The next implementation batch should land one bounded CSC-backed public
lifecycle semantics slice:

1. prove that large-`n` CSC-backed Cholesky refactor failure preserves the old
   usable factors on the public lifecycle path
2. prove that the same large-`n` lane rejects gross structure drift while still
   preserving old usable factors
3. only if the proof exposes a real gap, make the smallest `src/sparse_analysis.c`
   follow-through needed to keep the factor/refactor swap semantics uniform and
   explicit

### Exact touched-file fence

Required:

- `tests/test_integration.c`

Likely:

- `src/sparse_analysis.c`

Likely header truth follow-through only if the landed semantics actually move
the contract wording:

- `include/sparse_analysis.h`

Optional only if the proof burden forces it:

- `tests/test_chol_csc.c`
- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

### Intended proof shape

The intended Day 10 proof should stay in `tests/test_integration.c`, not widen
into a new harness.

The best shape is:

- start from the existing large-`n` Cholesky public lifecycle lane (`n = 120`)
- first build a usable baseline factor and solve
- then refactor on:
  - a same-pattern but no-longer-SPD matrix, or
  - a gross-structure-drift matrix,
  whichever closes the strongest missing CSC-backed retention fact first
- prove the failing refactor returns the expected error
- prove a later solve with the old factors still succeeds and matches the
  pre-failure solution

That keeps the proof aligned with the strongest public contract:

- reuse preserves symbolic/permutation setup
- failed refactor does not silently destroy the previous usable numeric factor

### Explicit non-goals

Day 10 should not widen into:

- LU wrapper follow-through
- LDL^T symmetry cleanup
- QR comparison work
- benchmark-governance or packaging/platform work
- broad docs/example cleanup unless the landed semantics actually require a
  wording correction

### Day 9 Close

Sprint 63 Day 9 fixes one exact implementation fence for the remaining queue:

- the shared direct lifecycle mechanism itself is already mostly right
- the missing strength is explicit large-`n` CSC-backed Cholesky
  failure-preserve proof, with code follow-through only if the proof exposes a
  real semantics gap
- Day 10 can now land a bounded `tests/test_integration.c`-first batch instead
  of reopening a general lifecycle redesign

## Day 10

**Objective:** Land the bounded large-`n` CSC-backed Cholesky public lifecycle
semantics batch by proving non-SPD and nnz-drift refactor failures preserve the
old usable factors, and make only the smallest CSC supernodal implementation
follow-through needed to keep that proof truthful.

### Commands Run

1. Re-read the Day 9 design fence and the live touched seams:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_63/artifacts/day9-solve-refactor-semantics-design.md`
   - `sed -n '1750,1905p' tests/test_integration.c`
   - `sed -n '160,260p' src/sparse_chol_csc_supernodal.c`
2. Inspect the live Day 10 diff while iterating on the proof:
   - `git diff -- src/sparse_chol_csc_supernodal.c tests/test_integration.c`
3. Run the required code-touch validation gate on the landed tree:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Day 10 Findings

#### 1. The missing Sprint 63 proof really did sit on the large-`n` CSC-backed Cholesky public lifecycle lane

The landed Day 10 proof stayed in `tests/test_integration.c` exactly as
planned and added two CSC-backed public lifecycle regressions at `n = 120`,
which is safely on the CSC side of `SPARSE_CSC_THRESHOLD`:

- `test_public_lifecycle_cholesky_csc_refactor_preserves_old_factors_on_failure`
- `test_public_lifecycle_cholesky_csc_refactor_rejects_nnz_drift_and_preserves_old_factors`

That keeps the new coverage on the highest-signal public contract surface
instead of widening into a family-local-only proof.

#### 2. The strongest same-pattern failure proof now uses an unambiguous non-SPD trigger

The non-SPD CSC-backed refactor proof now forces failure with a stored negative
diagonal on the retry matrix:

- `sparse_set(A_bad, 0, 0, -1.0)`

That avoids relying on a weaker off-diagonal perturbation interpretation and
pins the contract to the clearest possible public failure condition:

- failing CSC-backed Cholesky refactor returns `SPARSE_ERR_NOT_SPD`
- the old factors remain usable
- a later solve still matches the pre-failure solution

#### 3. One small CSC supernodal guard was enough to keep the CSC-backed failure contract explicit

The only implementation follow-through needed was in
`src/sparse_chol_csc_supernodal.c`.

The landed guard now rejects a non-positive stored diagonal before supernode
dispatch begins:

- iterate columns
- read the first stored entry in each non-empty column
- if the diagonal entry is already non-positive, return `SPARSE_ERR_NOT_SPD`

This keeps the supernodal CSC path aligned with the scalar CSC path on the
simplest SPD contract instead of letting an already-invalid stored diagonal
flow deeper into batched elimination.

#### 4. Gross structure drift is now pinned on the same large-`n` CSC-backed lifecycle lane

The second new integration proof removes both symmetric off-diagonal entries:

- `sparse_set(A_bad, 0, 1, 0.0)`
- `sparse_set(A_bad, 1, 0, 0.0)`

and proves:

- `sparse_refactor_numeric(...)` returns `SPARSE_ERR_BADARG`
- the old CSC-backed factors are preserved
- a later solve still succeeds and matches the baseline solution

This closes the last Sprint 63 Day 9 asymmetry without widening into LDL^T or
QR.

### Validation

Ran and passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 359.66 sec`

### Non-Blocking Note

The reviewed CMake rebuild again emitted the existing
`bench_eigs_reuse.c` double-promotion warnings while rebuilding that bench
binary, but the full reviewed path still completed cleanly and passed all
parity gates.

### Day 10 Close

Sprint 63 Day 10 closes one bounded shared-lifecycle semantics slice:

- large-`n` CSC-backed Cholesky refactor failure now has explicit public
  old-factor-preservation proof for both non-SPD failure and gross nnz drift
- the CSC supernodal path now rejects an already non-positive stored diagonal
  up front
- the sprint can move into final compatibility/documentation follow-through
  without reopening the broader lifecycle design

## Day 11

**Objective:** Tighten the post-Day-10 lifecycle and CSC compatibility surface
by adding the missing family-local CSC regression, removing stale public
header wording, and revalidating the landed state from the strongest reviewed
baseline.

### Commands Run

1. Re-read the Day 10 landing and the Day 11 plan fence:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_63/artifacts/day10-large-n-csc-cholesky-lifecycle-semantics-batch.md`
   - `sed -n '379,414p' docs/planning/EPIC_6/SPRINT_63/PLAN.md`
2. Inspect the live touched CSC/header seams:
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,260p' include/sparse_cholesky.h`
   - `rg -n "supernodal|invalid backend|invalid reorder|stored diagonal" tests/test_chol_csc.c`
3. Land the bounded compatibility/regression patch:
   - `apply_patch` on:
     - `include/sparse_lu.h`
     - `include/sparse_cholesky.h`
     - `tests/test_chol_csc.c`
4. Run the required code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
5. Run the bounded targeted follow-ons for the selected direct proof homes:
   - `./build/test_chol_csc`
   - `./build/test_integration`

### Day 11 Findings

#### 1. The missing family-local CSC proof was exactly one early-rejection regression

Day 10 added the public lifecycle proof and one small CSC supernodal guard.
The strongest remaining gap was a family-local regression that proves the new
guard fires before the supernodal path mutates CSC state.

Day 11 adds that exact proof in `tests/test_chol_csc.c`:

- `test_eliminate_supernodal_rejects_nonpositive_stored_diagonal`

It builds a small CSC input with a stored negative diagonal and proves:

- `chol_csc_eliminate_supernodal(...)` returns `SPARSE_ERR_NOT_SPD`
- the stored diagonal entry remains unchanged at the point of rejection

Interpretation:

- the Day 10 supernodal guard is now directly regression-proven at the
  family-local CSC seam
- the public lifecycle proof in `tests/test_integration.c` is no longer the
  only evidence for that rejection path

#### 2. The touched LU and Cholesky headers now state the shipped early-rejection semantics directly

The highest-value stale wording left after Days 6-10 was not broad workflow
story drift; it was the family-local precondition wording on the touched
wrapper entry points.

Day 11 tightens:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`

The landed wording now says explicitly that invalid pivot, reorder, or backend
enums are rejected before reorder or factor mutation begins.

Interpretation:

- the public header comments now match the shipped Day 6-Day 10 behavior
- callers no longer need to infer the safety property from tests or
  implementation shape alone

#### 3. The Day 11 proof burden stayed bounded to the selected CSC home

No implementation widening was needed on Day 11.

Touched:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `tests/test_chol_csc.c`

Not widened into:

- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_analysis.c`
- `src/sparse_chol_csc_supernodal.c`
- docs/example/benchmark surfaces

Interpretation:

- Day 11 stayed a true compatibility/regression sweep
- Sprint 63 remains a bounded lifecycle-uniformity sprint instead of drifting
  into another implementation phase after the Day 10 semantics batch

### Validation

Ran and passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `./build/test_chol_csc`
- `./build/test_integration`

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 396.79 sec`

Focused retained proof points:

- `test_chol_csc` passed with the new Day 11 CSC regression:
  - `test_eliminate_supernodal_rejects_nonpositive_stored_diagonal`
- `test_integration` stayed clean at `47 / 47`

### Non-Blocking Note

The reviewed CMake rebuild again emitted the existing
`bench_eigs_reuse.c` double-promotion warnings while rebuilding that bench
binary, but the full reviewed path still completed cleanly and passed all
parity gates.

### Day 11 Close

Sprint 63 Day 11 closes the planned compatibility/regression sweep cleanly:

- the missing family-local CSC proof now covers early supernodal rejection on
  a stored non-positive diagonal
- the touched LU and Cholesky headers now state the shipped early-rejection
  semantics directly
- the full reviewed validation path and bounded targeted follow-ons both
  passed from the landed tree

## Day 12

**Objective:** Align the highest-signal caller-facing and maintainer-facing
docs, examples, and benchmark wording with the landed Sprint 63
lifecycle-uniformity story, while keeping the batch bounded and docs-only.

### Commands Run

1. Re-read the Day 12 plan fence and the Day 11 validated close:
   - `sed -n '412,470p' docs/planning/EPIC_6/SPRINT_63/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_63/artifacts/day11-compatibility-layer-and-regression-sweep.md`
2. Inspect the highest-signal adoption and maintainer surfaces:
   - `sed -n '1,220p' README.md`
   - `sed -n '1,260p' docs/tutorial.md`
   - `sed -n '1,220p' examples/README.md`
   - `sed -n '1,240p' benchmarks/README.md`
   - `sed -n '260,360p' docs/maintainer_guide.md`
3. Land the bounded docs-only follow-through patch:
   - `apply_patch` on:
     - `README.md`
     - `docs/tutorial.md`
     - `examples/README.md`
     - `benchmarks/README.md`
     - `docs/maintainer_guide.md`
4. Run the targeted Day 12 sanity set:
   - `git diff -- README.md docs/tutorial.md examples/README.md benchmarks/README.md docs/maintainer_guide.md`
   - `rg -n "old-factor|nnz drift|example_analysis|bench_refactor_csc|repeated-run direct" README.md docs/tutorial.md examples/README.md benchmarks/README.md docs/maintainer_guide.md`
   - `wc -l README.md docs/tutorial.md examples/README.md benchmarks/README.md docs/maintainer_guide.md`
   - `git status --short --branch`

### Day 12 Findings

#### 1. The top-level repeated-run direct story now says the shipped failure-preserve rule directly

The highest-value missing caller-facing point after Day 11 was in the
top-level repeated-run direct workflow summary in `README.md`.

Day 12 adds the missing explicit rule:

- failed `sparse_refactor_numeric(...)` calls preserve the previous usable
  factor state
- the large-`n` CSC-backed Cholesky lane follows that same rule on same-pattern
  non-SPD failure and on obvious nnz drift rejection

Interpretation:

- the README no longer stops at “same-pattern numeric refresh”
- the repeated-run direct lifecycle now tells callers what happens on the
  highest-signal failure lane, not just on the success lane

#### 2. The tutorial now teaches the repeated-run direct failure contract at the actual adoption point

The strongest bounded tutorial insertion point was the Cholesky section
immediately after the `example_analysis.c` handoff.

Day 12 adds one explicit tutorial note:

- failed same-pattern refactors keep the previous usable factor state intact
- obvious nnz drift is rejected as a lifecycle-contract violation, not treated
  as an implicit rebuild request

Interpretation:

- the tutorial now teaches the Sprint 63 public lifecycle contract where users
  actually move from one-shot direct solves to analyze/factor/refactor reuse
- the new wording stays usage-focused instead of expanding into a maintainer
  policy block

#### 3. Example and benchmark docs now separate adoption proof from error-path proof

The example and benchmark follow-through stayed intentionally small:

- `examples/README.md`
  - `example_analysis` is now called out as the main adoption example for the
    repeated-run direct path, not as the full error-path contract reference
- `benchmarks/README.md`
  - `bench_refactor_csc` is now described as the main throughput/proof surface
    for the large-`n` CSC-backed repeated-run direct lane, while failed
    refactor preservation remains owned by `tests/test_integration.c`

Interpretation:

- the docs no longer blur together:
  - adoption example
  - throughput benchmark
  - failure-contract proof
- the strongest user-facing surfaces now point to the correct proof home
  without widening into general documentation cleanup

#### 4. The maintainer guide now owns the post-Sprint-63 interpretation explicitly

`docs/maintainer_guide.md` now updates the direct-family interpretation from
Sprint 62 to Sprint 63:

- invalid LU pivot/reorder enums and invalid Cholesky reorder/backend enums
  reject before reorder or factor mutation begins
- the public repeated-run direct lifecycle preserves old usable factors on
  refactor failure
- the large-`n` CSC-backed Cholesky lane follows that same old-factor-
  preservation rule on same-pattern non-SPD failure and obvious nnz drift

Interpretation:

- maintainer ownership now matches the landed Day 6-Day 11 implementation and
  proof state
- the remaining deferred queue stays explicit without reopening the sprint

#### 5. The batch stayed bounded and docs-only

Touched:

- `README.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

Not widened into:

- public headers
- implementation
- tests
- benchmarks or examples themselves

Measured touched-surface result:

- `README.md`: `982 -> 988`
- `docs/tutorial.md`: `464 -> 469`
- `examples/README.md`: `142 -> 147`
- `benchmarks/README.md`: `246 -> 249`
- `docs/maintainer_guide.md`: `391 -> 398`

Interpretation:

- Day 12 stayed a true docs/example/benchmark follow-through pass
- Sprint 63 remains ready for final validation without another implementation
  widening

### Day 12 Close

Sprint 63 Day 12 completes the bounded public/maintainer wording follow-through:

- the README and tutorial now state the shipped repeated-run direct
  failure-preserve rule directly
- the example and benchmark docs now separate adoption, throughput, and
  failure-proof roles more cleanly
- the maintainer guide now owns the post-Sprint-63 direct-family
  interpretation explicitly

## Day 13

**Objective:** Revalidate the full Sprint 63 landed state from the strongest
reviewed baseline, then rerun the highest-signal lifecycle, CSC, example, and
benchmark proof surfaces and capture the retained signals.

### Commands Run

1. Re-read the Day 13 validation fence and confirm clean starting state:
   - `sed -n '440,520p' docs/planning/EPIC_6/SPRINT_63/PLAN.md`
   - `tail -n 120 docs/planning/EPIC_6/SPRINT_63/WORKING_NOTES.md`
   - `git status --short --branch`
2. Run the full required validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
3. Run the targeted Sprint 63 rerun set:
   - `./build/test_integration`
   - `./build/test_sparse_lu`
   - `./build/test_cholesky`
   - `./build/test_chol_csc`
   - `./build/test_ldlt`
   - `./build/test_ldlt_csc`
   - `./build/test_iterative`
   - `./build/test_eigs`
   - `./build/test_eigs_lobpcg`
   - `./build/example_analysis`
   - `./build/example_basic_solve`
   - `./build/example_ldlt`
   - `./build/example_iterative`
   - `./build/example_ic_minres`
   - `./build/example_eigs`
   - `./build/example_svd_lowrank`
   - `./build/bench_refactor`
   - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/bench_iterative_reuse`
   - `./build/bench_eigs_reuse`

### Day 13 Findings

#### 1. The strongest reviewed baseline passed end to end without reopening any Sprint 63 code or docs decisions

The full Day 13 gate passed cleanly:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 348.10 sec`

Interpretation:

- Sprint 63 now has a validated branch state from the strongest local reviewed
  baseline
- the branch is ready to close from measured validation rather than inferred
  pass-forward confidence

#### 2. The targeted direct-lifecycle and CSC proof homes all stayed green

The focused Sprint 63 proof surfaces all passed:

- `./build/test_integration` -> `47 / 47`
- `./build/test_sparse_lu` -> `37 / 37`
- `./build/test_cholesky` -> `21 / 21`
- `./build/test_chol_csc` -> `140 / 140`
- `./build/test_ldlt` -> `84 / 84`
- `./build/test_ldlt_csc` -> `96 / 96`
- `./build/test_iterative` -> `79 / 79`
- `./build/test_eigs` -> `30 / 30`
- `./build/test_eigs_lobpcg` -> `26 / 26`

Interpretation:

- the landed Sprint 63 LU and CSC lifecycle work still holds on both:
  - the public repeated-run direct proof surface
  - the family-local CSC proof surface
- no adjacent iterative or eigensolver drift appeared while validating the
  full tree

#### 3. The representative examples still tell the intended adoption story with stable numerical signals

All targeted examples passed:

- `./build/example_analysis`
- `./build/example_basic_solve`
- `./build/example_ldlt`
- `./build/example_iterative`
- `./build/example_ic_minres`
- `./build/example_eigs`
- `./build/example_svd_lowrank`

Representative retained outputs:

- `example_analysis` residual stayed `4.44e-16`
- `example_basic_solve` residual stayed `0.00e+00`
- `example_ldlt` relative residual stayed `1.555e-16`
- `example_iterative`: GMRES `25` iterations unpreconditioned, `9` with ILU(0)
- `example_ic_minres`: MINRES on KKT `42x42` at `39` iterations, Jacobi-MINRES
  at `26`
- `example_eigs`: `nos4` `5 / 5` pairs in `115` Lanczos iterations; KKT
  nearest-sigma `3 / 3` in `6`; explicit `LOBPCG` on `bcsstk04` `3 / 3` in
  `62` outer iterations with residual `8.808e-09`
- `example_svd_lowrank`: sparse low-rank `k=2` kept `22 -> 6` nnz and `3.7x`
  compression

Interpretation:

- the adoption surfaces still demonstrate the intended one-shot versus
  repeated-run split cleanly
- the Sprint 63 lifecycle uniformity changes did not degrade the representative
  caller-facing demos

#### 4. The representative benchmark surfaces kept the expected repeated-run direct and handle-path signals

All targeted benchmarks passed:

- `./build/bench_refactor`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Representative retained outputs:

- `bench_refactor`: `tridiag-200 1.46x`, `tridiag-500 1.27x`, `bcsstk04 1.36x`,
  `nos4 1.45x`
- `bench_refactor_csc nos4`: `speedup_refactor = 1.68x`, residuals
  `8.24e-16` / `7.06e-16`
- `bench_iterative_reuse`: `cg-tridiag-300 1.21x`, `gmres-unsym-220 1.05x`,
  `minres-kkt-42 0.99x`
- `bench_eigs_reuse`: `growm-nos4-k5 1.03x`, `thick-bcsstk14-k5 1.02x`,
  `lobpcg-diag40-k3 1.00x`, with `|lambda|max diff = 0.000e+00`

Interpretation:

- the repeated-run direct benchmark story remains intact after Sprint 63
- the adjacent iterative/eigensolver handle proof surfaces still preserve
  parity and expected reuse behavior

### Non-Blocking Note

The reviewed CMake rebuild again emitted the existing `bench_eigs_reuse.c`
double-promotion warnings while rebuilding that bench binary, but the full
reviewed path still completed cleanly and passed all parity gates.

### Day 13 Close

Sprint 63 Day 13 validates the full landed branch state cleanly:

- the strongest local reviewed baseline passed end to end
- the targeted direct-lifecycle, CSC, example, and benchmark rerun set all
  passed
- Sprint 63 is now ready to close from a validated branch state
