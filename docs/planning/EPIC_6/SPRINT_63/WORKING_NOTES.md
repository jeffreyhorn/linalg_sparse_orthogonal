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
