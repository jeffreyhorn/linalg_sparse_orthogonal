# Sprint 62 Working Notes

## Day 1

**Objective:** Turn the Sprint 62 project-plan scope plus the Sprint 61
validated close into a concrete direct-usability implementation starting point
by confirming the preserved reviewed baseline, fixing the strongest live
direct-solver usability hotspots, and making the sprint workstreams explicit
before design or code work begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 62 project-plan source and the new sprint plan:
   - `sed -n '88,117p' docs/planning/EPIC_6/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_62/PLAN.md`
3. Re-read the strongest inherited Sprint 61 closeout source:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_61/RETROSPECTIVE.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_61/artifacts/day14-closeout-and-handoff.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Inventory the strongest current direct-usability seams:
   - `rg -n "analysis|factor|refactor|copy|mutat|lifecycle|one-shot|wrapper|reuse" README.md docs/tutorial.md docs/maintainer_guide.md include/sparse_analysis.h include/sparse_cholesky.h include/sparse_ldlt.h include/sparse_lu.h src/sparse_analysis.c src/sparse_chol_csc.c src/sparse_ldlt.c src/sparse_lu.c tests/test_integration.c tests/test_chol_csc.c tests/test_ldlt.c tests/test_sparse_lu.c`
7. Measure the main Sprint 62 public/direct/docs/proof hotspots:
   - `wc -l README.md docs/tutorial.md docs/maintainer_guide.md include/sparse_analysis.h include/sparse_cholesky.h include/sparse_ldlt.h include/sparse_lu.h src/sparse_analysis.c src/sparse_chol_csc.c src/sparse_ldlt.c src/sparse_lu.c tests/test_integration.c tests/test_chol_csc.c tests/test_ldlt.c tests/test_sparse_lu.c`

### Day 1 Findings

#### 1. Sprint 62 starts from the Sprint 61 validated close, not from renewed configuration or workflow-boundary work

Sprint 61 already closed the first Epic 6 typed-configuration package:

- the main analysis/reorder control plane now has a typed front door
- the compatibility rule is frozen:
  - explicit typed value
  - then legacy compatibility override when unspecified
  - then internal default
- the repeated-run workflow fence is still explicit and unchanged
- the strongest local reviewed baseline is still the authoritative close state

Interpretation:

- Sprint 62 is not another configuration-first sprint
- Sprint 62 is not reopening the repeated-run support boundary
- Sprint 62 is the first bounded Epic 6 direct-usability sprint
- the main Day 1 job is to translate that frozen state into an exact
  one-shot-wrapper and lifecycle-coherence map

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible through the entire direct-usability sprint

The maintained local truth surfaces remain:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the current reviewed wrapper still presents the expected maintained path.

Interpretation:

- Sprint 62 should inherit the exact Sprint 61 truthfulness wording
- later `*.c` / `*.h` landing days should still default to:
  - `make format`
  - `make lint`
  - `make test`
- substantial direct-control or lifecycle-sensitive days should still treat
  `make quality-review-full` as the stronger default

#### 3. The broad Epic 6 direct-usability claim is already concentrated in one-shot direct wrappers and lifecycle-adjacent helper seams

The live repo now shows the strongest current usability pressure clustered in:

- one-shot direct caller stories and wording:
  - `README.md`
  - `docs/tutorial.md`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_lu.h`
- lifecycle and factor-state translation seams:
  - `include/sparse_analysis.h`
  - `src/sparse_analysis.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_lu.c`
- proof surfaces that already carry the mutation/reuse burden:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_sparse_lu.c`

Interpretation:

- Sprint 62 should not pretend direct usability is equally spread across every
  solver family
- the highest-value first batch is one-shot direct wrapper hardening plus
  lifecycle-story clarification
- the existing explicit `analysis` / `factors` lifecycle remains the stable
  ownership model rather than something Sprint 62 should blur away

#### 4. Sprint 62 reduces cleanly to seven bounded implementation workstreams

The project-plan scope collapses to:

1. direct-usability audit
2. lifecycle/wrapper coherence design
3. one-shot hardening
4. explicit lifecycle convergence
5. example/docs adoption
6. regression expansion
7. validation and closeout

Interpretation:

- the Sprint 62 implementation order is already smaller and clearer than a
  generic “direct usability improvement” description suggests
- the right Day 1 deliverable is not “solve direct usability”
- the right Day 1 deliverable is a bounded implementation map with a fixed
  safety and non-goal fence

#### 5. The strongest likely Sprint 62 touch surfaces are now explicit from the live tree

The highest-value current Sprint 62 surfaces are:

- caller-facing docs and lifecycle explanation surfaces:
  - `README.md` = `982`
  - `docs/tutorial.md` = `454`
  - `docs/maintainer_guide.md` = `367`
- public direct and lifecycle headers:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_lu.h` = `337`
- strongest implementation/lifecycle seams:
  - `src/sparse_analysis.c` = `1020`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_lu.c` = `1040`
- strongest proof surfaces likely to matter in Phase 1:
  - `tests/test_integration.c` = `1976`
  - `tests/test_chol_csc.c` = `4552`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_sparse_lu.c` = `908`

Interpretation:

- the early code pressure is concentrated enough to support a bounded first
  landing
- the strongest proof pressure is already split between:
  - public lifecycle regression
  - direct solver family mutation/reuse proof
- docs should follow the landed direct-usage story rather than lead it

#### 6. Sprint 62 needs an explicit Day 1 non-goal fence before any direct-usability design begins

The preserved non-goal fence for Sprint 62 is:

- no reopening the repeated-run workflow fence
- no broad configuration-surface rewrite in the same batch
- no packaging/platform widening disguised as usability work
- no fake convergence between one-shot and lifecycle direct APIs that breaks
  explicit ownership or compatibility
- no broad backend/AUTO-policy work unless a direct-usability landing proves it
  is actually blocking the direct control plane

Interpretation:

- Sprint 62 should make the existing direct-solver story safer and clearer
- compatibility behavior should tighten without pretending the explicit
  lifecycle no longer matters
- success is more coherent one-shot and lifecycle behavior, not erasing the
  distinction between them

### Day 1 Close

Sprint 62 now starts from one explicit direct-usability baseline:

- the Sprint 61 typed-configuration close remains frozen and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 direct-usability claim has already narrowed to one-shot
  wrapper, lifecycle, and factor-state seams
- the public/implementation/docs/proof hotspots for the first hardening batch
  are explicit
- the next step is to turn that live direct-usability map into a ranked
  one-shot and lifecycle coherence design before code changes begin

## Day 2

**Objective:** Freeze the validation and truthfulness baseline that Sprint 62
direct-usability implementation work must preserve before the sprint moves
into the deeper one-shot wrapper and lifecycle coherence audit.

### Commands Run

1. Confirm branch cleanliness before the Day 2 pass:
   - `git status --short --branch`
2. Re-read the current Sprint 62 notes plus the Day 2 plan slice:
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_62/WORKING_NOTES.md`
   - `sed -n '85,150p' docs/planning/EPIC_6/SPRINT_62/PLAN.md`
3. Re-read the strongest inherited Day 2 shape from Sprint 61:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_61/artifacts/day2-validation-baseline-and-touched-surface-recheck.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Re-read the current quality/truthfulness wording:
   - `sed -n '1,220p' README.md`
   - `sed -n '1,260p' docs/maintainer_guide.md`
   - `rg -n "quality-review-full|quality-review-cmake|deadcode|Windows|macOS|Linux|coverage" README.md docs/maintainer_guide.md Makefile .github/workflows`
7. Confirm the Sprint 62 targeted rerun-set presence in the live build tree:
   - `for f in ./build/test_integration ./build/test_chol_csc ./build/test_ldlt_csc ./build/test_cholesky ./build/test_ldlt ./build/test_sparse_lu ./build/test_iterative ./build/test_eigs ./build/test_eigs_lobpcg ./build/example_analysis ./build/example_basic_solve ./build/example_ldlt ./build/example_iterative ./build/example_ic_minres ./build/example_eigs ./build/example_svd_lowrank ./build/bench_refactor ./build/bench_refactor_csc ./build/bench_iterative_reuse ./build/bench_eigs_reuse; do [ -e "$f" ] && echo "$f"; done`

### Day 2 Findings

#### 1. The strongest local reviewed baseline is still `make quality-review-full`

Sprint 62 inherits the same authoritative local validation command as the
Sprint 61 close state:

- `make quality-review-full`

That remains the strongest local reviewed baseline because it preserves both:

- the reviewed Makefile path
- the reviewed CMake parity path

This should remain the top-level local trust anchor unless a later Epic 6
implementation sprint proves the contract itself must change.

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
- stronger default for substantial direct-control or lifecycle-sensitive work:
  - `make quality-review-full`
- docs-only days:
  - no automatic code-quality gate required
  - use targeted sanity checks instead

This remains consistent with the repo’s current Sprint 61 close discipline and
does not need reinterpretation on Sprint 62 Day 2.

#### 4. The current quality/platform story is coherent across README, maintainer guide, Makefile, and workflows

The main maintained surfaces still agree on the current contract:

- Linux remains the enforced reviewed source-of-truth path
- macOS remains reviewed but narrower, with dead-code still staged
- Windows keeps the reviewed CMake subset enforced while the Makefile reviewed
  wrappers and dead-code stay staged
- coverage remains a supplemental signal, not an active reviewed-baseline
  residual
- dead-code remains operationally serialized and separate from `lint` and
  `test`

That means Sprint 62 can proceed from a stable truthfulness contract rather
than needing a wording-reconciliation batch just to start direct-usability
implementation work.

#### 5. The targeted Sprint 62 rerun set is present and now aligned to the actual direct caller-risk surface

The confirmed rerun set is:

- direct lifecycle and integration proofs:
  - `./build/test_integration`
- direct solver family proofs:
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_sparse_lu`
- adjacent repeated-run solver proofs that should not drift:
  - `./build/test_iterative`
  - `./build/test_eigs`
  - `./build/test_eigs_lobpcg`
- representative direct and adjacent examples:
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

- one-shot direct wrapper hardening
- lifecycle/factor-state compatibility verification
- representative workflow-example sanity checks after direct behavior changes
- adjacent repeated-run regression verification so Sprint 62 does not widen
  solver-support boundaries by accident

### Authoritative Day 2 Validation Boundary

- docs-only days:
  - use targeted sanity checks, not the full code-day gate by default
- bounded `*.c` / `*.h` days:
  - run:
    - `make format`
    - `make lint`
    - `make test`
- substantial direct-control or lifecycle-sensitive code days:
  - prefer:
    - `make quality-review-full`
  - and refresh representative proof/benchmark/example surfaces as needed

### Day 2 Exit State

Sprint 62 now has a written validation baseline that matches the live repo:

- strongest local reviewed baseline unchanged
- reviewed CMake parity anchor unchanged
- rerun set fixed from the current build tree around direct lifecycle,
  one-shot direct solver, and adjacent repeated-run regression surfaces
- docs-only versus code-day versus stronger-review path split fixed explicitly
- no contradiction across the main quality/truthfulness surfaces

## Day 3

**Objective:** Re-rank the live one-shot direct-solver pain points by mutation
surprise, lifecycle ambiguity, cancellation/progress behavior, and caller-risk
so Sprint 62 can move into design from a concrete ranked map rather than a
generic usability claim.

### Commands Run

1. Confirm branch cleanliness before the Day 3 audit:
   - `git status --short --branch`
2. Re-read the current Sprint 62 notes and the Day 3 plan slice:
   - `sed -n '1,420p' docs/planning/EPIC_6/SPRINT_62/WORKING_NOTES.md`
   - `sed -n '151,240p' docs/planning/EPIC_6/SPRINT_62/PLAN.md`
3. Inventory the strongest current direct-usage seams:
   - `rg -n "sparse_(lu_factor|chol|cholesky|ldlt|qr)|factor_norm|refactor|analysis|progress_cb|cancel|in-place|bit-identical|copy\\(|mutable|mutates|input matrix|one-shot|wrapper" README.md docs/tutorial.md include/sparse_analysis.h include/sparse_cholesky.h include/sparse_ldlt.h include/sparse_lu.h include/sparse_qr.h src/sparse_analysis.c src/sparse_chol_csc.c src/sparse_ldlt.c src/sparse_lu.c src/sparse_qr.c tests/test_integration.c tests/test_chol_csc.c tests/test_ldlt.c tests/test_sparse_lu.c tests/test_qr.c`
4. Re-read the direct public contract surfaces:
   - `sed -n '1,320p' include/sparse_lu.h`
   - `sed -n '1,260p' include/sparse_cholesky.h`
   - `sed -n '1,360p' include/sparse_ldlt.h`
   - `sed -n '1,260p' include/sparse_qr.h`
5. Re-read the strongest implementation seams:
   - `sed -n '1,120p' src/sparse_lu.c`
   - `sed -n '330,470p' src/sparse_lu.c`
   - `sed -n '1,260p' src/sparse_analysis.c`
6. Re-read the strongest proof and teaching surfaces:
   - `sed -n '740,960p' tests/test_integration.c`
   - `sed -n '1,260p' docs/tutorial.md`
   - `rg -n "test_.*(wrapper|one_shot|lifecycle|cancel|copy|preserve|matches_explicit_analysis_path|matches_default_opts|refactor_same_pattern|invalidates_old_factor)" tests/test_integration.c tests/test_chol_csc.c tests/test_ldlt.c tests/test_sparse_lu.c tests/test_qr.c`

### Day 3 Findings

#### 1. The broad Sprint 62 “direct usability” problem now reduces to four concrete pain-point classes

The strongest live direct-usability pressure clusters into:

1. mutable-matrix surprise on one-shot in-place paths
2. wrapper versus explicit lifecycle ambiguity
3. cancellation/progress semantics that differ by solver family
4. copy-discipline and “fresh original matrix” friction in docs/examples

Interpretation:

- Sprint 62 does not need a vague product-usability sweep
- it needs to tighten a small set of caller-risk seams that already recur
  across direct docs, headers, wrappers, and integration proofs

#### 2. LU is the strongest first hardening target

LU is now the clearest first target because it has the most mixed public story:

- it is still a first-class one-shot in-place surface
- `sparse_lu_factor_opts(...)` can silently route through the shared
  `analysis` / `factors` lifecycle when the option shape matches the shared
  lifecycle fast-path criteria
- reorder can mutate matrix layout before factorization proper begins
- cancellation semantics are more nuanced than the other direct families:
  - cancel-at-step-0 can preserve some compatibility mirrors
  - reordered one-shot entry still does not promise a bit-identical matrix on
    every cancellation path

Interpretation:

- LU has the highest wrapper/lifecycle ambiguity
- LU has the highest “simple one-shot API, but subtle actual behavior” risk
- LU is the strongest first Sprint 62 hardening seam

#### 3. Cholesky is the strongest second target, but for a different reason than LU

Cholesky is already more explicit than LU about being a copied-matrix one-shot
surface, but it still carries meaningful usability risk:

- the matrix is always mutated in place
- the upper triangle is stripped during factorization
- the CSC/linked-list backend split adds behavior detail to the same public
  wrapper
- cancellation semantics are family-specific and not bit-identical to the
  pre-call state

Interpretation:

- Cholesky’s main risk is mutation surprise and backend-behavior opacity, not
  lifecycle mixing
- it is the strongest second target after LU
- Sprint 62 should not treat “direct solvers” as one homogeneous usability
  problem

#### 4. LDL^T is cleaner than the Epic 6 review summary implied

LDL^T still matters to the Sprint 62 story, but it is not the strongest first
landing target:

- the main family-local surface uses an owned `sparse_ldlt_t`, not in-place
  matrix mutation
- the header already states the distinction between the family-local factor
  object and the shared repeated-run direct lifecycle
- cancellation behavior is simpler because the input matrix is never mutated
- the strongest remaining risk is coherence with the shared lifecycle story,
  not one-shot mutation surprise

Interpretation:

- LDL^T should stay in the audit and later regression/design set
- it should not define the first hardening batch
- the repo is already closer to a coherent LDL^T usability story than the
  top-level Epic 6 review suggested

#### 5. QR matters mainly as a contrast surface, not as the defining Sprint 62 landing target

QR still intersects the direct-usage story where caller expectations can drift:

- it requires an unfactored, unreordered matrix with identity permutations
- docs already tell callers to start from a fresh `sparse_copy()` when the
  matrix may have been reused elsewhere
- it has cancellation/progress semantics, but not the same shared
  direct-lifecycle convergence story as LU/Cholesky/LDL^T

Interpretation:

- QR belongs in Sprint 62 mainly as a shared-expectation comparison surface
- it should not be the main Batch 1 usability target
- forcing QR into the first landing would spread the sprint too widely for too
  little caller-value gain

#### 6. The strongest current proof burden already sits in `tests/test_integration.c`, not in a missing test story

The live proof map is already substantial:

- wrapper/default-parity proofs exist
- explicit-analysis-path parity proofs exist
- lifecycle rejection/preservation proofs exist
- cancellation proofs exist
- one-shot versus lifecycle equivalence already exists for at least one
  Cholesky repeated-run story

Interpretation:

- Sprint 62 is not blocked by missing proof infrastructure
- the stronger need is to rebalance what the public and internal direct
  surfaces promise, then add only the smallest new regression surface needed
  to lock that contract down

### Day 3 Close

Sprint 62 now has a ranked direct-usability map instead of a generic problem
statement:

- strongest first target:
  - LU one-shot wrapper and lifecycle coherence
- strongest second target:
  - Cholesky one-shot mutation and backend clarity
- later target:
  - LDL^T coherence follow-through
- contrast/deferred surface:
  - QR

The next step is to turn that ranked map into an exact lifecycle/wrapper
coherence design with preserved compatibility rules before code changes begin.

## Day 4

**Objective:** Define the bounded Sprint 62 lifecycle/wrapper hardening model
and the exact preserved compatibility rules before code changes begin, so the
first direct-usability batch lands against a real safety contract instead of a
generic cleanup goal.

### Commands Run

1. Confirm branch cleanliness before the Day 4 design pass:
   - `git status --short --branch`
2. Re-read the current Sprint 62 notes and the Day 4-5 plan slice:
   - `sed -n '1,560p' docs/planning/EPIC_6/SPRINT_62/WORKING_NOTES.md`
   - `sed -n '181,320p' docs/planning/EPIC_6/SPRINT_62/PLAN.md`
3. Re-read a recent Epic 6 design-contract artifact for shape:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_61/artifacts/day4-typed-options-design-and-precedence-contract.md`
4. Re-read the strongest current direct wrapper control flow:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,320p' include/sparse_lu.h`
   - `sed -n '1,260p' include/sparse_cholesky.h`
   - `sed -n '1,360p' include/sparse_ldlt.h`
   - `sed -n '220,340p' src/sparse_cholesky.c`
   - `sed -n '330,470p' src/sparse_lu.c`
   - `sed -n '1040,1135p' src/sparse_ldlt.c`

### Day 4 Findings

#### 1. The repeated-run direct lifecycle remains the canonical reuse contract

Sprint 62 should preserve the current direct ownership split exactly:

- one-shot wrappers remain first-class/default caller entry points
- the explicit repeated-run contract remains:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`
- the one-shot wrappers may internally reuse lifecycle plumbing where they
  already do so, but that does not make them the same public workflow

Interpretation:

- Sprint 62 should improve coherence, not erase the boundary
- the lifecycle API remains the only public analyze-once / factor-many story
- any usability gain must preserve that explicit ownership contract

#### 2. Sprint 62 should reduce surprise by clarifying mutation and state publication, not by hiding it

The strongest design rule is:

- do not silently copy inside one-shot wrappers to “protect” callers
- do not change family-local ownership models
- do tighten preconditions, cleanup, invalidation, and documentation so the
  actual mutation semantics are easier to understand and harder to misuse

Family-specific preserved model:

- LU:
  - caller-owned matrix is still the one-shot factor container
- Cholesky:
  - caller-owned copied matrix is still mutated in place
- LDL^T:
  - family-local owned factor object is still the one-shot result
- QR:
  - unfactored/unreordered original-matrix expectation remains explicit

Interpretation:

- Sprint 62 is not a “transparent copy semantics” sprint
- it is a “less surprising direct ownership semantics” sprint

#### 3. LU should receive the first hardening batch, but only on bounded wrapper/lifecycle seams

The strongest first landing should stay focused on LU:

- wrapper/lifecycle crossover clarity
- reorder-before-factor invalidation clarity
- cancellation and compatibility-mirror cleanup behavior
- preservation of the current one-shot public shape

Recommended Day 6-7 target:

- make the LU one-shot wrapper easier to reason about when:
  - it stays in the one-shot path
  - it routes through the shared lifecycle fast-path
  - it exits early from reorder or cancellation-sensitive control flow

Interpretation:

- Sprint 62 should not start by touching all direct families at once
- LU gives the highest value-to-risk ratio for the first hardening batch

#### 4. Cholesky should be the strongest second batch, not part of the first fence

Cholesky still matters, but its risk is different enough that it should follow
the LU batch instead of mixing into it:

- mutation surprise
- backend clarity
- cancellation caveats

Interpretation:

- Day 4 should keep Cholesky in the design contract
- Day 5 should likely keep Cholesky out of the first exact touched-file fence
- mixing LU and Cholesky too early would blur two different usability
  problems

#### 5. The implementation ownership split is now explicit

Public wrapper behavior should own:

- clearer precondition and mutation wording
- clearer one-shot versus explicit lifecycle positioning
- default-wrapper behavior normalization where the implementation already
  promises equivalence

Internal factor-state hardening should own:

- reorder metadata invalidation/retention discipline
- compatibility-mirror cleanup around early exits
- wrapper/lifecycle fast-path coherence

Lifecycle helper plumbing should own:

- shared helper use only where it reduces ambiguity without widening API
- alignment between one-shot wrappers and explicit lifecycle publish/free rules

Docs/examples should own:

- copy-discipline guidance
- one-shot versus repeated-run workflow choice
- family-specific mutation caveats only where callers actually need them

#### 6. The explicit compatibility contract is now fixed

Sprint 62 should preserve:

- one-shot wrappers remain available as the default/simple entry points
- no new top-level direct lifecycle object
- no broad API rename or removal
- no hidden broadening of repeated-run support boundaries
- no silent semantic change that promises bit-identical no-mutation behavior on
  paths that still mutate caller-owned matrices

Sprint 62 may clarify:

- when one-shot wrappers are the right default
- when explicit lifecycle should replace them
- which state becomes invalid on reorder/cancel/factor transitions
- which wrapper behaviors already match explicit lifecycle outputs on the
  supported path

### Day 4 Close

Sprint 62 now has an explicit lifecycle/wrapper safety contract:

- repeated-run direct lifecycle remains the canonical reuse path
- one-shot wrappers remain first-class/default peer entry points
- Sprint 62 should reduce surprise by tightening mutation/state semantics, not
  by hiding them
- LU is the fixed first implementation target
- Cholesky is the fixed second target
- Day 5 can now define the exact first touched-file fence against this
  preserved contract

## Day 5

**Objective:** Turn the Day 4 lifecycle/wrapper safety contract into an exact
touched-file and API/implementation boundary plan so the first LU hardening
batch stays bounded and does not expand into a broad direct-solver rewrite.

### Commands Run

1. Confirm branch cleanliness before the Day 5 design pass:
   - `git status --short --branch`
2. Re-read the current Sprint 62 notes and the Day 5-8 plan slice:
   - `sed -n '1,760p' docs/planning/EPIC_6/SPRINT_62/WORKING_NOTES.md`
   - `sed -n '241,380p' docs/planning/EPIC_6/SPRINT_62/PLAN.md`
3. Re-read a recent Epic 6 landing-design artifact for shape:
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_61/artifacts/day5-header-and-internal-surface-landing-design.md`
4. Re-read the strongest LU proof/control seams:
   - `sed -n '500,880p' tests/test_integration.c`
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,140p' README.md`
   - `rg -n "reorder_perm|factor_state|publish_factored|require_original_row_col_state|restore_compat_on_premutation_exit|progress_cb|CANCELLED" src/sparse_lu.c src/sparse_matrix_state_internal.h src/sparse_factor_state_internal.c`

### Day 5 Findings

#### 1. The minimum viable public surface for the first batch is LU-only

Public surfaces to touch first:

- `include/sparse_lu.h`

Public surfaces to keep untouched in Batch 1:

- `include/sparse_analysis.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_qr.h`

Public design rule:

- the first batch should normalize the LU one-shot contract in place
- it should not widen the shared lifecycle header
- it should not try to normalize every direct-family header at once

#### 2. The smallest viable implementation bridge stays inside the LU wrapper and factor-state seam

The core implementation lane should stay inside:

- `src/sparse_lu.c`

With helper/state support allowed only if the landed behavior proves it is
necessary:

- `src/sparse_factor_state_internal.c`
- `src/sparse_matrix_state_internal.h`

Why this matters:

- the main Sprint 62 LU risk is wrapper/lifecycle crossover plus state
  invalidation around reorder/cancel paths
- those seams already live in `src/sparse_lu.c` and the factor-state helper
  lane
- widening into `src/sparse_analysis.c` on the first batch would turn the
  usability sprint into a lifecycle-core rewrite too early

#### 3. The first proof home should stay integration-led, with unit-level LU proof only if the landed path forces it

Required proof surface:

- `tests/test_integration.c`

Optional only if the landed change needs tighter family-local proof:

- `tests/test_sparse_lu.c`

Reason:

- the hardest Sprint 62 risks already show up at the integration boundary:
  - cancel
  - reorder invalidation
  - one-shot versus explicit lifecycle parity
- `tests/test_integration.c` is already the authoritative proof home for those
  seams
- a mandatory `test_sparse_lu.c` expansion on Day 6 would widen the batch
  before the actual hardening proves it is needed

#### 4. Docs follow-through should stay out of the Day 6 first patch unless the code change forces a compile-visible contract update

Likely later docs surfaces:

- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`

But the first code batch should avoid starting there unless:

- the LU header wording must move in the same patch for truthfulness
- or the implementation change cannot be described correctly without a matching
  public-header edit

Interpretation:

- Day 6 should prioritize code and proof
- docs should mostly follow the landed LU behavior rather than precede it

#### 5. The Day 6 versus Day 7 split is now explicit

Day 6 target:

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`

Day 6 focus:

- public wrapper wording normalization where needed
- first LU wrapper/lifecycle hardening slice
- first bounded regression additions for the exact landed path

Day 7 optional/support set:

- `src/sparse_factor_state_internal.c`
- `src/sparse_matrix_state_internal.h`
- `tests/test_sparse_lu.c`
- small LU comment/wording follow-through in touched files only

Day 7 focus:

- cleanup/state-preservation tightening if the Day 6 landing proves it is
  needed
- helper hardening only where it directly supports the LU batch
- regression expansion only where the Day 6 behavior exposed a real proof gap

#### 6. The exact non-touch set for the first landing is now fixed

Do not widen the first batch into:

- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt.c`
- `src/sparse_analysis.c`
- `src/sparse_qr.c`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_qr.h`
- broad docs simplification
- benchmark/example edits
- packaging/platform work
- configuration-surface work

### Day 5 Close

Sprint 62 now has a precise first implementation boundary:

- the minimum viable public surface is fixed to `include/sparse_lu.h`
- the primary implementation seam is fixed to `src/sparse_lu.c`
- the required proof home is fixed to `tests/test_integration.c`
- the helper/state support lane is bounded and optional rather than assumed
- the Day 6 versus Day 7 split is fixed
- the non-touch set is fixed before public-header or implementation edits begin

## Day 6

**Objective:** Land the first bounded LU one-shot hardening slice by making
the one-shot wrapper contract reject reused matrix state earlier and more
explicitly, while preserving the explicit `analysis` / `factors` lifecycle as
the canonical repeated-run path.

### Commands Run

1. Confirm branch cleanliness before the Day 6 landing:
   - `git status --short --branch`
2. Re-read the current Sprint 62 notes plus the Day 6 plan slice:
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_62/WORKING_NOTES.md`
   - `sed -n '220,280p' docs/planning/EPIC_6/SPRINT_62/PLAN.md`
3. Re-read the Day 5 landing design:
   - `sed -n '1,240p' docs/planning/EPIC_6/SPRINT_62/artifacts/day5-lu-landing-design.md`
4. Inspect the live LU wrapper and matrix-state seams:
   - `rg -n "sparse_lu_factor|sparse_lu_factor_opts|require_original_row_col_state|reorder|progress" include/sparse_lu.h src/sparse_lu.c src/sparse_matrix_state_internal.h tests/test_integration.c`
5. Land the Day 6 code batch:
   - `apply_patch` on:
     - `include/sparse_lu.h`
     - `src/sparse_lu.c`
     - `tests/test_integration.c`
6. Run the required code-day gate:
   - `make format`
   - `make lint`
   - `make test`
7. Run the stronger reviewed baseline for this lifecycle-sensitive change:
   - `make quality-review-full`

### Day 6 Findings

#### 1. LU one-shot wrappers now reject reused matrix state up front instead of re-entering the wrapper path on an already reordered/factored matrix

The landed LU hardening slice adds
`sparse_matrix_require_original_row_col_state(mat)` to:

- `sparse_lu_factor(...)`
- `sparse_lu_factor_opts(...)`

Interpretation:

- LU now matches the intended one-shot contract more closely
- callers no longer get a second wrapper attempt on a matrix that has already
  been reordered, pivoted, or factored
- the explicit repeated-run lifecycle remains the supported reuse path rather
  than something LU one-shot wrappers should approximate implicitly

#### 2. The public LU header now states the one-shot versus explicit lifecycle boundary more directly

The touched public header now makes three caller-facing points clearer:

- LU one-shot entry points should be called on a fresh matrix or on a
  `sparse_copy(...)`
- stable-pattern repeated runs belong on the explicit
  `sparse_analysis.h` lifecycle
- reordered/default-compatible LU calls may internally reuse lifecycle
  plumbing, but that does not relax the public one-shot matrix-state contract

Interpretation:

- Sprint 62 Day 6 reduced wrapper/lifecycle ambiguity without widening the
  lifecycle API
- the public story is now stricter and easier to follow
- this is a usability hardening batch, not a hidden copy-semantics batch

#### 3. The Day 6 proof stayed bounded to the required integration home and now proves old-factor preservation explicitly

The integration proof home now carries the renamed regression:

- `test_lu_refactor_attempt_rejects_existing_reordered_factor_and_preserves_old_factor`

That proof now checks:

- a reordered LU factorization succeeds
- a second one-shot LU wrapper call on that same matrix is rejected with
  `SPARSE_ERR_BADARG`
- the previously built LU factor remains usable and produces the same solve
  result

Interpretation:

- the most important Day 6 caller-risk seam is now covered where the public
  lifecycle story already lives
- the batch did not need to widen into `tests/test_sparse_lu.c`
- Day 6 stayed inside the exact Day 5 proof fence

#### 4. The only real Day 6 regression was error-code flattening, and the landed fix preserved the original NULL-path semantics

The first cut of the state guard incorrectly flattened every precondition
failure to `SPARSE_ERR_BADARG`.

That broke:

- `test_lu_null_matrix`

The landed fix returns the exact `state_err` from
`sparse_matrix_require_original_row_col_state(...)`.

Interpretation:

- the hardening change is still bounded and correct
- the important compatibility detail is preserved:
  - `NULL` stays `SPARSE_ERR_NULL`
  - reused row/column state still rejects with `SPARSE_ERR_BADARG`

#### 5. The full Day 6 validation close is clean

The required code-day gate passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also passed:

- `make quality-review-full`

Reviewed anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 222.54 sec`

One non-blocking note remains the same as earlier sprint validation closes:

- the reviewed CMake rebuild emitted ordinary compiler warnings while
  rebuilding `bench_eigs_reuse`, but the full reviewed path still completed
  cleanly and passed all parity gates

### Day 6 Close

Sprint 62 Day 6 landed one coherent first LU hardening slice:

- LU one-shot wrappers now reject reused matrix state explicitly
- the public LU header now states the one-shot versus explicit lifecycle split
  more directly
- the integration proof now checks both rejection and old-factor preservation
- the batch stayed inside the Day 5 touched-file and proof fence
- the full required and reviewed validation close passed from the landed state

## Day 7

**Objective:** Complete the bounded LU one-shot hardening follow-through by
tightening the reordered wrapper publication seam so cancelled or failed
reordered LU one-shot attempts preserve the caller-owned matrix state until a
factorization actually succeeds.

### Commands Run

1. Confirm branch cleanliness before the Day 7 batch:
   - `git status --short --branch`
2. Re-read the Sprint 62 plan slice and Day 6 landed notes:
   - `sed -n '150,230p' docs/planning/EPIC_6/SPRINT_62/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_6/SPRINT_62/WORKING_NOTES.md`
   - `sed -n '1,220p' docs/planning/EPIC_6/SPRINT_62/artifacts/day6-one-shot-hardening-batch1.md`
3. Audit the remaining LU publication and factor-state seam:
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,520p' src/sparse_lu.c`
   - `sed -n '1,260p' src/sparse_factor_state_internal.c`
   - `sed -n '1,260p' src/sparse_matrix_state_internal.h`
   - `sed -n '1,260p' src/sparse_reorder.c`
   - `sed -n '1,2400p' tests/test_integration.c`
4. Land the bounded Day 7 code batch:
   - `include/sparse_lu.h`
   - `src/sparse_lu.c`
   - `tests/test_integration.c`
5. Run the required code-day gate:
   - `make format`
   - `make lint`
   - `make test`
6. Run the stronger reviewed baseline for this lifecycle-sensitive change:
   - `make quality-review-full`

### Day 7 Findings

#### 1. Reordered LU one-shot factorization now publishes back to the caller matrix only on success

The strongest remaining Day 6 seam was not the fresh-matrix guard itself; it
was the reordered one-shot publication boundary inside
`sparse_lu_factor_opts(...)`.

Before the Day 7 batch:

- LU could reorder into a temporary matrix
- steal the reordered payload back into `mat`
- then fail or cancel during numeric factorization

That left the caller-owned matrix partially transformed even though no usable
LU factor had actually been produced.

The landed Day 7 helper:

- `s62_lu_factor_reordered_working_copy(...)`

now keeps reordered LU one-shot work on a temporary copy and only steals the
payload back into `mat` after `sparse_lu_factor_inner(...)` succeeds.

Interpretation:

- cancelled or failed reordered LU one-shot attempts no longer strand the
  caller matrix in an intermediate reordered state
- this tightens the one-shot safety contract without widening the explicit
  repeated-run lifecycle API
- the batch finishes the first LU hardening seam by fixing publication
  timing rather than by adding hidden copy semantics to all direct wrappers

#### 2. The public LU header now describes the reordered one-shot preservation rule truthfully

`include/sparse_lu.h` now states that reordered one-shot LU calls outside the
default-compatible fast path:

- factor a temporary reordered working copy
- publish back to the caller matrix only on success

Interpretation:

- the public LU contract now matches the actual reordered failure/cancel
  behavior
- callers have a clearer rule for what remains untouched after an interrupted
  reordered one-shot attempt
- Sprint 62 still preserves the same larger story:
  - one-shot LU remains first-class/default
  - explicit `analysis` / `factors` remains the canonical stable-pattern
    repeated-run path

#### 3. The Day 7 proof stayed bounded to `test_integration.c` and now covers reordered cancel preservation explicitly

The new integration proof:

- `test_progress_cb_lu_cancel_after_reorder_preserves_original_matrix`

checks that a reordered LU one-shot call cancelled at the first progress step:

- returns `SPARSE_ERR_CANCELLED`
- leaves row/column permutation state at identity
- preserves the original tridiagonal matrix entries
- leaves the matrix unfactored for `sparse_lu_solve(...)`
- still allows a later successful reordered LU one-shot retry

Interpretation:

- the strongest caller-risk seam now has direct public-lifecycle coverage
- the proof stayed in the Day 5 required proof home instead of widening into
  `tests/test_sparse_lu.c`
- Day 7 remained a bounded support batch, not a broad direct-family sweep

#### 4. The only Day 7 regression was a test-side retry mistake, not an implementation flaw

The first Day 7 test cut incorrectly reused the same cancelling options for
the “retry succeeds” half of the new integration proof.

That caused the second factorization attempt to cancel again.

The landed fix introduces a separate `retry_opts` object with:

- the same reorder and pivot choices
- no cancellation callback

Interpretation:

- the implementation seam held up under the first validation attempt
- the only failure was in the test harness
- the landed regression proof now matches the intended public caller story

#### 5. The full Day 7 validation close is clean

The required code-day gate passed:

- `make format`
- `make lint`
- `make test`

The stronger reviewed baseline also passed:

- `make quality-review-full`

Reviewed anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 360.27 sec`

One non-blocking note remains explicit:

- the reviewed CMake rebuild again emitted ordinary compiler warnings while
  rebuilding `bench_eigs_reuse`, but the full reviewed path still completed
  cleanly and passed all parity gates

### Day 7 Close

Sprint 62 Day 7 completed the first bounded LU usability package:

- reordered LU one-shot attempts now preserve the caller-owned matrix until
  numeric factorization actually succeeds
- the public LU header now describes that reordered preservation rule
  directly
- the integration proof now covers cancelled reordered one-shot preservation
  plus later successful retry
- the batch stayed inside the Day 5 optional support lane without widening
  into Cholesky, LDL^T, QR, or broad docs/example work
- the full required and reviewed validation close passed from the landed state
