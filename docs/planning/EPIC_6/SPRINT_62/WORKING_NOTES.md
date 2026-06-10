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
