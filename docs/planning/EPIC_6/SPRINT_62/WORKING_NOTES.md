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
