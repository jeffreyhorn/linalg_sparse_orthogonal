# Sprint 177 Day 12: Sprint Handoff Package

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Requested sprint path:** `docs/planning/EPIC_15/SPRINT_177/`
**Status:** Complete

## Purpose

Prepare actionable handoffs for the first two Epic 16 implementation sprints:
Sprint 178 allocation-failure proof batch 2 and Sprint 179 generated API HTML
publication/status closure. These handoffs are meant to let those sprints
begin without repeating Sprint 177 baseline work.

## Sprint 178 Handoff: Allocation-Failure Proof Batch 2

### Target

Add deterministic allocation-failure cleanup evidence for one additional
high-risk subsystem beyond the existing CG/GMRES/MINRES repeated-run handle
proof.

### Source Inputs

- Day 3 residual: S177-R01.
- Matrix row: ESM-010.
- Gate: Day 8 Gate 1.
- Quality map: Day 10 C/header, C test, Makefile, CMake, and focused
  allocation-failure rows.
- Prior pattern: Sprint 176 focused iterative allocation-failure lane.

### Recommended First Actions

1. Select one subsystem with high allocation ownership and bounded public
   state, such as one matrix construction/conversion path, one direct solver
   setup path, or one decomposition workspace owner.
2. Inventory failure sites and ownership cleanup expectations before editing
   implementation code.
3. Write a cleanup invariant artifact that states no stale public state
   publication, successful retry semantics, and unsupported breadth.
4. Reuse or extend the private allocation hook semantics only as far as needed
   for the selected subsystem.
5. Add focused regression coverage before broadening public wording.

### Owner Files To Inspect First

| Area | Files |
| --- | --- |
| Allocation hook and private semantics | `src/sparse_alloc_internal.c`, `src/sparse_alloc_internal.h` |
| Existing proof pattern | `tests/test_iterative.c`, `src/sparse_iterative.c`, `Makefile`, `CMakeLists.txt` |
| Candidate matrix paths | `src/sparse_matrix.c`, `src/sparse_matrix_build_internal.c`, `src/sparse_csr.c`, public matrix headers |
| Candidate direct solver paths | `src/sparse_lu.c`, `src/sparse_lu_csr.c`, `src/sparse_ldlt*.c`, `src/sparse_qr.c` |
| Public wording | `README.md`, `docs/maintainer_guide.md`, sprint artifacts |

### Required Validation

If C or header files change, run:

```bash
make format && make lint && make test
```

Additional validation depends on the selected subsystem:

- focused new allocation-failure Make/CTest target;
- `make iterative-allocation-failure-gate` as a pattern regression check if
  shared hook behavior changes;
- CMake/CTest registration checks if a new test target or label is added;
- `git diff --check`.

### Pass Criteria

- One named subsystem has deterministic injected allocation-failure coverage.
- Failure paths assert cleanup and no stale public state publication.
- A successful retry after reset is proven.
- The focused validation target is documented.
- README/maintainer wording names the selected subsystem and preserves the
  local/focused support tier.

### Stop Criteria

Stop and ask for direction if:

- failure injection cannot be made deterministic without broad hook redesign;
- cleanup state cannot be observed without changing public API;
- a selected subsystem requires invasive behavior changes beyond one sprint;
- validation fails and the failure is not clearly unrelated;
- docs pressure would require a broad allocation-failure claim.

### Review Traps

- Keep "allocation-failure" terminology consistent.
- Preserve existing public error ordering, especially NULL-handle contracts.
- Do not publish partial handles, partial matrices, or stale result state on
  injected failure.
- Do not imply coverage for direct solvers, eigensolvers, matrix construction,
  package/install flows, generated tooling, or unrelated allocation paths
  unless they are the selected subsystem and fully proven.

## Sprint 179 Handoff: Generated API HTML Publication Decision

### Target

Close generated API HTML product status by selecting and enforcing exactly one
of: hosted publication, retained CI artifact, committed generated output, or
stronger local-only status.

### Source Inputs

- Day 3 residual: S177-R02.
- Matrix rows: ESM-005 and ESM-011.
- Gate: Day 8 Gate 2.
- Quality map: Day 10 generated API docs, documentation-only, workflow, and
  public-header rows.
- Current policy: generated API HTML is local-only, ignored, untracked,
  unstaged, and validated by `make api-docs-freshness`.

### Recommended First Actions

1. Audit current Doxygen inputs, generated output location, ignored paths,
   staging guard, and API navigation wording.
2. Compare feasible product statuses:
   - keep local-only but strengthen guard/docs;
   - upload retained CI artifact;
   - publish hosted generated docs;
   - commit generated output.
3. Select one status in a decision artifact before changing workflows or
   public navigation.
4. Implement only the chosen path and reject ambiguous mixed status.
5. Update public docs to point to the chosen status with adjacent non-claims.

### Owner Files To Inspect First

| Area | Files |
| --- | --- |
| Doxygen generation | `Doxyfile`, `Makefile` docs targets |
| API docs checks | `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh` |
| Public API inputs | `include/*.h`, `docs/api_reference.md` |
| User navigation | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md` |
| Maintainer wording | `docs/maintainer_guide.md` |
| Optional hosted/artifact path | `.github/workflows/*.yml` if publication or retained artifacts are selected |

### Required Validation

For generated API status changes:

```bash
make docs-check
make api-docs-freshness
git diff --check
```

Also run:

- workflow syntax/guard checks if CI artifact or hosted publication changes;
- `make format && make lint && make test` if public headers or C files change;
- declaration-preservation checks if header declarations move.

### Pass Criteria

- One product status is selected and documented.
- Generated output freshness and staging/publication behavior match that
  status.
- Docs navigation points users to the supported API reference path.
- The local-only or hosted/artifact guard fails clearly on stale, missing, or
  accidentally staged output.
- Public wording does not imply broader package, ABI, platform, or release
  evidence.

### Stop Criteria

Stop and ask for direction if:

- hosted publication requires credentials, infrastructure, or branch settings
  not available in the repo;
- generated output would need to be committed without an explicit product
  decision;
- Doxygen warnings or missing pages indicate API documentation regressions;
- generated files are staged unexpectedly after the local-only guard;
- public wording would imply hosted docs before evidence lands.

### Review Traps

- Do not let generated HTML be both ignored/local-only and publicly described
  as hosted or source-controlled.
- Do not cite a source-controlled metadata row as proof that docs were just
  regenerated.
- Keep generated `sparse_version.h` treatment aligned with current configured
  Doxygen input policy.
- Preserve non-claims for dynamic ABI, shared-library, package-manager,
  Windows Makefile/pkg-config parity, hosted documentation publication, and
  completeness beyond configured Doxygen inputs.

## Shared Handoff Rules

- Use the Day 10 quality map to select validation before closeout.
- Record a sprint artifact before broad public wording changes.
- Keep positive claims adjacent to their non-claims.
- If a sprint selects deferral rather than promotion, strengthen the guard and
  public non-claim instead of weakening the boundary.
- If validation fails, stop and treat the failure as a blocker unless it is
  clearly unrelated and documented.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 178 can begin without redoing baseline work | Complete | Handoff names target, owner files, first actions, validation, pass/stop criteria, and review traps. |
| Sprint 179 can begin with a clear publication decision frame | Complete | Handoff enumerates allowed status choices and requires one decision before implementation. |
| Handoffs preserve scoped claim boundaries | Complete | Both handoffs include protected non-claims and stop criteria for overbroad wording. |
