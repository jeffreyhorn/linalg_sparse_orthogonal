# Epic 20 Gap-Closure Checklist (2026-10-19)

Companion to the [code review](review-codex-2026-10-19.md). The order below is
dependency order, not a claim that every historical residual can be closed in
ten sprints. Each funded step names a binary completion gate. `complete`
means the named bounded capability works and its evidence is retained; a
design, guard, or explicit re-deferral alone does not count as implementation.
The [project plan](../PROJECT_PLAN.md) funds steps 1-10 at approximately 140
hours each, leaving up to 28 hours of contingency per 14-day sprint.

## Funded Steps

### 1. Make shared vector diagnostics numerically reliable (Sprint 217)

1. Record `vec_norm2`, `vec_norminf`, and iterative call sites; define finite,
   infinity, NaN, and zero-length behavior in public documentation.
2. Add exponent-stratified finite vectors whose true norms remain
   representable even when naive squares overflow or underflow; add explicit
   NaN/infinity inputs and cancellation-adjacent iterative cases.
3. Implement scaled sum-of-squares for `vec_norm2` and deliberate non-finite
   propagation for both norm helpers. Review relative-residual divisions and
   stopping checks in CG/GMRES/MINRES/BiCGSTAB so invalid norms cannot become
   a successful convergence result.
4. Strengthen `west0067` ILUT coverage with finite-solution and independent
   residual/backward-error checks; if the solver really yields non-finite
   output, fix or fail it explicitly rather than accepting `norm > 0`.
5. Pass focused vector/ILUT/iterative tests, sanitizer checks, `make format`,
   `make lint`, and `make test`. Record the before/after extreme-scale output.

**Done when:** finite representable norms stay finite, non-finite inputs and
solver outputs cannot silently pass, and the ILUT fixture has a meaningful
numerical assertion. This closes one shared arithmetic/diagnostic owner, not
all numerical robustness (review findings 1 and 9).

### 2. Remove the quadratic `sparse_matmul` row flush (Sprint 218)

1. Freeze output semantics: duplicate accumulation, sorted indices,
   zero-drop threshold, ownership, overflow, and failure atomicity. Pin
   narrow/wide/sparse/dense-result matrices and baseline time/memory samples.
2. Build a row-append/compressed temporary output path that does not restart
   a sorted row scan for every touched entry; preserve column traversal or
   convert once under a documented ownership boundary.
3. Test against a simple independent dense oracle on rectangular, cancellation,
   zero, and randomized seeded matrices. Check deterministic ordering.
4. Extend the existing matmul allocation-failure sweep to every new
   allocation/rollback site; ensure `*C == NULL` on failure and inputs remain
   unchanged.
5. Profile at least three pinned shapes and sizes, retain raw timing and peak
   memory samples, and state whether the selected kernel improved. Run full
   C quality, focused CMake/CTest, and sanitizer gates.

**Done when:** the repeated head-scan cost is removed, output semantics and
failure behavior are preserved, and a reproducible measured kernel result is
available. No portable speed claim is unlocked (review finding 3; Epic 19
RQ-17).

### 3. Enforce the installed 32/64-bit index contract (Sprint 219)

1. Name `SPARSE_IDX_BITS` as a validated CMake and Make configuration value;
   reject unsupported widths at configure/build time.
2. Propagate the chosen width to installed CMake target usage requirements and
   `pkg-config` cflags, and expose a package/runtime width check that a
   downstream static consumer can execute before passing `idx_t` arrays.
3. Test clean-prefix installed consumers for both widths; add deliberate
   mismatched-width negative cases and check static archive identity.
4. Run a selected 64-bit solver/IO/conversion suite, then full default-width
   tests on Linux and the relevant hosted install lanes.
5. Update headers, INSTALL, and package docs to say exactly what is tested.

**Done when:** an installed consumer gets the correct width automatically or
fails on mismatch, and both widths have actual build/test evidence. This does
not promise cross-version/shared ABI stability (review finding 4).

### 4. Reduce workflow-guard parser risk without weakening policy (Sprint 220)

1. Inventory existing YAML anchor/alias, flow, folding, quote, escape,
   shell, PowerShell, and publication fixtures in
   `check_api_docs_local_only.sh` and its tests. Freeze expected decisions.
2. Choose a maintained structured YAML parser or a documented, fail-closed
   subset. Parse workflow steps and run scalars structurally while preserving
   step boundaries and effective shell identity.
3. Keep shell-aware command/path analysis, canonicalize path tokens, and
   reject undecidable publication constructions rather than accepting them.
4. Differentially run the old and new guards on every fixture, investigate
   every discrepancy, and add benign-comment plus encoded/bypass regressions.
5. Replace the old pass only after existing and new fixtures, real workflows,
   docs-check, and CI pass; document parser dependency and update procedure.

**Done when:** the local-only publication policy has a smaller structured
ownership surface with no known regression against the frozen fixture corpus.
Generated HTML remains local-only (review finding 7; Epic 19 RQ-19).

### 5. Implement the external evidence v1 acceptance gate (Sprint 221)

1. Freeze the v1 schema from Sprint 215, canonical cell-ID algorithm, status
   vocabulary, allowed units, safe paths, and complete provenance fields.
2. Implement a duplicate-key-rejecting JSON validator with schema, hash,
   relationship, status-transition, and claim-boundary checks.
3. Add valid planning, blocked, and observed fixtures plus negative duplicate,
   missing-hash, path-traversal, noncanonical-ID, false-pass, and unit cases.
4. Integrate the validator self-test into the optional pilot preflight and CI;
   keep it separate from the existing selected-report normalizer.
5. Run test mutations showing a malformed object cannot be accepted even if
   its JSON syntax is valid. Document versioning and raw-artifact retention.

**Done when:** the pilot can pass preflight with a real validator and invalid
or incomplete evidence fails closed. This proves the format, not a solver
result (review finding 2; Epic 19 RQ-01).

### 6. Execute and review one independent CHOLMOD correctness cell (Sprint 222)

1. Pin the exact SPD matrix/RHS and conversion bytes; verify attribution and
   redistribution rights before committing or publishing any derived bundle.
2. Capture project and CHOLMOD versions, compiler/flags, effective ordering,
   threads, loaded libraries and transitive hashes, host, and run identity.
3. Execute both solves; retain raw stdout/stderr, status, solution vectors,
   residual and backward error computed by a shared independent checker.
4. Pass v1 validation on the complete cell, rerun it reproducibly, and retain
   failed attempts as failed records rather than rewriting history.
5. Obtain explicit rights/claim review for the exact bundle; update public
   wording only if approved. Run optional pilot tests and affected docs gates.

**Done when:** one named local cell has a validator-accepted two-sided
correctness result and an explicit publication/retention decision. If rights
are denied, keep the validated result private and do not claim public parity;
that is an external approval blocker, not a technical pass (review finding 2;
Epic 19 RQ-02/RQ-03).

### 7. Close selected Windows Cholesky freshness (Sprint 223)

1. Reproduce the exact selected `cholesky-spd-tridiag-5` job on a current
   commit; inspect run/job identity and the six uploaded files.
2. Verify all rows, hashes, paths, and support metadata against the selected
   manifest and report normalizer.
3. Promote manifest, generator support tier, docs, and non-claim text as one
   change; add negatives for stale/missing/wrong files and contradictory tier.
4. Run local fixture guards and hosted Windows lane on the proposed commit;
   retain artifact IDs and the final claim review.

**Done when:** that exact target's Windows freshness is evidenced and
metadata/wording agree. If hosted evidence is unavailable or mismatched,
promotion is blocked; do not count a workflow-only deferral as closure
(review finding 6; Epic 19 RQ-04).

### 8. Close selected Windows QR incompatible freshness (Sprint 224)

1. Run the current `qr-incompatible-ls` MSVC collection lane and inspect its
   exact six-file uploaded bundle, including diagnostic and path forms.
2. Verify selected row identity and generated files; compare manifest,
   normalizer, workflow and PowerShell guard decisions.
3. Promote target metadata, generator tier, docs, and negative guards together.
4. Repeat hosted and local checks on the final commit and retain run/artifact
   provenance plus claim approval.

**Done when:** the QR target, and only that target, has validated selected
Windows freshness. Missing hosted artifacts block promotion (review finding
6; Epic 19 RQ-05).

### 9. Establish one controlled canonical timing gate (Sprint 225)

1. Select one runner class and the existing `bench_refactor_csc`/`nos4`
   workload; pre-register compiler, backend, warmup, repeat count, affinity or
   host constraints, raw sample retention, and measurement exclusions.
2. Collect a versioned baseline and repeated candidate samples on the same
   class; choose a variance-aware threshold and document its false-alarm risk.
3. Implement report fields and pass/fail checks distinct from freshness;
   retain outliers/invalid trials and exact artifact hashes.
4. Add negative fixtures for stale baseline, one sample, unknown host,
   missing variance, changed workload, and threshold overclaims.
5. Run multiple hosted repetitions and approve only a scoped regression-gate
   statement, not portable speed or Linux/macOS timing parity.

**Done when:** one named CI workload has a reproducible, evidence-backed
timing regression policy (review finding 5; Epic 19 RQ-08).

### 10. Validate one static release candidate and simplify first use (Sprint 226)

1. Select a concrete clean commit and apply the Sprint 214 static release
   checklist; freeze version, changelog, source manifest, license inventory,
   rollback owner, and artifact naming.
2. Build the source archive twice independently and compare hashes; extract
   both and run static installed consumers on every platform to be claimed.
3. Run full source, CMake, install, docs, policy, and relevant hosted checks on
   the exact candidate; retain run/artifact IDs and source-to-install mapping.
4. Exercise a clean-prefix README -> INSTALL -> example/API first-use route;
   shorten duplicated historical status prose without removing boundaries,
   and verify links/public-header coverage.
5. Record candidate-specific sign-off and a precise readiness decision. A tag
   or public release requires explicit owner approval; no shared ABI/provider
   language follows from the static candidate.

**Done when:** one exact candidate has reproducible static source-install
evidence, complete checklist disposition, and a tested first-use route. If an
external approval is pending, label readiness blocked rather than publishing
or claiming a release (review findings 8 and 10; Epic 19 RQ-06/RQ-20).

## Explicitly Not Funded In Epic 20

- Shared-library export/SONAME/install-name/DLL and cross-version ABI
  compatibility (Epic 19 RQ-13): needs a product choice and platform-specific
  loader/consumer program, not merely `BUILD_SHARED_LIBS=ON`.
- Public Homebrew/other provider recipes or binary distribution (RQ-07/RQ-16):
  requires owned provider submission and its own non-local install/uninstall
  evidence after a stable source artifact exists.
- Hosted generated Doxygen HTML and public benchmark report publication
  (RQ-14/RQ-21): each needs an audience, destination, retention, freshness and
  rollback decision; keep the current local-only policy.
- General Windows, external-library, solver-family, platform, timing, memory,
  or state-of-the-art parity (RQ-11/RQ-12/RQ-15/RQ-18): the selected cells and
  lanes above are prerequisites, not representative coverage.
- A repository-wide linked-list-to-compressed architecture rewrite or
  all-solver allocation-failure guarantee (RQ-09/RQ-17): Sprint 218 closes one
  measured kernel only; rank subsequent owners from its evidence.

## Change-Control Rule

For each step, retain exact command, commit, host/toolchain, fixture hashes,
raw artifact, exit status, scope-specific claim, and reviewer decision. A
failed, skipped, unavailable, or unrun check is visible and blocks the related
promotion. Documentation must describe the earned selected scope, not a
planned or neighboring capability.
