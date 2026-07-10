# Sprint 118 Day 7 Product Truth Map Design

## Purpose

Day 7 designs the product truth map that Day 8 will complete. The map will
freeze the current public and maintainer truth for compressed-first workflows,
mutable-shell compatibility, solver-family support, package/platform support,
benchmark evidence, validation/reporting, adoption surfaces, and explicit
non-claims.

The design intentionally separates baseline truth from Epic 11 candidate
claims. Candidate claims remain fenced until an owner sprint implements,
validates, documents, and publicly calibrates them.

## Product-Truth Categories

| Category | Current truth question | Day 8 fill-in expectation |
|---|---|---|
| Compressed-first storage and construction | What is the preferred path when callers already have CSR/CSC data? | Record supported CSR/CSC export/import, compressed-first constructors, and one-shot direct entry routes. |
| Mutable-shell compatibility | What role does the orthogonal linked-list shell still play? | Record supported mutation, traversal, Matrix Market load/save, and compatibility behavior without making it the performance center. |
| Direct solvers | What direct solver families are currently supported and how are repeated-run paths framed? | Record LU, Cholesky, LDLT, QR, CSR/CSC dispatch, repeated direct lifecycle, and current evidence boundaries. |
| Iterative solvers | Which iterative workflows and repeated handles are supported? | Record CG, GMRES, MINRES, BiCGSTAB, block variants, preconditioners, diagnostics, and handle limits. |
| Eigensolver surfaces | What symmetric eigensolver support is current truth? | Record Lanczos, thick restart, LOBPCG, shift-invert, repeated-run handle, and current source-boundary residuals. |
| SVD/QR/rank surfaces | What SVD, QR, pseudoinverse, low-rank, and rank behavior is supported? | Record full/partial SVD, QR/least-squares, rank/condition/pseudoinverse/low-rank evidence and non-parity boundaries. |
| Matrix Market I/O | What file formats and failure modes are supported? | Record coordinate real/pattern/symmetric support, duplicate handling, unsupported features, and SuiteSparse fixture use. |
| Graph and reordering | What reorder/graph methods are supported and what claims are bounded? | Record RCM, AMD, ND, COLAMD, graph partition surfaces, fill/guardrail evidence, and universal-fill non-claims. |
| Package/install/platform | What package and platform support is maintained now? | Record static-first install, `pkg-config`, `find_package`, Linux/macOS/Windows tiers, staged exclusions, and ABI/package-manager non-claims. |
| Benchmark/performance | How should benchmark and sentinel evidence be interpreted? | Record local measurement surfaces, compile-only gates, canonical reports, sentinel/guardrail boundaries, and portable-speed non-claims. |
| Validation/reporting | What validation and report surfaces define current evidence? | Record `make quality-review-full`, CMake parity, source-list, dead-code, coverage/report classification, and expected counts. |
| Adoption/docs | What user-facing routes are currently reliable? | Record README routes, solver-selection, tutorial, examples, install docs, maintainer guide, and known scanability residuals. |
| Explicit non-claims | Which claims must remain unearned until future evidence exists? | Record state-of-the-art replacement, ecosystem parity, every-family external validation, portable speed, dynamic ABI, package-manager, GPU, distributed-memory, and broad precision non-claims. |

## Evidence-Source Inventory

| Evidence source | Truth categories it can support | Notes for Day 8 |
|---|---|---|
| `README.md` | Product identity, compressed-first routes, solver summaries, benchmark caveats, CI/support summary, install summary, non-claims by omission/fences. | Use as public front-door truth, but cross-check against deeper docs before expanding wording. |
| `INSTALL.md` | Package/install/platform, static-first contract, supported platform table, install validation scripts, support boundaries. | Use for package/platform truth and staged install non-claims. |
| `docs/solver_selection.md` | Direct/iterative/eigensolver/SVD/Matrix Market workflow choice. | Use for user-facing solver-family assumptions and workflow routing. |
| `docs/tutorial.md` | Repeated-run and adoption workflows. | Use for supported first-use flows, not for broad claim expansion. |
| `docs/matrix_market.md` | Matrix Market I/O support and unsupported features. | Use for exact file-format truth and unsupported Matrix Market variants. |
| `benchmarks/README.md` | Benchmark command groups, CSV fields, report artifacts, measurement caveats. | Use for local measurement truth and portable-performance fences. |
| `examples/README.md` | Maintained examples and cookbook candidates. | Use for current adoption route inventory and Day 126 handoff candidates. |
| `docs/maintainer_guide.md` | Quality-contract interpretation, reviewed/supplemental/staged lanes, maintainer rules. | Use as maintainer truth for validation and report classification. |
| `include/*.h` | Public API surface. | Use to confirm actual public function/status/options names and ownership contracts. |
| `src/*.c` and `src/*.h` | Implementation truth and source-boundary reality. | Use cautiously in Day 8 only for existing behavior; source movement remains future work. |
| `tests/*.c` | Behavior proof owners, CTest membership, fixture-local evidence. | Use to cite evidence breadth without implying every-family external parity. |
| `benchmarks/*.c` | Benchmark surfaces and local sentinel drivers. | Use for measurement surface inventory only. |
| `examples/*.c` | Adoption examples. | Use to confirm current example coverage. |
| `Makefile` | Local quality, source-list, CMake parity wrappers, benchmark/report, coverage, install targets. | Use for validation/report truth and command names. |
| `CMakeLists.txt` and `cmake/` | CMake build/install/export truth. | Use for CMake package and CTest membership truth. |
| `.github/workflows/*.yml` | CI platform truth. | Use Day 4 classification; do not infer unsupported parity. |
| Sprint 118 Day 2-6 artifacts | Baseline validation, platform truth, residual owner map. | Use as current Sprint 118 evidence spine. |
| Epic 10/Sprint 117 retrospectives | Earned claims, residuals, and non-claims. | Use as final pre-Epic-11 truth baseline. |

## Classification Rules

| Classification | Definition | Required evidence before Day 8 uses it |
|---|---|---|
| Baseline truth | Already implemented, documented, and supported by current tests, validation, or public docs. | Existing public docs plus Day 3 validation or named test/implementation evidence. |
| Baseline truth with caveat | Implemented and supported, but limited by platform, fixture, local measurement, staged lane, or API contract. | Evidence source plus explicit caveat text. |
| Epic 11 candidate claim | Desired future claim scheduled for Sprints 119-127. | Must remain fenced until owner sprint completes implementation/proof/validation/docs. |
| Explicit non-claim | Broad claim intentionally not earned by current evidence. | Day 5-6 residual map, Epic 10 retrospective, Epic 11 review/todo, or Day 4 platform truth. |
| Future-epic deferral | Work outside current Epic 11 scope or likely to remain after owner sprint defers it. | Named future-epic bucket and reason. |
| Unknown or needs audit | Claim needs Day 8 or Day 13 evidence review before disposition. | Mark as unresolved rather than promoting. |

## Candidate Claim Fences

Day 8 must not promote any of these to baseline truth unless a later sprint
actually earns evidence:

- broad state-of-the-art sparse linear algebra replacement;
- SuiteSparse, PETSc, Trilinos, ARPACK/LAPACK, SciPy/NumPy, GraphBLAS, or
  vendor-backend parity;
- every solver family has broad external oracle coverage;
- portable performance superiority;
- universal reorder/fill superiority;
- shared-library package support or dynamic ABI guarantee;
- package-manager support;
- symmetric Linux/macOS/Windows reviewed parity;
- Windows Makefile, install-validation, thread/fuzz/property, or full CTest
  parity;
- GPU support;
- distributed-memory support;
- broad complex or mixed-precision maturity.

## Day 8 Product Truth Map Template

Day 8 should fill the following template, one section per category:

```markdown
## <Category>

| Field | Current truth |
|---|---|
| Baseline truth | ... |
| Evidence sources | ... |
| Caveats | ... |
| Epic 11 candidate claims | ... |
| Explicit non-claims | ... |
| Owner sprint or future owner | ... |
```

Day 8 should also produce a summary table:

```markdown
| Category | Baseline truth? | Candidate claims? | Explicit non-claims? | Owner |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |
```

## Day 8 Completion Checklist

Day 8 is complete when:

- every product-truth category has a baseline-truth entry;
- every entry cites at least one evidence source;
- caveats and staged boundaries are explicit;
- Epic 11 candidate claims remain fenced and assigned to owner sprints;
- explicit non-claims are carried forward without dilution;
- package/platform truth matches Day 4;
- residual owners match Day 6;
- the map can feed Day 13 public-claim drift audit without rediscovering
  evidence sources.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Product-truth category list is complete. | Complete. |
| Evidence-source inventory is recorded. | Complete. |
| Baseline/candidate/non-claim classification rules are defined. | Complete. |
| Day 8 truth-map template is drafted. | Complete. |
| Day 8 completion checklist is ready. | Complete. |
| Candidate claims remain fenced until future proof exists. | Complete. |
