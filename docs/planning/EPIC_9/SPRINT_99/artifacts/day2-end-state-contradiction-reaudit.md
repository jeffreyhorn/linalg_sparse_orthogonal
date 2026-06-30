# Sprint 99 Day 2: End-State Contradiction Re-audit

## Purpose

Day 2 reconstructs the original Epic 9 contradiction classes from Sprint 90
and compares them against the live Sprint 99 tree. The result is a closeout
classification map that separates resolved work, partial resolutions, active
residuals, and deliberate non-claims before the final comparison sweep begins.

## Original Contradiction Classes

Sprint 90 reduced Epic 9 to these durable contradiction classes:

1. Linked-list-first public/product ownership versus compressed-first compute
   reality.
2. Dense/backend maturity ceiling beyond the builtin scalar core and bounded
   platform-specific acceleration seams.
3. Capability breadth ceiling around real-only scalar, bounded solver-family
   breadth, and compile-time index maturity.
4. Runtime/threading and ABI/index follow-through.
5. Large mixed-role implementation hotspot concentration.
6. Giant-test and proof-owner concentration.
7. Sprint-era chronology and historical naming in permanent surfaces.
8. Build, package, and workflow duplication.
9. Insufficient maintained comparison depth.
10. Explicit non-goals that must remain non-claims unless real proof landed.

## Live-Tree Classification

| Class | Sprint 90 contradiction | Live Sprint 99 status | Closeout reading |
|---:|---|---|---|
| 1 | Product model read as linked-list-first rather than compressed-first for direct and interop workflows | Partially resolved | CSR/CSC export and compressed-first construction paths are public and documented; the linked-list shell remains the mutable compatibility owner and still appears in the front-door identity |
| 2 | Backend maturity ceiling beyond scalar builtin and bounded Accelerate seams | Partially resolved | Dense/backend seams and LDLT CSC backend adoption improved; broad portable backend maturity is still residual |
| 3 | Capability breadth ceiling around real-only scalar and bounded solver/eigensolver breadth | Partially resolved | scalar/index seams and selected capability surfaces improved; broad complex and mixed-precision maturity remain non-claims |
| 4 | Runtime/threading and ABI/index follow-through | Partially resolved | runtime-control and benchmark evidence improved; OpenMP remains localized and no universal runtime claim is supported |
| 5 | Large mixed-role source owners | Still active residual | `src/sparse_ldlt_csc.c` is smaller than the Sprint 90 baseline but still large; several direct, QR, eigensolver, SVD, and matrix owners remain mixed-role |
| 6 | Giant-test and proof-owner concentration | Still active residual | `tests/test_chol_csc.c` is much smaller after extraction, but `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `tests/test_integration.c`, and others remain large proof owners |
| 7 | Sprint-era chronology in permanent product/support/proof surfaces | Partially resolved | public docs are cleaner; lower-level tests and implementation comments still contain sprint/day chronology and historical proof names |
| 8 | Build/package/workflow duplication | Partially resolved | source-list manifest/checker and reviewed install/export assertions reduce drift; Make, CMake, and CI remain intentionally separate proof surfaces |
| 9 | Maintained comparison depth too narrow | Partially resolved | Cholesky CSC and LDLT CSC external dense-reference lanes plus reorder/fill calibration exist; broader solver-family and ecosystem comparisons remain residual |
| 10 | Invalid broad claims: fake platform symmetry, broad shared-library maturity, broad complex/mixed precision, benchmark supremacy | Deliberate non-claim | README, INSTALL, workflow comments, benchmark docs, and maintainer guide continue to fence these claims explicitly |

## Evidence Notes

### Product Model

Current evidence:

- `README.md` still opens with the orthogonal linked-list representation.
- `README.md`, `include/sparse_csr.h`, and `include/sparse_matrix.h` now
  describe CSR/CSC compressed-first construction and one-shot direct workflow
  entry paths.
- `include/sparse_matrix.h` continues to define the matrix shell as the public
  compatibility owner while acknowledging bounded compressed-first helpers.

Classification:

- Partially resolved.
- Not a final-fix blocker unless Day 3-5 finds a specific overclaim or stale
  contradiction in public wording.

### Backend Maturity

Current evidence:

- `src/sparse_dense.c`, `src/sparse_ldlt_dense.c`, and backend override/test
  surfaces show a stronger dense/backend seam than the Sprint 90 baseline.
- LDLT CSC proof and benchmark support are materially stronger after Sprints
  92, 96, and 98.
- Public docs still keep acceleration and platform support bounded.

Classification:

- Partially resolved.
- Residual work remains around broader portable backend maturity, but that is
  too large for an unplanned Sprint 99 closeout fix.

### Capability Breadth

Current evidence:

- `include/sparse_types.h` keeps scalar/index aliases explicit.
- `README.md` and maintainer docs still state that scalar widening does not
  imply complex support or broad generic-scalar maturity.
- Solver-family, eigensolver, SVD, QR, and block/reuse surfaces are broader
  than the Epic 9 start, but not a broad general sparse numerical platform.

Classification:

- Partially resolved with deliberate non-claims.
- Broad complex and mixed precision remain intentionally outside the Epic 9
  closeout claim.

### Runtime, Threading, and ABI/Index Follow-through

Current evidence:

- Reviewed runtime and benchmark surfaces are clearer:
  - `make bench-canonical-report`
  - `make bench-reorder-sprint86`
  - `make bench-fast`
- `README.md` and `INSTALL.md` keep OpenMP as optional/localized rather than a
  product-wide runtime model.
- 64-bit index support and ABI surface language are clearer but still bounded.

Classification:

- Partially resolved.
- The final comparison sweep should validate benchmark/reporting commands, but
  no broad runtime claim is available.

### Maintainability and Proof-Owner Concentration

Current evidence:

- Approximate current line counts show several source owners remain large:
  - `src/sparse_ldlt_csc.c`: 2174
  - `src/sparse_lu_csr.c`: 1665
  - `src/sparse_qr.c`: 1563
  - `src/sparse_ldlt.c`: 1535
  - `src/sparse_eigs.c`: 1534
  - `src/sparse_iterative.c`: 1495
  - `src/sparse_matrix.c`: 1355
  - `src/sparse_svd.c`: 1319
- Several giant tests remain:
  - `tests/test_ldlt_csc.c`: 3878
  - `tests/test_integration.c`: 3421
  - `tests/test_qr.c`: 3234
  - `tests/test_ldlt.c`: 2977
  - `tests/test_etree.c`: 2962
  - `tests/test_graph.c`: 2925
  - `tests/test_iterative.c`: 2841
  - `tests/test_chol_csc.c`: 2617

Classification:

- Still active residual.
- Sprint 96 improved the shape materially, including Cholesky CSC proof-owner
  extraction, but Epic 9 should not claim complete hotspot elimination.

### Chronology and Naming

Current evidence:

- Permanent public docs are cleaner after Sprint 95.
- Several tests, comments, and lower-level proof surfaces still carry
  sprint/day chronology.
- Historical planning artifacts appropriately remain historical.

Classification:

- Partially resolved.
- Closeout language should say chronology was reduced and support surfaces were
  cleaned, not that every historical anchor was removed.

### Build, Package, and Workflow Duplication

Current evidence:

- `build-metadata/library_sources.txt` and `scripts/check_library_sources.py`
  now support source-list consistency.
- `make source-list-check` is wired into `quality-review-compile`.
- `tests/test_install.sh` and `tests/test_cmake_install.sh` remain local
  Unix-side package/export proof owners.
- Workflows keep explicit platform asymmetry:
  - Linux is strongest reviewed source of truth.
  - macOS is reviewed Apple Clang plus supplemental confidence.
  - Windows is reviewed CMake-first subset, currently expecting 51 CTest
    registrations.

Classification:

- Partially resolved.
- Static-first package story is resolved enough for Epic 9 closeout; full
  shared-library and symmetric platform maturity remain non-claims.

### Maintained Comparison Depth

Current evidence:

- Cholesky CSC external dense-reference proof remains anchored in:
  - `tests/test_chol_csc.c`
  - `tests/chol_external_dense_reference.py`
  - `nos4`
  - `bcsstk04`
- Sprint 98 added LDLT CSC external dense-reference proof:
  - `tests/test_ldlt_csc.c`
  - `tests/ldlt_external_dense_reference.py`
  - `kkt5`
  - `kkt10`
- Sprint 98 bounded reorder/fill calibration around:
  - `make bench-reorder-sprint86`
  - `nnz_L` as primary fill field
  - `reorder_ms` as local context

Classification:

- Partially resolved.
- The final comparison sweep should validate the maintained lanes, but broader
  iterative, eigensolver/LOBPCG, QR, SVD, Matrix Market LDLT, and ecosystem
  runtime comparisons remain residual unless separately designed.

## Deliberate Non-claims to Preserve

Sprint 99 closeout must continue to reject these claims unless later days find
new proof that actually changes them:

- full compressed-first replacement of the linked-list shell
- broad complex-scalar support
- broad mixed-precision support
- broad backend-neutral acceleration maturity
- symmetric Linux/macOS/Windows reviewed parity
- shared-library-first or dynamic-ABI product maturity
- universal best-in-class runtime, solver, or ordering performance
- broad external comparison across every solver family
- coverage-threshold or coverage-quality expansion beyond current supplemental
  coverage ownership

## Candidate Final-Fix Queue

Day 2 found no clear implementation blocker that must be fixed before
closeout. The candidate queue for Days 3-6 is intentionally narrow:

| Candidate | Trigger required before fixing | Likely touch surface |
|---|---|---|
| stale public or maintainer wording | Day 3-5 evidence finds a live overclaim or contradiction | docs only |
| workflow/test-count drift | final CTest or workflow-surface check disagrees with documented counts/scope | workflow or docs |
| install/export proof drift | package proof commands fail or contradict static-first docs | scripts, docs, or build files |
| benchmark/reporting claim drift | selected benchmark/report command output no longer matches docs or guardrails | benchmark docs or scripts |
| residual queue ambiguity | unresolved items are duplicated, stale, or phrased as accidental claims | Sprint 99 artifacts/docs |

## Day 2 Conclusion

Epic 9 is materially advanced relative to the Sprint 90 baseline, but several
original contradiction classes are only partially resolved by design. That is
not itself a closeout blocker because the Sprint 90 target was bounded and
included explicit non-claims. Day 3 should freeze a final comparison scope that
tests the partially resolved classes and prevents Sprint 99 from turning
residual work into unsupported final claims.
