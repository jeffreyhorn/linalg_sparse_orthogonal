# Sprint 90 Day 4: Maintainability, Coherence, and Workflow Audit

## Purpose

Reduce Sprint 90's remaining maintainability, documentation-coherence, and
build/workflow-duplication problem to one ranked live contradiction map so
Epic 9 can choose bounded structural cleanup lanes instead of another generic
"split big files and rewrite docs" bucket.

## Main Result

Sprint 90's broad maintainability/coherence/duplication problem is now
reduced to one ranked live contradiction map:

- strongest first target:
  - large mixed-role implementation hotspot reduction centered first on the
    largest direct-family and solver-family owners
- strongest second target:
  - giant-test and proof-owner concentration cleanup centered on the broadest
    family-local and integration proof surfaces
- strongest third target:
  - sprint-era chronology removal from permanent product, maintainer, header,
    benchmark, and test surfaces
- strongest fourth target:
  - build/package/workflow topology convergence where long-term hand-maintained
    duplication is still high
- strongest support-only but real target:
  - public/support surface simplification where the structural cleanup truly
    changes owner, naming, or navigation expectations

## Strongest Current Contradiction

The strongest current contradiction is not simply that some files are large:

- the largest implementation hotspots remain large *and* mixed-role:
  - `src/sparse_ldlt_csc.c` = `2694`
  - `src/sparse_iterative.c` = `1854`
  - `src/sparse_lu_csr.c` = `1665`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_ldlt.c` = `1535`
  - `src/sparse_eigs.c` = `1534`
  - `src/sparse_svd.c` = `1319`
  - `src/sparse_matrix.c` = `1297`
  - `src/sparse_chol_csc.c` = `1279`
- these files still combine algorithm detail, lifecycle policy, helper logic,
  and family-local orchestration in the same retained owners

That fixes the strongest first Epic 9 maintainability move:

- reduce the largest mixed-role implementation owners first
- especially on the families Epic 9 is most likely to touch again:
  - LDL^T CSC
  - iterative solvers
  - QR
  - eigensolvers
  - the matrix shell

## Second-Tier Contradictions

### Giant-Test and Proof-Owner Concentration

The strongest second contradiction is proof concentration:

- `tests/test_chol_csc.c` = `4987`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_qr.c` = `3234`
- `tests/test_integration.c` = `3197`
- `tests/test_etree.c` = `2962`
- `tests/test_graph.c` = `2925`
- `tests/test_ldlt.c` = `2921`
- `tests/test_iterative.c` = `2841`
- `tests/test_reorder_nd.c` = `2340`

The live tree therefore still has broad family-local proof owners that carry
large helper, registration, historical, and policy concentration. This is real
Epic 9 work, but it still reads after the first implementation-hotspot lane
rather than before it.

### Sprint-Era Chronology in Permanent Surfaces

The strongest third contradiction is chronology residue:

- `README.md` still contains many public sprint-era references and historical
  note blocks
- `docs/maintainer_guide.md` still contains many "after Sprint X" or similar
  anchors
- `INSTALL.md`, `benchmarks/README.md`, and several public headers still carry
  sprint/date/history references in permanent product surfaces
- sprint-named proof owners still exist:
  - `tests/test_sprint20_integration.c`
  - `tests/test_sprint19_integration.c`
  - `tests/test_sprint11_integration.c`
  - `tests/test_sprint8_integration.c`
  - `tests/test_sprint6_integration.c`
  - `tests/test_sprint5_integration.c`
- many major family-local tests and benchmark files still explain durable
  behavior through day-by-day sprint chronology rather than through stable
  product-oriented ownership language

Epic 8 improved truthfulness and front-door flow, but the permanent surfaces
still often read like a high-quality planning archive instead of like a
history-light technical product.

### Build, Package, and Workflow Duplication

The strongest fourth contradiction is topology duplication:

- `Makefile` = `908`
- `CMakeLists.txt` = `416`
- `.github/workflows/ci.yml` = `223`
- `.github/workflows/macos-ci.yml` = `104`
- `.github/workflows/windows-ci.yml` = `63`
- `Makefile` still hand-enumerates library sources, test sources, benchmark
  sources, and many workflow/reporting targets
- `CMakeLists.txt` still duplicates the library/test/benchmark/example
  topology separately
- both build systems remain first-class product paths
- workflow truth remains intentionally asymmetric across Linux, macOS, and
  Windows, which is honest but still increases long-term maintenance cost

This remains real Epic 9 work, but it still reads after the largest source and
proof-owner concentration lanes rather than before them.

## Highest-Friction Public and Support Surfaces

The highest-friction public/support surfaces are now explicit:

- `README.md` = `1113`
- `docs/maintainer_guide.md` = `727`
- `docs/tutorial.md` = `473`
- `benchmarks/README.md` = `399`
- `INSTALL.md` = `315`

These are not bad surfaces. They are high-friction because the repo still
asks users and contributors to cross several long owners while also carrying
too much sprint-era chronology inside those permanent references.

## Deferred Maintainability and Coherence Claims

Broad claim widening remains lower-value first work:

- no claim that every large source owner will be fully decomposed in one epic
- no claim that every proof owner will be split into many smaller binaries
- no claim that product docs will become chronology-free without deliberate
  naming and ownership work
- no claim that Make/CMake/workflow duplication can be removed without
  preserving reviewed parity and install/export proof
- no reopening product-model, backend, or capability lanes as if these
  maintainability surfaces now outrank them

## Interpretation

The useful Day 4 clarification is now explicit:

- Epic 9's maintainability and coherence problem is real, but it is not one
  undifferentiated cleanup bucket
- the best first cleanup lane is the largest mixed-role implementation
  concentration
- giant-test and proof-owner concentration follows next
- chronology cleanup follows after that
- build/package/workflow convergence follows after that
- support surfaces remain support-only unless a landed structural cleanup truly
  changes naming, navigation, or owner expectations

## Exit State

- Sprint 90 now has one ranked live maintainability/coherence/duplication map
  grounded in the current tree.
- The strongest residual source hotspots, proof hotspots, chronology residue,
  and duplication surfaces are explicit.
- Day 5 can now define the Epic 9 target state against the full contradiction
  inventory instead of against only the product/backend/capability side.
