# Sprint 87 Day 3: Release / Package Gap Audit

## Purpose

Reduce Sprint 87's broad packaging and ABI problem to one ranked live
contradiction map so the sprint can choose one bounded product-contract lane
instead of another generic build or release bucket.

## Main Result

Sprint 87's broad package / ABI / consumer problem is now reduced to one
ranked live contradiction map:

- strongest first target:
  - bounded product-matrix design centered on `CMakeLists.txt`,
    `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, and the matching package
    wording in `README.md`, `INSTALL.md`, and
    `docs/maintainer_guide.md`
- strongest second target:
  - bounded consumer-proof expansion centered on `tests/test_install.sh`,
    `tests/test_cmake_install.sh`, and
    `examples/cmake_example/CMakeLists.txt`
- strongest third target:
  - bounded workflow / platform follow-through centered on
    `.github/workflows/macos-ci.yml` and `.github/workflows/windows-ci.yml`
    after the product contract is explicit
- strongest support-only but real target:
  - support-surface alignment across `README.md`, `INSTALL.md`,
    `docs/maintainer_guide.md`, and narrow benchmark/docs wording only where
    landed package work truly changes the contract

## Strongest Current Contradiction

The strongest current contradiction is now explicit:

- the maintained docs consistently say the package surface is static-first and
  does not imply a broad shared-library or dynamic-ABI guarantee
- the live CMake install/export path already emits package-version metadata
  through `SparseConfigVersion.cmake` with `SameMajorVersion` compatibility
- the configure path also accepts `BUILD_SHARED_LIBS=ON` only to continue
  producing a static target

That makes the strongest first Sprint 87 move clear:

- do not jump straight to "enable shared"
- first define the exact product matrix the repo is willing to support
- then make the build/export surface and the docs language match that contract
  cleanly

## Second-Tier Contradictions

### Downstream Consumer Asymmetry

The strongest second contradiction is downstream consumer asymmetry:

- the local proof story is real on Unix:
  - `tests/test_install.sh` proves Make install/uninstall + `pkg-config`
  - `tests/test_cmake_install.sh` proves CMake install/export +
    `find_package(Sparse)`
- but the installed surfaces remain asymmetric:
  - Make installs the archive, headers, and `sparse.pc`
  - CMake installs the archive, headers, exported targets, and package config
- the representative downstream example is CMake-only

This makes consumer-proof expansion real Sprint 87 work, but it still reads as
second after the product-matrix contract is explicit.

### Workflow / Platform Asymmetry

The strongest third contradiction is workflow/platform asymmetry:

- Linux remains the strongest reviewed source of truth, but its package proof
  stays developer-side rather than a separate reviewed CI lane
- macOS carries only a narrower supplemental Make install/`pkg-config`
  confidence path
- Windows keeps the reviewed CMake-first consumer subset and explicitly does
  not claim a separate reviewed install-validation lane

This means workflow follow-through is real Sprint 87 work, but it remains
bounded and must stay behind a truthful product contract.

### Support-Surface Follow-Through

The strongest support-only follow-through remains bounded:

- `README.md` = `1050`
- `INSTALL.md` = `265`
- `docs/maintainer_guide.md` = `726`
- `benchmarks/README.md` = `399`

These remain support-only unless landed package work truly changes the
contract, local proof interpretation, or workflow reading.

## Deferred Claims

Broad product and ABI widening remains lower-value first work:

- no broad shared-library product claim without bounded proof
- no dynamic-ABI promise detached from explicit validation ownership
- no generic build-system rewrite detached from the chosen product contract
- no workflow widening that outruns maintained local proof
- no support-surface churn detached from a real landed packaging seam

## Interpretation

The useful Day 3 clarification is now explicit:

- the best first Sprint 87 move is not generic "improve packaging"
- it is one bounded product-matrix design pass on the static/shared and ABI
  contract
- consumer-proof expansion follows next where local install/export and
  downstream evidence can be strengthened against that contract
- workflow/platform follow-through comes after that where the contract exposes
  a real maintained gap
- support surfaces remain support-only unless implementation truly changes the
  package truth

The Sprint 80 and Sprint 86 carry-forward reading is now fixed:

- Sprint 80 already pushed the repo toward a static-first maintained package
  truth
- Sprint 86 already removed reviewed-runtime as the strongest first-tier Epic
  8 contradiction
- Sprint 87 therefore begins with package-contract truthfulness rather than
  another runtime or maintainability lane

## Exit State

- Sprint 87 now has one ranked live package / ABI / consumer contradiction map
  grounded in the current tree and maintained package contract.
- The first implementation center is fixed to bounded product-matrix design,
  not immediate shared-library widening.
- Later consumer-proof expansion, workflow/platform follow-through, and
  support-surface alignment are explicitly ordered behind that first lane.
