# Sprint 215 Day 2: Candidate Baselines And Matrices

## Purpose And Status

Screen independently released sparse-solver baselines and source-controlled
matrix inputs for a reproducible, narrowly scoped future comparison. This is
a **candidate survey**, not the Day 3 baseline decision and not a numerical
or timing result. Repository `scripts/run_external_comparison.py` uses
source-controlled dense helpers; those helpers are not either external
candidate below.

## Versioned External Candidates

| Candidate and source | Acquisition, license and prerequisites | Operation and platform/compiler scope | Local availability and disposition |
| --- | --- | --- | --- |
| [SuiteSparse `v7.12.2` CHOLMOD 5.3.4](https://github.com/DrTimothyAldenDavis/SuiteSparse/releases/tag/v7.12.2), tag commit `42151688813c45846a597edcb601435a0e38f3dd` | Acquire official tagged source at that commit, or record the exact installed Homebrew `suite-sparse 7.12.2` package and linkage. [Tagged build guide](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.12.2/README.md) lists CHOLMOD's AMD/COLAMD/CCOLAMD, BLAS and LAPACK requirements; METIS is optional. [Tagged header](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.12.2/CHOLMOD/Include/cholmod.h) identifies LGPL-2.1+ Cholesky/Utility and GPL-2.0+ optional modules including Supernodal. Module selection and redistribution terms need review. | C `cholmod_analyze` / `cholmod_factorize` / `cholmod_solve` for real sparse SPD factor-and-solve. Upstream CMake documents Unix and MSVC routes; only local macOS x86-64/Apple Clang package discovery was checked today. No Linux/Windows execution or compiler compatibility was tested. | `brew list --versions suite-sparse` returned `7.12.2`; `pkg-config --modversion cholmod` returned `5.3.4`, prefix `/usr/local/Cellar/suite-sparse/7.12.2`, and link flags resolve. Highest-feasibility candidate, **not yet selected**. Installed package version alone does not record build options or prove a working pilot. |
| [SciPy `v1.17.0` sparse LU](https://github.com/scipy/scipy/releases/tag/v1.17.0), tag commit `8c75ae75176236f233824e9a0483c26a69e6dfec` | Acquire [PyPI `scipy==1.17.0`](https://pypi.org/project/scipy/1.17.0/) in an isolated environment. For this host's CPython 3.14/macOS x86-64, the `cp314-cp314-macosx_14_0_x86_64` wheel is listed with SHA-256 `aabf057c632798832f071a8dde013c2e26284043934f53b00489f1773b33527e`; source tarball is another pinned route. PyPI identifies BSD license and Python >=3.11. Pin NumPy version, wheel/BLAS provenance and any bundled dependency details before a pilot. | [`scipy.sparse.linalg.splu`](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.sparse.linalg.splu.html) factors a sparse square CSC input using SuperLU and returns an object with `solve`. PyPI lists macOS, Linux and Windows wheels, but no lane was exercised here. A prebuilt wheel has no compiler choice at install time; its build compiler/flags are not yet recorded. The candidate host is macOS x86-64/CPython 3.14, not a cross-platform claim. | `python3` is 3.14.5; import discovery found neither SciPy nor NumPy. Plausible alternative, but package installation, transitive pinning and LU semantics are unresolved. **Not selected.** |

The tag commits were resolved from upstream GitHub refs on Day 2. Neither
candidate was installed, rebuilt or executed by this survey; the existing
Homebrew CHOLMOD package was only discovered. No project or external solver
output was generated.

## Solver-Operation Mapping

| Candidate operation | Repository counterpart | Mismatch or control requiring Day 3 resolution |
| --- | --- | --- |
| CHOLMOD SPD analyze/factorize/solve | `sparse_cholesky_factor_opts` and `sparse_cholesky_solve` in `include/sparse_cholesky.h`, on a fresh matrix copy. Force `SPARSE_REORDER_NONE` and an explicit linked-list or CSC backend for a named cell. | CHOLMOD may choose an ordering and simplicial/supernodal method; the project's `AUTO` backend changes at `n=100`. Pin both choices and account for symbolic analysis, matrix conversion, factorization, solve, thread/BLAS context and memory ownership separately. Do not compare only function names or `bench_refactor_csc`'s internal speedup ratio. |
| SciPy SuperLU `splu(A).solve(b)` | `sparse_lu_factor_opts` with explicit partial pivot, no reordering and absolute pivot tolerance, then `sparse_lu_solve` in `include/sparse_lu.h`; use a fresh input copy. | SciPy defaults to COLAMD column ordering and exposes `permc_spec`, diagonal pivot threshold and equilibration/refinement options. Repository LU uses its own row/column permutation, drop-tolerance and pivot policy. Set options explicitly, preserve the same float64 matrix/RHS and compare residual/status first; numerical agreement does not imply factor-layout or timing parity. |

QR minimum-norm/least-squares and partial-SVD top-k remain **out of this
Day 2 shortlist**. The current selected rows for those operations are bounded
dense-helper comparisons, not evidence that an external QR/SVD package offers
matching rank thresholds, stopping conditions, vector orientation or timed
phases. No third-party QR/SVD version is selected by omission.

## Candidate Matrix Subset And Provenance

SHA-256 values below are for the exact checked-in file bytes, not a
normalized or expanded matrix. Matrix Market `symmetric` storage must be
expanded once; `general` storage must not be symmetrized by filename.

| Candidate input | Properties and SHA-256 | Provenance/license gate | Intended role |
| --- | --- | --- | --- |
| `tests/data/symmetric_4.mtx` | 4 x 4 real symmetric; 7 stored coordinate entries, 10 after symmetric expansion; diagonally dominant SPD by inspection. `ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335` | Added in repository commit `aa1c04ea` with no upstream attribution in file. Repository `LICENSE` is MIT; confirm authorship/file terms before publishing results. | First CHOLMOD/project correctness-only pilot candidate: small, deterministic, no optional download. |
| `tests/data/tridiagonal_20.mtx` | 20 x 20 real `general`; 58 stored entries; symmetric tridiagonal with diagonal 2 and adjacent -1, hence SPD by construction. `691cae8a5b7b59cbfb43fb7e9a8c2446ce935c9cad4824770a6be2a25982830e` | Added in `aa1c04ea`; no upstream attribution in file; same repository-license/authorship check. | Secondary SPD shape/control check, not a speed-ranking dataset. |
| `tests/data/suitesparse/nos4.mtx` | `HB/nos4`, 100 x 100 real symmetric; 347 stored entries, 594 expanded nonzeros per [collection record](https://sparse.tamu.edu/HB/nos4). `6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80` | File header attributes H. Simon and HB editors; added in repository commit `462020e0`. [Collection terms](https://sparse.tamu.edu/about) state CC-BY 4.0 for matrices and request source citation and metadata preservation. Verify checked-in bytes against upstream and retain attribution before any external publication. | Secondary CHOLMOD/CSC fixture; already used by selected internal canonical benchmark, but that does not make its timing externally comparable. |
| `tests/data/unsymm_5.mtx` | 5 x 5 real `general`; 13 stored entries; listed off-diagonals are paired with equal transposes, so values are symmetric despite the filename. `8877593ef2e566a7fcd6e9243f6c222ed9c64e4a3552d07c9173f28dd56a6c8a` | Added in `aa1c04ea`; no upstream attribution in file; same repository-license/authorship check. | **Excluded** as a genuinely nonsymmetric LU representative. It may test general-format parsing, but not nonsymmetric solver breadth. |
| `lu_nonsym_square_5` inline fixture in `scripts/run_external_comparison.py` | 5 x 5 real, genuinely nonsymmetric entries; not a standalone matrix file, so no matrix-byte hash yet. | Source-controlled script defines the triplets and RHS; repository MIT license applies to the code, subject to normal provenance check. Freeze a canonical Matrix Market or equivalent byte representation and hash it before pilot use. | Potential SciPy LU cell only after exact input materialization; not ready for measurement. |

`tests/corpus/manifests/optional_data.tsv` keeps the optional SuiteSparse
rank-deficient QR subset disabled by default and flags terms for review. It is
not part of this candidate subset and cannot silently become a passed row.

## Pilot-Feasibility Shortlist And Exclusions

1. **Most feasible, not selected:** local macOS SPD factor-and-solve on exact
   `symmetric_4.mtx` bytes, project explicit no-reorder backend versus
   CHOLMOD from SuiteSparse `v7.12.2`. Record one fixed RHS, matrix expansion,
   float64 path, ordering/backend, dependency build, raw outputs and residuals.
   Initial pilot would be correctness-only; any timing requires separate
   methodology and repeated matched measurements.
2. **Conditional alternative:** project one-shot LU versus SciPy 1.17.0
   `splu(...).solve(...)` on a materialized/hash-pinned
   `lu_nonsym_square_5` input. Blocked for now by absent SciPy/NumPy locally,
   unpinned dependency/build provenance and nonidentical pivot/ordering
   defaults.
3. **Secondary scope, not first pilot:** `tridiagonal_20.mtx` and `HB/nos4`
   broaden SPD shape only after the first cell passes. `HB/nos4` additionally
   needs attribution and checked-in/upstream byte review. Its selected
   threshold-free canonical benchmark row cannot be reinterpreted as an
   external performance baseline.

Do not rank the candidates by runtime or claim solver-family superiority from
availability. Even a valid one-cell numerical comparison would not establish
SuiteSparse/SciPy parity, portable performance, platform parity or
state-of-the-art status.

## Day 3 Decision Gates

- Choose one immutable external package/version and one exact matrix/RHS/
  operation cell, or record the precise blocker and executable future route.
- Verify module license/configuration and fixture authorship/attribution;
  capture installed library path, linked dependencies, build flags and
  artifact hash. A matching version string is not enough.
- Freeze float64 input conversion, symmetry expansion, solver backend,
  ordering, pivoting, threads and numerical status before selecting tolerances
  or collecting timings.
- Separate project/CHOLMOD/SciPy correctness checks from Sprint 212 selected
  benchmark freshness, local sentinels and any future performance methodology.

## Validation Performed

Read source-controlled fixture and solver contracts; computed the four file
SHA-256 hashes with `shasum -a 256`; resolved upstream version tags and
primary release, API, license and collection pages; checked local package
discovery with `brew`, `pkg-config` and Python import lookup. No package was
installed, no solver was run, and no C, header, manifest, schema, workflow or
public documentation file was changed. `make docs-check` passed locally with
18 checked-in public headers and matching generated reference/source pages;
trailing-whitespace inspection found no matches in the edited Markdown. The
full C quality gate was not required for planning-only changes.
