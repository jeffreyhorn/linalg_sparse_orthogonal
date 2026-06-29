# Sprint 95 Day 1: Public-Surface Inventory

## Purpose

Day 1 builds the evidence map for Sprint 95's public narrative cleanup. The goal
is not to rewrite product docs yet. The goal is to identify which permanent
surfaces still read like sprint history, which surfaces own stable public
workflow truth, and which historical documents should remain untouched.

## Starting Truth

Sprint 95 starts from a repo where the accumulated planning record is valuable,
but some of that chronology has leaked into permanent reader-facing surfaces.
The cleanup needs to preserve technical truth while moving historical context
back toward planning artifacts, maintainer-only references, or concise proof
links.

## Scope Split

| Surface class | Cleanup stance |
|---|---|
| Permanent product docs | In scope for audit and later rewrite where chronology or duplicate narrative harms reader understanding. |
| Public headers | In scope when comments become user-visible API documentation or generated docs input. |
| Examples and benchmark drivers | In scope when comments, names, or help text expose sprint chronology to users. |
| Build and support workflow files | In scope where public commands or comments create confusing support, benchmark, or proof ownership. |
| Tests and proof owners | In scope for Day 10-11 naming or regrouping decisions, not for Day 1 rename work. |
| Generated API HTML | Derived output; do not hand-edit directly. Fix source comments first. |
| `docs/planning/**` | Intentionally historical archive; exclude from cleanup except as evidence and handoff input. |

## Initial Public-Surface Inventory

| Surface | Public role | Day 1 evidence | Initial cleanup pressure |
|---|---|---|---|
| `README.md` | Primary adoption front door, capability summary, build/test/install entry, examples and benchmark pointer. | Contains many sprint-era capability and performance sections, including symmetric eigensolver, CSC Cholesky, LDLT, callback, integration-test, warning-baseline, and historical measurement notes. | High. Keep as concise product front door and move detailed chronology behind stable links. |
| `INSTALL.md` | Install, package, platform, validation, and support workflow owner. | Contains useful support and reviewed-platform guidance, but some notes are framed as Sprint 28/29 incidents and inherited thresholds. | High. Preserve operational truth while removing incident-log phrasing. |
| `docs/tutorial.md` | Primary learning path after README. | Needs duplicate-onboarding and workflow-owner review against README and examples. | Medium. Cleanup depends on Day 3 ownership model. |
| `docs/algorithm.md` | Public technical reference for current algorithm behavior. | Heavy sprint-by-sprint chronology across CSC Cholesky, LDLT, AMD, ND, eigensolver, and performance sections. | High. Most likely needs a separate style decision because some historical context may be useful technical provenance. |
| `docs/matrix_market.md` | Focused format/reference documentation. | Included in public doc inventory; no Day 1 high-pressure evidence found compared with README, install, algorithm, and headers. | Low. Recheck during Day 2 ranking. |
| `docs/maintainer_guide.md` | Maintainer workflow, review, release, and support process. | Likely owner for internal proof/review context that should not stay in adoption docs. | Medium. Use as destination for maintainer-only history where appropriate. |
| `examples/README.md` and `examples/*.c` | User-facing usage examples and copy-paste workflows. | Day 1 inventory includes all example entry points for duplicate quick-start and stale-workflow review. | Medium. Rewrite only where comments or descriptions duplicate README/tutorial or expose chronology. |
| `include/*.h` | Public API source and generated API-doc input. | `include/sparse_matrix.h`, `include/sparse_qr.h`, `include/sparse_svd.h`, and `include/sparse_eigs.h` contain sprint/day notes in comments visible to API readers. | High. Public behavior comments should describe stable contracts, not implementation chronology. |
| `benchmarks/README.md` and `benchmarks/*.c` | Performance workflow, benchmark interpretation, and benchmark CLI help. | Benchmark drivers include sprint/day provenance such as Cholesky, eigensolver, reorder, and reuse benchmark history. | Medium. Keep reproducible benchmark ownership, remove unnecessary chronology from public help and descriptions. |
| `Makefile` | Primary local workflow and quality command surface. | Contains sprint-named test targets and sprint/day comments, plus current reviewed quality paths. | Medium. Need Day 3 ownership and Day 10-11 proof naming decisions before changing target names. |
| `CMakeLists.txt` | CMake package/test/build workflow. | Contains sprint/day comments and sprint-named integration tests. | Medium. Any rename work must preserve Makefile/CMake parity. |
| `tests/test_sprint*_integration.c` | Historical cross-feature proof owners. | Test files and suite names expose sprint-number ownership directly. | High for proof naming, but deferred. Renames have validation risk. |

## Obvious Chronology And Duplication Evidence

- `README.md` contains sprint-labeled feature sections and integration history
  such as "Symmetric eigensolvers (Sprint 20)", "CSC Cholesky Speedup (Sprint 17
  + Sprint 18)", "CSC LDL^T (Sprint 17 scaffolding + Sprint 18 native + Sprint
  19 row-adj + supernodal)", sprint-specific callback notes, sprint-specific
  integration-test lists, warning baselines, and end-of-sprint snapshot wording.
- `INSTALL.md` contains platform and coverage guidance tied to Sprint 28/29
  inheritance and Sprint 29 Day 12 threshold decisions.
- `docs/algorithm.md` includes extensive development chronology in otherwise
  permanent technical reference sections for Cholesky, LDLT, AMD, ND, FM,
  spectral bisection, eigensolvers, and benchmark conclusions.
- Public headers include sprint/day notes in comments that can flow into
  generated API documentation, including silent-zero contracts, callback
  behavior, SVD mode behavior, and QR progress/cancellation options.
- Benchmark drivers include sprint/day provenance and option labels, including
  `bench_chol_csc.c`, `bench_eigs.c`, `bench_reorder.c`, and reuse benchmarks.
- Makefile and CMake expose sprint-named integration tests and sprint-era
  comments in build/test orchestration.
- `tests/test_sprint*_integration.c` files preserve proof history directly in
  filenames, suite names, and comments. These are likely valid proof owners, but
  the public product story should not require readers to understand sprint
  numbering.

## Initial Cleanup Queue Seeds

These are not the final Day 2 rankings. They are the Day 1 evidence buckets to
rank next.

| Bucket | Candidate surfaces | Why it matters |
|---|---|---|
| Adoption front door | `README.md`, `docs/tutorial.md`, `examples/README.md` | New readers should see current capabilities and next steps without sprint archaeology. |
| Install and support workflow | `INSTALL.md`, `Makefile`, `docs/maintainer_guide.md` | Support guidance should distinguish user install validation from maintainer proof paths. |
| API narrative | `include/*.h`, generated `docs/api/html/**` | Header comments define public contract language and generated API docs. |
| Technical reference chronology | `docs/algorithm.md`, benchmark docs | Algorithm truth should remain precise without embedding full sprint decision history in every section. |
| Proof-owner naming | `tests/test_sprint*_integration.c`, Makefile, CMake | Rename/regrouping may improve product vocabulary but carries build and parity risk. |
| Benchmark story | `benchmarks/README.md`, `benchmarks/*.c`, README benchmark sections | Benchmark claims need clear ownership and reproducibility without sprint closeout framing. |

## Preserved Fence

- Do not edit planning, retrospective, or sprint artifact history just because it
  contains chronology.
- Do not hand-edit generated API HTML as the source of truth.
- Do not rename tests, targets, examples, or benchmark options during Day 1.
- Do not collapse benchmark or algorithm evidence until Day 2 ranking and Day 3
  audience ownership rules define what belongs in public docs versus maintainer
  artifacts.
- Do not change code behavior as part of the inventory pass.

## Day 1 Result

Sprint 95 now has a concrete public-surface map and initial evidence list. The
next step is to rank the findings by reader impact, truth risk, and validation
risk so rewrite work starts from a prioritized queue instead of broad docs
polish.
