# Sprint 95 Day 4: README Boundary and Rewrite Outline

## Purpose

Day 4 fixes the README boundary before the rewrite lands. The README should be
the concise adoption front door, not a sprint ledger, benchmark report,
maintainer handbook, or install manual.

## README Permanent Responsibilities

The README should own:

- project identity and one-sentence value proposition
- compact current capability story
- first successful local build and solve path
- solver workflow chooser
- short command map for common local actions
- compact API surface map
- short quality/testing pointer
- short install pointer
- links to the owner surfaces for tutorial, examples, benchmarks, install,
  maintainer policy, generated API docs, and planning history

The README should not own:

- sprint-by-sprint implementation history
- long performance tables or benchmark closeout narratives
- benchmark CSV schema or command details
- full reviewed-platform policy
- dead-code workflow interpretation
- full test-suite inventory by historical sprint owner
- platform incident history
- install validation script detail
- generated API reference content

## Current README Boundary Findings

| Current section | Keep in README? | Boundary decision |
|---|---|---|
| `Start Here` | Yes | Keep as the front-door router; make it shorter if Day 5 needs room. |
| `Features` | Yes | Keep current capability summary; remove sprint chronology and very deep benchmark/proof detail. |
| `Choose a Workflow` | Yes | Keep as one of README's strongest owned sections. |
| `Building` | Yes | Keep compact local command map; move long quality/policy interpretation behind links. |
| `Quick Start` | Yes | Keep the first successful direct solve. |
| `Iterative Solver Example` | Maybe | Keep only if it remains compact; otherwise point to examples/tutorial. |
| `Repeated-Run Lifecycle Handles` | Maybe | Keep compact chooser-level explanation; deeper walkthrough belongs in tutorial. |
| `Repeated-Run Direct Workflow` | Maybe | Keep enough to select the workflow; detailed lifecycle explanation belongs in tutorial/examples. |
| `API Overview` | Yes | Keep a concise API map; exact contracts belong in headers. |
| `Performance Characteristics` | Shorten heavily | Replace detailed sprint-era speedup narratives with a compact benchmark summary and links. |
| `Thread Safety` | Yes | Keep current stable user contract; trim if it duplicates headers. |
| `Known Limitations` | Yes | Keep current user-facing limits. |
| `Testing` | Shorten heavily | Keep command summary and link to maintainer guide/tests; remove sprint-named proof ledger from README. |
| `Test Category Policy` | Maybe | Keep only a compact supported opt-in category summary. |
| `Dead-Code Workflow` | Move mostly | Maintainer guide and Makefile own detail; README should link. |
| `Reviewed Local Quality Path` | Move mostly | Keep command names only if useful; maintainer guide owns interpretation. |
| `Cross-Platform CI Contract` | Move mostly | README may summarize supported platform stance; maintainer guide owns matrix interpretation. |
| `Quality Readiness Checklist` | Move | Maintainer guide owns readiness policy. |
| `Maintainer References` | Shorten | Keep one link cluster only. |
| `Project Structure` | Maybe | Keep only if concise and useful to first-time readers. |
| `Installation` | Shorten | Keep compact summary and link to INSTALL. |
| `Documentation` | Yes | Keep compact link list. |
| `License` | Yes | Keep. |

## Proposed Clean README Structure

```text
# linalg_sparse_orthogonal

## Start Here
## Current Capabilities
## Choose a Workflow
## Build and Run
## Quick Start
## API Map
## Performance and Benchmarks
## Thread Safety and Limits
## Testing and Quality
## Installation
## Documentation
## Project Structure
## License
```

## Section Intent

### Start Here

Keep the existing router shape:

- first local solve
- workflow choice
- install/downstream consumer setup
- examples, benchmarks, and maintainer policy links

Avoid duplicating details that appear later.

### Current Capabilities

Collapse `Features` into a current-state capability list:

- sparse matrix representation
- direct solvers
- repeated-run direct lifecycle
- iterative solvers
- eigensolvers
- SVD and dense helpers
- reordering/preconditioning
- Matrix Market and compressed storage interop
- quality and observability features

Remove sprint labels and long benchmark/proof explanations from this section.

### Choose a Workflow

Keep as a concise decision map:

- one-shot direct
- compressed-first one-shot
- stable-pattern repeated direct
- repeated-run iterative handles
- repeated-run eigensolver handles
- examples/tutorial/benchmarks/tests ownership split

### Build and Run

Keep common commands:

- `make`
- `make test`
- `make tooling-build`
- `make lint`
- `make quality-review`
- `make quality-review-full`
- `make bench`
- `make examples`
- `make docs`
- CMake build/install summary

Move command expansion, reviewed-policy interpretation, and readiness checklist
detail to `Makefile` and `docs/maintainer_guide.md`.

### Quick Start

Keep the C snippet for one direct solve if it remains short and correct. Point
to `examples/README.md` and `docs/tutorial.md` for executable follow-through and
fuller walkthroughs.

### API Map

Keep a compact table of public headers and what they own. Exact function
contracts stay in headers and generated API docs.

### Performance and Benchmarks

Replace detailed historical speedup sections with:

- one short paragraph saying benchmark surfaces exist
- a compact list of benchmark families
- links to `benchmarks/README.md`
- optionally one current high-level note that dispatch-backed CSR/CSC paths
  exist

Do not keep sprint-era speedup tables in README unless Day 5 decides one compact
headline table is essential and links to benchmark owner surfaces for detail.

### Thread Safety and Limits

Keep stable user-facing contract and limitations:

- concurrent solve/read behavior
- mutation/factorization caveats
- `SPARSE_MUTEX` caveat if still accurate
- default index width
- in-place factorization
- real-only scalar support

Trim anything that repeats header-level details too deeply.

### Testing and Quality

Keep a compact operator map:

- `make test`
- `make smoke`
- sanitizer commands
- coverage command
- opt-in test environment variables
- link to maintainer guide for reviewed baseline interpretation

Remove the long sprint-named test inventory from README.

### Installation

Keep compact install summary and link to `INSTALL.md`. INSTALL owns platform,
staged install, CMake package, `pkg-config`, and validation-script detail.

### Documentation

Keep link list:

- tutorial
- examples
- benchmarks
- matrix market
- maintainer guide
- generated API docs if present
- planning docs as history, not user workflow

## Move/Delete List

| README content | Day 5 action | Destination or reason |
|---|---|---|
| Sprint labels in feature bullets, especially progress/cancel callbacks. | Rewrite. | Headers own API contract; planning docs own chronology. |
| Symmetric eigensolver sprint references in API overview. | Rewrite. | Current API behavior only; detailed backend contracts in headers. |
| Detailed CSC Cholesky speedup section. | Shorten or move behind link. | Benchmark details belong in `benchmarks/README.md` or planning evidence. |
| Detailed CSC LDLT sprint history. | Shorten or move behind link. | README should summarize current dispatch semantics only. |
| End-of-sprint benchmark snapshot links. | Move/delete from README. | Planning docs preserve historical evidence. |
| Long proof-owner lists under Cholesky/LDLT performance. | Move/delete from README. | Tests and maintainer guide own proof interpretation. |
| Full sprint-named test inventory. | Replace. | Keep compact test command summary; proof naming cleanup happens later. |
| Dead-code workflow details. | Shorten. | Maintainer guide and Makefile own interpretation and execution detail. |
| Reviewed local quality path details. | Shorten. | Maintainer guide owns reviewed-policy meaning; Makefile owns commands. |
| Cross-platform CI contract table. | Shorten or move. | Maintainer guide owns platform interpretation. |
| Quality readiness checklist. | Move/delete from README. | Maintainer guide owns readiness policy. |
| Sprint 30 warning-baseline references. | Move behind maintainer guide link. | Maintainer guide owns warning authority. |
| Installation details repeated after project structure. | Shorten. | INSTALL owns full install story. |

## Claim-Check List Before Day 5 Rewrite

| Claim to preserve | Check against | Risk if skipped |
|---|---|---|
| One-shot direct solves include LU, Cholesky, LDLT, and QR. | Public headers and examples. | README could overstate current entry points. |
| Repeated-run direct lifecycle uses `sparse_analyze`, `sparse_factor_numeric`, `sparse_factor_solve`, and `sparse_refactor_numeric`. | `include/sparse_analysis.h`, tutorial, `example_analysis`. | Workflow chooser could point to stale names. |
| Repeated-run iterative handles are bounded to CG, GMRES, and MINRES. | Public iterative headers, examples, tests. | README could imply BiCGSTAB/block handles exist. |
| Repeated-run eigensolver handles cover grow-m Lanczos, thick-restart Lanczos, and explicit LOBPCG. | `include/sparse_eigs.h`, examples, benchmarks. | README could overstate reusable eigensolver support. |
| `bench_eigs` modes and benchmark family names remain current. | `benchmarks/README.md`, benchmark source help. | Link summary could point to stale CLI. |
| `make tooling-build`, `make quality-review`, and `make quality-review-full` remain current. | Makefile. | Command map could fail or misdescribe reviewed path. |
| CMake install supports `find_package(Sparse)`. | `INSTALL.md`, CMake config files. | Install summary could overstate package support. |
| Testing surface count or exact suite inventory. | CTest discovery or Makefile. | Best avoided in README unless freshly measured. |
| Thread-safety contract. | Public headers and tests. | User-facing concurrency claims are sensitive. |
| Default index width and real-only scalar support. | Public types/config headers. | Limitations must stay exact. |
| Generated API docs path. | `make docs`, `docs/api/html/`. | Documentation link could point to stale output. |

## Day 5 Rewrite Guardrails

- Preserve current behavior claims unless the owner surface proves they are
  stale.
- Prefer deleting detailed chronology over moving it wholesale into another
  public doc.
- Keep README links stable where possible; if headings change, update inbound
  links discovered during Day 5.
- Do not rename test files, Makefile targets, CMake targets, benchmark options,
  or examples as part of the README rewrite.
- If README edits only touch Markdown, code quality checks are not required; run
  Markdown/link sanity checks available in the repo.

## Day 4 Result

The README boundary is fixed: README owns adoption, current capability summary,
workflow choice, compact commands, quick start, and links. Detailed benchmark
history, proof ownership, install mechanics, CI policy, warning authority, and
sprint chronology move behind their owning surfaces or remain in planning
history.
