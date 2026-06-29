# Sprint 95 Day 3: Audience Ownership Model

## Purpose

Day 3 defines the stable audience split and ownership rules for Sprint 95
cleanup. Later rewrite days should use this model to shorten public docs without
losing technical truth or moving maintainer-only context into the wrong place.

## Audience Split

| Audience | Primary question | Start surface | Secondary surfaces |
|---|---|---|---|
| First-time adopters | How do I build once and solve one problem? | `README.md` | `examples/README.md`, `docs/tutorial.md` |
| Solver workflow choosers | Which public API path fits my workload? | `README.md` | `docs/tutorial.md`, public headers |
| API users | What does this function or option promise at call sites? | `include/*.h` | generated API docs, examples |
| Example readers | Which shipped executable shows this usage pattern? | `examples/README.md` | individual `examples/*.c`, tutorial |
| Install and package consumers | How do I install or consume the library downstream? | `INSTALL.md` | `examples/cmake_example/`, README install summary |
| Benchmark and performance readers | Which benchmark command or CSV field answers my question? | `benchmarks/README.md` | benchmark drivers, README summary |
| Maintainers and reviewers | What is the reviewed quality, proof, or policy interpretation? | `docs/maintainer_guide.md` | Makefile, CMake, CI, scripts, tests |
| Historical auditors | Why did this behavior or limit evolve? | `docs/planning/**` | specific linked sprint artifacts |

## Narrative Ownership Map

| Narrative | Owning surface | Allowed support surfaces | Rule |
|---|---|---|---|
| Project identity and compact capability story | `README.md` | tutorial, examples, headers | README states current capabilities and links outward; it does not narrate sprint history. |
| First successful local solve | `README.md` | `examples/README.md` | README owns the first path; examples own executable follow-through. |
| Solver workflow choice | `README.md` | tutorial, examples, headers | README owns the decision map; tutorial owns the longer walkthrough. |
| Full learning path | `docs/tutorial.md` | examples, headers | Tutorial explains sequence and rationale; it should not duplicate every README feature list. |
| Example selection and local example behavior | `examples/README.md` | individual examples, tutorial | Examples README maps binaries to workflows and keeps proof/benchmark policy short. |
| API-local contracts | `include/*.h` | generated API docs, examples | Headers describe stable behavior, parameters, errors, caveats, and compatibility. |
| Generated API reference | `docs/api/html/**` | headers | Generated output follows source comments; do not hand-edit as the owner. |
| Operational install and package setup | `INSTALL.md` | README install summary, CMake example | INSTALL owns setup, staged installs, downstream consumers, and install validation. |
| Reviewed platform and quality interpretation | `docs/maintainer_guide.md` | Makefile, CI, scripts | Maintainer guide owns interpretation; executable command detail stays with tooling. |
| Benchmark commands and measurement interpretation | `benchmarks/README.md` | benchmark drivers, Makefile | Benchmark README owns command groups, CSV fields, and measurement caveats. |
| Build/test executable truth | `Makefile`, `CMakeLists.txt`, CI, scripts | maintainer guide | Tooling files own what actually runs; docs explain how to interpret it. |
| Regression, oracle, and property guarantees | tests | README, maintainer guide | Tests own proof; public docs link to product-oriented proof names where needed. |
| Historical provenance | `docs/planning/**` | maintainer guide links | Planning docs preserve chronology; permanent docs link only when provenance is necessary. |

## Surface Responsibilities

### README

The README is the concise adoption front door. It should contain:

- project identity
- the smallest build-and-solve path
- current capability summary
- solver workflow chooser
- compact build/test/install command map
- short links to tutorial, examples, benchmarks, install docs, and maintainer
  guide

It should not contain:

- sprint-by-sprint feature chronology
- long benchmark evidence narratives
- detailed platform incident history
- repeated maintainer policy
- exhaustive proof-owner lists

### Tutorial

The tutorial is the longer user learning path after README. It should contain:

- fuller repeated-run and API walkthroughs
- explanation of matrix-state and ownership expectations
- cross-links to examples and headers where they clarify usage

It should not become:

- a second README feature ledger
- the benchmark proof owner
- the install guide

### Examples

Examples are compact executable usage references. They should contain:

- which binary to run for each common workflow
- small usage-specific caveats needed to understand the example
- links to tutorial for broader explanation

They should not contain:

- broad quality or proof policy
- benchmark interpretation beyond "go to benchmarks"
- sprint provenance

### INSTALL

INSTALL owns operational setup and installed-consumer detail. It should contain:

- prerequisites
- Make and CMake install flows
- static package shape
- downstream `pkg-config` and `find_package(Sparse)` use
- install validation scripts
- platform setup notes

It should not contain:

- first-use solver selection
- benchmark command reference
- repo-wide maintainer policy beyond links
- platform history phrased as sprint incident logs

### Benchmarks

Benchmarks own measurement workflow and interpretation. They should contain:

- benchmark binary groups
- command syntax and target mapping
- CSV schema and report artifact meaning
- current measurement caveats

They should not contain:

- adoption tutorials
- generic support policy
- sprint closeout narrative unless a historical link is necessary

### Maintainer Guide

The maintainer guide owns policy interpretation. It should contain:

- reviewed baseline meaning
- quality-contract interpretation
- proof ownership rules
- documentation ownership rules
- residual and platform interpretation

It should not contain:

- executable command expansion already owned by Makefile/scripts
- first-time user guidance
- benchmark command syntax already owned by benchmarks README

### Public Headers

Public headers own API-local truth. They should contain:

- stable function, option, result, and error contracts
- compatibility notes that affect caller behavior
- caveats that must be visible at call sites

They should not contain:

- sprint/day implementation provenance
- broad proof-owner narratives
- maintainer policy

## Naming And Style Rules

1. Describe current behavior before history.
2. Prefer product behavior names over sprint names on permanent public
   surfaces.
3. Link to proof owners without narrating every sprint that created them.
4. Keep historical context in `docs/planning/**` unless it changes how a user
   should call or validate the library today.
5. Keep each long-form explanation in one owning surface and use short links
   elsewhere.
6. Treat public header comments as API documentation, not internal closeout
   notes.
7. Do not hand-edit generated API HTML; update source comments first.
8. Do not rename public benchmark options, Makefile targets, test files, or
   CMake targets without an explicit compatibility and validation decision.
9. Keep executable command detail with tooling; docs may summarize intent and
   link to the command owner.
10. Use "maintained", "reviewed", and "supported" only where the owning surface
    defines the scope.

## Ownership Header Candidates

These surfaces would benefit from explicit ownership text or shorter versions of
existing ownership text during rewrite days:

| Surface | Needed pattern |
|---|---|
| `README.md` | Short "Start Here" router plus one sentence that README is the front door, not the detailed owner for install, benchmarks, or maintainer policy. |
| `INSTALL.md` | Keep existing operational-setup scope, but remove duplicate support split where README links are enough. |
| `docs/tutorial.md` | Add or preserve a clear learning-path boundary against README and examples. |
| `examples/README.md` | Keep compact example-selection boundary; collapse repeated proof policy into links. |
| `benchmarks/README.md` | Keep benchmark-local command and interpretation boundary; reduce sprint-governance phrasing. |
| `docs/maintainer_guide.md` | Keep as policy-interpretation owner; avoid absorbing every historical detail moved out of public docs. |
| `include/*.h` | No ownership banner needed; comments should read as direct API contracts. |
| `Makefile` and `CMakeLists.txt` | Comments should explain command or target behavior, not sprint provenance. |

## Link Pattern Rules

| From | Link to | Use when |
|---|---|---|
| README | tutorial | The reader needs a fuller API walkthrough. |
| README | examples README | The reader needs executable usage references. |
| README | INSTALL | The reader needs install, package, or downstream consumer detail. |
| README | benchmarks README | The reader needs command syntax, CSV schema, or measurement interpretation. |
| README | maintainer guide | The reader needs reviewed quality or policy interpretation. |
| Tutorial | headers | The reader needs exact API contract detail. |
| Examples README | tutorial | The reader needs broader explanation beyond a small example. |
| INSTALL | maintainer guide | The reader needs reviewed-platform interpretation rather than install mechanics. |
| Benchmarks README | maintainer guide | The reader needs quality-policy interpretation rather than benchmark command usage. |
| Public docs | planning docs | The reader needs provenance for a surprising limit or compatibility decision. |

## Planning-History Rule

Permanent docs may link to planning history when all of these are true:

- the historical decision explains a current limitation, default, compatibility
  behavior, or validation boundary;
- the permanent doc can state the current behavior first;
- a short link is enough for provenance;
- repeating the sprint chronology inline would not help the main audience.

If those conditions are not met, leave the history in `docs/planning/**` and
remove the chronology from the permanent surface.

## Day 3 Result

Sprint 95 now has one ownership model for the rewrite days. The highest-value
next step is to apply this model to the README boundary: keep README as the
adoption router and current capability summary, then push detailed install,
benchmark, support, proof, and historical explanations to their owning surfaces.
