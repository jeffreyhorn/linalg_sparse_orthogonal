# Sprint 95 Day 6: Tutorial and Quick-Start Cleanup

## Purpose

Day 6 aligns the tutorial and example quick-start surfaces with the cleaned
README. The tutorial should be the fuller learning path, and examples should be
compact executable references.

## Changed Surfaces

| Surface | Change | Ownership result |
|---|---|---|
| `docs/tutorial.md` | Added a four-step learning sequence after the opening. | Tutorial owns the fuller path after README. |
| `docs/tutorial.md` | Added install and benchmark owner links in the build section. | INSTALL owns package setup; benchmarks README owns measurement. |
| `docs/tutorial.md` | Shortened Cholesky repeated-run prose that explained benchmark/proof ownership inline. | Tutorial teaches workflow; benchmarks/tests/maintainer docs own proof interpretation. |
| `examples/README.md` | Removed repeated support-split bullets. | Examples README stays focused on executable examples. |
| `examples/README.md` | Replaced proof-heavy wording with measurement/validation owner links. | Benchmarks own measurement; maintainer guide owns quality/validation interpretation. |

## Terminology Alignment

| Term | Day 6 usage |
|---|---|
| README | Front door and workflow chooser. |
| Tutorial | Fuller repeated-run and API learning path. |
| Examples | Runnable usage references and next-step map. |
| Benchmarks | Measurement workflows and retained performance artifacts. |
| Maintainer guide | Quality, validation, reviewed-baseline, and documentation-ownership interpretation. |
| Headers | API-local contracts and exact call-site behavior. |

## Follow-Up Queue

- Day 7 should review `docs/algorithm.md` and other public reference docs for
  chronology that no longer belongs in README or tutorial.
- Day 8 should clean public header comments that still expose sprint history;
  that will require full quality checks because headers are `.h` files.
- Day 9 should revisit examples after header wording is cleaned, especially
  `example_eigs`, `example_iterative`, and `example_analysis` descriptions.
- Day 10-11 proof-owner naming work may require README, tutorial, or examples
  link updates if product-oriented names replace sprint-oriented ones.

## Day 6 Result

The README, tutorial, and examples now have clearer roles:

- README routes readers and summarizes current capabilities.
- Tutorial teaches the fuller workflow sequence.
- Examples show runnable entry points.
- Benchmarks, maintainer docs, headers, and install docs own their specialized
  details.
