# Sprint 95 Day 7: Public Docs Coherence

## Purpose

Day 7 aligns the remaining high-value public docs with the Sprint 95 ownership
model after the README and tutorial cleanup. The goal is current-state public
language, not a full rewrite of every historical technical note.

## Changed Surfaces

| Surface | Change | Ownership result |
|---|---|---|
| `INSTALL.md` | Replaced sprint-incident notes for macOS TSan, Homebrew GCC coverage fallback, and Linux coverage threshold with current operational wording. | INSTALL owns setup and install validation. |
| `INSTALL.md` | Replaced install "proof" wording with "validation" wording. | Maintainer guide owns proof interpretation; INSTALL owns install validation. |
| `benchmarks/README.md` | Replaced several "proof surface" phrases with "measurement surface" and "performance guarantee" wording. | Benchmarks own measurement workflows and emitted fields. |
| `benchmarks/README.md` | Removed sprint-governance labels from benchmark category descriptions while preserving public option names such as `--sprint86-slice`. | Public CLI compatibility stays unchanged. |
| `docs/algorithm.md` | Removed sprint labels from the CSC Cholesky backend heading and supernodal heading. | Algorithm docs start from current technical behavior. |
| `docs/algorithm.md` | Rewrote first Cholesky CSC takeaways to describe current scaling behavior without sprint-day framing. | Historical captures remain linked only as evidence. |

## Updated Audit Queue

| Queue item | Status after Day 7 | Notes |
|---|---|---|
| README front-door overload | Complete for first cleanup batch. | Day 5 landed the main README rewrite. |
| Tutorial and quick-start duplication | Complete for first cleanup batch. | Day 6 aligned tutorial and examples. |
| Install/support workflow repetition | Partially complete. | INSTALL now uses current-state install validation wording; maintainer guide still owns policy. |
| Benchmark narrative cleanup | Partially complete. | Overview/category wording is cleaner; deeper benchmark-specific sections may still need Day 12/13 residual cleanup. |
| Algorithm reference sprint ledger | Partially complete. | Cholesky CSC headings/takeaways cleaned; broader AMD/ND/eigensolver history remains residual. |
| Public header sprint residue | Pending. | Scheduled for Day 8; `.h` edits require full quality checks. |
| Example wording cleanup | Pending. | Scheduled for Day 9 after header wording decisions. |
| Proof-owner naming cleanup | Pending. | Scheduled for Day 10-11. |

## Deferred Residuals

- Full `docs/algorithm.md` chronology rewrite across AMD, ND, LDL^T, wall-check,
  and eigensolver sections.
- Public benchmark option renames such as `--sprint86-slice`; compatibility
  needs a deliberate decision.
- Broad maintainer guide chronology cleanup. The guide is the policy owner and
  intentionally retains more historical context than public adoption docs.
- Any proof-owner filename, suite-name, Makefile, or CMake target changes.

## Cross-Link Notes

- README points benchmark readers to `benchmarks/README.md`.
- Tutorial points install readers to `INSTALL.md` and benchmark readers to
  `benchmarks/README.md`.
- Examples point measurement readers to `benchmarks/README.md` and quality or
  validation readers to `docs/maintainer_guide.md`.
- INSTALL keeps reviewed-platform interpretation narrow and does not try to
  become the maintainer-policy home.

## Day 7 Result

The main public docs now have clearer separation:

- README routes and summarizes.
- Tutorial teaches.
- Examples demonstrate.
- INSTALL validates install/package workflows.
- Benchmarks measure.
- Maintainer guide interprets quality and proof policy.
- Algorithm docs remain the technical reference, with deeper historical cleanup
  explicitly tracked as residual work.
