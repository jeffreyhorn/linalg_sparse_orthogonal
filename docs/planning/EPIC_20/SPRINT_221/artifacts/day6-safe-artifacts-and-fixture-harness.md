# Sprint 221 Day 6: Safe Artifacts And Fixture Harness

## Scope

The v1 validator now binds the selected matrix, methodology references and
declared run artifacts to actual file bytes. The Day 5 status, identity,
numerical and claim checks remain in force. This closes item 221.2's
first-pass implementation boundary, but it does not release the pilot:
`--self-test` still exits with `implementation_pending` until later negative
fixture coverage is complete. No external solver or timing run occurred.

## Artifact Contract

- Repository inputs are the pinned matrix and the frozen Sprint 215 Day 3,
  Day 4 and Day 5 methodology files. The matrix's declared SHA-256 is checked
  against its bytes. Generated records must provide methodology SHA-256 values,
  which are checked against those source files. Planning records may leave
  methodology hashes null but still require the named files to exist.
- Run artifacts are read only under an explicit bundle root. Every declared
  artifact path is unique and every declared SHA-256 is checked against the
  file. Solver-side raw references, invalid-timing attempts and publication
  review references must point to listed artifacts. A missing bundle root is
  fatal when artifacts exist; an explicitly supplied nonexistent or
  non-directory root is always fatal.
- Paths must be normalized relative POSIX paths. Absolute paths, `.`/`..`,
  empty components, Windows drives and backslashes are rejected. The reader
  walks from a directory descriptor with `O_NOFOLLOW`, rejects symlinks in
  every component and accepts regular files only. Read/open failures become
  `artifact_path_or_hash` diagnostics at the relevant JSON pointer.
- Measured raw timing JSON is parsed from the same verified bytes used for
  its digest, with duplicate-key, UTF-8 and nonfinite-number rejection. Raw
  sample files have a 4 MiB v1 limit. The Day 5 timing checker recomputes
  summaries from that parsed object, not caller-supplied sample data.

The interface is `validate_record(record, schema, repo_root, bundle_root)`;
the CLI uses the source tree as its repository root and accepts
`--bundle-root` for generated artifacts. The CLI still returns
`implementation_pending` after a valid record until the negative self-test
is implemented. File integrity is not proof that a solver ran or that a
publication review was substantively approved. Raw solver-output content
binding and stronger synthetic observed fixtures remain follow-on test work.

## Fixture And Regression Results

The existing Sprint 215 blueprint is the positive planning fixture:
`run_id=null`, `cell_id=null`, both sides `not_run`, timing `not_run`, no
generated artifacts. A temporary-bundle helper writes synthetic solver and
timing files, computes SHA-256 from the bytes it wrote and fills methodology
hashes from source files. No source-controlled manifest, corpus input or
benchmark bundle is modified. Its observed and measured records exercise
the integrated validator without claiming a real CHOLMOD observation.

Negative tests cover missing/changed bundle files, missing and malformed
roots, inaccessible files, duplicate paths, unlisted review references,
symlinked files and parent directories, traversal/absolute/Windows-style
paths, changed matrix bytes, wrong methodology digest, and timing JSON whose
digest is correct but whose content has a duplicate key. The CLI reports
`record_id` and `/bundle_root` for a generated record without a bundle root.

## Verification

`python3 tests/test_validate_external_baseline_evidence.py` passed 29 tests;
`python3 tests/test_run_cholmod_pilot.py` passed 13 tests;
`python3 scripts/validate_corpus_schema.py`, Python byte-compilation of the
validator and test, and `make docs-check` passed. A direct `--self-test`
call exited 1 with `implementation_pending`. No `.c` or `.h` file changed,
so `make format && make lint && make test` was not required.
