# Sprint 162 Day 4 Product Decision Matrix

## Scope

Day 4 decides whether Sprint 162 should promote Windows `pkg-config`
execution parity, Windows Makefile parity, both, or neither. The decision is
based on the Day 2 parity audit and the Day 3 package metadata boundary
review.

The selected decision must be narrow enough to close during Sprint 162 without
weakening the already-reviewed Windows CMake install/downstream proof.

## Decision Matrix

Scoring uses a 1 to 5 scale where 5 is strongest for the project. For
maintainer cost, CI availability, portability risk, and documentation
complexity, a higher score means lower cost, better availability, lower risk,
and simpler documentation.

| Option | Maintainer Cost | CI Availability | User Value | Portability Risk | Documentation Complexity | Total | Summary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Promote Windows `pkg-config` only | 2 | 2 | 3 | 2 | 2 | 11 | Useful for Unix-like Windows environments, but needs a selected provider, compiler flag route, and downstream proof. |
| Promote Windows Makefile parity only | 1 | 2 | 2 | 1 | 2 | 8 | High implementation risk because current Make install/uninstall assumptions are POSIX-oriented. |
| Promote both Windows `pkg-config` and Makefile parity | 1 | 1 | 4 | 1 | 1 | 8 | Combines both technical risks and expands support claims beyond the current evidence base. |
| Retain both non-claims with stronger guards | 5 | 5 | 3 | 5 | 4 | 22 | Fully closes the ambiguity gap while preserving the reviewed CMake-first Windows package proof. |

## Selected Product Decision

Sprint 162 will retain both Windows Makefile parity and Windows `pkg-config`
execution parity as explicit non-claims, and will close the package parity gap
by strengthening guards, documentation wording, workflow assertions, and
evidence separation around the existing Windows CMake-first package proof.

This means:

- Windows remains a reviewed CMake install/downstream consumer surface.
- Windows continues to install and inspect `sparse.pc` as package metadata.
- Windows does not claim `pkg-config` command execution parity.
- Windows does not claim Makefile install/uninstall parity.
- Linux/macOS remain the reviewed Make install and `pkg-config` execution
  proof surfaces.
- The static archive contract remains the only package contract.

## Rationale

The Day 2 audit showed that the Windows package gap is not installed artifact
shape. Windows already proves:

- installed static `.lib`;
- installed headers and generated version header;
- installed CMake package metadata;
- installed `sparse.pc` metadata;
- static imported CMake target metadata;
- exact-version CMake consumer behavior;
- version mismatch rejection;
- generated and maintained downstream CMake consumers;
- absence of DLLs, shared imported metadata, loader metadata, and unsupported
  package wording.

The remaining deltas are execution-front-end deltas:

- Windows does not run `make install` or `make uninstall`.
- Windows does not run `pkg-config`.
- Windows does not compile/link/run a downstream consumer from `pkg-config`
  output.

Promoting either execution front end would require new support decisions that
are not already owned by the repository:

- a Windows `pkg-config` provider;
- a compiler/toolchain path for Unix-style `pkg-config` flags or a translation
  layer for MSVC;
- a Windows Make shell/toolchain route;
- install/uninstall semantics that do not conflict with the CMake-first
  package contract.

Retaining both non-claims is therefore the only option that can be completely
closed within Sprint 162 while keeping support claims aligned with evidence.

## Independent Treatment Of Package Front Ends

Makefile parity and `pkg-config` parity remain independent decisions.

| Front End | Day 4 Decision | Reason |
| --- | --- | --- |
| Windows CMake install/downstream | Retain and protect as reviewed support. | Already validated in CI and aligned with the static archive package contract. |
| Windows `pkg-config` execution | Retain as non-claim. | `sparse.pc` is installed as metadata, but no reviewed Windows provider, command execution, flags, or downstream consumer proof exists. |
| Windows Makefile install/uninstall | Retain as non-claim. | Current Make install/uninstall proof is POSIX shell and utility oriented. |
| Linux/macOS Make and `pkg-config` | Retain as reviewed support. | Existing CI proves Make install/uninstall and downstream `pkg-config` compile/link/run behavior. |

## Required Retained Non-Claim Proof

The selected decision requires guard evidence, not new Windows execution proof.
Days 5-14 should implement and validate these requirements:

1. Public docs must state that Windows package validation is CMake-first and
   does not prove Windows Makefile or `pkg-config` execution parity.
2. Windows workflow comments or checks must separate installed `sparse.pc`
   metadata from `pkg-config` command execution.
3. Static package guards must continue rejecting shared-library, dynamic ABI,
   runtime-loader, package-manager, and static/shared selector claims.
4. Package metadata checks must preserve the static archive description and
   reject unsupported package or ABI wording.
5. Evidence indexes or planning artifacts must distinguish CMake install
   consumer proof from Make/pkg-config execution proof.
6. Existing Linux/macOS Make and `pkg-config` validation must remain
   unchanged.
7. Existing Windows CMake install/downstream validation must remain unchanged
   or become stricter only around non-claim clarity.

## Rollback Criteria

If later implementation cannot preserve the retained non-claim cleanly, Sprint
162 should stop and reassess rather than quietly expanding Windows support
claims. Roll back or ask for a new product decision if:

- a guard requires deleting or weakening Windows CMake install/downstream
  validation;
- documentation cannot distinguish installed `sparse.pc` metadata from
  `pkg-config` execution;
- a workflow starts running `pkg-config` on Windows without a selected provider
  and downstream consumer proof;
- a Makefile path is added on Windows without install and uninstall proof;
- package metadata starts implying shared-library support, dynamic ABI,
  runtime-loader behavior, package-manager support, or static/shared selectors;
- Linux/macOS Make install and `pkg-config` proof regresses.

## Day 4 Conclusion

The selected Sprint 162 product decision is:

**Retain Windows Makefile parity and Windows `pkg-config` execution parity as
non-claims, and close the sprint by strengthening static-first/CMake-first
guards and documentation around the existing Windows package evidence.**

This gives Days 5-14 a bounded implementation path: design the exact retained
guard checks, add them, validate them locally, ensure hosted CI keeps the
current CMake-first package proof, and close the sprint with an evidence-based
retrospective.
