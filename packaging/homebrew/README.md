# Homebrew Local Formula Proof

This directory contains local proof material for the Sprint 180 Homebrew
provider decision. It is not a Homebrew/core formula, a tap, a bottle, a
Linuxbrew claim, or general package-manager support.

The source-controlled template is:

- `sparse-lu-ortho.rb.in`

The template is not installed directly. The future proof script renders it into
a temporary local formula by injecting:

- the current checkout version from `VERSION`;
- a temporary `file://` source archive URL;
- that archive's SHA-256 checksum;
- accurate local-proof license metadata.

Generated formula files, local taps, source archives, logs, Homebrew caches,
build trees, install prefixes, and bottle outputs are proof outputs and must
not be committed.

The selected proof boundary is local source formula only, static archive only,
and macOS-local first. Homebrew/core, bottles, Linuxbrew, hosted binaries,
registry readiness, shared-library support, dynamic ABI support, static/shared
selectors, and broad package-manager support remain unsupported unless a later
product decision adds separate evidence.
