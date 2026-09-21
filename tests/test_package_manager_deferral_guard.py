#!/usr/bin/env python3
"""Regression tests for the package-manager deferral guard."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "package_manager_deferral_check.sh"


def run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_fixture(root: Path) -> None:
    (root / "scripts").mkdir()
    (root / "docs" / "planning" / "EPIC_15" / "SPRINT_171" / "artifacts").mkdir(
        parents=True
    )
    (root / "docs" / "planning" / "EPIC_18" / "SPRINT_198" / "artifacts").mkdir(
        parents=True
    )
    (root / "docs" / "planning" / "EPIC_19" / "SPRINT_207" / "artifacts").mkdir(
        parents=True
    )
    (root / "docs").mkdir(exist_ok=True)
    (root / "packaging" / "homebrew").mkdir(parents=True)

    (root / "scripts" / SCRIPT.name).write_text(
        SCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (root / "sparse.pc.in").write_text(
        "Name: sparse\n"
        "Description: Static archive package metadata for sparse linear algebra\n",
        encoding="utf-8",
    )
    (root / "cmake").mkdir()
    (root / "cmake" / "SparseConfig.cmake.in").write_text(
        '@PACKAGE_INIT@\ninclude("${CMAKE_CURRENT_LIST_DIR}/SparseTargets.cmake")\n',
        encoding="utf-8",
    )
    (root / "packaging" / "homebrew" / "README.md").write_text(
        "This directory contains proof material for the local static source "
        "formula only. Homebrew/core, bottles, Linuxbrew, hosted binaries, "
        "selectors, and broad package-manager support remain unsupported. "
        "It is proof material for the local static boundary.\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "Shared-library packaging, dynamic ABI support, Windows Makefile parity, "
        "Windows `pkg-config` parity, Homebrew/core readiness, bottles, "
        "Linuxbrew, public taps, binary packages, and broad package-manager "
        "distribution are not claimed. Homebrew proof is a developer-mode "
        "local static source formula proof only. package-manager distribution "
        "are not claimed.\n",
        encoding="utf-8",
    )
    (root / "INSTALL.md").write_text(
        "- package-manager deferral:\n"
        "  - Homebrew local formula proof artifacts exist\n"
        "  - not a user-facing Homebrew installation path\n"
        "  - distribution, static/shared selectors remain out of scope\n",
        encoding="utf-8",
    )
    (root / "docs" / "maintainer_guide.md").write_text(
        "package-manager support remains scoped as a non-claim. "
        "scripts/homebrew_local_formula_proof.sh records local proof. "
        "do not infer shared-library support, ABI stability, "
        "package-manager support from package evidence.\n",
        encoding="utf-8",
    )
    (root / "docs" / "planning" / "EPIC_15" / "SPRINT_171" / "artifacts" /
     "day5-package-manager-deferral.md").write_text(
        "Package-manager support is formally deferred. No vcpkg. Homebrew. "
        "Conan. pkgsrc. provider registry readiness. Evidence Needed To "
        "Revisit. Downstream consumer proof. Guard coverage.\n",
        encoding="utf-8",
    )
    (root / "docs" / "planning" / "EPIC_18" / "SPRINT_198" / "artifacts" /
     "day2-license-metadata-decision.md").write_text(
        "must not invent license terms\n",
        encoding="utf-8",
    )
    (root / "docs" / "planning" / "EPIC_18" / "SPRINT_198" / "artifacts" /
     "day9-end-to-end-proof-run.md").write_text(
        "Homebrew/core readiness, bottles, Linuxbrew support, public tap "
        "maintenance remain unclaimed\n",
        encoding="utf-8",
    )
    (root / "docs" / "planning" / "EPIC_18" / "SPRINT_198" / "artifacts" /
     "day14-closeout-review.md").write_text(
        "SPARSE_HOMEBREW_LICENSE=MIT HOMEBREW_DEVELOPER=1 temporary local "
        "tap proof exit `0` developer-mode local static source formula proof "
        "broad package-manager support remain unclaimed\n",
        encoding="utf-8",
    )
    (root / "docs" / "planning" / "EPIC_19" / "SPRINT_207" / "artifacts" /
     "day5-provider-decision.md").write_text(
        "continued deferral with stronger guards. will not promote a public "
        "Homebrew tap/source formula. claiming Homebrew/core readiness is out "
        "of scope. stable source archive and SHA-256 provenance. "
        "Homebrew/core-style formula audit evidence.\n",
        encoding="utf-8",
    )
    (root / "docs" / "planning" / "EPIC_19" / "SPRINT_207" / "artifacts" /
     "day6-proof-deferral-design.md").write_text(
        "| Pass | `0` | local proof completed | developer-mode local static "
        "source formula proof only |\n"
        "Reject public tap support wording without evidence\n"
        "Reject Homebrew/core readiness wording without evidence\n",
        encoding="utf-8",
    )


def assert_guard_fails_with(mutator, expected: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_fixture(root)
        mutator(root)
        result = run(["bash", "scripts/package_manager_deferral_check.sh"], root)
        if result.returncode == 0:
            raise AssertionError("expected guard failure")
        message = result.stdout + result.stderr
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}")


def test_positive_homebrew_core_readiness_claim_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "README.md").write_text(
            (root / "README.md").read_text(encoding="utf-8")
            + "\nHomebrew/core readiness is supported.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "unsupported provider claim")


def test_positive_public_tap_claim_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "INSTALL.md").write_text(
            (root / "INSTALL.md").read_text(encoding="utf-8")
            + "\npublic tap support is available.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "unsupported provider claim")


def test_positive_plural_public_taps_claim_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "README.md").write_text(
            (root / "README.md").read_text(encoding="utf-8")
            + "\npublic taps are supported.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "unsupported provider claim")


def test_positive_package_manager_support_claim_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "docs" / "maintainer_guide.md").write_text(
            (root / "docs" / "maintainer_guide.md").read_text(encoding="utf-8")
            + "\npackage-manager support is available.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "unsupported provider claim")


def test_positive_binary_package_claim_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "docs" / "maintainer_guide.md").write_text(
            (root / "docs" / "maintainer_guide.md").read_text(encoding="utf-8")
            + "\nbinary package support is provided.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "unsupported provider claim")


def test_positive_bottle_support_claim_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "README.md").write_text(
            (root / "README.md").read_text(encoding="utf-8")
            + "\nbottle support is available.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "unsupported provider claim")


def test_positive_linuxbrew_support_claim_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "INSTALL.md").write_text(
            (root / "INSTALL.md").read_text(encoding="utf-8")
            + "\nLinuxbrew support is provided.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "unsupported provider claim")


def test_positive_release_package_claim_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "packaging" / "homebrew" / "README.md").write_text(
            (root / "packaging" / "homebrew" / "README.md").read_text(
                encoding="utf-8"
            )
            + "\nrelease packages are available.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "unsupported provider claim")


def main() -> None:
    test_positive_homebrew_core_readiness_claim_fails_clearly()
    test_positive_public_tap_claim_fails_clearly()
    test_positive_plural_public_taps_claim_fails_clearly()
    test_positive_package_manager_support_claim_fails_clearly()
    test_positive_binary_package_claim_fails_clearly()
    test_positive_bottle_support_claim_fails_clearly()
    test_positive_linuxbrew_support_claim_fails_clearly()
    test_positive_release_package_claim_fails_clearly()


if __name__ == "__main__":
    main()
