#!/usr/bin/env python3
"""Validate local-only generated API routing docs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

API_ROUTING_FILES = (
    "README.md",
    "INSTALL.md",
    "docs/api_reference.md",
    "docs/maintainer_guide.md",
)

MAKEFILE_ROUTING_TARGET = re.compile(
    r"(?m)^api-docs-routing:\n"
    r"\t@python3 scripts/check_api_docs_routing\.py\n"
    r"\t@python3 tests/test_api_docs_routing\.py$"
)
MAKEFILE_VALIDATE_DEP = re.compile(r"(?m)^api-docs-validate:\s+.*\bapi-docs-routing\b")

REQUIRED_ROUTES = {
    "README.md": (
        "docs/api_reference.md",
        "include/",
        "INSTALL.md#support-readiness-matrix",
    ),
    "INSTALL.md": (),
    "docs/api_reference.md": (
        "../include/",
        "../Doxyfile",
        "../INSTALL.md#support-readiness-matrix",
        "tutorial.md",
        "cookbook.md",
        "solver_selection.md",
        "maintainer_guide.md",
    ),
    "docs/maintainer_guide.md": (),
}

REQUIRED_TEXT = {
    "docs/api_reference.md": (
        "The generated HTML tree is local-only generated output.",
        "is not a hosted or source-controlled publication surface.",
        "source-controlled API reference path.",
    ),
    "README.md": (
        "API reference entry point: docs/api_reference.md",
        "Generated API HTML is not hosted documentation, a retained CI artifact,",
    ),
    "INSTALL.md": (
        "| Local generated API HTML | local-only |",
        "No hosted API publication, retained generated-doc artifact, committed generated HTML,",
    ),
    "docs/maintainer_guide.md": (
        "`docs/api_reference.md` is the user-facing API reference entry point.",
        "`docs/api/html/` is generated Doxygen output",
        "`make api-docs-freshness` runs `docs-check` plus the local-only generated",
        "`api-docs-routing` proves user-facing docs route API readers",
        "source-controlled reference path is `docs/api_reference.md` plus checked-in",
        "retained generated-doc artifacts",
        "removes `api-docs-routing` from `make api-docs-freshness` without replacing",
    ),
}

FORBIDDEN_LINK_TARGET_PATTERNS = (
    re.compile(r"(^|/|\\.\\.)docs/api(/|$)"),
    re.compile(r"docs/api/html"),
    re.compile(r"https?://[^)]+(?:api|doxygen|pages|github\\.io)", re.IGNORECASE),
)


class RoutingError(RuntimeError):
    pass


def markdown_links(text: str) -> list[str]:
    return [match.group(1).strip() for match in LINK_PATTERN.finditer(text)]


def strip_fragment(target: str) -> str:
    return target.split("#", 1)[0]


def is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


def validate_target_exists(root: Path, source: Path, target: str) -> None:
    if is_external(target):
        return

    path_part = strip_fragment(target)
    if not path_part:
        return

    resolved = (source.parent / path_part).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RoutingError(f"{source.relative_to(root)} link escapes repository: {target}") from exc

    if not resolved.exists():
        raise RoutingError(f"{source.relative_to(root)} links to missing API route target: {target}")


def validate_required_routes(root: Path, rel_path: str, path: Path, text: str) -> None:
    links = set(markdown_links(text))
    missing = [target for target in REQUIRED_ROUTES[rel_path] if target not in links]
    if missing:
        joined = ", ".join(missing)
        raise RoutingError(f"{rel_path} missing required API route link(s): {joined}")

    for target in REQUIRED_ROUTES[rel_path]:
        validate_target_exists(root, path, target)

    for needle in REQUIRED_TEXT[rel_path]:
        if needle not in text:
            raise RoutingError(f"{rel_path} missing required local-only API routing text: {needle}")


def validate_no_forbidden_links(rel_path: str, text: str) -> None:
    for target in markdown_links(text):
        for pattern in FORBIDDEN_LINK_TARGET_PATTERNS:
            if pattern.search(target):
                raise RoutingError(
                    f"{rel_path} links to unsupported generated or hosted API publication target: {target}"
                )


def validate_makefile_wiring(root: Path) -> None:
    makefile = root / "Makefile"
    if not makefile.is_file():
        raise RoutingError("Makefile is missing; cannot verify api-docs-routing wiring")

    text = makefile.read_text(encoding="utf-8")
    if not MAKEFILE_ROUTING_TARGET.search(text):
        raise RoutingError("Makefile must define api-docs-routing with the routing guard and regression suite")
    if not MAKEFILE_VALIDATE_DEP.search(text):
        raise RoutingError("Makefile api-docs-validate must depend on api-docs-routing")


def validate_api_routes(root: Path) -> None:
    for rel_path in API_ROUTING_FILES:
        path = root / rel_path
        if not path.is_file():
            raise RoutingError(f"required API routing document missing: {rel_path}")

        text = path.read_text(encoding="utf-8")
        validate_required_routes(root, rel_path, path, text)
        validate_no_forbidden_links(rel_path, text)

    validate_makefile_wiring(root)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local-only generated API routing docs.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    try:
        validate_api_routes(root)
    except RoutingError as exc:
        print(f"api-docs-routing: FAIL: {exc}", file=sys.stderr)
        return 1

    print("api-docs-routing: PASS")
    print(f"  checked routing documents: {len(API_ROUTING_FILES)}")
    print("  Makefile api-docs-routing wiring: present")
    print("  generated API publication links: absent")
    print("  source-controlled API entry point: docs/api_reference.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
