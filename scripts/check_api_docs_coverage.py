#!/usr/bin/env python3
"""Check generated Doxygen API page coverage for public headers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CoverageError(RuntimeError):
    pass


def doxygen_header_stem(header: Path) -> str:
    """Return Doxygen's current file-page stem for a header basename."""
    if header.suffix != ".h":
        raise CoverageError(f"unsupported public header extension for {header}")
    return header.stem.replace("_", "__") + "_8h"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def checked_in_headers(include_dir: Path) -> list[Path]:
    if not include_dir.is_dir():
        raise CoverageError(f"include directory not found: {include_dir}")

    headers = sorted(path for path in include_dir.glob("*.h") if path.is_file())
    if not headers:
        raise CoverageError(f"no checked-in public headers found under {include_dir}")
    return headers


def check_coverage(root: Path, include_dir: Path, html_dir: Path) -> tuple[int, int, int]:
    if not html_dir.is_dir():
        raise CoverageError(f"generated API HTML directory not found: {rel(html_dir, root)}; run `make docs-check`")

    index = html_dir / "index.html"
    if not index.is_file():
        raise CoverageError(f"generated API index not found: {rel(index, root)}; rerun `make docs-check`")

    headers = checked_in_headers(include_dir)
    missing: list[str] = []
    reference_count = 0
    source_count = 0

    for header in headers:
        stem = doxygen_header_stem(header)
        reference_page = html_dir / f"{stem}.html"
        source_page = html_dir / f"{stem}_source.html"

        if reference_page.is_file():
            reference_count += 1
        else:
            missing.append(f"{rel(header, root)} -> missing reference page {rel(reference_page, root)}")

        if source_page.is_file():
            source_count += 1
        else:
            missing.append(f"{rel(header, root)} -> missing source page {rel(source_page, root)}")

    if missing:
        raise CoverageError("missing generated API pages:\n" + "\n".join(f"  - {item}" for item in missing))

    return len(headers), reference_count, source_count


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that generated Doxygen HTML contains reference and source "
            "pages for checked-in public headers."
        )
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument(
        "--include-dir",
        type=Path,
        default=None,
        help="public header directory; defaults to <root>/include",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=None,
        help="generated Doxygen HTML directory; defaults to <root>/docs/api/html",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    include_dir = (args.include_dir or root / "include").resolve()
    html_dir = (args.html_dir or root / "docs" / "api" / "html").resolve()

    try:
        header_count, reference_count, source_count = check_coverage(root, include_dir, html_dir)
    except CoverageError as exc:
        print(f"api-docs-coverage: FAIL: {exc}", file=sys.stderr)
        return 1

    print("api-docs-coverage: PASS")
    print(f"  checked-in public headers: {header_count}")
    print(f"  generated reference pages: {reference_count}")
    print(f"  generated source pages:    {source_count}")
    print("  generated sparse_version.h: separate installed-header policy row; not an expected page")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
