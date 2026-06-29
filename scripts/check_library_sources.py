#!/usr/bin/env python3
"""Check library source membership across manifest, Makefile, and CMake."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "build-metadata" / "library_sources.txt"
MAKEFILE = ROOT / "Makefile"
CMAKE = ROOT / "CMakeLists.txt"


class CheckError(RuntimeError):
    pass


def normalize_path(raw: str) -> str:
    path = raw.strip().strip('"').strip("'")
    path = path.replace("$(SRCDIR)/", "src/")
    path = path.replace("${CMAKE_CURRENT_SOURCE_DIR}/", "")
    path = path.replace("\\", "/")
    return path


def read_manifest() -> list[str]:
    if not MANIFEST.exists():
        raise CheckError(f"manifest not found: {MANIFEST.relative_to(ROOT)}")

    sources: list[str] = []
    for lineno, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        path = normalize_path(stripped)
        if not re.fullmatch(r"src/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.c", path):
            raise CheckError(f"{MANIFEST.relative_to(ROOT)}:{lineno}: invalid source path: {stripped}")
        sources.append(path)
    return sources


def parse_make_lib_sources() -> list[str]:
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^LIB_SRCS\s*=\s*(.*?)(?=\n[A-Z0-9_]+\s*=|\n\n)", text, re.S | re.M)
    if not match:
        raise CheckError("could not find Makefile LIB_SRCS block")

    block = match.group(1).replace("\\\n", " ")
    sources = [normalize_path(token) for token in block.split() if token.strip()]
    sources = [source for source in sources if source.endswith(".c")]
    if not sources:
        raise CheckError("Makefile LIB_SRCS block did not contain any .c sources")
    return sources


def parse_cmake_library_sources() -> list[str]:
    text = CMAKE.read_text(encoding="utf-8")
    match = re.search(r"add_library\s*\(\s*sparse_lu_ortho\s+STATIC\s*(.*?)\n\)", text, re.S)
    if not match:
        raise CheckError("could not find CMake sparse_lu_ortho STATIC add_library block")

    sources: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        source = normalize_path(line)
        if source.endswith(".c"):
            sources.append(source)
    if not sources:
        raise CheckError("CMake add_library block did not contain any .c sources")
    return sources


def duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for value in values:
        if value in seen and value not in dupes:
            dupes.append(value)
        seen.add(value)
    return dupes


def compare(expected: list[str], actual: list[str], name: str) -> list[str]:
    messages: list[str] = []
    expected_set = set(expected)
    actual_set = set(actual)

    missing = [source for source in expected if source not in actual_set]
    extra = [source for source in actual if source not in expected_set]
    if missing:
        messages.append(f"{name} missing manifest entries:\n" + "\n".join(f"  - {source}" for source in missing))
    if extra:
        messages.append(f"{name} has entries not in manifest:\n" + "\n".join(f"  - {source}" for source in extra))
    if not missing and not extra and expected != actual:
        lines = [f"{name} order differs from manifest:"]
        for index, (left, right) in enumerate(zip(expected, actual), 1):
            if left != right:
                lines.append(f"  position {index}: manifest={left} {name}={right}")
        messages.append("\n".join(lines))
    return messages


def validate_unique(name: str, values: list[str]) -> list[str]:
    dupes = duplicates(values)
    if not dupes:
        return []
    return [f"{name} contains duplicate entries:\n" + "\n".join(f"  - {source}" for source in dupes)]


def main() -> int:
    try:
        manifest = read_manifest()
        make_sources = parse_make_lib_sources()
        cmake_sources = parse_cmake_library_sources()
    except CheckError as exc:
        print(f"source-list-check: ERROR: {exc}", file=sys.stderr)
        return 2

    messages: list[str] = []
    messages.extend(validate_unique("manifest", manifest))
    messages.extend(validate_unique("Makefile LIB_SRCS", make_sources))
    messages.extend(validate_unique("CMake add_library", cmake_sources))
    messages.extend(compare(manifest, make_sources, "Makefile LIB_SRCS"))
    messages.extend(compare(manifest, cmake_sources, "CMake add_library"))

    if messages:
        print("source-list-check: FAIL")
        for message in messages:
            print()
            print(message)
        return 1

    print(f"source-list-check: PASS ({len(manifest)} library sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
