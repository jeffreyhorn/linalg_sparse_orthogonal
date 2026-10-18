#!/usr/bin/env python3
"""Validate the installed static package without inferring a shared ABI."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PC_FIELDS = {
    "Name": "sparse",
    "Description": "Static archive package metadata for sparse linear algebra",
    "Cflags": "-I${includedir}",
    "Libs": "-L${libdir} -lsparse_lu_ortho -lm",
}
PC_VARIABLES = {"prefix", "exec_prefix", "libdir", "includedir"}
FORBIDDEN_CMAKE = re.compile(
    r"\b(?:SHARED IMPORTED|MODULE IMPORTED|IMPORTED_IMPLIB(?:_[A-Z_]+)?|"
    r"IMPORTED_SONAME(?:_[A-Z_]+)?|IMPORTED_NO_SONAME(?:_[A-Z_]+)?|"
    r"SOVERSION|INSTALL_NAME_DIR|MACOSX_RPATH|CMAKE_INSTALL_RPATH|"
    r"BUILD_SHARED_LIBS|SPARSE_(?:ABI|EXPORT|IMPORT|LINKAGE|LINK_MODE|"
    r"LIBRARY_TYPE|LIBRARY_KIND)(?:_[A-Z0-9_]*)?|"
    r"SPARSE_[A-Z0-9_]*(?:SHARED|STATIC)[A-Z0-9_]*)\b|"
    r"\bCOMPONENTS?\s+(?:STATIC|SHARED)\b|Sparse::[^\s\"']*shared",
    re.IGNORECASE,
)
IMPORTED_DYNAMIC_LOCATION = re.compile(
    r"\bIMPORTED_LOCATION[A-Z_]*[^\n]*(?:\.so(?:\.[0-9]+)*|\.dylib|\.dll)",
    re.IGNORECASE,
)
IMPORTED_TARGET = re.compile(
    r"(?m)^\s*add_library\s*\(\s*(Sparse::[^\s)]+)\s+([A-Z]+)\s+IMPORTED"
    r"(?:\s+GLOBAL)?\s*\)",
    re.IGNORECASE,
)


def read_text(path: Path, errors: list[str]) -> str:
    if not path.is_file() or path.is_symlink():
        errors.append(f"missing regular file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def check_pc(path: Path, prefix: Path, version: str, errors: list[str]) -> None:
    fields: dict[str, str] = {}
    variables: dict[str, str] = {}
    for line in read_text(path, errors).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        assignment = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        field = re.fullmatch(r"([A-Za-z][A-Za-z0-9_.]*):\s*(.*)", line)
        if assignment:
            key = assignment.group(1)
            if key not in PC_VARIABLES or key in variables:
                errors.append(f"unsupported or duplicate pkg-config variable: {key}")
            variables[key] = assignment.group(2)
        elif field:
            key = field.group(1)
            if key in fields:
                errors.append(f"duplicate pkg-config field: {key}")
            fields[key] = " ".join(field.group(2).split())
        else:
            errors.append(f"unrecognized pkg-config line: {line}")

    if set(variables) != PC_VARIABLES:
        errors.append(f"pkg-config variables differ: {sorted(variables)}")
    if "prefix" in variables and Path(variables["prefix"]).resolve() != prefix.resolve():
        errors.append(f"pkg-config prefix differs from install prefix: {variables['prefix']}")
    for key, expected_value in {
        "exec_prefix": "${prefix}",
        "libdir": "${exec_prefix}/lib",
        "includedir": "${prefix}/include",
    }.items():
        if key in variables and variables[key] != expected_value:
            errors.append(f"pkg-config {key} differs from {expected_value}")
    expected = {**PC_FIELDS, "Version": version}
    if fields != expected:
        errors.append(f"pkg-config fields differ from static contract: {fields}")


def check_cmake(package_dir: Path, archive_name: str, version: str,
                errors: list[str]) -> None:
    required = {
        "SparseConfig.cmake",
        "SparseConfigVersion.cmake",
        "SparseTargets.cmake",
    }
    files = sorted(package_dir.glob("*.cmake"))
    names = {path.name for path in files}
    contents: dict[str, str] = {}
    missing = required - names
    if missing:
        errors.append(f"missing CMake package files: {sorted(missing)}")
    if not any(re.fullmatch(r"SparseTargets-[A-Za-z0-9_]+\.cmake", name)
               for name in names):
        errors.append("missing CMake imported archive configuration")
    for path in files:
        if path.name not in required and not re.fullmatch(
            r"SparseTargets-[A-Za-z0-9_]+\.cmake", path.name
        ):
            errors.append(f"unexpected CMake package file: {path.name}")
        content = read_text(path, errors)
        contents[path.name] = content
        match = FORBIDDEN_CMAKE.search(content)
        if match:
            errors.append(f"unsupported CMake loader/selector metadata in {path.name}: "
                          f"{match.group(0)}")
        dynamic_location = IMPORTED_DYNAMIC_LOCATION.search(content)
        if dynamic_location:
            errors.append(f"dynamic imported location in {path.name}: "
                          f"{dynamic_location.group(0)}")

    declarations = [(target, library_type.upper()) for content in contents.values()
                    for target, library_type in IMPORTED_TARGET.findall(content)]
    if declarations != [("Sparse::sparse_lu_ortho", "STATIC")]:
        errors.append(f"CMake imported targets differ from static contract: "
                      f"{declarations}")
    package_contents = "\n".join(contents.values())
    expected_location = "${_IMPORT_PREFIX}/lib/" + archive_name
    locations = [quoted or unquoted for quoted, unquoted in re.findall(
        r'(?<!\w)"?IMPORTED_LOCATION(?:_[A-Z0-9_]+)?"?\s+'
        r'(?:"([^"]+)"|([^\s)]+))',
        package_contents,
    )]
    if not locations or any(location != expected_location for location in locations):
        errors.append(
            f"CMake imported locations differ from {expected_location}: {locations}"
        )
    if "SparseTargets.cmake" not in contents.get("SparseConfig.cmake", ""):
        errors.append("CMake config does not load the static target")
    if f'PACKAGE_VERSION "{version}"' not in contents.get(
        "SparseConfigVersion.cmake", ""
    ):
        errors.append(f"CMake package version does not match {version}")


def validate(prefix: Path, mode: str, platform: str, version: str) -> list[str]:
    errors: list[str] = []
    archive_name = (
        "sparse_lu_ortho.lib" if platform == "windows" else "libsparse_lu_ortho.a"
    )
    library_dir = prefix / "lib"
    package_dir = library_dir / "cmake" / "Sparse"
    allowed = {Path("lib") / archive_name, Path("lib/pkgconfig/sparse.pc")}
    if mode == "cmake":
        allowed.update(path.relative_to(prefix) for path in package_dir.glob("*.cmake"))

    if not (library_dir / archive_name).is_file() or (library_dir / archive_name).is_symlink():
        errors.append(f"missing regular static archive: {library_dir / archive_name}")
    if library_dir.exists():
        for path in library_dir.rglob("*"):
            if (path.is_file() or path.is_symlink()) and path.relative_to(prefix) not in allowed:
                errors.append(f"unexpected installed library/metadata artifact: {path}")
    bin_dir = prefix / "bin"
    if bin_dir.exists() and any(bin_dir.rglob("*")):
        errors.append(f"unexpected installed runtime artifact under {bin_dir}")
    for path in prefix.rglob("*"):
        name = path.name.lower()
        if (path.is_file() or path.is_symlink() or path.is_dir()) and (
            name.endswith((".so", ".dylib", ".dll", ".dll.a", ".tbd", ".exp",
                           ".framework", ".xcframework"))
            or ".so." in name
        ):
            errors.append(f"unexpected shared/import artifact: {path}")

    check_pc(library_dir / "pkgconfig" / "sparse.pc", prefix, version, errors)
    if mode == "cmake":
        check_cmake(package_dir, archive_name, version, errors)
    elif package_dir.exists():
        errors.append("Make install unexpectedly contains CMake package metadata")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--mode", choices=("make", "cmake"), required=True)
    parser.add_argument("--platform", choices=("unix", "windows"), required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    errors = validate(args.prefix, args.mode, args.platform, args.version)
    if errors:
        for error in errors:
            print(f"static-install-contract: FAIL: {error}")
        return 1
    print("static-install-contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
