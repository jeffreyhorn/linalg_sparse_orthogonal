#!/usr/bin/env python3
"""Mutation fixtures for the installed static-package contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_static_install_contract import validate  # noqa: E402


VERSION = "2.2.0"
PC = """prefix=/tmp/sparse-prefix
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: sparse
Description: Static archive package metadata for sparse linear algebra
Version: 2.2.0
Cflags: -I${includedir}
Libs: -L${libdir} -lsparse_lu_ortho -lm
"""


def write_fixture(prefix: Path, mode: str = "cmake", platform: str = "unix") -> None:
    library_dir = prefix / "lib"
    (library_dir / "pkgconfig").mkdir(parents=True)
    archive = (
        "sparse_lu_ortho.lib" if platform == "windows" else "libsparse_lu_ortho.a"
    )
    (library_dir / archive).write_bytes(b"static archive fixture")
    (library_dir / "pkgconfig" / "sparse.pc").write_text(
        PC.replace("prefix=/tmp/sparse-prefix", f"prefix={prefix}"),
        encoding="utf-8",
    )
    if mode == "make":
        return
    package_dir = library_dir / "cmake" / "Sparse"
    package_dir.mkdir(parents=True)
    (package_dir / "SparseConfig.cmake").write_text(
        'include("${CMAKE_CURRENT_LIST_DIR}/SparseTargets.cmake")\n',
        encoding="utf-8",
    )
    (package_dir / "SparseConfigVersion.cmake").write_text(
        'set(PACKAGE_VERSION "2.2.0")\n', encoding="utf-8"
    )
    (package_dir / "SparseTargets.cmake").write_text(
        'add_library(Sparse::sparse_lu_ortho STATIC IMPORTED)\n',
        encoding="utf-8",
    )
    (package_dir / "SparseTargets-release.cmake").write_text(
        'set_target_properties(Sparse::sparse_lu_ortho PROPERTIES\n'
        f'  IMPORTED_LOCATION_RELEASE "${{_IMPORT_PREFIX}}/lib/{archive}")\n',
        encoding="utf-8",
    )


class StaticInstallContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.prefix = Path(self.temp_dir.name) / "prefix"
        write_fixture(self.prefix)

    def errors(self, mode: str = "cmake", platform: str = "unix") -> list[str]:
        return validate(self.prefix, mode, platform, VERSION)

    def assert_rejects(self, marker: str) -> None:
        errors = self.errors()
        self.assertTrue(any(marker in error for error in errors), errors)

    def test_valid_unix_cmake_and_make_packages(self) -> None:
        self.assertEqual(self.errors(), [])
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            write_fixture(prefix, mode="make")
            self.assertEqual(validate(prefix, "make", "unix", VERSION), [])

    def test_expected_windows_static_lib_name_and_metadata_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            write_fixture(prefix, platform="windows")
            self.assertEqual(validate(prefix, "cmake", "windows", VERSION), [])
            (prefix / "lib" / "sparse_lu_ortho_import.lib").write_bytes(b"import")
            errors = validate(prefix, "cmake", "windows", VERSION)
            self.assertTrue(any("unexpected installed" in error for error in errors))

    def test_rejects_shared_and_import_artifacts(self) -> None:
        names = (
            "libsparse_lu_ortho.so",
            "libsparse_lu_ortho.so.1",
            "libsparse_lu_ortho.dylib",
            "sparse_lu_ortho.dll",
            "libsparse_lu_ortho.dll.a",
            "libsparse_lu_ortho.tbd",
        )
        for name in names:
            with self.subTest(name=name):
                path = self.prefix / "lib" / name
                path.write_bytes(b"unexpected")
                self.assert_rejects("unexpected shared/import artifact")
                path.unlink()

    def test_rejects_runtime_file_outside_library_dir(self) -> None:
        bin_dir = self.prefix / "bin"
        bin_dir.mkdir()
        (bin_dir / "sparse_lu_ortho.dll").write_bytes(b"dll")
        self.assert_rejects("unexpected installed runtime artifact")

    def test_rejects_missing_static_archive(self) -> None:
        (self.prefix / "lib/libsparse_lu_ortho.a").unlink()
        self.assert_rejects("missing regular static archive")

    def test_rejects_missing_pkgconfig_metadata(self) -> None:
        (self.prefix / "lib/pkgconfig/sparse.pc").unlink()
        self.assert_rejects("missing regular file")

    def test_rejects_missing_cmake_export_metadata(self) -> None:
        (self.prefix / "lib/cmake/Sparse/SparseTargets.cmake").unlink()
        self.assert_rejects("missing CMake package files")

    def test_rejects_wrong_cmake_package_version(self) -> None:
        version_file = self.prefix / "lib/cmake/Sparse/SparseConfigVersion.cmake"
        version_file.write_text('set(PACKAGE_VERSION "2.3.0")\n', encoding="utf-8")
        self.assert_rejects("CMake package version does not match")

    def test_rejects_cmake_loader_and_selector_metadata(self) -> None:
        path = self.prefix / "lib/cmake/Sparse/SparseTargets.cmake"
        original = path.read_text(encoding="utf-8")
        for addition in (
            'add_library(Sparse::shared SHARED IMPORTED)\n',
            'set_property(TARGET Sparse::sparse_lu_ortho PROPERTY IMPORTED_IMPLIB "x")\n',
            'set(SPARSE_ABI 1)\n',
            'set(SPARSE_ABI_EPOCH 1)\n',
            'set(SPARSE_EXPORT_SYMBOLS "extra")\n',
            'set(SPARSE_IMPORT_TABLE "extra")\n',
            'set(SPARSE_BUILD_SHARED ON)\n',
            'set(SPARSE_BUILD_STATIC ON)\n',
            'set(SPARSE_LINKAGE shared)\n',
            'set(IMPORTED_SONAME "libsparse_lu_ortho.so.1")\n',
        ):
            with self.subTest(addition=addition):
                path.write_text(original + addition, encoding="utf-8")
                self.assert_rejects("unsupported CMake loader/selector metadata")
        path.write_text(original, encoding="utf-8")

    def test_rejects_extra_cmake_package_file_and_wrong_archive_location(self) -> None:
        package_dir = self.prefix / "lib/cmake/Sparse"
        extra = package_dir / "SparseSharedTargets.cmake"
        extra.write_text("", encoding="utf-8")
        self.assert_rejects("unexpected CMake package file")
        extra.unlink()
        config = package_dir / "SparseTargets-release.cmake"
        config.write_text('set(IMPORTED_LOCATION_RELEASE "build/library.so")\n',
                          encoding="utf-8")
        self.assert_rejects("CMake imported locations differ")

    def test_rejects_extra_target_declared_in_package_config(self) -> None:
        config = self.prefix / "lib/cmake/Sparse/SparseConfig.cmake"
        config.write_text(
            config.read_text(encoding="utf-8")
            + 'add_library(Sparse::extra STATIC IMPORTED)\n',
            encoding="utf-8",
        )
        self.assert_rejects("CMake imported targets differ")

    def test_rejects_imported_location_override_in_package_config(self) -> None:
        config = self.prefix / "lib/cmake/Sparse/SparseConfig.cmake"
        original = config.read_text(encoding="utf-8")
        for property_name in ("IMPORTED_LOCATION_RELEASE", "IMPORTED_LOCATION_NOCONFIG"):
            with self.subTest(property_name=property_name):
                config.write_text(
                    original
                    + 'set_target_properties(Sparse::sparse_lu_ortho PROPERTIES\n'
                    + f'  {property_name} "/outside/other.a")\n',
                    encoding="utf-8",
                )
                self.assert_rejects("CMake imported locations differ")

    def test_rejects_second_dynamic_imported_location(self) -> None:
        config = self.prefix / "lib/cmake/Sparse/SparseTargets-release.cmake"
        config.write_text(
            config.read_text(encoding="utf-8")
            + 'set(IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/sparse.so")\n',
            encoding="utf-8",
        )
        self.assert_rejects("dynamic imported location")

    def test_rejects_second_wrong_static_imported_location(self) -> None:
        config = self.prefix / "lib/cmake/Sparse/SparseTargets-release.cmake"
        config.write_text(
            config.read_text(encoding="utf-8")
            + 'set(IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/other.a")\n',
            encoding="utf-8",
        )
        self.assert_rejects("CMake imported locations differ")

    def test_rejects_pkgconfig_claim_and_version_drift(self) -> None:
        path = self.prefix / "lib/pkgconfig/sparse.pc"
        original = path.read_text(encoding="utf-8")
        for changed in (
            original + "Libs.private: -lshared\n",
            original.replace("Version: 2.2.0", "Version: 2.3.0"),
            original.replace("Static archive package", "Shared-library package"),
            original + "SPARSE_ABI=1\n",
            original.replace("libdir=${exec_prefix}/lib", "libdir=${exec_prefix}/other"),
            original.replace(f"prefix={self.prefix}", "prefix=/another/prefix"),
        ):
            with self.subTest(changed=changed[-40:]):
                path.write_text(changed, encoding="utf-8")
                self.assert_rejects("pkg-config")

    def test_rejects_cmake_metadata_in_make_install(self) -> None:
        self.assertIn(
            "Make install unexpectedly contains CMake package metadata",
            self.errors(mode="make"),
        )


if __name__ == "__main__":
    unittest.main()
