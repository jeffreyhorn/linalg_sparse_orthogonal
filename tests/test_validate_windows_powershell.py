#!/usr/bin/env python3
"""Guard Windows PowerShell validation ownership checks."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_windows_powershell as validator  # noqa: E402


def read_workflow() -> str:
    return validator.WINDOWS_WORKFLOW.read_text(encoding="utf-8")


def assert_raises_with(fn, expected: str) -> None:
    try:
        fn()
    except validator.ValidationError as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}") from exc
        return
    raise AssertionError(f"expected validation failure containing {expected!r}")


def selected_steps_from(text: str) -> list[validator.Step]:
    rows = validator.selected_report_targets()
    steps = validator.validate_workflow_structure(
        text,
        validator.selected_report_freshness_tokens(rows),
    )
    return validator.validate_required_steps(steps)


def manifest_rows() -> list[dict[str, str]]:
    return validator.selected_report_targets()


def selected_steps() -> list[validator.Step]:
    return selected_steps_from(read_workflow())


def write_fake_pwsh(directory: Path, body: str) -> Path:
    fake = directory / "pwsh"
    fake.write_text("#!/usr/bin/env sh\n" + body, encoding="utf-8")
    fake.chmod(0o755)
    return fake


def run_with_path(argv: list[str], path: str) -> int:
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = path
    try:
        return validator.main(argv)
    finally:
        os.environ["PATH"] = old_path


def test_current_windows_workflow_structural_validation() -> None:
    validator.validate_hosted_validation_wiring(read_workflow())
    assert len(selected_steps()) == len(validator.STEP_REQUIREMENTS)


def test_shell_drift_fails_clearly() -> None:
    drifted = read_workflow().replace("        shell: pwsh", "        shell: cmd", 1)
    steps = validator.validate_workflow_structure(drifted)
    assert_raises_with(
        lambda: validator.validate_required_steps(steps),
        "must declare shell: pwsh",
    )


def test_command_anchor_drift_fails_clearly() -> None:
    drifted = read_workflow().replace("cmake -S . -B build", "cmake --preset windows", 1)
    steps = validator.validate_workflow_structure(drifted)
    assert_raises_with(
        lambda: validator.validate_required_steps(steps),
        "missing token 'cmake -S . -B build'",
    )


def test_unowned_powershell_step_fails_clearly() -> None:
    marker = "      - name: Run enforced reviewed CMake configure path"
    injected = (
        "      - name: Unowned PowerShell report-adjacent drift\n"
        "        run: Write-Host \"drift\"\n"
        "        shell: pwsh\n\n"
    )
    drifted = read_workflow().replace(marker, injected + marker, 1)
    steps = validator.validate_workflow_structure(drifted)
    assert_raises_with(
        lambda: validator.validate_required_steps(steps),
        "windows workflow has unowned PowerShell steps",
    )


def test_forbidden_windows_report_freshness_command_fails_clearly() -> None:
    drifted = read_workflow() + "\n# drift\nrun: make report-index-comparison-freshness\n"
    assert_raises_with(
        lambda: validator.validate_workflow_structure(
            drifted,
            validator.selected_report_freshness_tokens(manifest_rows()),
        ),
        "windows workflow must not run or upload selected report freshness",
    )


def test_extra_windows_upload_artifact_fails_outside_selected_lane() -> None:
    drifted = read_workflow() + "\n      - uses: actions/upload-artifact@v4\n"
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "must not publish hosted artifacts outside the owned selected comparison freshness lanes",
    )


def test_selected_cholesky_lane_missing_target_fails_clearly() -> None:
    drifted = read_workflow().replace("--selected-target cholesky-spd-tridiag-5", "", 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "missing selected Cholesky token",
    )


def test_selected_cholesky_lane_generator_target_drift_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "--target cholesky-spd-tridiag-5", "--target qr-minnorm", 1
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "missing selected Cholesky token",
    )


def test_selected_cholesky_lane_artifact_name_drift_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "name: sprint190-windows-selected-comparison-cholesky",
        "name: sprint190-windows-selected-comparison-qr",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "sprint190-windows-selected-comparison-cholesky",
    )


def test_selected_cholesky_lane_upload_must_fail_closed() -> None:
    drifted = read_workflow().replace("if-no-files-found: error\n", "", 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "if-no-files-found: error",
    )


def test_selected_cholesky_lane_missing_timeout_fails_clearly() -> None:
    drifted = read_workflow().replace("    timeout-minutes: 20\n", "", 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "must declare timeout-minutes: 20",
    )


def test_selected_cholesky_lane_broad_upload_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "            build/comparison/cholesky_spd_tridiag_5/project_observations.tsv",
        "            build/comparison/**",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "must not use broad comparison artifact paths",
    )


def test_selected_cholesky_lane_missing_required_upload_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "            build/comparison/cholesky_spd_tridiag_5/manifest.tsv\n",
        "",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "missing upload path",
    )


def test_selected_qr_lane_missing_target_fails_clearly() -> None:
    drifted = read_workflow().replace("--selected-target qr-incompatible-ls", "", 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "missing selected QR token",
    )


def test_selected_qr_lane_generator_target_drift_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "--target qr-incompatible-ls", "--target qr-minnorm", 1
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "missing selected QR token",
    )


def test_selected_qr_lane_artifact_name_drift_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "name: sprint209-windows-selected-comparison-qr-incompatible",
        "name: sprint203-windows-selected-comparison-qr-incompatible",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "sprint209-windows-selected-comparison-qr-incompatible",
    )


def test_selected_qr_lane_upload_must_fail_closed() -> None:
    qr_block = validator.find_job_block(read_workflow(), validator.WINDOWS_SELECTED_QR_JOB)
    drifted_qr_block = qr_block.replace("if-no-files-found: error\n", "", 1)
    drifted = read_workflow().replace(qr_block, drifted_qr_block, 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "if-no-files-found: error",
    )


def test_selected_qr_lane_fail_closed_must_belong_to_named_artifact() -> None:
    qr_block = validator.find_job_block(read_workflow(), validator.WINDOWS_SELECTED_QR_JOB)
    drifted_qr_block = qr_block.replace("          if-no-files-found: error\n", "", 1)
    later_upload = (
        "\n      - name: Upload unrelated QR comparison artifact\n"
        "        uses: actions/upload-artifact@v4\n"
        "        with:\n"
        "          name: unrelated-sprint209-qr-debug\n"
        "          if-no-files-found: error\n"
        "          path: |\n"
        "            build/comparison/qr_incompatible_ls/debug.log\n"
    )
    drifted_qr_block = drifted_qr_block.replace(
        "\n  # Verify the reviewed static-first CMake install",
        later_upload + "\n  # Verify the reviewed static-first CMake install",
        1,
    )
    drifted = read_workflow().replace(qr_block, drifted_qr_block, 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "exactly one upload-artifact step",
    )


def test_selected_qr_lane_missing_timeout_fails_clearly() -> None:
    qr_block = validator.find_job_block(read_workflow(), validator.WINDOWS_SELECTED_QR_JOB)
    drifted_qr_block = qr_block.replace("    timeout-minutes: 20\n", "", 1)
    drifted = read_workflow().replace(qr_block, drifted_qr_block, 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "must declare timeout-minutes: 20",
    )


def test_selected_qr_lane_broad_upload_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "            build/comparison/qr_incompatible_ls/project_observations.tsv",
        "            build/comparison/**",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "must not use broad or stale QR artifact paths",
    )


def test_selected_qr_lane_missing_required_upload_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "            build/comparison/qr_incompatible_ls/manifest.tsv\n",
        "",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "missing upload path",
    )


def test_selected_qr_lane_extra_upload_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "            build/comparison/qr_incompatible_ls/manifest.tsv\n",
        "            build/comparison/qr_incompatible_ls/manifest.tsv\n"
        "            build/comparison/qr_incompatible_ls/debug.log\n",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "exact selected QR six-file contract",
    )


def test_selected_qr_lane_extra_upload_step_fails_clearly() -> None:
    qr_block = validator.find_job_block(read_workflow(), validator.WINDOWS_SELECTED_QR_JOB)
    later_upload = (
        "\n      - name: Upload unrelated QR comparison artifact\n"
        "        uses: actions/upload-artifact@v4\n"
        "        with:\n"
        "          name: unrelated-sprint209-qr-debug\n"
        "          if-no-files-found: error\n"
        "          path: |\n"
        "            build/comparison/qr_incompatible_ls/debug.log\n"
    )
    drifted_qr_block = qr_block.replace(
        "\n  # Verify the reviewed static-first CMake install",
        later_upload + "\n  # Verify the reviewed static-first CMake install",
        1,
    )
    drifted = read_workflow().replace(qr_block, drifted_qr_block, 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "exactly one upload-artifact step",
    )


def test_selected_qr_lane_upload_paths_must_belong_to_named_artifact() -> None:
    qr_block = validator.find_job_block(read_workflow(), validator.WINDOWS_SELECTED_QR_JOB)
    original_paths = "".join(f"            {path}\n" for path in validator.WINDOWS_SELECTED_QR_REQUIRED_FILES)
    drifted_qr_block = qr_block.replace(
        original_paths,
        "            build/comparison/qr_incompatible_ls/debug.log\n",
        1,
    )
    later_upload = (
        "\n      - name: Upload unrelated QR comparison artifact\n"
        "        uses: actions/upload-artifact@v4\n"
        "        with:\n"
        "          name: unrelated-sprint209-qr-debug\n"
        "          if-no-files-found: error\n"
        "          path: |\n"
        f"{original_paths}"
    )
    drifted_qr_block = drifted_qr_block.replace(
        "\n  # Verify the reviewed static-first CMake install",
        later_upload + "\n  # Verify the reviewed static-first CMake install",
        1,
    )
    drifted = read_workflow().replace(qr_block, drifted_qr_block, 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "exactly one upload-artifact step",
    )


def test_manifest_derived_artifact_name_is_forbidden_on_windows() -> None:
    artifact = manifest_rows()[0]["workflow_artifact"]
    drifted = read_workflow() + f"\n# drift\nname: {artifact}\n"
    assert_raises_with(
        lambda: validator.validate_workflow_structure(
            drifted,
            validator.selected_report_freshness_tokens(manifest_rows()),
        ),
        f"selected report freshness {artifact!r}",
    )


def test_claim_boundaries_validate_current_docs() -> None:
    validator.validate_claim_boundaries()


def test_claim_boundary_missing_marker_fails_clearly() -> None:
    path = validator.REPO_ROOT / "README.md"
    text = path.read_text(encoding="utf-8").replace(
        "Sprint 190 adds one guarded Windows hosted",
        "Windows report freshness follows the hosted validation lane",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "missing Windows/PowerShell non-claim marker",
    )


def test_claim_boundary_missing_public_qr_marker_fails_clearly() -> None:
    path = validator.REPO_ROOT / "INSTALL.md"
    text = path.read_text(encoding="utf-8").replace(
        "Sprint 209 adds one bounded Windows QR incompatible evidence-collection lane",
        "Sprint 209 adds Windows QR freshness support",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "INSTALL.md missing Windows/PowerShell non-claim marker",
    )


def test_report_index_claim_boundary_missing_qr_marker_fails_clearly() -> None:
    path = (
        validator.REPO_ROOT
        / "tests"
        / "corpus"
        / "schemas"
        / "report_index_fields.md"
    )
    text = path.read_text(encoding="utf-8").replace(
        "The QR incompatible least-squares target remains outside Windows selected\n"
        "freshness",
        "The QR incompatible least-squares target is ready for selected freshness",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "report_index_fields.md missing Windows/PowerShell non-claim marker",
    )


def test_maintainer_claim_boundary_missing_sprint209_marker_fails_clearly() -> None:
    path = validator.REPO_ROOT / "docs" / "maintainer_guide.md"
    text = path.read_text(encoding="utf-8").replace(
        "Sprint 209 adds one bounded Windows QR incompatible evidence-collection lane",
        "Sprint 209 adds a Windows QR freshness lane",
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "maintainer_guide.md missing Windows/PowerShell non-claim marker",
    )


def test_corpus_claim_boundary_missing_sprint209_marker_fails_clearly() -> None:
    path = validator.REPO_ROOT / "tests" / "corpus" / "README.md"
    text = path.read_text(encoding="utf-8").replace(
        "Sprint 209 adds one bounded Windows QR incompatible\n"
        "evidence-collection lane",
        "Sprint 209 adds Windows QR freshness evidence",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "tests/corpus/README.md missing Windows/PowerShell non-claim marker",
    )


def test_report_index_claim_boundary_missing_sprint209_marker_fails_clearly() -> None:
    path = (
        validator.REPO_ROOT
        / "tests"
        / "corpus"
        / "schemas"
        / "report_index_fields.md"
    )
    text = path.read_text(encoding="utf-8").replace(
        "Sprint 209 adds one bounded Windows QR incompatible\n"
        "evidence-collection lane",
        "Sprint 209 adds Windows QR freshness metadata",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "report_index_fields.md missing Windows/PowerShell non-claim marker",
    )


def test_claim_boundary_promotion_wording_fails_clearly() -> None:
    path = validator.REPO_ROOT / "docs" / "maintainer_guide.md"
    text = path.read_text(encoding="utf-8") + (
        "\nPowerShell validation proves Windows report freshness.\n"
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "unsupported Windows/PowerShell claim",
    )


def test_claim_boundary_windows_selected_promotion_wording_fails_clearly() -> None:
    path = validator.REPO_ROOT / "README.md"
    text = path.read_text(encoding="utf-8") + (
        "\nWindows selected Cholesky freshness is promoted.\n"
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "unsupported Windows/PowerShell claim",
    )


def test_claim_boundary_selected_windows_promotion_wording_fails_clearly() -> None:
    path = validator.REPO_ROOT / "README.md"
    text = path.read_text(encoding="utf-8") + (
        "\nSelected Windows Cholesky freshness is promoted.\n"
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "unsupported Windows/PowerShell claim",
    )


def test_claim_boundary_selected_windows_generic_promotion_fails_clearly() -> None:
    path = validator.REPO_ROOT / "INSTALL.md"
    text = path.read_text(encoding="utf-8") + (
        "\nSelected Windows freshness is promoted.\n"
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "unsupported Windows/PowerShell claim",
    )


def test_claim_boundary_qr_windows_promotion_wording_fails_clearly() -> None:
    path = validator.REPO_ROOT / "README.md"
    text = path.read_text(encoding="utf-8") + (
        "\nQR incompatible least-squares Windows selected freshness is promoted.\n"
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "unsupported Windows/PowerShell claim",
    )


def test_claim_boundary_windows_selected_qr_promotion_wording_fails_clearly() -> None:
    path = validator.REPO_ROOT / "README.md"
    text = path.read_text(encoding="utf-8") + (
        "\nWindows selected QR incompatible freshness is promoted.\n"
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "unsupported Windows/PowerShell claim",
    )


def test_hosted_validation_wiring_requires_fail_closed_command() -> None:
    drifted = read_workflow().replace(" --require-pwsh", "", 1)
    assert_raises_with(
        lambda: validator.validate_hosted_validation_wiring(drifted),
        "must run",
    )


def test_hosted_validation_wiring_requires_windows_runner() -> None:
    drifted = read_workflow().replace(
        "  powershell-validation:\n"
        "    name: Windows PowerShell validation ownership\n"
        "    runs-on: windows-2022",
        "  powershell-validation:\n"
        "    name: Windows PowerShell validation ownership\n"
        "    runs-on: ubuntu-latest",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_hosted_validation_wiring(drifted),
        "must run on windows-2022",
    )


def test_hosted_validation_wiring_does_not_use_pwsh_shell() -> None:
    drifted = read_workflow().replace(
        "        run: python scripts/validate_windows_powershell.py --require-pwsh\n"
        "        shell: cmd",
        "        run: python scripts/validate_windows_powershell.py --require-pwsh\n"
        "        shell: pwsh",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_hosted_validation_wiring(drifted),
        "must declare shell: cmd",
    )


def test_hosted_validation_pwsh_shell_is_unowned() -> None:
    drifted = read_workflow().replace(
        "        run: python scripts/validate_windows_powershell.py --require-pwsh\n"
        "        shell: cmd",
        "        run: python scripts/validate_windows_powershell.py --require-pwsh\n"
        "        shell: pwsh",
        1,
    )
    steps = validator.validate_workflow_structure(drifted)
    assert_raises_with(
        lambda: validator.validate_required_steps(steps),
        "windows workflow has unowned PowerShell steps",
    )


def test_selected_report_manifest_references_validate() -> None:
    validator.validate_selected_report_references(manifest_rows())


def test_missing_manifest_workflow_file_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0] = dict(rows[0])
    rows[0]["workflow_file"] = ".github/workflows/missing-windows.yml"
    assert_raises_with(
        lambda: validator.validate_selected_report_references(rows),
        "references missing workflow_file",
    )


def test_manifest_windows_deferral_validation() -> None:
    validator.validate_manifest_windows_deferral(manifest_rows())


def test_manifest_windows_deferral_rejects_cholesky_windows_platform() -> None:
    rows = manifest_rows()
    cholesky = next(
        row
        for row in rows
        if row["target_id"] == validator.WINDOWS_SELECTED_CHOLESKY_TARGET_ID
    )
    cholesky["workflow_platforms"] = f"{cholesky['workflow_platforms']};windows"
    cholesky["workflow_file"] = (
        f"{cholesky['workflow_file']};{validator.WINDOWS_WORKFLOW.relative_to(validator.REPO_ROOT)}"
    )
    cholesky["workflow_artifact"] = (
        f"{cholesky['workflow_artifact']};{validator.WINDOWS_SELECTED_CHOLESKY_ARTIFACT}"
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "selected_report_targets.tsv must not list windows while Windows report freshness is deferred",
    )


def test_manifest_windows_deferral_rejects_cholesky_artifact_drift() -> None:
    rows = manifest_rows()
    cholesky = next(
        row
        for row in rows
        if row["target_id"] == validator.WINDOWS_SELECTED_CHOLESKY_TARGET_ID
    )
    cholesky["workflow_artifact"] = cholesky["workflow_artifact"].replace(
        "sprint175-macos-selected-comparison-freshness",
        validator.WINDOWS_SELECTED_CHOLESKY_ARTIFACT,
        1,
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must retain deferred Windows manifest workflow_artifact metadata",
    )


def test_manifest_windows_deferral_rejects_cholesky_identity_drift() -> None:
    rows = manifest_rows()
    cholesky = next(
        row
        for row in rows
        if row["target_id"] == validator.WINDOWS_SELECTED_CHOLESKY_TARGET_ID
    )
    cholesky["target_key"] = "cholesky-spd-tridiag-5-windows"
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must retain deferred Windows manifest target_key metadata",
    )


def test_manifest_windows_deferral_rejects_cholesky_row_contract_drift() -> None:
    rows = manifest_rows()
    cholesky = next(
        row
        for row in rows
        if row["target_id"] == validator.WINDOWS_SELECTED_CHOLESKY_TARGET_ID
    )
    cholesky["expected_row_ids"] = cholesky["expected_row_ids"].replace(
        "comparison_cholesky_spd_tridiag_5_project_status_v1",
        "comparison_cholesky_spd_tridiag_5_windows_project_status_v1",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must retain deferred Windows manifest expected_row_ids metadata",
    )


def test_manifest_windows_deferral_rejects_non_cholesky_windows_metadata() -> None:
    rows = manifest_rows()
    non_cholesky = next(
        row
        for row in rows
        if row["target_id"] != validator.WINDOWS_SELECTED_CHOLESKY_TARGET_ID
    )
    non_cholesky["workflow_file"] = (
        f"{non_cholesky['workflow_file']};"
        f"{validator.WINDOWS_WORKFLOW.relative_to(validator.REPO_ROOT)}"
    )
    non_cholesky["workflow_artifact"] = (
        f"{non_cholesky['workflow_artifact']};"
        f"{validator.WINDOWS_SELECTED_CHOLESKY_ARTIFACT}"
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "non-Cholesky selected rows must not reference Windows selected Cholesky",
    )


def test_manifest_windows_deferral_rejects_qr_windows_workflow_metadata() -> None:
    rows = manifest_rows()
    qr = next(
        row for row in rows if row["target_id"] == validator.WINDOWS_SELECTED_QR_TARGET_ID
    )
    qr["workflow_file"] = (
        f"{qr['workflow_file']};"
        f"{validator.WINDOWS_WORKFLOW.relative_to(validator.REPO_ROOT)}"
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must not list Sprint 209 Windows QR workflow file metadata",
    )


def test_manifest_windows_deferral_rejects_qr_windows_job_metadata() -> None:
    rows = manifest_rows()
    qr = next(
        row for row in rows if row["target_id"] == validator.WINDOWS_SELECTED_QR_TARGET_ID
    )
    qr["workflow_job"] = f"{qr['workflow_job']};{validator.WINDOWS_SELECTED_QR_JOB}"
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must not list Sprint 209 Windows QR workflow job metadata",
    )


def test_manifest_windows_deferral_rejects_qr_windows_artifact_metadata() -> None:
    rows = manifest_rows()
    qr = next(
        row for row in rows if row["target_id"] == validator.WINDOWS_SELECTED_QR_TARGET_ID
    )
    qr["workflow_artifact"] = (
        f"{qr['workflow_artifact']};{validator.WINDOWS_SELECTED_QR_ARTIFACT}"
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must not list Sprint 209 Windows QR workflow artifact metadata",
    )


def test_manifest_windows_deferral_rejects_qr_windows_platform() -> None:
    rows = manifest_rows()
    qr = next(
        row for row in rows if row["target_id"] == validator.WINDOWS_SELECTED_QR_TARGET_ID
    )
    qr["workflow_platforms"] = f"{qr['workflow_platforms']};windows"
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "selected_report_targets.tsv must not list windows while Windows report freshness is deferred",
    )


def test_manifest_windows_deferral_rejects_qr_identity_drift() -> None:
    rows = manifest_rows()
    qr = next(
        row for row in rows if row["target_id"] == validator.WINDOWS_SELECTED_QR_TARGET_ID
    )
    qr["target_key"] = "qr-incompatible-ls-windows"
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must retain re-deferred Windows QR manifest target_key metadata",
    )


def test_manifest_windows_deferral_rejects_qr_required_files_drift() -> None:
    rows = manifest_rows()
    qr = next(
        row for row in rows if row["target_id"] == validator.WINDOWS_SELECTED_QR_TARGET_ID
    )
    qr["required_files"] = qr["required_files"].replace(
        ";dependency_status.tsv", "", 1
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must retain re-deferred Windows QR manifest required_files metadata",
    )


def test_manifest_windows_deferral_rejects_qr_removed_windows_non_claim() -> None:
    rows = manifest_rows()
    qr = next(
        row for row in rows if row["target_id"] == validator.WINDOWS_SELECTED_QR_TARGET_ID
    )
    qr["non_claims"] = qr["non_claims"].replace(
        ";no Windows report freshness", "", 1
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must retain re-deferred Windows QR manifest non_claims metadata",
    )


def test_manifest_windows_deferral_rejects_non_qr_qr_windows_metadata() -> None:
    rows = manifest_rows()
    non_qr = next(
        row for row in rows if row["target_id"] != validator.WINDOWS_SELECTED_QR_TARGET_ID
    )
    non_qr["workflow_job"] = (
        f"{non_qr['workflow_job']};{validator.WINDOWS_SELECTED_QR_JOB}"
    )
    non_qr["workflow_artifact"] = (
        f"{non_qr['workflow_artifact']};{validator.WINDOWS_SELECTED_QR_ARTIFACT}"
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "non-QR selected rows must not reference Sprint 209 Windows QR",
    )


def test_manifest_windows_deferral_rejects_claim_scope_drift() -> None:
    rows = manifest_rows()
    cholesky = next(
        row
        for row in rows
        if row["target_id"] == validator.WINDOWS_SELECTED_CHOLESKY_TARGET_ID
    )
    cholesky["claim_scope"] = (
        "Selected Cholesky rows are fresh on reviewed Windows hosted lanes."
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must retain deferred Windows manifest claim_scope metadata",
    )


def test_manifest_windows_deferral_rejects_removed_windows_non_claim() -> None:
    rows = manifest_rows()
    cholesky = next(
        row
        for row in rows
        if row["target_id"] == validator.WINDOWS_SELECTED_CHOLESKY_TARGET_ID
    )
    cholesky["non_claims"] = cholesky["non_claims"].replace(
        ";no Windows report freshness", "", 1
    )
    assert_raises_with(
        lambda: validator.validate_manifest_windows_deferral(rows),
        "must retain deferred Windows manifest non_claims metadata",
    )


def test_deferral_record_validation() -> None:
    validator.validate_deferral_record()


def test_parse_with_fake_pwsh_accepts_selected_snippets() -> None:
    fake_body = """
test "$1" = "-NoProfile" || exit 11
test "$2" = "-NonInteractive" || exit 12
test "$3" = "-Command" || exit 13
test -n "$SPARSE_PWSH_SNIPPET" || exit 14
test -f "$SPARSE_PWSH_SNIPPET" || exit 15
grep -Eq 'cmake|ctest|sparse_lu_ortho.lib' "$SPARSE_PWSH_SNIPPET" || exit 16
printf '%s\\n' "$SPARSE_PWSH_SNIPPET" >> "$SPARSE_FAKE_PWSH_LOG"
"""
    with tempfile.TemporaryDirectory(prefix="sparse-fake-pwsh-") as tmp:
        tmpdir = Path(tmp)
        fake = write_fake_pwsh(tmpdir, fake_body)
        log = tmpdir / "pwsh.log"
        old_log = os.environ.get("SPARSE_FAKE_PWSH_LOG")
        os.environ["SPARSE_FAKE_PWSH_LOG"] = str(log)
        try:
            validator.parse_with_pwsh(str(fake), selected_steps())
        finally:
            if old_log is None:
                os.environ.pop("SPARSE_FAKE_PWSH_LOG", None)
            else:
                os.environ["SPARSE_FAKE_PWSH_LOG"] = old_log
        assert len(log.read_text(encoding="utf-8").splitlines()) == len(
            validator.STEP_REQUIREMENTS
        )


def test_parse_with_fake_pwsh_failure_is_actionable() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-fake-pwsh-") as tmp:
        fake = write_fake_pwsh(Path(tmp), "printf 'parse bad\\n' >&2\nexit 42\n")
        assert_raises_with(
            lambda: validator.parse_with_pwsh(str(fake), selected_steps()),
            "PowerShell parse failed",
        )


def test_main_with_fake_pwsh_returns_pass() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-fake-pwsh-") as tmp:
        tmpdir = Path(tmp)
        write_fake_pwsh(
            tmpdir,
            """
test -n "$SPARSE_PWSH_SNIPPET" || exit 14
test -f "$SPARSE_PWSH_SNIPPET" || exit 15
exit 0
""",
        )
        fake_pwsh_path = str(tmpdir) + os.pathsep + os.environ.get("PATH", "")
        assert run_with_path([], fake_pwsh_path) == 0
        assert run_with_path(["--require-pwsh"], fake_pwsh_path) == 0


def test_local_missing_pwsh_returns_unavailable() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-no-pwsh-") as tmp:
        assert run_with_path([], tmp) == 2


def test_require_pwsh_fails_closed_when_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-no-pwsh-") as tmp:
        assert run_with_path(["--require-pwsh"], tmp) == 1


def test_unavailable_output_keeps_non_pass_evidence_wording() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-no-pwsh-") as tmp:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = run_with_path([], tmp)
    assert rc == 2
    output = stderr.getvalue()
    assert "UNAVAILABLE: pwsh not found; structural checks passed" in output
    assert "local unavailable PowerShell is not pass evidence" in output


if __name__ == "__main__":
    test_current_windows_workflow_structural_validation()
    test_shell_drift_fails_clearly()
    test_command_anchor_drift_fails_clearly()
    test_unowned_powershell_step_fails_clearly()
    test_forbidden_windows_report_freshness_command_fails_clearly()
    test_extra_windows_upload_artifact_fails_outside_selected_lane()
    test_selected_cholesky_lane_missing_target_fails_clearly()
    test_selected_cholesky_lane_generator_target_drift_fails_clearly()
    test_selected_cholesky_lane_artifact_name_drift_fails_clearly()
    test_selected_cholesky_lane_upload_must_fail_closed()
    test_selected_cholesky_lane_missing_timeout_fails_clearly()
    test_selected_cholesky_lane_broad_upload_fails_clearly()
    test_selected_cholesky_lane_missing_required_upload_fails_clearly()
    test_selected_qr_lane_missing_target_fails_clearly()
    test_selected_qr_lane_generator_target_drift_fails_clearly()
    test_selected_qr_lane_artifact_name_drift_fails_clearly()
    test_selected_qr_lane_upload_must_fail_closed()
    test_selected_qr_lane_fail_closed_must_belong_to_named_artifact()
    test_selected_qr_lane_missing_timeout_fails_clearly()
    test_selected_qr_lane_broad_upload_fails_clearly()
    test_selected_qr_lane_missing_required_upload_fails_clearly()
    test_selected_qr_lane_extra_upload_fails_clearly()
    test_selected_qr_lane_extra_upload_step_fails_clearly()
    test_selected_qr_lane_upload_paths_must_belong_to_named_artifact()
    test_manifest_derived_artifact_name_is_forbidden_on_windows()
    test_claim_boundaries_validate_current_docs()
    test_claim_boundary_missing_marker_fails_clearly()
    test_claim_boundary_missing_public_qr_marker_fails_clearly()
    test_report_index_claim_boundary_missing_qr_marker_fails_clearly()
    test_maintainer_claim_boundary_missing_sprint209_marker_fails_clearly()
    test_corpus_claim_boundary_missing_sprint209_marker_fails_clearly()
    test_report_index_claim_boundary_missing_sprint209_marker_fails_clearly()
    test_claim_boundary_promotion_wording_fails_clearly()
    test_claim_boundary_windows_selected_promotion_wording_fails_clearly()
    test_claim_boundary_selected_windows_promotion_wording_fails_clearly()
    test_claim_boundary_selected_windows_generic_promotion_fails_clearly()
    test_claim_boundary_qr_windows_promotion_wording_fails_clearly()
    test_claim_boundary_windows_selected_qr_promotion_wording_fails_clearly()
    test_hosted_validation_wiring_requires_fail_closed_command()
    test_hosted_validation_wiring_requires_windows_runner()
    test_hosted_validation_wiring_does_not_use_pwsh_shell()
    test_hosted_validation_pwsh_shell_is_unowned()
    test_selected_report_manifest_references_validate()
    test_missing_manifest_workflow_file_fails_clearly()
    test_manifest_windows_deferral_validation()
    test_manifest_windows_deferral_rejects_cholesky_windows_platform()
    test_manifest_windows_deferral_rejects_cholesky_artifact_drift()
    test_manifest_windows_deferral_rejects_cholesky_identity_drift()
    test_manifest_windows_deferral_rejects_cholesky_row_contract_drift()
    test_manifest_windows_deferral_rejects_non_cholesky_windows_metadata()
    test_manifest_windows_deferral_rejects_qr_windows_workflow_metadata()
    test_manifest_windows_deferral_rejects_qr_windows_job_metadata()
    test_manifest_windows_deferral_rejects_qr_windows_artifact_metadata()
    test_manifest_windows_deferral_rejects_qr_windows_platform()
    test_manifest_windows_deferral_rejects_qr_identity_drift()
    test_manifest_windows_deferral_rejects_qr_required_files_drift()
    test_manifest_windows_deferral_rejects_qr_removed_windows_non_claim()
    test_manifest_windows_deferral_rejects_non_qr_qr_windows_metadata()
    test_manifest_windows_deferral_rejects_claim_scope_drift()
    test_manifest_windows_deferral_rejects_removed_windows_non_claim()
    test_deferral_record_validation()
    test_parse_with_fake_pwsh_accepts_selected_snippets()
    test_parse_with_fake_pwsh_failure_is_actionable()
    test_main_with_fake_pwsh_returns_pass()
    test_local_missing_pwsh_returns_unavailable()
    test_require_pwsh_fails_closed_when_missing()
    test_unavailable_output_keeps_non_pass_evidence_wording()
    print("test-validate-windows-powershell: ok")
