#!/usr/bin/env python3
"""Regression checks for the frozen workflow-policy differential harness."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_api_docs_workflow_differential import (  # noqa: E402
    MIXED_VARIANTS, _classify, _edge_stage_matches, _unexpected_count, load_oracle, run,
)


class DifferentialHarnessTests(unittest.TestCase):
    def test_frozen_registration_and_mixed_variant_counts(self):
        _, oracle, registered = load_oracle()
        self.assertEqual(190, len(oracle))
        self.assertEqual(190, len(registered))
        self.assertEqual(26, sum(map(len, MIXED_VARIANTS.values())))

    def test_classification_keeps_disagreements_visible(self):
        self.assertEqual(
            ["legacy_missed_bypass"],
            _classify("REJECT", "ALLOW", "REJECT", "REJECT", ""),
        )
        self.assertEqual(
            ["structured_false_positive", "parser_error"],
            _classify("ALLOW", "ALLOW", "REJECT", "ALLOW", "parse: invalid YAML"),
        )
        self.assertEqual(0, _unexpected_count(Counter({"legacy_missed_bypass": 3})))
        self.assertEqual(2, _unexpected_count(Counter({
            "structured_missed_bypass": 1, "current_guard_regression": 1,
        })))
        self.assertEqual(1, _unexpected_count(Counter({"edge_stage_mismatch": 1})))
        self.assertTrue(_edge_stage_matches("parse", "fixture.yml:3:2: parse: duplicate key"))
        self.assertFalse(_edge_stage_matches("parse", "fixture.yml:3:2: jobs.test: path rejected"))
        self.assertTrue(_edge_stage_matches("policy", "fixture.yml:3:2: jobs.test: path rejected"))
        self.assertFalse(_edge_stage_matches("policy", "fixture.yml:3:2: parse: jobs.test invalid"))

    def test_one_fixture_reports_expected_verdict_and_location(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "differential.jsonl"
            errors, counts = run(output, frozenset(("test_workflow_staged_docs_artifact_upload_fails_clearly",)))
            self.assertEqual(0, errors)
            self.assertFalse(counts)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(1, len(rows))
            row = rows[0]
            self.assertEqual("REJECT", row["expected"])
            self.assertEqual("REJECT", row["legacy"]["verdict"])
            self.assertEqual("REJECT", row["structured"]["verdict"])
            self.assertEqual("REJECT", row["current"]["verdict"])
            self.assertEqual(".github/workflows/api-docs.yml", row["workflow_location"]["path"])
            self.assertNotIn(directory, json.dumps(row))

    def test_doxyfile_output_contract_survives_workflow_policy_handoff(self):
        from test_api_docs_local_only_guard import run_guard, write_fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            doxyfile = root / "Doxyfile"
            doxyfile.write_text(
                doxyfile.read_text(encoding="utf-8").replace(
                    "OUTPUT_DIRECTORY       = docs/api", "OUTPUT_DIRECTORY       = build/api"
                ),
                encoding="utf-8",
            )
            result = run_guard(root)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("Doxyfile must keep OUTPUT_DIRECTORY = docs/api", result.stderr)


if __name__ == "__main__":
    unittest.main()
