#!/usr/bin/env python3
"""
Unit tests for evaluation/evaluate.py.

All data here is synthetic (fabricated only to exercise the parsing and
metric logic) and must never be reported as real experimental results.

Usage:
    python3 -m pytest tests/test_evaluation.py
    python3 tests/test_evaluation.py
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import (
    REQUIRED_COLUMNS,
    compute_metrics,
    load_trials,
    parse_bool,
    parse_float,
)


def write_csv(rows, fieldnames=REQUIRED_COLUMNS):
    tmp = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    tmp.close()
    return tmp.name


def base_row(**overrides):
    row = {
        "trial_id": "T1-01",
        "prompt": "pick up the red cube",
        "expected_object": "red_cube",
        "selected_object": "red_cube",
        "expected_zone": "red_zone",
        "actual_zone": "red_zone",
        "selection_correct": "True",
        "pick_success": "True",
        "place_success": "True",
        "duration_s": "12.5",
        "notes": "",
    }
    row.update(overrides)
    return row


class TestParseBool(unittest.TestCase):
    def test_true_variants(self):
        for v in ["True", "true", "1", "yes", "Y"]:
            self.assertTrue(parse_bool(v))

    def test_false_variants(self):
        for v in ["False", "false", "0", "no", "N"]:
            self.assertFalse(parse_bool(v))

    def test_missing_is_unknown_not_false(self):
        self.assertIsNone(parse_bool(""))
        self.assertIsNone(parse_bool(None))
        self.assertIsNone(parse_bool("  "))

    def test_garbage_is_unknown(self):
        self.assertIsNone(parse_bool("unknown"))
        self.assertIsNone(parse_bool("maybe"))


class TestParseFloat(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_float("12.5"), 12.5)

    def test_missing(self):
        self.assertIsNone(parse_float(""))
        self.assertIsNone(parse_float(None))

    def test_invalid(self):
        self.assertIsNone(parse_float("not-a-number"))


class TestLoadTrials(unittest.TestCase):
    def test_empty_file_returns_no_rows(self):
        path = write_csv([])
        rows = load_trials(path)
        self.assertEqual(rows, [])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_trials("/nonexistent/path/trials.csv")

    def test_missing_required_column_raises(self):
        path = write_csv([{"trial_id": "T1"}], fieldnames=["trial_id"])
        with self.assertRaises(ValueError):
            load_trials(path)

    def test_valid_file_loads_rows(self):
        path = write_csv([base_row()])
        rows = load_trials(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trial_id"], "T1-01")


class TestComputeMetrics(unittest.TestCase):
    def test_no_rows_all_metrics_none(self):
        metrics, per_trial = compute_metrics([])
        self.assertEqual(metrics["total_trials_in_file"], 0)
        self.assertIsNone(metrics["object_selection_accuracy"]["value"])
        self.assertIsNone(metrics["pick_success_rate"]["value"])
        self.assertIsNone(metrics["place_success_rate"]["value"])
        self.assertIsNone(metrics["end_to_end_task_success_rate"]["value"])
        self.assertIsNone(metrics["average_execution_time_s"]["value"])
        self.assertEqual(per_trial, [])

    def test_all_success_synthetic(self):
        rows = [base_row(trial_id="T1-01"), base_row(trial_id="T2-01")]
        metrics, _ = compute_metrics(rows)
        self.assertEqual(metrics["object_selection_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["object_selection_accuracy"]["denominator"], 2)
        self.assertEqual(metrics["pick_success_rate"]["value"], 1.0)
        self.assertEqual(metrics["place_success_rate"]["value"], 1.0)
        self.assertEqual(metrics["end_to_end_task_success_rate"]["value"], 1.0)
        self.assertAlmostEqual(metrics["average_execution_time_s"]["value"], 12.5)

    def test_missing_data_not_counted_as_success(self):
        rows = [
            base_row(trial_id="T1-01", pick_success="", place_success="", selection_correct=""),
        ]
        metrics, _ = compute_metrics(rows)
        # Nothing was recorded, so denominators are 0 and value is None,
        # NOT a false "0% failure" or a counted success.
        self.assertIsNone(metrics["object_selection_accuracy"]["value"])
        self.assertEqual(metrics["object_selection_accuracy"]["denominator"], 0)
        self.assertIsNone(metrics["pick_success_rate"]["value"])
        self.assertIsNone(metrics["end_to_end_task_success_rate"]["value"])

    def test_partial_failure_lowers_rates(self):
        rows = [
            base_row(trial_id="T1-01", selection_correct="True", pick_success="True", place_success="True"),
            base_row(trial_id="T2-01", selection_correct="False", pick_success="False", place_success="False"),
        ]
        metrics, _ = compute_metrics(rows)
        self.assertEqual(metrics["object_selection_accuracy"]["value"], 0.5)
        self.assertEqual(metrics["pick_success_rate"]["value"], 0.5)
        self.assertEqual(metrics["place_success_rate"]["value"], 0.5)
        self.assertEqual(metrics["end_to_end_task_success_rate"]["value"], 0.5)

    def test_end_to_end_requires_all_three_stages_known(self):
        rows = [
            # Selection correct and pick succeeded, but place was never recorded.
            base_row(trial_id="T1-01", selection_correct="True", pick_success="True", place_success=""),
        ]
        metrics, _ = compute_metrics(rows)
        # Should not count toward end-to-end denominator since place_success is unknown.
        self.assertEqual(metrics["end_to_end_task_success_rate"]["denominator"], 0)
        self.assertIsNone(metrics["end_to_end_task_success_rate"]["value"])
        # But selection and pick metrics are still measurable independently.
        self.assertEqual(metrics["object_selection_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["pick_success_rate"]["value"], 1.0)

    def test_unavailable_object_trial_correctly_marked(self):
        # T5-style trial: expected_object is unavailable, correct behavior is
        # to select nothing, so selection_correct=True with no pick/place.
        rows = [
            base_row(
                trial_id="T5-01",
                prompt="pick up the purple cube",
                expected_object="none",
                selected_object="",
                expected_zone="none",
                actual_zone="",
                selection_correct="True",
                pick_success="",
                place_success="",
                duration_s="",
            )
        ]
        metrics, _ = compute_metrics(rows)
        self.assertEqual(metrics["object_selection_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["pick_success_rate"]["denominator"], 0)
        self.assertEqual(metrics["average_execution_time_s"]["denominator"], 0)

    def test_average_duration_ignores_missing(self):
        rows = [
            base_row(trial_id="T1-01", duration_s="10"),
            base_row(trial_id="T2-01", duration_s=""),
            base_row(trial_id="T3-01", duration_s="20"),
        ]
        metrics, _ = compute_metrics(rows)
        self.assertEqual(metrics["average_execution_time_s"]["denominator"], 2)
        self.assertAlmostEqual(metrics["average_execution_time_s"]["value"], 15.0)


class TestTemplateCsv(unittest.TestCase):
    def test_template_csv_is_parseable(self):
        template_path = Path(__file__).resolve().parent.parent / "evaluation" / "trials_template.csv"
        rows = load_trials(str(template_path))
        self.assertEqual(len(rows), 5)
        ids = [r["trial_id"] for r in rows]
        self.assertEqual(ids, ["T1-01", "T2-01", "T3-01", "T4-01", "T5-01"])


if __name__ == "__main__":
    unittest.main()
