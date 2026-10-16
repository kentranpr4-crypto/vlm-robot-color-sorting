#!/usr/bin/env python3
"""
Evaluation script for the VLM-guided pick-and-place robot.

Reads a trial-results CSV (see trials_template.csv for the schema) and
computes reproducible metrics from it. This script does not run the
simulator or generate any experimental data itself -- it only summarizes
whatever a human (or an automated harness) recorded while running
color_sorting_vlm.py / panda_vision_simulation.py.

Usage:
    python3 evaluation/evaluate.py path/to/trials.csv
    python3 evaluation/evaluate.py evaluation/trials_template.csv --json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REQUIRED_COLUMNS = [
    "trial_id",
    "prompt",
    "expected_object",
    "selected_object",
    "expected_zone",
    "actual_zone",
    "selection_correct",
    "pick_success",
    "place_success",
    "duration_s",
    "notes",
]

TRUE_STRINGS = {"true", "1", "yes", "y"}
FALSE_STRINGS = {"false", "0", "no", "n"}


def parse_bool(value):
    """Parse a boolean-ish CSV cell. Returns True, False, or None (unknown).

    Missing / empty / unrecognized values are treated as unknown and must
    never be counted as a success.
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v == "":
        return None
    if v in TRUE_STRINGS:
        return True
    if v in FALSE_STRINGS:
        return False
    return None


def parse_float(value):
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_trials(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Trials file not found: {csv_path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Trials CSV is missing required columns: {missing}. "
                f"Expected columns: {REQUIRED_COLUMNS}"
            )
        rows = [row for row in reader]
    return rows


def compute_metrics(rows):
    """Compute evaluation metrics. Every metric reports its own denominator
    because different trials may have missing/unknown fields for different
    stages (e.g. a trial where selection failed has no pick/place data).
    """
    total_trials = len(rows)

    selection_known = 0
    selection_correct = 0

    pick_known = 0
    pick_success = 0

    place_known = 0
    place_success = 0

    end_to_end_known = 0
    end_to_end_success = 0

    durations = []

    per_trial = []

    for row in rows:
        trial_id = row.get("trial_id", "").strip()
        sel_correct = parse_bool(row.get("selection_correct"))
        pick_ok = parse_bool(row.get("pick_success"))
        place_ok = parse_bool(row.get("place_success"))
        duration = parse_float(row.get("duration_s"))

        if sel_correct is not None:
            selection_known += 1
            if sel_correct:
                selection_correct += 1

        if pick_ok is not None:
            pick_known += 1
            if pick_ok:
                pick_success += 1

        if place_ok is not None:
            place_known += 1
            if place_ok:
                place_success += 1

        # End-to-end success requires ALL three stages to be known and True.
        # If any stage is unknown or False, the trial cannot count as a
        # successful end-to-end run. It only counts toward the known
        # denominator when all three stages were actually recorded.
        if sel_correct is not None and pick_ok is not None and place_ok is not None:
            end_to_end_known += 1
            if sel_correct and pick_ok and place_ok:
                end_to_end_success += 1

        if duration is not None:
            durations.append(duration)

        per_trial.append(
            {
                "trial_id": trial_id,
                "selection_correct": sel_correct,
                "pick_success": pick_ok,
                "place_success": place_ok,
                "duration_s": duration,
            }
        )

    def rate(numerator, denominator):
        if denominator == 0:
            return None
        return numerator / denominator

    metrics = {
        "total_trials_in_file": total_trials,
        "object_selection_accuracy": {
            "value": rate(selection_correct, selection_known),
            "numerator": selection_correct,
            "denominator": selection_known,
            "denominator_description": "trials with a recorded selection_correct value",
        },
        "pick_success_rate": {
            "value": rate(pick_success, pick_known),
            "numerator": pick_success,
            "denominator": pick_known,
            "denominator_description": "trials with a recorded pick_success value",
        },
        "place_success_rate": {
            "value": rate(place_success, place_known),
            "numerator": place_success,
            "denominator": place_known,
            "denominator_description": "trials with a recorded place_success value",
        },
        "end_to_end_task_success_rate": {
            "value": rate(end_to_end_success, end_to_end_known),
            "numerator": end_to_end_success,
            "denominator": end_to_end_known,
            "denominator_description": (
                "trials with selection_correct, pick_success, and place_success "
                "all recorded (not missing)"
            ),
        },
        "average_execution_time_s": {
            "value": (sum(durations) / len(durations)) if durations else None,
            "numerator": sum(durations) if durations else 0.0,
            "denominator": len(durations),
            "denominator_description": "trials with a recorded duration_s value",
        },
    }

    return metrics, per_trial


def format_rate(entry):
    if entry["value"] is None:
        return f"N/A (0/{entry['denominator']})"
    return f"{entry['value']*100:.1f}% ({entry['numerator']}/{entry['denominator']})"


def print_report(metrics, csv_path):
    print("=" * 70)
    print(f"EVALUATION REPORT: {csv_path}")
    print("=" * 70)
    print(f"Total trial rows in file: {metrics['total_trials_in_file']}")
    print()
    print(f"1. Object Selection Accuracy : {format_rate(metrics['object_selection_accuracy'])}")
    print(f"   denominator = {metrics['object_selection_accuracy']['denominator_description']}")
    print()
    print(f"2. Pick Success Rate         : {format_rate(metrics['pick_success_rate'])}")
    print(f"   denominator = {metrics['pick_success_rate']['denominator_description']}")
    print()
    print(f"3. Place Success Rate        : {format_rate(metrics['place_success_rate'])}")
    print(f"   denominator = {metrics['place_success_rate']['denominator_description']}")
    print()
    print(f"4. End-to-End Task Success   : {format_rate(metrics['end_to_end_task_success_rate'])}")
    print(f"   denominator = {metrics['end_to_end_task_success_rate']['denominator_description']}")
    print()
    avg = metrics["average_execution_time_s"]
    if avg["value"] is None:
        print(f"5. Average Execution Time    : N/A (0/{avg['denominator']})")
    else:
        print(f"5. Average Execution Time    : {avg['value']:.2f}s ({avg['denominator']} trials)")
    print(f"   denominator = {avg['denominator_description']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM robot pick-and-place trial results.")
    parser.add_argument("csv_path", help="Path to a trial-results CSV file (see trials_template.csv)")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON instead of a text report")
    args = parser.parse_args()

    try:
        rows = load_trials(args.csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print(f"Warning: '{args.csv_path}' contains no trial rows. No metrics to report.", file=sys.stderr)
        empty_metrics, _ = compute_metrics([])
        if args.json:
            print(json.dumps(empty_metrics, indent=2))
        else:
            print_report(empty_metrics, args.csv_path)
        sys.exit(0)

    metrics, per_trial = compute_metrics(rows)

    if args.json:
        print(json.dumps({"metrics": metrics, "per_trial": per_trial}, indent=2))
    else:
        print_report(metrics, args.csv_path)


if __name__ == "__main__":
    main()
