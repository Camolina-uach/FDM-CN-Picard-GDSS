"""Compare regenerated CSV outputs with the archived reference tables.

Values above the round-off scale use rtol=1e-3 and atol=1e-10.  Quantities
whose reference magnitude is at or below 1e-10 are checked with the absolute
tolerance only because their platform-to-platform relative variation is not
meaningful.
"""
from __future__ import annotations

import csv
import math
import os


ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "paper", "results_fdm")
REFERENCES = os.path.join(ROOT, "reference_results")
RTOL = 1e-3
ATOL = 1e-10
ROUND_OFF_SCALE = 1e-10
IGNORED_COLUMNS = {
    # Wall-clock measurements are intentionally reported but are not suitable
    # as cross-platform regression targets.
    "cpu", "lu_factor_s", "lu_total_setup_s", "cg_total_setup_s",
    "lu_total_s", "lu_step_s", "cg_total_s", "cg_step_s",
    "lu_wave_fraction", "cg_wave_fraction",
}


def _read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_csv(reference_path, current_path):
    reference = _read_csv(reference_path)
    current = _read_csv(current_path)
    if len(reference) != len(current):
        raise AssertionError(
            f"row-count mismatch for {os.path.basename(current_path)}: "
            f"reference={len(reference)}, current={len(current)}"
        )
    failures = []
    for row_index, (expected, actual) in enumerate(zip(reference, current), start=2):
        if set(expected) != set(actual):
            raise AssertionError(f"column mismatch for {os.path.basename(current_path)}")
        for column in expected:
            if column in IGNORED_COLUMNS:
                continue
            evalue = _as_float(expected[column])
            avalue = _as_float(actual[column])
            if evalue is None or avalue is None:
                if expected[column] != actual[column]:
                    failures.append((row_index, column, expected[column], actual[column]))
                continue
            if not (math.isfinite(evalue) and math.isfinite(avalue)):
                if evalue != avalue:
                    failures.append((row_index, column, evalue, avalue))
                continue
            tolerance = ATOL if abs(evalue) <= ROUND_OFF_SCALE else ATOL + RTOL * abs(evalue)
            if abs(avalue - evalue) > tolerance:
                failures.append((row_index, column, evalue, avalue))
    if failures:
        details = "\n".join(
            f"  row {row}, {column}: expected {expected}, got {actual}"
            for row, column, expected, actual in failures[:20]
        )
        raise AssertionError(f"reference mismatch in {os.path.basename(current_path)}:\n{details}")


def verify_if_available(skip_names=()):
    if not os.path.isdir(REFERENCES):
        print("  reference_results/ is not present; comparison skipped")
        return []
    checked = []
    for name in sorted(os.listdir(REFERENCES)):
        if not name.endswith(".csv"):
            continue
        if name in skip_names:
            print(f"  SKIP {name} (reduced run)")
            continue
        reference_path = os.path.join(REFERENCES, name)
        current_path = os.path.join(RESULTS, name)
        if not os.path.exists(current_path):
            raise FileNotFoundError(f"missing regenerated result: {current_path}")
        compare_csv(reference_path, current_path)
        checked.append(name)
        print(f"  PASS {name}")
    if not checked:
        raise RuntimeError("reference_results/ contains no CSV files")
    print(f"  {len(checked)} archived tables passed (rtol={RTOL:g}, atol={ATOL:g})")
    return checked


if __name__ == "__main__":
    verify_if_available()
