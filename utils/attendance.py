"""
Attendance logging utilities — CSV persistence with per-day deduplication.
"""
from __future__ import annotations

import csv
import os
from datetime import date, datetime
from pathlib import Path

ATTENDANCE_DIR = "attendance_logs"
CSV_COLUMNS = ["student_id", "name", "department", "date", "timestamp", "snapshot"]


def _csv_path(for_date: date | None = None) -> str:
    """Return the CSV file path for a given date (defaults to today)."""
    d = for_date or date.today()
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    return os.path.join(ATTENDANCE_DIR, f"attendance_{d.isoformat()}.csv")


def _ensure_header(path: str) -> None:
    """Write CSV header if the file is new / empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def already_marked(student_id: str, for_date: date | None = None) -> bool:
    """Return True if the student already has an entry for the given date."""
    path = _csv_path(for_date)
    if not os.path.exists(path):
        return False
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("student_id") == student_id:
                return True
    return False


def log_attendance(
    student_id: str,
    name: str,
    department: str,
    snapshot_path: str,
    for_date: date | None = None,
) -> bool:
    """
    Append an attendance record to today's CSV.
    Returns True if written, False if already marked (deduplicated).
    """
    target_date = for_date or date.today()
    if already_marked(student_id, target_date):
        return False

    path = _csv_path(target_date)
    _ensure_header(path)

    row = {
        "student_id": student_id,
        "name": name,
        "department": department,
        "date": target_date.isoformat(),
        "timestamp": datetime.now().isoformat(),
        "snapshot": snapshot_path,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
    return True


def get_attendance_dates() -> list[str]:
    """Return sorted list of dates (YYYY-MM-DD) that have attendance logs."""
    if not os.path.exists(ATTENDANCE_DIR):
        return []
    dates = []
    for fname in os.listdir(ATTENDANCE_DIR):
        if fname.startswith("attendance_") and fname.endswith(".csv"):
            d = fname.replace("attendance_", "").replace(".csv", "")
            dates.append(d)
    return sorted(dates, reverse=True)


def get_attendance_records(for_date: str | None = None) -> list[dict]:
    """
    Return all attendance records for a given date string (YYYY-MM-DD).
    Defaults to today.
    """
    target = for_date or date.today().isoformat()
    path = os.path.join(ATTENDANCE_DIR, f"attendance_{target}.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_attendance_csv_path(for_date: str) -> str | None:
    """Return the CSV path for a date string, or None if it doesn't exist."""
    path = os.path.join(ATTENDANCE_DIR, f"attendance_{for_date}.csv")
    return path if os.path.exists(path) else None
