"""
Attendance Manager Module
=========================
Handles all attendance record operations:
  - Marking attendance (CSV + Redis, deduplicated per day)
  - Querying records by date / student / department
  - Daily summary generation
  - Automatic daily CSV file creation
"""
from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from utils.attendance import (
    log_attendance,
    already_marked,
    get_attendance_records,
    get_attendance_dates,
    get_attendance_csv_path,
    ATTENDANCE_DIR,
    CSV_COLUMNS,
)
from utils.database import get_redis_client, get_all_students

# ─── Core Mark Attendance ──────────────────────────────────────────────────────

def mark_attendance(
    student_id: str,
    name: str,
    department: str,
    snapshot_path: str = "",
    for_date: Optional[date] = None,
) -> dict:
    """
    Mark attendance for a student.

    Returns a result dict:
        {
            "success": bool,
            "status": "marked" | "duplicate" | "error",
            "message": str,
            "record": dict | None,
        }
    """
    target_date = for_date or date.today()

    try:
        # Write to CSV (deduplicated)
        written = log_attendance(
            student_id=student_id,
            name=name,
            department=department,
            snapshot_path=snapshot_path,
            for_date=target_date,
        )

        if not written:
            return {
                "success": False,
                "status": "duplicate",
                "message": f"{name} ({student_id}) already marked present on {target_date.isoformat()}.",
                "record": None,
            }

        # Also persist to Redis for fast querying
        _redis_log_attendance(
            student_id=student_id,
            name=name,
            department=department,
            snapshot_path=snapshot_path,
            target_date=target_date,
        )

        record = {
            "student_id": student_id,
            "name": name,
            "department": department,
            "date": target_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "snapshot": snapshot_path,
        }

        return {
            "success": True,
            "status": "marked",
            "message": f"Attendance marked for {name} ({student_id}).",
            "record": record,
        }

    except Exception as exc:
        return {
            "success": False,
            "status": "error",
            "message": f"Error marking attendance: {exc}",
            "record": None,
        }


def _redis_log_attendance(
    student_id: str,
    name: str,
    department: str,
    snapshot_path: str,
    target_date: date,
) -> None:
    """Store attendance record in Redis for fast lookup."""
    try:
        client = get_redis_client()
        key = f"attendance:{target_date.isoformat()}:{student_id}"
        record = {
            "student_id": student_id,
            "name": name,
            "department": department,
            "date": target_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "snapshot": snapshot_path,
        }
        client.set(key, json.dumps(record))
        # Add to daily set for fast enumeration
        client.sadd(f"attendance:date:{target_date.isoformat()}", student_id)
        # Add to student history set
        client.sadd(f"attendance:student:{student_id}", target_date.isoformat())
        # Keep 90 days of date keys
        client.expire(key, 90 * 86400)
    except Exception:
        pass  # Redis failure is non-fatal; CSV is the source of truth


# ─── Query Functions ───────────────────────────────────────────────────────────

def get_daily_summary(for_date: Optional[str] = None) -> dict:
    """
    Return a summary dict for a given date (YYYY-MM-DD string).
    {
        "date": str,
        "total_present": int,
        "by_department": {dept: count},
        "records": [list of dicts],
        "first_entry": str | None,
        "last_entry": str | None,
    }
    """
    target = for_date or date.today().isoformat()
    records = get_attendance_records(target)

    by_dept: dict[str, int] = {}
    for r in records:
        dept = r.get("department", "Unknown")
        by_dept[dept] = by_dept.get(dept, 0) + 1

    timestamps = [r.get("timestamp", "") for r in records if r.get("timestamp")]
    timestamps_sorted = sorted(timestamps)

    return {
        "date": target,
        "total_present": len(records),
        "by_department": by_dept,
        "records": records,
        "first_entry": timestamps_sorted[0][:19] if timestamps_sorted else None,
        "last_entry": timestamps_sorted[-1][:19] if timestamps_sorted else None,
    }


def get_student_attendance_history(student_id: str) -> list[dict]:
    """Return all attendance records across all dates for a specific student."""
    all_dates = get_attendance_dates()
    history = []
    for d in all_dates:
        records = get_attendance_records(d)
        for r in records:
            if r.get("student_id") == student_id:
                history.append(r)
    return history


def get_attendance_range(start_date: str, end_date: str) -> list[dict]:
    """
    Return all attendance records between start_date and end_date (inclusive).
    Both dates are YYYY-MM-DD strings.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    all_records = []

    current = start
    while current <= end:
        records = get_attendance_records(current.isoformat())
        all_records.extend(records)
        current += timedelta(days=1)

    return all_records


def get_attendance_by_department(department: str, for_date: Optional[str] = None) -> list[dict]:
    """Return attendance records filtered by department for a given date."""
    target = for_date or date.today().isoformat()
    records = get_attendance_records(target)
    return [r for r in records if r.get("department", "").lower() == department.lower()]


def get_absentees(for_date: Optional[str] = None) -> list[dict]:
    """
    Return list of registered students who are NOT present on the given date.
    Requires Redis to be available for student list.
    """
    target = for_date or date.today().isoformat()
    present_ids = {r["student_id"] for r in get_attendance_records(target)}

    try:
        all_students = get_all_students()
    except Exception:
        return []

    return [
        s for s in all_students
        if s["student_id"] not in present_ids
    ]


# ─── Reporting Helpers ─────────────────────────────────────────────────────────

def build_attendance_dataframe(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    department: Optional[str] = None,
    student_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a filtered Pandas DataFrame of attendance records.
    If no date range given, returns all available records.
    """
    if start_date and end_date:
        records = get_attendance_range(start_date, end_date)
    elif start_date:
        records = get_attendance_records(start_date)
    else:
        # All available dates
        all_dates = get_attendance_dates()
        records = []
        for d in all_dates:
            records.extend(get_attendance_records(d))

    if not records:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.DataFrame(records)

    if department:
        df = df[df["department"].str.lower() == department.lower()]
    if student_id:
        df = df[df["student_id"] == student_id]

    # Ensure correct dtypes
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    return df.reset_index(drop=True)


def get_attendance_stats() -> dict:
    """
    Return aggregate statistics across all attendance data.
    {
        "total_records": int,
        "unique_students": int,
        "total_days": int,
        "avg_daily_attendance": float,
        "most_active_dept": str | None,
    }
    """
    all_dates = get_attendance_dates()
    if not all_dates:
        return {
            "total_records": 0,
            "unique_students": 0,
            "total_days": 0,
            "avg_daily_attendance": 0.0,
            "most_active_dept": None,
        }

    all_records: list[dict] = []
    for d in all_dates:
        all_records.extend(get_attendance_records(d))

    unique_students = len({r["student_id"] for r in all_records})
    dept_counts: dict[str, int] = {}
    for r in all_records:
        dept = r.get("department", "Unknown")
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    most_active = max(dept_counts, key=dept_counts.get) if dept_counts else None

    return {
        "total_records": len(all_records),
        "unique_students": unique_students,
        "total_days": len(all_dates),
        "avg_daily_attendance": round(len(all_records) / len(all_dates), 1) if all_dates else 0.0,
        "most_active_dept": most_active,
    }


# ─── Auto Daily File ───────────────────────────────────────────────────────────

def ensure_today_csv() -> str:
    """
    Ensure today's attendance CSV exists (with header).
    Returns the file path.
    """
    from utils.attendance import _csv_path, _ensure_header
    path = _csv_path(date.today())
    _ensure_header(path)
    return path
