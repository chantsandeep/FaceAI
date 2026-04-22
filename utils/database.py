"""
Redis database utility
======================
Handles all Redis operations:
  - Student embeddings and metadata
  - Attendance records (secondary store; CSV is source of truth)
  - Unknown face event log
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime

import numpy as np
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None) or None
REDIS_DB = int(os.getenv("REDIS_DB", 0))


def get_redis_client() -> redis.Redis:
    """Create and return a Redis client. Raises ConnectionError if unreachable."""
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=False,
            socket_connect_timeout=3,
        )
        client.ping()
        return client
    except redis.ConnectionError as e:
        raise ConnectionError(
            f"Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}. Error: {e}"
        )


# ─── Student Operations ────────────────────────────────────────────────────────

def save_student(
    student_id: str, name: str, department: str, embedding: np.ndarray
) -> bool:
    """Save student details and face embedding to Redis."""
    client = get_redis_client()
    key = f"student:{student_id}"
    data = {
        "student_id": student_id,
        "name": name,
        "department": department,
        "registered_at": datetime.now().isoformat(),
        "embedding": json.dumps(embedding.tolist()),
    }
    # hmset works on all Redis versions including 3.x (hset mapping= requires Redis 4+)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client.hmset(key, {k: str(v) for k, v in data.items()})
    client.sadd("students:all", student_id)
    return True


def get_student(student_id: str) -> dict | None:
    """Retrieve a student record by ID."""
    client = get_redis_client()
    key = f"student:{student_id}"
    raw = client.hgetall(key)
    if not raw:
        return None
    decoded = {k.decode(): v.decode() for k, v in raw.items()}
    decoded["embedding"] = np.array(
        json.loads(decoded["embedding"]), dtype=np.float32
    )
    return decoded


def get_all_students() -> list[dict]:
    """Return all registered students with their embeddings."""
    client = get_redis_client()
    ids = client.smembers("students:all")
    students = []
    for sid in ids:
        student = get_student(sid.decode())
        if student:
            students.append(student)
    return students


def delete_student(student_id: str) -> bool:
    """Remove a student and their embedding from Redis."""
    client = get_redis_client()
    client.delete(f"student:{student_id}")
    client.srem("students:all", student_id)
    return True


def student_exists(student_id: str) -> bool:
    """Check if a student ID is already registered."""
    client = get_redis_client()
    return client.exists(f"student:{student_id}") > 0


# ─── Attendance Operations (Redis secondary store) ─────────────────────────────

def redis_log_attendance(
    student_id: str,
    name: str,
    department: str,
    snapshot_path: str,
    target_date_str: str,
) -> None:
    """
    Store an attendance record in Redis.
    Key: attendance:{date}:{student_id}
    Also maintains a set per date and a set per student for fast lookups.
    TTL: 90 days.
    """
    try:
        client = get_redis_client()
        key = f"attendance:{target_date_str}:{student_id}"
        record = {
            "student_id": student_id,
            "name": name,
            "department": department,
            "date": target_date_str,
            "timestamp": datetime.now().isoformat(),
            "snapshot": snapshot_path,
        }
        client.set(key, json.dumps(record), ex=90 * 86400)
        client.sadd(f"attendance:date:{target_date_str}", student_id)
        client.sadd(f"attendance:student:{student_id}", target_date_str)
    except Exception:
        pass  # Non-fatal; CSV is source of truth


def redis_get_attendance_by_date(date_str: str) -> list[dict]:
    """Return all attendance records for a date from Redis."""
    try:
        client = get_redis_client()
        student_ids = client.smembers(f"attendance:date:{date_str}")
        records = []
        for sid in student_ids:
            raw = client.get(f"attendance:{date_str}:{sid.decode()}")
            if raw:
                records.append(json.loads(raw.decode()))
        return records
    except Exception:
        return []


def redis_get_student_attendance_history(student_id: str) -> list[str]:
    """Return list of date strings when a student was present."""
    try:
        client = get_redis_client()
        dates = client.smembers(f"attendance:student:{student_id}")
        return sorted([d.decode() for d in dates], reverse=True)
    except Exception:
        return []


# ─── Unknown Face Log ──────────────────────────────────────────────────────────

def log_unknown_face(snapshot_path: str) -> None:
    """Log an unknown face detection event to Redis."""
    try:
        client = get_redis_client()
        timestamp = datetime.now().isoformat()
        client.lpush(
            "unknown_faces",
            json.dumps({"timestamp": timestamp, "snapshot": snapshot_path}),
        )
        client.ltrim("unknown_faces", 0, 499)  # Keep last 500 entries
    except Exception:
        pass


def get_unknown_faces(limit: int = 50) -> list[dict]:
    """Retrieve recent unknown face log entries."""
    try:
        client = get_redis_client()
        raw = client.lrange("unknown_faces", 0, limit - 1)
        return [json.loads(r.decode()) for r in raw]
    except Exception:
        return []


# ─── Health ────────────────────────────────────────────────────────────────────

def redis_health_check() -> bool:
    """Return True if Redis is reachable."""
    try:
        get_redis_client()
        return True
    except Exception:
        return False
