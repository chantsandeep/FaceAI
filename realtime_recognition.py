"""
Real-Time Recognition Module  —  Anti-Proxy Edition
=====================================================
Marks attendance ONLY after passing all liveness checks:
  1. Texture analysis  — phone screens are flat, real faces have skin texture
  2. Embedding variance — static photo produces identical embeddings every frame
  3. Face size check   — phone-in-frame produces a tiny face
  4. Blink requirement — must blink at least once (photos can't blink)
  5. Multi-frame confirmation — face must appear in N consecutive frames
"""
from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from attendance_manager import mark_attendance
from utils.database import get_all_students, log_unknown_face
from utils.face_utils import (
    LivenessState,
    detect_faces,
    draw_face_box,
    find_best_match,
    get_embedding,
    save_snapshot,
)

SNAPSHOTS_DIR   = "snapshots"
UNKNOWN_DIR     = os.path.join(SNAPSHOTS_DIR, "unknown")
DEFAULT_THRESHOLD = 0.45
CONFIRM_FRAMES  = 5   # consecutive frames needed after liveness passes


def recognition_page():
    st.title("📷 Real-Time Attendance Capture")
    st.markdown(
        "Faces are verified for **liveness** before attendance is marked. "
        "Photos and phone screens are automatically rejected."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    col_feed, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.subheader("⚙️ Settings")
        threshold = st.slider(
            "Match Threshold", 0.3, 0.9, DEFAULT_THRESHOLD, 0.05,
            help="Cosine similarity cutoff. Higher = stricter.",
        )
        show_confidence = False
        show_liveness   = False
        st.markdown("---")

        # Liveness info box
        st.markdown("""
**🛡️ Anti-Proxy Checks Active:**
- 🔄 Head movement challenge
- 💡 Screen brightness detection
- 📐 Face size check
- 📊 Movement variance
        """)
        st.markdown("---")
        stop_btn = st.button("⏹ Stop Camera", use_container_width=True, type="primary")
        st.markdown("---")
        status_area = st.empty()

    with col_feed:
        frame_placeholder = st.empty()

    log_area = st.empty()

    # ── Load student DB ───────────────────────────────────────────────────────
    try:
        students = get_all_students()
    except Exception as e:
        st.error(f"Cannot connect to Redis: {e}")
        return

    if not students:
        st.warning("⚠️ No students registered. Go to **Register Student** first.")
        return

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
        return

    # ── Per-student state ─────────────────────────────────────────────────────
    liveness_states:    dict[str, LivenessState] = {}
    last_lv_result:     dict[str, dict]          = {}  # last result dict per student
    consecutive_hits:   dict[str, int]           = {}
    marked_this_session: set[str]                = set()
    attendance_log:     list[dict]               = []
    last_log_length:    int                      = 0
    unknown_frame_ctr:  int                      = 0

    st.session_state["recognition_running"] = True

    # ── Main loop ─────────────────────────────────────────────────────────────
    while st.session_state.get("recognition_running", True):
        ret, frame = cap.read()
        if not ret:
            break

        display  = frame.copy()
        faces    = detect_faces(frame)
        seen_ids: set[str] = set()

        for face in faces:
            emb   = get_embedding(face)
            match, score = find_best_match(emb, students, threshold)

            if match:
                sid = match["student_id"]
                seen_ids.add(sid)

                # Initialise liveness state for this student
                if sid not in liveness_states:
                    liveness_states[sid]  = LivenessState()
                    consecutive_hits[sid] = 0

                # Run all liveness checks
                lv = liveness_states[sid].update(face, frame, emb)
                last_lv_result[sid] = lv

                if lv["live"]:
                    consecutive_hits[sid] = consecutive_hits.get(sid, 0) + 1
                    color  = (0, 255, 0)
                    status_text = lv["reason"]
                else:
                    consecutive_hits[sid] = 0   # reset — must re-confirm after passing
                    color  = (0, 165, 255)       # orange = detected but not verified
                    status_text = lv["reason"]

                # Mark attendance only after liveness + N consecutive confirmed frames
                if (lv["live"]
                        and consecutive_hits[sid] >= CONFIRM_FRAMES
                        and sid not in marked_this_session):

                    snapshot_path = _save_attendance_snapshot(frame, sid)
                    result = mark_attendance(
                        student_id=sid,
                        name=match["name"],
                        department=match["department"],
                        snapshot_path=snapshot_path,
                    )
                    marked_this_session.add(sid)

                    entry = {
                        "Time":   datetime.now().strftime("%H:%M:%S"),
                        "ID":     sid,
                        "Name":   match["name"],
                        "Dept":   match["department"],
                        "Score":  f"{score:.2f}",
                        "Status": "✅ Marked" if result["status"] == "marked"
                                  else "⚠️ Already marked",
                    }
                    attendance_log.insert(0, entry)

                    if result["status"] == "marked":
                        status_area.success(f"✅ {match['name']} ({sid}) — Liveness verified")
                    else:
                        status_area.info(f"ℹ️ Already marked: {match['name']}")

                # Build label
                if sid in marked_this_session:
                    label = f"✓ {match['name']}"
                    color = (0, 255, 0)
                else:
                    label = match["name"]

                # Show liveness detail under box
                detail = ""
                if show_liveness and sid not in marked_this_session:
                    detail = status_text
                    if lv["live"] and consecutive_hits.get(sid, 0) < CONFIRM_FRAMES:
                        remaining = CONFIRM_FRAMES - consecutive_hits.get(sid, 0)
                        detail = f"✓ Live — confirming ({remaining} frames)"
                    elif not lv["live"]:
                        ps = lv.get("phone_score", lv.get("diff", 0))
                        detail = f"{status_text} [d:{ps:.1f}]"

            else:
                label  = "Unknown"
                color  = (0, 0, 255)
                score  = None
                detail = ""

                unknown_frame_ctr += 1
                if unknown_frame_ctr % 30 == 0:
                    unk_path = _save_unknown_snapshot(frame)
                    try:
                        log_unknown_face(unk_path)
                    except Exception:
                        pass

            draw_face_box(
                display, face, label, color,
                confidence=score if show_confidence else None,
                status=detail,
            )

        # Reset state for faces that left the frame
        for sid in list(consecutive_hits.keys()):
            if sid not in seen_ids:
                consecutive_hits[sid] = 0
                if sid not in marked_this_session and sid in liveness_states:
                    liveness_states[sid].reset()
                    last_lv_result.pop(sid, None)

        # HUD
        hud = (
            f"Faces: {len(faces)}  |  "
            f"DB: {len(students)} students  |  "
            f"Marked: {len(marked_this_session)}"
        )
        cv2.putText(display, hud,
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Draw head movement challenge instruction at top of frame
        for sid in list(last_lv_result.keys()):
            if sid in marked_this_session:
                continue
            lv_r = last_lv_result[sid]
            if not lv_r.get("challenge_passed", True):
                reason = lv_r.get("reason", "")
                if "Turn" in reason:
                    text = reason.split("(")[0].strip()
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    tx = max(0, (display.shape[1] - tw) // 2)
                    cv2.rectangle(display, (tx - 8, 8), (tx + tw + 8, th + 24), (0, 120, 255), -1)
                    cv2.putText(display, text, (tx, th + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                break

        frame_placeholder.image(
            cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_column_width=True,
        )

        # Update log table only when new entry added
        if attendance_log and len(attendance_log) != last_log_length:
            import pandas as pd
            last_log_length = len(attendance_log)
            log_area.dataframe(
                pd.DataFrame(attendance_log[:10]),
                use_container_width=True,
                hide_index=True,
                height=388,
            )

        if stop_btn:
            break

    cap.release()
    st.session_state["recognition_running"] = False
    st.info(
        f"📷 Camera stopped. "
        f"**{len(marked_this_session)}** student(s) marked present this session."
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save_attendance_snapshot(frame: np.ndarray, student_id: str) -> str:
    folder   = os.path.join(SNAPSHOTS_DIR, student_id)
    filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    return save_snapshot(frame, folder, filename)


def _save_unknown_snapshot(frame: np.ndarray) -> str:
    filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
    return save_snapshot(frame, UNKNOWN_DIR, filename)
