"""
Registration Module
===================
Capture face samples via webcam, generate averaged embeddings,
and store student data in Redis.

Features:
  - Multi-frame capture with live preview
  - Face detection validation per frame
  - Averaged embedding for robustness
  - Re-registration (update) support
  - Sample images saved to dataset/
"""
from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from utils.database import (
    delete_student,
    get_all_students,
    get_student,
    save_student,
    student_exists,
)
from utils.face_utils import detect_faces, frames_to_embedding, save_snapshot

DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"


def registration_page():
    st.title("📋 Student Registration")
    st.markdown("Register a new student by capturing their face via webcam.")

    tab_new, tab_update, tab_list = st.tabs(
        ["➕ New Student", "🔄 Update / Re-register", "📄 Registered Students"]
    )

    with tab_new:
        _register_form(update_mode=False)

    with tab_update:
        _register_form(update_mode=True)

    with tab_list:
        _student_list()


# ─── Registration Form ─────────────────────────────────────────────────────────

def _register_form(update_mode: bool = False):
    label = "Update Student" if update_mode else "New Student"
    form_key = "update_form" if update_mode else "registration_form"

    if update_mode:
        st.markdown(
            "Re-capture face samples for an existing student. "
            "This **replaces** their stored embedding."
        )
    else:
        st.markdown("Fill in the student details and capture face samples via webcam.")

    with st.form(form_key):
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input(
                "Student ID",
                placeholder="e.g. STU001",
                help="Unique identifier for the student.",
            )
            name = st.text_input("Full Name", placeholder="e.g. John Doe")
        with col2:
            department = st.text_input("Department", placeholder="e.g. Computer Science")
            num_samples = st.slider(
                "Capture samples",
                min_value=5, max_value=30, value=10,
                help="More samples = more robust embedding.",
            )

        submitted = st.form_submit_button(
            f"{'🔄 Update' if update_mode else '🚀 Start'} Registration",
            use_container_width=True,
        )

    if not submitted:
        return

    # ── Validation ─────────────────────────────────────────────────────────────
    if not student_id.strip() or not name.strip() or not department.strip():
        st.error("❌ Please fill in all fields.")
        return

    student_id = student_id.strip().upper()

    if update_mode:
        if not student_exists(student_id):
            st.error(f"❌ Student ID **{student_id}** is not registered. Use 'New Student' tab.")
            return
        # Delete old record before re-registering
        delete_student(student_id)
        st.info(f"ℹ️ Old record for **{student_id}** removed. Capturing new samples...")
    else:
        if student_exists(student_id):
            st.warning(
                f"⚠️ Student ID **{student_id}** is already registered. "
                "Use the **Update / Re-register** tab to replace their embedding."
            )
            return

    # ── Webcam capture ─────────────────────────────────────────────────────────
    st.info(f"📷 Opening webcam — capturing {num_samples} frames. Look directly at the camera.")
    progress = st.progress(0, text="Initializing camera...")
    status_box = st.empty()
    preview = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Check your camera connection.")
        return

    captured_frames: list[np.ndarray] = []
    student_dir = os.path.join(DATASET_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)

    frame_count = 0
    attempts = 0
    max_attempts = num_samples * 15  # Allow retries for frames without faces

    while len(captured_frames) < num_samples and attempts < max_attempts:
        ret, frame = cap.read()
        attempts += 1
        if not ret:
            continue

        faces = detect_faces(frame)
        display = frame.copy()

        if faces:
            face = faces[0]
            bbox = face.bbox.astype(int)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"Captured: {len(captured_frames) + 1}/{num_samples}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            captured_frames.append(frame.copy())
            frame_count += 1

            # Save sample image to dataset/
            img_path = os.path.join(student_dir, f"sample_{frame_count}.jpg")
            cv2.imwrite(img_path, frame)
        else:
            cv2.putText(
                display,
                "No face detected — adjust position",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        preview.image(
            cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_column_width=True,
        )
        progress.progress(
            len(captured_frames) / num_samples,
            text=f"Captured {len(captured_frames)}/{num_samples} frames...",
        )

    cap.release()
    preview.empty()

    if len(captured_frames) < num_samples:
        st.error(
            f"❌ Only captured {len(captured_frames)}/{num_samples} frames. "
            "Ensure your face is clearly visible and well-lit."
        )
        return

    # ── Generate embedding ─────────────────────────────────────────────────────
    status_box.info("⚙️ Generating face embedding from captured frames...")
    embedding = frames_to_embedding(captured_frames)

    if embedding is None:
        st.error(
            "❌ Failed to extract face embedding. "
            "Try again with better lighting and a clear view of your face."
        )
        return

    # ── Save to Redis ──────────────────────────────────────────────────────────
    save_student(student_id, name.strip(), department.strip(), embedding)
    status_box.empty()
    progress.empty()

    action = "updated" if update_mode else "registered"
    st.success(
        f"✅ Student **{name}** (ID: `{student_id}`) {action} successfully "
        f"with **{len(captured_frames)}** samples!"
    )
    st.balloons()


# ─── Student List ──────────────────────────────────────────────────────────────

def _student_list():
    st.subheader("All Registered Students")

    try:
        students = get_all_students()
    except Exception as e:
        st.error(f"Cannot connect to Redis: {e}")
        return

    if not students:
        st.info("No students registered yet.")
        return

    import pandas as pd

    rows = [
        {
            "Student ID": s["student_id"],
            "Name": s["name"],
            "Department": s["department"],
            "Registered At": s.get("registered_at", "—")[:19],
            "Samples": len(
                os.listdir(os.path.join(DATASET_DIR, s["student_id"]))
            ) if os.path.isdir(os.path.join(DATASET_DIR, s["student_id"])) else 0,
        }
        for s in students
    ]
    df = pd.DataFrame(rows).sort_values("Student ID").reset_index(drop=True)

    st.metric("Total Registered", len(df))
    st.dataframe(df, use_container_width=True, hide_index=True, height=320)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Student List (CSV)",
        data=csv_bytes,
        file_name="registered_students.csv",
        mime="text/csv",
    )
