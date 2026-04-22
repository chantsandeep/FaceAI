"""
Management dashboard — view/delete students, attendance records, unknown faces, export CSV.
"""
from __future__ import annotations

import os
import streamlit as st
import pandas as pd
from datetime import date

from utils.database import get_all_students, delete_student, get_unknown_faces
from utils.attendance import (
    get_attendance_dates,
    get_attendance_records,
    get_attendance_csv_path,
)


def management_page():
    st.title("⚙️ Management Dashboard")

    tab1, tab2, tab3 = st.tabs(["👥 Students", "📅 Attendance Records", "🚨 Unknown Faces"])

    with tab1:
        _students_tab()

    with tab2:
        _attendance_tab()

    with tab3:
        _unknown_faces_tab()


# ─── Students Tab ──────────────────────────────────────────────────────────────

def _students_tab():
    st.subheader("Registered Students")

    students = get_all_students()

    if not students:
        st.info("No students registered yet.")
        return

    # Build display dataframe (exclude raw embedding)
    rows = [
        {
            "Student ID": s["student_id"],
            "Name": s["name"],
            "Department": s["department"],
            "Registered At": s.get("registered_at", "—"),
        }
        for s in students
    ]
    df = pd.DataFrame(rows).sort_values("Student ID").reset_index(drop=True)

    st.metric("Total Registered", len(df))
    st.dataframe(df, use_container_width=True, hide_index=True, height=320)

    # Export
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Student List (CSV)",
        data=csv_bytes,
        file_name=f"students_{date.today().isoformat()}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Delete Student")
    st.warning("⚠️ Deleting a student removes their embedding from Redis. Attendance history is preserved.")

    student_ids = [s["student_id"] for s in students]
    selected_id = st.selectbox("Select Student ID to delete", options=["— select —"] + student_ids)

    if selected_id and selected_id != "— select —":
        # Show student info
        student = next((s for s in students if s["student_id"] == selected_id), None)
        if student:
            st.markdown(f"**Name:** {student['name']}  |  **Department:** {student['department']}")

        confirm = st.checkbox(f"I confirm I want to delete student **{selected_id}**")
        if st.button("🗑️ Delete Student", type="primary", disabled=not confirm):
            delete_student(selected_id)
            st.success(f"Student {selected_id} deleted successfully.")
            st.rerun()


# ─── Attendance Tab ────────────────────────────────────────────────────────────

def _attendance_tab():
    st.subheader("Attendance Records")

    dates = get_attendance_dates()

    if not dates:
        st.info("No attendance records found.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_date = st.selectbox("Select Date", options=dates, index=0)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_path = get_attendance_csv_path(selected_date)
        if csv_path:
            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇️ Download CSV",
                    data=f.read(),
                    file_name=os.path.basename(csv_path),
                    mime="text/csv",
                    use_container_width=True,
                )

    records = get_attendance_records(selected_date)

    if not records:
        st.info(f"No records for {selected_date}.")
        return

    df = pd.DataFrame(records)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Present", len(df))
    if "department" in df.columns:
        m2.metric("Departments", df["department"].nunique())
    if "timestamp" in df.columns:
        first_in = df["timestamp"].min()
        m3.metric("First Entry", str(first_in)[:19] if first_in else "—")

    # Filter by department
    if "department" in df.columns:
        depts = ["All"] + sorted(df["department"].unique().tolist())
        dept_filter = st.selectbox("Filter by Department", depts)
        if dept_filter != "All":
            df = df[df["department"] == dept_filter]

    # Search by name / ID
    search = st.text_input("🔍 Search by name or ID", placeholder="Type to filter...")
    if search:
        mask = (
            df["name"].str.contains(search, case=False, na=False)
            | df["student_id"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    st.dataframe(df, use_container_width=True, hide_index=True, height=360)

    # Attendance chart
    if "department" in df.columns and len(df) > 0:
        st.markdown("---")
        st.subheader("Attendance by Department")
        dept_counts = df["department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        st.bar_chart(dept_counts.set_index("Department"))


# ─── Unknown Faces Tab ─────────────────────────────────────────────────────────

def _unknown_faces_tab():
    st.subheader("Unknown Face Detections")
    st.markdown("Faces detected during recognition sessions that did not match any registered student.")

    unknown = get_unknown_faces(limit=100)

    if not unknown:
        st.info("No unknown face events logged.")
        return

    st.metric("Total Unknown Events (last 100)", len(unknown))

    rows = [
        {
            "Timestamp": u.get("timestamp", "—"),
            "Snapshot Path": u.get("snapshot", "—"),
        }
        for u in unknown
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=320)

    # Show snapshot thumbnails for existing files
    st.markdown("---")
    st.subheader("Recent Snapshots")
    cols = st.columns(4)
    shown = 0
    for u in unknown[:20]:
        snap = u.get("snapshot", "")
        if snap and os.path.exists(snap):
            with cols[shown % 4]:
                st.image(snap, caption=u.get("timestamp", "")[:19], use_column_width=True)
            shown += 1

    if shown == 0:
        st.info("No snapshot files found on disk for recent events.")
