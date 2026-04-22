"""
Reports Module
==============
Streamlit page for attendance analytics and reporting.

Features:
  - Filter by date range, student, department
  - Summary metrics
  - Matplotlib charts (daily trend, dept breakdown, hourly heatmap)
  - Full data table with search
  - CSV export
  - Absentee report
"""
from __future__ import annotations

import io
import os
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

from attendance_manager import (
    build_attendance_dataframe,
    get_attendance_stats,
    get_absentees,
    get_daily_summary,
)
from utils.attendance import get_attendance_dates
from utils.database import get_all_students


# ─── Colour palette (dark-mode friendly) ──────────────────────────────────────
_BG = "#0e1117"
_SURFACE = "#161b22"
_ACCENT = "#1f6feb"
_GREEN = "#3fb950"
_RED = "#f85149"
_YELLOW = "#d29922"
_TEXT = "#c9d1d9"
_GRID = "#21262d"


def _apply_dark_style(fig: plt.Figure, ax) -> None:
    """Apply dark-mode styling to a Matplotlib figure."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_SURFACE)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(color=_GRID, linestyle="--", linewidth=0.5, alpha=0.7)


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return buf.read()


# ─── Chart builders ───────────────────────────────────────────────────────────

def _chart_daily_trend(df: pd.DataFrame) -> plt.Figure:
    """Line chart: attendance count per day."""
    if df.empty or "date" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=_TEXT)
        _apply_dark_style(fig, ax)
        return fig

    daily = df.groupby("date").size().reset_index(name="count")
    daily = daily.sort_values("date")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(
        daily["date"].astype(str), daily["count"],
        marker="o", color=_ACCENT, linewidth=2, markersize=5,
    )
    ax.fill_between(
        daily["date"].astype(str), daily["count"],
        alpha=0.15, color=_ACCENT,
    )
    ax.set_title("Daily Attendance Trend", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Students Present")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=30, ha="right")
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


def _chart_dept_breakdown(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart: attendance by department."""
    if df.empty or "department" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=_TEXT)
        _apply_dark_style(fig, ax)
        return fig

    dept_counts = df["department"].value_counts()
    colors = [_ACCENT, _GREEN, _YELLOW, _RED, "#8b949e", "#a371f7"]

    fig, ax = plt.subplots(figsize=(7, max(3, len(dept_counts) * 0.6 + 1)))
    bars = ax.barh(
        dept_counts.index.tolist(),
        dept_counts.values.tolist(),
        color=colors[: len(dept_counts)],
        edgecolor="none",
        height=0.55,
    )
    ax.bar_label(bars, padding=4, color=_TEXT, fontsize=9)
    ax.set_title("Attendance by Department", fontsize=12, fontweight="bold")
    ax.set_xlabel("Students Present")
    ax.invert_yaxis()
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


def _chart_hourly_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Bar chart: attendance entries by hour of day."""
    if df.empty or "timestamp" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=_TEXT)
        _apply_dark_style(fig, ax)
        return fig

    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No timestamp data", ha="center", va="center", color=_TEXT)
        _apply_dark_style(fig, ax)
        return fig

    hours = ts.dt.hour.value_counts().sort_index()
    all_hours = pd.Series(0, index=range(24))
    all_hours.update(hours)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bar_colors = [_ACCENT if v > 0 else _GRID for v in all_hours.values]
    ax.bar(all_hours.index, all_hours.values, color=bar_colors, edgecolor="none", width=0.7)
    ax.set_title("Attendance by Hour of Day", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour (24h)")
    ax.set_ylabel("Entries")
    ax.set_xticks(range(0, 24, 2))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


def _chart_top_students(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Bar chart: students with most attendance days."""
    if df.empty or "student_id" not in df.columns:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=_TEXT)
        _apply_dark_style(fig, ax)
        return fig

    counts = df.groupby(["student_id", "name"]).size().reset_index(name="days")
    counts["label"] = counts["name"] + "\n(" + counts["student_id"] + ")"
    counts = counts.nlargest(top_n, "days")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(
        counts["label"], counts["days"],
        color=_GREEN, edgecolor="none", width=0.55,
    )
    ax.bar_label(bars, padding=3, color=_TEXT, fontsize=8)
    ax.set_title(f"Top {top_n} Students by Attendance Days", fontsize=12, fontweight="bold")
    ax.set_ylabel("Days Present")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    return fig


# ─── Main Page ─────────────────────────────────────────────────────────────────

def reports_page():
    st.title("📊 Attendance Reports")
    st.markdown("Analyze attendance data with filters, charts, and CSV exports.")

    # ── Sidebar-style filter panel ─────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)

        available_dates = get_attendance_dates()
        today_str = date.today().isoformat()

        with fcol1:
            date_mode = st.selectbox(
                "Date Range",
                ["Today", "Last 7 Days", "Last 30 Days", "All Time", "Custom Range"],
            )

        # Resolve date range
        if date_mode == "Today":
            start_str = today_str
            end_str = today_str
        elif date_mode == "Last 7 Days":
            start_str = (date.today() - timedelta(days=6)).isoformat()
            end_str = today_str
        elif date_mode == "Last 30 Days":
            start_str = (date.today() - timedelta(days=29)).isoformat()
            end_str = today_str
        elif date_mode == "All Time":
            start_str = available_dates[-1] if available_dates else today_str
            end_str = available_dates[0] if available_dates else today_str
        else:  # Custom Range
            with fcol2:
                start_str = st.date_input(
                    "From",
                    value=date.today() - timedelta(days=6),
                    max_value=date.today(),
                ).isoformat()
            with fcol3:
                end_str = st.date_input(
                    "To",
                    value=date.today(),
                    max_value=date.today(),
                ).isoformat()

        # Department filter
        try:
            all_students = get_all_students()
            depts = sorted({s["department"] for s in all_students})
        except Exception:
            depts = []

        with fcol2 if date_mode != "Custom Range" else fcol4:
            dept_filter = st.selectbox("Department", ["All"] + depts)

        with fcol3 if date_mode != "Custom Range" else fcol1:
            search_query = st.text_input("Search Name / ID", placeholder="Leave blank for all")

    # ── Load data ──────────────────────────────────────────────────────────────
    df = build_attendance_dataframe(
        start_date=start_str,
        end_date=end_str,
        department=dept_filter if dept_filter != "All" else None,
    )

    if search_query:
        mask = (
            df["name"].str.contains(search_query, case=False, na=False)
            | df["student_id"].str.contains(search_query, case=False, na=False)
        )
        df = df[mask]

    # ── Top-level metrics ──────────────────────────────────────────────────────
    st.markdown("---")
    stats = get_attendance_stats()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Records (filtered)", len(df))
    m2.metric("Unique Students", df["student_id"].nunique() if not df.empty else 0)
    m3.metric("Total Registered", stats["total_records"])
    m4.metric("Avg Daily Attendance", stats["avg_daily_attendance"])
    m5.metric("Top Department", stats["most_active_dept"] or "—")

    if df.empty:
        st.info("No attendance records match the selected filters.")
        _absentee_section(today_str)
        return

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_trend = _chart_daily_trend(df)
        st.pyplot(fig_trend, use_container_width=True)
        plt.close(fig_trend)

    with chart_col2:
        fig_dept = _chart_dept_breakdown(df)
        st.pyplot(fig_dept, use_container_width=True)
        plt.close(fig_dept)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig_hourly = _chart_hourly_heatmap(df)
        st.pyplot(fig_hourly, use_container_width=True)
        plt.close(fig_hourly)

    with chart_col4:
        fig_top = _chart_top_students(df)
        st.pyplot(fig_top, use_container_width=True)
        plt.close(fig_top)

    # ── Data Table ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Attendance Records")

    display_df = df.copy()
    if "timestamp" in display_df.columns:
        display_df["timestamp"] = display_df["timestamp"].astype(str).str[:19]
    if "date" in display_df.columns:
        display_df["date"] = display_df["date"].astype(str)

    # Drop snapshot column from display (path is long and not useful in table)
    display_cols = [c for c in display_df.columns if c != "snapshot"]
    st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True, height=400)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Export")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        filename = f"attendance_{start_str}_to_{end_str}.csv"
        st.download_button(
            "📥 Download Filtered Records (CSV)",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        # Export charts as PNG bundle
        chart_bytes_trend = _fig_to_bytes(_chart_daily_trend(df))
        st.download_button(
            "📥 Download Daily Trend Chart (PNG)",
            data=chart_bytes_trend,
            file_name=f"trend_{start_str}_to_{end_str}.png",
            mime="image/png",
            use_container_width=True,
        )

    # ── Absentee Report ────────────────────────────────────────────────────────
    st.markdown("---")
    _absentee_section(today_str)


def _absentee_section(today_str: str) -> None:
    """Show today's absentee list."""
    st.subheader("🔴 Absentees Today")

    try:
        absentees = get_absentees(today_str)
    except Exception as e:
        st.warning(f"Could not load absentee list: {e}")
        return

    if not absentees:
        st.success("All registered students are present today! 🎉")
        return

    rows = [
        {
            "Student ID": a["student_id"],
            "Name": a["name"],
            "Department": a["department"],
        }
        for a in absentees
    ]
    absent_df = pd.DataFrame(rows).sort_values("Department").reset_index(drop=True)
    st.metric("Absent Today", len(absent_df))
    st.dataframe(absent_df, use_container_width=True, hide_index=True, height=280)

    csv_bytes = absent_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Absentee List (CSV)",
        data=csv_bytes,
        file_name=f"absentees_{today_str}.csv",
        mime="text/csv",
    )
