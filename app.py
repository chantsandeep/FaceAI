import streamlit as st
import os

# Load .env file if present (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Mode CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stSidebar { background-color: #161b22; }

    .stButton > button {
        background-color: #1f6feb; color: white;
        border: none; border-radius: 6px;
        padding: 0.4rem 1rem; font-weight: 600;
        transition: background-color 0.2s;
    }
    .stButton > button:hover { background-color: #388bfd; }

    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
    }

    .status-badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }
    .badge-green { background: #1a4731; color: #3fb950; }
    .badge-red   { background: #4a1a1a; color: #f85149; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 6px 6px 0 0;
        padding: 6px 16px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }

    .stDataFrame { border: 1px solid #21262d; border-radius: 6px; }

    .streamlit-expanderHeader {
        background-color: #161b22;
        border-radius: 6px;
    }

    .stAlert { border-radius: 6px; }

    .stRadio > div { gap: 4px; }
    .stRadio label {
        padding: 6px 12px;
        border-radius: 6px;
        cursor: pointer;
    }
    .stRadio label:hover { background-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Admin Login ───────────────────────────────────────────────────────────────
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")


def check_login():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        _, center, _ = st.columns([1, 1.2, 1])
        with center:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.image("https://img.icons8.com/fluency/96/face-id.png", width=72)
            st.title("🔐 Admin Login")
            st.markdown("Please log in to access the attendance system.")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="admin")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                login_btn = st.form_submit_button("Login", use_container_width=True)

            if login_btn:
                if username == ADMIN_USER and password == ADMIN_PASS:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
        st.stop()


# ── Cached data loaders (TTL = 30 s, stable across re-runs) ──────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _cached_students():
    from utils.database import get_all_students
    try:
        return get_all_students()
    except Exception:
        return []

@st.cache_data(ttl=30, show_spinner=False)
def _cached_today_records():
    from utils.attendance import get_attendance_records
    from datetime import date
    return get_attendance_records(date.today().isoformat())

@st.cache_data(ttl=30, show_spinner=False)
def _cached_stats():
    from attendance_manager import get_attendance_stats
    return get_attendance_stats()

@st.cache_data(ttl=30, show_spinner=False)
def _cached_recent_summary():
    from utils.attendance import get_attendance_dates, get_attendance_records
    dates = get_attendance_dates()[:7]
    rows = []
    for d in dates:
        recs = get_attendance_records(d)
        depts = len({r.get("department", "") for r in recs})
        rows.append({
            "Date": d,
            "Students Present": len(recs),
            "Departments": depts,
        })
    return rows

@st.cache_data(ttl=10, show_spinner=False)
def _cached_redis_status():
    from utils.database import redis_health_check
    return redis_health_check()


# ── Home Page ─────────────────────────────────────────────────────────────────
def home_page():
    import pandas as pd

    st.title("🎓 AI Face Recognition Attendance System")
    st.markdown(
        "Automated attendance tracking powered by deep learning face recognition "
        "using **InsightFace** · **OpenCV** · **Redis** · **Streamlit**"
    )

    # Manual refresh button — clears cache and reloads data
    col_title, col_refresh = st.columns([6, 1])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # ── Load cached data ──────────────────────────────────────────────────────
    students    = _cached_students()
    today_recs  = _cached_today_records()
    stats       = _cached_stats()
    redis_ok    = _cached_redis_status()

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Registered Students", len(students))
    c2.metric("✅ Present Today",        len(today_recs))
    c3.metric("📅 Total Days Logged",    stats["total_days"])
    c4.metric("📊 Avg Daily Attendance", stats["avg_daily_attendance"])
    c5.metric("🔴 Redis", "Online" if redis_ok else "Offline")

    st.markdown("---")

    # ── Today's attendance + Quick actions ────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📋 Today's Attendance")
        if today_recs:
            df = pd.DataFrame(today_recs)
            display_cols = [c for c in ["student_id", "name", "department", "timestamp"] if c in df.columns]
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(str).str[:19]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True, height=320)
        else:
            st.info("No attendance marked yet today. Start the **Live Recognition** session.")

    with col_right:
        st.subheader("🗂️ Quick Actions")
        st.markdown("""
**📋 Register Student**
Capture face samples and register a new student.

---

**📷 Live Recognition**
Start real-time face recognition to mark attendance.

---

**📊 Reports**
View analytics, charts, and export attendance data.

---

**⚙️ Management**
Manage students, view unknown faces, download CSVs.
        """)

    # ── Recent summary table ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 Recent Attendance Summary")

    summary_rows = _cached_recent_summary()
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True, height=280)
    else:
        st.info("No historical attendance data yet.")


# ── Run login check ───────────────────────────────────────────────────────────
check_login()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/face-id.png", width=72)
    st.title("AI Attendance")
    st.caption("Face Recognition System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📋 Register Student",
            "📷 Live Recognition",
            "📊 Reports",
            "⚙️ Management",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    redis_ok = _cached_redis_status()
    badge_class = "badge-green" if redis_ok else "badge-red"
    badge_text  = "Redis Connected" if redis_ok else "Redis Offline"
    st.markdown(
        f'<span class="status-badge {badge_class}">⬤ {badge_text}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.cache_data.clear()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("v1.0.0 · InsightFace buffalo_l")

# ── Page Routing ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    home_page()
elif page == "📋 Register Student":
    from registration import registration_page
    registration_page()
elif page == "📷 Live Recognition":
    from realtime_recognition import recognition_page
    recognition_page()
elif page == "📊 Reports":
    from reports import reports_page
    reports_page()
elif page == "⚙️ Management":
    from management import management_page
    management_page()
