# 🎓 AI Face Recognition Attendance System

> A real-time, AI-powered attendance management system that automatically detects faces, verifies liveness, and marks attendance — all through a web browser. No ID cards. No manual entry. Just look at the camera.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [AI Components](#-ai-components)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [System Architecture](#-system-architecture)
7. [Workflow Diagrams](#-workflow-diagrams)
8. [Anti-Proxy System](#-anti-proxy--liveness-detection)
9. [Database Design](#-database-design)
10. [Pages & UI](#-pages--ui)
11. [How to Run](#-how-to-run)
12. [Configuration](#-configuration)
13. [How Data Flows](#-how-data-flows)
14. [File-by-File Explanation](#-file-by-file-explanation)

---

## 🌟 Project Overview

This system replaces traditional paper-based or RFID attendance with **AI face recognition**. A webcam captures live video, neural networks detect and identify faces, and attendance is automatically logged to a CSV file and Redis database.

The system includes:
- A **Streamlit web dashboard** accessible from any browser
- **Student registration** via webcam
- **Real-time face recognition** with anti-proxy protection
- **Attendance reports** with charts and CSV export
- **Admin management** panel

---

## ✅ Key Features

| Feature | Description |
|---|---|
| 🔐 Admin Login | Session-based authentication with `.env` credentials |
| 📋 Student Registration | Webcam capture → face embedding → Redis storage |
| 📷 Real-Time Recognition | Live webcam feed with face detection and matching |
| 🛡️ Anti-Proxy Protection | Head movement challenge + screen brightness detection |
| 📅 Daily CSV Logs | Auto-generated per-day attendance files |
| 🔁 Deduplication | Each student marked only once per day |
| 📸 Snapshot Capture | Photo saved when attendance is marked |
| 📊 Reports & Analytics | Charts, filters, date range, department breakdown |
| 🚨 Unknown Face Logging | Unrecognized faces saved and logged |
| ⚙️ Management Dashboard | View/delete students, browse records, export data |
| 🌙 Dark Mode UI | Full dark theme via custom Streamlit CSS |
| 🐳 Docker Support | One-command deployment with docker-compose |

---

## 🧠 AI Components

This project uses AI in **6 places**:

### 1. Face Detection — RetinaFace (Deep Learning CNN)
- **Model:** `det_10g.onnx` inside InsightFace `buffalo_l`
- **What it does:** Scans every webcam frame and finds all faces (bounding boxes)
- **Type:** Convolutional Neural Network trained on millions of face images
- **Output:** Bounding box coordinates `[x1, y1, x2, y2]` for each face

### 2. Face Recognition — ArcFace ResNet-50
- **Model:** `w600k_r50.onnx` inside InsightFace `buffalo_l`
- **What it does:** Converts any face into a **512-dimensional embedding vector**
- **Type:** ResNet-50 deep neural network trained with ArcFace loss
- **Output:** A list of 512 numbers that uniquely describes a face
- **Key property:** Same person → similar vectors. Different person → very different vectors.

### 3. Cosine Similarity Matching
- **What it does:** Compares two 512-d vectors to measure how similar two faces are
- **Formula:** `similarity = dot(A,B) / (|A| × |B|)`
- **Range:** 0.0 (no match) → 1.0 (perfect match)
- **Threshold:** 0.45 (configurable) — above this = same person

### 4. Head Pose Estimation
- **Model:** `genderage.onnx` inside InsightFace `buffalo_l`
- **What it does:** Estimates 3D head rotation — pitch (up/down), yaw (left/right), roll (tilt)
- **Used for:** Head movement challenge in anti-proxy system
- **Output:** `[pitch, yaw, roll]` in degrees

### 5. Multi-Frame Embedding Averaging
- **What it does:** During registration, runs the AI on 10+ frames and averages all embeddings
- **Why:** Reduces noise, creates a more robust face representation
- **Result:** One stable 512-d vector stored per student

### 6. Embedding Variance Analysis
- **What it does:** Tracks how much the face embedding changes across frames
- **Logic:** Real face moves slightly → embeddings vary. Static photo → identical embeddings every frame
- **Used for:** Detecting printed photos or static video playback

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core programming language |
| **Web UI** | Streamlit 1.28 | Browser-based dashboard |
| **Face AI** | InsightFace 0.7.3 (buffalo_l) | Face detection + recognition |
| **Model Runtime** | ONNX Runtime 1.16.1 | Runs neural network models on CPU |
| **Computer Vision** | OpenCV 4.8 | Webcam access, image processing |
| **Database** | Redis 3.x / 7.x | Stores embeddings and attendance |
| **Data Processing** | NumPy, Pandas | Embeddings math, CSV handling |
| **Visualization** | Matplotlib 3.7 | Charts in reports page |
| **Image Handling** | Pillow 10.0 | Image utilities |
| **Config** | python-dotenv | Load `.env` credentials |
| **Deployment** | Docker + docker-compose | Containerized deployment |

---

## 📁 Project Structure

```
AI_Attendance_System/
│
├── app.py                    ← Entry point: login, navigation, page routing
├── registration.py           ← Student registration (webcam + embedding)
├── realtime_recognition.py   ← Live recognition + attendance marking
├── attendance_manager.py     ← Business logic: mark, query, summarize
├── reports.py                ← Reports page with Matplotlib charts
├── management.py             ← Admin: manage students, records, unknowns
│
├── utils/
│   ├── __init__.py
│   ├── face_utils.py         ← ALL AI logic: detect, embed, match, liveness
│   ├── database.py           ← All Redis read/write operations
│   └── attendance.py         ← CSV read/write with deduplication
│
├── dataset/                  ← Face sample images from registration
│   └── {student_id}/
│       ├── sample_1.jpg
│       └── sample_2.jpg ...
│
├── snapshots/                ← Photos taken when attendance is marked
│   ├── {student_id}/
│   │   └── {id}_20260420_091532.jpg
│   └── unknown/              ← Unrecognized face snapshots
│
├── attendance_logs/          ← Daily attendance CSV files
│   └── attendance_2026-04-21.csv
│
├── embeddings/               ← Reserved for future disk cache
│
├── .env                      ← Your credentials (not in git)
├── .env.example              ← Template for credentials
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← Docker build instructions
├── docker-compose.yml        ← Redis + App together
└── README.md                 ← This file
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BROWSER  (http://localhost:8501)                 │
│                                                                         │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │   Home   │  │  Register    │  │    Live      │  │   Reports /  │  │
│   │Dashboard │  │  Student     │  │ Recognition  │  │  Management  │  │
│   └──────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              │  Streamlit (app.py)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      registration.py  realtime_recognition  reports.py / management.py
              │               │
              └───────┬───────┘
                      ▼
            utils/face_utils.py
            ┌─────────────────────────────────┐
            │  InsightFace buffalo_l           │
            │  ┌─────────────┐ ┌───────────┐  │
            │  │ RetinaFace  │ │  ArcFace  │  │
            │  │  Detector   │ │ ResNet-50 │  │
            │  │ (det_10g)   │ │(w600k_r50)│  │
            │  └─────────────┘ └───────────┘  │
            │  ┌─────────────────────────────┐ │
            │  │   Head Pose Estimator       │ │
            │  │      (genderage)            │ │
            │  └─────────────────────────────┘ │
            └─────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
  utils/database.py        utils/attendance.py
          │                       │
       Redis DB             CSV Files
  ┌─────────────────┐    ┌──────────────────────┐
  │ student:{id}    │    │ attendance_logs/      │
  │ students:all    │    │ attendance_YYYY-MM-DD │
  │ attendance:...  │    │ .csv                 │
  │ unknown_faces   │    └──────────────────────┘
  └─────────────────┘
```

---

## 🔄 Workflow Diagrams

### Registration Workflow

```
Admin opens Register Student page
          │
          ▼
Fill form: Student ID, Name, Department, Sample Count
          │
          ▼
Click "Start Registration"
          │
          ▼
Webcam opens
          │
          ▼
┌─────────────────────────────────┐
│  LOOP until N samples captured  │
│                                 │
│  Capture frame from webcam      │
│         │                       │
│         ▼                       │
│  RetinaFace detects face?       │
│    YES → save frame + image     │
│    NO  → show "No face" warning │
└─────────────────────────────────┘
          │
          ▼
ArcFace extracts 512-d embedding from each frame
          │
          ▼
Average all embeddings → 1 robust embedding
          │
          ▼
Save to Redis:
  student:{id} → { name, dept, embedding }
  students:all → add id to set
          │
          ▼
Save sample images to dataset/{student_id}/
          │
          ▼
✅ "Student registered successfully!"
```

---

### Recognition & Attendance Workflow

```
Admin opens Live Recognition page
          │
          ▼
Load all students from Redis
          │
          ▼
Open webcam (cv2.VideoCapture(0))
          │
          ▼
┌──────────────────────────────────────────────────────┐
│  MAIN LOOP (runs every frame ~30fps)                 │
│                                                      │
│  Read frame from webcam                              │
│         │                                            │
│         ▼                                            │
│  RetinaFace → detect all faces in frame              │
│         │                                            │
│         ▼  (for each face)                           │
│  ArcFace → extract 512-d embedding                   │
│         │                                            │
│         ▼                                            │
│  Cosine similarity vs ALL stored embeddings          │
│         │                                            │
│    ┌────┴────┐                                       │
│    │         │                                       │
│  MATCH    NO MATCH                                   │
│    │         │                                       │
│    ▼         ▼                                       │
│  Run      Label "Unknown"                            │
│  Liveness  Save snapshot                             │
│  Checks    Log to Redis                              │
│    │                                                 │
│    ▼                                                 │
│  ┌─────────────────────────────────────────┐         │
│  │  LIVENESS CHECKS (anti-proxy)           │         │
│  │                                         │         │
│  │  1. Warm up 10 frames                   │         │
│  │  2. Screen brightness check             │         │
│  │     (phone screen >> surroundings?)     │         │
│  │  3. Embedding variance check            │         │
│  │     (static photo = no variance)        │         │
│  │  4. Head movement challenge             │         │
│  │     "Turn LEFT" or "Turn RIGHT"         │         │
│  │     (measure yaw angle change)          │         │
│  │  5. Face size check                     │         │
│  │     (must be ≥5% of frame)              │         │
│  └─────────────────────────────────────────┘         │
│         │                                            │
│    ┌────┴────┐                                       │
│    │         │                                       │
│  LIVE     NOT LIVE                                   │
│    │         │                                       │
│    ▼         ▼                                       │
│  N consecutive  Show reason on screen                │
│  frames passed? "⚠ Phone screen detected"            │
│    │            "➡ Turn your head RIGHT"             │
│    ▼                                                 │
│  Mark Attendance                                     │
│    │                                                 │
│    ├── Write to CSV (attendance_logs/)               │
│    ├── Write to Redis (attendance:{date}:{id})       │
│    └── Save snapshot (snapshots/{id}/)               │
│                                                      │
│  Draw bounding box + name on frame                   │
│  Display frame in browser                            │
└──────────────────────────────────────────────────────┘
          │
          ▼
Admin clicks "Stop Camera"
          │
          ▼
Show summary: "X students marked present"
```

---

### Anti-Proxy Decision Tree

```
Face detected in frame
          │
          ▼
Frame count < 10?
  YES → "Verifying... (X/10)"  ──────────────────────────► WAIT
          │
          NO
          ▼
Face size < 5% of frame?
  YES → "⚠ Move closer to camera"  ──────────────────────► BLOCK
          │
          NO
          ▼
Screen brightness > 150 AND diff vs surroundings > 35?
  YES → "⚠ Phone screen detected"  ──────────────────────► BLOCK
          │
          NO
          ▼
Embedding variance too low (static image)?
  YES → "⚠ No movement — proxy blocked"  ────────────────► BLOCK
          │
          NO
          ▼
Head movement challenge passed?
  NO  → "⬅ Turn your head LEFT" / "➡ Turn RIGHT"  ──────► WAIT
          │
          YES
          ▼
All checks passed → LIVENESS VERIFIED ──────────────────► MARK ATTENDANCE
```

---

## 🛡️ Anti-Proxy / Liveness Detection

The system uses **4 layers** to prevent proxy attendance:

### Layer 1 — Screen Brightness Contrast
```
Phone screen scenario:
  Face region (on screen) = very bright (mean > 150)
  Surrounding area (hand, wall, room) = dark
  Difference > 35 brightness units → PHONE DETECTED

Real face scenario:
  Face and background lit by same ambient light
  Difference is small (< 25) → REAL FACE
```

### Layer 2 — Embedding Variance
```
Static photo/video:
  Same face every frame → embeddings nearly identical
  Variance across 12 frames ≈ 0 → BLOCKED

Real face:
  Natural micro-movements → embeddings vary slightly
  Variance > threshold → PASS
```

### Layer 3 — Head Movement Challenge
```
Random challenge assigned: "Turn LEFT" or "Turn RIGHT"

InsightFace measures yaw angle (left-right rotation):
  Baseline yaw recorded when face first appears
  User must rotate head ≥ 15° from baseline
  
  Phone screen: yaw stays constant no matter how you tilt it
  Real person: yaw changes when they turn their head
```

### Layer 4 — Face Size Check
```
Phone held at arm's length:
  Face-within-phone occupies tiny fraction of frame
  Face area < 5% of total frame → BLOCKED

Real person standing in front of camera:
  Face occupies reasonable portion of frame → PASS
```

---

## 🗄️ Database Design

### Redis Key Structure

| Key | Type | Contents | TTL |
|---|---|---|---|
| `student:{id}` | Hash | student_id, name, department, registered_at, embedding (JSON) | None |
| `students:all` | Set | All registered student IDs | None |
| `attendance:{date}:{id}` | String | JSON attendance record | 90 days |
| `attendance:date:{date}` | Set | Student IDs present on date | 90 days |
| `attendance:student:{id}` | Set | Dates student was present | 90 days |
| `unknown_faces` | List | Last 500 unknown face events | None |

### CSV Format

**File:** `attendance_logs/attendance_2026-04-21.csv`

```
student_id,name,department,date,timestamp,snapshot
20857,Sahil Singh,BCA,2026-04-21,2026-04-21T09:15:32,snapshots/20857/20857_20260421_091532.jpg
20858,Aditya,BCA,2026-04-21,2026-04-21T09:17:44,snapshots/20858/20858_20260421_091744.jpg
```

**Rules:**
- One file per day, auto-created
- One row per student per day (duplicates blocked)
- CSV is the permanent record — Redis expires after 90 days

---

## 🖥️ Pages & UI

### 🏠 Home Page
- 5 metric cards: Registered Students, Present Today, Total Days Logged, Avg Daily Attendance, Redis Status
- Today's attendance table
- 7-day recent summary table
- 🔄 Refresh button

### 📋 Register Student
- **New Student tab** — form + webcam capture
- **Update/Re-register tab** — replace existing embedding
- **Registered Students tab** — view all, export CSV

### 📷 Live Recognition
- Live webcam feed with face bounding boxes
- Match threshold slider
- Anti-proxy checks panel
- Head movement challenge banner on video
- Live attendance log table

### 📊 Reports
- Date range filter (Today / Last 7 / Last 30 / All Time / Custom)
- Department and name/ID search filters
- 4 Matplotlib charts:
  - Daily attendance trend (line chart)
  - Attendance by department (horizontal bar)
  - Attendance by hour of day (bar chart)
  - Top students by attendance days (bar chart)
- Full data table with export
- Absentee report for today

### ⚙️ Management
- **Students tab** — view all, export, delete with confirmation
- **Attendance Records tab** — browse by date, filter, download CSV
- **Unknown Faces tab** — event log + snapshot thumbnails

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- Redis server
- Webcam

### Option A — Local

```bash
# 1. Start Redis
docker run -d -p 6379:6379 --name attendance_redis redis:7-alpine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env → set ADMIN_USER and ADMIN_PASS

# 4. Run
streamlit run app.py
```

Open **http://localhost:8501**

### Option B — Docker Compose

```bash
docker compose up --build
```

Open **http://localhost:8501**

### Default Login
- **Username:** `admin`
- **Password:** `admin123`

---

## ⚙️ Configuration

All settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis server address |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | *(empty)* | Redis password |
| `REDIS_DB` | `0` | Redis database index |
| `ADMIN_USER` | `admin` | Login username |
| `ADMIN_PASS` | `admin123` | Login password |

Recognition settings (in `realtime_recognition.py`):

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_THRESHOLD` | `0.45` | Cosine similarity cutoff |
| `CONFIRM_FRAMES` | `5` | Consecutive frames before marking |
| `_YAW_THRESHOLD` | `15.0°` | Head rotation needed for challenge |

---

## 🔁 How Data Flows

```
REGISTRATION:
  Webcam → OpenCV → InsightFace (detect) → InsightFace (embed)
  → Average 10 embeddings → Redis (student:{id})
  → Save images to dataset/{id}/

RECOGNITION:
  Webcam → OpenCV → InsightFace (detect) → InsightFace (embed)
  → Cosine similarity vs Redis embeddings
  → Liveness checks (brightness + variance + head pose)
  → attendance_logs/attendance_YYYY-MM-DD.csv
  → Redis (attendance:{date}:{id})
  → snapshots/{id}/photo.jpg

REPORTING:
  attendance_logs/*.csv → Pandas DataFrame
  → Matplotlib charts → Streamlit display
  → CSV download
```

---

## 📄 File-by-File Explanation

| File | What it does |
|---|---|
| `app.py` | Main entry point. Handles admin login, dark mode CSS, sidebar navigation, page routing. Caches data with `@st.cache_data` to prevent UI flickering. |
| `registration.py` | Student registration UI. Opens webcam, captures N frames, calls InsightFace, averages embeddings, saves to Redis. Also handles update/re-register. |
| `realtime_recognition.py` | Live recognition loop. Reads webcam frames, detects faces, matches embeddings, runs liveness checks, marks attendance, draws bounding boxes. |
| `attendance_manager.py` | Business logic layer. `mark_attendance()` writes to both CSV and Redis. Query functions for daily summary, date range, absentees, stats. |
| `reports.py` | Reports page. Builds filtered DataFrames, renders 4 Matplotlib charts, shows absentee list, provides CSV downloads. |
| `management.py` | Admin dashboard. Student list with delete, attendance records browser, unknown faces viewer with thumbnails. |
| `utils/face_utils.py` | All AI logic. InsightFace singleton, face detection, embedding extraction, cosine similarity, liveness state machine (screen brightness + embedding variance + head pose challenge). |
| `utils/database.py` | All Redis operations. Student CRUD, attendance read/write, unknown face log, health check. Uses `hmset` for Redis 3.x compatibility. |
| `utils/attendance.py` | CSV operations. Write attendance row, check duplicates, read records by date, list available dates. |

---

## 📦 Requirements

```
opencv-python==4.8.1.78     # Webcam + image processing
numpy==1.24.3               # Embedding math
pandas==2.0.3               # CSV + DataFrames
streamlit==1.28.0           # Web UI
insightface==0.7.3          # Face detection + recognition AI
onnxruntime==1.16.1         # Neural network inference (CPU)
redis==5.0.1                # Database client
matplotlib==3.7.2           # Charts
Pillow==10.0.1              # Image utilities
scikit-learn==1.3.0         # ML utilities
scipy==1.11.3               # Scientific computing
dlib==19.24.2               # Face landmark utilities
python-dotenv==1.0.0        # Load .env file
```

---

## 🔒 Security Notes

- Admin credentials stored in `.env` — never commit this file to git
- `.env.example` is the safe template to share
- Redis attendance records expire after 90 days — CSV files are permanent
- Snapshots stored locally — back up `snapshots/` and `attendance_logs/` regularly
- Anti-proxy system prevents photo/phone/video spoofing

---

## 👨‍💻 Author

Built as a Capstone Project — AI-Based Face Recognition Attendance System  
Stack: Python · InsightFace · OpenCV · Redis · Streamlit · Matplotlib · Docker
