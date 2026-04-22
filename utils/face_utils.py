"""
Face detection, embedding, and anti-proxy liveness utilities.

Anti-proxy checks:
  1. Head movement challenge  — random left/right turn instruction, photo can't comply
  2. Screen brightness contrast — phone screen much brighter than surroundings
  3. Embedding variance         — static photo gives identical embeddings
  4. Face size check            — phone held far = tiny face
"""
from __future__ import annotations

import random
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

_face_app = None


def get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def detect_faces(frame: np.ndarray) -> list:
    return get_face_app().get(frame)


def get_embedding(face) -> np.ndarray:
    emb = face.embedding
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6))


def find_best_match(
    query_embedding: np.ndarray,
    students: list[dict],
    threshold: float = 0.45,
) -> tuple[dict | None, float]:
    best_score, best_student = -1.0, None
    for student in students:
        score = cosine_similarity(query_embedding, student["embedding"])
        if score > best_score:
            best_score = score
            best_student = student
    return (best_student, best_score) if best_score >= threshold else (None, best_score)


def draw_face_box(
    frame: np.ndarray, face, label: str, color: tuple,
    confidence: float = None, status: str = "",
) -> np.ndarray:
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    display = f"{label} ({confidence:.2f})" if confidence is not None else label
    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, display, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if status:
        cv2.putText(frame, status, (x1, y2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
    return frame


def save_snapshot(frame: np.ndarray, folder: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame)
    return path


# ─── Head Pose ────────────────────────────────────────────────────────────────

def get_yaw(face) -> float | None:
    """
    Extract yaw (left-right head rotation) from InsightFace pose attribute.
    Positive yaw  = face turned RIGHT (from person's perspective).
    Negative yaw  = face turned LEFT.
    Returns None if pose not available.
    """
    try:
        pose = face.pose  # [pitch, yaw, roll]
        if pose is None:
            return None
        return float(pose[1])
    except Exception:
        return None


# ─── Screen Brightness Contrast ───────────────────────────────────────────────

def check_screen_brightness_contrast(face, frame: np.ndarray) -> tuple[bool, float, float, float]:
    """
    Phone screen is much brighter than its surroundings.
    Returns (is_real, screen_mean, surround_mean, diff).
    """
    try:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        face_w, face_h = x2 - x1, y2 - y1

        ix1, iy1 = max(0, x1), max(0, y1)
        ix2, iy2 = min(fw, x2), min(fh, y2)

        pad_x, pad_y = int(face_w * 0.9), int(face_h * 0.9)
        ox1 = max(0, x1 - pad_x); oy1 = max(0, y1 - pad_y)
        ox2 = min(fw, x2 + pad_x); oy2 = min(fh, y2 + pad_y)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_region = gray[iy1:iy2, ix1:ix2]
        screen_mean = float(np.mean(face_region)) if face_region.size > 0 else 128.0

        outer = gray[oy1:oy2, ox1:ox2].copy().astype(np.float32)
        rx1, ry1 = ix1 - ox1, iy1 - oy1
        rx2, ry2 = rx1 + (ix2 - ix1), ry1 + (iy2 - iy1)
        outer[ry1:ry2, rx1:rx2] = np.nan
        surr = outer[~np.isnan(outer)]
        surround_mean = float(np.nanmean(surr)) if surr.size > 0 else 128.0

        diff = screen_mean - surround_mean
        is_phone = (screen_mean > 150) and (diff > 35)
        return not is_phone, round(screen_mean, 1), round(surround_mean, 1), round(diff, 1)
    except Exception:
        return True, 128.0, 128.0, 0.0


def check_embedding_variance(embeddings: list[np.ndarray], min_var: float = 8e-6) -> bool:
    if len(embeddings) < 5:
        return True
    return float(np.mean(np.var(np.stack(embeddings, axis=0), axis=0))) > min_var


def check_face_size(face, frame: np.ndarray, min_ratio: float = 0.05) -> bool:
    try:
        b = face.bbox.astype(int)
        return ((b[2]-b[0]) * (b[3]-b[1])) / (frame.shape[0] * frame.shape[1]) >= min_ratio
    except Exception:
        return True


# ─── Liveness State Machine ───────────────────────────────────────────────────

# Head movement challenge config
_CHALLENGES      = ["left", "right"]   # directions to pick from
_YAW_THRESHOLD   = 15.0               # degrees of yaw needed to pass
_CENTER_RANGE    = 10.0               # yaw within ±10° = looking straight


class LivenessState:
    """
    Per-student liveness state.

    Stages:
      0 — Warming up (10 frames)
      1 — Head movement challenge (turn left / turn right)
      2 — All checks passed → live
    """

    def __init__(self):
        # Brightness / size / variance tracking
        self.diffs:         list[float]      = []
        self.screen_means:  list[float]      = []
        self.embeddings:    list[np.ndarray] = []
        self.size_ok_count: int              = 0
        self.frame_count:   int              = 0

        # Head movement challenge
        self.challenge:         str   = random.choice(_CHALLENGES)  # "left" or "right"
        self.challenge_passed:  bool  = False
        self.baseline_yaw:      float | None = None   # yaw when looking straight
        self.yaw_history:       list[float]  = []

    def update(self, face, frame: np.ndarray, embedding: np.ndarray) -> dict:
        self.frame_count += 1

        # ── Basic checks ──────────────────────────────────────────────────────
        is_real, scr, sur, diff = check_screen_brightness_contrast(face, frame)
        size_ok = check_face_size(face, frame)

        self.diffs.append(diff)
        self.screen_means.append(scr)
        if size_ok:
            self.size_ok_count += 1

        self.embeddings.append(embedding)
        if len(self.embeddings) > 12:
            self.embeddings.pop(0)

        var_ok   = check_embedding_variance(self.embeddings)
        avg_diff = float(np.mean(self.diffs[-10:]))
        avg_scr  = float(np.mean(self.screen_means[-10:]))

        # ── Yaw tracking ──────────────────────────────────────────────────────
        yaw = get_yaw(face)
        if yaw is not None:
            self.yaw_history.append(yaw)
            if len(self.yaw_history) > 30:
                self.yaw_history.pop(0)

        # ── Stage 0: warm-up ──────────────────────────────────────────────────
        if self.frame_count < 10:
            return self._r(False, f"Verifying... ({self.frame_count}/10)",
                           avg_diff, avg_scr, var_ok, size_ok)

        # ── Hard blocks (phone / too small / no movement) ─────────────────────
        if self.size_ok_count < self.frame_count * 0.5:
            return self._r(False, "⚠ Move closer to camera",
                           avg_diff, avg_scr, var_ok, size_ok)

        if avg_scr > 150 and avg_diff > 35:
            return self._r(False, "⚠ Phone screen detected — proxy blocked",
                           avg_diff, avg_scr, var_ok, size_ok)

        if not var_ok:
            return self._r(False, "⚠ No movement — proxy blocked",
                           avg_diff, avg_scr, var_ok, size_ok)

        # ── Stage 1: head movement challenge ──────────────────────────────────
        if not self.challenge_passed:
            return self._head_challenge(yaw, avg_diff, avg_scr, var_ok, size_ok)

        # ── Stage 2: all passed ───────────────────────────────────────────────
        return self._r(True, "✓ Liveness verified",
                       avg_diff, avg_scr, var_ok, size_ok)

    def _head_challenge(self, yaw, avg_diff, avg_scr, var_ok, size_ok) -> dict:
        """Check if the user has turned their head in the required direction."""

        if yaw is None:
            # Pose not available — skip challenge, pass anyway
            self.challenge_passed = True
            return self._r(True, "✓ Liveness verified",
                           avg_diff, avg_scr, var_ok, size_ok)

        # Establish baseline yaw from first few straight-ahead frames
        if self.baseline_yaw is None:
            recent = self.yaw_history[-5:] if len(self.yaw_history) >= 5 else self.yaw_history
            if recent:
                self.baseline_yaw = float(np.mean(recent))
            return self._r(False,
                f"{'⬅ Turn LEFT' if self.challenge == 'left' else '➡ Turn RIGHT'}",
                avg_diff, avg_scr, var_ok, size_ok)

        # Relative yaw from baseline
        rel_yaw = yaw - self.baseline_yaw

        # InsightFace yaw convention:
        #   positive yaw = face turned to person's LEFT  (camera sees right side)
        #   negative yaw = face turned to person's RIGHT (camera sees left side)
        if self.challenge == "left":
            turned = rel_yaw > _YAW_THRESHOLD
            instruction = f"⬅ Turn your head LEFT  (yaw:{rel_yaw:+.0f}°)"
        else:
            turned = rel_yaw < -_YAW_THRESHOLD
            instruction = f"➡ Turn your head RIGHT  (yaw:{rel_yaw:+.0f}°)"

        if turned:
            self.challenge_passed = True
            return self._r(True, "✓ Liveness verified",
                           avg_diff, avg_scr, var_ok, size_ok)

        return self._r(False, instruction, avg_diff, avg_scr, var_ok, size_ok)

    def _r(self, live, reason, diff, scr, var_ok, size_ok):
        return {
            "live": live, "reason": reason,
            "diff": diff, "screen_mean": scr,
            "variance_ok": var_ok, "size_ok": size_ok,
            "challenge": self.challenge,
            "challenge_passed": self.challenge_passed,
        }

    def reset(self):
        self.__init__()


# ─── Registration helper ──────────────────────────────────────────────────────

def frames_to_embedding(frames: list[np.ndarray]) -> np.ndarray | None:
    embeddings = []
    for frame in frames:
        faces = detect_faces(frame)
        if faces:
            embeddings.append(get_embedding(faces[0]))
    if not embeddings:
        return None
    avg  = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


# backward compat stub
def detect_blink(face, prev_ear=None, threshold=0.28):
    return False, 1.0
