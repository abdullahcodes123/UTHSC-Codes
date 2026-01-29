"""
================================================================================
MASTER-CLOCK 3-DOMAIN ALIGNMENT + FIBER PHOTOMETRY PERI-EVENT ANALYSIS (INCREASE-only)
SELF_ADMIN-only — VIDEO+FIBER => MASTER (trim+rezero) — MedPC => MASTER via MONOTONIC DP (NO PENALTIES)
NO MedPC→Fiber warping (MF alignment removed)
================================================================================

HIGH-LEVEL MAP (who maps to what?)
----------------------------------
(1) Define MASTER time by trimming the same amount from VIDEO and FIBER, then re-zero both to t=0.

    VIDEO (original time)  --trim TRIM_START_SEC-->  VIDEO_MASTER time (t=0 at trim point)
    FIBER (original time)  --trim TRIM_START_SEC-->  FIBER_MASTER time (t=0 at trim point)

(2) Align MedPC event time into MASTER with a single affine transform:
        t_master_event = a_mv * t_medpc + b_mv

(3) Do peri-event analysis on FIBER_MASTER (no warping), and compute:
        peak_offset (relative to event, in seconds on MASTER)
        t_master_peak = t_master_event + peak_offset

(4) Extract video clips using MASTER peak time:
        MASTER time -> MASTER frame index -> ORIGINAL video frame index (via start_frame_offset)

ARCHITECTURE (TARGET)
---------------------
VIDEO  --- trim+rezero ---┐
                          ├── MASTER CLOCK (t_master)
FIBER  --- trim+rezero ---┘

MedPC  ---[monotonic DP matching, penalty-free objective]---> MASTER

MASTER MedPC events
        ↓
Fiber peri-event (on MASTER fiber time)
        ↓
Peak (offset in MASTER seconds)
        ↓
Video clips (MASTER→frame index + original frame offset)
================================================================================

NOTES
-----
- MASTER clock is *defined* by trimming both VIDEO and FIBER by the same TRIM_START_SEC and re-zeroing.
- MedPC is aligned *only* to MASTER. There is no MedPC→Fiber non-linear warping.
- Alignment uses a monotonic DP matcher with a penalty-free objective:
    - Primary objective: maximize number of matched anchors within tolerance
    - Secondary objective: minimize total absolute timing error across matched anchors
  (No miss penalties, no gap penalties, no parameter regularization.)
- Peri-event peak detection is INCREASE-only (max robust-Z within [PEAK_SEARCH_START, PEAK_SEARCH_END]).
"""

from __future__ import annotations

# =============================================================================
# EDITABLE VARIABLES (ALL TUNABLES LIVE HERE)
# =============================================================================

from pathlib import Path
from typing import Tuple, List, Optional, Dict
import math
import json
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# -------------------------
# Paths
# -------------------------
DESKTOP = Path("/content/drive/MyDrive/deneme")
BEH_FILE = DESKTOP / "deneme.csv"
FIBER_FILE = DESKTOP / "fiber.csv"
VIDEO_FILE = DESKTOP / "Abdullah.avi"

# Output root directory for all generated files (figures, CSVs, reports, clips)
ROOT = DESKTOP / "abdullah_masterclock_penaltyfree_v20"

# -------------------------
# Debug / runtime
# -------------------------
USE_FAKE_VIDEO = False
PREFER_GPU = True

DEBUG_MATCH_PRINT = True
DEBUG_MATCH_MAX = 10

# -------------------------
# MASTER clock trimming
# -------------------------
TRIM_START_SEC = 0.0

# -------------------------
# MedPC settings
# -------------------------
MEDPC_COL = "self_administration"
MEDPC_DT_SEC = 1.0
CLUSTER_WIN = 0.0

# -------------------------
# Fiber preprocessing
# -------------------------
PB_TAU_SEC = 60.0

# -------------------------
# Peri-event analysis (fiber)
# -------------------------
PRE = 5.0
POST = 5.0

BASE_START = -5.0
BASE_END = 0.0

PEAK_SEARCH_START = 0.0
PEAK_SEARCH_END = 5.0

# -------------------------
# Clip extraction (MASTER timeline)
# -------------------------
CLIP_PRE = 5.0
CLIP_POST = 5.0
TARGET_CLIPS = 10

# -------------------------
# Event cleanup / selection
# -------------------------
EVENT_REFRACTORY = 5.0

# -------------------------
# MV (MedPC -> MASTER) grid search controls
# -------------------------
TOL_MANUAL = 15.0

# Search grid for slope a around A_PRIOR
A_PRIOR = 1.0
A_MV_PAD = 0.015
A_MV_STEP = 0.005

# Search grid for intercept b around b_prior (only used to define the search window)
B_PAD_AROUND_PRIOR = 120.0
B_MV_STEP = 1.0

# -------------------------
# Clock prior (only for centering b search window)
# -------------------------
MEDPC_START_CLOCK = "15:43:47"
VIDEO_START_CLOCK = "15:44:43"
USE_CLOCK_PRIOR = True
B_PRIOR_OVERRIDE: Optional[float] = None

# -------------------------
# Bootstrap settings (MV only; MF removed)
# -------------------------
BOOTSTRAP_SEED = 123
BOOTSTRAP_MV_N = 200

MV_BOOT_A_PAD = 0.005
MV_BOOT_A_STEP = 0.0005
MV_BOOT_B_PAD = 6.0
MV_BOOT_B_STEP = 0.1

# -------------------------
# Manual anchors (VIDEO timeline, "mm.ss")
# -------------------------
manual_0_10_txt = ["1.36", "2.52", "3.34", "4.23", "5.19", "6.49", "7.00", "7.52", "8.25", "9.50"]
manual_mid_txt = ["12.27", "13.19", "21.13", "27.23", "31.56"]
manual_35_45_txt = ["36.10", "38.15", "43.01", "43.11"]

# =============================================================================
# LIGHTWEIGHT PROGRESS BARS (Colab-friendly)
# =============================================================================
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _pbar(iterable=None, total=None, desc="", unit="", leave=True):
    if tqdm is None:
        return iterable if iterable is not None else range(int(total or 0))
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, mininterval=0.2)


# =============================================================================
# OPTIONAL GPU ACCELERATION (CuPy)
# =============================================================================
try:
    import cupy as cp
    _GPU_OK = True
except Exception:
    cp = None
    _GPU_OK = False


def xp_backend(prefer_gpu: bool = True):
    return cp if (prefer_gpu and _GPU_OK) else np


# =============================================================================
# HELPERS
# =============================================================================
def parse_mmss_to_seconds(s: str) -> float:
    """
    Convert a manual timestamp string to seconds.

    Expected formats:
    - "mm.ss" (e.g. "12.27" means 12 minutes + 27 seconds)
    - Or a plain number string that already represents seconds.
    """
    s = s.strip()
    if "." in s:
        mm, ss = s.split(".", 1)
        return int(mm) * 60 + int(ss)
    return float(s)


def cluster_events(times_s: np.ndarray, win_s: float) -> np.ndarray:
    """Cluster events closer than win_s seconds; keep the first event of each cluster."""
    x = np.sort(np.asarray(times_s, float))
    if x.size == 0:
        return x
    out = [float(x[0])]
    last = float(x[0])
    for t in x[1:]:
        t = float(t)
        if t - last > float(win_s):
            out.append(t)
            last = t
    return np.array(out, dtype=float)


def infer_fiber_cols(df: pd.DataFrame):
    """Infer time/signal/control columns across different acquisition CSV formats."""
    cols = list(df.columns)

    def pick(name_list, fallback_idx):
        for n in name_list:
            if n in cols:
                return n
        return cols[fallback_idx] if len(cols) > fallback_idx else cols[0]

    time_cands = [c for c in cols if "time" in c.lower()]
    sig_cands = [c for c in cols if ("sig" in c.lower()) or ("490" in c.lower())]
    ctl_cands = [c for c in cols if ("ctl" in c.lower()) or ("control" in c.lower()) or ("405" in c.lower()) or ("iso" in c.lower())]

    time_col = pick(["TimeStamp", "Time(s)"], 0)
    if time_col not in cols and time_cands:
        time_col = time_cands[0]

    sig_col = pick(["Signal"], 1)
    if sig_col not in cols and sig_cands:
        sig_col = sig_cands[0]

    ctl_col = pick(["Control"], 2)
    if ctl_col not in cols and ctl_cands:
        ctl_col = ctl_cands[0]

    return time_col, sig_col, ctl_col


def normalize_to_seconds_by_dt(t_raw: np.ndarray):
    """
    Normalize a raw time array to seconds (tries to infer microseconds or milliseconds).
    Returns (t_seconds, scale_factor_applied).
    """
    t = np.asarray(t_raw, float)
    t = t - np.nanmin(t)
    dt = float(np.nanmedian(np.diff(np.sort(t))))
    if not np.isfinite(dt) or dt <= 0:
        return t, 1.0

    fs = 1.0 / dt
    scale = 1.0
    if fs > 50000:
        scale = 1e-6  # microseconds
    elif fs > 5000:
        scale = 1e-3  # milliseconds
    elif fs < 50 and np.nanmax(t) > 20000:
        scale = 1e-3

    t2 = (t * scale).astype(float)
    t2 = t2 - float(np.min(t2))
    return t2, scale


def photobleach_correct(sig: np.ndarray, ctl: np.ndarray, fs: float, tau_sec: float):
    """Photobleaching/drift correction using EMA trends (normalize each channel by its trend)."""
    span = int(max(10, round(tau_sec * fs)))
    s = pd.Series(sig)
    c = pd.Series(ctl)
    trend_sig = s.ewm(span=span, adjust=False).mean().to_numpy(dtype=float)
    trend_ctl = c.ewm(span=span, adjust=False).mean().to_numpy(dtype=float)
    eps = 1e-12
    sig_n = sig / (trend_sig + eps)
    ctl_n = ctl / (trend_ctl + eps)
    return sig_n, ctl_n, trend_sig, trend_ctl


def calculate_clock_offset(medpc_time: str, video_time: str) -> float:
    """
    b_prior from wall-clock start times:
      b_prior = (MedPC start) - (Video start), in seconds.
    Negative => MedPC started earlier.
    """
    from datetime import datetime
    fmt = "%H:%M:%S"
    t_med = datetime.strptime(medpc_time, fmt)
    t_vid = datetime.strptime(video_time, fmt)
    return (t_med - t_vid).total_seconds()


# =============================================================================
# VIDEO: VFR/FPS-DROP SAFE TIMELINE (PTS) EXTRACTION
# =============================================================================
def _run_ffprobe_frame_pts(video_path: Path) -> Optional[np.ndarray]:
    """Extract per-frame timestamps (PTS) using ffprobe; best for VFR/dropped frames."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        j = json.loads(out)
        frames = j.get("frames", [])
        pts = []
        for fr in frames:
            v = fr.get("best_effort_timestamp_time", None)
            if v is None:
                continue
            try:
                pts.append(float(v))
            except Exception:
                continue
        if len(pts) < 10:
            return None
        pts = np.asarray(pts, dtype=float)
        pts = np.maximum.accumulate(pts)
        pts = pts - float(np.min(pts))
        return pts
    except Exception:
        return None


def _scan_cv2_pos_msec(video_path: Path, max_frames: Optional[int] = None) -> Tuple[np.ndarray, int, int, float]:
    """Fallback: read frames with OpenCV and record CAP_PROP_POS_MSEC."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video file (codec).")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_meta = float(cap.get(cv2.CAP_PROP_FPS)) if np.isfinite(cap.get(cv2.CAP_PROP_FPS)) else float("nan")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_est = None
    if np.isfinite(frame_count) and frame_count > 0:
        total_est = int(frame_count)
        if max_frames is not None:
            total_est = min(total_est, int(max_frames))

    t_ms = []
    n = 0
    it = _pbar(total=total_est, desc="[VIDEO] OpenCV timestamp scan", unit="fr", leave=True)

    while True:
        ret, _ = cap.read()
        if not ret:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not np.isfinite(ms):
            ms = np.nan
        t_ms.append(ms)
        n += 1
        if tqdm is not None:
            it.update(1)
        if max_frames is not None and n >= int(max_frames):
            break

    if tqdm is not None:
        it.close()
    cap.release()

    t_ms = np.asarray(t_ms, dtype=float)
    good = np.isfinite(t_ms)
    if good.sum() < 10:
        raise RuntimeError("OpenCV timestamp scan failed (too few finite CAP_PROP_POS_MSEC values).")

    idx = np.arange(len(t_ms))
    t_ms2 = t_ms.copy()
    if not np.all(good):
        t_ms2[~good] = np.interp(idx[~good], idx[good], t_ms2[good])

    t_ms2 = np.maximum.accumulate(t_ms2)
    t_sec = (t_ms2 - float(np.min(t_ms2))) / 1000.0
    return t_sec, W, H, fps_meta


def build_video_timeline(video_path: Path) -> dict:
    """Build a VFR-safe video timeline: ffprobe PTS preferred, else OpenCV scan."""
    if tqdm is not None:
        print("[VIDEO] Building VFR-safe timeline (ffprobe preferred)...")

    pts = _run_ffprobe_frame_pts(video_path)
    if pts is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video file (codec).")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_meta = float(cap.get(cv2.CAP_PROP_FPS)) if np.isfinite(cap.get(cv2.CAP_PROP_FPS)) else float("nan")
        cap.release()
        t_frame = pts
        src = "ffprobe"
    else:
        if tqdm is not None:
            print("[VIDEO] ffprobe PTS failed/insufficient -> fallback OpenCV scan (can be slow)...")
        t_frame, W, H, fps_meta = _scan_cv2_pos_msec(video_path)
        src = "opencv_pos_msec"

    d = np.diff(t_frame)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 5:
        fps_nom = float(fps_meta) if np.isfinite(fps_meta) and fps_meta > 1 else 30.0
    else:
        fps_nom = float(1.0 / np.median(d))

    dur_s = float(t_frame[-1]) if len(t_frame) else float("nan")

    d_all = np.diff(t_frame)
    d_pos = d_all[np.isfinite(d_all) & (d_all > 0)]
    if len(d_pos) >= 5:
        q = np.quantile(d_pos, [0.01, 0.5, 0.99])
        vfr_ratio = float(q[2] / q[0]) if q[0] > 0 else float("inf")
        vfr_report = {
            "source": src,
            "n_frames": int(len(t_frame)),
            "dur_s_by_pts": float(dur_s),
            "fps_nominal_median": float(fps_nom),
            "dt_q01_s": float(q[0]),
            "dt_q50_s": float(q[1]),
            "dt_q99_s": float(q[2]),
            "vfr_ratio_q99_over_q01": float(vfr_ratio),
            "plateau_frames": int(np.sum(d_all <= 1e-6)),
        }
    else:
        vfr_report = {
            "source": src,
            "n_frames": int(len(t_frame)),
            "dur_s_by_pts": float(dur_s),
            "fps_nominal_median": float(fps_nom),
            "note": "insufficient dt samples",
        }

    return {
        "t_frame_s": t_frame,
        "W": W,
        "H": H,
        "fps_nominal": fps_nom,
        "dur_s": dur_s,
        "vfr_report": vfr_report,
    }


def time_to_frame_idx(t_frame_s: np.ndarray, t_s: float) -> int:
    """Time (seconds) -> nearest frame index using per-frame timeline (VFR-safe)."""
    if not np.isfinite(t_s):
        return 0
    t_s = float(t_s)
    if t_s <= float(t_frame_s[0]):
        return 0
    if t_s >= float(t_frame_s[-1]):
        return int(len(t_frame_s) - 1)

    j = int(np.searchsorted(t_frame_s, t_s, side="left"))
    if j <= 0:
        return 0
    if j >= len(t_frame_s):
        return int(len(t_frame_s) - 1)

    if abs(float(t_frame_s[j]) - t_s) < abs(float(t_frame_s[j - 1]) - t_s):
        return j
    return j - 1


# =============================================================================
# DP MONOTONIC MATCH (PENALTY-FREE OBJECTIVE)
# =============================================================================
def dp_monotonic_match_penaltyfree(manual: np.ndarray, cand: np.ndarray, tol: float):
    """
    Monotonic DP match between manual anchors (sorted) and candidate anchors (sorted).

    Allowed actions (monotonic):
      - skip candidate (advance j)
      - skip manual (advance i)
      - match manual i with cand j if |err| <= tol (advance i and j)

    Penalty-free objective:
      - PRIMARY: maximize number of matches
      - SECONDARY: among max-matches solutions, minimize total absolute error across matches

    Returns:
      mae: mean absolute error across matches (None if zero matches)
      misses: number of manual anchors not matched (informational only)
      pairs: list of matched index pairs (i_manual, j_cand, abs_err)
      n_matches: number of matched pairs
      total_abs_err: total absolute error across matches
    """
    m = np.sort(np.asarray(manual, float))
    c = np.sort(np.asarray(cand, float))
    M, N = m.size, c.size

    if M == 0:
        return (0.0, 0, [], 0, 0.0)
    if N == 0:
        return (None, int(M), [], 0, 0.0)

    # dp_k[i,j] = max matches using first i manuals and first j candidates
    # dp_e[i,j] = min total abs error among solutions that achieve dp_k[i,j]
    dp_k = np.full((M + 1, N + 1), -1, dtype=int)
    dp_e = np.full((M + 1, N + 1), np.inf, dtype=float)
    bt = np.full((M + 1, N + 1), -1, dtype=int)  # 0 skip cand, 1 skip manual, 2 match

    dp_k[0, 0] = 0
    dp_e[0, 0] = 0.0

    # Initialize borders (only skipping is possible)
    for j in range(1, N + 1):
        dp_k[0, j] = 0
        dp_e[0, j] = 0.0
        bt[0, j] = 0
    for i in range(1, M + 1):
        dp_k[i, 0] = 0
        dp_e[i, 0] = 0.0
        bt[i, 0] = 1

    def better(k1, e1, k2, e2):
        """Return True if (k1,e1) is better than (k2,e2) under lexicographic: max k, min e."""
        if k1 != k2:
            return k1 > k2
        return e1 < e2 - 1e-12

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            best_k = -1
            best_e = np.inf
            best_bt = -1

            # Option A: skip candidate j-1
            kA, eA = dp_k[i, j - 1], dp_e[i, j - 1]
            if better(kA, eA, best_k, best_e):
                best_k, best_e, best_bt = kA, eA, 0

            # Option B: skip manual i-1
            kB, eB = dp_k[i - 1, j], dp_e[i - 1, j]
            if better(kB, eB, best_k, best_e):
                best_k, best_e, best_bt = kB, eB, 1

            # Option C: match if within tolerance
            e_raw = abs(float(m[i - 1]) - float(c[j - 1]))
            if e_raw <= float(tol):
                kC = dp_k[i - 1, j - 1] + 1
                eC = dp_e[i - 1, j - 1] + float(e_raw)
                if better(kC, eC, best_k, best_e):
                    best_k, best_e, best_bt = kC, eC, 2

            dp_k[i, j] = best_k
            dp_e[i, j] = best_e
            bt[i, j] = best_bt

    n_matches = int(dp_k[M, N])
    total_abs_err = float(dp_e[M, N]) if np.isfinite(dp_e[M, N]) else float("inf")
    misses = int(M - n_matches)

    # Backtrack
    pairs = []
    i, j = M, N
    while i > 0 or j > 0:
        step = bt[i, j]
        if step == 0:
            j -= 1
        elif step == 1:
            i -= 1
        elif step == 2:
            pairs.append((i - 1, j - 1, abs(float(m[i - 1]) - float(c[j - 1]))))
            i -= 1
            j -= 1
        else:
            break
    pairs.reverse()

    if n_matches == 0:
        return (None, misses, pairs, n_matches, total_abs_err)

    mae = float(total_abs_err / n_matches)
    return (mae, misses, pairs, n_matches, total_abs_err)


# =============================================================================
# MV SCORING: MedPC -> MASTER (PENALTY-FREE)
# =============================================================================
def mv_eval_params_penaltyfree(
    windows: List[dict],
    a: float,
    b: float,
    tol: float,
) -> Tuple[Tuple[int, float], dict]:
    """
    Evaluate a candidate (a,b) mapping MedPC->MASTER across multiple anchor windows.

    For each window:
      - manual anchors are in MASTER time
      - MedPC anchors are in MedPC time; map them: mapped = a*mp + b (MASTER)
      - penalty-free DP matching returns:
          matches (maximize)
          total abs error among max-match solutions (minimize)

    Combined objective across windows (lexicographic):
      1) maximize total matches across all windows
      2) minimize mean absolute error across all matched pairs

    Returns:
      key: (matches_total, mae_total) to compare lexicographically as:
           prefer larger matches_total; if tie prefer smaller mae_total
      details: dict for reporting
    """
    usable = [w for w in windows if (len(w["manual"]) > 0 and len(w["mp"]) > 0)]
    if len(usable) == 0:
        return ((0, float("inf")), {"a": float(a), "b": float(b), "matches_total": 0, "mae_total": float("inf"), "per": []})

    matches_total = 0
    err_total = 0.0
    per = []

    for w in usable:
        manual = np.sort(np.asarray(w["manual"], float))
        mapped = np.sort(float(a) * np.asarray(w["mp"], float) + float(b))

        mae, misses, pairs, n_matches, total_abs_err = dp_monotonic_match_penaltyfree(manual, mapped, tol=tol)

        matches_total += int(n_matches)
        if np.isfinite(total_abs_err):
            err_total += float(total_abs_err)

        per.append({
            "name": w["name"],
            "n_manual": int(len(manual)),
            "n_candidates": int(len(mapped)),
            "n_matches": int(n_matches),
            "misses": int(misses),
            "mae": (float(mae) if mae is not None else None),
        })

    mae_total = float(err_total / matches_total) if matches_total > 0 else float("inf")

    details = {
        "a": float(a),
        "b": float(b),
        "matches_total": int(matches_total),
        "mae_total": float(mae_total),
        "err_total": float(err_total),
        "per": per,
    }
    key = (int(matches_total), float(mae_total))
    return key, details


def best_affine_medpc_to_master_penaltyfree(windows, tol: float, b_min: float, b_max: float):
    """
    Grid search for best (a,b) mapping MedPC->MASTER using a penalty-free objective.

    Compare candidates by:
      - higher matches_total is better
      - if ties, lower mae_total is better
    """
    usable = [w for w in windows if (len(w["manual"]) > 0 and len(w["mp"]) > 0)]
    if len(usable) == 0:
        raise RuntimeError("No usable MV windows (manual or MedPC windows empty).")

    a_list = np.arange(A_PRIOR - A_MV_PAD, A_PRIOR + A_MV_PAD + 1e-12, A_MV_STEP)
    b_list = np.arange(float(b_min), float(b_max) + 1e-12, B_MV_STEP)

    best_details = None
    best_key = None

    total_comb = int(len(a_list) * len(b_list))
    p = tqdm(total=total_comb, desc="[ALIGN MV] Grid search (a,b) -> MASTER (penalty-free)", unit="comb",
             leave=True, mininterval=0.2) if tqdm else None

    for a in a_list:
        for b in b_list:
            key, details = mv_eval_params_penaltyfree(usable, float(a), float(b), tol=tol)

            if best_key is None:
                best_key = key
                best_details = details
            else:
                # Prefer higher matches_total; if tie, lower mae_total
                if (key[0] > best_key[0]) or (key[0] == best_key[0] and key[1] < best_key[1] - 1e-12):
                    best_key = key
                    best_details = details

            if p:
                p.update(1)

    if p:
        p.close()

    if best_details is None:
        raise RuntimeError("Failed MV estimation.")
    return best_details


# =============================================================================
# GPU-VECTORIZED PERI-EVENT ROBUST Z + INCREASE PEAK
# =============================================================================
def peri_event_robustz_and_peak_gpu(
    t_fiber: np.ndarray,
    dff: np.ndarray,
    event_times: np.ndarray,
    epoch: np.ndarray,
    base_mask: np.ndarray,
    search_mask: np.ndarray,
    prefer_gpu: bool = True,
):
    """
    Peri-event extraction + robust Z scoring:
      1) interpolate dff at (event_time + epoch)
      2) baseline median and MAD in base_mask
      3) robust-Z = (trace - median) / MAD
      4) INCREASE-only peak = max robust-Z in search_mask
    """
    xp = xp_backend(prefer_gpu=prefer_gpu)
    t_x = xp.asarray(t_fiber, dtype=xp.float64)
    d_x = xp.asarray(dff, dtype=xp.float64)
    ev_x = xp.asarray(event_times, dtype=xp.float64)
    ep_x = xp.asarray(epoch, dtype=xp.float64)

    n_ev = int(ev_x.size)
    n_ep = int(ep_x.size)
    if n_ev == 0:
        return np.zeros((0, n_ep)), np.array([]), np.array([])

    q = (ev_x[:, None] + ep_x[None, :]).reshape(-1)
    yq = xp.interp(q, t_x, d_x).reshape(n_ev, n_ep)

    base = yq[:, base_mask]
    med = xp.median(base, axis=1)
    mad = xp.median(xp.abs(base - med[:, None]), axis=1)

    mad_safe = xp.where(mad <= 0, xp.nan, mad)
    rz = (yq - med[:, None]) / mad_safe[:, None]

    zz = rz[:, search_mask]
    zz2 = xp.nan_to_num(zz, nan=-xp.inf)
    kmax = xp.argmax(zz2, axis=1)
    peak_z = xp.take_along_axis(zz, kmax[:, None], axis=1).reshape(-1)

    search_t = xp.asarray(epoch[search_mask], dtype=xp.float64)
    peak_off = search_t[kmax]

    if xp is cp:
        return cp.asnumpy(rz), cp.asnumpy(peak_z), cp.asnumpy(peak_off)
    return np.asarray(rz), np.asarray(peak_z), np.asarray(peak_off)


def auc_trapz(y: np.ndarray, x: np.ndarray, t0: float, t1: float) -> float:
    """Trapezoidal AUC of y(x) between [t0,t1]."""
    m = (x >= t0) & (x <= t1) & np.isfinite(y) & np.isfinite(x)
    if int(np.sum(m)) < 2:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y[m], x[m]))
    return float(((y[m][1:] + y[m][:-1]) * 0.5 * np.diff(x[m])).sum())


def choose_top_indices(scores: np.ndarray, times_for_refractory: np.ndarray, refractory: float, max_keep: int = 400):
    """Pick indices by descending score with a refractory constraint on times_for_refractory."""
    key = -np.nan_to_num(scores, nan=-1e9)
    order = np.argsort(key)
    chosen = []
    used = []
    for idx in order:
        t0 = float(times_for_refractory[idx])
        if any(abs(t0 - u) < refractory for u in used):
            continue
        chosen.append(int(idx))
        used.append(t0)
        if len(chosen) >= max_keep:
            break
    return np.array(chosen, dtype=int)


# =============================================================================
# BOOTSTRAP: MV (PENALTY-FREE) around best params
# =============================================================================
def bootstrap_mv_penaltyfree(windows, best_a: float, best_b: float, tol: float, n_boot: int, seed: int):
    """
    Bootstrap stability analysis for MV parameters (a,b), penalty-free objective.

    Resampling:
      - resample manual anchors with replacement within each window
      - local grid search around best (a,b)
      - store best (a,b) + summary metrics
    """
    rng = np.random.default_rng(seed)
    usable = [w for w in windows if (len(w["manual"]) > 0 and len(w["mp"]) > 0)]
    if len(usable) == 0:
        return pd.DataFrame()

    a_list = np.arange(best_a - MV_BOOT_A_PAD, best_a + MV_BOOT_A_PAD + 1e-12, MV_BOOT_A_STEP)
    b_list = np.arange(best_b - MV_BOOT_B_PAD, best_b + MV_BOOT_B_PAD + 1e-12, MV_BOOT_B_STEP)

    rows = []
    boot_iter = _pbar(range(int(n_boot)), desc="[BOOTSTRAP MV] Resamples (penalty-free)", unit="boot", leave=True)
    for _ in boot_iter:
        boot_wins = []
        for w in usable:
            m = np.asarray(w["manual"], float)
            idx = rng.integers(0, len(m), size=len(m))
            boot_manual = m[idx]
            boot_wins.append({"name": w["name"], "manual": boot_manual, "mp": w["mp"]})

        best_key = None
        best_det = None
        for a in a_list:
            for b in b_list:
                key, det = mv_eval_params_penaltyfree(boot_wins, float(a), float(b), tol=tol)
                if best_key is None:
                    best_key, best_det = key, det
                else:
                    if (key[0] > best_key[0]) or (key[0] == best_key[0] and key[1] < best_key[1] - 1e-12):
                        best_key, best_det = key, det

        rows.append({
            "a": best_det["a"],
            "b": best_det["b"],
            "matches_total": int(best_det["matches_total"]),
            "mae_total": float(best_det["mae_total"]),
            "err_total": float(best_det["err_total"]),
        })

    return pd.DataFrame(rows)


def qstats(x: np.ndarray):
    """Median, 16th pct, 84th pct, std."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (float(np.median(x)), float(np.quantile(x, 0.16)), float(np.quantile(x, 0.84)), float(np.std(x)))


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
ROOT.mkdir(parents=True, exist_ok=True)

case_root = ROOT / "case_self_administrator"
figdir = case_root / "figures"
viddir = case_root / "videos_increase"
valdir = case_root / "validation"

case_root.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)
viddir.mkdir(parents=True, exist_ok=True)
valdir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 0) MANUAL ANCHORS (VIDEO -> MASTER)
# =============================================================================
manual_0_10_video = np.array([parse_mmss_to_seconds(x) for x in manual_0_10_txt], dtype=float)
manual_mid_video = np.array([parse_mmss_to_seconds(x) for x in manual_mid_txt], dtype=float)
manual_35_45_video = np.array([parse_mmss_to_seconds(x) for x in manual_35_45_txt], dtype=float)

def video_to_master_times(x_video: np.ndarray, trim_start: float) -> np.ndarray:
    """VIDEO seconds -> MASTER seconds by subtracting TRIM_START_SEC; drop negatives."""
    x = np.asarray(x_video, float) - float(trim_start)
    x = x[np.isfinite(x)]
    x = x[x >= 0.0]
    return np.sort(x)

manual_0_10_master = video_to_master_times(manual_0_10_video, TRIM_START_SEC)
manual_mid_master = video_to_master_times(manual_mid_video, TRIM_START_SEC)
manual_35_45_master = video_to_master_times(manual_35_45_video, TRIM_START_SEC)

# =============================================================================
# 1) LOAD VIDEO TIMELINE (VFR SAFE) -> TRIM + REZERO => MASTER VIDEO TIME
# =============================================================================
if USE_FAKE_VIDEO:
    fps_nominal = 20.0
    dur_v = 15 * 60
    t_frame = np.arange(0, dur_v, 1 / fps_nominal)
    W, H = 640, 480
    vt = {"t_frame_s": t_frame, "dur_s": dur_v, "vfr_report": {"source": "FAKE"}, "W": W, "H": H, "fps_nominal": fps_nominal}
    print("[VIDEO] FAKE VIDEO MODE ENABLED — skipping real video load")
else:
    vt = build_video_timeline(VIDEO_FILE)
    t_frame = vt["t_frame_s"]
    W = vt["W"]
    H = vt["H"]
    fps_nominal = vt["fps_nominal"]
    dur_v = vt["dur_s"]
    print(f"[VIDEO] frames={len(t_frame)} dur_by_pts≈{dur_v:.3f}s size={W}x{H} nominal_fps≈{fps_nominal:.3f}")
    print(f"[VIDEO/VFR] {vt['vfr_report']}")

keep_v = t_frame >= float(TRIM_START_SEC)
if int(np.sum(keep_v)) < 10:
    raise RuntimeError("Video too short after TRIM_START_SEC or invalid timestamps.")

start_frame_offset = int(np.argmax(keep_v))
t_frame_master = t_frame[keep_v] - float(TRIM_START_SEC)
dur_v_master = float(t_frame_master[-1])

vfr_report_path = valdir / "video_vfr_report.json"
with open(vfr_report_path, "w", encoding="utf-8") as f:
    json.dump(vt["vfr_report"], f, indent=2)
print(f"[OK] Saved: {vfr_report_path}")
print(f"[MASTER/VIDEO] TRIM_START_SEC={TRIM_START_SEC:.3f} start_frame_offset={start_frame_offset} dur_v_master≈{dur_v_master:.3f}s")

# =============================================================================
# 2) LOAD FIBER -> TRIM + REZERO => MASTER FIBER TIME
# =============================================================================
fdf = pd.read_csv(FIBER_FILE)
time_col, sig_col, ctl_col = infer_fiber_cols(fdf)

t_raw = pd.to_numeric(fdf[time_col], errors="coerce")
sig_raw = pd.to_numeric(fdf[sig_col], errors="coerce")
ctl_raw = pd.to_numeric(fdf[ctl_col], errors="coerce")
mask = t_raw.notna() & sig_raw.notna() & ctl_raw.notna()

t_raw = t_raw[mask].to_numpy(dtype=float)
sig_raw = sig_raw[mask].to_numpy(dtype=float)
ctl_raw = ctl_raw[mask].to_numpy(dtype=float)

t_sec, t_scale = normalize_to_seconds_by_dt(t_raw)

keep_f = t_sec >= float(TRIM_START_SEC)
if int(np.sum(keep_f)) < 10:
    raise RuntimeError("Fiber too short after TRIM_START_SEC or invalid timestamps.")

t_master = t_sec[keep_f] - float(TRIM_START_SEC)
sig = sig_raw[keep_f]
ctl = ctl_raw[keep_f]

dt = float(np.nanmedian(np.diff(t_master)))
if not np.isfinite(dt) or dt <= 0:
    raise RuntimeError("Fiber timestamps invalid (dt<=0).")
fs = 1.0 / dt

print(
    f"[FIBER] time={time_col} signal={sig_col} control={ctl_col} "
    f"time_scale={t_scale} Fs≈{fs:.3f}Hz "
    f"MASTER range={t_master.min():.2f}..{t_master.max():.2f}s"
)

sig_n, ctl_n, trend_sig, trend_ctl = photobleach_correct(sig, ctl, fs=fs, tau_sec=PB_TAU_SEC)
a_fit, b_fit = np.polyfit(ctl_n, sig_n, 1)
fit_ctl = a_fit * ctl_n + b_fit
dff = (sig_n - fit_ctl) / fit_ctl
print(f"[DFF] OLS on PB-corrected: a={a_fit:.6f} b={b_fit:.6f}")

epoch = np.arange(-PRE, POST + dt, dt)
base_mask = (epoch >= BASE_START) & (epoch < BASE_END)
search_mask = (epoch >= PEAK_SEARCH_START) & (epoch <= PEAK_SEARCH_END)

pb_png = figdir / "photobleach_trends_master.png"
plt.figure(figsize=(12, 5))
plt.plot(t_master, sig, linewidth=0.7, label="Signal raw", alpha=0.7)
plt.plot(t_master, trend_sig, linewidth=2.0, label=f"Signal EMA trend (tau={PB_TAU_SEC}s)")
plt.plot(t_master, ctl, linewidth=0.7, label="Control raw", alpha=0.7)
plt.plot(t_master, trend_ctl, linewidth=2.0, label=f"Control EMA trend (tau={PB_TAU_SEC}s)")
plt.xlabel("MASTER time (s)")
plt.ylabel("Fluorescence (raw units)")
plt.title("Photobleaching / drift trends (EMA) on MASTER fiber timeline")
plt.legend()
plt.tight_layout()
plt.savefig(pb_png, dpi=200)
plt.close()
print(f"[OK] Saved: {pb_png}")

# =============================================================================
# 3) DEFINE MASTER DURATION AND TRUNCATE FIBER TO MATCH VIDEO
# =============================================================================
dur_master = float(min(dur_v_master, float(np.nanmax(t_master))))
keep_master = t_master <= dur_master

t_master = t_master[keep_master]
dff = dff[keep_master]
sig = sig[keep_master]
ctl = ctl[keep_master]
trend_sig = trend_sig[keep_master]
trend_ctl = trend_ctl[keep_master]

print(f"[MASTER] dur_master=min(video,fiber)={dur_master:.3f}s | fiber points kept={len(t_master)}")

# =============================================================================
# 4) LOAD MEDPC (SELF_ADMIN ONLY) + APPLY DT (NO REZERO)
# =============================================================================
bdf = pd.read_csv(BEH_FILE)
if MEDPC_COL not in bdf.columns:
    raise KeyError(f"Column '{MEDPC_COL}' not found in {BEH_FILE.name}.")

mp_raw = pd.to_numeric(bdf[MEDPC_COL], errors="coerce").dropna().to_numpy(dtype=float)
mp_raw = mp_raw[mp_raw > 0]
mp_raw = np.sort(mp_raw)

mp_s = mp_raw * float(MEDPC_DT_SEC)
mp_s = cluster_events(mp_s, CLUSTER_WIN)
mp_idx = np.arange(1, len(mp_s) + 1, dtype=int)

if mp_s.size:
    print(f"[MEDPC] col={MEDPC_COL} n={len(mp_s)} dt={MEDPC_DT_SEC}s range={mp_s.min():.2f}..{mp_s.max():.2f}s")
else:
    print(f"[MEDPC] col={MEDPC_COL} n=0")

# =============================================================================
# 5) MV ALIGNMENT: MedPC -> MASTER (PENALTY-FREE DP)
# =============================================================================
if USE_CLOCK_PRIOR:
    b_prior = calculate_clock_offset(MEDPC_START_CLOCK, VIDEO_START_CLOCK)
else:
    b_prior = 0.0

if B_PRIOR_OVERRIDE is not None:
    b_prior = float(B_PRIOR_OVERRIDE)

B_MV_MIN = float(b_prior - B_PAD_AROUND_PRIOR)
B_MV_MAX = float(b_prior + B_PAD_AROUND_PRIOR)

print(f"[MV PRIOR] b_prior (MedPC - VideoClock) = {b_prior:.3f}s -> search b in [{B_MV_MIN:.3f},{B_MV_MAX:.3f}]")
print("[MV TARGET] t_master = a_mv * t_medpc + b_mv (MASTER seconds)")
print("[MV OBJECTIVE] Penalty-free: maximize matches (within tolerance), then minimize absolute error.")

mv_windows = [
    {"name": "w0_10", "manual": manual_0_10_master, "mp": mp_s},
    {"name": "wmid", "manual": manual_mid_master, "mp": mp_s},
    {"name": "w35_45", "manual": manual_35_45_master, "mp": mp_s},
]

mv_best = best_affine_medpc_to_master_penaltyfree(
    mv_windows, tol=TOL_MANUAL, b_min=B_MV_MIN, b_max=B_MV_MAX
)
a_mv = float(mv_best["a"])
b_mv = float(mv_best["b"])

print(f"[ALIGN MV] t_master = {a_mv:.9f} * t_medpc + {b_mv:.6f}")
print(f"          matches_total={mv_best['matches_total']} mae_total={mv_best['mae_total']:.6f}s (across matched anchors)")
for it in mv_best["per"]:
    mae_str = "None" if it["mae"] is None else f"{it['mae']:.3f}s"
    print(f"  {it['name']}: matches={it['n_matches']} / manual={it['n_manual']} | mae={mae_str}")

def print_nearest_for_manual(name, manual, mp_s, a, b, k=3):
    manual = np.sort(np.asarray(manual, float))
    mapped = np.sort(a*np.asarray(mp_s, float) + b)
    print(f"\n[NEAREST] {name}")
    for t in manual:
        j = int(np.searchsorted(mapped, t))
        cand_idx = np.clip(np.array([j-1, j, j+1]), 0, len(mapped)-1)
        cand_idx = np.unique(cand_idx)
        errs = [(mapped[ii], abs(mapped[ii]-t)) for ii in cand_idx]
        errs.sort(key=lambda x: x[1])
        top = errs[:k]
        print(f"  manual {t:8.3f}s -> " + " | ".join([f"{tm:8.3f}s err={e:6.3f}" for tm,e in top]))

for w in mv_windows:
    print_nearest_for_manual(w["name"], w["manual"], mp_s, a_mv, b_mv, k=3)

if DEBUG_MATCH_PRINT:
    for w in mv_windows:
        if w["name"] != "wmid":
            continue
        manual = np.sort(np.asarray(w["manual"], float))
        mapped = np.sort(a_mv * np.asarray(w["mp"], float) + b_mv)
        mae, miss, pairs, n_matches, total_abs_err = dp_monotonic_match_penaltyfree(manual, mapped, tol=TOL_MANUAL)
        print(f"[DEBUG wmid] matches={n_matches} miss={miss} mae={mae} tol={TOL_MANUAL}")
        for kk, (i_m, j_c, e) in enumerate(pairs[:DEBUG_MATCH_MAX], start=1):
            print(f"   pair#{kk}: manual[{i_m}]={manual[i_m]:.3f}s  cand[{j_c}]={mapped[j_c]:.3f}s  |err|={e:.3f}s")

mv_report = valdir / "mv_penaltyfree_master_report.json"
with open(mv_report, "w", encoding="utf-8") as f:
    json.dump(mv_best, f, indent=2)
print(f"[OK] Saved: {mv_report}")

# =============================================================================
# 6) ALL EVENTS: peri-event robust Z + INCREASE peak (ALL ON MASTER)
# =============================================================================
t_event_master = a_mv * mp_s + b_mv

ok_fiber = (t_event_master - PRE >= float(t_master.min())) & (t_event_master + POST <= float(t_master.max()))
mp_ok = mp_s[ok_fiber]
mp_ok_idx = mp_idx[ok_fiber]
t_event_ok = t_event_master[ok_fiber]

print(f"[OK] Events usable in fiber (MASTER): n={len(t_event_ok)}")

rz_trials, peak_z, peak_off = peri_event_robustz_and_peak_gpu(
    t_fiber=t_master,
    dff=dff,
    event_times=t_event_ok,
    epoch=epoch,
    base_mask=base_mask,
    search_mask=search_mask,
    prefer_gpu=PREFER_GPU,
)

t_peak_master = t_event_ok + peak_off

ok_video = (t_peak_master - CLIP_PRE >= 0) & (t_peak_master + CLIP_POST <= dur_master) & np.isfinite(t_peak_master)
print(f"[OK] Events usable in video clips (MASTER, +/- {CLIP_PRE:.1f}s): n={int(np.sum(ok_video))}")

all_events_df = pd.DataFrame({
    "nosepoke_index": mp_ok_idx.astype(int),
    "t_medpc_s": mp_ok.astype(float),
    "t_master_event_s": t_event_ok.astype(float),
    "peak_increase_robust_z": peak_z.astype(float),
    "peak_increase_offset_master_s": peak_off.astype(float),
    "t_master_peak_s": t_peak_master.astype(float),
    "usable_in_video": ok_video.astype(bool),
    "a_mv": a_mv,
    "b_mv": b_mv,
    "TRIM_START_SEC": float(TRIM_START_SEC),
    "start_frame_offset": int(start_frame_offset),
})
all_events_csv = case_root / "all_aligned_events_increase_only_master.csv"
all_events_df.to_csv(all_events_csv, index=False)
print(f"[OK] Saved: {all_events_csv}")

# =============================================================================
# 7) SELECT TOP CLIPS (INCREASE-only) WITH REFRACTORY (MEDPC refractory)
# =============================================================================
chosen = choose_top_indices(peak_z, mp_ok, EVENT_REFRACTORY, max_keep=400)

valid = []
for idx0 in chosen:
    if not ok_video[idx0]:
        continue
    valid.append((idx0, float(t_peak_master[idx0]), float(t_event_ok[idx0])))
    if len(valid) >= TARGET_CLIPS:
        break
print(f"[OK] Valid INCREASE clips: n={len(valid)} (target={TARGET_CLIPS})")

top10 = valid[:10]
summary_rows = []
for rank, (idx0, tpkm, tev_m) in enumerate(top10, start=1):
    summary_rows.append({
        "rank": rank,
        "nosepoke_index": int(mp_ok_idx[idx0]),
        "t_medpc_s": float(mp_ok[idx0]),
        "t_master_event_s": float(tev_m),
        "peak_increase_robust_z": float(peak_z[idx0]),
        "peak_increase_offset_master_s": float(peak_off[idx0]),
        "t_master_peak_s": float(tpkm),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = case_root / "top10_increase_summary_master.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"[OK] Saved: {summary_csv}")

# =============================================================================
# 8) TOP10 PERI-EVENT GRID PLOT (INCREASE-only)
# =============================================================================
grid_png = figdir / "top10_peri_event_robustZ_grid_increase_master.png"
if len(top10) > 0:
    plt.figure(figsize=(18, 6))
    ncols = 5
    nrows = int(math.ceil(len(top10) / ncols))
    for j, (idx0, _, _) in enumerate(top10, start=1):
        ax = plt.subplot(nrows, ncols, j)
        ax.plot(epoch, rz_trials[idx0], linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axvline(float(peak_off[idx0]), linestyle=":", linewidth=1)
        ax.set_title(f"INC #{j} | Z={peak_z[idx0]:.2f} | NP#{int(mp_ok_idx[idx0])} | off={peak_off[idx0]:.2f}s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Robust Z (dF/F)")
        ax.set_xlim(-PRE, POST)
    plt.tight_layout()
    plt.savefig(grid_png, dpi=200)
    plt.close()
    print(f"[OK] Saved: {grid_png}")

# =============================================================================
# 9) MEAN PERI-EVENT + AUC
# =============================================================================
finite_rows = np.all(np.isfinite(rz_trials), axis=1)
rz_valid = rz_trials[finite_rows]

if rz_valid.shape[0] >= 5:
    mean_rz = np.nanmean(rz_valid, axis=0)
    sem_rz = np.nanstd(rz_valid, axis=0, ddof=1) / np.sqrt(rz_valid.shape[0])

    auc_net_0_3 = auc_trapz(mean_rz, epoch, 0.0, 3.0)
    auc_net_0_5 = auc_trapz(mean_rz, epoch, 0.0, 5.0)

    mean_pos = np.where(mean_rz > 0, mean_rz, 0.0)
    auc_pos_0_3 = auc_trapz(mean_pos, epoch, 0.0, 3.0)
    auc_pos_0_5 = auc_trapz(mean_pos, epoch, 0.0, 5.0)

    auc_df = pd.DataFrame([{
        "medpc_column": MEDPC_COL,
        "n_events_used": int(rz_valid.shape[0]),
        "AUC_net_mean_trace_0_3s": auc_net_0_3,
        "AUC_net_mean_trace_0_5s": auc_net_0_5,
        "AUC_posonly_mean_trace_0_3s": auc_pos_0_3,
        "AUC_posonly_mean_trace_0_5s": auc_pos_0_5,
        "MASTER_trim_sec": float(TRIM_START_SEC),
        "dur_master_s": float(dur_master),
    }])
    auc_csv = case_root / "all_events_mean_trace_auc_increase_only_master.csv"
    auc_df.to_csv(auc_csv, index=False)
    print(f"[OK] Saved: {auc_csv}")

    avg_png = figdir / "all_events_mean_peri_event_robustZ_master.png"
    plt.figure(figsize=(10, 5))
    plt.plot(epoch, mean_rz, linewidth=2, label="Mean Robust Z")
    plt.fill_between(epoch, mean_rz - sem_rz, mean_rz + sem_rz, alpha=0.25, label="SEM")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlim(-PRE, POST)
    plt.xlabel("Time (s) relative to event")
    plt.ylabel("Robust Z (dF/F)")
    plt.title(f"All aligned events (n={rz_valid.shape[0]}) | NetAUC(0-3)={auc_net_0_3:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(avg_png, dpi=200)
    plt.close()
    print(f"[OK] Saved: {avg_png}")
else:
    print("[WARN] Not enough finite peri-event trials for mean/SEM and AUC.")

# =============================================================================
# 10) WRITE INCREASE CLIPS (VFR SAFE) USING MASTER -> ORIGINAL FRAME OFFSET
# =============================================================================
def write_increase_clips_master_vfrsafe(valid_list, out_dir: Path, t_frame_master_s: np.ndarray, fps_write: float, start_frame_offset: int):
    """Export clips for selected peaks using MASTER->frame index mapping and original frame offset."""
    if USE_FAKE_VIDEO:
        print("[FAKE VIDEO] skipping clip writing")
        return

    cap = cv2.VideoCapture(str(VIDEO_FILE))
    if not cap.isOpened():
        raise RuntimeError("Could not open video file (codec).")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    clip_iter = _pbar(list(enumerate(valid_list, start=1)), desc="[VIDEO] Writing clips (MASTER)", unit="clip", leave=True)
    for rank, (idx0, tpk_master, _tev_master) in clip_iter:
        start_t = float(tpk_master - CLIP_PRE)
        end_t = float(tpk_master + CLIP_POST)

        start_fr_m = time_to_frame_idx(t_frame_master_s, start_t)
        end_fr_m = time_to_frame_idx(t_frame_master_s, end_t)

        start_fr_m = max(0, min(int(start_fr_m), int(len(t_frame_master_s) - 1)))
        end_fr_m = max(0, min(int(end_fr_m), int(len(t_frame_master_s) - 1)))
        if end_fr_m < start_fr_m:
            start_fr_m, end_fr_m = end_fr_m, start_fr_m

        start_fr_orig = int(start_frame_offset + start_fr_m)
        end_fr_orig = int(start_frame_offset + end_fr_m)

        out_clip = out_dir / f"{rank}.mp4"
        writer = cv2.VideoWriter(str(out_clip), fourcc, float(fps_write), (W, H))

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_fr_orig))
        cur = int(start_fr_orig)

        np_index = int(mp_ok_idx[idx0])
        t_medpc = float(mp_ok[idx0])
        zpk = float(peak_z[idx0])
        pkoff = float(peak_off[idx0])

        frames_to_write = int(end_fr_orig - start_fr_orig + 1)
        p = tqdm(total=frames_to_write, desc=f"  clip {rank} frames", unit="fr", leave=False, mininterval=0.3) if tqdm else None

        while cur <= int(end_fr_orig):
            ret, frame = cap.read()
            if not ret:
                break

            line1 = f"{MEDPC_COL} | INCREASE | Clip {rank} | PeakZ={zpk:.2f} | MasterPeak={tpk_master:.2f}s"
            line2 = f"MedPC={t_medpc:.2f}s (dt={MEDPC_DT_SEC:.1f}s) | NosepokeIndex={np_index} | PeakOffset={pkoff:.2f}s"
            line3 = f"MASTER=Video+Fiber trimmed {TRIM_START_SEC:.2f}s then rezero | VFR-safe timeline | orig_frame_offset={start_frame_offset}"

            cv2.putText(frame, line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, line2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, line3, (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            cur += 1
            if p:
                p.update(1)

        if p:
            p.close()
        writer.release()
        print(f"[OK] Clip saved: {out_clip} (MASTER t {start_t:.3f}..{end_t:.3f}s)")

    cap.release()


write_increase_clips_master_vfrsafe(
    valid_list=valid,
    out_dir=viddir,
    t_frame_master_s=t_frame_master,
    fps_write=fps_nominal,
    start_frame_offset=start_frame_offset,
)

# =============================================================================
# 11) BOOTSTRAP: MV stability (PENALTY-FREE)
# =============================================================================
print("[BOOTSTRAP] Running MV bootstrap (penalty-free DP) on MASTER...")
mv_boot = bootstrap_mv_penaltyfree(
    windows=mv_windows,
    best_a=a_mv,
    best_b=b_mv,
    tol=TOL_MANUAL,
    n_boot=BOOTSTRAP_MV_N,
    seed=BOOTSTRAP_SEED,
)
mv_boot_csv = valdir / "bootstrap_mv_penaltyfree_master.csv"
mv_boot.to_csv(mv_boot_csv, index=False)
print(f"[OK] Saved: {mv_boot_csv}")

mv_a_med, mv_a_q16, mv_a_q84, mv_a_sd = qstats(mv_boot["a"].to_numpy()) if len(mv_boot) else (np.nan, np.nan, np.nan, np.nan)
mv_b_med, mv_b_q16, mv_b_q84, mv_b_sd = qstats(mv_boot["b"].to_numpy()) if len(mv_boot) else (np.nan, np.nan, np.nan, np.nan)
mv_m_med, mv_m_q16, mv_m_q84, mv_m_sd = qstats(mv_boot["matches_total"].to_numpy()) if len(mv_boot) else (np.nan, np.nan, np.nan, np.nan)
mv_e_med, mv_e_q16, mv_e_q84, mv_e_sd = qstats(mv_boot["mae_total"].to_numpy()) if len(mv_boot) else (np.nan, np.nan, np.nan, np.nan)

# =============================================================================
# 12) VALIDATION REPORT (TEXT) INCLUDING MASTER CONCEPT
# =============================================================================
val_report = valdir / "validation_report_master.txt"
with open(val_report, "w", encoding="utf-8") as f:
    f.write("VALIDATION REPORT (MASTER-CLOCK + MV PENALTY-FREE DP + BOOTSTRAP)\n\n")

    f.write("MASTER clock definition:\n")
    f.write(f"  TRIM_START_SEC={TRIM_START_SEC:.6f}\n")
    f.write("  MASTER = VIDEO and FIBER trimmed by TRIM_START_SEC then re-zeroed\n")
    f.write(f"  Video: start_frame_offset={start_frame_offset} dur_v_master={dur_v_master:.6f}\n")
    f.write(f"  Fiber: dur_f_master={float(np.nanmax(t_master)):.6f}\n")
    f.write(f"  dur_master=min(video,fiber)={dur_master:.6f}\n\n")

    f.write("MV Prior (only for b search centering):\n")
    f.write(f"  USE_CLOCK_PRIOR={USE_CLOCK_PRIOR}\n")
    f.write(f"  MEDPC_START_CLOCK={MEDPC_START_CLOCK}\n")
    f.write(f"  VIDEO_START_CLOCK={VIDEO_START_CLOCK}\n")
    f.write(f"  b_prior={b_prior:.6f}\n")
    f.write(f"  b_search=[{B_MV_MIN:.3f},{B_MV_MAX:.3f}] step={B_MV_STEP}\n\n")

    f.write("MV Objective (penalty-free):\n")
    f.write("  1) maximize number of matched anchors within tolerance\n")
    f.write("  2) among those, minimize total absolute error across matches\n\n")

    f.write("MV Best (MedPC -> MASTER):\n")
    f.write(f"  t_master = {a_mv:.12f} * t_medpc + {b_mv:.6f}\n")
    f.write(f"  matches_total={mv_best['matches_total']} mae_total={mv_best['mae_total']:.6f}\n\n")

    f.write("No MedPC->Fiber warping:\n")
    f.write("  Fiber is already on MASTER time; events are projected into MASTER via MV only.\n")
    f.write("  Peak time: t_master_peak = t_master_event + peak_offset\n\n")

    f.write("Bootstraps:\n")
    f.write(f"  MV boot N={BOOTSTRAP_MV_N}\n")
    f.write(f"    a_med={mv_a_med:.12f} a_q16={mv_a_q16:.12f} a_q84={mv_a_q84:.12f}\n")
    f.write(f"    b_med={mv_b_med:.6f} b_q16={mv_b_q16:.6f} b_q84={mv_b_q84:.6f}\n")
    f.write(f"    matches_med={mv_m_med:.3f} matches_q16={mv_m_q16:.3f} matches_q84={mv_m_q84:.3f}\n")
    f.write(f"    mae_med={mv_e_med:.6f} mae_q16={mv_e_q16:.6f} mae_q84={mv_e_q84:.6f}\n")

print(f"[OK] Saved: {val_report}")
print(f"[DONE] Outputs: {case_root}")
