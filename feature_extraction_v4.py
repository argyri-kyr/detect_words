"""
=============================================================
GREEK READ-ALOUD CLASSIFIER  —  Feature Extraction Module
=============================================================

Folder structure handled:
  Case A:  subject_{id}/ee_14334_1_visitloud_wa_4_5_2025-...avi
           → extract .wav from .avi, save as subject_{id}/{id}_visitloud.wav
  Case B:  subject_{id}/ste_11896_1_visitloud_.../AudioRecording.wav
           → use .wav directly

Class codes extracted from filename prefix:
  'be', 'ge', 'de', 'ee', 'ste', 'ag', 'bg', 'gg'

FEATURES EXTRACTED (8 raw → 8 z-scored):
  total_duration_sec      ← model coef 0.794  (strongest predictor)
  speech_duration_sec     ← model coef 0.652
  pause_count             ← model coef 0.649
  speech_segments_count   ← model coef 0.576
  pause_long_count        ← model coef 0.568
  pause_ratio             ← selected 4/5 folds
  pause_mean_sec          ← kept for report generation
  pause_short_count       ← kept for report generation

REMOVED vs previous version:
  Acoustic     : MFCCs, spectral_centroid, RMS, ZCR, F0
                 → dominated by mic/room variation, not reading ability
  Quality flags: audio_quality_very_quiet/noisy/too_short, audio_speech_ratio
                 → binary, unreliable, don't predict reading difficulty
  Rate         : syllables_per_sec, onset_count
                 → inconsistent direction across classes, never selected
  Syllabic     : short_burst_ratio, burst_mean_sec, short_burst_count
                 → correlated with speech_segments_count, never selected
  Redundant    : silence_duration_sec, pause_std_sec, pause_max_sec,
                 pause_total_sec, burst_std_sec

All features stored RAW here.
Per-class z-score normalisation happens AFTER all subjects are
extracted, once population statistics (mean/std per class) are known.
"""

import os
import json
import warnings
import numpy as np
import librosa

warnings.filterwarnings("ignore")


from globalVars import SAMPLE_RATE, KNOWN_CLASSES, VAD_TOP_DB, PAUSE_MIN_SEC, PAUSE_SHORT, PAUSE_LONG, FEATURES_TO_NORMALISE


# ─────────────────────────────────────────────
# 1.  GET CLASS CODE FROM SUBJECT FOLDER
# ─────────────────────────────────────────────

def get_class_code(sub_path: str) -> str:
    """
    Extract class code from the AVI/WAV filename or subfolder name
    inside the subject folder.

    Looks for an entry that starts with a known class code + '_'.

    Examples:
        ee_14334_1_visitloud_wa_4_5_2025-10-24-15.37.14.avi  → 'ee'
        ste_11896_1_visitloud_wa_4_6_2022-06-04-12.57.59/    → 'ste'

    Returns 'unknown' if no match found.
    """
    try:
        entries = os.listdir(sub_path)
    except FileNotFoundError:
        return "unknown"

    for entry in entries:
        name_lower = entry.lower()
        for code in KNOWN_CLASSES:
            if name_lower.startswith(code + "_"):
                return code

    return "unknown"


# ─────────────────────────────────────────────
# 2.  AUDIO LOADING
# ─────────────────────────────────────────────

def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE):
    """
    Load audio file, resample to target_sr, convert to mono.
    Returns (y: np.ndarray, sr: int).
    """
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return y, sr


# ─────────────────────────────────────────────
# 3.  VOICE ACTIVITY DETECTION
# ─────────────────────────────────────────────

def get_speech_silence(y: np.ndarray, sr: int, top_db: float = VAD_TOP_DB):
    """
    Split waveform into speech and silence intervals using
    librosa's energy-based VAD (librosa.effects.split).

    This is the core of Pipeline B — all features derive from
    these two interval lists.

    Returns:
        speech_intervals  : list of (start_sec, end_sec)
        silence_intervals : list of (start_sec, end_sec)
        total_speech_sec  : float
    """
    non_silent = librosa.effects.split(y, top_db=top_db)

    speech_intervals  = []
    silence_intervals = []
    prev_end          = 0.0

    for start_frame, end_frame in non_silent:
        s = start_frame / sr
        e = end_frame   / sr
        if s > prev_end:                     # gap before this segment = pause
            silence_intervals.append((prev_end, s))
        speech_intervals.append((s, e))
        prev_end = e

    total_dur = len(y) / sr
    if prev_end < total_dur:                 # trailing silence
        silence_intervals.append((prev_end, total_dur))

    total_speech = sum(e - s for s, e in speech_intervals)

    return speech_intervals, silence_intervals, total_speech


# ─────────────────────────────────────────────
# 4.  PAUSE ANALYSIS
# ─────────────────────────────────────────────

def analyze_pauses(silence_intervals: list) -> dict:
    """
    Categorise silence intervals into pauses and compute statistics.

    Gaps < PAUSE_MIN_SEC (50ms) are ignored — VAD frame noise.

    Features:
        pause_count       — total pauses           ← model coef 0.649
        pause_long_count  — pauses > 0.5s          ← model coef 0.568
        pause_short_count — pauses 0.15–0.5s       ← kept for reporting
        pause_mean_sec    — mean pause duration     ← kept for reporting
    """
    durations = [e - s for s, e in silence_intervals if (e - s) >= PAUSE_MIN_SEC]

    if not durations:
        return {
            "pause_count":       0,
            "pause_short_count": 0,
            "pause_long_count":  0,
            "pause_mean_sec":    0.0,
        }

    return {
        "pause_count":       len(durations),
        "pause_short_count": sum(1 for d in durations if PAUSE_SHORT <= d < PAUSE_LONG),
        "pause_long_count":  sum(1 for d in durations if d >= PAUSE_LONG),
        "pause_mean_sec":    float(np.mean(durations)),
    }


# ─────────────────────────────────────────────
# 5.  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def extract_reader_features(audio_path: str) -> dict:
    """
    Full feature extraction for one reader's audio file.

    Returns a flat dict of all raw features (not yet z-scored).
    Z-score normalisation is applied separately after the full
    population is extracted, so per-class mean/std can be computed.

    Parameters
    ----------
    audio_path : str
        Path to cleaned .wav file.

    Returns
    -------
    dict with keys:
        total_duration_sec    — total recording length (seconds)
        speech_duration_sec   — time actually speaking (seconds)
        pause_ratio           — fraction of recording that is silence
        speech_segments_count — number of continuous speech bursts
        pause_count           — number of pauses detected
        pause_short_count     — pauses 0.15–0.5s
        pause_long_count      — pauses > 0.5s
        pause_mean_sec        — mean pause duration (seconds)
    """
    feats = {}

    # ── Load audio ────────────────────────────────────────────────────
    y, sr     = load_audio(audio_path)
    total_dur = len(y) / sr
    feats["total_duration_sec"] = round(total_dur, 4)

    # ── Speech/silence segmentation ───────────────────────────────────
    # All features derive from these two interval lists
    speech_segs, silence_segs, speech_sec = get_speech_silence(y, sr)

    feats["speech_duration_sec"]   = round(speech_sec, 4)
    feats["pause_ratio"]           = round(1.0 - (speech_sec / total_dur), 4) if total_dur > 0 else 0.0
    feats["speech_segments_count"] = len(speech_segs)  # fragmentation count

    # ── Pause analysis ────────────────────────────────────────────────
    feats.update(analyze_pauses(silence_segs))

    # ── Round all floats for clean Excel output ───────────────────────
    for k, v in feats.items():
        if isinstance(v, float):
            feats[k] = round(v, 6)

    return feats


# ─────────────────────────────────────────────
# 6.  POST-PROCESSING: CLASS-LEVEL Z-SCORE NORMALISATION
#     Run AFTER all subjects extracted (df is complete)
# ─────────────────────────────────────────────

def add_class_normalised_features(df):
    """
    Add z-score columns for each feature in FEATURES_TO_NORMALISE.

    For each class (group_code), compute mean and std from the
    training population, then:
        {feature}_z = (value - class_mean) / class_std

    Interpretation:
        positive z on pause_count    → more pauses than classmates → bad sign
        negative z on total_duration → faster than classmates      → good sign

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'group_code' column and all feature columns.

    Returns
    -------
    df          : pd.DataFrame with *_z columns added
    class_stats : dict  {class_code: {feature: {mean, std}}}
                  Save to JSON — needed for inference on new subjects.
    """

    class_stats = {}

    for code in df["group_code"].dropna().unique():
        mask = df["group_code"] == code
        class_stats[code] = {}
        for feat in FEATURES_TO_NORMALISE:
            if feat not in df.columns:
                continue
            vals = df.loc[mask, feat].dropna()
            m    = float(vals.mean()) if len(vals) > 1 else 0.0
            s    = float(vals.std())  if len(vals) > 1 else 1.0
            if s == 0:
                s = 1.0  # avoid division by zero for uniform classes
            class_stats[code][feat] = {"mean": m, "std": s}

    # Add z-score columns
    for feat in FEATURES_TO_NORMALISE:
        if feat not in df.columns:
            continue
        z_col = feat + "_z"
        df[z_col] = df.apply(
            lambda row: (
                (row[feat] - class_stats
                    .get(row["group_code"], {})
                    .get(feat, {})
                    .get("mean", 0.0))
                / class_stats
                    .get(row["group_code"], {})
                    .get(feat, {})
                    .get("std", 1.0)
            )
            if row["group_code"] in class_stats else np.nan,
            axis=1,
        ).round(4)

    return df, class_stats


def save_class_stats(class_stats: dict, path: str):
    """Save per-class normalisation stats to JSON for reuse on new subjects."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_stats, f, indent=2, ensure_ascii=False)
    print(f"Class stats saved → {path}")


def load_class_stats(path: str) -> dict:
    """Load previously saved class stats for inference on new subjects."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalise_single_subject(feats: dict,
                              class_stats: dict,
                              group_code: str) -> dict:
    """
    Apply saved class norms to a single new subject's feature dict.

    Used at inference time — no need to recompute population stats.
    The same mean/std from the training population is applied so
    new subjects are compared against the same reference population.
    """
    stats = class_stats.get(group_code, {})
    for feat in FEATURES_TO_NORMALISE:
        if feat not in feats:
            continue
        m = stats.get(feat, {}).get("mean", 0.0)
        s = stats.get(feat, {}).get("std",  1.0)
        feats[feat + "_z"] = round((feats[feat] - m) / s, 4)
    return feats