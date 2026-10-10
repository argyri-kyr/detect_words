

"""
feature_extraction_pipeline_a.py
─────────────────────────────────────────────────────────────────────────────
Extract word-level timing features from WhisperX detected_words XLSX files.

WHERE TO CALL THIS:
    In your main pipeline (pipeline_b_main.py or wherever Phase 1 runs),
    after extract_reader_features() for each subject:

        # Pipeline B features (VAD-based)
        feats_b = extract_reader_features(clean_audio_path)

        # Pipeline A features (word timestamps)
        xlsx_path = find_pipeline_a_xlsx(sub_path, sub_id)   # see helper below
        if xlsx_path:
            feats_a = extract_pipeline_a_features(xlsx_path, expected_word_count=181)
            feats_b.update(feats_a)   # merge into one flat dict

        all_features.append(feats_b)

    The merged dict is then written to signal_reading_features.xlsx as usual.
    add the new feature names to FEATURES_TO_NORMALISE in feature_extraction_v4.py.

NOISE HANDLING:
    WhisperX word timestamps have three known error types:
      1. Wrong word recognized       → doesn't affect timing features
      2. Child mispronounced word     → doesn't affect timing features
      3. Extra/missing small words    → handled via coverage_ratio
      4. Stretched word duration      → filtered by WORD_DUR_MAX threshold

    Inter-word gaps are the most reliable feature — the gap between
    end[i] and start[i+1] reflects hesitation at word boundaries
    regardless of whether the word itself was recognized correctly.
"""

import os
import numpy as np
import pandas as pd

from globalVars import WORD_DUR_MIN, WORD_DUR_MAX, GAP_MIN, GAP_LONG, GAP_VERY_LONG, WPM_MIN, WPM_MAX


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def extract_reading_features_DetectedWords(
    xlsx_path: str,
    expected_word_count: int,
) -> dict:
    """
    Extract word-level timing features from a WhisperX detected_words XLSX file.

    Parameters
    ----------
    xlsx_path           : path to detected_words_v3_{id}.xlsx
    expected_word_count : number of words in the actual text
                          used to compute coverage_ratio

    Returns
    -------
    dict of raw features (not yet z-scored), prefixed with 'pa_'
    to distinguish from Pipeline B features.

    Features:
        pa_coverage_ratio       — detected_words / expected_word_count
        pa_words_per_minute     — reading speed (words per minute)
        pa_articulation_rate    — detected_words / total_speech_sec
        pa_word_dur_mean        — mean word duration (outliers filtered)
        pa_word_dur_std         — std of word duration
        pa_gap_mean             — mean inter-word gap
        pa_gap_std              — std of inter-word gaps
        pa_long_gap_count       — gaps > 1.0s  (hesitations)
        pa_very_long_gap_count  — gaps > 2.0s  (strong difficulty signal)
        pa_long_gap_ratio       — long_gap_count / total_gaps
        pa_failed               — 1 if extraction failed, 0 otherwise
    """
    failed_result = {
        "pa_coverage_ratio":      np.nan,
        "pa_words_per_minute":    np.nan,
        "pa_articulation_rate":   np.nan,
        "pa_word_dur_mean":       np.nan,
        "pa_word_dur_std":        np.nan,
        "pa_gap_mean":            np.nan,
        "pa_gap_std":             np.nan,
        "pa_long_gap_count":      np.nan,
        "pa_very_long_gap_count": np.nan,
        "pa_long_gap_ratio":      np.nan,
        "pa_failed":              1,
    }

    # ── Load XLSX ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"  [WARN] Pipeline A: could not read {xlsx_path}: {e}")
        return failed_result

    required_cols = {"start_s", "end_s"}
    if not required_cols.issubset(df.columns):
        print(f"  [WARN] Pipeline A: missing columns in {xlsx_path}")
        return failed_result

    if len(df) < 5:
        print(f"  [WARN] Pipeline A: too few words detected ({len(df)}) in {xlsx_path}")
        return failed_result

    df = df.sort_values("start_s").reset_index(drop=True)

    starts = df["start_s"].values
    ends   = df["end_s"].values
    n_detected = len(df)

    # ── Coverage ──────────────────────────────────────────────────────────────
    coverage_ratio = round(n_detected / expected_word_count, 4) if expected_word_count > 0 else np.nan

    # ── Reading speed ─────────────────────────────────────────────────────────
    total_duration = ends[-1] - starts[0]   # first word start → last word end
    total_min      = total_duration / 60.0
    wpm = round(n_detected / total_min, 2) if total_min > 0 else np.nan

    # Sanity check
    if wpm is not np.nan and not (WPM_MIN <= wpm <= WPM_MAX):
        print(f"  [WARN] Pipeline A: implausible WPM={wpm:.1f} in {xlsx_path}")
        wpm = np.nan

    # Articulation rate: words per second of actual speech
    # Use total_duration as proxy (we don't have VAD here, Pipeline B has it)
    articulation_rate = round(n_detected / total_duration, 4) if total_duration > 0 else np.nan

    # ── Word durations (filtered) ─────────────────────────────────────────────
    # WhisperX sometimes assigns very long durations to words when it's
    # struggling with a segment — filter these out before computing stats.
    word_durs = ends - starts
    valid_durs = word_durs[(word_durs >= WORD_DUR_MIN) & (word_durs <= WORD_DUR_MAX)]

    if len(valid_durs) >= 3:
        word_dur_mean = round(float(np.mean(valid_durs)), 4)
        word_dur_std  = round(float(np.std(valid_durs)),  4)
    else:
        word_dur_mean = np.nan
        word_dur_std  = np.nan

    # ── Inter-word gaps ───────────────────────────────────────────────────────
    # Gap between end of word[i] and start of word[i+1].
    # This is the most reliable feature — reflects hesitation at word
    # boundaries regardless of recognition accuracy.
    gaps = starts[1:] - ends[:-1]
    gaps = gaps[gaps >= GAP_MIN]   # remove negative gaps (alignment artifacts)

    if len(gaps) >= 3:
        gap_mean = round(float(np.mean(gaps)), 4)
        gap_std  = round(float(np.std(gaps)),  4)

        long_gap_count      = int(np.sum(gaps >= GAP_LONG))
        very_long_gap_count = int(np.sum(gaps >= GAP_VERY_LONG))
        long_gap_ratio      = round(long_gap_count / len(gaps), 4)
    else:
        gap_mean            = np.nan
        gap_std             = np.nan
        long_gap_count      = np.nan
        very_long_gap_count = np.nan
        long_gap_ratio      = np.nan

    return {
        "pa_coverage_ratio":      coverage_ratio,
        "pa_words_per_minute":    wpm,
        "pa_articulation_rate":   articulation_rate,
        "pa_word_dur_mean":       word_dur_mean,
        "pa_word_dur_std":        word_dur_std,
        "pa_gap_mean":            gap_mean,
        "pa_gap_std":             gap_std,
        "pa_long_gap_count":      long_gap_count,
        "pa_very_long_gap_count": very_long_gap_count,
        "pa_long_gap_ratio":      long_gap_ratio,
        "pa_failed":              0,
    }


# ─────────────────────────────────────────────
# HELPER — find the XLSX for a subject
# ─────────────────────────────────────────────

def find_detected_words_xlsx(sub_path, sub_id):
    """
    Look for the detected_words XLSX inside the subject folder.

    Tries common naming patterns:
        detected_words_v3_{id}.xlsx
        detected_words_{id}.xlsx
        words_{id}.xlsx

    Returns the path if found, None otherwise.
    """
    patterns = [
        f"detected_words_v3_{sub_id}.xlsx"
        # f"detected_words_timestamps_{sub_id}.csv",
        # f"detected_words_{sub_id}.csv",
        # f"words_{sub_id}.csv",
    ]
    for fname in patterns:
        full = os.path.join(sub_path, fname)
        if os.path.exists(full):
            return full

    # Also scan folder in case filename doesn't match exactly
    try:
        for entry in os.listdir(sub_path):
            if entry.endswith(".xlsx") and "word" in entry.lower() and 'v3' in entry.lower():
                return os.path.join(sub_path, entry)
    except FileNotFoundError:
        pass

    return None