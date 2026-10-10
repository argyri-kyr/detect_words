

folder1 = '/_Iro_analysis_tobii7_HD/'

# folder2 = 'individual_decodewordspseudo_apokor/'
# folder2 = 'individual_decodewordspseudo_apokor_without_prompt/'
# folder2 = 'individual_decodewordspseudo_larisa_cherso2'
folder2 = 'individual/'


# ─────────────────────────────────────────────
# PREPROCESSING SETTINGS
# ─────────────────────────────────────────────

# Noise reduction aggressiveness 0.0–1.0
# 0.75 : strong, good for hissy/humming recordings
# 0.50 : moderate, use if cleaned speech sounds robotic/distorted
NOISE_PROP_DECREASE = 0.75

# Target loudness after normalisation (dBFS)
# -20 is comfortable for speech; lower = quieter output
NORMALISE_TARGET_DB = -20.0

# Set True to re-clean audio even if _clean.wav already exists
FORCE_REPROCESS = False

SAMPLE_RATE = 16000   # all audio standardised to 16 kHz


# ─────────────────────────────────────────────
# KNOWN CLASS CODES  (ordered by grade)
# ─────────────────────────────────────────────
# be  = Β' Δημοτικού  (2nd elementary)
# ge  = Γ' Δημοτικού  (3rd elementary)
# de  = Δ' Δημοτικού  (4th elementary)
# ee  = Ε' Δημοτικού  (5th elementary)
# ste = ΣΤ' Δημοτικού (6th elementary)
# ag  = Α' Γυμνασίου  (1st middle school)
# bg  = Β' Γυμνασίου  (2nd middle school)
# gg  = Γ' Γυμνασίου  (3rd middle school)

KNOWN_CLASSES = ['be', 'ge', 'de', 'ee', 'ste', 'ag', 'bg', 'gg']

VAD_TOP_DB    = 25     # dB below peak to treat as silence
                       # raise to 30-35 for noisy recordings
PAUSE_MIN_SEC = 0.05   # gaps below this = VAD frame noise, not real pauses
PAUSE_SHORT   = 0.15   # seconds — lower bound for a real hesitation pause
PAUSE_LONG    = 0.50   # seconds — above this: strong sign of decoding difficulty

# Only features with demonstrated discriminatory power.
# Order matches model feature importance (descending).
FEATURES_TO_NORMALISE = [
    "total_duration_sec",     # model coef 0.794 — strongest predictor
    "speech_duration_sec",    # model coef 0.652
    "pause_count",            # model coef 0.649
    "speech_segments_count",  # model coef 0.576
    "pause_long_count",       # model coef 0.568
    "pause_ratio",            # selected 4/5 folds
    "pause_mean_sec",         # for reporting
    "pause_short_count",      # for reporting
    
    "pa_word_dur_mean",
    "pa_gap_mean",
    "pa_long_gap_count",
    "pa_very_long_gap_count",
    "pa_long_gap_ratio",
    "pa_words_per_minute",    
    "pa_articulation_rate",   
]

# Key z-score features to inspect — edit if your column names differ
KEY_Z_FEATURES = [
    # Pipeline B — VAD features
    "total_duration_sec_z",
    "speech_duration_sec_z",
    "pause_count_z",
    "speech_segments_count_z",
    "pause_long_count_z",
    "pause_ratio_z",
    "pause_mean_sec_z",
    "pause_short_count_z",
    # Pipeline A — word timing features (redundant speed measures removed)
    "pa_word_dur_mean_z",
    "pa_gap_mean_z",
    "pa_long_gap_count_z",
    "pa_very_long_gap_count_z",
    "pa_long_gap_ratio_z",
    "pa_words_per_minute_z",  
    "pa_articulation_rate_z",
]

# ─────────────────────────────────────────────
# THRESHOLDS 
# ─────────────────────────────────────────────

WORD_DUR_MIN   = 0.05   # seconds — below this: alignment noise, not a real word
WORD_DUR_MAX   = 2.00   # seconds — above this: WhisperX stretched boundary error
GAP_MIN        = 0.0    # seconds — negative gaps can occur with alignment errors
GAP_LONG       = 1.00   # seconds — hesitation at word boundary
GAP_VERY_LONG  = 2.00   # seconds — strong decoding difficulty signal
WPM_MIN        = 10     # sanity check — below this: something went wrong
WPM_MAX        = 250    # sanity check — above this: something went wrong


# ─────────────────────────────────────────────────────────────────────────────
# Feature descriptions — plain Greek-school-reading context
# ─────────────────────────────────────────────────────────────────────────────

# For each feature:
#   "label"     : human-readable name
#   "unit"      : unit of measurement
#   "good_dir"  : which direction is "good" — "low" or "high"
#   "explain"   : what this feature means in plain words

FEATURE_META = {
    "total_duration_sec": {
        "label":    "Total reading time",
        "unit":     "seconds",
        "good_dir": "low",
        "explain":  "How long the subject took to read the full text from start to finish.",
    },
    "speech_duration_sec": {
        "label":    "Active speech time",
        "unit":     "seconds",
        "good_dir": "low",
        "explain":  "Total time the subject was actually speaking (pauses excluded).",
    },
    "pause_count": {
        "label":    "Number of pauses",
        "unit":     "pauses",
        "good_dir": "low",
        "explain":  "How many times the subject stopped between words.",
    },
    "pause_long_count": {
        "label":    "Number of long pauses",
        "unit":     "pauses",
        "good_dir": "low",
        "explain":  "How many pauses lasted longer than 0.5 seconds — a strong sign of decoding difficulty.",
    },
    "speech_segments_count": {
        "label":    "Speech fragments",
        "unit":     "segments",
        "good_dir": "low",
        "explain":  "How many separate bursts of speech were detected. "
                    "High values indicate fragmented, syllable-by-syllable reading.",
    },
    "pa_word_dur_mean": {
        "label":    "Average word duration (WD)",
        "unit":     "seconds",
        "good_dir": "low",
        "explain":  "Average time spent pronouncing each detected word. "
                    "Longer durations indicate slower, more labored word production.",
    },
    "pa_gap_mean": {
        "label":    "Average gap between words (WD)",
        "unit":     "seconds",
        "good_dir": "low",
        "explain":  "Average silence between consecutive words. "
                    "Larger gaps indicate hesitation at word boundaries.",
    },
    "pa_long_gap_count": {
        "label":    "Number of long word gaps (WD)",
        "unit":     "gaps",
        "good_dir": "low",
        "explain":  "How many gaps between words lasted longer than 1 second. "
                    "Each marks a point where the child struggled to retrieve the next word.",
    },
    "pa_very_long_gap_count": {
        "label":    "Number of very long word gaps (WD)",
        "unit":     "gaps",
        "good_dir": "low",
        "explain":  "How many gaps between words lasted longer than 2 seconds. "
                    "A strong indicator of significant decoding difficulty.",
    },
    "pa_long_gap_ratio": {
        "label":    "Proportion of long word gaps (WD)",
        "unit":     "",
        "good_dir": "low",
        "explain":  "Fraction of all inter-word gaps that lasted longer than 1 second. "
                    "High values mean the child hesitated frequently throughout the reading.",
    },
    "pa_words_per_minute": {
        "label":    "Words per minute (WD)",
        "unit":     "wpm",
        "good_dir": "high",
        "explain":  "Reading speed in words per minute based on detected words.",
    },
    "pa_articulation_rate": {
        "label":    "Articulation rate (WD)",
        "unit":     "words/sec",
        "good_dir": "high",
        "explain":  "Number of detected words per second of total recording time.",
    },
    "pause_ratio": {
        "label":    "Pause ratio",
        "unit":     "",
        "good_dir": "low",
        "explain":  "Fraction of the total recording that was silence. "
                    "Higher values mean the child spent more time pausing than reading.",
    },
    "pause_mean_sec": {
        "label":    "Mean pause duration",
        "unit":     "seconds",
        "good_dir": "low",
        "explain":  "Average length of each pause. "
                    "Longer average pauses indicate more severe hesitation between words.",
    },
    "pause_short_count": {
        "label":    "Number of short pauses",
        "unit":     "pauses",
        "good_dir": "low",
        "explain":  "How many pauses lasted between 0.15 and 0.5 seconds. "
                    "Frequent short pauses indicate choppy, effortful reading.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────

