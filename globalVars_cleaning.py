

"""
All configuration for the audio cleaning pipeline
(final_voice_studio_cleaner).

Edit this file to tune cleaning behaviour — nothing else needs to change.
"""

# ─────────────────────────────────────────────
# WHISPER SETTINGS  (phrase/filler detection)
# ─────────────────────────────────────────────

# Use "large-v3" for best Greek accuracy; "medium" is faster.
WHISPER_MODEL    = "medium"
WHISPER_LANGUAGE = "el"
DEVICE           = "cpu"    # "cpu" or "cuda"
COMPUTE_TYPE     = "int8"   # CPU: "int8" | CUDA: "float16"

# ─────────────────────────────────────────────
# PHRASE / FILLER REMOVAL
# ─────────────────────────────────────────────

# Main phrases to delete. Accents are ignored automatically.
# Add aliases if Whisper hears something slightly different.
PHRASE_GROUPS = {
    "δεν καταλαβαινω": [
        "δεν καταλαβαινω",
        "δε καταλαβαινω",
        "δεν το καταλαβαινω",
        "δεν καταλαβαινεις",
        "δε καταλαβαινεις",
    ],
    "συνεχισε": [
        "συνεχισε",
        "συνεχιστε",
    ],
    "προχωρας": [
        "προχωρας",
        "προχωρα",
    ],
    "προχωρας στην": [
        "προχωρας στην",
        "προχωρα στην",
    ],
    "επομενη": [
        "επομενη",
    ],
    "στην επομενη": [
        "στην επομενη",
    ],
    "ωραια": [
        "ωραια",
    ],
    "καλα": [
        "καλα",
    ],
    "καταλαβαινω": [
        "καταλαβαινω",
        "καταλαβαινεις",
    ],
    "μπραβο": [
        "μπραβο",
    ],
    "μπραβο σου": [
        "μπραβο σου",
    ],
    "παμε": [
        "παμε",
    ],
    "προχωρας στην επομενη": [
        "προχωρας στην επομενη",
        "προχωρα στην επομενη",
        "προχωρα στη επομενη",
        "προχωρας στη επομενη",
    ],
    "πειραζει": [
        "πειραζει",
    ],
}

# Set True to also remove filler words/phrases.
# Keep conservative for Greek read-aloud audio.
REMOVE_FILLERS = True
FILLER_GROUPS = {
    "εε":       ["εε", "ε", "εεε", "εμμ", "μμ", "μμμ"],
    "λοιπον":   ["λοιπον", "λοιπόν"],
    "δηλαδη":   ["δηλαδη", "δηλαδή"],
    "ας πουμε": ["ας πουμε", "ας πούμε"],
}

# Optional manual cuts in seconds, used only when Whisper misses something.
# Example: MANUAL_CUT_INTERVALS_SEC = [(12.40, 13.20), (45.10, 46.00)]
MANUAL_CUT_INTERVALS_SEC = []

# Extra milliseconds removed before/after detected phrases and fillers.
PHRASE_PADDING_MS = 250
FILLER_PADDING_MS = 120
CROSSFADE_MS      = 20

# Robust phrase matching.
ENABLE_FUZZY_MATCHING       = True
FUZZY_THRESHOLD_MULTI_WORD  = 0.80
FUZZY_THRESHOLD_SINGLE_WORD = 0.90

# Print transcript to help tune PHRASE_GROUPS / FILLER_GROUPS.
PRINT_TRANSCRIPT      = True
PRINT_WORD_TIMESTAMPS = False

# ─────────────────────────────────────────────
# STUDIO ENHANCEMENT
# ─────────────────────────────────────────────

ENABLE_STUDIO_ENHANCEMENT = True
USE_AI_DENOISE            = True    # tries DeepFilterNet CLI first if installed
FALLBACK_NOISE_REDUCTION  = 0.45   # 0.45 gentle, 0.65 medium, 0.80 aggressive
TARGET_LUFS               = -18.5  # -16 podcast-style loud; -18/-20 gentler

# ─────────────────────────────────────────────
# VOICE-ONLY / NON-SPEECH HANDLING
# ─────────────────────────────────────────────

# Options:
#   "attenuate"             = keep timing, make non-speech very quiet
#   "shorten"               = shorten long silences, keep short natural gaps
#   "attenuate_and_shorten" = recommended voice-only result
#   "off"                   = no voice-only gating/shortening
NON_SPEECH_MODE           = "attenuate_and_shorten"
VOICE_TOP_DB              = 32      # higher keeps more quiet material
VOICE_PADDING_MS          = 200
NON_SPEECH_ATTENUATION_DB = -45.0
MAX_SILENCE_KEEP_MS       = 200
MIN_SPEECH_SEGMENT_MS     = 60

# ─────────────────────────────────────────────
# BREATH / MOUTH-CLICK CLEANUP
# ─────────────────────────────────────────────

REDUCE_HARSH_BREATHS   = True
BREATH_REDUCTION_DB    = -18.0
BREATH_RMS_MIN_DB      = -55.0
BREATH_RMS_MAX_DB      = -28.0
BREATH_CENTROID_MIN_HZ = 2800.0
BREATH_ZCR_MIN         = 0.08

LIGHT_DECLICK    = True
DECLICK_STRENGTH = 0.65  # 0.0 off, 1.0 stronger

# ─────────────────────────────────────────────
# MILD DEREVERB / TAIL SUPPRESSION
# ─────────────────────────────────────────────

# Not true neural dereverb — reduces room tails between phrases after denoise.
TAIL_EXPANDER     = True
TAIL_THRESHOLD_DB = -38.0
TAIL_REDUCTION_DB = -10.0