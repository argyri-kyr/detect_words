
from pathlib import Path


from globalVars import FORCE_REPROCESS, NOISE_PROP_DECREASE, NORMALISE_TARGET_DB, FORCE_REPROCESS
# from audio_preprocessing_v4 import ensure_clean_audio
from final_voice_studio_cleaner import clean_wav_voice_only_studio



# ── Noise removal → saves _clean.wav ─────────────
#
# Pipeline applied to each recording (final_voice_studio_cleaner):
#   1. Greek phrase/filler removal via Whisper word timestamps
#   2. Clipping soft repair
#   3. High-pass filter (rumble < 120 Hz)
#   4. AI denoise via DeepFilterNet (falls back to noisereduce)
#   5. Tail expander / mild dereverb
#   6. Speech EQ + de-esser
#   7. Breath / mouth-click reduction
#   8. Voice-only gating (attenuate + shorten non-speech)
#   9. Compression / levelling
#  10. LUFS loudness normalisation + final limiter
#
# Output: {stem}_clean.wav saved alongside the raw file.
# Already-cleaned files are reused (skipped) automatically.
# Set FORCE_REPROCESS = True in globalVars to redo all cleaning.
#
def clean_audio_pipeline(sub_id, raw_audio_path):
    p = Path(raw_audio_path)
    clean_audio_path = str(p.parent / f"{p.stem}_clean{p.suffix}")

    # ── Cache: skip if already cleaned ───────────────────────────────
    if Path(clean_audio_path).exists() and not FORCE_REPROCESS:
        print(f"  [Cache] Clean audio exists: {Path(clean_audio_path).name}")
        return clean_audio_path

    try:
        # clean_audio_path = ensure_clean_audio(raw_audio_path, prop_decrease=NOISE_PROP_DECREASE, target_db=NORMALISE_TARGET_DB, force_reprocess=FORCE_REPROCESS,verbose=True)
        clean_audio_path = clean_wav_voice_only_studio(raw_audio_path, clean_audio_path)
    except Exception as e:
        print(f"  [WARN] Audio cleaning failed for {sub_id}: {e}")
        print(f"  [WARN] Falling back to raw audio.")
        clean_audio_path = raw_audio_path   # use raw if cleaning fails

    return clean_audio_path