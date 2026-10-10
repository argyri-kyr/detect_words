"""
=============================================================
GREEK READ-ALOUD CLASSIFIER  —  audio_preprocessing.py
=============================================================

Noise removal and audio cleanup pipeline.
Designed for classroom/office recordings with:
  - Microphone hiss & recording artifacts
  - Constant background hum / AC / room tone
  - Low volume (child far from microphone)
  - Clipping (signal saturation)

Pipeline order (matters):
  1. Load & resample to 16 kHz mono
  2. Clipping repair  (soft-limit saturated samples)
  3. High-pass filter (remove low-freq hum / rumble < 80 Hz)
  4. Stationary noise reduction  (noisereduce — removes hiss/AC hum)
  5. Volume normalisation  (fixes "far from mic" issue)
  6. Save as {sub_id}_visitloud_clean.wav  (original untouched)

Dependencies:
    pip install noisereduce scipy soundfile numpy librosa
"""

import os
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfilt
from pathlib import Path

from globalVars import SAMPLE_RATE


# ─────────────────────────────────────────────
# 1.  LOAD
# ─────────────────────────────────────────────

def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE):
    """Load audio, resample to target_sr, convert to mono float32 [-1, 1]."""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return y.astype(np.float32), sr


# ─────────────────────────────────────────────
# 2.  CLIPPING REPAIR
# ─────────────────────────────────────────────

def repair_clipping(y: np.ndarray, clip_threshold: float = 0.98) -> np.ndarray:
    """
    Soft-limit samples that exceed clip_threshold.

    Clipping happens when the microphone/preamp saturates — the
    waveform is hard-cut at ±1.0 (or close to it), causing harsh
    distortion. We apply a tanh-based soft limiter which smoothly
    compresses the peaks without introducing hard discontinuities.

    clip_threshold : samples above this absolute value are considered clipped.
                     0.98 catches near-saturation without touching normal peaks.
    """
    # Find clipped regions
    clipped_mask = np.abs(y) >= clip_threshold

    if not np.any(clipped_mask):
        return y   # nothing to fix

    clipped_fraction = clipped_mask.mean()

    # Soft limiter via tanh scaling
    # Scale so that clip_threshold maps to tanh input of ~2.0
    scale = 2.0 / clip_threshold
    y_repaired = np.tanh(y * scale) / np.tanh(scale)

    print(f"  [Clipping] {clipped_fraction:.1%} of samples were clipped — repaired.")
    return y_repaired.astype(np.float32)


# ─────────────────────────────────────────────
# 3.  HIGH-PASS FILTER  (remove hum / low-freq rumble)
# ─────────────────────────────────────────────

def highpass_filter(y: np.ndarray, sr: int,
                    cutoff_hz: float = 80.0,
                    order: int = 4) -> np.ndarray:
    """
    Remove frequencies below cutoff_hz using a Butterworth high-pass filter.

    Eliminates:
      - AC electrical hum (50/60 Hz)
      - Low-frequency room rumble
      - Handling noise from microphone cable

    cutoff_hz : 80 Hz is safe for speech (fundamental F0 starts ~80 Hz
                for adult males, ~160 Hz for children — we keep all speech
                but cut the sub-speech rumble).
    order     : filter steepness. 4 is a good balance.
    """
    sos = butter(order, cutoff_hz, btype="high", fs=sr, output="sos")
    y_filtered = sosfilt(sos, y)
    return y_filtered.astype(np.float32)


# ─────────────────────────────────────────────
# 4.  STATIONARY NOISE REDUCTION  (hiss, AC hum, room tone)
# ─────────────────────────────────────────────

def reduce_stationary_noise(y: np.ndarray, sr: int,
                             prop_decrease: float = 0.75,
                             use_silence_as_noise_ref: bool = True) -> np.ndarray:
    """
    Remove stationary background noise using spectral subtraction
    (noisereduce library — no GPU needed, works on CPU).

    How it works:
      - Estimates the noise spectral profile from a reference segment
        (either the first/last 0.5 s of audio, which are usually silence,
         or the full signal in stationary mode)
      - Subtracts that noise profile from the entire signal

    Best for: hiss, AC hum, constant fan noise, room tone.
    NOT suitable for: intermittent noise (coughs, other voices).

    prop_decrease : 0.0 = no reduction, 1.0 = full removal.
                    0.75 is aggressive but preserves speech.
                    Lower to 0.5 if you hear speech artifacts after.

    use_silence_as_noise_ref : if True, use first+last 0.5 s as noise
                                profile (more accurate).
                                If False, noisereduce estimates noise
                                from the full signal (more robust when
                                there is no clear silence at start/end).
    """
    if use_silence_as_noise_ref:
        # Use first 0.5 s + last 0.5 s as noise reference
        # (typically silence before/after the reader starts)
        n_ref = int(0.5 * sr)
        noise_ref = np.concatenate([y[:n_ref], y[-n_ref:]])
        y_denoised = nr.reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_ref,
            prop_decrease=prop_decrease,
            stationary=True,
            n_fft=1024,
            n_jobs=1,
        )
    else:
        y_denoised = nr.reduce_noise(
            y=y,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=True,
            n_fft=1024,
            n_jobs=1,
        )

    return y_denoised.astype(np.float32)


# ─────────────────────────────────────────────
# 5.  VOLUME NORMALISATION  (fix "far from mic" low volume)
# ─────────────────────────────────────────────

def normalise_volume(y: np.ndarray,
                     target_db: float = -20.0) -> np.ndarray:
    """
    Normalise audio loudness to a target RMS level in dBFS.

    Fixes the common problem of a child sitting far from the
    microphone — the recording is very quiet and VAD / feature
    extraction performs poorly.

    target_db : target RMS level in dBFS.
                -20 dBFS is a comfortable speech level.
                -23 dBFS matches EBU R128 broadcast standard.

    Does NOT clip: if normalisation would push peaks above 0 dBFS,
    we scale to peak = 0.99 instead (peak normalisation fallback).
    """
    rms = np.sqrt(np.mean(y ** 2))

    if rms < 1e-9:
        # Essentially silent — nothing to normalise
        print("  [Volume] Warning: near-silent audio, skipping normalisation.")
        return y

    # Current RMS in dBFS
    current_db = 20 * np.log10(rms)
    gain_db    = target_db - current_db
    gain_linear = 10 ** (gain_db / 20.0)

    y_normalised = y * gain_linear

    # Safety: if any sample exceeds ±1.0 after gain, peak-normalise
    peak = np.max(np.abs(y_normalised))
    if peak > 0.99:
        y_normalised = y_normalised * (0.99 / peak)
        print(f"  [Volume] Peak limiting applied after gain of {gain_db:+.1f} dB.")
    else:
        print(f"  [Volume] Gain applied: {gain_db:+.1f} dB  "
              f"(was {current_db:.1f} dBFS → target {target_db:.1f} dBFS)")

    return y_normalised.astype(np.float32)


# ─────────────────────────────────────────────
# 6.  AUDIO QUALITY ASSESSMENT  (before & after)
# ─────────────────────────────────────────────

def assess_audio_quality(y: np.ndarray, sr: int, label: str = "") -> dict:
    """
    Compute quick quality metrics to compare before/after cleaning.
    Printed as a report; also returned as a dict for logging.

    Metrics:
      rms_db        : overall loudness
      peak_db       : peak level (clipping risk if near 0 dBFS)
      snr_estimate  : rough SNR estimate using librosa VAD
                      (speech RMS vs silence RMS ratio)
      clipping_pct  : percentage of samples at/near saturation
    """
    rms    = np.sqrt(np.mean(y ** 2))
    peak   = np.max(np.abs(y))
    rms_db = 20 * np.log10(rms  + 1e-9)
    peak_db= 20 * np.log10(peak + 1e-9)
    clipping_pct = float(np.mean(np.abs(y) >= 0.98) * 100)

    # SNR estimate: speech segments vs silence segments
    non_silent = librosa.effects.split(y, top_db=25)
    if len(non_silent) > 0:
        speech_frames  = np.concatenate([y[s:e] for s, e in non_silent])
        silence_mask   = np.ones(len(y), dtype=bool)
        for s, e in non_silent:
            silence_mask[s:e] = False
        silence_frames = y[silence_mask]

        speech_rms  = np.sqrt(np.mean(speech_frames  ** 2)) if len(speech_frames)  else 1e-9
        silence_rms = np.sqrt(np.mean(silence_frames ** 2)) if len(silence_frames) else 1e-9
        snr_db = 20 * np.log10(speech_rms / (silence_rms + 1e-9))
    else:
        snr_db = 0.0

    result = {
        "rms_db":        round(rms_db,        2),
        "peak_db":       round(peak_db,       2),
        "snr_estimate":  round(snr_db,        2),
        "clipping_pct":  round(clipping_pct,  4),
    }

    tag = f"[{label}]" if label else ""
    print(f"  {tag} RMS={rms_db:.1f} dBFS  |  Peak={peak_db:.1f} dBFS  |  "
          f"SNR≈{snr_db:.1f} dB  |  Clipping={clipping_pct:.2f}%")

    return result


# ─────────────────────────────────────────────
# 7.  FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess_audio(audio_path: str,
                     output_path: str = None,
                     prop_decrease: float = 0.75,
                     target_db: float = -20.0,
                     highpass_cutoff: float = 80.0,
                     verbose: bool = True) -> tuple:
    """
    Full noise removal pipeline for one audio file.

    Steps:
      1. Load & resample
      2. Clipping repair
      3. High-pass filter (hum / rumble)
      4. Stationary noise reduction (hiss / AC)
      5. Volume normalisation
      6. Save cleaned file

    Parameters
    ----------
    audio_path     : path to input .wav file
    output_path    : where to save the clean file.
                     If None, saves as {stem}_clean.wav in same folder.
    prop_decrease  : noise reduction aggressiveness (0.0–1.0).
                     0.75 recommended. Lower if speech sounds robotic.
    target_db      : target loudness after normalisation (-20 dBFS default).
    highpass_cutoff: high-pass filter cutoff in Hz (80 recommended).
    verbose        : print quality report before/after.

    Returns
    -------
    y_clean   : np.ndarray  cleaned waveform
    sr        : int         sample rate (always SAMPLE_RATE)
    out_path  : str         path to saved clean file
    qa_before : dict        quality metrics before cleaning
    qa_after  : dict        quality metrics after cleaning
    """
    # ── Auto output path ──────────────────────────────────────────────
    if output_path is None:
        p = Path(audio_path)
        output_path = str(p.parent / f"{p.stem}_clean{p.suffix}")

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Preprocessing: {Path(audio_path).name}")
        print(f"{'─'*55}")

    # ── Step 1: Load ──────────────────────────────────────────────────
    y, sr = load_audio(audio_path)
    if verbose:
        print(f"  Loaded: {len(y)/sr:.1f} s  @ {sr} Hz")

    # ── Quality BEFORE ────────────────────────────────────────────────
    qa_before = assess_audio_quality(y, sr, label="BEFORE") if verbose else {}

    # ── Step 2: Clipping repair ───────────────────────────────────────
    y = repair_clipping(y, clip_threshold=0.98)

    # ── Step 3: High-pass filter ──────────────────────────────────────
    y = highpass_filter(y, sr, cutoff_hz=highpass_cutoff)

    # ── Step 4: Stationary noise reduction ───────────────────────────
    y = reduce_stationary_noise(y, sr, prop_decrease=prop_decrease)

    # ── Step 5: Volume normalisation ─────────────────────────────────
    y = normalise_volume(y, target_db=target_db)

    # ── Quality AFTER ────────────────────────────────────────────────
    qa_after = assess_audio_quality(y, sr, label="AFTER ") if verbose else {}

    # ── Step 6: Save ─────────────────────────────────────────────────
    sf.write(output_path, y, sr, subtype="PCM_16")
    if verbose:
        print(f"  ✅ Saved clean audio → {Path(output_path).name}")

    return y, sr, output_path, qa_before, qa_after


# ─────────────────────────────────────────────
# 8.  BATCH HELPER  (called from main.py)
# ─────────────────────────────────────────────

def get_clean_audio_path(audio_path: str) -> str:
    """
    Return the path where the clean version of this audio should live.
    Convention: {stem}_clean.wav  in the same folder as the original.
    """
    p = Path(audio_path)
    return str(p.parent / f"{p.stem}_clean{p.suffix}")


def ensure_clean_audio(audio_path: str,
                       prop_decrease: float = 0.75,
                       target_db: float = -20.0,
                       force_reprocess: bool = False,
                       verbose: bool = True) -> str:
    """
    Return path to cleaned audio, creating it if it doesn't exist yet.

    If the _clean.wav file already exists, skip reprocessing
    (unless force_reprocess=True).

    This is the function to call from main.py — it integrates
    cleanly into the existing subject loop.

    Parameters
    ----------
    audio_path      : path to raw .wav (already extracted from .avi)
    prop_decrease   : noise reduction strength
    target_db       : normalisation target
    force_reprocess : re-run even if _clean.wav already exists
    verbose         : print quality report

    Returns
    -------
    clean_path : str  path to the cleaned .wav file
    """
    clean_path = get_clean_audio_path(audio_path)

    if os.path.exists(clean_path) and not force_reprocess:
        if verbose:
            print(f"  [Cache] Clean audio exists: {Path(clean_path).name}")
        return clean_path

    _, _, clean_path, _, _ = preprocess_audio(
        audio_path,
        output_path=clean_path,
        prop_decrease=prop_decrease,
        target_db=target_db,
        verbose=verbose,
    )

    return clean_path
