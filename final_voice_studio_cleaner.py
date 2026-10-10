"""
final_voice_only_studio_cleaner_16khz_no_args.py

Greek WAV cleaner + voice-only studio-style enhancer.

What it does:
1) Transcribes Greek audio with word timestamps using faster-whisper.
2) Removes configured phrases, e.g. "δεν καταλαβαινω" and "συνεχισε".
3) Optionally removes filler words/phrases.
4) Applies voice-only cleanup:
   - suppresses non-speech sections
   - shortens long dead-air/silence
   - reduces harsh breath-like noisy frames
   - light declicking for mouth clicks
5) Applies studio-style enhancement:
   - clipping soft repair
   - high-pass rumble removal
   - optional DeepFilterNet AI denoise if installed
   - noisereduce fallback denoise
   - mild dereverb/tail suppression via expansion/gating
   - speech EQ
   - de-essing
   - compression/leveling
   - LUFS loudness normalization
   - final limiter
6) Saves a 16 kHz mono PCM WAV.

Install minimum:
    pip install faster-whisper numpy scipy soundfile librosa noisereduce pyloudnorm

Optional, stronger AI denoise/enhancement:
    pip install torch torchaudio deepfilternet

Usage:
    1) Edit INPUT_WAV and OUTPUT_WAV below.
    2) Run this file with no command-line arguments:

       python final_voice_only_studio_cleaner_16khz_no_args.py

Notes:
    - This is an open-source approximation of "studio sound", not the proprietary
      Cleanvoice model.
    - "Voice only" processing is aggressive by nature. If the output sounds choppy,
      reduce STRICTNESS settings near the top of this file.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import butter, lfilter, medfilt, sosfiltfilt

try:
    from faster_whisper import WhisperModel
except Exception as exc:  # pragma: no cover
    WhisperModel = None
    _WHISPER_IMPORT_ERROR = exc

try:
    import noisereduce as nr
except Exception:  # pragma: no cover
    nr = None

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover
    pyln = None


# =====================================================================
# SETTINGS — imported from globalVars / globalVars_cleaning
# =====================================================================

from globalVars import SAMPLE_RATE as OUTPUT_SAMPLE_RATE
from globalVars_cleaning import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    DEVICE,
    COMPUTE_TYPE,
    PHRASE_GROUPS,
    REMOVE_FILLERS,
    FILLER_GROUPS,
    MANUAL_CUT_INTERVALS_SEC,
    PHRASE_PADDING_MS,
    FILLER_PADDING_MS,
    CROSSFADE_MS,
    ENABLE_FUZZY_MATCHING,
    FUZZY_THRESHOLD_MULTI_WORD,
    FUZZY_THRESHOLD_SINGLE_WORD,
    PRINT_TRANSCRIPT,
    PRINT_WORD_TIMESTAMPS,
    ENABLE_STUDIO_ENHANCEMENT,
    USE_AI_DENOISE,
    FALLBACK_NOISE_REDUCTION,
    TARGET_LUFS,
    NON_SPEECH_MODE,
    VOICE_TOP_DB,
    VOICE_PADDING_MS,
    NON_SPEECH_ATTENUATION_DB,
    MAX_SILENCE_KEEP_MS,
    MIN_SPEECH_SEGMENT_MS,
    REDUCE_HARSH_BREATHS,
    BREATH_REDUCTION_DB,
    BREATH_RMS_MIN_DB,
    BREATH_RMS_MAX_DB,
    BREATH_CENTROID_MIN_HZ,
    BREATH_ZCR_MIN,
    LIGHT_DECLICK,
    DECLICK_STRENGTH,
    TAIL_EXPANDER,
    TAIL_THRESHOLD_DB,
    TAIL_REDUCTION_DB,
)

# Optional intermediate files. Use None to disable.
SAVE_CUT_ONLY_WAV = None
SAVE_BEFORE_FINAL_VOICE_ONLY_WAV = None



# =====================================================================
# Utility functions
# =====================================================================

@dataclass
class Interval:
    start_ms: int
    end_ms: int
    label: str


def safe_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(x: np.ndarray | float, eps: float = 1e-9) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(x), eps))


def load_audio_mono(audio_path: str, target_sr: int = OUTPUT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return safe_float32(y), target_sr


def save_wav(path: str, y: np.ndarray, sr: int) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, safe_float32(y), sr, subtype="PCM_16")
    return path


# =====================================================================
# Greek text normalization and robust phrase matching
# =====================================================================

def normalize_greek(text: str) -> str:
    """Lowercase, remove Greek accents, normalize final sigma, remove punctuation."""
    text = text.lower().strip().replace("ς", "σ")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\sα-ω]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _as_phrase_groups(phrases) -> dict[str, list[str]]:
    if isinstance(phrases, dict):
        return {str(label): [str(x) for x in aliases] for label, aliases in phrases.items()}
    return {str(p): [str(p)] for p in phrases}


def transcript_from_words(words: Sequence[dict]) -> str:
    return " ".join(str(w["text"]).strip() for w in words if str(w.get("text", "")).strip())


def transcribe_words(
    audio_path: str,
    model_size: str = WHISPER_MODEL,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
    language: str = WHISPER_LANGUAGE,
    beam_size: int = 5,
) -> list[dict]:
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Install it with:\n"
            "    pip install faster-whisper"
        ) from _WHISPER_IMPORT_ERROR

    print(f"[Whisper] Loading model={model_size}, device={device}, compute_type={compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("[Whisper] Transcribing with word timestamps...")
    segments, _info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        vad_filter=False,
    )

    words: list[dict] = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            if word.start is None or word.end is None:
                continue
            text = str(word.word).strip()
            if text:
                words.append({"text": text, "start": float(word.start), "end": float(word.end)})

    print(f"[Whisper] Detected {len(words)} words.")
    return words


def merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x.start_ms, x.end_ms))
    merged = [ordered[0]]
    for cur in ordered[1:]:
        prev = merged[-1]
        if cur.start_ms <= prev.end_ms:
            labels = prev.label if cur.label in prev.label else f"{prev.label}; {cur.label}"
            merged[-1] = Interval(prev.start_ms, max(prev.end_ms, cur.end_ms), labels)
        else:
            merged.append(cur)
    return merged


def find_phrase_intervals_ms(
    words: Sequence[dict],
    phrases,
    padding_ms: int,
    enable_fuzzy: bool = ENABLE_FUZZY_MATCHING,
    fuzzy_threshold_multi_word: float = FUZZY_THRESHOLD_MULTI_WORD,
    fuzzy_threshold_single_word: float = FUZZY_THRESHOLD_SINGLE_WORD,
    label_prefix: str = "phrase",
) -> list[Interval]:
    normalized_words = []
    for w in words:
        nw = normalize_greek(w["text"])
        if nw:
            normalized_words.append({
                "text": nw,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "raw": str(w["text"]).strip(),
            })

    phrase_groups = _as_phrase_groups(phrases)
    intervals: list[Interval] = []
    used_spans: set[tuple[int, int, str]] = set()

    def add_interval(i: int, j: int, label: str, method: str, score: float = 1.0):
        if i < 0 or j <= i or j > len(normalized_words):
            return
        key = (i, j, label)
        if key in used_spans:
            return
        used_spans.add(key)

        start_ms = int(normalized_words[i]["start"] * 1000) - padding_ms
        end_ms = int(normalized_words[j - 1]["end"] * 1000) + padding_ms
        raw_match = " ".join(w["raw"] for w in normalized_words[i:j])
        print(f"  [Match:{method}] {label_prefix}:{label!r} <= {raw_match!r} score={score:.2f}")
        intervals.append(Interval(max(0, start_ms), max(0, end_ms), f"{label_prefix}: {label}"))

    for label, aliases in phrase_groups.items():
        for alias in aliases:
            alias_norm = normalize_greek(alias)
            target = alias_norm.split()
            n = len(target)
            if n == 0:
                continue

            # Exact contiguous match.
            for i in range(0, len(normalized_words) - n + 1):
                window_words = [w["text"] for w in normalized_words[i : i + n]]
                if window_words == target:
                    add_interval(i, i + n, label, "exact", 1.0)

            if not enable_fuzzy:
                continue

            # Fuzzy matching. For one-word fillers, keep strict to avoid accidental deletion.
            if n == 1:
                sizes = [1]
                threshold = fuzzy_threshold_single_word
            else:
                sizes = list(range(max(1, n - 1), min(len(normalized_words), n + 2) + 1))
                threshold = fuzzy_threshold_multi_word

            for size in sizes:
                for i in range(0, len(normalized_words) - size + 1):
                    window_text = " ".join(w["text"] for w in normalized_words[i : i + size])
                    score = SequenceMatcher(None, alias_norm, window_text).ratio()
                    if score >= threshold:
                        add_interval(i, i + size, label, "fuzzy", score)

    return merge_intervals(intervals)


def manual_intervals_ms(manual_intervals_sec: Sequence[tuple[float, float]], padding_ms: int = 0) -> list[Interval]:
    intervals = []
    for start_s, end_s in manual_intervals_sec:
        start_ms = int(float(start_s) * 1000) - padding_ms
        end_ms = int(float(end_s) * 1000) + padding_ms
        if end_ms > start_ms:
            intervals.append(Interval(max(0, start_ms), max(0, end_ms), "manual"))
    return intervals


# =====================================================================
# Cutting with crossfade
# =====================================================================

def concat_with_crossfade(parts: Sequence[np.ndarray], sr: int, crossfade_ms: int = CROSSFADE_MS) -> np.ndarray:
    valid_parts = [safe_float32(p) for p in parts if len(p) > 0]
    if not valid_parts:
        return np.zeros(0, dtype=np.float32)

    crossfade_samples = int(sr * crossfade_ms / 1000)
    output = valid_parts[0]

    for part in valid_parts[1:]:
        n = min(crossfade_samples, len(output), len(part))
        if n <= 1:
            output = np.concatenate([output, part])
            continue
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
        overlap = output[-n:] * fade_out + part[:n] * fade_in
        output = np.concatenate([output[:-n], overlap, part[n:]])

    return safe_float32(output)


def cut_intervals_from_audio(y: np.ndarray, sr: int, intervals: Sequence[Interval], crossfade_ms: int = CROSSFADE_MS) -> np.ndarray:
    if not intervals:
        return safe_float32(y)

    total_ms = int(len(y) / sr * 1000)
    cleaned_intervals = []
    for iv in merge_intervals(intervals):
        start_ms = max(0, min(total_ms, int(iv.start_ms)))
        end_ms = max(0, min(total_ms, int(iv.end_ms)))
        if end_ms > start_ms:
            cleaned_intervals.append(Interval(start_ms, end_ms, iv.label))

    parts = []
    cursor_ms = 0
    for iv in cleaned_intervals:
        start_sample = int(cursor_ms * sr / 1000)
        end_sample = int(iv.start_ms * sr / 1000)
        if end_sample > start_sample:
            parts.append(y[start_sample:end_sample])
        cursor_ms = iv.end_ms

    cursor_sample = int(cursor_ms * sr / 1000)
    if cursor_sample < len(y):
        parts.append(y[cursor_sample:])

    return concat_with_crossfade(parts, sr, crossfade_ms=crossfade_ms)


# =====================================================================
# Filters, EQ, dynamics
# =====================================================================

def butter_filter(y: np.ndarray, sr: int, cutoff_hz: float, btype: str, order: int = 4) -> np.ndarray:
    cutoff_hz = min(cutoff_hz, sr / 2 - 50.0)
    sos = butter(order, cutoff_hz, btype=btype, fs=sr, output="sos")
    return safe_float32(sosfiltfilt(sos, y))


def bandpass_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    high_hz = min(high_hz, sr / 2 - 50.0)
    if high_hz <= low_hz:
        return np.zeros_like(y, dtype=np.float32)
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def biquad_peaking_eq(y: np.ndarray, sr: int, freq_hz: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    if freq_hz >= sr / 2 - 100:
        return safe_float32(y)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return safe_float32(lfilter(b, a, y))


def smooth_envelope(abs_y: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    attack = np.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr))
    release = np.exp(-1.0 / max(1.0, release_ms * 0.001 * sr))
    env = np.zeros_like(abs_y, dtype=np.float32)
    previous = 0.0
    for i, x in enumerate(abs_y):
        coeff = attack if x > previous else release
        previous = coeff * previous + (1.0 - coeff) * float(x)
        env[i] = previous
    return env


def repair_clipping_soft(y: np.ndarray, clip_threshold: float = 0.98) -> np.ndarray:
    y = safe_float32(y)
    clipped = np.mean(np.abs(y) >= clip_threshold)
    if clipped <= 0:
        return y
    scale = 2.0 / clip_threshold
    print(f"[Enhance] Soft clipping repair applied to {clipped * 100:.3f}% near-clipped samples.")
    return safe_float32(np.tanh(y * scale) / np.tanh(scale))


def light_declick(y: np.ndarray, strength: float = DECLICK_STRENGTH) -> np.ndarray:
    """Very light mouth-click impulse smoothing. Conservative to avoid damaging speech."""
    if not LIGHT_DECLICK or strength <= 0:
        return safe_float32(y)
    y = safe_float32(y)
    if len(y) < 15:
        return y

    # Median-filter estimate of local waveform. Replace only strong tiny impulses.
    med = medfilt(y, kernel_size=7).astype(np.float32)
    residual = np.abs(y - med)
    threshold = np.percentile(residual, 99.85) * 1.2 + 1e-6
    click_mask = residual > threshold
    click_fraction = float(np.mean(click_mask))
    if click_fraction > 0:
        print(f"[Enhance] Light declick affected {click_fraction * 100:.3f}% samples.")
        y = np.where(click_mask, y * (1.0 - strength) + med * strength, y)
    return safe_float32(y)


def speech_eq(y: np.ndarray, sr: int) -> np.ndarray:
    """Mild voice EQ for 16 kHz speech."""
    y = butter_filter(y, sr, 70.0, "highpass", order=4)
    y = biquad_peaking_eq(y, sr, 250.0, gain_db=-2.0, q=1.0)   # less boxy
    y = biquad_peaking_eq(y, sr, 3400.0, gain_db=2.0, q=0.8)   # clarity/presence
    y = butter_filter(y, sr, min(7400.0, sr / 2 - 100.0), "lowpass", order=4)
    return safe_float32(y)


def de_esser(y: np.ndarray, sr: int, threshold_db: float = -30.0, ratio: float = 4.0) -> np.ndarray:
    y = safe_float32(y)
    low_hz = 5000.0
    high_hz = min(7600.0, sr / 2 - 100.0)
    if high_hz <= low_hz:
        return y

    sib = bandpass_filter(y, sr, low_hz, high_hz, order=3)
    env = smooth_envelope(np.abs(sib), sr, attack_ms=2.0, release_ms=60.0)
    env_db = amp_to_db(env)
    over_db = np.maximum(env_db - threshold_db, 0.0)
    reduction_db = over_db * (1.0 - 1.0 / ratio)
    gain = 10.0 ** (-reduction_db / 20.0)
    return safe_float32(y - sib * (1.0 - gain))


def compressor(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 1.8,
    attack_ms: float = 8.0,
    release_ms: float = 120.0,
    makeup_db: float = 1.5,
) -> np.ndarray:
    y = safe_float32(y)
    env = smooth_envelope(np.abs(y), sr, attack_ms, release_ms)
    env_db = amp_to_db(env)
    over_db = np.maximum(env_db - threshold_db, 0.0)
    gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
    gain_db = makeup_db - gain_reduction_db
    gain = 10.0 ** (gain_db / 20.0)
    return safe_float32(y * gain)


def tail_expander(y: np.ndarray, sr: int) -> np.ndarray:
    """Downward expansion for quiet room/reverb tails between speech."""
    if not TAIL_EXPANDER:
        return safe_float32(y)
    y = safe_float32(y)
    env = smooth_envelope(np.abs(y), sr, attack_ms=5.0, release_ms=180.0)
    env_db = amp_to_db(env)
    quiet = env_db < TAIL_THRESHOLD_DB
    target_gain = np.ones_like(y, dtype=np.float32)
    target_gain[quiet] = db_to_amp(TAIL_REDUCTION_DB)
    # Smooth mask to avoid pumping.
    win = max(3, int(sr * 0.020))
    kernel = np.ones(win, dtype=np.float32) / win
    gain = np.convolve(target_gain, kernel, mode="same")
    return safe_float32(y * gain)


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    y = safe_float32(y)
    if len(y) == 0:
        return y
    if pyln is not None:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if np.isfinite(loudness):
            print(f"[Enhance] Loudness {loudness:.1f} LUFS -> {target_lufs:.1f} LUFS")
            return safe_float32(pyln.normalize.loudness(y, loudness, target_lufs))

    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    target_rms = db_to_amp(-20.0)
    print("[Enhance] pyloudnorm missing/unusable. RMS normalization fallback applied.")
    return safe_float32(y * (target_rms / rms))


def final_limiter(y: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    y = safe_float32(y)
    if len(y) == 0:
        return y
    ceiling = db_to_amp(ceiling_db)
    drive = 1.25
    y = np.tanh(y * drive) / np.tanh(drive)
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > ceiling:
        y = y * (ceiling / peak)
    return safe_float32(y)


# =====================================================================
# AI denoise / fallback denoise
# =====================================================================

def find_deepfilter_cli() -> Optional[str]:
    return shutil.which("deepFilter") or shutil.which("deep-filter")


def deepfilter_enhance_if_available(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    exe = find_deepfilter_cli()
    if exe is None:
        print("[Enhance] DeepFilterNet CLI not found. Using noisereduce fallback.")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_wav = tmp / "input.wav"
        output_dir = tmp / "deepfilter_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_wav(str(input_wav), y, sr)

        command_variants = [
            [exe, "--output-dir", str(output_dir), str(input_wav)],
            [exe, str(input_wav), "--output-dir", str(output_dir)],
            [exe, "-o", str(output_dir), str(input_wav)],
            [exe, str(input_wav), "-o", str(output_dir)],
        ]

        for cmd in command_variants:
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                wavs = sorted(output_dir.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
                if wavs:
                    enhanced, _ = load_audio_mono(str(wavs[0]), target_sr=sr)
                    print("[Enhance] DeepFilterNet applied.")
                    return enhanced
            except Exception:
                continue

    print("[Enhance] DeepFilterNet found but failed. Using noisereduce fallback.")
    return None


def reduce_noise_fallback(y: np.ndarray, sr: int, prop_decrease: float = FALLBACK_NOISE_REDUCTION) -> np.ndarray:
    if nr is None:
        print("[Enhance] noisereduce not installed. Skipping fallback denoise.")
        return safe_float32(y)

    print(f"[Enhance] noisereduce fallback, prop_decrease={prop_decrease}")
    return safe_float32(
        nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=False,
            prop_decrease=prop_decrease,
            time_constant_s=1.7,
            n_fft=1024,
            hop_length=256,
            n_jobs=1,
        )
    )


# =====================================================================
# Voice-only cleanup: non-speech, silence, breaths
# =====================================================================

def get_speech_intervals_samples(y: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Energy-based speech-ish intervals. Works after denoise; not a neural VAD."""
    if len(y) == 0:
        return []

    raw_intervals = librosa.effects.split(y, top_db=VOICE_TOP_DB)
    pad = int(sr * VOICE_PADDING_MS / 1000)
    min_len = int(sr * MIN_SPEECH_SEGMENT_MS / 1000)

    intervals = []
    for start, end in raw_intervals:
        if end - start < min_len:
            continue
        intervals.append((max(0, start - pad), min(len(y), end + pad)))

    # Merge touching/overlapping intervals.
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        ps, pe = merged[-1]
        if start <= pe:
            merged[-1] = (ps, max(pe, end))
        else:
            merged.append((start, end))
    return merged


def attenuate_non_speech(y: np.ndarray, sr: int, intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    if not intervals:
        return safe_float32(y)
    y = safe_float32(y)
    gain = np.full(len(y), db_to_amp(NON_SPEECH_ATTENUATION_DB), dtype=np.float32)
    for start, end in intervals:
        gain[start:end] = 1.0

    # Smooth gain changes to avoid clicks/pumping.
    win = max(3, int(sr * 0.025))
    kernel = np.ones(win, dtype=np.float32) / win
    gain = np.convolve(gain, kernel, mode="same")
    print(f"[VoiceOnly] Non-speech attenuated by {NON_SPEECH_ATTENUATION_DB:.0f} dB.")
    return safe_float32(y * gain)


def shorten_long_silences(y: np.ndarray, sr: int, intervals: Sequence[tuple[int, int]], keep_ms: int = MAX_SILENCE_KEEP_MS) -> np.ndarray:
    if not intervals:
        return safe_float32(y)

    keep_samples = int(sr * keep_ms / 1000)
    pieces: list[np.ndarray] = []
    cursor = 0

    for start, end in intervals:
        # Keep only a small amount of silence before each speech segment.
        gap_start = cursor
        gap_end = start
        gap_len = max(0, gap_end - gap_start)

        if gap_len > 0:
            if gap_len <= keep_samples:
                pieces.append(y[gap_start:gap_end])
            else:
                # Keep half before/half after so transitions sound less abrupt.
                first = keep_samples // 2
                second = keep_samples - first
                pieces.append(y[gap_start:gap_start + first])
                pieces.append(y[gap_end - second:gap_end])

        pieces.append(y[start:end])
        cursor = end

    # Keep final tail only if short.
    if cursor < len(y):
        tail = y[cursor:]
        if len(tail) <= keep_samples:
            pieces.append(tail)
        else:
            pieces.append(tail[:keep_samples])

    out = concat_with_crossfade(pieces, sr, crossfade_ms=CROSSFADE_MS)
    removed_sec = max(0.0, (len(y) - len(out)) / sr)
    if removed_sec > 0:
        print(f"[VoiceOnly] Shortened long silence/dead-air by about {removed_sec:.2f}s.")
    return safe_float32(out)


def reduce_harsh_breaths(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Conservative breath reducer based on frame energy + spectral centroid + zero-crossing.
    This targets noisy low/mid-level breath frames; it may also soften some fricatives if too aggressive.
    """
    if not REDUCE_HARSH_BREATHS or len(y) == 0:
        return safe_float32(y)

    frame_length = 512
    hop_length = 128
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    rms_db = amp_to_db(rms)

    breath_frames = (
        (rms_db > BREATH_RMS_MIN_DB)
        & (rms_db < BREATH_RMS_MAX_DB)
        & (centroid > BREATH_CENTROID_MIN_HZ)
        & (zcr > BREATH_ZCR_MIN)
    )

    # Smooth/dilate frame decisions.
    breath_frames = binary_dilation(breath_frames, iterations=2)

    if not np.any(breath_frames):
        print("[VoiceOnly] No strong breath-like frames detected.")
        return safe_float32(y)

    frame_gain = np.ones_like(rms, dtype=np.float32)
    frame_gain[breath_frames] = db_to_amp(BREATH_REDUCTION_DB)

    # Convert frame gain to sample gain.
    sample_gain = np.ones(len(y), dtype=np.float32)
    for idx, g in enumerate(frame_gain):
        start = idx * hop_length
        end = min(len(y), start + frame_length)
        sample_gain[start:end] = np.minimum(sample_gain[start:end], g)

    win = max(3, int(sr * 0.020))
    kernel = np.ones(win, dtype=np.float32) / win
    sample_gain = np.convolve(sample_gain, kernel, mode="same")

    affected = float(np.mean(sample_gain < 0.99))
    print(f"[VoiceOnly] Breath-like material reduced in {affected * 100:.2f}% of samples.")
    return safe_float32(y * sample_gain)


def voice_only_cleanup(y: np.ndarray, sr: int, mode: str = NON_SPEECH_MODE) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "off":
        return safe_float32(y)

    intervals = get_speech_intervals_samples(y, sr)
    if not intervals:
        print("[VoiceOnly] No speech intervals detected. Skipping voice-only cleanup.")
        return safe_float32(y)

    total_speech = sum(end - start for start, end in intervals) / sr
    print(f"[VoiceOnly] Detected {len(intervals)} speech regions, speech-ish time ≈ {total_speech:.2f}s.")

    out = y
    if mode in {"attenuate", "attenuate_and_shorten"}:
        out = attenuate_non_speech(out, sr, intervals)

    # Important: if we shortened after attenuation, use intervals from the original out. This is okay because
    # shorten_long_silences consumes the same pre-shortened signal.
    if mode in {"shorten", "attenuate_and_shorten"}:
        out = shorten_long_silences(out, sr, intervals, keep_ms=MAX_SILENCE_KEEP_MS)

    return safe_float32(out)


# =====================================================================
# Full studio enhancement chain
# =====================================================================

def studio_enhance_array(y, sr, use_ai=USE_AI_DENOISE,
                         target_lufs=TARGET_LUFS,
                         fallback_noise_reduction=FALLBACK_NOISE_REDUCTION):
    if len(y) == 0:
        return y

    y = repair_clipping_soft(y)
    y = light_declick(y)
    y = butter_filter(y, sr, 120.0, "highpass", order=4)  # 70 → 120

    enhanced = deepfilter_enhance_if_available(y, sr) if use_ai else None
    if enhanced is not None:
        y = enhanced
    else:
        y = reduce_noise_fallback(y, sr, prop_decrease=fallback_noise_reduction)

    y = tail_expander(y, sr)
    y = speech_eq(y, sr)
    y = de_esser(y, sr)
    y = reduce_harsh_breaths(y, sr)

    # Voice cleanup ΠΡΙΝ τον compressor — έτσι κρίνει το φυσικό επίπεδο
    y = voice_only_cleanup(y, sr, mode=NON_SPEECH_MODE)

    y = compressor(y, sr)
    y = loudness_normalize(y, sr, target_lufs=target_lufs)
    y = final_limiter(y, ceiling_db=-1.0)
    # ΔΕΝ υπάρχει δεύτερο tail_expander/loudness/limiter — αυτό έκανε over-compression
    return safe_float32(y)


# =====================================================================
# End-to-end pipeline
# =====================================================================

def collect_cut_intervals(words: Sequence[dict]) -> list[Interval]:
    intervals: list[Interval] = []

    print("\n[Cut] Searching target phrases...")
    intervals += find_phrase_intervals_ms(
        words,
        PHRASE_GROUPS,
        padding_ms=PHRASE_PADDING_MS,
        label_prefix="phrase",
    )

    if REMOVE_FILLERS:
        print("\n[Cut] Searching filler phrases...")
        intervals += find_phrase_intervals_ms(
            words,
            FILLER_GROUPS,
            padding_ms=FILLER_PADDING_MS,
            label_prefix="filler",
            # Slightly stricter for fillers.
            fuzzy_threshold_multi_word=max(0.86, FUZZY_THRESHOLD_MULTI_WORD),
            fuzzy_threshold_single_word=max(0.95, FUZZY_THRESHOLD_SINGLE_WORD),
        )

    intervals += manual_intervals_ms(MANUAL_CUT_INTERVALS_SEC, padding_ms=0)
    return merge_intervals(intervals)


def clean_wav_voice_only_studio(input_wav: str, output_wav: str) -> str:
    input_wav = str(input_wav)
    output_wav = str(output_wav)

    # 1) Transcribe original audio for phrase/filler timestamps.
    words = transcribe_words(input_wav)

    if PRINT_TRANSCRIPT:
        txt = transcript_from_words(words)
        print("\n[Transcript]")
        print(txt)
        print("\n[Transcript normalized]")
        print(normalize_greek(txt))

    if PRINT_WORD_TIMESTAMPS:
        print("\n[Word timestamps]")
        for w in words:
            print(f"  {w['start']:.2f}s - {w['end']:.2f}s : {w['text']}")

    # 2) Collect intervals to remove.
    intervals = collect_cut_intervals(words)

    print("\n[Cut] Intervals to remove:")
    if not intervals:
        print("  No target phrases/fillers found. Audio will not be phrase-cut.")
        print("  Use [Transcript normalized] to add variants to PHRASE_GROUPS/FILLER_GROUPS.")
    else:
        for iv in intervals:
            print(f"  {iv.start_ms / 1000:.2f}s - {iv.end_ms / 1000:.2f}s : {iv.label}")

    # 3) Load 16 kHz and cut.
    y, sr = load_audio_mono(input_wav, target_sr=OUTPUT_SAMPLE_RATE)
    y_cut = cut_intervals_from_audio(y, sr, intervals, crossfade_ms=CROSSFADE_MS)

    if SAVE_CUT_ONLY_WAV:
        save_wav(SAVE_CUT_ONLY_WAV, y_cut, sr)
        print(f"[Cut] Saved cut-only WAV -> {SAVE_CUT_ONLY_WAV}")

    # 4) Studio enhancement + voice-only cleanup.
    if ENABLE_STUDIO_ENHANCEMENT:
        if SAVE_BEFORE_FINAL_VOICE_ONLY_WAV:
            save_wav(SAVE_BEFORE_FINAL_VOICE_ONLY_WAV, y_cut, sr)
        y_final = studio_enhance_array(y_cut, sr)
    else:
        y_final = voice_only_cleanup(y_cut, sr, mode=NON_SPEECH_MODE)

    # 5) Save 16 kHz output.
    save_wav(output_wav, y_final, sr)
    print(f"\n[DONE] Saved final 16 kHz WAV -> {output_wav}")
    print(f"[DONE] Duration: input={len(y)/sr:.2f}s output={len(y_final)/sr:.2f}s")
    return output_wav