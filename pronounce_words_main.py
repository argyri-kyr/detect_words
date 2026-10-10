# slice_words_audio.py
# Slices an audio file into per-word clips based on a word timestamps file.
# Accepts both .xlsx and .csv (auto-detected by extension).
#
# Requirements:
#   pip install pydub openpyxl
#   Also needs ffmpeg installed: https://ffmpeg.org/download.html

import os
import re
import pandas as pd
from pydub import AudioSegment

from globalVars import folder1, folder2
from edit_audio import resolve_audio_path
from clean_audio_pipeline import clean_audio_pipeline


# ---------- config ----------
# SUBS = [subno for subno in range(14501, 14514)]
# SUBS += [13512, 13514, 13515]

# SUBS         = [14502, 14503, 14508, 14510, 14512]

# SUBS = [14356]

# SUBS = [subno for subno in range(14302, 14438)]

SUBS = [1973, 1976]

# SUBS = [14379, 14356, 14359, 14391, 14398, 14399, 14408, 14428, 14336]

        
PADDING_START   = 300    # extra ms to add before each word
PADDING_END = 0          # extra ms to add after each word (not implemented yet)
ONLY_ALIGNED = False     # set True to skip words where aligned=False
FMT          = "wav"     # "wav" or "mp3"
SPEED        = 1.0       # playback speed: 1.0 = normal, 0.75 = slower, 1.5 = faster
# ----------------------------


def sanitize(word):
    return re.sub(r'[\\/*?:"<>|]', "_", word)


def change_speed(clip, speed):
    if speed == 1.0:
        return clip
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        clip.export(src, format="wav")
        if 0.5 <= speed <= 2.0:
            atempo = f"atempo={speed}"
        elif speed < 0.5:
            atempo = f"atempo={speed*2},atempo=0.5"
        else:
            atempo = f"atempo=2.0,atempo={speed/2}"
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-filter:a", atempo, dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return AudioSegment.from_wav(dst)

def load_timestamps(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        # utf-8-sig strips the BOM automatically; quoting handled by pandas
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx or .csv.")
    df.columns = df.columns.str.strip().str.lower()
    # strip punctuation attached to words e.g. "λοιπόν," -> "λοιπόν"
    df['word'] = df['word'].astype(str).str.strip().str.strip('",.')
    return df


def find_timestamps_file(sub_path, sub_id, text):
    for ext in (".xlsx", ".csv"):
        path = os.path.join(sub_path, f"{text}_detected_words_v3_{sub_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def slice_audio(sub_id, text, audio_path, timestamps_path, padding_start=0, padding_end=0, only_aligned=False, fmt="wav", speed=1.0):

    out_dir = os.path.join(os.path.dirname(timestamps_path), f"{text}_word_clips_{sub_id}")
    os.makedirs(out_dir, exist_ok=True)

    df = load_timestamps(timestamps_path)

    if 'start_ms' in df.columns and 'end_ms' in df.columns:
        df['_start_ms'] = df['start_ms'].astype(float)
        df['_end_ms']   = df['end_ms'].astype(float)
    elif 'start_s' in df.columns and 'end_s' in df.columns:
        df['_start_ms'] = df['start_s'].astype(float) * 1000
        df['_end_ms']   = df['end_s'].astype(float) * 1000
    else:
        raise ValueError(f"[{sub_id}] File must have start_ms/end_ms or start_s/end_s columns.")

    if 'word' not in df.columns:
        raise ValueError(f"[{sub_id}] File must have a 'word' column.")

    if only_aligned and 'aligned' in df.columns:
        before = len(df)
        df = df[df['aligned'].astype(str).str.lower() == 'true']
        print(f"  [{sub_id}] Aligned-only filter: {len(df)}/{before} words kept.")

    ext = os.path.splitext(audio_path)[-1].lower().lstrip('.')
    print(f"  [{sub_id}] Loading audio: {audio_path}")
    audio    = AudioSegment.from_file(audio_path, format=ext if ext else "wav")
    total_ms = len(audio)
    print(f"  [{sub_id}] Duration: {total_ms} ms, words to slice: {len(df)}, speed: {speed}x")

    skipped = 0
    for i, row in df.iterrows():
        word  = str(row['word'])
        start = max(0, row['_start_ms'] - padding_start)
        end   = min(total_ms, row['_end_ms'] + padding_end)

        if start >= end:
            print(f"  [{sub_id}] SKIP row {i} '{word}': invalid range {start}-{end} ms")
            skipped += 1
            continue

        clip  = change_speed(audio[start:end], speed)
        fname = f"{i:04d}_{sanitize(word)}.{fmt}"
        clip.export(os.path.join(out_dir, fname), format=fmt)

    saved = len(df) - skipped
    print(f"  [{sub_id}] Done. {saved} clips saved to '{out_dir}/', {skipped} skipped.\n")


def main():
    text = 'decodewordspseudo'
    
    base_folder = os.path.join(folder1, folder2)

    for sub_id in SUBS:
        sub_path        = os.path.join(base_folder, f"subject_{sub_id}")
        
        if not os.path.exists(sub_path):
            continue
        
        raw_audio       = resolve_audio_path(sub_id, sub_path, text)
        audio_path      = clean_audio_pipeline(sub_id, raw_audio)
        timestamps_path = find_timestamps_file(sub_path, sub_id, text)
        
        if not os.path.exists(audio_path):
            print(f"[{sub_id}] Audio not found: {audio_path} — skipping.")
            continue
        if timestamps_path is None:
            print(f"[{sub_id}] No timestamps file found in {sub_path} — skipping.")
            continue

        slice_audio(sub_id, text, audio_path, timestamps_path,
                    padding_start=PADDING_START, padding_end=PADDING_END, only_aligned=ONLY_ALIGNED, fmt=FMT, speed=SPEED)


if __name__ == "__main__":
    main()