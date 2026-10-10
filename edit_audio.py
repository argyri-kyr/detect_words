

from pathlib import Path
import os
from pydub import AudioSegment

from globalVars import folder1, folder2
# ─────────────────────────────────────────────
# AVI → WAV CONVERSION
# ─────────────────────────────────────────────

def convert_avi_to_wav(sub_id, avi_path, text):
    # # # # # # Define output path based on subject folder structure # # # # # # 
    avi_path    = Path(avi_path)
    
    # Walk up the path until we find the subject_{id} folder
    subject_dir = avi_path
    while subject_dir.name != f"subject_{sub_id}":
        subject_dir = subject_dir.parent
        if subject_dir == subject_dir.parent:
            raise ValueError(f"Could not find subject_{sub_id} in path: {avi_path}")
    
    output_path = subject_dir / f"{sub_id}_{text}.wav"
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    if not output_path.exists():
        audio = AudioSegment.from_file(avi_path, format="avi")
        audio.export(str(output_path), format="wav")
        print(f"  [AVI→WAV] {output_path.name}")
    return str(output_path)


# ─────────────────────────────────────────────
# RESOLVE RAW AUDIO PATH
# ─────────────────────────────────────────────

def resolve_audio_path(sub_id, sub_path, text):
    """
    Find the raw .wav for a subject (Case A: from .avi, Case B: existing .wav).
    Returns raw audio path (str) or None if nothing found.
    """
    wav_files  = []
    avi_files  = []
    wav_found  = False
    audio_path = None

    text_folder_name = [
        f for f in os.listdir(sub_path)
        if text in f and not f.endswith(".avi") and not f.endswith(".wav") and not f.endswith(".txt") and not f.endswith(".json") and not f.endswith(".xlsx") and not f.endswith(".pkl") and 'clips' not in f
    ]

    if text_folder_name:
        vl_path   = os.path.join(sub_path, text_folder_name[0])
        wav_files = [f for f in os.listdir(vl_path) if f.endswith(".wav")]
        avi_files = [f for f in os.listdir(vl_path) if f.endswith(".avi")]
        if wav_files:
            audio_path = os.path.join(vl_path, wav_files[0])
            wav_found  = True

    wav_files += [f for f in os.listdir(sub_path) if f.endswith(".wav")]
    avi_files += [f for f in os.listdir(sub_path) if f.endswith(".avi")]

    if not wav_files and not avi_files:
        print(f"  [SKIP] No .wav or .avi in {sub_path}")
        return None

    if not wav_files and avi_files:
        vl_path    = os.path.join(sub_path, text_folder_name[0]) if text_folder_name else sub_path
        avi_path   = os.path.join(vl_path, avi_files[0])
        audio_path = convert_avi_to_wav(sub_id, avi_path, text)
    elif wav_files and not wav_found:
        audio_path = os.path.join(sub_path, wav_files[0])

    return audio_path
