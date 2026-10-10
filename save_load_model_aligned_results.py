import json
import numpy as np


def save_alignment_results(word_list, filepath):
    # convert np.float64 to plain float so json can serialize them
    serializable = [
        {
            "word": entry["word"],
            "start": float(entry["start"]),
            "end": float(entry["end"]),
            "score": float(entry["score"]),
        }
        for entry in word_list
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_alignment_results(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # restore np.float64 so the struct matches what WhisperX originally returned
    return [
        {
            "word": entry["word"],
            "start": np.float64(entry["start"]),
            "end": np.float64(entry["end"]),
            "score": np.float64(entry["score"]),
        }
        for entry in raw
    ]