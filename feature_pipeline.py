


import numpy as np
import os

from feature_extraction_v4 import extract_reader_features

from detect_words_faster import detect_words
from detect_words_hierarchical import detect_words_hierarchical
from extract_reading_features_DetectedWords import extract_reading_features_DetectedWords, find_detected_words_xlsx


# ── Feature extraction ─────────────
def feature_pipeline(sub_id, text, sub_path, clean_audio_path, device, model, align_model, metadata, expected_words):
    failed = False
    
    try:
        ########### Features from Reading Text/Based on Words Detected ###########
        xlsx_path = find_detected_words_xlsx(sub_path, sub_id)
        
        if xlsx_path:
            os.remove(xlsx_path)  # Remove existing XLSX to ensure fresh detection
        
        # No XLSX — run WhisperX and save it
        print(f"  Pipeline A: no XLSX — running WhisperX...")
        detect_words(sub_id, text, sub_path, clean_audio_path, device, model, align_model, metadata, expected_words)
        # detect_words_hierarchical(sub_id, text, sub_path, clean_audio_path, device, model, expected_words)
        xlsx_path = find_detected_words_xlsx(sub_path, sub_id)  # re-check after detection
    
    except Exception as e:
        print(f"  [ERROR] Feature extraction failed for {sub_id}: {e}")
        failed = True
    
    return failed