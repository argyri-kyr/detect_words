
import os
from tqdm import tqdm


from globalVars import folder1, folder2

from edit_audio import resolve_audio_path

from clean_audio_pipeline import clean_audio_pipeline

# from load_whisperx_model import load_whisperx_model
from load_faster_whisperx_model import load_whisperx_model

from feature_pipeline import feature_pipeline

from feature_extraction_v4 import get_class_code



# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    text = 'decodewordspseudo'
    
    # subs = [subno for subno in range(14501, 14514)]
    # subs += [13512, 13514, 13515]
    
    # subs = [14504, 14508, 14505, 14502]
    
    # subs = [14317, 14318, 14322, 14323, 14325, 14326, 14329, 14333, 14334, 14336, 14339, 14341, 14345, 14346, 14347]
    
    # subs = [subno for subno in range(14302, 14438)]
    # subs = [14395]
    
    # subs = [14302, 14329, 14345, 14350, 14363, 14371, 14375, 14379, 14394, 14433, 14435, 14436]
    
    # subs = [14363, 14371, 14302, 14388, 14341, 14339, 14367, 14356]
    
    # subs = [14379, 14356, 14359, 14391, 14398, 14399, 14408, 14428, 14336]
    
    subs = [1973, 1976]
    
    # Prepare Model and cache for whisperx 
    device, model, align_model, metadata = load_whisperx_model()
    # # # # # # # # # # # # # # # # # #  #

    # Load text
    with open(folder1 + f'texts/{text}_1920/alltext.txt', encoding="utf-8") as f:
        expected_words = f.read().split()

    expected_words = expected_words[:24]
    # ── PHASE 1: Extract features ─────────────────────────────────────
    for sub_id in tqdm(subs, desc="Processing subjects"):

        sub_path = os.path.join(folder1, folder2, f"subject_{sub_id}")
        if not os.path.exists(sub_path):
            continue

        print(f"\n{'='*55}")
        print(f"  Subject {sub_id}")
        print(f"{'='*55}")

        # Class code from AVI / folder filename prefix
        group_code = get_class_code(sub_path)
        print(f"  Class: {group_code}")

        # ── Step A: Resolve raw audio (.avi → .wav if needed) ─────
        raw_audio_path = resolve_audio_path(sub_id, sub_path, text)
        if raw_audio_path is None:
            continue

        # ── Step B: Noise removal → saves _clean.wav ─────────────
        clean_audio_path = clean_audio_pipeline(sub_id, raw_audio_path)
        
        # ── Step C: Feature extraction ─────────────
        continue_next = feature_pipeline(sub_id, text, sub_path, clean_audio_path, device, model, align_model, metadata, expected_words)
        if continue_next:
            continue


main()