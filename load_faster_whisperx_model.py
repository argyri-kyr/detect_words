

import os
import whisperx
from faster_whisper import WhisperModel # Import this directly


# -------------------------------------------------------------------------
# 1. Updated Model Loading (Loading two specialized components)
# -------------------------------------------------------------------------
def load_whisperx_model():
    os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
    device = "cpu"
    
    # Transcription Engine (Supports initial_prompt)
    model = WhisperModel("large-v3", device=device, compute_type="float32")
    
    # Alignment Engine (Your specialized Greek Aligner)
    align_model, metadata = whisperx.load_align_model(
        language_code="el", 
        device=device, 
        model_name="lighteternal/wav2vec2-large-xlsr-53-greek"
    )
    
    return device, model, align_model, metadata