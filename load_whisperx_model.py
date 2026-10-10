    

import os
import whisperx


def load_whisperx_model():
    os.environ["HF_HOME"]           = os.path.join(os.getcwd(), "hf_cache")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "hf_cache")
    
    device = "cpu"
    model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type="float32",
        language="el",
        vad_options={"min_silence_duration_ms": 50, "speech_pad_ms": 20},
    )
    align_model, metadata = whisperx.load_align_model(language_code="el", device=device, model_name="lighteternal/wav2vec2-large-xlsr-53-greek")
    # align_model, metadata = whisperx.load_align_model(language_code="el", device=device)
    
    return device, model, align_model, metadata