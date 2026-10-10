import os
import re
import pandas as pd
import whisperx
from rapidfuzz import fuzz

def find_word_boundaries(transcript_words, expected_words, similarity_threshold=65):
    """
    Core logic to map expected words to WhisperX timestamps, 
    accounting for syllable-by-syllable decoding.
    """
    results = []
    current_transcript_idx = 0
    valid_words = [w for w in transcript_words if 'start' in w and 'end' in w]

    for expected_word in expected_words:
        expected_word_clean = expected_word.lower().strip()
        match_found = False
        
        for i in range(current_transcript_idx, len(valid_words)):
            # Clean punctuation from ASR output
            transcribed_word = re.sub(r'[^\w\s]', '', valid_words[i]['word'].lower().strip())
            
            # Fuzzy match to catch pseudowords or slight mispronunciations
            similarity = fuzz.ratio(expected_word_clean, transcribed_word)
            
            if similarity >= similarity_threshold or transcribed_word in expected_word_clean:
                end_time = valid_words[i]['end']
                
                # LOOK-BACK: Capture the start of the first syllable/attempt
                first_attempt_idx = i
                for j in range(i - 1, current_transcript_idx - 1, -1):
                    prev_word = re.sub(r'[^\w\s]', '', valid_words[j]['word'].lower().strip())
                    
                    # Logic: If it's a short fragment or shares a prefix, it's an attempt
                    is_short_fragment = len(prev_word) <= 3
                    shares_prefix = prev_word[:2] == expected_word_clean[:2] if len(expected_word_clean) >= 2 else False
                    
                    if shares_prefix or is_short_fragment:
                        first_attempt_idx = j
                    else:
                        break 
                
                start_time = valid_words[first_attempt_idx]['start']
                
                results.append({
                    'word': expected_word,
                    'start_s': start_time,
                    'end_s': end_time,
                    'status': 'Found',
                    'matched_transcription': " ".join([tw['word'] for tw in valid_words[first_attempt_idx:i+1]])
                })
                
                current_transcript_idx = i + 1
                match_found = True
                break
        
        if not match_found:
            results.append({
                'word': expected_word,
                'start_s': 0,
                'end_s': 0,
                'status': 'Not Found',
                'matched_transcription': None,
            })

    return results

def detect_words(sub_id, sub_path, clean_audio_path, device, model, align_model, metadata, expected_words):
    print(f"\n--- Processing Subject: {sub_id} ---")
    
    # 1. WhisperX Processing
    audio = whisperx.load_audio(clean_audio_path)
    result = model.transcribe(audio)
    
    # Align using your specific Greek alignment model
    aligned_result = whisperx.align(
        result["segments"], 
        align_model, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False
    )
    
    # Flatten word segments
    word_segments = []
    for segment in aligned_result["segments"]:
        if "words" in segment:
            word_segments.extend(segment["words"])
            
    # 2. Matching logic
    matched_data = find_word_boundaries(word_segments, expected_words)
    
    # 3. Save to Excel
    df_words = pd.DataFrame(matched_data)
    
    # User requested save path logic
    save_path = os.path.join(sub_path, f"detected_words_v3_{sub_id}.xlsx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        df_words.to_excel(save_path, index=False)
        print(f"Successfully saved results to: {save_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

    # Return structured metadata for further use in script
    return {sub_id: matched_data}