

import os
import re
import pandas as pd
import whisperx
from rapidfuzz import fuzz
import unicodedata


from save_load_model_aligned_results import load_alignment_results, save_alignment_results


def remove_accents(input_str):
    # Κανονικοποίηση σε μορφή NFD (διαχωρίζει τον χαρακτήρα από τον τόνο του)
    nfkd_form = unicodedata.normalize('NFD', input_str)
    # Φιλτράρισμα μόνο των χαρακτήρων που δεν είναι συνδυαστικά σημεία (τόνοι)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize_greek_phonetic(text):
    if not text:
        return ""

    # 1. Lowercase
    text = remove_accents(text.lower().strip())

    # 2. Handle Accented Pairs (Before stripping accents)
    # Mapping αϊ/εϊ to αη/εη
    text = text.replace('αϊ', 'αη').replace('εϊ', 'εη')

    # 3.  Protect 'ου' and the accents 'ευ/αυ'
    text = text.replace('αυ', 'αβ')
    text = text.replace('ευ', 'εβ')
    text = text.replace('ηυ', 'η')
    text = text.replace('αι', 'ε')
    text = text.replace('ου', 'U_PLACE')
    
    # Mapping the "i-sounds" to 'η' as requested
    text = re.sub(r'ει|οι|υι', 'η', text)
    
    # Mapping the "e-sound" αι to 'ε'
    text = text.replace('αι', 'ε')

    # 4. Remove remaining Accents (Diacritics)
    # This cleans up single vowels like ά, έ, ή, ό, ύ, ώ
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # 5. Final Single Letter Normalization
    # η -> η (already handled by digraphs, but for safety)
    # υ -> η (as per your logic for i-sounds mapping to η)
    text = text.replace('υ', 'η')
    
    # ω -> ο
    text = text.replace('ω', 'ο')
    
    # Optional: If you want all "i-sounds" to end up as the same character:
    text = text.replace('η', 'ι') 
    
    # Restore Protected Sound 'ου'
    text = text.replace('U_PLACE', 'ου')

    return text



def find_scores(transcript_words, expected_words):
    valid_words = [w for w in transcript_words if 'start' in w and 'end' in w]
    expected_phonetics = [normalize_greek_phonetic(w.lower().strip()) for w in expected_words]

    # matrix[i][j] = phonetic score of expected_words[i] vs valid_words[j]
    matrix = []
    
    # Append header row with valid words
    row = ['Expected\Detected'] + [w['word'] for w in valid_words]
    matrix.append(row)
    
    for index, exp_phon in enumerate(expected_phonetics):
        row = []
        row.append(expected_words[index])
        for det_word in valid_words:
            det_text = re.sub(r'[^\w\s]', '', det_word['word'].lower().strip())
            score = fuzz.ratio(exp_phon, normalize_greek_phonetic(det_text))
            row.append(score)
        matrix.append(row)

    return matrix


def find_word_boundaries(matrix, expected_words, valid_words, min_score):
    results = []
    last_matched_idx = None

    # Find the best match from the detected words, for this expected word
    for i, exp_word in enumerate(expected_words):
        # row i+1 because row 0 is the header, skip first cell (label)
        scores = matrix[i + 1][1:]

        # Define some variables
        best_j_list = []
        best_j = None
        first_idx = None
        last_idx = None
        
        
        # # # # # 1. Collect all indexes of the best/max score # # # # #
        best_score = max(scores) if scores else 0
        best_j_list = [j for j, score in enumerate(scores) if score == best_score]   
        
        if best_j_list==[]:
            results.append({
                'word': exp_word,
                'matched_transcription': None,
                'start_s': 0,
                'end_s': 0,
                'phonetic_score': 0,
                'status': 'Not Found'
            })
            continue
        
        
        # # # # # 2. Define which is gonna be the best_j eventually # # # # #
        
        # Case: More than one best_j see for example subject 14354, in word υπεύθυνος, there are two from the detected one that has the same max score) 
        # Choose which one whould be the best_j now
        # My idea is to pick the one that is closer to the last best, 
        # so based on the previous best_j i will pick the one which is closer to this, and this is gona be the best_j for this word
        # The though behind this is that we move to the next word, so the new best will be close to the previous best
        if len(best_j_list)>1:
            min_dist = None
            # This is if I have found two best scores for the first word
            # Then probably the first appearance will be the correct one as we are about the first word in this specific case
            if last_matched_idx==None:
                best_j = best_j_list[0]
            else:
                for idx in best_j_list:
                    if min_dist==None or min_dist>abs(idx-last_matched_idx):
                        min_dist = abs(idx-last_matched_idx)
                        best_j = idx
        else:
            best_j = best_j_list[0]
        
        
        # # # # # 3. After defining best_j, we have to define first_idx, last_idx # # # # #
        first_idx = best_j
        last_idx = best_j
        
        
        # # # # # 4. MERGE # # # # #
        
        # This will work only if I have the word in syllabes, not autocorrection, ( VALID; 'παυσι', 'πονας'     INVALID: 'παυσι', 'πονας', 'παυσιπονο' )
        # as if I have autocorection then best match will be the autocorrected and then if we merge it with something else the score will be lower
    
        # MERGING LOGIC: Check if merging with adjacent words improves the score, which can help with split detections or reading in syllables.
        if best_j_list!=[]:
            if best_j-2>=0:
                merged_word = valid_words[best_j-2]['word'] + valid_words[best_j-1]['word'] + valid_words[best_j]['word']
                merged_score = fuzz.ratio(normalize_greek_phonetic(exp_word), normalize_greek_phonetic(merged_word))
                if merged_score >= best_score:
                    best_score = merged_score
                    first_idx = best_j - 2
            if best_j-1>=0:
                merged_word = valid_words[best_j-1]['word'] + valid_words[best_j]['word']
                merged_score = fuzz.ratio(normalize_greek_phonetic(exp_word), normalize_greek_phonetic(merged_word))
                if merged_score >= best_score:
                    best_score = merged_score
                    first_idx = best_j - 1
            if best_j+1<len(valid_words):
                merged_word = valid_words[best_j]['word'] + valid_words[best_j+1]['word']
                merged_score = fuzz.ratio(normalize_greek_phonetic(exp_word), normalize_greek_phonetic(merged_word))
                if merged_score >= best_score:
                    best_score = merged_score
                    last_idx = best_j + 1
            if best_j+2<len(valid_words):
                merged_word = valid_words[best_j]['word'] + valid_words[best_j+1]['word'] + valid_words[best_j+2]['word']
                merged_score = fuzz.ratio(normalize_greek_phonetic(exp_word), normalize_greek_phonetic(merged_word))
                if merged_score >= best_score:
                    best_score = merged_score
                    last_idx = best_j + 2


        # # # # # 5. LOOK-BACK & LOOK-AHEAD # # # # #

        # This will work only if I have the autocorrection, not the word in syllabes, ( VALID; 'παυσιμονο', 'παυσιπονο' )
        
        # Look-back & Look-ahead logic: Check if the previous or next detected word has a similar score, which might indicate autocorrection.
        if best_j - 1 >= 0 and scores[best_j - 1] > min_score:
            first_idx = best_j - 1
        if best_j + 1 < len(scores) and scores[best_j + 1] > min_score:
            last_idx = best_j + 1
            
            
        # # # # # 6. BACKWARD SEARCH: Look for earlier attempts (syllable-by-syllable reads) # # # # #
        # Case: 'παυσι', 'πονας', 'παυσιμονο', 'παυσιπονο'  → capture all four attempts
        # Accumulate syllables backward and check if the merged set matches the expected word
        first_new_idx = None
        if first_idx > 0:
            for prev_j in range(first_idx - 1, -1, -1):
                # Stop if we reach or pass the previous word's end boundary
                if last_matched_idx is not None and prev_j <= last_matched_idx:
                    break
                
                # Merge from prev_j ALL THE WAY to (first_idx - 1), accumulating backward
                merged_text = " ".join([valid_words[k]['word'] for k in range(prev_j, first_idx)])
                merged_score = fuzz.ratio(normalize_greek_phonetic(exp_word), normalize_greek_phonetic(merged_text.replace(" ", "")))
                
                if merged_score >= min_score:
                    first_new_idx = prev_j
                    
        first_idx = first_new_idx if first_new_idx!=None else first_idx
        
        # # # # # 7. FORWARD SEARCH: Look for follow-up attempts (syllable breakdown after correct word) # # # # #
        # Case: 'παυσιπονο', 'παυσι', 'πονας' → capture all three attempts
        # Accumulate syllables forward and check if the merged set matches the expected word
        last_new_idx = None
        if last_idx < len(valid_words) - 1:
            for next_j in range(last_idx + 1, len(valid_words)):
                # Merge from (last_idx + 1) ALL THE WAY to next_j, accumulating forward
                merged_text = " ".join([valid_words[k]['word'] for k in range(last_idx + 1, next_j + 1)])
                merged_score = fuzz.ratio(normalize_greek_phonetic(exp_word), normalize_greek_phonetic(merged_text.replace(" ", "")))
                
                if merged_score >= min_score:
                    last_new_idx = next_j
        
        last_idx = last_new_idx if last_new_idx!=None else last_idx    
            


        # # # # # 8. Construct the full matched transcription from first_idx to last_idx AND save the result # # # # #
        full_attempt_text = " ".join([valid_words[k]['word'] for k in range(first_idx, last_idx + 1)])

        results.append({
            'word': exp_word,
            'matched_transcription': full_attempt_text,
            'start_s': valid_words[first_idx]['start'],
            'end_s': valid_words[last_idx]['end'],
            'phonetic_score': best_score,
            'status': 'Found'
        })
        
        last_matched_idx = last_idx

    return results


# -------------------------------------------------------------------------
# 2. Updated detect_words with "Prompting" logic
# -------------------------------------------------------------------------
def detect_words(sub_id, text, sub_path, clean_audio_path, device, model, align_model, metadata, expected_words):
    raw_detected_results_filepath = sub_path + f"/{text}_raw_detected_words_{sub_id}.json"
    
    if os.path.exists(raw_detected_results_filepath):
        print(f"  Found existing aligned results for subject {sub_id}. Loading...")
        word_segments = load_alignment_results(raw_detected_results_filepath)
        
        # print(f"  Found existing aligned results for subject {sub_id}. Removing & Recalculating...")
        # os.remove(raw_detected_results_filepath)

    else:
        print(f"\n--- Processing Subject: {sub_id} ---")
        
        # A. Transcribe using faster-whisper directly (This supports initial_prompt)
        print("Transcribing with vocabulary biasing...")
        prompt_str = " ".join(expected_words)
        segments, info = model.transcribe(
            clean_audio_path, 
            language="el", 
            temperature = 0,
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
            # initial_prompt=prompt_str,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=200,   # ήταν 50 — πολύ aggressive, τώρα 500ms
                speech_pad_ms=200,             # padding γύρω από κάθε speech segment
                threshold=0.1,                 # default=0.5, χαμηλότερο = πιο ευαίσθητο
            ),
        )
        
        # Convert segments to the format WhisperX alignment expects
        segments_list = []
        for s in segments:
            segments_list.append({"start": s.start, "end": s.end, "text": s.text})

        # B. Align using WhisperX and your Greek Wav2Vec2 model
        audio = whisperx.load_audio(clean_audio_path)
        aligned_result = whisperx.align(
            segments_list, 
            align_model, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        # C. Flatten and Match
        word_segments = []
        for segment in aligned_result["segments"]:
            if "words" in segment:
                word_segments.extend(segment["words"])
        
        print("end")
        
        # Normalize detected words for better matching
        # Remove punctuation and lowercase for better fuzzy matching, but keep the original word for output
        for word in word_segments:
            word['word'] = re.sub(r'[^\w\s]', '', word['word'].lower().strip())
        
        save_alignment_results(word_segments, raw_detected_results_filepath)
        
    #########################################
    
    print("Finding word boundaries...")
    scores_matrix = find_scores(word_segments, expected_words)
    
    matched_data = find_word_boundaries(scores_matrix, expected_words, word_segments, min_score=75)
    
    # D. Save to Excel
    df_scores = pd.DataFrame(scores_matrix)
    save_path_scores = os.path.join(sub_path, f"scores_matrix_{sub_id}.xlsx")
    os.makedirs(os.path.dirname(save_path_scores), exist_ok=True)
    df_scores.to_excel(save_path_scores, index=False)
    
    
    df_words = pd.DataFrame(matched_data)
    save_path_words = os.path.join(sub_path, f"{text}_detected_words_v3_{sub_id}.xlsx")
    os.makedirs(os.path.dirname(save_path_words), exist_ok=True)
    df_words.to_excel(save_path_words, index=False)
    
    print(f"Success! Saved to: {save_path_scores}")
    print(f"Success! Saved to: {save_path_words}")
    return {sub_id: matched_data}






# def find_word_boundaries(transcript_words, expected_words, similarity_threshold=60):
#     results = []
#     current_transcript_idx = 0
#     valid_words = [w for w in transcript_words if 'start' in w and 'end' in w]

#     for expected_word in expected_words:
#         target_raw = expected_word.lower().strip()
#         target_phonetic = normalize_greek_phonetic(target_raw)
#         match_found = False
        
#         i = current_transcript_idx
#         while i < len(valid_words):
#             # 1. SLIDING WINDOW (Look ahead for split words like "στα βιδιασμένα")
#             candidates = []
#             w1 = re.sub(r'[^\w\s]', '', valid_words[i]['word'].lower().strip())
#             candidates.append({'text': w1, 'end_idx': i, 'raw_list': [valid_words[i]['word']]})
            
#             if i + 1 < len(valid_words):
#                 w2 = re.sub(r'[^\w\s]', '', valid_words[i+1]['word'].lower().strip())
#                 candidates.append({'text': w1 + w2, 'end_idx': i+1, 'raw_list': [valid_words[i]['word'], valid_words[i+1]['word']]})
                
#             if i + 2 < len(valid_words):
#                 w3 = re.sub(r'[^\w\s]', '', valid_words[i+2]['word'].lower().strip())
#                 candidates.append({'text': w1 + w2 + w3, 'end_idx': i+2, 'raw_list': [valid_words[i]['word'], valid_words[i+1]['word'], valid_words[i+2]['word']]})
            
#             if i + 3 < len(valid_words):
#                 w4 = re.sub(r'[^\w\s]', '', valid_words[i+3]['word'].lower().strip())
#                 candidates.append({'text': w1 + w2 + w3 + w4, 'end_idx': i+3, 'raw_list': [valid_words[i]['word'], valid_words[i+1]['word'], valid_words[i+2]['word'], valid_words[i+3]['word']]})

#             if i + 4 < len(valid_words):
#                 w5 = re.sub(r'[^\w\s]', '', valid_words[i+4]['word'].lower().strip())
#                 candidates.append({'text': w1 + w2 + w3 + w4 + w5, 'end_idx': i+4, 'raw_list': [valid_words[i]['word'], valid_words[i+1]['word'], valid_words[i+2]['word'], valid_words[i+3]['word'], valid_words[i+4]['word']]})

#             if i + 5 < len(valid_words):
#                 w6 = re.sub(r'[^\w\s]', '', valid_words[i+5]['word'].lower().strip())
#                 candidates.append({'text': w1 + w2 + w3 + w4 + w5 + w6, 'end_idx': i+5, 'raw_list': [valid_words[i]['word'], valid_words[i+1]['word'], valid_words[i+2]['word'], valid_words[i+3]['word'], valid_words[i+4]['word'], valid_words[i+5]['word']]})


#             best_cand = None
#             max_sim = 0
#             for cand in candidates:
#                 sim = fuzz.ratio(target_phonetic, normalize_greek_phonetic(cand['text']))
#                 if sim > max_sim:
#                     max_sim = sim
#                     best_cand = cand
                
                
#             # 2. IF MATCH FOUND, PERFORM LOOK-BACK
#             if best_cand:
#                 # Start with the current match
#                 first_idx = i
#                 last_idx = best_cand['end_idx']
                
#                 # Look-back: Check words BEFORE the match for stutters/syllables
#                 # We stop at current_transcript_idx to avoid eating the previous word
#                 for j in range(i - 1, current_transcript_idx - 1, -1):
#                     prev_raw = re.sub(r'[^\w\s]', '', valid_words[j]['word'].lower().strip())
#                     prev_phon = normalize_greek_phonetic(prev_raw)
                    
#                     # If it's a syllable of the target or a very short fragment (<3 chars)
#                     if target_phonetic.startswith(prev_phon) or len(prev_raw) <= 2:
#                         first_idx = j
#                     else:
#                         break
                
#                 # 3. CONSTRUCT RESULT
#                 # Use the first_idx to get the absolute start of the attempt
#                 full_attempt_text = " ".join([w['word'] for w in valid_words[first_idx : last_idx + 1]])
                
#                 results.append({
#                     'word': expected_word,
#                     'matched_transcription': full_attempt_text,
#                     'start_s': valid_words[first_idx]['start'],
#                     'end_s': valid_words[last_idx]['end'],
#                     'phonetic_score': max_sim,
#                     'status': 'Found'
#                 })
                
#                 current_transcript_idx = last_idx + 1
#                 match_found = True
#                 break
            
#             i += 1
            
#         if not match_found:
#             results.append({
#                 'word': expected_word,
#                 'matched_transcription': None,
#                 'start_s': 0,
#                 'end_s': 0,
#                 'phonetic_score': 0,
#                 'status': 'Not Found'
#             })

#     return results