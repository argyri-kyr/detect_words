

import os
import re
import pandas as pd
import whisperx
from rapidfuzz import fuzz
import unicodedata
import numpy as np
from rapidfuzz.distance import Jaro

from save_load_model_aligned_results import load_alignment_results, save_alignment_results


# -------------------------------------------------------------------------
# Phonetic normalization 
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# Scoring function to create the matrix of phonetic similarity scores
# -------------------------------------------------------------------------

def find_scores(transcript_words, expected_words):
    valid_words = [w for w in transcript_words if 'start' in w and 'end' in w]
    expected_phonetics = [normalize_greek_phonetic(w.lower().strip()) for w in expected_words]

    # matrix[i][j] = phonetic score of expected_words[i] vs valid_words[j]
    matrix = []
    jaro_matrix = []

    # Append header row with valid words
    row = ['Expected\Detected'] + [w['word'] for w in valid_words]
    matrix.append(row)
    jaro_matrix.append(row)
    
    for index, exp_phon in enumerate(expected_phonetics):
        row = []
        row.append(expected_words[index])
        jaro_row = []
        jaro_row.append(expected_words[index])  # Header for Jaro matrix
        for det_word in valid_words:
            det_text = re.sub(r'[^\w\s]', '', det_word['word'].lower().strip())
            score = fuzz.ratio(exp_phon, normalize_greek_phonetic(det_text))
            jaro_score = Jaro.similarity(exp_phon, normalize_greek_phonetic(det_text)) * 100
            row.append(score)
            jaro_row.append(jaro_score)
        matrix.append(row)
        jaro_matrix.append(jaro_row)

    return matrix, jaro_matrix


# -------------------------------------------------------------------------
# Merge validation helper
# -------------------------------------------------------------------------

def merge_score(expected_word, detected_words_list):
    """
    Compute the phonetic similarity between an expected word and
    a list of detected words merged together (concatenated).
    
    Example: merge_score("πλυντήρια", ["πλην", "τήρια"]) → high score
    """
    merged_text = "".join([w['word'] for w in detected_words_list])
    return fuzz.ratio(
        normalize_greek_phonetic(expected_word),
        normalize_greek_phonetic(merged_text)
    )


# -------------------------------------------------------------------------
# DP Alignment — replaces find_word_boundaries
# -------------------------------------------------------------------------

def find_word_boundaries_dp(matrix, expected_words, valid_words, min_score=75):
    """
    DP-based alignment that replaces the greedy find_word_boundaries.
    
    How it works:
      dp[i][j] = best total score when we've aligned expected_words[0..i-1]
                  using valid_words[0..j-1]
    
    At each cell (i, j) we consider 5 options:
      A. 1:1 match     — expected[i] ↔ detected[j]
      B. 1:2 merge      — expected[i] ↔ detected[j-1] + detected[j]
      C. 1:3 merge      — expected[i] ↔ detected[j-2] + detected[j-1] + detected[j]
      D. Skip detected  — detected[j] is noise/stutter, skip it
      E. Skip expected   — child didn't say this word, penalty
    
    The alignment is monotonic BY CONSTRUCTION — no transition goes backwards.
    This solves the repeated-word problem (e.g. "το" matched to wrong instance).
    
    Parameters:
      matrix:         score matrix from find_scores (row 0 = header, col 0 = labels)
      expected_words: list of expected words (from the text)
      valid_words:    list of detected word dicts (with 'word', 'start', 'end')
      min_score:      minimum phonetic score to accept a match
    
    Returns:
      list of dicts, same format as original find_word_boundaries
    """
    
    T = len(expected_words)   # number of expected (target) words
    D = len(valid_words)      # number of detected words
    
    NEG_INF = -1e9
    
    # Penalties — tune these based on your data
    SKIP_DETECTED_PENALTY = 0.0     # no cost to skip a detected word (noise)
    SKIP_EXPECTED_PENALTY = -20.0   # cost when child didn't say a word
    MERGE_BONUS = 5.0               # small bonus for valid merges
    
    
    # # # # # 1. Extract the score matrix into a numpy array # # # # #
    # matrix[0] = header row, matrix[i+1][0] = expected word label
    # matrix[i+1][j+1] = phonetic score of expected_words[i] vs valid_words[j]
    
    scores = np.zeros((T, D), dtype=np.float64)
    for i in range(T):
        for j in range(D):
            scores[i, j] = float(matrix[i + 1][j + 1])
    
    
    # # # # # 2. Build the DP table # # # # #
    # dp[i][j] = best alignment score for expected[0..i-1] using detected[0..j-1]
    
    dp = np.full((T + 1, D + 1), NEG_INF, dtype=np.float64)
    
    # Backtrack table: stores (prev_i, prev_j, action_type, merge_count)
    # action_type: 'match', 'skip_det', 'skip_exp'
    # merge_count: how many detected words are consumed (1, 2, or 3)
    bt = [[None for _ in range(D + 1)] for _ in range(T + 1)]
    
    # Base case: 0 expected words aligned to 0..j detected words (all skipped)
    dp[0][0] = 0.0
    for j in range(1, D + 1):
        dp[0][j] = dp[0][j - 1] + SKIP_DETECTED_PENALTY
        bt[0][j] = (0, j - 1, 'skip_det', 0)
    
    
    # # # # # 3. Fill the DP table # # # # #
    
    for i in range(1, T + 1):
        # Allow skipping expected words from the start
        dp[i][0] = dp[i - 1][0] + SKIP_EXPECTED_PENALTY
        bt[i][0] = (i - 1, 0, 'skip_exp', 0)
        
        for j in range(1, D + 1):
            best = NEG_INF
            best_bt = None
            
            if(i==175 and j==165):
                print()
            # --- Option A: 1:1 match (expected[i-1] ↔ detected[j-1]) ---
            s = scores[i - 1, j - 1]
            if s >= min_score:
                candidate = dp[i - 1][j - 1] + s
                if candidate > best:
                    best = candidate
                    best_bt = (i - 1, j - 1, 'match', 1)
            
            
            # --- Option B: 1:2 merge (expected[i-1] ↔ detected[j-2] + detected[j-1]) ---
            if j >= 2:
                merged = merge_score(
                    expected_words[i - 1], 
                    [valid_words[j - 2], valid_words[j - 1]]
                )
                if merged >= min_score:
                    candidate = dp[i - 1][j - 2] + merged + MERGE_BONUS
                    if candidate > best:
                        best = candidate
                        best_bt = (i - 1, j - 2, 'match', 2)
            
            
            # --- Option C: 1:3 merge (expected[i-1] ↔ detected[j-3..j-1]) ---
            if j >= 3:
                merged = merge_score(
                    expected_words[i - 1],
                    [valid_words[j - 3], valid_words[j - 2], valid_words[j - 1]]
                )
                if merged >= min_score:
                    candidate = dp[i - 1][j - 3] + merged + MERGE_BONUS
                    if candidate > best:
                        best = candidate
                        best_bt = (i - 1, j - 3, 'match', 3)
            
            
            # --- Option D: Skip detected word (noise/stutter) ---
            candidate = dp[i][j - 1] + SKIP_DETECTED_PENALTY
            if candidate > best:
                best = candidate
                best_bt = (i, j - 1, 'skip_det', 0)
            
            
            # --- Option E: Skip expected word (child didn't say it) ---
            candidate = dp[i - 1][j] + SKIP_EXPECTED_PENALTY
            if candidate > best:
                best = candidate
                best_bt = (i - 1, j, 'skip_exp', 0)
            
            
            dp[i][j] = best
            bt[i][j] = best_bt
    
    
    # # # # # 4. Backtrack to recover the alignment # # # # #
    # alignment[i] = (first_det_idx, last_det_idx, score) for expected word i
    # or None if the word was skipped
    
    alignment = {}  # expected_word_idx → (first_idx, last_idx, phonetic_score)
    
    i, j = T, D
    while i > 0 or j > 0:
        prev_i, prev_j, action, merge_count = bt[i][j]
        
        if action == 'match':
            # The matched detected words are from prev_j to prev_j + merge_count - 1
            first_idx = prev_j
            last_idx = prev_j + merge_count - 1
            
            # Compute the phonetic score for this match
            if merge_count == 1:
                match_score = scores[prev_i, first_idx]
            else:
                # For merges, use the merge_score (phonetic concatenation score)
                match_score = merge_score(
                    expected_words[prev_i],
                    valid_words[first_idx : last_idx + 1]
                )
            
            alignment[prev_i] = (first_idx, last_idx, match_score)
        
        # skip_det and skip_exp: no entry needed
        
        i, j = prev_i, prev_j
    
    
    # # # # # 5. Build the results in the same format as original # # # # #
    
    results = []
    
    for i, exp_word in enumerate(expected_words):
        if i in alignment:
            first_idx, last_idx, match_score = alignment[i]
            
            # Construct the full matched transcription
            matched_words = [valid_words[k]['word'] for k in range(first_idx, last_idx + 1)]
            full_attempt_text = " ".join(matched_words)
            
            results.append({
                'word': exp_word,
                'matched_transcription': full_attempt_text,
                'start_s': valid_words[first_idx]['start'],
                'end_s': valid_words[last_idx]['end'],
                'phonetic_score': round(match_score, 2),
                'status': 'Found'
            })
        else:
            results.append({
                'word': exp_word,
                'matched_transcription': None,
                'start_s': 0,
                'end_s': 0,
                'phonetic_score': 0,
                'status': 'Not Found'
            })
    
    return results

# -------------------------------------------------------------------------
# Post-processing: Rescue 'Not Found' words via best-in-window fallback
# Post-processing: Rescue contested words — interpolate timestamps from neighbors
# -------------------------------------------------------------------------
def rescue_not_found_contested_words(results, valid_words, scores_matrix, min_score_window=45, min_score_contested=65):
    """
    Two-stage rescue for 'Not Found' words:
    1. Best-in-window: Search between neighboring matched words (conservative)
    2. Best-globally: If still not found, search entire matrix (aggressive fallback)
    
    Parameters:
        results: List of word detection results from DP alignment
        valid_words: List of detected word dicts with timestamps
        scores_matrix: Phonetic similarity matrix from find_scores
        min_score_window: Minimum score for local window search (default: 45)
        min_score_contested: Minimum score for contested word rescue (default: 65)
    
    Returns:
        Updated results with rescued words marked as 'Mispronounced' or 'Estimated'
    """
    
    # Build a numpy scores array
    T = len(results)
    D = len(valid_words)
    scores = np.zeros((T, D), dtype=np.float64)
    for i in range(T):
        for j in range(D):
            scores[i, j] = float(scores_matrix[i + 1][j + 1])

    # Build a position index map from (start, end) → index in valid_words
    time_to_idx = {}
    for j, w in enumerate(valid_words):
        time_to_idx[(round(w['start'], 3), round(w['end'], 3))] = j

    for i, r in enumerate(results):
        # Skip already matched words
        if r['status'] in ('Found', 'Mispronounced'):
            continue

        # Skip very short words (likely articles/particles)
        if len(r['word']) <= 3:
            continue

        # --- STAGE 1: Search within the local window ---
        # Find the boundary before this word (last matched word's position)
        prev_idx = -1
        for prev in range(i - 1, -1, -1):
            if results[prev]['status'] in ('Found', 'Mispronounced'):
                key = (round(results[prev]['start_s'], 3), round(results[prev]['end_s'], 3))
                idx = time_to_idx.get(key, None)
                if idx is not None:
                    prev_idx = idx
                break

        # Find the boundary after this word (next matched word's position)
        next_idx = D
        for nxt in range(i + 1, len(results)):
            if results[nxt]['status'] in ('Found', 'Mispronounced'):
                key = (round(results[nxt]['start_s'], 3), round(results[nxt]['end_s'], 3))
                idx = time_to_idx.get(key, None)
                if idx is not None:
                    next_idx = idx
                break

        # Find the best-scoring detected word within the index window
        best_j = None
        best_s = -1.0

        for j in range(prev_idx + 1, next_idx):
            s = scores[i, j]
            if s > best_s:
                best_s = s
                best_j = j

        # Assign if above the threshold
        if best_j is not None and best_s >= min_score_window:
            results[i] = {
                'word': r['word'],
                'matched_transcription': valid_words[best_j]['word'],
                'start_s': valid_words[best_j]['start'],
                'end_s': valid_words[best_j]['end'],
                'phonetic_score': round(best_s, 2),
                'status': 'Mispronounced'
            }
            continue  # Successfully rescued, move to next word
        
        # --- STAGE 2: If still not found, search globally (contested words) ---
        # Double-check that Stage 1 didn't rescue it
        if results[i]['status'] == 'Not Found':
            # Vectorized search: find best score across ALL detected words
            row = scores[i, :]  # Get entire row for this expected word
            best_j_global = np.argmax(row)  # Index of maximum
            best_s_global = row[best_j_global]  # Maximum value

            if best_s_global >= min_score_contested:
                # Use the best match's actual timestamps
                results[i] = {
                    'word': r['word'],
                    'matched_transcription': valid_words[best_j_global]['word'],
                    'start_s': valid_words[best_j_global]['start'],
                    'end_s': valid_words[best_j_global]['end'],
                    'phonetic_score': round(best_s_global, 2),
                    'status': 'Estimated'
                }

    return results


# -------------------------------------------------------------------------
# Optional: Post-processing to capture reading attempts
# -------------------------------------------------------------------------

def expand_with_reading_attempts(results, valid_words, scores_matrix, jaro_score_matrix, min_score=60):
    """
    Post-processing step after DP alignment — expands each matched word's time
    range to include reading attempts (autocorrections, syllable fragments) that
    the DP couldn't capture because it only assigns one detected word per expected.

    Two mechanisms:
      A. LOOK-BACK / LOOK-AHEAD (autocorrection attempts)
         The child said the full word twice, e.g. 'παυσιμονο' then 'παυσιπονο'.
         The DP matched one; we expand to include the other if it scores high
         against the same expected word AND scores higher here than against the
         neighbouring expected word (contest).

      B. BACKWARD / FORWARD SYLLABLE SEARCH (syllable attempts)
         The child broke the word into syllables before or after saying it whole.
         We merge adjacent gap words and check if the merged string phonetically
         resembles the expected word.
         Hard boundary: we never enter territory the contest assigned to a neighbour.

    Parameters:
      results:           output from find_word_boundaries_dp / rescue step
      valid_words:       detected word dicts with 'word', 'start', 'end'
      scores_matrix:     fuzz.ratio phonetic score matrix from find_scores
      jaro_score_matrix: Jaro similarity matrix from find_scores
      min_score:         minimum score for a gap word to count as an attempt
    """

    # --- 1. Build matched_ranges ---
    # Map each result index → (first_det_idx, last_det_idx) in valid_words.
    # This is what lets us know the hard core boundaries from the DP.
    matched_ranges = []
    for r in results:
        if r['status'] in ('Found', 'Mispronounced'):
            first = last = None
            for k, w in enumerate(valid_words):
                if abs(w['start'] - r['start_s']) < 0.001 and first is None:
                    first = k
                if abs(w['end'] - r['end_s']) < 0.001:
                    last = k
            matched_ranges.append((first, last))
        else:
            matched_ranges.append((None, None))


    # --- 2. Score helper ---
    # Take max of fuzz.ratio and Jaro so we don't miss near-matches that one
    # metric catches but the other doesn't.
    def get_score(exp_idx, j):
        s  = float(scores_matrix[exp_idx + 1][j + 1])
        js = float(jaro_score_matrix[exp_idx + 1][j + 1])
        return max(s, js)


    # --- 3. Main loop ---
    for idx, r in enumerate(results):
        if r['status'] not in ('Found', 'Mispronounced'):
            continue

        first_idx, last_idx = matched_ranges[idx]
        if first_idx is None:
            continue

        exp_word = r['word']

        # Outer boundaries: the last detected index owned by the previous matched
        # word, and the first detected index owned by the next matched word.
        # We never cross these — they are the DP's hard territory lines.
        prev_boundary = -1
        for prev in range(idx - 1, -1, -1):
            if matched_ranges[prev][1] is not None:
                prev_boundary = matched_ranges[prev][1]
                break

        next_boundary = len(valid_words)
        for nxt in range(idx + 1, len(results)):
            if matched_ranges[nxt][0] is not None:
                next_boundary = matched_ranges[nxt][0]
                break

        # Find the neighbouring expected word indices for the contest comparisons
        left_exp_idx = None
        for prev in range(idx - 1, -1, -1):
            if matched_ranges[prev][0] is not None:
                left_exp_idx = prev
                break

        right_exp_idx = None
        for nxt in range(idx + 1, len(results)):
            if matched_ranges[nxt][0] is not None:
                right_exp_idx = nxt
                break

        new_first = first_idx
        new_last  = last_idx


        # --- 3a. LOOK-BACK: walk leftward through the gap ---
        # For each gap word, contest against the left neighbour expected word.
        # Stop as soon as the left neighbour wins — everything further left
        # belongs to it, so there's no point continuing.
        # Record where the left neighbour first wins as contest_boundary_left,
        # which becomes the hard stop for the syllable search below.

        contest_boundary_left = prev_boundary  # default: no contest, full gap is ours

        for prev_j in range(first_idx - 1, prev_boundary, -1):
            my_score    = get_score(idx, prev_j)
            rival_score = get_score(left_exp_idx, prev_j) if left_exp_idx is not None else 0

            if my_score >= min_score and my_score >= rival_score:
                new_first = prev_j  # we win this slot, extend leftward
            else:
                # Left neighbour wins here — record the hard boundary and stop.
                # We must not enter this slot or anything further left.
                contest_boundary_left = prev_j
                break


        # --- 3b. LOOK-AHEAD: walk rightward through the gap ---
        # Symmetric to look-back: contest each gap word against the right neighbour.
        # Stop as soon as the right neighbour wins and record contest_boundary_right.

        contest_boundary_right = next_boundary  # default: no contest, full gap is ours

        for next_j in range(last_idx + 1, next_boundary):
            my_score    = get_score(idx, next_j)
            rival_score = get_score(right_exp_idx, next_j) if right_exp_idx is not None else 0

            if my_score >= min_score and my_score >= rival_score:
                new_last = next_j  # we win this slot, extend rightward
            else:
                # Right neighbour wins here — record and stop.
                contest_boundary_right = next_j
                break


        # --- 3c. BACKWARD SYLLABLE SEARCH ---
        # The child may have sounded out syllables before saying the full word,
        # e.g. 'παυ', 'σι', 'παυσιπονο'. We check if merging consecutive gap
        # words leftward produces something phonetically close to the expected word.
        #
        # Hard stop: contest_boundary_left (not prev_boundary) — we must not
        # enter any slot the left neighbour won in the contest above.

        if new_first > contest_boundary_left + 1:
            best_backward = new_first
            for prev_j in range(new_first - 1, contest_boundary_left, -1):
                # Merge everything from prev_j up to (but not including) new_first
                merged_text = "".join([valid_words[k]['word'] for k in range(prev_j, new_first)])
                ms = fuzz.ratio(
                    normalize_greek_phonetic(exp_word),
                    normalize_greek_phonetic(merged_text)
                )
                if ms >= min_score:
                    best_backward = prev_j  # this merge qualifies, keep extending
            new_first = best_backward


        # --- 3d. FORWARD SYLLABLE SEARCH ---
        # Symmetric: child may have re-read in syllables after the full word.
        # Hard stop: contest_boundary_right instead of next_boundary.

        if new_last < contest_boundary_right - 1:
            best_forward = new_last
            for next_j in range(new_last + 1, contest_boundary_right):
                # Merge everything from (new_last + 1) up to next_j inclusive
                merged_text = "".join([valid_words[k]['word'] for k in range(new_last + 1, next_j + 1)])
                ms = fuzz.ratio(
                    normalize_greek_phonetic(exp_word),
                    normalize_greek_phonetic(merged_text)
                )
                if ms >= min_score:
                    best_forward = next_j  # this merge qualifies, keep extending
            new_last = best_forward


        # --- 4. Write back if we expanded ---
        if new_first != first_idx or new_last != last_idx:
            full_text = " ".join([valid_words[k]['word'] for k in range(new_first, new_last + 1)])
            results[idx]['matched_transcription'] = full_text
            results[idx]['start_s'] = valid_words[new_first]['start']
            results[idx]['end_s']   = valid_words[new_last]['end']
            # Update matched_ranges so the next word's boundary calculation
            # sees our expanded territory, not just the DP core.
            matched_ranges[idx] = (new_first, new_last)

    return results



# -------------------------------------------------------------------------------------------------------------
# Main function to detect words for a subject — includes transcription, alignment, scoring, and saving results
# -------------------------------------------------------------------------------------------------------------
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
            initial_prompt=prompt_str,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=200,   # min_silence_duration_ms=50
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
        
        
        # Normalize detected words for better matching
        # Remove punctuation and lowercase for better fuzzy matching, but keep the original word for output
        for word in word_segments:
            word['word'] = re.sub(r'[^\w\s]', '', word['word'].lower().strip())
        
        save_alignment_results(word_segments, raw_detected_results_filepath)
        
    #########################################
    
    print("Finding word boundaries...")
    scores_matrix, jaro_score_matrix = find_scores(word_segments, expected_words)
    
    # Filter to only words with timestamps (same filter as find_scores uses internally)
    valid_words = [w for w in word_segments if 'start' in w and 'end' in w]

    # DP alignment
    matched_data = find_word_boundaries_dp(scores_matrix, expected_words, valid_words, min_score=75)

    # Strategy 4: Rescue 'Not Found' words with best-in-window fallback
    matched_data = rescue_not_found_contested_words(matched_data, valid_words, scores_matrix, min_score_window=40, min_score_contested=65)

    # Expand matched words to include reading attempts (syllables, autocorrections)
    matched_data = expand_with_reading_attempts(matched_data, valid_words, scores_matrix, jaro_score_matrix, min_score=60)

    
    # D. Save to Excel
    df_scores = pd.DataFrame(scores_matrix)
    save_path_scores = os.path.join(sub_path, f"scores_matrix_{sub_id}.xlsx")
    os.makedirs(os.path.dirname(save_path_scores), exist_ok=True)
    df_scores.to_excel(save_path_scores, index=False)

    df_jaro_scores = pd.DataFrame(jaro_score_matrix)
    save_path_jaro_scores = os.path.join(sub_path, f"jaro_scores_matrix_{sub_id}.xlsx")
    os.makedirs(os.path.dirname(save_path_jaro_scores), exist_ok=True)
    df_jaro_scores.to_excel(save_path_jaro_scores, index=False)


    df_words = pd.DataFrame(matched_data)
    save_path_words = os.path.join(sub_path, f"{text}_detected_words_v3_{sub_id}.xlsx")
    os.makedirs(os.path.dirname(save_path_words), exist_ok=True)
    df_words.to_excel(save_path_words, index=False)
    
    print(f"Success! Saved to: {save_path_scores}")
    print(f"Success! Saved to: {save_path_words}")
    return {sub_id: matched_data}





