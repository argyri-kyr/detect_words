# detect_words_hierarchical.py
#
# Hierarchical transcription — replaces detect_words() in detect_words_faster.py
# for subjects where forced alignment fails.
#
# PASS 1 — full audio, all 24 words as prompt → reliable segment boundaries
# PASS 2 — per segment: re-transcribe slice with focused prompt (only words
#           expected in that window) + word_timestamps=True → per-word times
#           Words assigned as secondary to a segment are included in its prompt
#           so cross-segment words (e.g. αλάτια spanning two segments) can be
#           detected in both.
# MERGE  — for words spanning two segments: take start from first occurrence,
#           end from last — but only if both matches score > MIN_SECONDARY_SCORE.
# REFINE — per-word re-transcription in a tight window using prev.end_s as
#           left boundary. Skipped for words that already have score=100 and
#           a tight window (< REFINE_SKIP_THRESHOLD seconds).
#
# No pass 3 — pass 2 with focused prompts is sufficient.
#
# Drop-in for detect_words() — same return format, no align_model needed.

import os
import re
import json
import tempfile
import unicodedata
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from rapidfuzz import fuzz

from save_load_model_aligned_results import save_alignment_results


# -----------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------

PASS1_VAD_PARAMS = dict(
    min_silence_duration_ms=200,
    speech_pad_ms=200,
    threshold=0.1,
)

PASS2_VAD_PARAMS = dict(
    min_silence_duration_ms=200,
    speech_pad_ms=100,
    threshold=0.1,
)

# Minimum phonetic score to accept a primary match
MIN_SCORE = 65

# Minimum phonetic score to accept a secondary (cross-segment) match.
MIN_SECONDARY_SCORE = 70

# Refinement pass settings
REFINE_WIDE_THRESHOLD = 15.0   # seconds — above this use 3-word prompt
REFINE_SKIP_THRESHOLD = 8.0    # seconds — skip refinement if window < this AND score=100

# Examiner speech tokens to filter out before matching.
EXAMINER_PHRASES = [
    'δεν', 'καταλαβαίνω', 'καταλαβαινω', 'πειράζει', 'πειραζει',
    'προχώρα', 'προχωρα', 'επόμενη', 'επομενη', 'πες', 'πάμε',
]

SAMPLE_RATE = 16000


# -----------------------------------------------------------------------
# Phonetic normalization
# -----------------------------------------------------------------------
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize_greek_phonetic(text):
    if not text:
        return ""
    text = remove_accents(text.lower().strip())
    text = text.replace('αϊ', 'αη').replace('εϊ', 'εη')
    text = text.replace('αυ', 'αβ').replace('ευ', 'εβ').replace('ηυ', 'η')
    text = text.replace('αι', 'ε').replace('ου', 'U_PLACE')
    text = re.sub(r'ει|οι|υι', 'η', text)
    text = text.replace('αι', 'ε')
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = text.replace('υ', 'η').replace('ω', 'ο').replace('η', 'ι')
    text = text.replace('U_PLACE', 'ου')
    return text


def phonetic_score(a, b):
    return fuzz.ratio(normalize_greek_phonetic(a), normalize_greek_phonetic(b))


# -----------------------------------------------------------------------
# Examiner speech filter
# -----------------------------------------------------------------------
def is_examiner_token(token_word):
    w = remove_accents(token_word.lower().strip())
    for phrase in EXAMINER_PHRASES:
        if w == remove_accents(phrase):
            return True
    return False


def filter_examiner_tokens(word_tokens):
    filtered = []
    for tok in word_tokens:
        w = tok.get('word', '')
        if is_examiner_token(w):
            print(f"    ⚠️  Dropping examiner token: '{w}' at {tok.get('start', '?'):.1f}s")
            continue
        has_greek = any('\u0370' <= c <= '\u03ff' or '\u1f00' <= c <= '\u1fff' for c in w)
        if w and not has_greek:
            print(f"    ⚠️  Dropping non-Greek hallucination: '{w}'")
            continue
        filtered.append(tok)
    return filtered


# -----------------------------------------------------------------------
# Audio slicing
# -----------------------------------------------------------------------
def slice_audio_to_tempfile(audio_path, start_s, end_s):
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    slice_wave = waveform[start_sample:end_sample]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, slice_wave, sr)
    tmp.close()
    return tmp.name


# -----------------------------------------------------------------------
# Pass 1: full audio → segment boundaries
# -----------------------------------------------------------------------
def run_pass1(audio_path, model, expected_words, cache_path):
    if os.path.exists(cache_path):
        print("  [Pass 1] Loading cached segments...")
        with open(cache_path, encoding="utf-8") as f:
            segments = json.load(f)
        if segments and 'text' in segments[0]:
            for seg in segments:
                print(f"    [{seg['start']:.1f}–{seg['end']:.1f}] {seg['text'].strip()}")
            return segments
        print("  [Pass 1] Cache is old format — re-transcribing.")

    print("  [Pass 1] Transcribing full audio...")
    prompt_str = " ".join(expected_words)
    segments_gen, _ = model.transcribe(
        audio_path,
        language="el",
        temperature=0,
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        initial_prompt=prompt_str,
        vad_filter=True,
        vad_parameters=PASS1_VAD_PARAMS,
    )

    segments = []
    for s in segments_gen:
        segments.append({"start": s.start, "end": s.end, "text": s.text})

    print(f"  [Pass 1] Found {len(segments)} segments:")
    for seg in segments:
        print(f"    [{seg['start']:.1f}–{seg['end']:.1f}] {seg['text'].strip()}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return segments


# -----------------------------------------------------------------------
# Assign expected words to segments (with optional secondary)
# -----------------------------------------------------------------------
def assign_words_to_segments(segments, expected_words):
    score_matrix = []
    for exp_word in expected_words:
        seg_scores = []
        for seg in segments:
            seg_text = re.sub(r'[^\w\s]', '', seg['text'].lower().strip())
            full_score = phonetic_score(exp_word, seg_text)
            best_word_score = max(
                (phonetic_score(exp_word, w) for w in seg_text.split()),
                default=0,
            )
            seg_scores.append(max(full_score, best_word_score))
        score_matrix.append(seg_scores)

    assignments = []
    for i, exp_word in enumerate(expected_words):
        seg_scores = score_matrix[i]
        primary_idx = int(np.argmax(seg_scores))
        primary_score = seg_scores[primary_idx]

        secondary_idx = None
        for candidate_seg in [primary_idx - 1, primary_idx + 1]:
            if candidate_seg < 0 or candidate_seg >= len(segments):
                continue

            candidate_score = seg_scores[candidate_seg]
            if candidate_score < 40:
                continue

            neighbor_scores = []
            if i - 1 >= 0:
                neighbor_scores.append(score_matrix[i - 1][candidate_seg])
            if i + 1 < len(expected_words):
                neighbor_scores.append(score_matrix[i + 1][candidate_seg])

            current_word_wins = all(
                candidate_score >= ns for ns in neighbor_scores
            )

            if current_word_wins:
                secondary_idx = candidate_seg
                break

        assignments.append({
            "word": exp_word,
            "primary_seg_idx": primary_idx,
            "secondary_seg_idx": secondary_idx,
            "score": primary_score,
        })

    # Enforce sequential constraint
    for i in range(1, len(assignments)):
        if assignments[i]["primary_seg_idx"] < assignments[i - 1]["primary_seg_idx"]:
            assignments[i]["primary_seg_idx"] = assignments[i - 1]["primary_seg_idx"]
            assignments[i]["secondary_seg_idx"] = None
            assignments[i]["score"] = 0

    return assignments, score_matrix


# -----------------------------------------------------------------------
# Transcribe a slice with focused prompt
# -----------------------------------------------------------------------
def transcribe_slice(audio_path, model, slice_start, slice_end, words_in_slice, label=""):
    tmp_path = slice_audio_to_tempfile(audio_path, slice_start, slice_end)

    try:
        prompt_str = " ".join(words_in_slice)
        segments_gen, _ = model.transcribe(
            tmp_path,
            language="el",
            temperature=0,
            condition_on_previous_text=False,
            no_repeat_ngram_size=2,
            initial_prompt=prompt_str,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=PASS2_VAD_PARAMS,
            no_speech_threshold=0.8,
            suppress_blank=True,
        )

        word_tokens = []
        for seg in segments_gen:
            if seg.words:
                for w in seg.words:
                    word_tokens.append({
                        "word": re.sub(r'[^\w\s]', '', w.word.lower().strip()),
                        "start": round(w.start + slice_start, 3),
                        "end": round(w.end + slice_start, 3),
                    })

        word_tokens = filter_examiner_tokens(word_tokens)

        token_str = ' '.join(t['word'] for t in word_tokens)
        print(f"    {label} [{slice_start:.1f}–{slice_end:.1f}] → {len(word_tokens)} tokens: {token_str}")

        return word_tokens

    finally:
        os.remove(tmp_path)


# -----------------------------------------------------------------------
# Match tokens to expected words
# -----------------------------------------------------------------------
def match_tokens_to_words(word_tokens, expected_words):
    valid_tokens = [t for t in word_tokens if t.get('word')]

    if not valid_tokens:
        return [
            {"word": w, "start_s": None, "end_s": None,
             "matched_transcription": None, "phonetic_score": 0}
            for w in expected_words
        ]

    results = []
    last_matched_idx = None

    for exp_word in expected_words:
        exp_phon = normalize_greek_phonetic(exp_word)

        best_score = -1
        best_first_idx = 0
        best_last_idx = 0

        search_start = last_matched_idx if last_matched_idx is not None else 0

        for j in range(search_start, len(valid_tokens)):
            s1 = fuzz.ratio(exp_phon, normalize_greek_phonetic(valid_tokens[j]['word']))
            if s1 > best_score:
                best_score = s1
                best_first_idx = j
                best_last_idx = j

            if j + 1 < len(valid_tokens):
                merged2 = valid_tokens[j]['word'] + valid_tokens[j + 1]['word']
                s2 = fuzz.ratio(exp_phon, normalize_greek_phonetic(merged2))
                if s2 > best_score:
                    best_score = s2
                    best_first_idx = j
                    best_last_idx = j + 1

            if j + 2 < len(valid_tokens):
                merged3 = (valid_tokens[j]['word']
                           + valid_tokens[j + 1]['word']
                           + valid_tokens[j + 2]['word'])
                s3 = fuzz.ratio(exp_phon, normalize_greek_phonetic(merged3))
                if s3 > best_score:
                    best_score = s3
                    best_first_idx = j
                    best_last_idx = j + 2

        matched_text = " ".join(
            valid_tokens[k]['word'] for k in range(best_first_idx, best_last_idx + 1)
        )

        results.append({
            "word": exp_word,
            "start_s": valid_tokens[best_first_idx]['start'],
            "end_s": valid_tokens[best_last_idx]['end'],
            "matched_transcription": matched_text,
            "phonetic_score": best_score,
        })

        last_matched_idx = best_last_idx

    return results


# -----------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------
def detect_words_hierarchical(
    sub_id,
    text,
    sub_path,
    clean_audio_path,
    device,
    model,
    expected_words,
):
    # ----------------------------------------------------------------
    # Pass 1 — full audio → segment boundaries
    # ----------------------------------------------------------------
    pass1_cache = os.path.join(sub_path, f"{text}_pass1_segments_{sub_id}.json")
    segments = run_pass1(clean_audio_path, model, expected_words, pass1_cache)

    if not segments:
        print(f"  [{sub_id}] ⚠️  Pass 1 returned no segments.")
        return {sub_id: []}

    # ----------------------------------------------------------------
    # Assign expected words to segments
    # ----------------------------------------------------------------
    assignments, score_matrix = assign_words_to_segments(segments, expected_words)

    print("\n  [Assignment plan]")
    for a in assignments:
        sec = f" + seg {a['secondary_seg_idx']}" if a['secondary_seg_idx'] is not None else ""
        print(f"    {a['word']:<25} → seg {a['primary_seg_idx']}{sec}  (score={a['score']:.0f})")

    # ----------------------------------------------------------------
    # Build per-segment word lists for pass 2 prompts
    # ----------------------------------------------------------------
    seg_primary_words = {}
    seg_all_words = {}

    for a in assignments:
        p = a['primary_seg_idx']
        seg_primary_words.setdefault(p, []).append(a['word'])
        seg_all_words.setdefault(p, []).append(a['word'])

        if a['secondary_seg_idx'] is not None:
            s = a['secondary_seg_idx']
            seg_all_words.setdefault(s, [])
            if a['word'] not in seg_all_words[s]:
                seg_all_words[s].append(a['word'])

    # ----------------------------------------------------------------
    # Pass 2 — re-transcribe each segment with focused prompt
    # ----------------------------------------------------------------
    print("\n  [Pass 2] Re-transcribing each segment with focused prompt...")
    seg_word_tokens = {}

    for seg_idx in sorted(seg_all_words.keys()):
        seg = segments[seg_idx]
        words_for_prompt = seg_all_words[seg_idx]

        tokens = transcribe_slice(
            clean_audio_path,
            model,
            seg['start'],
            seg['end'],
            words_for_prompt,
            label=f"[Pass 2 seg {seg_idx}]",
        )

        seg_word_tokens[seg_idx] = tokens

    # ----------------------------------------------------------------
    # Match tokens to expected words per segment
    # ----------------------------------------------------------------
    print("\n  [Matching] Assigning timestamps to expected words...")

    word_results = {}

    for seg_idx in sorted(seg_primary_words.keys()):
        primary_words = seg_primary_words[seg_idx]
        tokens = seg_word_tokens.get(seg_idx, [])

        matched = match_tokens_to_words(tokens, primary_words)

        for m in matched:
            word_results[m['word']] = {
                "start_s": m['start_s'],
                "end_s": m['end_s'],
                "matched_transcription": m['matched_transcription'],
                "phonetic_score": m['phonetic_score'],
                "primary_seg": seg_idx,
            }

    # ----------------------------------------------------------------
    # Cross-segment merge for secondary assignments
    # ----------------------------------------------------------------
    for a in assignments:
        if a['secondary_seg_idx'] is None:
            continue

        word = a['word']
        sec_idx = a['secondary_seg_idx']
        tokens = seg_word_tokens.get(sec_idx, [])

        if not tokens:
            continue

        secondary_match = match_tokens_to_words(tokens, [word])
        if not secondary_match:
            continue

        sec_result = secondary_match[0]

        if sec_result['phonetic_score'] < MIN_SECONDARY_SCORE:
            print(
                f"    ℹ️  '{word}' secondary in seg {sec_idx} scored "
                f"{sec_result['phonetic_score']:.0f} < {MIN_SECONDARY_SCORE} — not merging"
            )
            continue

        existing = word_results.get(word)
        if existing is None or existing['start_s'] is None:
            word_results[word] = {
                "start_s": sec_result['start_s'],
                "end_s": sec_result['end_s'],
                "matched_transcription": sec_result['matched_transcription'],
                "phonetic_score": sec_result['phonetic_score'],
                "primary_seg": sec_idx,
            }
        else:
            merged_start = min(existing['start_s'], sec_result['start_s'])
            merged_end = max(existing['end_s'], sec_result['end_s'])
            merged_text = (
                (existing['matched_transcription'] or '')
                + ' '
                + (sec_result['matched_transcription'] or '')
            ).strip()
            merged_score = max(existing['phonetic_score'], sec_result['phonetic_score'])

            print(
                f"    🔗 '{word}' merged across seg {existing['primary_seg']} + seg {sec_idx}: "
                f"[{merged_start:.1f}–{merged_end:.1f}]"
            )

            word_results[word] = {
                "start_s": merged_start,
                "end_s": merged_end,
                "matched_transcription": merged_text,
                "phonetic_score": merged_score,
                "primary_seg": existing['primary_seg'],
            }

    # ----------------------------------------------------------------
    # Build matched_data in reading order
    # ----------------------------------------------------------------
    matched_data = []

    for exp_word in expected_words:
        res = word_results.get(exp_word)
        if res and res['start_s'] is not None and res['phonetic_score'] >= MIN_SCORE:
            matched_data.append({
                "word": exp_word,
                "matched_transcription": res['matched_transcription'],
                "start_s": round(res['start_s'], 3),
                "end_s": round(res['end_s'], 3),
                "phonetic_score": res['phonetic_score'],
                "status": "Found",
            })
        else:
            matched_data.append({
                "word": exp_word,
                "matched_transcription": res['matched_transcription'] if res else None,
                "start_s": 0,
                "end_s": 0,
                "phonetic_score": res['phonetic_score'] if res else 0,
                "status": "Not Found",
            })

    # ----------------------------------------------------------------
    # Refinement pass
    #
    # Left boundary  = end of previous word (prev.end_s) — tighter than
    #                  prev.start_s, starts after previous word finished.
    # Right boundary = end of next word (next.end_s).
    #
    # Skip if: score == 100 AND window < REFINE_SKIP_THRESHOLD seconds
    #          (word is already correctly placed, no need to re-transcribe)
    #
    # Wide window (> REFINE_WIDE_THRESHOLD): use 3-word prompt for context.
    # Tight window: single-word prompt is enough.
    #
    # Sequential: matched_data[i] updated before i+1 is processed,
    # so each word benefits from the refined timestamp of the previous one.
    # ----------------------------------------------------------------
    print(f"\n  [Refinement pass] Re-transcribing words that need improvement...")

    audio_duration = segments[-1]['end']

    for i, row in enumerate(matched_data):
        word = row['word']

        # Left boundary: end of previous word (or 0 for first word)
        if i == 0:
            left = 0.0
        else:
            prev_end = matched_data[i - 1]['end_s']
            left = prev_end if prev_end and prev_end > 0 else 0.0

        # Right boundary: end of next word (or end of audio for last word)
        if i == len(matched_data) - 1:
            right = audio_duration
        else:
            next_end = matched_data[i + 1]['end_s']
            right = next_end if next_end and next_end > 0 else audio_duration

        # Safety: ensure minimum window size
        if right - left < 1.0:
            right = left + 3.0

        window_dur = right - left

        # Skip refinement if word already has perfect score and tight window
        # — it's correctly placed and re-transcribing risks breaking it
        if row['phonetic_score'] >= 100 and window_dur < REFINE_SKIP_THRESHOLD:
            print(f"    ✅ '{word}': score=100 window={window_dur:.1f}s — skipping refinement")
            continue

        # Build prompt
        if window_dur > REFINE_WIDE_THRESHOLD:
            prev_word = expected_words[i - 1] if i > 0 else ""
            next_word = expected_words[i + 1] if i < len(expected_words) - 1 else ""
            prompt_words = [w for w in [prev_word, word, next_word] if w]
        else:
            prompt_words = [word]

        tokens = transcribe_slice(
            clean_audio_path,
            model,
            left,
            right,
            prompt_words,
            label=f"[Refine {i:02d} {word}]",
        )

        if not tokens:
            print(f"    ⚠️  No tokens for '{word}' — keeping original")
            continue

        refined = match_tokens_to_words(tokens, [word])

        if not refined or refined[0]['start_s'] is None:
            print(f"    ⚠️  No match for '{word}' — keeping original")
            continue

        r = refined[0]

        if r['phonetic_score'] < MIN_SCORE:
            print(
                f"    ⚠️  '{word}' refined score {r['phonetic_score']:.0f} < {MIN_SCORE} "
                f"— keeping original"
            )
            continue

        old_start = row['start_s']
        old_end = row['end_s']
        matched_data[i]['start_s'] = round(r['start_s'], 3)
        matched_data[i]['end_s'] = round(r['end_s'], 3)
        matched_data[i]['matched_transcription'] = r['matched_transcription']
        matched_data[i]['phonetic_score'] = r['phonetic_score']

        print(
            f"    🔧 '{word}': [{old_start:.1f}–{old_end:.1f}] "
            f"→ [{r['start_s']:.1f}–{r['end_s']:.1f}]  "
            f"score={r['phonetic_score']:.0f}  window={window_dur:.1f}s"
        )

    # ----------------------------------------------------------------
    # Save to Excel
    # ----------------------------------------------------------------
    df_words = pd.DataFrame(matched_data)
    save_path = os.path.join(sub_path, f"{text}_detected_words_v3_{sub_id}.xlsx")
    df_words.to_excel(save_path, index=False)

    print(f"\n  {'Word':<25} {'Start':>6}  {'End':>6}  {'Score':>5}  Matched")
    print(f"  {'-' * 70}")
    for row in matched_data:
        print(
            f"  {row['word']:<25} {row['start_s']:>6.1f}  {row['end_s']:>6.1f}  "
            f"{row['phonetic_score']:>5.0f}  {row['matched_transcription'] or 'NOT FOUND'}"
        )

    print(f"\n  ✅ Saved: {save_path}")

    # Cache flat token list for compatibility with old pipeline readers
    all_tokens = []
    for seg_idx in sorted(seg_word_tokens.keys()):
        all_tokens.extend(seg_word_tokens[seg_idx])
    raw_cache = os.path.join(sub_path, f"{text}_raw_detected_words_{sub_id}.json")
    save_alignment_results(all_tokens, raw_cache)

    return {sub_id: matched_data}