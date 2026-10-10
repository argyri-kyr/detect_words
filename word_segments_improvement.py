
import re

def clean_token_for_timing(text):
    return re.sub(r'[^\w\s]', '', str(text).lower().strip())


def split_text_tokens(text):
    tokens = []
    for t in str(text).split():
        cleaned = clean_token_for_timing(t)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def proportional_word_timestamps(seg_start, seg_end, tokens, min_word_dur=0.20):
    """
    Fallback timestamps when WhisperX collapses words.
    Distributes segment time across words by character length.
    """

    seg_start = float(seg_start)
    seg_end = float(seg_end)

    if not tokens:
        return []

    duration = max(0.0, seg_end - seg_start)

    if duration <= 0:
        return []

    weights = [max(1, len(t)) for t in tokens]
    total_weight = sum(weights)

    results = []
    cursor = seg_start

    for i, token in enumerate(tokens):
        if i == len(tokens) - 1:
            word_end = seg_end
        else:
            word_dur = duration * (weights[i] / total_weight)
            word_dur = max(min_word_dur, word_dur)
            word_end = min(seg_end, cursor + word_dur)

        results.append({
            "word": token,
            "start": cursor,
            "end": word_end,
            "score": 0.0,
            "timing_source": "proportional_fallback",
        })

        cursor = word_end

    return results


def repair_aligned_segment_if_collapsed(
    input_segment,
    aligned_segment,
    min_coverage_ratio=0.45,
    min_avg_word_dur=0.35,
    low_score_threshold=0.35,
):
    """
    Repairs one WhisperX aligned segment if word timings are obviously collapsed.

    Example bad case:
    Segment window: 19.64 → 30.40
    WhisperX words: 19.64 → 20.68
    Coverage is tiny, so replace with proportional timings.
    """

    seg_start = float(input_segment["start"])
    seg_end = float(input_segment["end"])
    seg_duration = max(0.001, seg_end - seg_start)

    tokens = split_text_tokens(input_segment["text"])

    words = aligned_segment.get("words", [])

    valid_words = [
        w for w in words
        if "start" in w and "end" in w and clean_token_for_timing(w.get("word", ""))
    ]

    if not valid_words:
        return proportional_word_timestamps(seg_start, seg_end, tokens)

    aligned_start = min(float(w["start"]) for w in valid_words)
    aligned_end = max(float(w["end"]) for w in valid_words)
    aligned_coverage = max(0.0, aligned_end - aligned_start) / seg_duration

    durations = [
        max(0.0, float(w["end"]) - float(w["start"]))
        for w in valid_words
    ]

    avg_word_dur = sum(durations) / max(1, len(durations))

    scores = [
        float(w.get("score", 1.0))
        for w in valid_words
        if w.get("score") is not None
    ]

    avg_score = sum(scores) / max(1, len(scores)) if scores else 1.0

    # Main collapse detector
    collapsed = (
        aligned_coverage < min_coverage_ratio
        or avg_word_dur < min_avg_word_dur
        or avg_score < low_score_threshold
    )

    if collapsed:
        print(
            "COLLAPSED ALIGNMENT REPAIR:",
            f"{seg_start:.2f}-{seg_end:.2f}",
            "|",
            input_segment["text"],
            "| coverage:",
            round(aligned_coverage, 2),
            "avg_dur:",
            round(avg_word_dur, 2),
            "avg_score:",
            round(avg_score, 2),
        )

        return proportional_word_timestamps(seg_start, seg_end, tokens)

    repaired = []

    for w in valid_words:
        ww = dict(w)
        ww["word"] = clean_token_for_timing(ww.get("word", ""))
        ww["timing_source"] = "whisperx"
        repaired.append(ww)

    return repaired