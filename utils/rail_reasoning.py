import re

import torch


RAIL_REASONING_DECISION_PATTERN = re.compile(
    r"This is a (?P<switch>turnout|merge) switch\. "
    r"The right blade is (?P<right_state>open|closed) and the left blade is (?P<left_state>open|closed)\. "
    r"The open (?P<open_side>right|left) blade and the closed (?P<closed_side>right|left) blade together create a continuous rail "
    r"connection toward the (?P<ego_path>right-hand|left-hand) path and break continuity with the (?P<other_path>right-hand|left-hand) path\. "
    r"Therefore, the ego-path follows the (?P<final_path>right-hand|left-hand) path\."
    r"(?: (?:It is \[SEG\]\.|Sure, \[SEG\]\.|Sure, it is \[SEG\]\.|The segmentation result is \[SEG\]\.|\[SEG\]\.))?$"
)

# Per-slot CE weights for Rail ReasonSeg explanations. Slots not listed here default to 1.0.
RAIL_REASONING_DECISION_GROUP_WEIGHTS = {
    "switch": 60.0,
    "right_state": 60.0,
    "left_state": 1.0,
    "open_side": 1.0,
    "closed_side": 1.0,
    "ego_path": 1.0,
    "other_path": 1.0,
    "final_path": 1.0,
}

RAIL_REASONING_PROMPT_GROUPS = ("open_side",)

RAIL_REASONING_MASK_ONLY_ANSWERS = {
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "The segmentation result is [SEG].",
    "[SEG].",
}

# Shared CE weight for the [SEG] token across all mask-producing scenarios.
# Set to 1.0 to disable extra weighting.
SEG_TOKEN_CE_WEIGHT = 3.0


def _decode_token_ids(tokenizer, token_ids):
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def _find_token_sequence(sequence, subsequence):
    if not subsequence or len(subsequence) > len(sequence):
        return []

    last_start = len(sequence) - len(subsequence) + 1
    return [
        start
        for start in range(last_start)
        if sequence[start : start + len(subsequence)] == subsequence
    ]


def locate_rail_reasoning_group_token_positions(
    assistant_text,
    actual_token_ids,
    tokenizer,
    search_start=0,
    search_end=None,
):
    """Locate and verify reasoning slots in the already-tokenized conversation.

    Tokenizing arbitrary text prefixes can change SentencePiece boundaries. This
    function instead tokenizes the complete canonical reasoning sentence once,
    anchors that exact sequence inside ``actual_token_ids``, and maps each regex
    group through cumulative decoding of that anchored sequence.
    """
    actual_token_ids = list(actual_token_ids)
    if search_end is None:
        search_end = len(actual_token_ids)
    if not 0 <= search_start <= search_end <= len(actual_token_ids):
        raise ValueError(
            "Invalid Rail reasoning token search span: "
            f"[{search_start}, {search_end}) for {len(actual_token_ids)} tokens"
        )

    normalized_assistant_text = assistant_text.strip()
    match = RAIL_REASONING_DECISION_PATTERN.search(normalized_assistant_text)
    if match is None:
        if normalized_assistant_text in RAIL_REASONING_MASK_ONLY_ANSWERS:
            return {}
        raise ValueError(
            "Rail reasoning answer is neither a canonical explanation nor a "
            f"supported mask-only answer: {assistant_text!r}"
        )

    reasoning_start = match.start()
    final_period = normalized_assistant_text.find(
        ".",
        match.end("final_path"),
    )
    if final_period < 0:
        raise ValueError("Could not find the final period in Rail reasoning text")
    reasoning_text = normalized_assistant_text[
        reasoning_start : final_period + 1
    ]
    reasoning_token_ids = tokenizer(
        reasoning_text,
        add_special_tokens=False,
    ).input_ids
    if not reasoning_token_ids:
        raise ValueError("Rail reasoning text produced no tokens")

    sequence_start = None
    anchor_token_offset = None
    anchor_matches = []
    search_token_ids = actual_token_ids[search_start:search_end]
    # A slow LLaMA tokenizer can alter the first token after a special token.
    # Every named decision group begins after "This is a", so anchoring from
    # one of the next few tokens remains exact without guessing an offset.
    for token_offset in range(min(4, len(reasoning_token_ids))):
        candidate_matches = _find_token_sequence(
            search_token_ids,
            list(reasoning_token_ids[token_offset:]),
        )
        anchor_matches.append((token_offset, candidate_matches))
        if len(candidate_matches) == 1:
            sequence_start = search_start + candidate_matches[0]
            anchor_token_offset = token_offset
            break

    if sequence_start is None:
        raise ValueError(
            "Could not uniquely anchor Rail reasoning tokens in the conversation: "
            f"search_span=[{search_start}, {search_end}) "
            f"matches={anchor_matches} reasoning={reasoning_text!r}"
        )

    decoded_reasoning = _decode_token_ids(tokenizer, reasoning_token_ids)
    decoded_reasoning_offset = decoded_reasoning.find(reasoning_text)
    if decoded_reasoning_offset < 0:
        if decoded_reasoning.strip() == reasoning_text:
            decoded_reasoning_offset = len(decoded_reasoning) - len(
                decoded_reasoning.lstrip()
            )
        else:
            raise ValueError(
                "Rail reasoning changed during tokenizer round-trip: "
                f"expected={reasoning_text!r} decoded={decoded_reasoning!r}"
            )

    decoded_prefix_lengths = [
        len(_decode_token_ids(tokenizer, reasoning_token_ids[:token_end]))
        for token_end in range(len(reasoning_token_ids) + 1)
    ]
    if any(
        end < start
        for start, end in zip(
            decoded_prefix_lengths,
            decoded_prefix_lengths[1:],
        )
    ):
        raise ValueError("Tokenizer decoding produced non-monotonic prefix lengths")

    group_positions = {}
    for group_name, expected_text in match.groupdict().items():
        relative_char_start = match.start(group_name) - reasoning_start
        relative_char_end = match.end(group_name) - reasoning_start
        decoded_char_start = decoded_reasoning_offset + relative_char_start
        decoded_char_end = decoded_reasoning_offset + relative_char_end

        relative_token_positions = [
            token_idx
            for token_idx, (token_char_start, token_char_end) in enumerate(
                zip(decoded_prefix_lengths, decoded_prefix_lengths[1:])
            )
            if token_char_end > decoded_char_start
            and token_char_start < decoded_char_end
        ]
        if not relative_token_positions:
            raise ValueError(
                f"No tokens aligned to Rail reasoning group {group_name!r}"
            )
        if relative_token_positions[0] < anchor_token_offset:
            raise ValueError(
                "Rail reasoning group occurs before the verified token anchor: "
                f"group={group_name!r} token_positions={relative_token_positions} "
                f"anchor_offset={anchor_token_offset}"
            )

        selected_token_ids = [
            reasoning_token_ids[token_idx]
            for token_idx in relative_token_positions
        ]
        decoded_group = _decode_token_ids(tokenizer, selected_token_ids).strip()
        if decoded_group != expected_text:
            raise ValueError(
                "Rail reasoning token alignment failed: "
                f"group={group_name!r} expected={expected_text!r} "
                f"decoded={decoded_group!r} token_ids={selected_token_ids}"
            )

        absolute_positions = [
            sequence_start + token_idx - anchor_token_offset
            for token_idx in relative_token_positions
        ]
        actual_selected_ids = [
            actual_token_ids[position] for position in absolute_positions
        ]
        actual_decoded_group = _decode_token_ids(
            tokenizer,
            actual_selected_ids,
        ).strip()
        if actual_decoded_group != expected_text:
            raise ValueError(
                "Anchored Rail reasoning tokens failed verification: "
                f"group={group_name!r} expected={expected_text!r} "
                f"decoded={actual_decoded_group!r} "
                f"positions={absolute_positions}"
            )

        group_positions[group_name] = absolute_positions

    return group_positions


def build_rail_reasoning_decision_token_mask(text, tokenizer, target_len):
    token_mask = torch.zeros(target_len, dtype=torch.bool)

    match = RAIL_REASONING_DECISION_PATTERN.search(text)
    if match is None:
        return token_mask

    for group_name in RAIL_REASONING_PROMPT_GROUPS:
        char_start, char_end = match.span(group_name)
        token_start = len(
            tokenizer(text[:char_start].rstrip(), add_special_tokens=False).input_ids
        )
        token_end = len(
            tokenizer(text[:char_end], add_special_tokens=False).input_ids
        )
        token_start = max(0, min(target_len, token_start))
        token_end = max(token_start, min(target_len, token_end))
        if token_end > token_start:
            token_mask[token_start:token_end] = True

    return token_mask

