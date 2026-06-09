import re

import torch


RAIL_REASONING_DECISION_PATTERN = re.compile(
    r"This is a (?P<switch>turnout|merge) switch\. "
    r"The right blade is (?P<right_state>open|closed) and the left blade is (?P<left_state>open|closed)\. "
    r"The open (?P<open_side>right|left) blade and the closed (?P<closed_side>right|left) blade together create a continuous rail "
    r"connection toward the (?P<ego_path>right-hand|left-hand) path and break continuity with the (?P<other_path>right-hand|left-hand) path\. "
    r"Therefore, the ego-path follows the (?P<final_path>right-hand|left-hand) path\."
    r"(?: (?:It is \[SEG\]\.|Sure, \[SEG\]\.|Sure, it is \[SEG\]\.|Sure, the segmentation result is \[SEG\]\.|\[SEG\]\.))?$"
)

# Per-slot CE weights for Rail ReasonSeg explanations. Slots not listed here default to 1.0.
RAIL_REASONING_DECISION_GROUP_WEIGHTS = {
    "switch": 3.0,
    "right_state": 2.5,
    "left_state": 2.5,
    "open_side": 2.5,
    "closed_side": 2.5,
    "ego_path": 3.0,
    "other_path": 2.0,
    "final_path": 3.5,
}

RAIL_REASONING_PROMPT_GROUPS = ("open_side",)

# Shared CE weight for the [SEG] token across all mask-producing scenarios.
# Set to 1.0 to disable extra weighting.
SEG_TOKEN_CE_WEIGHT = 3.0


def build_rail_reasoning_decision_token_mask(text, tokenizer, target_len):
    token_mask = torch.zeros(target_len, dtype=torch.bool)

    match = RAIL_REASONING_DECISION_PATTERN.search(text)
    if match is None:
        return token_mask

    for group_name in RAIL_REASONING_PROMPT_GROUPS:
        char_start, char_end = match.span(group_name)
        token_start = len(tokenizer(text[:char_start], add_special_tokens=False).input_ids)
        token_end = len(tokenizer(text[:char_end], add_special_tokens=False).input_ids)
        token_start = max(0, min(target_len, token_start))
        token_end = max(token_start, min(target_len, token_end))
        if token_end > token_start:
            token_mask[token_start:token_end] = True

    return token_mask
