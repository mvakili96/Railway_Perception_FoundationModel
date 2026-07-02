import re

import numpy as np


_DIRECTION_PATTERN = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)


def swap_left_right(text):
    def replace_direction(match):
        direction = match.group(0)
        replacement = "right" if direction.lower() == "left" else "left"
        if direction.isupper():
            return replacement.upper()
        if direction[0].isupper():
            return replacement.capitalize()
        return replacement

    return _DIRECTION_PATTERN.sub(replace_direction, text)


def flip_rail_reasoning_explanation(text, decision_pattern):
    match = decision_pattern.search(text)
    if match is None:
        return swap_left_right(text)

    replacements = {
        "right_state": match.group("left_state"),
        "left_state": match.group("right_state"),
        "open_side": swap_left_right(match.group("open_side")),
        "closed_side": swap_left_right(match.group("closed_side")),
        "ego_path": swap_left_right(match.group("ego_path")),
        "other_path": swap_left_right(match.group("other_path")),
        "final_path": swap_left_right(match.group("final_path")),
    }
    for group_name, replacement in sorted(
        replacements.items(),
        key=lambda item: match.start(item[0]),
        reverse=True,
    ):
        start, end = match.span(group_name)
        text = text[:start] + replacement + text[end:]

    return text


def horizontal_flip(array):
    return np.ascontiguousarray(np.flip(array, axis=1))
