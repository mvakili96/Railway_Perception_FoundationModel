"""Strict Qwen text/token alignment helpers (no torch or model imports)."""

import re


ASSISTANT_END = "<|im_end|>"
_IMAGE_LINE = re.compile(r"(?m)^[ \t]*<image>[ \t]*(?:\r?\n|$)")
_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|image_pad|>",
                   "<|video_pad|>", "<|vision_start|>", "<|vision_end|>", "<|placeholder|>")


def question_content(question):
    if any(token in question for token in _CONTROL_TOKENS):
        raise ValueError("Dataset question contains a reserved Qwen control token.")
    markers = list(_IMAGE_LINE.finditer(question))
    if len(markers) > 1:
        raise ValueError("Each segmentation conversation must contain exactly one image.")
    if not markers:
        return [{"type": "image"}, {"type": "text", "text": question}]
    marker = markers[0]
    content = []
    if question[:marker.start()]:
        content.append({"type": "text", "text": question[:marker.start()]})
    content.append({"type": "image"})
    if question[marker.end():]:
        content.append({"type": "text", "text": question[marker.end():]})
    return content


def render_qwen_pair(processor, question, answer, inference=False):
    if any(token in answer for token in _CONTROL_TOKENS):
        raise ValueError("Dataset answer contains a reserved Qwen control token.")
    user = {"role": "user", "content": question_content(question)}
    prefix = processor.apply_chat_template(
        [user], tokenize=False, add_generation_prompt=True,
    )
    if inference:
        return prefix, None
    rendered = processor.apply_chat_template(
        [user, {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False,
    )
    # Verify the actual template, not an assumed LLaVA/BOS token offset.
    if rendered != prefix + answer + ASSISTANT_END + "\n":
        raise ValueError("Native Qwen template changed the answer or assistant boundary.")
    return rendered, len(prefix)


def expand_image_for_alignment(text, image_token, visual_tokens):
    """Reconstruct native expansion only to interpret tokenizer char offsets.

    Pixel processing and the model's input IDs still come from AutoProcessor.
    The collator verifies this string's token IDs against those native IDs.
    """
    if visual_tokens <= 0 or text.count(image_token) != 1:
        raise ValueError("Expected one native image placeholder and a positive visual-token count.")
    return text.replace(image_token, image_token * visual_tokens, 1)


def decode_exact(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def assistant_positions(ids, offsets, expanded_text, char_start, char_end, tokenizer):
    if len(ids) != len(offsets) or not 0 <= char_start < char_end <= len(expanded_text):
        raise ValueError("Invalid assistant token/character bounds.")
    positions = []
    for index, (start, end) in enumerate(offsets):
        if end > char_start and start < char_end:
            if start < char_start or end > char_end:
                raise ValueError("A token crosses the assistant boundary; refusing prefix supervision.")
            positions.append(index)
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("Assistant tokens must form a nonempty contiguous span.")
    decoded = decode_exact(tokenizer, [ids[index] for index in positions])
    expected = expanded_text[char_start:char_end]
    if decoded != expected:
        raise ValueError(f"Assistant round-trip changed: expected={expected!r}, decoded={decoded!r}.")
    return positions


def reasoning_positions(answer, offsets, char_start, ids, tokenizer, pattern, mask_only_answers):
    stripped = answer.strip()
    match = pattern.search(stripped)
    if match is None:
        if stripped in mask_only_answers:
            return {}
        raise ValueError("Rail answer is neither a canonical explanation nor a supported mask-only answer.")
    leading_space = len(answer) - len(answer.lstrip())
    groups = {}
    for name, expected in match.groupdict().items():
        start = char_start + leading_space + match.start(name)
        end = char_start + leading_space + match.end(name)
        positions = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
        decoded = decode_exact(tokenizer, [ids[i] for i in positions]).strip()
        if not positions or decoded != expected:
            raise ValueError(
                f"Rail field {name!r} failed Qwen alignment: expected={expected!r}, "
                f"decoded={decoded!r}, positions={positions}."
            )
        groups[name] = positions
    return groups
