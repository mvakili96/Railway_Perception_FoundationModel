import re
import unittest

from utils.rail_reasoning import (
    locate_rail_reasoning_group_token_positions,
)


class _WordPieceTokenizer:
    _pattern = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|[.,:]")

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def _id_for_token(self, token):
        if token not in self.token_to_id:
            token_id = len(self.token_to_id) + 1
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        tokens = []
        for token in self._pattern.findall(text):
            if token == "turnout":
                tokens.extend(["turn", "##out"])
            else:
                tokens.append(token)
        return type(
            "Tokenized",
            (),
            {"input_ids": [self._id_for_token(token) for token in tokens]},
        )()

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        output = ""
        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if token.startswith("##"):
                output += token[2:]
            elif token in ".,:":
                output += token
            else:
                if output:
                    output += " "
                output += token
        return output


class RailReasoningTest(unittest.TestCase):
    explanation = (
        "This is a turnout switch. The right blade is open and the left blade "
        "is closed. The open right blade and the closed left blade together "
        "create a continuous rail connection toward the right-hand path and "
        "break continuity with the left-hand path. Therefore, the ego-path "
        "follows the right-hand path."
    )
    closed_explanation = (
        "This is a merge switch. The right blade is closed and the left blade "
        "is open. The open left blade and the closed right blade together "
        "create a continuous rail connection toward the left-hand path and "
        "break continuity with the right-hand path. Therefore, the ego-path "
        "follows the left-hand path."
    )

    def test_group_positions_anchor_to_actual_conversation_tokens(self):
        tokenizer = _WordPieceTokenizer()
        conversation = "ASSISTANT: " + self.explanation
        conversation_ids = tokenizer(
            conversation,
            add_special_tokens=False,
        ).input_ids

        positions = locate_rail_reasoning_group_token_positions(
            self.explanation,
            conversation_ids,
            tokenizer,
        )

        switch_ids = [conversation_ids[index] for index in positions["switch"]]
        right_state_ids = [
            conversation_ids[index] for index in positions["right_state"]
        ]
        self.assertEqual(tokenizer.decode(switch_ids), "turnout")
        self.assertEqual(len(switch_ids), 2)
        self.assertEqual(tokenizer.decode(right_state_ids), "open")

    def test_closed_right_state_anchors_to_actual_conversation_tokens(self):
        tokenizer = _WordPieceTokenizer()
        conversation = "ASSISTANT: " + self.closed_explanation
        conversation_ids = tokenizer(
            conversation,
            add_special_tokens=False,
        ).input_ids

        positions = locate_rail_reasoning_group_token_positions(
            self.closed_explanation,
            conversation_ids,
            tokenizer,
        )
        right_state_ids = [
            conversation_ids[index] for index in positions["right_state"]
        ]

        self.assertEqual(tokenizer.decode(right_state_ids), "closed")

    def test_anchor_tolerates_context_dependent_first_token(self):
        tokenizer = _WordPieceTokenizer()
        conversation = "ASSISTANT: " + self.explanation
        conversation_ids = tokenizer(
            conversation,
            add_special_tokens=False,
        ).input_ids
        standalone_ids = tokenizer(
            self.explanation,
            add_special_tokens=False,
        ).input_ids
        reasoning_start = conversation_ids.index(standalone_ids[0])
        alternate_this_id = len(tokenizer.id_to_token) + 1
        tokenizer.id_to_token[alternate_this_id] = "This"
        conversation_ids[reasoning_start] = alternate_this_id

        positions = locate_rail_reasoning_group_token_positions(
            self.explanation,
            conversation_ids,
            tokenizer,
        )

        switch_ids = [conversation_ids[index] for index in positions["switch"]]
        self.assertEqual(tokenizer.decode(switch_ids), "turnout")
        self.assertEqual(len(switch_ids), 2)

    def test_search_span_ignores_an_echoed_explanation(self):
        tokenizer = _WordPieceTokenizer()
        prefix = self.explanation + " ASSISTANT: "
        conversation = prefix + self.explanation
        conversation_ids = tokenizer(
            conversation,
            add_special_tokens=False,
        ).input_ids
        search_start = len(
            tokenizer(prefix, add_special_tokens=False).input_ids
        )

        positions = locate_rail_reasoning_group_token_positions(
            self.explanation,
            conversation_ids,
            tokenizer,
            search_start=search_start,
            search_end=len(conversation_ids),
        )

        self.assertGreaterEqual(min(positions["switch"]), search_start)

    def test_group_positions_fail_when_actual_tokens_do_not_contain_reasoning(self):
        tokenizer = _WordPieceTokenizer()
        unrelated_ids = tokenizer(
            "ASSISTANT: The segmentation result is complete.",
            add_special_tokens=False,
        ).input_ids

        with self.assertRaisesRegex(ValueError, "Could not uniquely anchor"):
            locate_rail_reasoning_group_token_positions(
                self.explanation,
                unrelated_ids,
                tokenizer,
            )

    def test_mask_only_answer_has_no_decision_positions(self):
        tokenizer = _WordPieceTokenizer()
        answer = "The segmentation result is [SEG]."

        self.assertEqual(
            locate_rail_reasoning_group_token_positions(
                answer,
                tokenizer(answer, add_special_tokens=False).input_ids,
                tokenizer,
            ),
            {},
        )

    def test_trailing_conversation_whitespace_is_ignored(self):
        tokenizer = _WordPieceTokenizer()
        assistant_text = self.explanation + " "
        actual_ids = tokenizer(
            assistant_text,
            add_special_tokens=False,
        ).input_ids

        positions = locate_rail_reasoning_group_token_positions(
            assistant_text,
            actual_ids,
            tokenizer,
        )

        switch_ids = [actual_ids[index] for index in positions["switch"]]
        self.assertEqual(tokenizer.decode(switch_ids), "turnout")

    def test_noncanonical_explanation_fails_loudly(self):
        tokenizer = _WordPieceTokenizer()
        malformed = self.explanation.replace("right blade is open", "right blade open")

        with self.assertRaisesRegex(ValueError, "neither a canonical explanation"):
            locate_rail_reasoning_group_token_positions(
                malformed,
                tokenizer(malformed, add_special_tokens=False).input_ids,
                tokenizer,
            )


if __name__ == "__main__":
    unittest.main()

