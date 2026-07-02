import unittest
import re

import numpy as np

from utils.rail_augmentation import (flip_rail_reasoning_explanation,
                                     horizontal_flip, swap_left_right)


class RailAugmentationTest(unittest.TestCase):
    def test_swap_left_right_preserves_case_and_compound_terms(self):
        text = (
            "The LEFT blade selects the right-hand path while the "
            "Left rail avoids the bright marker."
        )

        self.assertEqual(
            swap_left_right(text),
            "The RIGHT blade selects the left-hand path while the "
            "Right rail avoids the bright marker.",
        )

    def test_horizontal_flip_keeps_image_and_mask_aligned(self):
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        mask = np.zeros((2, 4), dtype=np.uint8)
        image[0, 0] = 255
        mask[0, 0] = 1

        flipped_image = horizontal_flip(image)
        flipped_mask = horizontal_flip(mask)

        self.assertTrue(np.array_equal(flipped_image[0, 3], image[0, 0]))
        self.assertEqual(flipped_mask[0, 3], 1)
        self.assertTrue(flipped_image.flags.c_contiguous)
        self.assertTrue(flipped_mask.flags.c_contiguous)

    def test_explanation_flip_preserves_template_order(self):
        pattern = re.compile(
            r"The right blade is (?P<right_state>open|closed) "
            r"and the left blade is (?P<left_state>open|closed)\. "
            r"The open (?P<open_side>right|left) blade and the closed "
            r"(?P<closed_side>right|left) blade connect the "
            r"(?P<ego_path>right-hand|left-hand) path, not the "
            r"(?P<other_path>right-hand|left-hand) path\. "
            r"The ego-path follows the "
            r"(?P<final_path>right-hand|left-hand) path\."
        )
        explanation = (
            "The right blade is open and the left blade is closed. "
            "The open right blade and the closed left blade connect the "
            "right-hand path, not the left-hand path. "
            "The ego-path follows the right-hand path. [SEG]."
        )

        self.assertEqual(
            flip_rail_reasoning_explanation(explanation, pattern),
            "The right blade is closed and the left blade is open. "
            "The open left blade and the closed right blade connect the "
            "left-hand path, not the right-hand path. "
            "The ego-path follows the left-hand path. [SEG].",
        )


if __name__ == "__main__":
    unittest.main()
