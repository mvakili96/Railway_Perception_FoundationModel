import unittest

import torch

from model.seg_query_bridge import (
    FourQuerySegBridge,
    build_multimodal_position_masks,
    finalize_four_query_bridge_load,
    group_prompt_embeddings_by_image,
)
from model.segment_anything.modeling import (
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)


class MultimodalPositionMaskTest(unittest.TestCase):
    def test_maps_actual_seg_and_legacy_predictor_after_image_expansion(self):
        image_token = -200
        seg_token = 99
        input_ids = torch.tensor(
            [
                [1, image_token, 2, seg_token, 3, 0],
                [1, 2, seg_token, 0, 0, 0],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        valid, anchors, predictors = build_multimodal_position_masks(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expanded_length=8,
            seg_token_idx=seg_token,
            image_token_idx=image_token,
            image_patch_count=3,
        )

        self.assertEqual(
            valid[0].tolist(),
            [True, True, True, True, True, True, True, False],
        )
        self.assertEqual(
            valid[1].tolist(),
            [True, True, True, False, False, False, False, False],
        )
        self.assertEqual(anchors[0].nonzero().flatten().tolist(), [5])
        self.assertEqual(predictors[0].nonzero().flatten().tolist(), [4])
        self.assertEqual(anchors[1].nonzero().flatten().tolist(), [2])
        self.assertEqual(predictors[1].nonzero().flatten().tolist(), [1])

    def test_preserves_prompt_padding_gap_before_generated_seg(self):
        valid, anchors, predictors = build_multimodal_position_masks(
            input_ids=torch.tensor([[1, -200, 2, 0, 0, 99, 3]]),
            attention_mask=torch.tensor([[1, 1, 1, 0, 0, 1, 1]]),
            expanded_length=9,
            seg_token_idx=99,
            image_token_idx=-200,
            image_patch_count=3,
        )

        self.assertEqual(
            valid[0].tolist(),
            [True, True, True, True, True, False, False, True, True],
        )
        self.assertEqual(anchors[0].nonzero().flatten().tolist(), [7])
        self.assertEqual(predictors[0].nonzero().flatten().tolist(), [4])

    def test_maps_multiple_image_sentinels_before_seg(self):
        valid, anchors, predictors = build_multimodal_position_masks(
            input_ids=torch.tensor([[1, -200, 2, -200, 99, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]),
            expanded_length=8,
            seg_token_idx=99,
            image_token_idx=-200,
            image_patch_count=2,
        )

        self.assertEqual(
            valid[0].tolist(),
            [True, True, True, True, True, True, True, False],
        )
        self.assertEqual(anchors[0].nonzero().flatten().tolist(), [6])
        self.assertEqual(predictors[0].nonzero().flatten().tolist(), [5])

    def test_rejects_masked_gap_before_image_sentinel(self):
        with self.assertRaisesRegex(ValueError, "before an image sentinel"):
            build_multimodal_position_masks(
                input_ids=torch.tensor([[0, 1, -200, 2, 99]]),
                attention_mask=torch.tensor([[0, 1, 1, 1, 1]]),
                expanded_length=7,
                seg_token_idx=99,
                image_token_idx=-200,
                image_patch_count=3,
            )


class FourQuerySegBridgeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.hidden_size = 16
        self.sequence_length = 8
        self.bridge = FourQuerySegBridge(
            hidden_size=self.hidden_size,
            out_dim=256,
            num_queries=4,
            num_heads=8,
            num_hidden_layers=4,
        )
        self.anchors = torch.zeros(2, self.sequence_length, dtype=torch.bool)
        self.anchors[0, 3] = True
        self.anchors[1, 5] = True
        self.valid = torch.zeros_like(self.anchors)
        self.valid[0, :6] = True
        self.valid[1, :7] = True
        self.hidden_states = tuple(
            torch.randn(2, self.sequence_length, self.hidden_size)
            for _ in range(7)
        )

    def test_one_four_query_group_per_seg(self):
        prompts = self.bridge(
            self.hidden_states,
            self.anchors,
            self.valid,
        )

        self.assertEqual(tuple(self.bridge.learned_queries.shape), (4, 256))
        self.assertEqual(tuple(prompts.shape), (2, 4, 256))
        self.assertEqual(
            self.bridge.selected_layer_indices(33),
            [8, 16, 24, 32],
        )

    def test_full_reset_initializes_every_bridge_parameter(self):
        with torch.no_grad():
            for parameter in self.bridge.parameters():
                parameter.fill_(float("nan"))

        self.bridge.reset_parameters()
        self.bridge.validate_parameters()

        for parameter in self.bridge.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
        self.assertAlmostEqual(self.bridge.residual_gate.item(), 0.01, places=6)

    def test_legacy_checkpoint_initializes_the_complete_bridge(self):
        with torch.no_grad():
            for parameter in self.bridge.parameters():
                parameter.fill_(float("nan"))
        missing_keys = [
            f"model.seg_query_bridge.{name}"
            for name in self.bridge.state_dict().keys()
        ]

        status = finalize_four_query_bridge_load(self.bridge, missing_keys)

        self.assertEqual(status, "initialized_for_legacy_checkpoint")
        self.bridge.validate_parameters()

    def test_loaded_checkpoint_preserves_bridge_parameters(self):
        expected = {
            name: parameter.detach().clone()
            for name, parameter in self.bridge.named_parameters()
        }

        status = finalize_four_query_bridge_load(self.bridge, [])

        self.assertEqual(status, "loaded_from_checkpoint")
        for name, parameter in self.bridge.named_parameters():
            self.assertTrue(torch.equal(parameter, expected[name]))

    def test_old_four_query_checkpoint_only_initializes_missing_gate(self):
        expected = {
            name: parameter.detach().clone()
            for name, parameter in self.bridge.named_parameters()
            if name != "residual_gate"
        }
        with torch.no_grad():
            self.bridge.residual_gate.fill_(float("nan"))

        status = finalize_four_query_bridge_load(
            self.bridge,
            ["model.seg_query_bridge.residual_gate"],
        )

        self.assertEqual(status, "loaded_with_default_residual_gate")
        self.assertAlmostEqual(self.bridge.residual_gate.item(), 0.01, places=6)
        for name, parameter in self.bridge.named_parameters():
            if name != "residual_gate":
                self.assertTrue(torch.equal(parameter, expected[name]))

    def test_partially_missing_bridge_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "only part"):
            finalize_four_query_bridge_load(
                self.bridge,
                ["model.seg_query_bridge.memory_projection.weight"],
            )

    def test_no_seg_returns_an_empty_prompt_batch(self):
        prompts, keepalive_loss = self.bridge(
            self.hidden_states,
            torch.zeros_like(self.anchors),
            self.valid,
            return_keepalive_loss=True,
        )

        self.assertEqual(tuple(prompts.shape), (0, 4, 256))
        keepalive_loss.backward()
        for parameter in self.bridge.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_bridge_uses_stable_fp32_math_for_bfloat16_inputs(self):
        bridge = FourQuerySegBridge(
            hidden_size=self.hidden_size,
            out_dim=32,
            num_queries=4,
            num_heads=4,
            num_hidden_layers=3,
        ).to(dtype=torch.bfloat16)
        hidden_states = tuple(
            state.to(dtype=torch.bfloat16).requires_grad_()
            for state in self.hidden_states
        )

        prompts, keepalive_loss = bridge(
            hidden_states,
            self.anchors,
            self.valid,
            return_keepalive_loss=True,
            check_finite=True,
        )

        self.assertEqual(prompts.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(prompts).all())
        (prompts.float().sum() + keepalive_loss.float()).backward()
        for parameter in bridge.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_fp32_cross_attention_matches_torch_mha(self):
        queries = torch.randn(2, 4, 256)
        memory = torch.randn(2, 7, 256)
        padding_mask = torch.tensor(
            [
                [False, False, False, False, False, False, False],
                [False, False, False, False, True, True, True],
            ]
        )

        expected, _ = self.bridge.cross_attention(
            query=queries,
            key=memory,
            value=memory,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        actual = self.bridge._cross_attention_fp32(
            queries,
            memory,
            padding_mask,
        )

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-5))

    def test_numerics_debug_identifies_bad_selected_llama_layer(self):
        hidden_states = [state.clone() for state in self.hidden_states]
        selected_layer = self.bridge.selected_layer_indices(len(hidden_states))[0]
        hidden_states[selected_layer][0, 1, 0] = float("nan")

        with self.assertRaisesRegex(
            FloatingPointError,
            f"stage=llama_layer_{selected_layer}_causal_states",
        ):
            self.bridge(
                hidden_states,
                self.anchors,
                self.valid,
                check_finite=True,
            )

    def test_actual_seg_anchor_conditions_queries(self):
        bridge = FourQuerySegBridge(
            hidden_size=self.hidden_size,
            out_dim=256,
            num_queries=4,
            num_heads=8,
            num_hidden_layers=4,
        )
        with torch.no_grad():
            bridge.memory_projection.weight.zero_()
            bridge.memory_projection.bias.zero_()
        baseline = bridge(
            self.hidden_states,
            self.anchors,
            self.valid,
        )
        changed = [state.clone() for state in self.hidden_states]
        changed[-1][0, 3, 0] += 5.0

        anchored = bridge(changed, self.anchors, self.valid)

        self.assertFalse(torch.allclose(baseline[0], anchored[0]))
        self.assertTrue(torch.allclose(baseline[1], anchored[1]))

    def test_future_and_padding_states_cannot_change_prompts(self):
        baseline = self.bridge(
            self.hidden_states,
            self.anchors,
            self.valid,
        )
        changed = [state.clone() for state in self.hidden_states]
        for state in changed:
            state[0, 4:] += 1000.0
            state[1, 6:] -= 1000.0

        causal = self.bridge(changed, self.anchors, self.valid)

        self.assertTrue(torch.allclose(baseline, causal, atol=1e-6, rtol=1e-5))

    def test_valid_prefix_states_in_selected_layers_affect_prompts(self):
        baseline = self.bridge(
            self.hidden_states,
            self.anchors,
            self.valid,
        )
        changed = [state.clone() for state in self.hidden_states]
        selected_layer = self.bridge.selected_layer_indices(len(changed))[0]
        changed[selected_layer][0, 1, 0] += 20.0

        prefix_changed = self.bridge(changed, self.anchors, self.valid)

        self.assertFalse(torch.allclose(baseline[0], prefix_changed[0]))
        self.assertTrue(torch.allclose(baseline[1], prefix_changed[1]))

    def test_each_seg_event_has_its_own_causal_cutoff(self):
        anchors = torch.zeros(1, self.sequence_length, dtype=torch.bool)
        anchors[0, 2] = True
        anchors[0, 5] = True
        valid = torch.ones_like(anchors)
        hidden_states = tuple(
            torch.randn(1, self.sequence_length, self.hidden_size)
            for _ in range(7)
        )
        baseline = self.bridge(hidden_states, anchors, valid)
        changed = [state.clone() for state in hidden_states]
        selected_layer = self.bridge.selected_layer_indices(len(changed))[0]
        changed[selected_layer][0, 4, 0] += 20.0

        later_prefix_changed = self.bridge(changed, anchors, valid)

        self.assertTrue(torch.allclose(baseline[0], later_prefix_changed[0]))
        self.assertFalse(torch.allclose(baseline[1], later_prefix_changed[1]))

    def test_only_selected_hidden_layers_receive_memory_gradients(self):
        bridge = FourQuerySegBridge(
            hidden_size=self.hidden_size,
            out_dim=32,
            num_queries=4,
            num_heads=4,
            num_hidden_layers=3,
        )
        hidden_states = tuple(
            torch.randn(
                1,
                self.sequence_length,
                self.hidden_size,
                requires_grad=True,
            )
            for _ in range(9)
        )
        anchors = torch.zeros(1, self.sequence_length, dtype=torch.bool)
        anchors[0, 4] = True
        valid = torch.ones_like(anchors)
        prompts, keepalive_loss = bridge(
            hidden_states,
            anchors,
            valid,
            return_keepalive_loss=True,
        )
        weights = torch.linspace(0.1, 1.0, prompts.numel()).view_as(prompts)
        ((prompts * weights).sum() + keepalive_loss).backward()

        selected = set(bridge.selected_layer_indices(len(hidden_states)))
        self.assertNotIn(0, selected)
        for layer_index, state in enumerate(hidden_states):
            if layer_index in selected:
                self.assertIsNotNone(state.grad)
                self.assertTrue(torch.isfinite(state.grad).all())
            else:
                self.assertIsNone(state.grad)

        for parameter in bridge.parameters():
            self.assertTrue(parameter.requires_grad)
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(bridge.learned_queries.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            bridge.anchor_projection.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            bridge.memory_projection.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_state_dict_round_trip_is_strict_and_exact(self):
        expected = self.bridge(
            self.hidden_states,
            self.anchors,
            self.valid,
        )
        restored = FourQuerySegBridge(
            hidden_size=self.hidden_size,
            out_dim=256,
            num_queries=4,
            num_heads=8,
            num_hidden_layers=4,
        )
        legacy_state_dict = self.bridge.state_dict()
        legacy_state_dict.pop("residual_gate")
        restored.load_state_dict(legacy_state_dict, strict=True)

        actual = restored(self.hidden_states, self.anchors, self.valid)

        self.assertTrue(torch.equal(expected, actual))
        self.assertIn("learned_queries", restored.state_dict())
        self.assertIn("cross_attention.in_proj_weight", restored.state_dict())

    def test_scalar_residual_gate_checkpoint_is_reshaped(self):
        state_dict = self.bridge.state_dict()
        state_dict["residual_gate"] = torch.tensor(0.01)
        restored = FourQuerySegBridge(
            hidden_size=self.hidden_size,
            out_dim=256,
            num_queries=4,
            num_heads=8,
            num_hidden_layers=4,
        )

        restored.load_state_dict(state_dict, strict=True)

        self.assertEqual(tuple(restored.residual_gate.shape), (1,))
        self.assertAlmostEqual(restored.residual_gate.item(), 0.01, places=6)


class SegPromptGroupingTest(unittest.TestCase):
    def test_groups_conversations_by_source_image(self):
        prompts = torch.arange(3 * 4 * 256, dtype=torch.float32).view(
            3, 4, 256
        )

        grouped = group_prompt_embeddings_by_image(
            prompt_embeddings=prompts,
            seg_token_counts=torch.tensor([1, 0, 2]),
            offset=torch.tensor([0, 2, 3]),
        )

        self.assertEqual(
            [tuple(group.shape) for group in grouped],
            [(1, 4, 256), (2, 4, 256)],
        )
        self.assertTrue(torch.equal(grouped[0], prompts[:1]))
        self.assertTrue(torch.equal(grouped[1], prompts[1:]))

    def test_preserves_empty_prompt_group_for_an_image_without_seg(self):
        grouped = group_prompt_embeddings_by_image(
            prompt_embeddings=torch.empty(0, 4, 256),
            seg_token_counts=torch.tensor([0, 0]),
            offset=torch.tensor([0, 2]),
        )

        self.assertEqual(len(grouped), 1)
        self.assertEqual(tuple(grouped[0].shape), (0, 4, 256))

    def test_four_sparse_tokens_produce_one_sam_mask_per_seg(self):
        prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(2, 2),
            input_image_size=(8, 8),
            mask_in_chans=16,
        )
        mask_decoder = MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=1,
                embedding_dim=256,
                num_heads=8,
                mlp_dim=512,
            ),
            num_multimask_outputs=3,
            iou_head_depth=2,
            iou_head_hidden_dim=64,
        )
        prompt_groups = torch.randn(2, 4, 256)

        sparse, dense = prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=prompt_groups,
        )
        masks, scores = mask_decoder(
            image_embeddings=torch.randn(1, 256, 2, 2),
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        self.assertEqual(tuple(sparse.shape), (2, 4, 256))
        self.assertEqual(tuple(masks.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(scores.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
