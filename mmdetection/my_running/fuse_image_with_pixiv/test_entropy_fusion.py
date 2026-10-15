"""Focused checks for local entropy and cross-modal enhancement."""

import unittest

import cv2
import numpy as np

from cross_modal_enhancement import (
    enhance_ir_information_on_rgb,
    enhance_rgb_information_on_ir,
)
from image_entropy import compute_region_weights, image_entropy, weights_for_feature_size


class EntropyFusionTests(unittest.TestCase):
    def setUp(self):
        self.valid = np.ones((128, 128), dtype=np.float32)
        self.flat = np.full((128, 128), 128, dtype=np.uint8)
        line = np.array([20] * 16 + [220] * 16, dtype=np.uint8)
        self.structured = np.tile(line, (128, 4))

    def test_entropy_includes_255_and_ignores_empty_bins(self):
        self.assertAlmostEqual(image_entropy(np.full((16, 16), 255, np.uint8)), 0.0)
        pixels = np.tile(np.array([0, 255], np.uint8), (16, 8))
        self.assertAlmostEqual(image_entropy(pixels), 1.0)

    def test_low_information_modality_falls_back_to_other(self):
        weights = compute_region_weights(self.flat, self.structured, self.valid)
        self.assertFalse(weights.rgb_enabled)
        self.assertTrue(weights.ir_enabled)
        np.testing.assert_allclose(weights.ir, 1.0)
        np.testing.assert_allclose(weights.rgb + weights.ir, 1.0)

    def test_random_noise_does_not_win_on_entropy(self):
        noise = np.random.default_rng(4).integers(0, 256, self.flat.shape, np.uint8)
        self.assertGreater(image_entropy(noise), 7.0)
        weights = compute_region_weights(noise, self.structured, self.valid)
        self.assertFalse(weights.rgb_enabled)
        self.assertTrue(weights.ir_enabled)

    def test_one_high_entropy_ir_region_keeps_global_ir_weight_positive(self):
        noise = np.random.default_rng(7).normal(0, 1, self.flat.shape).astype(np.float32)
        rgb = cv2.normalize(
            cv2.GaussianBlur(noise, (0, 0), 2.0), None, 20, 230, cv2.NORM_MINMAX
        ).astype(np.uint8)
        ir = np.full(self.flat.shape, 100, np.uint8)
        ir[48:80, 48:80] = rgb[48:80, 48:80]
        weights = compute_region_weights(rgb, ir, self.valid)
        self.assertGreater(weights.ir_rescued_regions, 0)
        self.assertGreater(weights.rgb, 0.0)
        self.assertGreater(weights.ir, 0.0)
        self.assertAlmostEqual(weights.rgb + weights.ir, 1.0)

    def test_invalid_ir_border_cannot_receive_weight(self):
        valid = np.zeros_like(self.valid)
        weights = compute_region_weights(self.structured, self.structured, valid)
        self.assertEqual(weights.ir, 0.0)
        self.assertEqual(weights.rgb, 1.0)

    def test_feature_layer_weights_remain_complementary(self):
        weights = compute_region_weights(self.flat, self.structured, self.valid)
        rgb, ir = weights_for_feature_size(weights, 16, 16)
        self.assertEqual(rgb.shape, (16, 16))
        self.assertEqual(ir.dtype, np.float32)
        np.testing.assert_allclose(rgb + ir, 1.0)
        np.testing.assert_allclose(rgb, weights.rgb)
        np.testing.assert_allclose(ir, weights.ir)

    def test_rgb_trigger_brightens_heat_without_recoloring(self):
        rgb = np.full((128, 128, 3), 80, np.uint8)
        detail = np.zeros(self.flat.shape, np.float32)
        heat = np.ones(self.flat.shape, np.float32)
        off = enhance_ir_information_on_rgb(
            rgb, detail, detail, heat, np.zeros_like(self.valid), self.valid, 0.1
        )
        on = enhance_ir_information_on_rgb(
            rgb, detail, detail, heat, self.valid, self.valid, 0.1
        )
        self.assertLessEqual(np.max(np.abs(off.astype(int) - rgb.astype(int))), 1)
        self.assertGreater(float(np.mean(on.astype(int) - off.astype(int))), 60.0)
        self.assertLessEqual(int(np.max(on) - np.min(on)), 1)

    def test_hot_ir_edges_are_brightened_without_changing_rgb_chroma(self):
        rgb = np.full((128, 128, 3), (60, 100, 140), np.uint8)
        heat = np.zeros(self.flat.shape, np.float32)
        heat[32:96, 32:96] = 1.0
        detail = np.zeros(self.flat.shape, np.float32)
        result = enhance_ir_information_on_rgb(
            rgb, detail, detail, heat, self.valid, self.valid, 0.1
        )
        np.testing.assert_array_equal(result[10, 10], rgb[10, 10])
        self.assertGreater(float(np.mean(result[32, 40])), float(np.mean(result[10, 10])))
        self.assertGreater(float(np.mean(result[64, 64])), float(np.mean(result[10, 10])))
        original_chroma = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)[:, :, 1:]
        result_chroma = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)[:, :, 1:]
        self.assertLessEqual(
            int(np.max(np.abs(result_chroma.astype(int) - original_chroma.astype(int)))),
            2,
        )

    def test_ir_trigger_controls_rgb_edge_enhancement(self):
        ir = self.flat.copy()
        rgb_detail = np.zeros(self.flat.shape, np.float32)
        rgb_detail[:, 60:64] = 0.1
        edges = np.zeros(self.flat.shape, np.float32)
        edges[:, 60:64] = 1.0
        zero = np.zeros(self.flat.shape, np.float32)
        off = enhance_rgb_information_on_ir(
            ir, rgb_detail, zero, edges, zero, self.valid, 0.1
        )
        on = enhance_rgb_information_on_ir(
            ir, rgb_detail, zero, edges, self.valid, self.valid, 0.1
        )
        np.testing.assert_array_equal(off, ir)
        self.assertGreater(float(np.mean(on[:, 60:64])), float(np.mean(off[:, 60:64])))

    def test_invalid_black_and_white_ir_frames_are_repaired(self):
        zero = np.zeros(self.flat.shape, np.float32)
        valid = self.valid.copy()
        valid[:8, :] = valid[-8:, :] = 0.0
        valid[:, :8] = valid[:, -8:] = 0.0
        for border_value in (0, 255):
            ir = np.full(self.flat.shape, 100, np.uint8)
            ir[valid == 0] = border_value
            fused = enhance_rgb_information_on_ir(
                ir, zero, zero, zero, self.valid, valid, 0.1
            )
            self.assertLess(float(np.mean(np.abs(fused[valid == 0].astype(int) - 100))), 3.0)


if __name__ == "__main__":
    unittest.main()
