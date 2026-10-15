import unittest
from dataclasses import replace

import cv2
import numpy as np

from my_running.fuse_image_with_pixiv.test import (
    Config,
    candidate_score,
    fuse_details_into_ir,
    fuse_details_into_rgb,
    image_gradients,
    remap_with_flow,
    rgb_edge_highlight,
    sensor_border_weight,
    thermal_saliency,
)


class FeatureFusionTest(unittest.TestCase):
    def test_unmatched_ir_edge_lowers_candidate_score(self):
        rgb = np.zeros((32, 32), dtype=np.float32)
        rgb[:, 12:] = 1.0
        aligned_ir = rgb.copy()
        extra_ir = rgb.copy()
        extra_ir[:, 24:] = 0.0

        rgb_gx, rgb_gy, rgb_mag = image_gradients(rgb)
        support = cv2.GaussianBlur(rgb_mag, (0, 0), 1.5)
        support /= max(float(support.max()), 1e-6)

        def score(ir, penalty):
            ir_gx, ir_gy, ir_mag = image_gradients(ir)
            return candidate_score(
                rgb, ir, support, rgb_gx, rgb_gy, rgb_mag,
                ir_gx, ir_gy, ir_mag,
                replace(Config(), unmatched_edge_penalty=penalty),
            )

        old_gap = score(aligned_ir, 0.0) - score(extra_ir, 0.0)
        new_gap = score(aligned_ir, 0.20) - score(extra_ir, 0.20)
        self.assertGreater(new_gap, old_gap + 0.03)

    def test_sensor_border_fades_without_changing_interior(self):
        ir = np.full((768, 1024), 100, dtype=np.uint8)
        ir[0, :] = 0
        ir[:, -1] = 0
        weight = sensor_border_weight(ir)
        self.assertEqual(weight.shape, (768, 1024))
        self.assertEqual(float(weight[0, 512]), 0.0)
        self.assertEqual(float(weight[10, 512]), 0.0)
        self.assertEqual(float(weight[384, 1023]), 0.0)
        self.assertEqual(float(weight[384, 512]), 1.0)
        self.assertTrue(np.all((weight >= 0.0) & (weight <= 1.0)))

    def test_ir_without_sensor_frame_keeps_boundary(self):
        ir = np.full((96, 128), 100, dtype=np.uint8)
        weight = sensor_border_weight(ir)
        self.assertTrue(np.all(weight == 1.0))

    def test_bright_sensor_frame_fades(self):
        ir = np.full((96, 128), 100, dtype=np.uint8)
        ir[:, 0] = 255
        weight = sensor_border_weight(ir)
        self.assertEqual(float(weight[48, 0]), 0.0)
        self.assertEqual(float(weight[48, 64]), 1.0)

    def test_signed_ir_detail_preserves_color_and_masked_boundary(self):
        rgb = np.full((32, 32, 3), (50, 100, 150), dtype=np.uint8)
        rgb_detail = np.zeros((32, 32), dtype=np.float32)
        rgb_detail[:, :4] = 0.2
        rgb_detail[12:20, 12:20] = 0.05
        ir_detail = np.zeros((32, 32), dtype=np.float32)
        ir_detail[:, :4] = 0.5
        ir_detail[12:20, 12:16] = 0.1
        ir_detail[12:20, 16:20] = -0.1
        weight = np.ones((32, 32), dtype=np.float32)
        weight[:, :4] = 0.0
        fused = fuse_details_into_rgb(
            rgb, rgb_detail, ir_detail, weight, Config()
        )
        self.assertLessEqual(int(np.max(np.abs(fused[:, :4].astype(int) - rgb[:, :4]))), 1)
        self.assertGreater(int(fused[15, 14, 2]), int(rgb[15, 14, 2]))
        self.assertLess(int(fused[15, 18, 2]), int(rgb[15, 18, 2]))
        self.assertGreater(int(fused[15, 14, 2] - fused[15, 14, 0]), 80)

    def test_remap_uses_zero_outside_image(self):
        image = np.ones((8, 8), dtype=np.float32)
        dx = np.full((8, 8), 100.0, dtype=np.float32)
        dy = np.zeros((8, 8), dtype=np.float32)
        remapped = remap_with_flow(image, dx, dy)
        self.assertEqual(remapped.shape, image.shape)
        self.assertTrue(np.all(remapped == 0.0))

    def test_ir_output_repairs_black_and_white_sensor_borders(self):
        weight = np.ones((64, 64), dtype=np.float32)
        weight[:8] = 0.0
        weight[:, -8:] = 0.0
        detail = np.zeros((64, 64), dtype=np.float32)
        detail[24:40, 24:40] = 0.04
        for border_value in (0, 255):
            ir = np.full((64, 64), 100, dtype=np.uint8)
            ir[:8] = border_value
            ir[:, -8:] = border_value
            fused = fuse_details_into_ir(ir, weight, detail, np.zeros_like(detail))
            self.assertTrue(np.all(np.abs(fused[:8].astype(int) - 100) <= 5))
            self.assertTrue(np.all(np.abs(fused[:, -8:].astype(int) - 100) <= 5))
            self.assertGreater(int(fused[30, 30]), 100)

    def test_ir_repaired_boundary_uses_rgb_structure(self):
        ir = np.full((32, 32), 100, dtype=np.uint8)
        weight = np.ones((32, 32), dtype=np.float32)
        weight[:, :8] = 0.0
        rgb_detail = np.zeros((32, 32), dtype=np.float32)
        rgb_detail[8:24, :8] = 0.1
        result = fuse_details_into_ir(
            ir, weight, np.zeros_like(rgb_detail), rgb_detail
        )
        self.assertGreater(int(result[16, 4]), int(result[4, 4]) + 15)

    def test_hot_ir_region_is_highlighted_in_rgb(self):
        ir = np.full((96, 96), 80, dtype=np.uint8)
        ir[38:58, 38:58] = 230
        weight = np.ones((96, 96), dtype=np.float32)
        weight[:8] = 0.0
        heat = thermal_saliency(ir, weight)
        self.assertGreater(float(heat[48, 48]), 0.5)
        self.assertLess(float(heat[12, 12]), 0.05)
        self.assertEqual(float(heat[0, 48]), 0.0)

        rgb = np.full((96, 96, 3), (80, 100, 120), dtype=np.uint8)
        zero = np.zeros((96, 96), dtype=np.float32)
        fused = fuse_details_into_rgb(rgb, zero, zero, weight, Config(), heat)
        self.assertGreater(int(fused[48, 48, 2]), 155)
        self.assertLessEqual(int(np.max(np.abs(fused[12, 12].astype(int) - rgb[12, 12]))), 1)

    def test_rgb_edge_is_highlighted_in_ir(self):
        rgb = np.full((64, 64), 30, dtype=np.uint8)
        rgb[:, 32:] = 200
        edges = rgb_edge_highlight(rgb)
        self.assertGreater(float(edges[:, 31:33].max()), 0.5)
        self.assertLess(float(edges[:, :20].max()), 0.05)

        ir = np.full((64, 64), 100, dtype=np.uint8)
        weight = np.ones((64, 64), dtype=np.float32)
        zero = np.zeros((64, 64), dtype=np.float32)
        fused = fuse_details_into_ir(ir, weight, zero, zero, edges)
        self.assertGreater(int(fused[:, 31:33].max()), 145)
        self.assertEqual(int(fused[32, 10]), 100)


if __name__ == "__main__":
    unittest.main()
