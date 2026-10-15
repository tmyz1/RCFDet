"""Complementary detail enhancement driven by aligned region weights."""

import cv2
import numpy as np


def enhance_ir_information_on_rgb(
    rgb: np.ndarray,
    rgb_detail: np.ndarray,
    aligned_ir_detail: np.ndarray,
    heat_map: np.ndarray,
    rgb_trigger: np.ndarray,
    ir_validity: np.ndarray,
    detail_scale: float,
) -> np.ndarray:
    """Brighten hot IR regions and their edges without recoloring RGB."""
    trigger = np.clip(rgb_trigger * ir_validity, 0.0, 1.0)
    rgb_support = cv2.GaussianBlur(np.abs(rgb_detail), (0, 0), 1.2)
    unique_ir = np.clip(
        (np.abs(aligned_ir_detail) - 0.5 * rgb_support)
        / max(0.6 * detail_scale, 1e-6), 0.0, 1.0
    )
    heat = np.clip(heat_map, 0.0, 1.0).astype(np.float32)
    heat_edges = cv2.Canny(np.rint(heat * 255.0).astype(np.uint8), 40, 100)
    heat_edges = heat_edges.astype(np.float32) / 255.0
    heat_edges = np.clip(
        0.7 * heat_edges + 0.3 * cv2.GaussianBlur(heat_edges, (0, 0), 0.8),
        0.0, 1.0,
    )
    hot_support = np.clip(
        (cv2.GaussianBlur(heat, (0, 0), 2.0) - 0.3) / 0.5,
        0.0, 1.0,
    )
    bright_ir_detail = np.clip(
        aligned_ir_detail / max(detail_scale, 1e-6), 0.0, 1.0
    ) * unique_ir
    edge = np.maximum(heat_edges, bright_ir_detail) * hot_support * trigger
    heat *= trigger
    ycrcb = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0]
    luminance[:] = np.clip(
        luminance + (0.46 * heat + 0.28 * edge) * (1.0 - luminance),
        0.0, 1.0,
    )
    fused = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return np.rint(np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8)


def enhance_rgb_information_on_ir(
    aligned_ir_gray: np.ndarray,
    rgb_detail: np.ndarray,
    aligned_ir_detail: np.ndarray,
    rgb_edges: np.ndarray,
    ir_trigger: np.ndarray,
    ir_validity: np.ndarray,
    detail_scale: float,
) -> np.ndarray:
    """Repair sensor boundaries and show distinctive RGB edges where IR has weight."""
    ir_u8 = aligned_ir_gray.astype(np.uint8)
    validity = np.clip(ir_validity, 0.0, 1.0)
    if np.any(validity < 0.999):
        invalid = np.uint8(validity < 0.999) * 255
        repaired = cv2.inpaint(ir_u8, invalid, 3, cv2.INPAINT_TELEA)
        base = validity * ir_u8 + (1.0 - validity) * repaired
    else:
        base = ir_u8.astype(np.float32)
    ir_support = cv2.GaussianBlur(np.abs(aligned_ir_detail), (0, 0), 1.2)
    unique_rgb = np.clip(
        (np.abs(rgb_detail) - 0.4 * ir_support)
        / max(0.5 * detail_scale, 1e-6), 0.0, 1.0
    )
    trigger = np.clip(ir_trigger * validity, 0.0, 1.0)
    edge_density = cv2.GaussianBlur(
        (rgb_edges > 0.12).astype(np.float32), (0, 0), 3.0
    )
    sparse_support = np.clip((0.25 - edge_density) / 0.20, 0.25, 1.0)
    edge_boost = np.clip(
        0.26 * rgb_edges * unique_rgb * trigger * sparse_support, 0.0, 0.13
    )
    signed_change = np.clip(0.16 * rgb_detail * unique_rgb * trigger, -0.04, 0.04)
    fused = np.clip(base / 255.0 + edge_boost + signed_change, 0.0, 1.0)
    return np.rint(fused * 255.0).astype(np.uint8)
