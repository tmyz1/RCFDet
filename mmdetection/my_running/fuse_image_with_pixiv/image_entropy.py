"""Region-level entropy and quality weights for aligned visible/thermal images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EntropyConfig:
    quality_threshold: float = 0.68
    ir_rescue_entropy: float = 5.0
    ir_rescue_quality: float = 0.55


@dataclass
class RegionWeights:
    rgb: float
    ir: float
    rgb_entropy: np.ndarray
    ir_entropy: np.ndarray
    rgb_quality: float
    ir_quality: float
    rgb_enabled: bool
    ir_enabled: bool
    ir_rescued_regions: int


def image_entropy(gray: np.ndarray, valid: np.ndarray | None = None) -> float:
    """计算灰度图像的 Shannon 信息熵，单位为 bit，用于衡量像素灰度分布的不确定性：灰度越集中，熵越低；灰度分布越均匀，熵越高"""
    pixels = gray[valid] if valid is not None else gray.ravel()
    if pixels.size == 0:
        return 0.0
    hist = cv2.calcHist([np.ascontiguousarray(pixels)], [0], None, [256], [0, 256])
    probabilities = hist.ravel().astype(np.float64) / pixels.size
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _structure_and_noise(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """通过高斯平滑和 Sobel 梯度提取灰度图的结构强度，并计算原图与平滑图的绝对差，反映噪声及细节变化"""
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    residual = cv2.absdiff(gray, smooth).astype(np.float32)
    return cv2.magnitude(gx, gy), residual


def _quality(entropy: float, gradient: float, residual: float) -> float:
    """结合信息熵、梯度强度和梯度与残差的比值，计算 0～1 的质量评分，降低噪声造成的虚高评分"""
    information = np.clip(entropy / 6.0, 0.0, 1.0)
    structure = np.clip(gradient / 30.0, 0.0, 1.0)
    coherence = np.clip((gradient / max(residual, 1e-6) - 2.5) / 2.5, 0.0, 1.0)
    return float(information * (0.70 + 0.30 * structure) * coherence)


def _positions(length: int, block: int, stride: int) -> list[int]:
    """按指定步长生成滑动窗口的起始位置，并补上末端位置，确保覆盖边界"""
    positions = list(range(0, length - block + 1, stride))
    if positions[-1] != length - block:
        positions.append(length - block)
    return positions


def compute_region_weights(
    rgb_gray: np.ndarray,
    aligned_ir_gray: np.ndarray,
    ir_validity: np.ndarray,
    block: int = 32,
    stride: int = 16,
    config: EntropyConfig | None = None,
) -> RegionWeights:
    """Compare aligned regions and return one complementary weight pair per image.

    The image-level gate uses the median quality of all regions. If both modalities
    miss the gate, the better one remains available. High-information IR regions
    override an image-level IR rejection, except at invalid sensor boundaries.
    """
    settings = EntropyConfig() if config is None else config
    if rgb_gray.ndim != 2 or rgb_gray.dtype != np.uint8:
        raise ValueError("rgb_gray must be a uint8 grayscale image")
    if aligned_ir_gray.shape != rgb_gray.shape or aligned_ir_gray.dtype != np.uint8:
        raise ValueError("aligned_ir_gray must be a matching uint8 grayscale image")
    if ir_validity.shape != rgb_gray.shape:
        raise ValueError("ir_validity must match the image dimensions")
    if block < 1 or stride < 1:
        raise ValueError("block and stride must be positive")

    height, width = rgb_gray.shape
    block = min(block, height, width)
    ys = _positions(height, block, stride)
    xs = _positions(width, block, stride)
    ir_valid = np.asarray(ir_validity, dtype=np.float32) >= 0.9
    rgb_gradient, rgb_residual = _structure_and_noise(rgb_gray)
    ir_gradient, ir_residual = _structure_and_noise(aligned_ir_gray)
    shape = (len(ys), len(xs))
    rgb_entropy = np.zeros(shape, dtype=np.float32)
    ir_entropy = np.zeros(shape, dtype=np.float32)
    rgb_quality_grid = np.zeros(shape, dtype=np.float32)
    ir_quality_grid = np.zeros(shape, dtype=np.float32)
    valid_grid = np.zeros(shape, dtype=bool)

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            region = np.s_[y:y + block, x:x + block]
            valid = ir_valid[region]
            rgb_entropy[row, col] = image_entropy(rgb_gray[region])
            rgb_quality_grid[row, col] = _quality(
                rgb_entropy[row, col], float(np.mean(rgb_gradient[region])),
                float(np.mean(rgb_residual[region])),
            )
            if float(np.mean(valid)) < 0.8:
                continue
            valid_grid[row, col] = True
            ir_entropy[row, col] = image_entropy(aligned_ir_gray[region], valid)
            ir_quality_grid[row, col] = _quality(
                ir_entropy[row, col], float(np.mean(ir_gradient[region][valid])),
                float(np.mean(ir_residual[region][valid])),
            )

    rgb_quality = float(np.median(rgb_quality_grid))
    ir_quality = float(np.median(ir_quality_grid[valid_grid])) if np.any(valid_grid) else 0.0
    rgb_enabled = rgb_quality >= settings.quality_threshold
    ir_enabled = ir_quality >= settings.quality_threshold
    if not rgb_enabled and not ir_enabled:
        rgb_enabled = rgb_quality > ir_quality
        ir_enabled = not rgb_enabled and bool(np.any(valid_grid))
        if not ir_enabled:
            rgb_enabled = True

    rescue = (
        valid_grid
        & (ir_entropy >= settings.ir_rescue_entropy)
        & (ir_quality_grid >= settings.ir_rescue_quality)
    )
    rescued_regions = int(np.count_nonzero(rescue)) if not ir_enabled else 0
    if np.any(valid_grid):
        entropy_sum = np.maximum(rgb_entropy + ir_entropy, 1e-6)
        rgb_share = rgb_entropy / entropy_sum
        ir_share = ir_entropy / entropy_sum
        rgb_score = float(np.mean((rgb_quality_grid * rgb_share)[valid_grid]))
        ir_score = float(np.mean((ir_quality_grid * ir_share)[valid_grid]))
    else:
        rgb_score = rgb_quality
        ir_score = 0.0

    if rgb_enabled and ir_enabled:
        ir_weight = ir_score / max(rgb_score + ir_score, 1e-6)
    elif rgb_enabled and rescued_regions:
        rescue_fraction = rescued_regions / max(int(np.count_nonzero(valid_grid)), 1)
        rescue_score = max(
            0.05,
            float(np.mean(ir_quality_grid[rescue] * ir_share[rescue]))
            * np.sqrt(rescue_fraction),
        )
        ir_weight = rescue_score / max(rgb_score + rescue_score, 1e-6)
    else:
        ir_weight = 1.0 if ir_enabled else 0.0
    ir_weight = float(np.clip(ir_weight, 0.0, 1.0))
    rgb_weight = 1.0 - ir_weight
    return RegionWeights(
        rgb=rgb_weight,
        ir=ir_weight,
        rgb_entropy=rgb_entropy,
        ir_entropy=ir_entropy,
        rgb_quality=rgb_quality,
        ir_quality=ir_quality,
        rgb_enabled=rgb_weight > 0.0,
        ir_enabled=ir_weight > 0.0,
        ir_rescued_regions=rescued_regions,
    )


def weights_for_feature_size(
    weights: RegionWeights, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast the image-level weights to float32 HxW feature maps."""
    if height < 1 or width < 1:
        raise ValueError("feature dimensions must be positive")
    rgb = np.full((height, width), weights.rgb, dtype=np.float32)
    ir = np.full((height, width), weights.ir, dtype=np.float32)
    return rgb, ir
