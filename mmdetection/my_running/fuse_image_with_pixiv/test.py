"""Display selected RGB/IR high-pass feature comparison images.

The script shows:
    original RGB/IR images, Gaussian high-pass details, direct unaligned detail
    addition, and direct detail addition after cross-modal block alignment.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from cross_modal_enhancement import (
    enhance_ir_information_on_rgb,
    enhance_rgb_information_on_ir,
)
from image_entropy import EntropyConfig, compute_region_weights


@dataclass(frozen=True)
class Config:
    """All experiment parameters are collected here for reproducibility."""

    fft_sigma: float = 10.0
    fft_pad: int = 64
    denoise_diameter: int = 5
    denoise_sigma_color: float = 22.0
    denoise_sigma_space: float = 5.0
    improved_block: int = 32
    improved_stride: int = 16
    search_radius: int = 8
    search_step: int = 2
    feature_ncc_weight: float = 0.30
    gradient_ncc_weight: float = 0.30
    ngf_weight: float = 0.40
    unmatched_edge_penalty: float = 0.20
    displacement_penalty: float = 0.05
    minimum_match_score: float = 0.10
    confidence_margin: float = 0.08
    flow_smooth_sigma: float = 4.0
    rgb_detail_gain: float = 0.20
    ir_detail_gain: float = 0.60
    max_luma_change: float = 0.12
    heat_luma_gain: float = 0.18
    heat_color_alpha: float = 0.38
    rgb_edge_gain: float = 0.28


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an array to finite uint8 without changing spatial dimensions."""
    array = np.nan_to_num(np.asarray(image)).astype(np.float32)
    if array.size and float(np.max(array)) <= 1.5:
        array *= 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a uint8 grayscale image."""
    image_u8 = to_uint8(image)
    if image_u8.ndim == 2:
        return image_u8
    return cv2.cvtColor(image_u8, cv2.COLOR_BGR2GRAY)


def resize_like(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resize source to reference size when the two sensors differ."""
    height, width = reference.shape[:2]
    if source.shape[:2] == (height, width):
        return source
    return cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)


def normalize_float(image: np.ndarray) -> np.ndarray:
    """Min-max normalize an array into float32 [0, 1]."""
    array = np.asarray(image, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def denoise(gray: np.ndarray, config: Config) -> np.ndarray:
    """采用双边滤波 bilateral filter来进行去噪"""
    return cv2.bilateralFilter(
        gray,
        config.denoise_diameter,
        config.denoise_sigma_color,
        config.denoise_sigma_space,
        borderType=cv2.BORDER_REFLECT_101,
    )


def gaussian_highpass_signed(
    gray: np.ndarray,
    sigma: float,
    pad: int,
) -> np.ndarray:
    """Return signed Gaussian high-pass detail in the original [0, 1] scale."""
    source = to_gray(gray).astype(np.float32) / 255.0
    if pad > 0:
        source = cv2.copyMakeBorder(
            source,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_REFLECT_101,
        )

    rows, cols = source.shape
    #二维离散傅里叶变换DFT
    spectrum = cv2.dft(source, flags=cv2.DFT_COMPLEX_OUTPUT)
    #把低频移动到频谱中心
    spectrum = np.fft.fftshift(spectrum)
    center_y, center_x = rows // 2, cols // 2
    yy, xx = np.ogrid[:rows, :cols]
    distance_squared = (yy - center_y) ** 2 + (xx - center_x) ** 2
    #构造高斯高通滤波器 mask
    highpass_mask = 1.0 - np.exp(
        -distance_squared / (2.0 * max(sigma, 1e-6) ** 2)
    )
    filtered = spectrum * highpass_mask[:, :, None]
    #把滤波后的频谱变回空间域图像
    detail = cv2.idft(
        np.fft.ifftshift(filtered),
        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT,
    )
    if pad > 0:
        detail = detail[pad:-pad, pad:-pad]
    return detail.astype(np.float32)


def shared_detail_scale(details: List[np.ndarray]) -> float:
    """用所有细节图的绝对值共同计算一个鲁棒归一化尺度，让不同高频图显示时亮度可比较，同时避免被极端异常值影响。"""
    absolute_values = np.concatenate(
        [np.abs(detail).ravel() for detail in details]
    )
    scale = float(np.percentile(absolute_values, 99.7))
    return max(scale, 1e-8)


def detail_visualization(detail: np.ndarray, scale: float) -> np.ndarray:
    """提取出来的 高频细节图 转成可以显示的 uint8 图像"""
    normalized = np.clip(np.abs(detail) / scale, 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def ncc(first: np.ndarray, second: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation."""
    first_float = np.asarray(first, dtype=np.float32)
    second_float = np.asarray(second, dtype=np.float32)
    first_centered = first_float - float(np.mean(first_float))
    second_centered = second_float - float(np.mean(second_float))
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator < 1e-8:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def image_gradients(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算图像的 梯度信息，也就是图像在水平和垂直方向上的变化强度"""
    source = np.asarray(image, dtype=np.float32)
    grad_x = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return grad_x, grad_y, magnitude


def ngf_similarity(
    rgb_grad_x: np.ndarray,
    rgb_grad_y: np.ndarray,
    ir_grad_x: np.ndarray,
    ir_grad_y: np.ndarray,
    edge_epsilon: float = 0.02,
) -> float:
    """Contrast-invariant Normalized Gradient Fields similarity."""
    dot_product = rgb_grad_x * ir_grad_x + rgb_grad_y * ir_grad_y
    rgb_norm = np.sqrt(
        rgb_grad_x * rgb_grad_x + rgb_grad_y * rgb_grad_y + edge_epsilon ** 2
    )
    ir_norm = np.sqrt(
        ir_grad_x * ir_grad_x + ir_grad_y * ir_grad_y + edge_epsilon ** 2
    )
    normalized_dot = dot_product / (rgb_norm * ir_norm + 1e-8)
    # Squaring makes the measure robust to contrast inversion between sensors.
    return float(np.mean(normalized_dot * normalized_dot))


def candidate_score(
    rgb_feature: np.ndarray,
    ir_feature: np.ndarray,
    rgb_edge_support: np.ndarray,
    rgb_grad_x: np.ndarray,
    rgb_grad_y: np.ndarray,
    rgb_grad_mag: np.ndarray,
    ir_grad_x: np.ndarray,
    ir_grad_y: np.ndarray,
    ir_grad_mag: np.ndarray,
    config: Config,
) -> float:
    """NCC variant combining feature strength and modality-robust structure."""
    feature_score = ncc(rgb_feature, ir_feature)
    gradient_score = ncc(rgb_grad_mag, ir_grad_mag)
    orientation_score = ngf_similarity(
        rgb_grad_x,
        rgb_grad_y,
        ir_grad_x,
        ir_grad_y,
    )
    ir_edge_energy = float(np.sum(ir_grad_mag))
    edge_support = (
        float(np.sum(ir_grad_mag * rgb_edge_support)) / ir_edge_energy
        if ir_edge_energy > 1e-6 else 1.0
    )
    return float(
        config.feature_ncc_weight * feature_score
        + config.gradient_ncc_weight * gradient_score
        + config.ngf_weight * orientation_score
        - config.unmatched_edge_penalty * (1.0 - edge_support)
    )


def search_positions(origin: int, start: int, end: int, step: int) -> List[int]:
    """Create a sampled search axis that always includes the zero-offset point."""
    positions = list(range(start, end + 1, max(1, step)))
    positions.append(origin)
    return sorted(set(positions))


def axis_positions(length: int, block: int, stride: int) -> List[int]:
    """Generate overlapping block origins while covering the final border."""
    if length <= block:
        return [0]
    positions = list(range(0, length - block + 1, max(1, stride)))
    final_position = length - block
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def sensor_border_weight(ir_gray: np.ndarray) -> np.ndarray:
    """Fade sensor boundaries when the IR image has a dark or bright frame."""
    gray = to_gray(ir_gray)
    height, width = gray.shape
    inset = max(1, min(16, height // 4, width // 4))
    border_pairs = (
        (gray[0, :], gray[inset, :]),
        (gray[-1, :], gray[-1 - inset, :]),
        (gray[:, 0], gray[:, inset]),
        (gray[:, -1], gray[:, -1 - inset]),
    )
    has_sensor_frame = any(
        (
            float(np.mean(edge <= 8)) > 0.7
            and float(np.median(inner)) - float(np.median(edge)) > 16.0
        ) or (
            float(np.mean(edge >= 247)) > 0.7
            and float(np.median(edge)) - float(np.median(inner)) > 16.0
        )
        for edge, inner in border_pairs
    )
    if not has_sensor_frame:
        return np.ones((height, width), dtype=np.float32)

    fade = min(64, max(8, round(min(height, width) * 0.08)))
    dead = fade // 3
    yy, xx = np.ogrid[:height, :width]
    distance = np.minimum(
        np.minimum(yy, height - 1 - yy),
        np.minimum(xx, width - 1 - xx),
    )
    weight = np.clip((distance - dead) / max(fade - dead, 1), 0.0, 1.0)
    return (weight * weight * (3.0 - 2.0 * weight)).astype(np.float32)


def estimate_smooth_flow(
    rgb_detail: np.ndarray,
    ir_detail: np.ndarray,
    rgb_gray: np.ndarray,
    ir_gray: np.ndarray,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a confidence-weighted dense flow from overlapping block matches."""
    rgb_feature = normalize_float(np.abs(rgb_detail))
    ir_feature = normalize_float(np.abs(ir_detail))
    rgb_source = rgb_gray.astype(np.float32) / 255.0
    ir_source = ir_gray.astype(np.float32) / 255.0
    #计算RGB和IR图像在水平和垂直方向上的变化强度
    rgb_grad_x, rgb_grad_y, rgb_grad_mag = image_gradients(rgb_source)
    ir_grad_x, ir_grad_y, ir_grad_mag = image_gradients(ir_source)
    rgb_grad_mag = normalize_float(rgb_grad_mag)
    ir_grad_mag = normalize_float(ir_grad_mag)
    rgb_edge_support = cv2.GaussianBlur(rgb_grad_mag, (0, 0), 1.5)
    support_scale = max(float(np.percentile(rgb_edge_support, 85)), 1e-6)
    rgb_edge_support = np.clip(rgb_edge_support / support_scale, 0.0, 1.0)

    height, width = rgb_feature.shape
    #分区域
    block = min(config.improved_block, height, width)
    y_positions = axis_positions(height, block, config.improved_stride)
    x_positions = axis_positions(width, block, config.improved_stride)
    window_1d = np.hanning(block).astype(np.float32)
    window = np.maximum(np.outer(window_1d, window_1d), 0.03)

    dx_sum = np.zeros((height, width), dtype=np.float32)
    dy_sum = np.zeros((height, width), dtype=np.float32)
    confidence_sum = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)

    #遍历 RGB 图像上的块位置
    for y in y_positions:
        for x in x_positions:
            #定义当前 RGB 块区域以及对应的特征以及边缘变化幅度
            region = np.s_[y:y + block, x:x + block]
            rgb_feature_block = rgb_feature[region]
            rgb_grad_x_block = rgb_grad_x[region]
            rgb_grad_y_block = rgb_grad_y[region]
            rgb_grad_mag_block = rgb_grad_mag[region]
            rgb_edge_support_block = rgb_edge_support[region]

            #IR 图像里的搜索范围
            y_start = max(0, y - config.search_radius)
            y_end = min(height - block, y + config.search_radius)
            x_start = max(0, x - config.search_radius)
            x_end = min(width - block, x + config.search_radius)
            #生成候选搜索坐标
            candidate_ys = search_positions(
                y, y_start, y_end, config.search_step
            )
            candidate_xs = search_positions(
                x, x_start, x_end, config.search_step
            )

            best_score = -2.0
            second_score = -2.0
            best_x, best_y = x, y
            #遍历IR的候选搜索坐标
            for candidate_y in candidate_ys:
                for candidate_x in candidate_xs:
                    candidate_region = np.s_[
                        candidate_y:candidate_y + block,
                        candidate_x:candidate_x + block,
                    ]
                    #计算当前 RGB 块和当前 IR 候选块的匹配分数
                    score = candidate_score(
                        rgb_feature_block,
                        ir_feature[candidate_region],
                        rgb_edge_support_block,
                        rgb_grad_x_block,
                        rgb_grad_y_block,
                        rgb_grad_mag_block,
                        ir_grad_x[candidate_region],
                        ir_grad_y[candidate_region],
                        ir_grad_mag[candidate_region],
                        config,
                    )
                    #计算候选块相对于原位置的偏移距离，并归一化
                    normalized_distance = np.hypot(
                        candidate_x - x, candidate_y - y
                    ) / max(float(config.search_radius), 1.0)
                    #给大位移候选一个惩罚
                    score -= config.displacement_penalty * normalized_distance
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_x, best_y = candidate_x, candidate_y
                    elif score > second_score:
                        second_score = score

            if best_score < config.minimum_match_score:
                continue

            score_confidence = np.clip(
                (best_score - config.minimum_match_score)
                / (1.0 - config.minimum_match_score),
                0.0,
                1.0,
            )
            margin_confidence = np.clip(
                (best_score - second_score) / config.confidence_margin,
                0.0,
                1.0,
            )
            confidence = float(
                np.sqrt(score_confidence * (0.25 + 0.75 * margin_confidence))
            )
            weighted_window = window * confidence
            dx_sum[region] += (best_x - x) * weighted_window
            dy_sum[region] += (best_y - y) * weighted_window
            confidence_sum[region] += weighted_window
            coverage[region] += window

    valid = confidence_sum > 1e-6
    dx_map = np.zeros_like(dx_sum)
    dy_map = np.zeros_like(dy_sum)
    dx_map[valid] = dx_sum[valid] / confidence_sum[valid]
    dy_map[valid] = dy_sum[valid] / confidence_sum[valid]
    confidence_map = np.divide(
        confidence_sum,
        np.maximum(coverage, 1e-6),
        out=np.zeros_like(confidence_sum),
    )
    confidence_map = np.clip(confidence_map, 0.0, 1.0)

    # Ambiguous blocks contribute smaller displacement; smoothing removes seams.
    dx_map *= confidence_map
    dy_map *= confidence_map
    dx_map = cv2.GaussianBlur(dx_map, (0, 0), config.flow_smooth_sigma)
    dy_map = cv2.GaussianBlur(dy_map, (0, 0), config.flow_smooth_sigma)
    return dx_map, dy_map


def remap_with_flow(
    image: np.ndarray,
    dx_map: np.ndarray,
    dy_map: np.ndarray,
) -> np.ndarray:
    """把 IR 图像或 IR 高频细节图按照估计的偏移量重新采样，让它尽量对齐 RGB"""
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        image,
        grid_x + dx_map.astype(np.float32),
        grid_y + dy_map.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def signed_detail_change(
    rgb_detail: np.ndarray,
    ir_detail: np.ndarray,
    ir_weight: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Return the same bounded detail change for both output modalities."""
    rgb_structure = cv2.GaussianBlur(np.abs(rgb_detail), (0, 0), 2.0)
    structure_scale = max(float(np.percentile(rgb_structure, 80)), 0.01)
    rgb_support = np.clip(rgb_structure / (2.0 * structure_scale), 0.0, 1.0)
    change = (
        config.rgb_detail_gain * rgb_detail
        + config.ir_detail_gain * (0.1 + 0.9 * rgb_support) * ir_detail
    ) * ir_weight
    return np.clip(change, -config.max_luma_change, config.max_luma_change)


def thermal_saliency(aligned_ir_gray: np.ndarray, ir_weight: np.ndarray) -> np.ndarray:
    """Find locally hot IR regions without treating invalid borders as heat."""
    source = to_gray(aligned_ir_gray).astype(np.float32) / 255.0
    confidence = np.clip(ir_weight.astype(np.float32), 0.0, 1.0)
    local_weight = cv2.GaussianBlur(confidence, (0, 0), 2.0)
    smooth = cv2.GaussianBlur(source * confidence, (0, 0), 2.0)
    smooth /= np.maximum(local_weight, 1e-6)
    broad_weight = cv2.GaussianBlur(confidence, (0, 0), 24.0)
    background = cv2.GaussianBlur(smooth * confidence, (0, 0), 24.0)
    background /= np.maximum(broad_weight, 1e-6)
    contrast = smooth - background
    valid = confidence > 0.9
    samples = contrast[valid] if np.any(valid) else contrast.ravel()
    intensities = smooth[valid] if np.any(valid) else smooth.ravel()
    peak = max(float(np.percentile(samples, 99.5)), 0.16)
    low = float(np.percentile(intensities, 55))
    high = float(np.percentile(intensities, 90))
    local_heat = np.clip((contrast - 0.035) / (peak - 0.035), 0.0, 1.0)
    global_heat = np.clip((smooth - low) / max(high - low, 1e-6), 0.0, 1.0)
    heat = local_heat * (0.2 + 0.8 * global_heat)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    heat = cv2.morphologyEx(heat, cv2.MORPH_OPEN, kernel)
    heat = cv2.GaussianBlur(heat, (0, 0), 1.5)
    heat = np.clip((heat - 0.08) / 0.92, 0.0, 1.0)
    return np.clip(heat * confidence, 0.0, 1.0).astype(np.float32)


def rgb_edge_highlight(rgb_gray: np.ndarray) -> np.ndarray:
    """Extract crisp RGB edges for the IR output."""
    gray = to_gray(rgb_gray)
    _, _, magnitude = image_gradients(gray)
    threshold = int(np.clip(np.percentile(magnitude, 80), 20, 60))
    edges = cv2.Canny(gray, threshold, int(threshold * 2.5), L2gradient=True)
    strength_low = max(float(np.percentile(magnitude, 85)), 20.0)
    strength_high = max(float(np.percentile(magnitude, 98)), strength_low + 40.0)
    strength = np.clip(
        (magnitude - strength_low) / (strength_high - strength_low), 0.0, 1.0
    )
    core = (edges.astype(np.float32) / 255.0) * strength
    halo = cv2.GaussianBlur(core, (0, 0), 0.8)
    return np.clip(0.65 * core + 0.35 * halo, 0.0, 1.0)


def fuse_details_into_rgb(
    rgb: np.ndarray,
    rgb_detail: np.ndarray,
    ir_detail: np.ndarray,
    ir_weight: np.ndarray,
    config: Config,
    heat_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add signed, bounded RGB/IR detail to RGB luminance while keeping color."""
    ycrcb = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_BGR2YCrCb)
    change = signed_detail_change(rgb_detail, ir_detail, ir_weight, config)
    ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0] + change, 0.0, 1.0)
    if heat_map is not None:
        heat = np.clip(heat_map, 0.0, 1.0)
        ycrcb[:, :, 0] += config.heat_luma_gain * heat * (1.0 - ycrcb[:, :, 0])
    fused = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    if heat_map is not None:
        alpha = (config.heat_color_alpha * heat)[:, :, None]
        warm_bgr = np.array([0.10, 0.48, 1.0], dtype=np.float32)
        fused = fused * (1.0 - alpha) + warm_bgr * alpha
    return np.rint(np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8)


def fuse_details_into_ir(
    aligned_ir_gray: np.ndarray,
    ir_weight: np.ndarray,
    detail_change: np.ndarray,
    rgb_detail: np.ndarray,
    rgb_edges: Optional[np.ndarray] = None,
    edge_gain: float = 0.28,
) -> np.ndarray:
    """Repair invalid IR boundaries before adding the shared signed detail."""
    ir_u8 = to_gray(aligned_ir_gray)
    invalid = ir_weight < 0.999
    if np.any(invalid):
        mask = (invalid.astype(np.uint8) * 255)
        filled = cv2.inpaint(ir_u8, mask, 3, cv2.INPAINT_TELEA)
        border_detail = np.clip(rgb_detail, -0.15, 0.15) * 255.0
        repaired = np.clip(filled.astype(np.float32) + border_detail, 0.0, 255.0)
        base = ir_weight * ir_u8 + (1.0 - ir_weight) * repaired
    else:
        base = ir_u8.astype(np.float32)
    edge_change = 0.0 if rgb_edges is None else edge_gain * rgb_edges
    result = np.clip(base / 255.0 + detail_change + edge_change, 0.0, 1.0)
    return np.rint(result * 255.0).astype(np.uint8)


def show_images(outputs: Dict[str, np.ndarray]) -> None:
    """Display uint8 visual results with OpenCV windows."""
    for name, image in outputs.items():
        cv2.imshow(name, to_uint8(image))


def run(args: argparse.Namespace) -> None:
    """Display the selected RGB/IR feature comparison images."""
    rgb_path = Path(args.rgb)
    ir_path = Path(args.ir)

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError("Cannot decode RGB image: {}".format(rgb_path))
    if ir is None:
        raise FileNotFoundError("Cannot decode IR image: {}".format(ir_path))
    ir = resize_like(ir, rgb)
    config = Config(
        fft_sigma=args.fft_sigma,
        improved_block=args.block,
        improved_stride=max(1, args.block // 2),
        search_radius=args.search_radius,
        search_step=args.search_step,
        rgb_detail_gain=args.rgb_gain,
        ir_detail_gain=args.ir_gain,
        heat_luma_gain=args.heat_gain,
        rgb_edge_gain=args.edge_gain,
    )
    #转成灰度图
    rgb_gray = to_gray(rgb)
    ir_gray = to_gray(ir)
    #采用双边滤波
    rgb_denoised = denoise(rgb_gray, config)
    ir_denoised = denoise(ir_gray, config)
    #高斯高通滤波
    rgb_detail = gaussian_highpass_signed(
        rgb_denoised, config.fft_sigma, config.fft_pad
    )
    ir_detail = gaussian_highpass_signed(
        ir_denoised, config.fft_sigma, config.fft_pad
    )
    feature_scale = shared_detail_scale([rgb_detail, ir_detail])
    rgb_feature_vis = detail_visualization(rgb_detail, feature_scale)
    ir_feature_vis = detail_visualization(ir_detail, feature_scale)
    direct_unaligned_detail = np.abs(rgb_detail) + np.abs(ir_detail)
    direct_unaligned_vis = detail_visualization(
        direct_unaligned_detail, feature_scale
    )
    # 对齐RGB图像和IR图像
    dx_map, dy_map = estimate_smooth_flow(
        rgb_detail,
        ir_detail,
        rgb_denoised,
        ir_denoised,
        config,
    )
    improved_ir_detail = remap_with_flow(ir_detail, dx_map, dy_map)
    ir_border_weight = sensor_border_weight(ir_gray)
    aligned_ir_weight = remap_with_flow(ir_border_weight, dx_map, dy_map)
    direct_aligned_detail = (
        np.abs(rgb_detail) + aligned_ir_weight * np.abs(improved_ir_detail)
    )
    direct_aligned_vis = detail_visualization(
        direct_aligned_detail, feature_scale
    )
    aligned_ir_gray = remap_with_flow(ir_gray, dx_map, dy_map)
    region_weights = compute_region_weights(
        rgb_denoised,
        aligned_ir_gray,
        aligned_ir_weight,
        block=config.improved_block,
        stride=config.improved_stride,
        config=EntropyConfig(
            quality_threshold=args.quality_threshold,
            ir_rescue_entropy=args.ir_rescue_entropy,
        ),
    )
    print(region_weights.ir, region_weights.rgb)
    heat_map = thermal_saliency(aligned_ir_gray, aligned_ir_weight)
    rgb_edges = rgb_edge_highlight(rgb_denoised)
    #针对RGB图像增强IR细节
    fused_color = enhance_ir_information_on_rgb(
        rgb, rgb_detail, improved_ir_detail, heat_map,
        region_weights.rgb, aligned_ir_weight, feature_scale,
    )
    #针对IR图像增强RGB细节
    fused_ir = enhance_rgb_information_on_ir(
        aligned_ir_gray, rgb_detail, improved_ir_detail, rgb_edges,
        region_weights.ir, aligned_ir_weight, feature_scale,
    )
    # aligned_vis_bgr = cv2.cvtColor(direct_aligned_vis, cv2.COLOR_GRAY2BGR)
    # fused_ir_new = cv2.addWeighted(fused_color, 0.8, aligned_vis_bgr, 0.2, 0)
    fused_ir_new = cv2.addWeighted(ir, 0.5, rgb, 0.5, 0)

    print(
        "Region quality RGB={:.3f} ({}), IR={:.3f} ({}), IR rescue regions={}".format(
            region_weights.rgb_quality,
            "enabled" if region_weights.rgb_enabled else "off",
            region_weights.ir_quality,
            "enabled" if region_weights.ir_enabled else "off",
            region_weights.ir_rescued_regions,
        )
    )

    outputs = {
        "00_rgb_original": rgb,
        "01_ir_original": ir,
        "02_rgb_gaussian_highpass": rgb_feature_vis,
        "03_ir_gaussian_highpass": ir_feature_vis,
        "03a_direct_unaligned_add_feature": direct_unaligned_vis,
        "03b_direct_aligned_add_feature": direct_aligned_vis,
        "04_rgb_ir_fused_color": fused_color,
        "05_ir_rgb_fused_gray": fused_ir,
        "06_ir_rgb_fused_gray": fused_ir_new,
    }
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, image in outputs.items():
            cv2.imwrite(str(save_dir / "{}.png".format(name)), to_uint8(image))
        with (save_dir / "adaptive_weights.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "rgb": region_weights.rgb,
                    "ir": region_weights.ir,
                    "rgb_quality": region_weights.rgb_quality,
                    "ir_quality": region_weights.ir_quality,
                    "ir_rescued_regions": region_weights.ir_rescued_regions,
                },
                file,
                indent=2,
            )
    if not args.no_display:
        show_images(outputs)
        print("Press any key in an OpenCV image window to close all windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse experiment options."""
    parser = argparse.ArgumentParser(
        description="Display selected RGB/IR high-pass feature images."
    )
    parser.add_argument(
        "--rgb",
        default=r"C:\Users\23802\Desktop\LLVIP_raw_data\LLVIP_raw_data\LLVIP_raw_images\train\infrared\020001.jpg",
    )
    parser.add_argument(
        "--ir",
        default=r"C:\Users\23802\Desktop\LLVIP_raw_data\LLVIP_raw_data\LLVIP_raw_images\train\visible\020001.jpg",
    )
    parser.add_argument("--fft-sigma", type=float, default=10.0)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--search-step", type=int, default=2)
    parser.add_argument("--rgb-gain", type=float, default=0.20)
    parser.add_argument("--ir-gain", type=float, default=0.60)
    parser.add_argument("--heat-gain", type=float, default=0.18)
    parser.add_argument("--edge-gain", type=float, default=0.28)
    parser.add_argument("--quality-threshold", type=float, default=0.68)
    parser.add_argument("--ir-rescue-entropy", type=float, default=5.0)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
