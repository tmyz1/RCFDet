import argparse
from pathlib import Path

import cv2
import numpy as np


def normalize_u8(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    return np.uint8((x - mn) / (mx - mn) * 255.0)


def normalize_float(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_like(src, ref):
    h, w = ref.shape[:2]
    if src.shape[:2] == (h, w):
        return src
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)


def warp_image(img, dx, dy, border=cv2.BORDER_REFLECT_101):
    h, w = img.shape[:2]
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border,
    )


def gaussian_fft_highpass(gray, sigma=10.0, pad=64):
    """Gaussian high-pass filter with reflect padding to reduce FFT ringing."""
    gray = gray.astype(np.float32) / 255.0
    if pad > 0:
        src = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    else:
        src = gray

    rows, cols = src.shape
    dft = cv2.dft(src, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    vv, uu = np.meshgrid(v, u)
    dist2 = (uu - crow) ** 2 + (vv - ccol) ** 2
    mask = 1.0 - np.exp(-dist2 / (2.0 * sigma * sigma))
    mask = np.repeat(mask[:, :, None], 2, axis=2)

    filtered = dft_shift * mask
    inv = cv2.idft(np.fft.ifftshift(filtered))
    mag = cv2.magnitude(inv[:, :, 0], inv[:, :, 1])

    if pad > 0:
        mag = mag[pad:-pad, pad:-pad]
    return normalize_u8(mag)


def ncc(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.sum(a * b) / denom)


def edge_energy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(cv2.magnitude(gx, gy)))


def local_contrast(gray, block=16):
    h, w = gray.shape
    vals = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            vals.append(float(np.std(gray[y : min(y + block, h), x : min(x + block, w)])))
    return float(np.mean(vals))


def estimate_global_shift(rgb_feature, ir_feature, max_shift=24):
    """Estimate a global translation and choose the sign that best matches RGB."""
    rgb = normalize_float(cv2.GaussianBlur(rgb_feature, (0, 0), 1.2))
    ir = normalize_float(cv2.GaussianBlur(ir_feature, (0, 0), 1.2))

    shift, response = cv2.phaseCorrelate(rgb.astype(np.float32), ir.astype(np.float32))
    sx, sy = float(shift[0]), float(shift[1])
    sx = float(np.clip(sx, -max_shift, max_shift))
    sy = float(np.clip(sy, -max_shift, max_shift))

    candidates = [(0.0, 0.0), (sx, sy), (-sx, -sy)]
    best = (0.0, 0.0)
    best_score = -1e9
    for dx, dy in candidates:
        warped = warp_image(ir, dx, dy)
        score = ncc(rgb, warped)
        if score > best_score:
            best_score = score
            best = (dx, dy)
    return best[0], best[1], response, best_score


def build_local_flow(
    rgb_feature,
    ir_feature,
    block=16,
    search_radius=5,
    search_step=1,
    min_ncc=0.18,
    response_margin=0.08,
    ncc_weight=0.10,
    alpha_scale=0.45,
):
    """Build smooth local offsets and confidence instead of pasting blocks."""
    rgb = normalize_float(rgb_feature)
    ir = normalize_float(ir_feature)
    h, w = rgb.shape

    alpha = np.zeros((h, w), dtype=np.float32)
    dx_map = np.zeros((h, w), dtype=np.float32)
    dy_map = np.zeros((h, w), dtype=np.float32)
    selected = 0
    rejected = 0
    ncc_values = []

    for y in range(0, h, block):
        for x in range(0, w, block):
            y2 = min(y + block, h)
            x2 = min(x + block, w)
            bh, bw = y2 - y, x2 - x
            rgb_block = rgb[y:y2, x:x2]
            rgb_score = float(np.mean(rgb_block))

            y_start = max(0, y - search_radius)
            y_end = min(h - bh, y + search_radius)
            x_start = max(0, x - search_radius)
            x_end = min(w - bw, x + search_radius)
            yy_list = list(range(y_start, y_end + 1, max(1, search_step)))
            xx_list = list(range(x_start, x_end + 1, max(1, search_step)))
            if y not in yy_list:
                yy_list.append(y)
            if x not in xx_list:
                xx_list.append(x)

            best_score = -1e9
            best_ncc = -1.0
            best_x = x
            best_y = y
            for yy in sorted(set(yy_list)):
                for xx in sorted(set(xx_list)):
                    ir_block = ir[yy : yy + bh, xx : xx + bw]
                    cur_ncc = ncc(rgb_block, ir_block)
                    if cur_ncc < min_ncc:
                        continue
                    score = float(np.mean(ir_block)) + ncc_weight * cur_ncc
                    if score > best_score:
                        best_score = score
                        best_ncc = cur_ncc
                        best_x = xx
                        best_y = yy

            advantage = best_score - rgb_score
            if advantage > response_margin and best_ncc >= min_ncc:
                cur_alpha = np.clip((advantage / alpha_scale) * best_ncc, 0.0, 1.0)
                alpha[y:y2, x:x2] = cur_alpha
                dx_map[y:y2, x:x2] = best_x - x
                dy_map[y:y2, x:x2] = best_y - y
                selected += 1
                ncc_values.append(best_ncc)
            else:
                rejected += 1

    if np.any(alpha > 0):
        dx_map = cv2.medianBlur(dx_map.astype(np.float32), 5)
        dy_map = cv2.medianBlur(dy_map.astype(np.float32), 5)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 3.0)
        alpha = np.clip(alpha, 0.0, 1.0)

    stats = {
        "selected_blocks": selected,
        "rejected_blocks": rejected,
        "selected_ratio": selected / max(selected + rejected, 1),
        "mean_local_ncc": float(np.mean(ncc_values)) if ncc_values else 0.0,
        "mean_abs_dx": float(np.mean(np.abs(dx_map[alpha > 0.05]))) if np.any(alpha > 0.05) else 0.0,
        "mean_abs_dy": float(np.mean(np.abs(dy_map[alpha > 0.05]))) if np.any(alpha > 0.05) else 0.0,
    }
    return dx_map, dy_map, alpha, stats


def remap_with_flow(img, dx_map, dy_map):
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + dx_map.astype(np.float32)
    map_y = ys + dy_map.astype(np.float32)
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def fuse_gaussian_fft(
    rgb_img,
    ir_img,
    fft_sigma=10.0,
    fft_pad=64,
    global_max_shift=24,
    block=16,
    search_radius=5,
    search_step=1,
    min_ncc=0.18,
    response_margin=0.08,
    ncc_weight=0.10,
    detail_weight=0.45,
    ir_luma_weight=0.35,
    global_min_ncc=0.18,
    global_min_phase_response=0.03,
):
    rgb_gray = to_gray(rgb_img)
    ir_gray = to_gray(ir_img)

    rgb_feature = gaussian_fft_highpass(rgb_gray, sigma=fft_sigma, pad=fft_pad)
    ir_feature = gaussian_fft_highpass(ir_gray, sigma=fft_sigma, pad=fft_pad)

    gdx, gdy, phase_response, global_ncc = estimate_global_shift(
        rgb_feature, ir_feature, max_shift=global_max_shift
    )
    if phase_response < global_min_phase_response or global_ncc < global_min_ncc:
        gdx, gdy = 0.0, 0.0
    ir_feature_global = warp_image(ir_feature, gdx, gdy)
    ir_gray_global = warp_image(ir_gray, gdx, gdy)

    dx_map, dy_map, alpha, local_stats = build_local_flow(
        rgb_feature,
        ir_feature_global,
        block=block,
        search_radius=search_radius,
        search_step=search_step,
        min_ncc=min_ncc,
        response_margin=response_margin,
        ncc_weight=ncc_weight,
    )

    ir_feature_warped = remap_with_flow(ir_feature_global, dx_map, dy_map)
    ir_gray_warped = remap_with_flow(ir_gray_global, dx_map, dy_map)
    fused_float = (1.0 - alpha) * normalize_float(rgb_feature) + alpha * normalize_float(ir_feature_warped)
    fused_feature = normalize_u8(fused_float)

    detail = cv2.GaussianBlur(fused_feature, (0, 0), 0.6)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    rgb_v = hsv[:, :, 2]
    ir_hot = normalize_float(ir_gray_warped)
    hot_start = float(np.percentile(ir_hot, 72))
    hot_end = float(np.percentile(ir_hot, 98))
    ir_hot = np.clip((ir_hot - hot_start) / (hot_end - hot_start + 1e-8), 0.0, 1.0)
    alpha_body = cv2.GaussianBlur(alpha, (0, 0), 6.0)
    luma_alpha = np.clip(np.maximum(alpha, alpha_body * ir_hot), 0.0, 1.0)
    ir_positive_luma = np.maximum(ir_gray_warped.astype(np.float32) - rgb_v, 0.0)
    fused_v = (
        rgb_v
        + detail_weight * detail.astype(np.float32)
        + ir_luma_weight * luma_alpha * ir_positive_luma
    )
    hsv[:, :, 2] = np.clip(fused_v, 0, 255)
    fused_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    confidence = normalize_u8(alpha)
    offset_mag = normalize_u8(np.sqrt(dx_map * dx_map + dy_map * dy_map))

    base_gray = rgb_gray
    fused_gray = to_gray(fused_color)
    stats = {
        **local_stats,
        "global_dx": gdx,
        "global_dy": gdy,
        "phase_response": float(phase_response),
        "global_ncc": float(global_ncc),
        "edge_gain": edge_energy(fused_gray) - edge_energy(base_gray),
        "contrast_gain": local_contrast(fused_gray, block=block) - local_contrast(base_gray, block=block),
    }

    outputs = {
        "rgb_feature": rgb_feature,
        "ir_feature": ir_feature,
        "ir_feature_global_aligned": ir_feature_global,
        "ir_gray_global_aligned": ir_gray_global,
        "ir_feature_warped": ir_feature_warped,
        "ir_gray_warped": ir_gray_warped,
        "fused_feature": fused_feature,
        "fused_color": fused_color,
        "confidence": confidence,
        "offset_magnitude": offset_mag,
    }
    return outputs, stats


def add_label(img, text, label_h=30):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas = np.full((img.shape[0] + label_h, img.shape[1], 3), 245, dtype=np.uint8)
    canvas[label_h:] = img
    cv2.putText(canvas, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def resize_cell(img, width):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)


def make_grid(items, cell_width=320):
    cells = [add_label(resize_cell(img, cell_width), label) for img, label in items]
    min_h = min(c.shape[0] for c in cells)
    cells = [cv2.resize(c, (c.shape[1], min_h), interpolation=cv2.INTER_AREA) for c in cells]
    return np.hstack(cells)


def run(args):
    rgb_path = Path(args.rgb)
    ir_path = Path(args.ir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if ir is None:
        raise FileNotFoundError(ir_path)
    ir = resize_like(ir, rgb)

    outputs, stats = fuse_gaussian_fft(
        rgb,
        ir,
        fft_sigma=args.fft_sigma,
        fft_pad=args.fft_pad,
        global_max_shift=args.global_max_shift,
        block=args.block,
        search_radius=args.search_radius,
        search_step=args.search_step,
        min_ncc=args.min_ncc,
        response_margin=args.response_margin,
        ncc_weight=args.ncc_weight,
        detail_weight=args.detail_weight,
        ir_luma_weight=args.ir_luma_weight,
        global_min_ncc=args.global_min_ncc,
        global_min_phase_response=args.global_min_phase_response,
    )

    cv2.imwrite(str(out_dir / "00_rgb_original.png"), rgb)
    cv2.imwrite(str(out_dir / "00_ir_original.png"), ir)
    for name, img in outputs.items():
        cv2.imwrite(str(out_dir / f"gaussian_fft_{name}.png"), img)
    cv2.imwrite(str(out_dir / "gaussian_fft_rgb_filter_gray.png"), outputs["rgb_feature"])
    cv2.imwrite(str(out_dir / "gaussian_fft_ir_filter_gray.png"), outputs["ir_feature"])
    cv2.imwrite(str(out_dir / "gaussian_fft_fused_feature_gray.png"), outputs["fused_feature"])

    grid = make_grid(
        [
            (rgb, "RGB original"),
            (ir, "IR original"),
            (outputs["rgb_feature"], "RGB Gaussian high-pass"),
            (outputs["ir_feature_global_aligned"], "IR aligned high-pass"),
            (outputs["fused_feature"], "soft fused feature"),
            (outputs["confidence"], "IR confidence"),
            (outputs["fused_color"], "final color"),
        ],
        cell_width=args.grid_cell_width,
    )
    cv2.imwrite(str(out_dir / "gaussian_fft_alignment_fusion_grid.png"), grid)

    lines = [
        "Gaussian FFT high-pass fusion with global alignment, smooth local flow, and soft fusion.",
        f"RGB: {rgb_path}",
        f"IR:  {ir_path}",
        f"global_dx={stats['global_dx']:.3f}, global_dy={stats['global_dy']:.3f}",
        f"phase_response={stats['phase_response']:.4f}, global_ncc={stats['global_ncc']:.4f}",
        f"selected_ratio={stats['selected_ratio']:.4f}, mean_local_ncc={stats['mean_local_ncc']:.4f}",
        f"mean_abs_dx={stats['mean_abs_dx']:.4f}, mean_abs_dy={stats['mean_abs_dy']:.4f}",
        f"edge_gain={stats['edge_gain']:.4f}, contrast_gain={stats['contrast_gain']:.4f}",
    ]
    (out_dir / "gaussian_fft_fusion_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to: {out_dir.resolve()}")
    print("\n".join(lines))


def parse_args():
    default_rgb = r"D:\data\M3FD\cocodataset\images\visible\00000.png"
    default_ir = r"D:\data\M3FD\cocodataset\images\infrared\00000.png"
    default_out = Path(__file__).resolve().parent / "frequency_fusion_compare_vis"

    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", default=default_rgb)
    parser.add_argument("--ir", default=default_ir)
    parser.add_argument("--out-dir", default=str(default_out))
    parser.add_argument("--fft-sigma", type=float, default=10.0)
    parser.add_argument("--fft-pad", type=int, default=64)
    parser.add_argument("--global-max-shift", type=float, default=24.0)
    parser.add_argument("--global-min-ncc", type=float, default=0.18)
    parser.add_argument("--global-min-phase-response", type=float, default=0.03)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--search-radius", type=int, default=6)
    parser.add_argument("--search-step", type=int, default=1)
    parser.add_argument("--min-ncc", type=float, default=0.12)
    parser.add_argument("--response-margin", type=float, default=0.045)
    parser.add_argument("--ncc-weight", type=float, default=0.15)
    parser.add_argument("--detail-weight", type=float, default=1.05)
    parser.add_argument("--ir-luma-weight", type=float, default=1.00)
    parser.add_argument("--grid-cell-width", type=int, default=240)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
