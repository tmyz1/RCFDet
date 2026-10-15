"""Run the RGB/IR fusion pipeline on separated M3FD scenes and summarize results."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from fft_H import (
    FusionConfig,
    denoise_gray,
    fuse_rgb_ir_features,
    gaussian_fft_highpass,
    ncc,
    save_outputs,
    to_gray,
)


DEFAULT_IDS = (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4199)


def thumbnail(image: np.ndarray, label: str, size=(256, 192)) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    thumb = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (0, 0), (size[0], 25), (0, 0, 0), -1)
    cv2.putText(thumb, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return thumb


def measure(outputs: dict, stats: dict, image_id: str) -> dict:
    rgb = to_gray(outputs["00_rgb_original"]).astype(np.float32)
    rgb_result = to_gray(outputs["08_fused_back_to_rgb"]).astype(np.float32)
    ir = to_gray(outputs["04_ir_smooth_aligned"]).astype(np.float32)
    ir_result = to_gray(outputs["09_fused_back_to_ir"]).astype(np.float32)
    binary = outputs["05_fused_unique_binary"] > 0
    height, width = binary.shape
    border = np.ones_like(binary, dtype=bool)
    border[56:height - 56, 56:width - 56] = False
    quiet = cv2.dilate(binary.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    quiet &= cv2.Canny(rgb.astype(np.uint8), 40, 100) == 0
    lap_original = np.abs(cv2.Laplacian(rgb, cv2.CV_32F))
    lap_result = np.abs(cv2.Laplacian(rgb_result, cv2.CV_32F))
    config = FusionConfig()
    aligned_feature = gaussian_fft_highpass(
        denoise_gray(ir.astype(np.uint8), config),
        config.fft_sigma,
        config.fft_pad,
    )
    center = np.s_[60:height - 60, 60:width - 60]
    rgb_feature = outputs["02_rgb_gaussian_highpass"][center]
    return {
        "id": image_id,
        "binary_density": round(float(np.mean(binary)), 4),
        "border_white_ratio": round(float(np.mean(binary[border])), 4),
        "rgb_mean_abs_delta": round(float(np.mean(np.abs(rgb_result - rgb))), 2),
        "ir_mean_abs_delta": round(float(np.mean(np.abs(ir_result - ir))), 2),
        "quiet_laplacian_before": round(float(np.mean(lap_original[quiet])), 2),
        "quiet_laplacian_after": round(float(np.mean(lap_result[quiet])), 2),
        "rgb_voted_blocks": int(stats["unique_rgb_selected_blocks"]),
        "ir_voted_blocks": int(stats["unique_ir_selected_blocks"]),
        "local_ncc": round(stats["mean_match_ncc"], 3),
        "alignment_ncc_before": round(ncc(rgb_feature, outputs["03_ir_gaussian_highpass"][center]), 3),
        "alignment_ncc_after": round(ncc(rgb_feature, aligned_feature[center]), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\data\M3FD\cocodataset\images"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "fft_feature_fusion_ncc_fix_check")
    parser.add_argument("--ids", type=int, nargs="+", default=DEFAULT_IDS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sheets = []
    for number in args.ids:
        image_id = f"{number:05d}"
        rgb = cv2.imread(str(args.dataset_root / "visible" / f"{image_id}.png"))
        ir = cv2.imread(str(args.dataset_root / "infrared" / f"{image_id}.png"))
        if rgb is None or ir is None:
            raise FileNotFoundError(f"Missing RGB/IR pair {image_id}")
        outputs, stats = fuse_rgb_ir_features(rgb, ir, FusionConfig())
        case_dir = args.output_dir / image_id
        case_dir.mkdir(parents=True, exist_ok=True)
        save_outputs(case_dir, outputs)
        row = measure(outputs, stats, image_id)
        rows.append(row)
        print(row, flush=True)
        sheets.append(np.hstack([
            thumbnail(outputs[key], f"{image_id} {label}")
            for key, label in (
                ("00_rgb_original", "RGB"),
                ("08_fused_back_to_rgb", "RGB enhanced"),
                ("04_ir_smooth_aligned", "IR aligned"),
                ("09_fused_back_to_ir", "IR enhanced"),
                ("05_fused_unique_binary", "Unique binary"),
            )
        ]))
    with (args.output_dir / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(args.output_dir / "validation_contact_sheet.png"), np.vstack(sheets))


if __name__ == "__main__":
    main()
