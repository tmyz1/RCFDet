"""Save enhanced visible/infrared image pairs and their image-level weights."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import os
import shutil
import time
from pathlib import Path

import cv2

import test as fusion
from cross_modal_enhancement import (
    enhance_ir_information_on_rgb,
    enhance_rgb_information_on_ir,
)
from image_entropy import EntropyConfig, compute_region_weights


DEFAULT_INPUT_CANDIDATES = (
    Path("/root/autodl-tmp/M3FD/cocodataset/images"),
    Path(r"D:\data\M3FD\cocodataset\images"),
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def default_input_dir() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if (candidate / "visible").is_dir() and (candidate / "infrared").is_dir():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[0]


def process_pair(rgb_path: Path, ir_path: Path):
    """Run the same alignment and weighting pipeline as test.py for one pair."""
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    if rgb is None or ir is None:
        raise ValueError(f"Cannot decode image pair: {rgb_path}, {ir_path}")

    config = fusion.Config(improved_block=16, improved_stride=8)
    ir = fusion.resize_like(ir, rgb)
    rgb_gray = fusion.to_gray(rgb)
    ir_gray = fusion.to_gray(ir)
    rgb_denoised = fusion.denoise(rgb_gray, config)
    ir_denoised = fusion.denoise(ir_gray, config)
    rgb_detail = fusion.gaussian_highpass_signed(
        rgb_denoised, config.fft_sigma, config.fft_pad
    )
    ir_detail = fusion.gaussian_highpass_signed(
        ir_denoised, config.fft_sigma, config.fft_pad
    )
    dx_map, dy_map = fusion.estimate_smooth_flow(
        rgb_detail, ir_detail, rgb_denoised, ir_denoised, config
    )
    aligned_ir_detail = fusion.remap_with_flow(ir_detail, dx_map, dy_map)
    ir_validity = fusion.remap_with_flow(
        fusion.sensor_border_weight(ir_gray), dx_map, dy_map
    )
    aligned_ir_gray = fusion.remap_with_flow(ir_gray, dx_map, dy_map)
    weights = compute_region_weights(
        rgb_denoised,
        aligned_ir_gray,
        ir_validity,
        block=config.improved_block,
        stride=config.improved_stride,
        config=EntropyConfig(),
    )
    detail_scale = fusion.shared_detail_scale([rgb_detail, ir_detail])

    enhanced_rgb = None
    if weights.rgb > 0.0:
        heat_map = fusion.thermal_saliency(aligned_ir_gray, ir_validity)
        enhanced_rgb = enhance_ir_information_on_rgb(
            rgb, rgb_detail, aligned_ir_detail, heat_map,
            weights.rgb, ir_validity, detail_scale,
        )

    enhanced_ir = None
    if weights.ir > 0.0:
        rgb_edges = fusion.rgb_edge_highlight(rgb_denoised)
        enhanced_ir = enhance_rgb_information_on_ir(
            aligned_ir_gray, rgb_detail, aligned_ir_detail, rgb_edges,
            weights.ir, ir_validity, detail_scale,
        )
    return enhanced_rgb, enhanced_ir, weights


def save_image(source: Path, destination: Path, enhanced) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if enhanced is None:
        shutil.copy2(source, destination)
    elif not cv2.imwrite(str(destination), enhanced):
        raise OSError(f"Cannot save enhanced image: {destination}")


def process_one(name: Path, visible_dir: Path, infrared_dir: Path, destination: Path):
    rgb_source = visible_dir / name
    ir_source = infrared_dir / name
    rgb_result, ir_result, weights = process_pair(rgb_source, ir_source)
    save_image(rgb_source, destination / "visible" / name, rgb_result)
    save_image(ir_source, destination / "infrared" / name, ir_result)
    return name.as_posix(), float(weights.rgb), float(weights.ir)


def resolve_workers(requested_workers: int, image_count: int) -> int:
    if requested_workers < 0:
        raise ValueError("--workers must be 0 or a positive integer")
    if requested_workers == 0:
        requested_workers = os.cpu_count() or 1
    return max(1, min(requested_workers, image_count))


def process_dataset(input_dir: Path, output_dir: Path, workers: int = 1) -> None:
    source = input_dir.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Output directory must not overlap the input directory")
    visible_dir = source / "visible"
    infrared_dir = source / "infrared"
    if not visible_dir.is_dir() or not infrared_dir.is_dir():
        raise FileNotFoundError(
            "Input directory must contain visible/ and infrared/.\n"
            f"Checked input: {source}\n"
            f"Checked visible: {visible_dir}\n"
            f"Checked infrared: {infrared_dir}"
        )

    def image_names(folder: Path) -> set[Path]:
        return {
            path.relative_to(folder) for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }

    rgb_names = image_names(visible_dir)
    ir_names = image_names(infrared_dir)
    if not rgb_names:
        raise ValueError(f"No images found in {visible_dir}")
    if rgb_names != ir_names:
        only_rgb = sorted(rgb_names - ir_names)
        only_ir = sorted(ir_names - rgb_names)
        raise ValueError(
            f"Unpaired images: visible-only={only_rgb[:5]}, infrared-only={only_ir[:5]}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    names = sorted(rgb_names)
    worker_count = resolve_workers(workers, len(names))
    started = time.perf_counter()
    print(
        f"Processing {len(names)} pairs with {worker_count} worker(s)",
        flush=True,
    )

    results: dict[str, tuple[float, float]] = {}
    if worker_count == 1:
        for index, name in enumerate(names, 1):
            image_name, rgb_weight, ir_weight = process_one(
                name, visible_dir, infrared_dir, destination
            )
            results[image_name] = (rgb_weight, ir_weight)
            print(
                f"[{index}/{len(names)}] {image_name} "
                f"RGB={rgb_weight:.6f} IR={ir_weight:.6f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_one, name, visible_dir, infrared_dir, destination): name
                for name in names
            }
            for index, future in enumerate(as_completed(futures), 1):
                image_name, rgb_weight, ir_weight = future.result()
                results[image_name] = (rgb_weight, ir_weight)
                print(
                    f"[{index}/{len(names)}] {image_name} "
                    f"RGB={rgb_weight:.6f} IR={ir_weight:.6f}",
                    flush=True,
                )

    with (destination / "weights.txt").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["image", "rgb_weight", "ir_weight"])
        for name in names:
            image_name = name.as_posix()
            rgb_weight, ir_weight = results[image_name]
            writer.writerow([image_name, f"{rgb_weight:.8f}", f"{ir_weight:.8f}"])
    elapsed = time.perf_counter() - started
    rate = len(names) / elapsed if elapsed > 0.0 else 0.0
    print(
        f"Saved {len(names)} pairs and weights.txt to {destination} "
        f"in {elapsed:.2f}s ({rate:.2f} pairs/s)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=default_input_dir())
    parser.add_argument(
        "--output-dir", type=Path,
        help="Defaults to an <input>_enhanced directory next to the input directory",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes. Use 0 for all CPU cores.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output_dir or args.input_dir.with_name(args.input_dir.name + "_enhanced")
    process_dataset(args.input_dir, output, workers=args.workers)
