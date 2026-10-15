"""Show image-level adaptive RGB/IR weights on selected M3FD image pairs."""

import argparse
from pathlib import Path

import cv2

from get_fused_image import process_pair


def read_color(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return image


def annotate_image(image, modality: str, weight: float):
    result = image.copy()
    height, width = result.shape[:2]
    label = f"{modality} weight: {weight:.4f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.9, width / 1100)
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    pad = 14
    box_left = width - text_width - 2 * pad
    box_bottom = text_height + baseline + 2 * pad
    cv2.rectangle(result, (box_left, 0), (width - 1, box_bottom),
                  (0, 0, 0), -1)
    cv2.putText(result, label, (box_left + pad, text_height + pad),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return result


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        rgb_path = args.dataset / "visible" / f"{name}.png"
        ir_path = args.dataset / "infrared" / f"{name}.png"
        rgb = read_color(rgb_path)
        ir = read_color(ir_path)
        _, _, weights = process_pair(rgb_path, ir_path)
        for modality, image, weight in (
            ("rgb", rgb, weights.rgb), ("ir", ir, weights.ir)
        ):
            output_path = args.output_dir / f"{name}_{modality}_weight.png"
            if not cv2.imwrite(
                str(output_path), annotate_image(image, modality.upper(), weight)
            ):
                raise OSError(f"Cannot save image: {output_path}")
        print(f"{name}: RGB={weights.rgb:.6f}, IR={weights.ir:.6f}, "
              f"sum={weights.rgb + weights.ir:.6f} -> {args.output_dir}",
              flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=["00000", "00445", "00934"])
    parser.add_argument("--dataset", type=Path,
                        default=Path(r"D:\data\M3FD\cocodataset\images"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "adaptive_weight_figures")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
