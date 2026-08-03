from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from inference import EoMTPredictor
from inference.predictor import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description="Inference-only EoMT runner")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Must be a smaller inference/*.ckpt file")
    parser.add_argument("--input", type=Path, help="An image or directory of images")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/config without allocating the model")
    return parser.parse_args()


def colors(labels: np.ndarray) -> np.ndarray:
    ids = labels.astype(np.uint32)
    return np.stack(((ids * 37 + 17) % 255, (ids * 67 + 29) % 255, (ids * 97 + 43) % 255), axis=-1).astype(np.uint8)


def save_result(path: Path, image: Image.Image, result: dict[str, np.ndarray], output: Path):
    stem = path.stem
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{stem}.npz", **result)
    if "labels" in result and result["labels"].ndim == 2:
        labels = result["labels"]
    elif "semantic" in result:
        labels = result["semantic"]
    else:
        labels = np.zeros((image.height, image.width), dtype=np.int64)
        order = np.argsort(result["scores"])
        for index in order:
            labels[result["masks"][index]] = int(result["labels"][index]) + 1
    overlay = colors(labels)
    void = labels < 0
    overlay[void] = 0
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = (0.45 * rgb + 0.55 * overlay).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(output / f"{stem}_overlay.png")
    summary = {key: list(value.shape) for key, value in result.items()}
    (output / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["_config_path"] = str(args.config.resolve())
    data_class = config.get("data", {}).get("class_path")
    if data_class not in DATASETS:
        raise SystemExit(f"Unsupported data.class_path: {data_class!r}")
    if args.checkpoint.parent.name != "inference":
        raise SystemExit("Refusing full training checkpoint; select a checkpoint under inference/")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.config.stem != args.checkpoint.stem:
        raise SystemExit(
            "Config/checkpoint mismatch: filenames must have the same stem "
            f"({args.config.stem!r} != {args.checkpoint.stem!r})"
        )
    task = DATASETS[data_class][0]
    if args.dry_run:
        print(json.dumps({"status": "ok", "task": task, "config": str(args.config.resolve()), "checkpoint": str(args.checkpoint.resolve())}, indent=2))
        return
    if args.input is None:
        raise SystemExit("--input is required unless --dry-run is used")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = [args.input] if args.input.is_file() else sorted(path for path in args.input.iterdir() if path.suffix.lower() in extensions)
    if not paths:
        raise SystemExit(f"No supported images found at {args.input}")
    predictor = EoMTPredictor(config, args.checkpoint, args.device)
    for path in paths:
        image = Image.open(path).convert("RGB")
        save_result(path, image, predictor(image), args.output)
        print(f"saved {path.name} -> {args.output}")


if __name__ == "__main__":
    main()
