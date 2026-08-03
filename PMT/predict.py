#!/usr/bin/env python3
"""Run PMT-iFAN semantic, instance, or panoptic prediction on images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = PROJECT_ROOT / "image"
sys.path.insert(0, str(IMAGE_ROOT))

from inference.mask_classification_instance import MaskClassificationInstance
from inference.mask_classification_panoptic import MaskClassificationPanoptic
from inference.mask_classification_semantic import MaskClassificationSemantic
from models.pmt import PMT
from models.vit import ViT


TASK_CLASSES = {
    "semantic": MaskClassificationSemantic,
    "instance": MaskClassificationInstance,
    "panoptic": MaskClassificationPanoptic,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict segmentation for one image or a directory.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Defaults to checkpoints/<config-stem>.pth.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--backbone",
        help="Hugging Face model ID or local backbone directory; defaults to config value.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.30,
        help="Minimum final score saved for instance predictions.",
    )
    return parser.parse_args()


def resolve_config(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    candidate = PROJECT_ROOT / "configs" / path
    if candidate.is_file():
        return candidate.resolve()
    raise SystemExit(f"Config not found: {path}")


def resolve_checkpoint(config: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        result = checkpoint.expanduser().resolve()
    else:
        result = PROJECT_ROOT / "checkpoints" / config.name.replace(".yaml", ".pth")
    if not result.is_file():
        raise SystemExit(f"Checkpoint not found: {result}")
    return result


def task_from_config(config: dict) -> str:
    class_path = config["model"]["class_path"].lower()
    for task in TASK_CLASSES:
        if task in class_path:
            return task
    raise ValueError(f"Unsupported model class: {class_path}")


def build_model(config: dict, checkpoint: Path, backbone: str | None):
    data_args = config["data"]["init_args"]
    model_args = dict(config["model"]["init_args"])
    network_cfg = model_args.pop("network")
    network_args = dict(network_cfg["init_args"])
    encoder_cfg = network_args.pop("encoder")
    encoder_args = dict(encoder_cfg["init_args"])

    img_size = tuple(data_args["img_size"])
    num_classes = int(data_args["num_classes"])
    encoder_args["img_size"] = img_size
    if backbone:
        encoder_args["backbone_name"] = backbone
    encoder = ViT(**encoder_args)
    network_args["encoder"] = encoder
    network_args["num_classes"] = num_classes
    network = PMT(**network_args)

    task = task_from_config(config)
    model_args.update(
        network=network,
        img_size=img_size,
        num_classes=num_classes,
        ckpt_path=str(checkpoint),
    )
    if task == "panoptic":
        model_args["stuff_classes"] = list(data_args["stuff_classes"])
    model = TASK_CLASSES[task](**model_args)
    return task, model


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return device


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            return images
    raise SystemExit(f"No input images found: {path}")


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def color_for_id(value: int) -> np.ndarray:
    return np.array(
        [(37 * value + 23) % 256, (17 * value + 97) % 256, (29 * value + 53) % 256],
        dtype=np.uint8,
    )


def save_overlay(image: Image.Image, colors: np.ndarray, path: Path):
    base = image.convert("RGB")
    colored = Image.fromarray(colors)
    Image.blend(base, colored, 0.45).save(path)


@torch.inference_mode()
def predict_semantic(model, tensor, image, output_stem: Path):
    sizes = [tensor.shape[-2:]]
    crops, origins = model.window_imgs_semantic([tensor])
    masks, classes, quality = model._unpack_outputs(model(crops))
    mask_logits = F.interpolate(masks[-1], model.img_size, mode="bilinear")
    quality_logits = quality[-1] if quality is not None and model.quality_score_enabled else None
    crop_logits = model.to_per_pixel_logits_semantic(
        mask_logits,
        classes[-1],
        quality_logits=quality_logits,
        utility_score_temperature=model.utility_score_temperature,
        utility_score_bias=model.utility_score_bias,
    )
    logits = model.revert_window_logits_semantic(crop_logits, origins, sizes)[0]
    labels = logits.argmax(0).cpu().numpy().astype(np.uint16)
    np.save(output_stem.with_suffix(".labels.npy"), labels)
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        colors[labels == label] = color_for_id(int(label))
    save_overlay(image, colors, output_stem.with_suffix(".visualization.png"))
    output_stem.with_suffix(".json").write_text(
        json.dumps({"task": "semantic", "classes": [int(v) for v in np.unique(labels)]}, indent=2),
        encoding="utf-8",
    )


@torch.inference_mode()
def predict_instance(model, tensor, image, output_stem: Path, score_threshold: float):
    size = tensor.shape[-2:]
    transformed = model.resize_and_pad_imgs_instance_panoptic([tensor])
    masks, classes, quality = model._unpack_outputs(model(transformed))
    mask_logits = F.interpolate(masks[-1], model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, [size])[0]
    class_scores = classes[-1][0].softmax(dim=-1)[:, :-1]
    if quality is not None and model.quality_score_enabled:
        quality_scores = quality[-1][0].sigmoid().squeeze(-1)[:, None]
        class_scores = model.calibrated_utility_scores(
            class_scores,
            quality_scores.to(class_scores.dtype),
            model.utility_score_temperature,
            model.utility_score_bias,
        )
    labels = torch.arange(class_scores.shape[-1], device=tensor.device)[None].expand_as(class_scores).flatten()
    count = min(model.eval_top_k_instances, class_scores.numel())
    top_scores, indices = class_scores.flatten().topk(count, sorted=True)
    labels = labels[indices]
    query_indices = indices // class_scores.shape[-1]
    selected_logits = mask_logits[query_indices]
    selected_masks = selected_logits > 0
    mask_scores = (
        selected_logits.sigmoid().flatten(1) * selected_masks.flatten(1)
    ).sum(1) / (selected_masks.flatten(1).sum(1) + 1e-6)
    scores = top_scores * mask_scores
    keep = scores >= score_threshold
    masks_np = selected_masks[keep].cpu().numpy().astype(np.uint8)
    labels_np = labels[keep].cpu().numpy().astype(np.int32)
    scores_np = scores[keep].cpu().numpy().astype(np.float32)
    np.savez_compressed(
        output_stem.with_suffix(".instances.npz"),
        masks=masks_np,
        labels=labels_np,
        scores=scores_np,
    )
    colors = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    for index, mask in enumerate(masks_np):
        color = color_for_id(index + 1)
        colors[mask.astype(bool)] = (
            0.45 * colors[mask.astype(bool)] + 0.55 * color
        ).astype(np.uint8)
    Image.fromarray(colors).save(output_stem.with_suffix(".visualization.png"))
    records = [
        {"instance_id": i, "class_id": int(label), "score": float(score)}
        for i, (label, score) in enumerate(zip(labels_np, scores_np))
    ]
    output_stem.with_suffix(".json").write_text(
        json.dumps({"task": "instance", "instances": records}, indent=2), encoding="utf-8"
    )


@torch.inference_mode()
def predict_panoptic(model, tensor, image, output_stem: Path):
    size = tensor.shape[-2:]
    transformed = model.resize_and_pad_imgs_instance_panoptic([tensor])
    masks, classes, quality = model._unpack_outputs(model(transformed))
    mask_logits = F.interpolate(masks[-1], model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, [size])
    quality_logits = quality[-1] if quality is not None and model.quality_score_enabled else None
    prediction = model.to_per_pixel_preds_panoptic(
        mask_logits,
        classes[-1],
        model.stuff_classes,
        model.mask_thresh,
        model.overlap_thresh,
        quality_logits=quality_logits,
        utility_score_temperature=model.utility_score_temperature,
        utility_score_bias=model.utility_score_bias,
    )[0].cpu().numpy()
    class_ids = prediction[..., 0].astype(np.int32)
    segment_ids = prediction[..., 1].astype(np.int32)
    np.savez_compressed(
        output_stem.with_suffix(".panoptic.npz"),
        class_ids=class_ids,
        segment_ids=segment_ids,
    )
    colors = np.zeros((*segment_ids.shape, 3), dtype=np.uint8)
    segments = []
    for segment_id in np.unique(segment_ids):
        if segment_id < 0:
            continue
        mask = segment_ids == segment_id
        class_id = int(class_ids[mask][0])
        colors[mask] = color_for_id(int(segment_id) + 1)
        segments.append(
            {"segment_id": int(segment_id), "class_id": class_id, "area": int(mask.sum())}
        )
    save_overlay(image, colors, output_stem.with_suffix(".visualization.png"))
    output_stem.with_suffix(".json").write_text(
        json.dumps({"task": "panoptic", "segments": segments}, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    config_path = resolve_config(args.config)
    checkpoint_path = resolve_checkpoint(config_path, args.checkpoint)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = choose_device(args.device)
    task, model = build_model(config, checkpoint_path, args.backbone)
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    images = find_images(args.input)
    print(f"Task: {task}; device: {device}; images: {len(images)}")
    for input_path in images:
        image = Image.open(input_path).convert("RGB")
        tensor = image_tensor(image, device)
        output_stem = args.output / input_path.stem
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad()
        with context:
            if task == "semantic":
                predict_semantic(model, tensor, image, output_stem)
            elif task == "instance":
                predict_instance(model, tensor, image, output_stem, args.score_threshold)
            else:
                predict_panoptic(model, tensor, image, output_stem)
        print(f"Predicted: {input_path} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
