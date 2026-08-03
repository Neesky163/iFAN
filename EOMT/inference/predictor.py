from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.eomt import EoMT
from models.vit import ViT


DATASETS = {
    "datasets.coco_panoptic.COCOPanoptic": ("panoptic", 133, (640, 640)),
    "datasets.ade20k_panoptic.ADE20KPanoptic": ("panoptic", 150, (640, 640)),
    "datasets.coco_instance.COCOInstance": ("instance", 80, (640, 640)),
    "datasets.ade20k_semantic.ADE20KSemantic": ("semantic", 150, (512, 512)),
    "datasets.cityscapes_semantic.CityscapesSemantic": ("semantic", 19, (1024, 1024)),
}


def _tuple2(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected a two-element image size, got {value!r}")
    return int(value[0]), int(value[1])


class InferenceModel(nn.Module):
    """EoMT network plus inference preprocessing and task postprocessing."""

    def __init__(
        self,
        network: EoMT,
        task: str,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        mask_thresh: float,
        overlap_thresh: float,
        utility_score_temperature: float,
        utility_score_bias: float,
        top_k_instances: int = 100,
    ) -> None:
        super().__init__()
        self.network = network
        self.task = task
        self.img_size = img_size
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.utility_score_temperature = utility_score_temperature
        self.utility_score_bias = utility_score_bias
        self.top_k_instances = top_k_instances

    def forward(self, images: torch.Tensor):
        return self.network(images / 255.0)

    @staticmethod
    def _final_outputs(outputs):
        if len(outputs) == 3:
            masks, classes, qualities = outputs
            return masks[-1], classes[-1], qualities[-1]
        masks, classes = outputs
        return masks[-1], classes[-1], None

    @staticmethod
    def _pil_resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        pil = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
        pil = pil.resize((size[1], size[0]), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).to(image.device)

    def _scale_semantic(self, size: tuple[int, int]) -> tuple[int, int]:
        factor = max(self.img_size[0] / size[0], self.img_size[1] / size[1])
        return round(size[0] * factor), round(size[1] * factor)

    def _semantic_crops(self, image: torch.Tensor):
        original_size = tuple(image.shape[-2:])
        resized = self._pil_resize(image, self._scale_semantic(original_size))
        crop_size = min(self.img_size)
        num_crops = math.ceil(max(resized.shape[-2:]) / crop_size)
        overlap = num_crops * crop_size - max(resized.shape[-2:])
        overlap_per_crop = overlap / (num_crops - 1) if overlap > 0 else 0
        crops, origins = [], []
        for index in range(num_crops):
            start = int(index * (crop_size - overlap_per_crop))
            end = start + crop_size
            if resized.shape[-2] > resized.shape[-1]:
                crops.append(resized[:, start:end, :])
            else:
                crops.append(resized[:, :, start:end])
            origins.append((start, end))
        return torch.stack(crops), origins, original_size, tuple(resized.shape[-2:])

    @staticmethod
    def _calibrated_scores(class_scores, quality_scores, temperature, bias):
        eps = torch.finfo(class_scores.dtype).eps
        utility = (class_scores.clamp(eps, 1 - eps) * quality_scores.clamp(eps, 1 - eps)).clamp(eps, 1 - eps)
        logit = utility.log() - (1 - utility).log()
        return torch.sigmoid((logit - float(bias)) / max(float(temperature), 1e-6))

    def infer_semantic(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        crops, origins, original_size, resized_size = self._semantic_crops(image)
        masks, classes, qualities = self._final_outputs(self(crops))
        masks = F.interpolate(masks, self.img_size, mode="bilinear", align_corners=False)
        class_scores = classes.softmax(dim=-1)[..., :-1]
        if qualities is not None:
            class_scores = self._calibrated_scores(
                class_scores,
                qualities.sigmoid().squeeze(-1)[..., None].to(class_scores.dtype),
                self.utility_score_temperature,
                self.utility_score_bias,
            )
        crop_logits = torch.einsum("bqhw,bqc->bchw", masks.sigmoid(), class_scores)
        sums = crop_logits.new_zeros((self.num_classes, *resized_size))
        counts = crop_logits.new_zeros((self.num_classes, *resized_size))
        vertical = resized_size[0] > resized_size[1]
        for index, (start, end) in enumerate(origins):
            if vertical:
                sums[:, start:end, :] += crop_logits[index]
                counts[:, start:end, :] += 1
            else:
                sums[:, :, start:end] += crop_logits[index]
                counts[:, :, start:end] += 1
        logits = F.interpolate(
            (sums / counts.clamp_min(1))[None], original_size,
            mode="bilinear", align_corners=False,
        )[0]
        return {"labels": logits.argmax(0)}

    def _resize_and_pad(self, image: torch.Tensor):
        original_size = tuple(image.shape[-2:])
        factor = min(self.img_size[0] / original_size[0], self.img_size[1] / original_size[1])
        scaled_size = round(original_size[0] * factor), round(original_size[1] * factor)
        resized = self._pil_resize(image, scaled_size)
        padded = F.pad(resized, (0, self.img_size[1] - scaled_size[1], 0, self.img_size[0] - scaled_size[0]))
        return padded[None], original_size, scaled_size

    def _restore_masks(self, masks, original_size, scaled_size):
        masks = F.interpolate(masks, self.img_size, mode="bilinear", align_corners=False)
        masks = masks[0, :, : scaled_size[0], : scaled_size[1]]
        return F.interpolate(masks[None], original_size, mode="bilinear", align_corners=False)[0]

    def infer_instance(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        transformed, original_size, scaled_size = self._resize_and_pad(image)
        masks, classes, qualities = self._final_outputs(self(transformed))
        masks = self._restore_masks(masks, original_size, scaled_size)
        scores = classes[0].softmax(dim=-1)[:, :-1]
        if qualities is not None:
            scores = self._calibrated_scores(
                scores, qualities[0].sigmoid().squeeze(-1)[:, None].to(scores.dtype),
                self.utility_score_temperature, self.utility_score_bias,
            )
        labels = torch.arange(scores.shape[-1], device=scores.device)[None].expand(scores.shape[0], -1).flatten()
        k = min(self.top_k_instances, scores.numel())
        top_scores, indices = scores.flatten().topk(k, sorted=True)
        labels = labels[indices]
        selected = masks[indices // scores.shape[-1]]
        binary = selected > 0
        mask_scores = (selected.sigmoid().flatten(1) * binary.flatten(1)).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
        return {"masks": binary, "labels": labels, "scores": top_scores * mask_scores}

    def infer_panoptic(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        transformed, original_size, scaled_size = self._resize_and_pad(image)
        masks, classes_logits, qualities = self._final_outputs(self(transformed))
        masks = self._restore_masks(masks, original_size, scaled_size).sigmoid()
        class_scores, classes = classes_logits[0].softmax(dim=-1).max(-1)
        quality_scores = qualities[0].sigmoid().squeeze(-1).to(class_scores.dtype) if qualities is not None else torch.ones_like(class_scores)
        scores = self._calibrated_scores(class_scores, quality_scores, self.utility_score_temperature, self.utility_score_bias)
        keep = classes.ne(classes_logits.shape[-1] - 1) & scores.gt(self.mask_thresh)
        semantic = torch.full(original_size, self.num_classes, dtype=torch.long, device=image.device)
        instances = torch.full(original_size, -1, dtype=torch.long, device=image.device)
        if not keep.any():
            return {"semantic": semantic, "instance": instances}
        kept_masks, kept_scores, kept_classes = masks[keep], scores[keep], classes[keep]
        winners = (kept_scores[:, None, None] * kept_masks).argmax(0)
        stuff_ids: dict[int, int] = {}
        segment_id = 0
        for index, class_id in enumerate(kept_classes.tolist()):
            original_mask = kept_masks[index] >= 0.5
            assigned_mask = winners == index
            final_mask = original_mask & assigned_mask
            original_area, assigned_area, final_area = original_mask.sum().item(), assigned_mask.sum().item(), final_mask.sum().item()
            if not original_area or not assigned_area or not final_area or assigned_area / original_area < self.overlap_thresh:
                continue
            if class_id in self.stuff_classes and class_id in stuff_ids:
                sid = stuff_ids[class_id]
            else:
                sid = segment_id
                if class_id in self.stuff_classes:
                    stuff_ids[class_id] = sid
                segment_id += 1
            semantic[final_mask] = class_id
            instances[final_mask] = sid
        return {"semantic": semantic, "instance": instances}

    def infer(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.task == "semantic":
            return self.infer_semantic(image)
        if self.task == "instance":
            return self.infer_instance(image)
        return self.infer_panoptic(image)


class EoMTPredictor:
    """Build an inference-only EoMT model from a selected YAML and checkpoint."""

    def __init__(self, config: dict[str, Any], checkpoint: str | Path, device: str) -> None:
        data_path = config.get("data", {}).get("class_path")
        if data_path not in DATASETS:
            raise ValueError(f"Unsupported dataset class in config: {data_path!r}")
        task, default_classes, default_size = DATASETS[data_path]
        data_args = config.get("data", {}).get("init_args", {})
        model_args = config.get("model", {}).get("init_args", {})
        network_args = model_args.get("network", {}).get("init_args", {})
        encoder_args = network_args.get("encoder", {}).get("init_args", {})
        img_size = _tuple2(data_args.get("img_size", default_size))
        num_classes = int(data_args.get("num_classes", default_classes))
        checkpoint = Path(checkpoint).resolve()
        if checkpoint.parent.name != "inference":
            raise ValueError("Only the smaller inference checkpoint under an inference/ directory is accepted")
        config_path = config.get("_config_path")
        if config_path is not None and checkpoint.stem != Path(config_path).stem:
            raise ValueError("Config and checkpoint filenames must have the same stem")
        encoder = ViT(img_size=img_size, ckpt_path=str(checkpoint), **encoder_args)
        network = EoMT(
            encoder=encoder,
            num_classes=num_classes,
            num_q=int(network_args.get("num_q", 100)),
            num_blocks=int(network_args.get("num_blocks", 4)),
            quality_head_enabled=bool(network_args.get("quality_head_enabled", False)),
        )
        self.model = InferenceModel(
            network=network, task=task, img_size=img_size, num_classes=num_classes,
            stuff_classes=list(data_args.get("stuff_classes", [])),
            mask_thresh=float(model_args.get("mask_thresh", 0.8)),
            overlap_thresh=float(model_args.get("overlap_thresh", 0.8)),
            utility_score_temperature=float(model_args.get("utility_score_temperature", 1.0)),
            utility_score_bias=float(model_args.get("utility_score_bias", 0.0)),
            top_k_instances=int(model_args.get("eval_top_k_instances", 100)),
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = payload.get("state_dict", payload)
        state = {
            key.removeprefix("network.").replace("._orig_mod", ""): value
            for key, value in state.items()
            if key.startswith("network.") and key != "network.attn_mask_probs"
        }
        incompatible = self.model.network.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint/model mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
        self.device = torch.device(device)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> dict[str, np.ndarray]:
        tensor = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).to(self.device)
        device_type = self.device.type
        dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type in {"cuda", "cpu"}):
            result = self.model.infer(tensor)
        return {
            key: (value.float() if value.is_floating_point() else value)
            .detach()
            .cpu()
            .numpy()
            for key, value in result.items()
        }
