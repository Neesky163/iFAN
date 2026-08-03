# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from itertools import product
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from inference.lightning_module import LightningModule


class MaskClassificationPanoptic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        quality_score_enabled: bool = True,
        utility_score_temperature: float = 1.0,
        utility_score_bias: float = -1.5,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        postprocess_sweep_biases: Optional[list[float]] = None,
        postprocess_sweep_mask_thresholds: Optional[list[float]] = None,
        postprocess_sweep_overlap_thresholds: Optional[list[float]] = None,
        ckpt_path: Optional[str] = None,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            ckpt_path=ckpt_path,
            load_ckpt_class_head=load_ckpt_class_head,
        )
        self.save_hyperparameters(ignore=["network", "_class_path"])
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        sweep_args = (
            postprocess_sweep_biases,
            postprocess_sweep_mask_thresholds,
            postprocess_sweep_overlap_thresholds,
        )
        if any(value is not None for value in sweep_args):
            if not all(value for value in sweep_args):
                raise ValueError("All three postprocess sweep lists must be non-empty.")
            self.postprocess_sweep = list(product(*sweep_args))
        else:
            self.postprocess_sweep = []
        self.stuff_classes = stuff_classes
        self._configure_quality_inference(
            quality_score_enabled,
            utility_score_temperature,
            utility_score_bias,
        )
        thing_classes = [i for i in range(num_classes) if i not in stuff_classes]
        num_metrics = (
            len(self.postprocess_sweep)
            if self.postprocess_sweep
            else self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        )
        self.init_metrics_panoptic(thing_classes, stuff_classes, num_metrics)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer, quality_logits_per_layer = (
            self._unpack_outputs(self(transformed_imgs))
        )

        raw_targets = targets
        is_crowds = [target["is_crowd"] for target in raw_targets]
        targets = self.to_per_pixel_targets_panoptic(raw_targets)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            if self.postprocess_sweep and i != len(mask_logits_per_layer) - 1:
                continue
            quality_logits = (
                quality_logits_per_layer[i]
                if quality_logits_per_layer is not None and self.quality_score_enabled
                else None
            )
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )
            if i == len(mask_logits_per_layer) - 1:
                self._update_reliability_calibration(
                    mask_logits, class_logits, quality_logits, raw_targets
                )
            if self.postprocess_sweep:
                for metric_idx, (bias, mask_thresh, overlap_thresh) in enumerate(
                    self.postprocess_sweep
                ):
                    preds = self.to_per_pixel_preds_panoptic(
                        mask_logits,
                        class_logits,
                        self.stuff_classes,
                        mask_thresh,
                        overlap_thresh,
                        quality_logits=quality_logits,
                        utility_score_temperature=self.utility_score_temperature,
                        utility_score_bias=bias,
                    )
                    self.update_metrics_panoptic(
                        preds, targets, is_crowds, metric_idx
                    )
                continue
            preds = self.to_per_pixel_preds_panoptic(
                mask_logits,
                class_logits,
                self.stuff_classes,
                self.mask_thresh,
                self.overlap_thresh,
                quality_logits=quality_logits,
                utility_score_temperature=self.utility_score_temperature,
                utility_score_bias=self.utility_score_bias,
            )
            self.update_metrics_panoptic(preds, targets, is_crowds, i)

    def on_validation_epoch_end(self):
        if self.postprocess_sweep:
            for metric, (bias, mask_thresh, overlap_thresh) in zip(
                self.metrics, self.postprocess_sweep
            ):
                result = metric.compute()[:-1]
                metric.reset()
                pq = result[:, 0]
                num_things = len(metric.things)
                if self.trainer.is_global_zero:
                    print(
                        "SWEEP_RESULT "
                        f"bias={bias:g} mask_thresh={mask_thresh:g} "
                        f"overlap_thresh={overlap_thresh:g} "
                        f"pq_all={float(pq.mean()) * 100:.6f} "
                        f"pq_things={float(pq[:num_things].mean()) * 100:.6f} "
                        f"pq_stuff={float(pq[num_things:].mean()) * 100:.6f}",
                        flush=True,
                    )
            return
        self._on_eval_epoch_end_panoptic("val")
        self._log_reliability_calibration("val")

    def on_validation_end(self):
        if self.postprocess_sweep:
            return
        self._on_eval_end_panoptic("val")
