# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.lightning_module import LightningModule


class MaskClassificationInstance(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        quality_score_enabled: bool = True,
        utility_score_temperature: float = 1.0,
        utility_score_bias: float = -1.5,
        eval_top_k_instances: int = 100,
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
        self.stuff_classes: List[int] = []
        self.eval_top_k_instances = eval_top_k_instances
        self._configure_quality_inference(
            quality_score_enabled,
            utility_score_temperature,
            utility_score_bias,
        )
        self.init_metrics_instance(
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        )

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

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
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
                    mask_logits, class_logits, quality_logits, targets
                )

            preds, targets_ = [], []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                if quality_logits is not None:
                    quality_scores = quality_logits[j].sigmoid().squeeze(-1)[:, None]
                    scores = self.calibrated_utility_scores(
                        scores,
                        quality_scores.to(scores.dtype),
                        self.utility_score_temperature,
                        self.utility_score_bias,
                    )
                labels = (
                    torch.arange(scores.shape[-1], device=self.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    self.eval_top_k_instances, sorted=False
                )
                labels = labels[topk_indices]

                topk_indices = topk_indices // scores.shape[-1]
                mask_logits[j] = mask_logits[j][topk_indices]

                masks = mask_logits[j] > 0
                mask_scores = (
                    mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores

                preds.append(
                    dict(
                        masks=masks,
                        labels=labels,
                        scores=scores,
                    )
                )
                targets_.append(
                    dict(
                        masks=targets[j]["masks"],
                        labels=targets[j]["labels"],
                        iscrowd=targets[j]["is_crowd"],
                    )
                )

            self.update_metrics_instance(preds, targets_, i)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_instance("val")
        self._log_reliability_calibration("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")
