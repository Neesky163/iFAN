# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from inference.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        ignore_idx: int = 255,
        quality_score_enabled: bool = True,
        utility_score_temperature: float = 1.0,
        utility_score_bias: float = -1.5,
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
        self.ignore_idx = ignore_idx
        self.stuff_classes = range(num_classes)
        self._configure_quality_inference(
            quality_score_enabled,
            utility_score_temperature,
            utility_score_bias,
        )
        self.init_metrics_semantic(
            ignore_idx,
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1,
        )

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer, quality_logits_per_layer = (
            self._unpack_outputs(self(crops))
        )

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            quality_logits = (
                quality_logits_per_layer[i]
                if quality_logits_per_layer is not None and self.quality_score_enabled
                else None
            )
            if quality_logits is not None and i == len(mask_logits_per_layer) - 1:
                self.log(
                    f"metrics/{log_prefix}_quality_mean",
                    quality_logits.sigmoid().mean(),
                    sync_dist=True,
                )
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(
                mask_logits,
                class_logits,
                quality_logits=quality_logits,
                utility_score_temperature=self.utility_score_temperature,
                utility_score_bias=self.utility_score_bias,
            )
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
