"""Quality calibration utilities used by inference and validation."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


class QualityInferenceMixin:
    def _configure_quality_inference(
        self,
        quality_score_enabled: bool,
        utility_score_temperature: float,
        utility_score_bias: float,
    ):
        self.quality_score_enabled = quality_score_enabled
        self.utility_score_temperature = utility_score_temperature
        self.utility_score_bias = utility_score_bias
        self.reliability_calibration_bins = 10
        self.reliability_calibration_iou_size = 128
        for name in ("count", "pred_sum", "iou_sum", "tp_sum"):
            self.register_buffer(
                f"reliability_bin_{name}",
                torch.zeros(self.reliability_calibration_bins),
                persistent=False,
            )

    @staticmethod
    def _unpack_outputs(outputs):
        if len(outputs) == 3:
            return outputs
        masks, classes = outputs
        return masks, classes, None

    @staticmethod
    def calibrated_utility_scores(class_scores, quality_scores, temperature, bias):
        eps = torch.finfo(class_scores.dtype).eps
        utility = (
            class_scores.clamp(eps, 1.0 - eps)
            * quality_scores.clamp(eps, 1.0 - eps)
        ).clamp(eps, 1.0 - eps)
        utility_logit = utility.log() - (1.0 - utility).log()
        return torch.sigmoid(
            (utility_logit - float(bias)) / max(float(temperature), 1e-6)
        )

    def _query_reliability(self, class_logits, quality_logits):
        class_scores = class_logits.softmax(dim=-1)[..., :-1].max(dim=-1).values
        quality_scores = (
            quality_logits.sigmoid().squeeze(-1)
            if quality_logits is not None
            else torch.ones_like(class_scores)
        )
        return self.calibrated_utility_scores(
            class_scores,
            quality_scores.to(class_scores.dtype),
            getattr(self, "utility_score_temperature", 1.0),
            getattr(self, "utility_score_bias", 0.0),
        ).detach()

    @torch.no_grad()
    def _update_reliability_calibration(
        self, mask_logits, class_logits, quality_logits, targets
    ):
        bins = self.reliability_calibration_bins
        reliability = self._query_reliability(class_logits, quality_logits)
        classes = class_logits.softmax(dim=-1).max(dim=-1).indices
        no_object = class_logits.shape[-1] - 1
        for image_idx, image_logits in enumerate(mask_logits):
            candidate_idx = classes[image_idx].ne(no_object).nonzero(as_tuple=False).flatten()
            if candidate_idx.numel() == 0:
                continue
            pred_scores = reliability[image_idx, candidate_idx].clamp(0, 1)
            target = targets[image_idx]
            valid = ~target.get(
                "is_crowd",
                torch.zeros(
                    len(target["labels"]), dtype=torch.bool, device=image_logits.device
                ),
            ).to(image_logits.device).bool()
            target_labels = target["labels"].to(image_logits.device)[valid]
            target_masks = target["masks"].to(
                image_logits.device, image_logits.dtype
            )[valid]
            if target_masks.numel() == 0:
                matched_iou = torch.zeros_like(pred_scores)
            else:
                pred_masks = image_logits[candidate_idx].sigmoid()
                max_size = self.reliability_calibration_iou_size
                if max(pred_masks.shape[-2:]) > max_size:
                    scale = max_size / max(pred_masks.shape[-2:])
                    size = tuple(max(1, round(v * scale)) for v in pred_masks.shape[-2:])
                    pred_masks = F.interpolate(
                        pred_masks[:, None], size, mode="bilinear"
                    ).squeeze(1)
                    target_masks = F.interpolate(
                        target_masks[:, None], size, mode="nearest"
                    ).squeeze(1)
                pred_flat = pred_masks.flatten(1)
                target_flat = target_masks.flatten(1)
                intersection = pred_flat @ target_flat.T
                union = (
                    pred_flat.sum(1)[:, None]
                    + target_flat.sum(1)[None]
                    - intersection
                )
                ious = intersection / union.clamp_min(1e-6)
                same_class = classes[image_idx, candidate_idx, None].eq(
                    target_labels[None]
                )
                matched_iou = ious.masked_fill(~same_class, 0).max(dim=1).values
            tp = matched_iou.gt(0.5).to(pred_scores.dtype)
            bin_idx = (pred_scores * bins).long().clamp(max=bins - 1)
            for idx in bin_idx.unique():
                selected = bin_idx.eq(idx)
                self.reliability_bin_count[idx] += selected.sum()
                self.reliability_bin_pred_sum[idx] += pred_scores[selected].sum()
                self.reliability_bin_iou_sum[idx] += matched_iou[selected].sum()
                self.reliability_bin_tp_sum[idx] += tp[selected].sum()

    def _log_reliability_calibration(self, prefix):
        tensors = [
            self.reliability_bin_count.clone(),
            self.reliability_bin_pred_sum.clone(),
            self.reliability_bin_iou_sum.clone(),
            self.reliability_bin_tp_sum.clone(),
        ]
        if dist.is_available() and dist.is_initialized():
            for tensor in tensors:
                dist.all_reduce(tensor)
        count, pred_sum, iou_sum, tp_sum = tensors
        denom = count.clamp_min(1)
        pred, iou, tp = pred_sum / denom, iou_sum / denom, tp_sum / denom
        total = count.sum()
        self.log(
            f"metrics/{prefix}_reliability_ece_iou",
            ((pred - iou).abs() * count).sum() / total.clamp_min(1),
        )
        self.log(
            f"metrics/{prefix}_reliability_ece_tp",
            ((pred - tp).abs() * count).sum() / total.clamp_min(1),
        )
        for tensor in (
            self.reliability_bin_count,
            self.reliability_bin_pred_sum,
            self.reliability_bin_iou_sum,
            self.reliability_bin_tp_sum,
        ):
            tensor.zero_()

