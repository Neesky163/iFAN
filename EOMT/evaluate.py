from __future__ import annotations

import argparse
import csv
import json
import tarfile
import zipfile
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable, cast

import numpy as np
from PIL import Image
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection import PanopticQuality
from torchmetrics.functional.detection._panoptic_quality_common import (
    _Color,
    _calculate_iou,
    _get_color_areas,
    _prepocess_inputs,
)
import yaml

from inference import EoMTPredictor
from inference.predictor import DATASETS


COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93,
    95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138,
    141, 144, 145, 147, 148, 149, 151, 154, 155, 156, 159, 161, 166,
    168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186, 187, 188,
    189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
)
COCO_CATEGORY_TO_CONTIGUOUS = {
    category_id: index for index, category_id in enumerate(COCO_CATEGORY_IDS)
}

# ADE20K instance annotations use a 100-class thing index in channel 0.
ADE_INSTANCE_TO_SEMANTIC = (
    7, 8, 10, 12, 14, 15, 18, 19, 20, 22, 23, 24, 27, 30, 31, 32,
    33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47, 49, 50, 53, 55,
    56, 57, 58, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76,
    78, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 95, 97, 98,
    102, 103, 104, 107, 108, 110, 111, 112, 115, 116, 118, 119, 120,
    121, 123, 124, 125, 126, 127, 129, 130, 132, 133, 134, 135, 136,
    137, 138, 139, 142, 143, 144, 146, 147, 148, 149,
)

CITYSCAPES_ID_TO_TRAIN_ID = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone EoMT validation (no training or external source tree)."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="Validation dataset root")
    parser.add_argument("--output", type=Path, default=Path("metric_test_results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path) -> tuple[dict, str, str, int, list[int]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path.resolve())
    data_class = config.get("data", {}).get("class_path")
    if data_class not in DATASETS:
        raise SystemExit(f"Unsupported data.class_path: {data_class!r}")
    task, default_classes, _ = DATASETS[data_class]
    data_args = config.get("data", {}).get("init_args", {})
    num_classes = int(data_args.get("num_classes", default_classes))
    stuff_classes = [int(value) for value in data_args.get("stuff_classes", [])]
    return config, data_class, task, num_classes, stuff_classes


def validate_inputs(args: argparse.Namespace, task: str) -> None:
    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.checkpoint.parent.name != "inference":
        raise SystemExit("Only the smaller checkpoint under inference/ is accepted")
    if args.config.stem != args.checkpoint.stem:
        raise SystemExit(
            "Config/checkpoint mismatch: filenames must have the same stem "
            f"({args.config.stem!r} != {args.checkpoint.stem!r})"
        )
    if not args.data.is_dir():
        raise SystemExit(f"Dataset root not found: {args.data}")
    if task not in {"semantic", "instance", "panoptic"}:
        raise SystemExit(f"Unsupported task: {task}")


def open_json(root: Path, relative_path: str, archives: Iterable[str]) -> dict:
    direct = root / relative_path
    if direct.is_file():
        return json.loads(direct.read_text(encoding="utf-8"))
    for archive_name in archives:
        archive_path = root / archive_name
        if not archive_path.is_file():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            try:
                with archive.open(relative_path) as source:
                    return json.load(source)
            except KeyError:
                continue
    searched = ", ".join([str(direct), *(str(root / name) for name in archives)])
    raise FileNotFoundError(f"Could not find {relative_path}; searched: {searched}")


class ImageSource:
    def __init__(self, root: Path, dataset: str) -> None:
        self.root = root
        self.dataset = dataset
        self.archive: zipfile.ZipFile | None = None
        if dataset == "ade20k":
            self.directory = root / "ADEChallengeData2016/images/validation"
            self.archive_path = root / "ADEChallengeData2016.zip"
            self.prefix = "ADEChallengeData2016/images/validation/"
            self.suffix = ".jpg"
        elif dataset == "cityscapes":
            self.directory = root / "leftImg8bit/val"
            self.archive_path = root / "leftImg8bit_trainvaltest.zip"
            self.prefix = "leftImg8bit/val/"
            self.suffix = ".png"
        else:
            self.directory = root / "val2017"
            self.archive_path = root / "val2017.zip"
            self.prefix = "val2017/"
            self.suffix = ".jpg"

        if self.directory.is_dir():
            self.names = sorted(
                path.relative_to(self.directory).as_posix()
                for path in self.directory.rglob(f"*{self.suffix}")
            )
        elif self.archive_path.is_file():
            self.archive = zipfile.ZipFile(self.archive_path)
            self.names = sorted(
                name.removeprefix(self.prefix)
                for name in self.archive.namelist()
                if name.startswith(self.prefix)
                and name.lower().endswith(self.suffix)
                and not name.endswith("/")
            )
        else:
            raise FileNotFoundError(
                f"Images not found: expected {self.directory} or {self.archive_path}"
            )

    def open(self, filename: str) -> Image.Image:
        if self.archive is None:
            with Image.open(self.directory / filename) as image:
                return image.convert("RGB")
        with self.archive.open(self.prefix + filename) as source:
            with Image.open(source) as image:
                return image.convert("RGB")

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()


class ADETargetSource:
    def __init__(self, root: Path, panoptic: bool) -> None:
        self.semantic = zipfile.ZipFile(root / "ADEChallengeData2016.zip")
        self.instance: zipfile.ZipFile | tarfile.TarFile | None = None
        if panoptic:
            zip_path = root / "annotations_instance.zip"
            tar_path = root / "annotations_instance.tar"
            if zip_path.is_file():
                self.instance = zipfile.ZipFile(zip_path)
            elif tar_path.is_file():
                self.instance = tarfile.open(tar_path)
            else:
                raise FileNotFoundError(
                    f"Expected {zip_path.name} or {tar_path.name} in {root}"
                )

    @staticmethod
    def _image(stream: BinaryIO) -> np.ndarray:
        with Image.open(stream) as image:
            return np.asarray(image).copy()

    def semantic_array(self, stem: str) -> np.ndarray:
        member = f"ADEChallengeData2016/annotations/validation/{stem}.png"
        with self.semantic.open(member) as stream:
            return self._image(stream)

    def target_array(self, filename: str) -> np.ndarray:
        return self.semantic_array(Path(filename).stem).astype(np.int64) - 1

    def instance_array(self, stem: str) -> np.ndarray:
        assert self.instance is not None
        member = f"annotations_instance/validation/{stem}.png"
        if isinstance(self.instance, zipfile.ZipFile):
            with self.instance.open(member) as stream:
                return self._image(stream)
        info = self.instance.getmember(member)
        stream = self.instance.extractfile(info)
        if stream is None:
            raise FileNotFoundError(member)
        with stream:
            return self._image(stream)

    def close(self) -> None:
        self.semantic.close()
        if self.instance is not None:
            self.instance.close()


class CityscapesTargetSource:
    def __init__(self, root: Path) -> None:
        self.directory = root / "gtFine/val"
        self.archive_path = root / "gtFine_trainvaltest.zip"
        self.archive: zipfile.ZipFile | None = None
        if not self.directory.is_dir():
            if not self.archive_path.is_file():
                raise FileNotFoundError(
                    f"Targets not found: expected {self.directory} or {self.archive_path}"
                )
            self.archive = zipfile.ZipFile(self.archive_path)

        self.mapping = np.full(256, 255, dtype=np.int64)
        for label_id, train_id in CITYSCAPES_ID_TO_TRAIN_ID.items():
            self.mapping[label_id] = train_id

    @staticmethod
    def target_name(filename: str) -> str:
        if not filename.endswith("_leftImg8bit.png"):
            raise ValueError(f"Unexpected Cityscapes image name: {filename}")
        return filename.removesuffix("_leftImg8bit.png") + "_gtFine_labelIds.png"

    def target_array(self, filename: str) -> np.ndarray:
        target_name = self.target_name(filename)
        if self.archive is None:
            with Image.open(self.directory / target_name) as image:
                label_ids = np.asarray(image).copy()
        else:
            with self.archive.open(f"gtFine/val/{target_name}") as stream:
                with Image.open(stream) as image:
                    label_ids = np.asarray(image).copy()
        if label_ids.min() < 0 or label_ids.max() >= len(self.mapping):
            raise ValueError(f"Invalid Cityscapes labelIds in {target_name}")
        return self.mapping[label_ids]

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()


class COCOPanopticTargetSource:
    def __init__(self, root: Path) -> None:
        self.stack = ExitStack()
        self.archive: zipfile.ZipFile | None = None
        self.directory: Path | None = None
        for candidate in (
            root / "annotations/panoptic_val2017",
            root / "panoptic_val2017",
        ):
            if candidate.is_dir():
                self.directory = candidate
                return

        for candidate in (
            root / "annotations/panoptic_val2017.zip",
            root / "panoptic_val2017.zip",
        ):
            if candidate.is_file():
                self.archive = self.stack.enter_context(zipfile.ZipFile(candidate))
                return

        outer_path = root / "panoptic_annotations_trainval2017.zip"
        if outer_path.is_file():
            outer = self.stack.enter_context(zipfile.ZipFile(outer_path))
            with outer.open("annotations/panoptic_val2017.zip") as stream:
                nested = BytesIO(stream.read())
            self.archive = self.stack.enter_context(zipfile.ZipFile(nested))
            return
        raise FileNotFoundError("COCO panoptic_val2017 annotations were not found")

    def array(self, filename: str) -> np.ndarray:
        if self.directory is not None:
            with Image.open(self.directory / filename) as image:
                return np.asarray(image).copy()
        assert self.archive is not None
        with self.archive.open(filename) as stream:
            with Image.open(stream) as image:
                return np.asarray(image).copy()

    def close(self) -> None:
        self.stack.close()


def iter_names(names: list[str], limit: int | None) -> list[str]:
    return names if limit is None else names[:limit]


def print_progress(index: int, total: int, filename: str) -> None:
    if index == 1 or index == total or index % 50 == 0:
        print(f"[{index}/{total}] {filename}", flush=True)


def evaluate_semantic(
    predictor: EoMTPredictor,
    source: ImageSource,
    target_source: ADETargetSource | CityscapesTargetSource,
    num_classes: int,
    limit: int | None,
) -> dict[str, float]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    names = iter_names(source.names, limit)
    for index, filename in enumerate(names, 1):
        prediction = predictor(source.open(filename))["labels"].astype(np.int64)
        target = target_source.target_array(filename)
        valid = (target >= 0) & (target < num_classes)
        encoded = target[valid] * num_classes + prediction[valid]
        confusion += np.bincount(encoded, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
        print_progress(index, len(names), filename)
    intersection = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(
        intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0
    )
    return {"miou": float(iou.mean()), "images": len(names)}


def coco_ground_truth(root: Path) -> tuple[COCO, dict]:
    annotation_data = open_json(
        root,
        "annotations/instances_val2017.json",
        ("annotations_trainval2017.zip",),
    )
    coco = COCO()
    coco.dataset = annotation_data
    coco.createIndex()
    return coco, annotation_data


def evaluate_instance(
    predictor: EoMTPredictor,
    source: ImageSource,
    root: Path,
    limit: int | None,
) -> dict[str, float]:
    coco_gt, annotation_data = coco_ground_truth(root)
    image_id_by_name = {image["file_name"]: image["id"] for image in annotation_data["images"]}
    names = [name for name in source.names if name in image_id_by_name]
    names = iter_names(names, limit)
    predictions: list[dict] = []
    for index, filename in enumerate(names, 1):
        result = predictor(source.open(filename))
        masks = np.asfortranarray(result["masks"].astype(np.uint8).transpose(1, 2, 0))
        rles = coco_mask.encode(masks)
        for rle, label, score in zip(rles, result["labels"], result["scores"]):
            rle["counts"] = rle["counts"].decode("ascii")
            predictions.append(
                {
                    "image_id": int(image_id_by_name[filename]),
                    "category_id": int(COCO_CATEGORY_IDS[int(label)]),
                    "segmentation": rle,
                    "score": float(score),
                }
            )
        print_progress(index, len(names), filename)
    if not predictions:
        raise RuntimeError("No instance predictions were produced")
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    if limit is not None:
        coco_eval.params.imgIds = [image_id_by_name[name] for name in names]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return {
        "ap": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
        "images": len(names),
    }


def rgb_to_id(target: np.ndarray) -> np.ndarray:
    target = target.astype(np.int64)
    return target[..., 0] + 256 * target[..., 1] + 256**2 * target[..., 2]


def make_panoptic_target(
    height: int,
    width: int,
    num_classes: int,
    segments: Iterable[tuple[np.ndarray, int, int]],
) -> torch.Tensor:
    # Match the original validator: unlabeled and crowd pixels are unknown (-1).
    # TorchMetrics maps unknown target colors to its internal void color.
    target = torch.full((height, width, 2), -1, dtype=torch.long)
    for mask, class_id, instance_id in segments:
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.bool_))
        target[..., 0][mask_tensor] = class_id
        target[..., 1][mask_tensor] = instance_id
    return target


def coco_panoptic_target(
    png: np.ndarray, annotation: dict, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = rgb_to_id(png)
    segments = []
    is_crowds = []
    for instance_id, segment in enumerate(annotation["segments_info"]):
        is_crowds.append(bool(segment.get("iscrowd", 0)))
        class_id = COCO_CATEGORY_TO_CONTIGUOUS.get(segment["category_id"])
        if class_id is not None:
            segments.append((ids == segment["id"], class_id, instance_id))
    return (
        make_panoptic_target(ids.shape[0], ids.shape[1], num_classes, segments),
        torch.tensor(is_crowds, dtype=torch.bool),
    )


def ade_panoptic_target(
    semantic: np.ndarray,
    instance: np.ndarray,
    stuff_classes: list[int],
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    stuff_set = set(stuff_classes)
    segments: list[tuple[np.ndarray, int, int]] = []
    next_id = 0
    for raw_label in np.unique(semantic):
        class_id = int(raw_label) - 1
        if class_id in stuff_set:
            segments.append((semantic == raw_label, class_id, next_id))
            next_id += 1
    for raw_instance in np.unique(instance[..., 1]):
        if raw_instance == 0:
            continue
        mask = instance[..., 1] == raw_instance
        raw_class = np.unique(instance[..., 0][mask])
        if len(raw_class) != 1:
            raise RuntimeError("ADE20K instance mask contains multiple classes")
        thing_index = int(raw_class[0]) - 1
        if 0 <= thing_index < len(ADE_INSTANCE_TO_SEMANTIC):
            segments.append((mask, ADE_INSTANCE_TO_SEMANTIC[thing_index], next_id))
            next_id += 1
    return (
        make_panoptic_target(
            semantic.shape[0], semantic.shape[1], num_classes, segments
        ),
        torch.zeros(next_id, dtype=torch.bool),
    )


def update_panoptic_metric(
    metric: PanopticQuality,
    prediction: torch.Tensor,
    target: torch.Tensor,
    is_crowd: torch.Tensor,
) -> None:
    """Update PQ with the official category-aware COCO crowd handling."""
    flatten_pred = _prepocess_inputs(
        metric.things,
        metric.stuffs,
        prediction[None],
        metric.void_color,
        metric.allow_unknown_preds_category,
    )[0]
    flatten_target = _prepocess_inputs(
        metric.things,
        metric.stuffs,
        target[None],
        metric.void_color,
        True,
    )[0]
    pred_areas = cast(dict[_Color, torch.Tensor], _get_color_areas(flatten_pred))
    target_areas = cast(dict[_Color, torch.Tensor], _get_color_areas(flatten_target))
    intersection_matrix = torch.transpose(
        torch.stack((flatten_pred, flatten_target), -1), -1, -2
    )
    intersection_areas = cast(
        dict[tuple[_Color, _Color], torch.Tensor],
        _get_color_areas(intersection_matrix),
    )

    pred_segment_matched: set[_Color] = set()
    target_segment_matched: set[_Color] = set()
    for pred_color, target_color in intersection_areas:
        if target_color == metric.void_color:
            continue
        if bool(is_crowd[target_color[1]]):
            continue
        if pred_color[0] != target_color[0]:
            continue
        iou = _calculate_iou(
            pred_color,
            target_color,
            pred_areas,
            target_areas,
            intersection_areas,
            metric.void_color,
        )
        continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
        if iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            metric.iou_sum[continuous_id] += iou
            metric.true_positives[continuous_id] += 1

    false_negative_colors = set(target_areas) - target_segment_matched
    false_positive_colors = set(pred_areas) - pred_segment_matched
    false_negative_colors.discard(metric.void_color)
    false_positive_colors.discard(metric.void_color)

    for target_color in list(false_negative_colors):
        void_target_area = intersection_areas.get((metric.void_color, target_color), 0)
        if void_target_area / target_areas[target_color] > 0.5:
            false_negative_colors.discard(target_color)

    crowd_by_category: dict[int, int] = {}
    for target_color in false_negative_colors:
        if bool(is_crowd[target_color[1]]):
            crowd_by_category[target_color[0]] = target_color[1]
            continue
        continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
        metric.false_negatives[continuous_id] += 1

    for pred_color in list(false_positive_colors):
        pred_void_crowd_area = intersection_areas.get(
            (pred_color, metric.void_color), 0
        )
        if pred_color[0] in crowd_by_category:
            crowd_color = (pred_color[0], crowd_by_category[pred_color[0]])
            pred_void_crowd_area += intersection_areas.get(
                (pred_color, crowd_color), 0
            )
        if pred_void_crowd_area / pred_areas[pred_color] > 0.5:
            false_positive_colors.discard(pred_color)

    for pred_color in false_positive_colors:
        continuous_id = metric.cat_id_to_continuous_id[pred_color[0]]
        metric.false_positives[continuous_id] += 1
    # We update metric states directly to preserve category-aware crowd handling.
    metric._update_count += 1


def evaluate_panoptic(
    predictor: EoMTPredictor,
    source: ImageSource,
    root: Path,
    data_class: str,
    num_classes: int,
    stuff_classes: list[int],
    limit: int | None,
) -> dict[str, float]:
    things = [class_id for class_id in range(num_classes) if class_id not in stuff_classes]
    metric = PanopticQuality(
        things,
        stuff_classes + [num_classes],
        return_sq_and_rq=True,
        return_per_class=True,
    ).to(predictor.device)
    names = source.names
    coco_annotations = None
    coco_targets = None
    ade_targets = None
    if "coco_panoptic" in data_class:
        annotation_data = open_json(
            root,
            "annotations/panoptic_val2017.json",
            ("panoptic_annotations_trainval2017.zip",),
        )
        coco_annotations = {item["file_name"]: item for item in annotation_data["annotations"]}
        names = [name for name in names if f"{Path(name).stem}.png" in coco_annotations]
        coco_targets = COCOPanopticTargetSource(root)
    elif "ade20k_panoptic" in data_class:
        ade_targets = ADETargetSource(root, panoptic=True)
    else:
        raise SystemExit(f"Panoptic metrics are not implemented for {data_class}")

    names = iter_names(names, limit)
    try:
        for index, filename in enumerate(names, 1):
            result = predictor(source.open(filename))
            prediction = torch.from_numpy(
                np.stack((result["semantic"], result["instance"]), axis=-1)
            ).long().to(predictor.device)
            stem = Path(filename).stem
            if coco_targets is not None and coco_annotations is not None:
                target_name = f"{stem}.png"
                target, is_crowd = coco_panoptic_target(
                    coco_targets.array(target_name), coco_annotations[target_name], num_classes
                )
            else:
                assert ade_targets is not None
                target, is_crowd = ade_panoptic_target(
                    ade_targets.semantic_array(stem),
                    ade_targets.instance_array(stem),
                    stuff_classes,
                    num_classes,
                )
            update_panoptic_metric(
                metric,
                prediction,
                target.to(predictor.device),
                is_crowd.to(predictor.device),
            )
            print_progress(index, len(names), filename)
    finally:
        if coco_targets is not None:
            coco_targets.close()
        if ade_targets is not None:
            ade_targets.close()

    per_class = metric.compute()[:-1]
    pq, sq, rq = per_class[:, 0], per_class[:, 1], per_class[:, 2]
    num_things = len(things)
    return {
        "pq": float(pq.mean()),
        "sq": float(sq.mean()),
        "rq": float(rq.mean()),
        "pq_things": float(pq[:num_things].mean()) if num_things else 0.0,
        "pq_stuff": float(pq[num_things:].mean()) if len(stuff_classes) else 0.0,
        "images": len(names),
    }


def write_results(output_root: Path, task: str, model_name: str, results: dict) -> Path:
    output = output_root / task / model_name
    output.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "model": model_name, **results}
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(payload))
        writer.writeheader()
        writer.writerow(payload)
    return output


def main() -> None:
    args = parse_args()
    config, data_class, task, num_classes, stuff_classes = load_config(args.config)
    validate_inputs(args, task)
    if "ade20k" in data_class:
        dataset = "ade20k"
    elif "cityscapes" in data_class:
        dataset = "cityscapes"
    else:
        dataset = "coco"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "task": task,
                    "dataset": dataset,
                    "config": str(args.config.resolve()),
                    "checkpoint": str(args.checkpoint.resolve()),
                    "data": str(args.data.resolve()),
                },
                indent=2,
            )
        )
        return

    predictor = EoMTPredictor(config, args.checkpoint, args.device)
    source = ImageSource(args.data, dataset)
    try:
        if task == "semantic":
            if dataset == "ade20k":
                targets = ADETargetSource(args.data, panoptic=False)
            elif dataset == "cityscapes":
                targets = CityscapesTargetSource(args.data)
            else:
                raise SystemExit(f"Semantic metrics are not implemented for {data_class}")
            try:
                results = evaluate_semantic(
                    predictor, source, targets, num_classes, args.limit
                )
            finally:
                targets.close()
        elif task == "instance":
            results = evaluate_instance(predictor, source, args.data, args.limit)
        else:
            results = evaluate_panoptic(
                predictor,
                source,
                args.data,
                data_class,
                num_classes,
                stuff_classes,
                args.limit,
            )
    finally:
        source.close()

    output = write_results(args.output, task, args.config.stem, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Metrics written to {output}")


if __name__ == "__main__":
    main()
