# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Mask2Former repository
# by Facebook, Inc. and its affiliates, used under the Apache 2.0 License.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader

from datasets.dataset import Dataset
from datasets.lightning_data_module import LightningDataModule


CLASS_MAPPING = {i: i - 1 for i in range(1, 151)}

# Maps the 100 contiguous ADE20K instance IDs to the corresponding class ID
# in the 150-class semantic label space.
INSTANCE_MAPPING = (
    7, 8, 10, 12, 14, 15, 18, 19, 20, 22, 23, 24, 27, 30, 31, 32, 33,
    35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47, 49, 50, 53, 55, 56, 57,
    58, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 80, 81,
    82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 95, 97, 98, 102, 103, 104,
    107, 108, 110, 111, 112, 115, 116, 118, 119, 120, 121, 123, 124,
    125, 126, 127, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139,
    142, 143, 144, 146, 147, 148, 149,
)


class ADE20KPanoptic(LightningDataModule):
    def __init__(
        self,
        path,
        stuff_classes: list[int],
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 150,
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.stuff_classes = stuff_classes

    @staticmethod
    def target_parser(target, target_instance, stuff_classes, **kwargs):
        masks, labels = [], []

        for label_id in target[0].unique():
            raw_class_id = label_id.item()
            if raw_class_id not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[raw_class_id]
            if class_id not in stuff_classes:
                continue

            masks.append(target[0] == label_id)
            labels.append(class_id)

        for instance_id in target_instance[1].unique():
            if instance_id == 0:
                continue

            mask = target_instance[1] == instance_id
            contiguous_instance_class = target_instance[0][mask].unique().item() - 1
            masks.append(mask)
            labels.append(INSTANCE_MAPPING[contiguous_instance_class])

        return masks, labels, [False] * len(masks)

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_instance_zip_path": Path(self.path, "annotations_instance.zip"),
            "target_parser": self.target_parser,
            "stuff_classes": self.stuff_classes,
            "check_empty_targets": self.check_empty_targets,
        }
        if stage in (None, "validate"):
            self.val_dataset = Dataset(
                img_folder_path_in_zip=Path("./ADEChallengeData2016/images/validation"),
                target_folder_path_in_zip=Path(
                    "./ADEChallengeData2016/annotations/validation"
                ),
                target_instance_folder_path_in_zip=Path(
                    "./annotations_instance/validation"
                ),
                **dataset_kwargs,
            )
        return self

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
