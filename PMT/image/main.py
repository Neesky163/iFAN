"""Validation-only CLI for PMT-iFAN image segmentation."""

import logging
import os

import jsonargparse._typehints as _t
import torch
from lightning.pytorch import cli

from datasets.lightning_data_module import LightningDataModule
from inference.lightning_module import LightningModule
from utils import suppress_warnings


os.environ["TORCH_LOGS"] = "-dynamo"

# Keep compatibility with the jsonargparse version used by the original project.
_orig_single = _t.raise_unexpected_value
_orig_union = _t.raise_union_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


def _raise_union(subtypes, val, vals):
    for error in reversed(vals):
        if isinstance(error, Exception):
            raise error
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union


class InferenceCLI(cli.LightningCLI):
    @staticmethod
    def subcommands():
        return {"validate": {"model", "dataloaders", "datamodule"}}

    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        suppress_warnings()
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )
        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )


def cli_main():
    InferenceCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "devices": 1,
        },
    )


if __name__ == "__main__":
    cli_main()
