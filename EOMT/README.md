# EoMT-iFAN: Inference and Evaluation

[Repository overview](../README.md)

This directory contains a self-contained, inference-only EoMT-iFAN release for
semantic, instance, and panoptic segmentation. 

## Contents

```text
EOMT/
├── checkpoints/
│   ├── semantic/{configs,inference}/
│   ├── instance/{configs,inference}/
│   └── panoptic/{configs,inference}/
├── models/                     # EoMT and ViT definitions
├── scripts/                    # task-specific evaluation wrappers
├── infer.py                    # prediction for an image or directory
├── evaluate.py                 # standalone validation
└── requirements.txt
```

## Checkpoints

Pretrained EoMT-iFAN checkpoints are available upon request. Please submit the
[checkpoint access form](https://docs.google.com/forms/d/e/1FAIpQLSf3r8iXFp233mapvFjmfbQQSQEgyo0CdbCUKz2Z77Rwe-Seqg/viewform?usp=publish-editor); once your request is approved, you will be able to download the checkpoints.

## Installation

Python 3.10 or later and a CUDA-capable GPU are recommended. You can refer to the [EoMT](https://github.com/tue-mps/eomt) for configuration.

```bash
cd EOMT
python3 -m pip install -r requirements.txt
```


## Image prediction

`infer.py` accepts one image or all supported images in the top level of a
directory.

Semantic segmentation:

```bash
python3 infer.py \
  --config checkpoints/semantic/configs/ade20k_semantic_L_512.yaml \
  --checkpoint checkpoints/semantic/inference/ade20k_semantic_L_512.ckpt \
  --input /path/to/image-or-directory \
  --output outputs/semantic \
  --device cuda:0
```

Instance segmentation:

```bash
python3 infer.py \
  --config checkpoints/instance/configs/coco_instance_L_640.yaml \
  --checkpoint checkpoints/instance/inference/coco_instance_L_640.ckpt \
  --input /path/to/image-or-directory \
  --output outputs/instance \
  --device cuda:0
```

Panoptic segmentation:

```bash
python3 infer.py \
  --config checkpoints/panoptic/configs/coco_panoptic_vitl_640.yaml \
  --checkpoint checkpoints/panoptic/inference/coco_panoptic_vitl_640.ckpt \
  --input /path/to/image-or-directory \
  --output outputs/panoptic \
  --device cuda:0
```

Each image produces:

- `<name>.npz`: raw arrays returned by the task-specific predictor;
- `<name>_overlay.png`: an RGB segmentation overlay;
- `<name>.json`: array names and shapes.

Validate a config/checkpoint pair without loading the model:

```bash
python3 infer.py \
  --config checkpoints/semantic/configs/ade20k_semantic_L_512.yaml \
  --checkpoint checkpoints/semantic/inference/ade20k_semantic_L_512.ckpt \
  --dry-run
```

## Benchmark evaluation

`evaluate.py` reports mIoU for semantic segmentation, COCO mask AP for
instance segmentation, and PQ/SQ/RQ for panoptic segmentation. It reads either
the official archives or the corresponding extracted directories.

Expected dataset roots:

```text
/path/to/ade20k/
├── ADEChallengeData2016.zip
└── annotations_instance.zip       # .tar also supported; panoptic only

/path/to/cityscapes/
├── leftImg8bit_trainvaltest.zip
└── gtFine_trainvaltest.zip

/path/to/coco2017/
├── val2017.zip
├── annotations_trainval2017.zip
└── panoptic_annotations_trainval2017.zip
```

Run the default checkpoint for each task:

```bash
# ADE20K semantic, ViT-L/512
DATA_PATH=/path/to/ade20k GPU=0 bash scripts/test_semantic_metrics.sh

# COCO instance, ViT-L/640
DATA_PATH=/path/to/coco2017 GPU=0 bash scripts/test_instance_metrics.sh

# COCO panoptic, ViT-L/640
DATA_PATH=/path/to/coco2017 GPU=0 bash scripts/test_panoptic_metrics.sh
```

Override both paths to evaluate another released config/checkpoint pair:

```bash
CONFIG_PATH=checkpoints/panoptic/configs/ade20k_panoptic_vitg_1280.yaml \
CKPT_PATH=checkpoints/panoptic/inference/ade20k_panoptic_vitg_1280.ckpt \
DATA_PATH=/path/to/ade20k \
GPU=0 \
bash scripts/test_panoptic_metrics.sh
```

Cityscapes semantic evaluation uses the same semantic wrapper:

```bash
CONFIG_PATH=checkpoints/semantic/configs/cityscapes_semantic_L_1024.yaml \
CKPT_PATH=checkpoints/semantic/inference/cityscapes_semantic_L_1024.ckpt \
DATA_PATH=/path/to/cityscapes \
GPU=0 \
bash scripts/test_semantic_metrics.sh
```

Useful environment variables:

| Variable | Purpose |
|---|---|
| `GPU` | Physical GPU ID exposed to the evaluator; default: `0` |
| `LIMIT` | Evaluate only the first N images for a smoke test |
| `DRY_RUN=1` | Check paths and pairing without loading a model |
| `LOG_ROOT` | Override the default `metric_test_results/` output root |
| `CHECKPOINT_ROOT` | Use a checkpoint tree stored outside this directory |

Results are written to
`metric_test_results/<task>/<config-stem>/metrics.{json,csv}`. Runs with
`LIMIT` are smoke tests and are not comparable with full validation results.

## Released models and reference results

The checkpoint tree contains:

- ADE20K semantic: ViT-L, 512;
- Cityscapes semantic: ViT-L, 1024;
- COCO instance: ViT-L, 640 and 1280;
- ADE20K panoptic: ViT-L and ViT-G, 640 and 1280;
- COCO panoptic: ViT-L and ViT-G, 640 and 1280.
