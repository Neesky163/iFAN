# PMT-iFAN: Inference and Evaluation

[Repository overview](../README.md)

This directory contains the inference-only PMT-iFAN release with a frozen
DINOv3 ViT-L/16 encoder. It supports semantic, instance, and panoptic
segmentation on ADE20K and COCO 2017.

## Contents

```text
PMT/
├── checkpoints/          # seven released inference checkpoints
├── configs/              # matching inference configurations
├── image/                # model, datasets, post-processing, and metrics
├── results/              # validation logs and summary (created on demand)
├── predict.py            # prediction for an image or a directory
├── run_validation.py     # single- or multi-checkpoint evaluation
└── requirements.txt
```

Paths in the configs are portable: no checkpoint, dataset, user, or local
cache path is embedded in a YAML file.

The `trainer` and data-loader sections are intentionally retained because the
same configs drive Lightning's validation-only CLI. They control evaluation
devices, mixed precision, logging, batch size, and workers; they do not enable
training. Model/data `class_path` entries are likewise required to construct
the validation pipeline.

## Installation

Python 3.10 or later, a CUDA-capable GPU, and a recent NVIDIA driver are
recommended. Create a dedicated environment because the PMT and EoMT releases
pin different PyTorch versions.

```bash
cd PMT
python3 -m pip install -r requirements.txt
```

The encoder is loaded from the gated Hugging Face repository
`facebook/dinov3-vitl16-pretrain-lvd1689m`. Accept its access terms and log in
once:

```bash
hf auth login
```

For offline use, pass a local Transformers model directory through
`--backbone`.

## Image prediction

`predict.py` accepts one image or all supported images in the top level of a
directory. If `--checkpoint` is omitted, the checkpoint with the matching
config filename is selected from `checkpoints/`.

Semantic segmentation:

```bash
python3 predict.py \
  --config configs/ade20k_semantic_pmt_ifan_512_inference.yaml \
  --input /path/to/image-or-directory \
  --output outputs/semantic
```

Instance segmentation:

```bash
python3 predict.py \
  --config configs/coco_instance_pmt_ifan_640_inference.yaml \
  --input /path/to/image-or-directory \
  --output outputs/instance \
  --score-threshold 0.30
```

Panoptic segmentation:

```bash
python3 predict.py \
  --config configs/coco_panoptic_pmt_ifan_640_inference.yaml \
  --input /path/to/image-or-directory \
  --output outputs/panoptic
```

Common options:

| Option | Description |
|---|---|
| `--device auto\|cpu\|cuda\|cuda:N` | Inference device; default: `auto` |
| `--checkpoint PATH` | Override automatic checkpoint selection |
| `--backbone MODEL_OR_PATH` | Hugging Face model ID or local model directory |
| `--score-threshold FLOAT` | Instance score threshold; default: `0.30` |

For each input image, prediction writes a visualization, a JSON summary, and a
machine-readable array:

| Task | Array output | Contents |
|---|---|---|
| Semantic | `<name>.labels.npy` | `H x W` zero-based class IDs |
| Instance | `<name>.instances.npz` | binary masks, class IDs, and scores |
| Panoptic | `<name>.panoptic.npz` | per-pixel class IDs and segment IDs |

The visualization is saved as `<name>.visualization.png`; `<name>.json`
contains the predicted classes, instances, or segments. Dataset class names
are not stored in the checkpoints.

## Benchmark evaluation

Evaluation supports the official validation archives in the following layout:

```text
/path/to/ade20k/
├── ADEChallengeData2016.zip
└── annotations_instance.zip       # additionally required for panoptic

/path/to/coco2017/
├── val2017.zip
├── annotations_trainval2017.zip
└── panoptic_annotations_trainval2017.zip
```

Evaluate one checkpoint:

```bash
# ADE20K semantic segmentation
python3 run_validation.py ade20k_semantic_pmt_ifan_512 \
  --ade20k-root /path/to/ade20k \
  --gpus 0

# COCO instance segmentation
python3 run_validation.py coco_instance_pmt_ifan_640 \
  --coco-root /path/to/coco2017 \
  --gpus 0
```

Evaluate all seven checkpoints, distributing serial job queues across GPUs:

```bash
python3 run_validation.py \
  --ade20k-root /path/to/ade20k \
  --coco-root /path/to/coco2017 \
  --gpus 0,1,2,3
```

For an offline backbone or a custom checkpoint directory:

```bash
python3 run_validation.py \
  --ade20k-root /path/to/ade20k \
  --coco-root /path/to/coco2017 \
  --backbone /path/to/dinov3-vitl16-pretrain-lvd1689m \
  --checkpoint-dir /path/to/checkpoints \
  --gpus 0
```

Check config/checkpoint discovery without loading a model:

```bash
python3 run_validation.py --checkpoint-dir checkpoints --gpus 0 --dry-run
```

Additional Lightning CLI arguments can be repeated, for example:

```bash
python3 run_validation.py coco_instance_pmt_ifan_1280 \
  --coco-root /path/to/coco2017 \
  --gpus 0 \
  --extra-arg=--data.batch_size=1
```

Evaluation writes one log per checkpoint and a machine-readable summary to
`results/summary.json`.

## Released checkpoints and reference results

The following primary metrics were reproduced with the included checkpoints.
Values are shown as percentages; the JSON summary stores values in `[0, 1]`.

| Dataset | Task | Input | Primary metric | Result (%) |
|---|---|---:|---|---:|
| ADE20K | Semantic | 512 | mIoU | 59.50 |
| ADE20K | Panoptic | 640 | PQ | 50.67 |
| ADE20K | Panoptic | 1280 | PQ | 52.97 |
| COCO | Instance | 640 | mask AP | 46.63 |
| COCO | Instance | 1280 | mask AP | 50.56 |
| COCO | Panoptic | 640 | PQ | 56.69 |
| COCO | Panoptic | 1280 | PQ | 58.63 |

Running `run_validation.py` records full AP/PQ breakdowns, runtime, and the
executed commands in `results/summary.json`.

## Troubleshooting

- **Checkpoint not found:** preserve the released `*_inference.pth` filenames,
  or pass `--checkpoint`/`--checkpoint-dir` explicitly.
- **Hugging Face 401 or `GatedRepoError`:** accept the DINOv3 access terms and
  run `hf auth login`, or use a local backbone with `--backbone`.
- **CUDA out of memory:** use a 640-pixel config, set validation batch size to
  one, or predict images individually.
- **Dataset file not found:** pass the directory that directly contains the
  official archives shown above.
