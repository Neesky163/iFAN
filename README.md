# iFAN: Inference and Evaluation Code

This repository provides the inference-only release accompanying the iFAN
paper. It contains two self-contained implementations for semantic, instance,
and panoptic segmentation:

| Implementation | Backbone | Tasks | Documentation |
|---|---|---|---|
| PMT-iFAN | DINOv3 ViT-L/16 | ADE20K semantic/panoptic; COCO instance/panoptic | [PMT/README.md](PMT/README.md) |
| EoMT-iFAN | DINOv2 ViT-L/ViT-G | ADE20K and Cityscapes semantic; ADE20K and COCO panoptic; COCO instance | [EOMT/README.md](EOMT/README.md) |

The release is intentionally limited to model construction, checkpoint
loading, image prediction, and validation-set evaluation. Training code and
training-only dependencies are not included.

## Repository structure

```text
.
├── PMT/                 # PMT-iFAN code, configs, and weights
└── EOMT/                # EoMT-iFAN code, configs, and weights
```

PMT and EoMT use different pinned PyTorch versions. Install them in separate
environments and follow the instructions in the corresponding subdirectory.

## Quick start

For PMT-iFAN:

```bash
cd PMT
python3 -m pip install -r requirements.txt
python3 predict.py \
  --config configs/ade20k_semantic_pmt_ifan_512_inference.yaml \
  --input /path/to/image.jpg \
  --output outputs/semantic
```

For EoMT-iFAN:

```bash
cd EOMT
python3 -m pip install -r requirements.txt
python3 infer.py \
  --config checkpoints/semantic/configs/ade20k_semantic_L_512.yaml \
  --checkpoint checkpoints/semantic/inference/ade20k_semantic_L_512.ckpt \
  --input /path/to/image.jpg \
  --output outputs/semantic
```

See the implementation-specific READMEs for dataset preparation, all released
checkpoints, benchmark commands, output formats, and troubleshooting.

## Scope and reproducibility

- Released configs and inference checkpoints are paired by filename.
- Evaluation can read the official ADE20K, COCO 2017, and Cityscapes archives
  directly; the expected layouts are documented in each subdirectory.
- Evaluation commands produce machine-readable JSON/CSV summaries for
  reproducibility.
- Small differences in the last reported digits may occur across CUDA,
  PyTorch, and GPU versions.

## Citation

If you use this code or the released checkpoints, please cite the accompanying
iFAN paper. Citation metadata will be added after publication.
