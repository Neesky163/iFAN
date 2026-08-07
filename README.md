# iFAN: Inference-Aware Learning for Plain Mask Transformers

This repository currently provides inference code for EoMT-iFAN and PMT-iFAN only. The remaining code, including the training pipeline, will be released upon acceptance of the paper.

| Implementation | Backbone | Tasks | Documentation |
|---|---|---|---|
| PMT-iFAN | DINOv3 ViT-L/16 | ADE20K semantic/panoptic; COCO instance/panoptic | [PMT/README.md](PMT/README.md) |
| EoMT-iFAN | DINOv2 ViT-L/ViT-G | ADE20K and Cityscapes semantic; ADE20K and COCO panoptic; COCO instance | [EOMT/README.md](EOMT/README.md) |


## Repository structure

```text
.
├── PMT/                 # PMT-iFAN code, configs
└── EOMT/                # EoMT-iFAN code, configs
```

PMT and EoMT use different pinned PyTorch versions. Install them in separate
environments and follow the instructions in the corresponding subdirectory.

## Checkpoints

Pretrained checkpoints are available upon request. Please submit the
[checkpoint access form](https://docs.google.com/forms/d/e/1FAIpQLSf3r8iXFp233mapvFjmfbQQSQEgyo0CdbCUKz2Z77Rwe-Seqg/viewform?usp=publish-editor); once your request is approved, you will be able to download the checkpoints.

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


## Acknowledgments

We sincerely thank the authors of [EoMT](https://github.com/tue-mps/eomt) and [PMT](https://github.com/tue-mps/pmt) for open-sourcing their excellent work and codebases. Our implementation greatly benefits from these projects.
