#!/usr/bin/env python3
"""Validate one or more portable PMT-iFAN checkpoint/config pairs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PRIMARY_METRICS = (
    "metrics/val_iou_all",
    "metrics/val_ap_all",
    "metrics/val_pq_all",
)
METRIC_RE = re.compile(r"(metrics/val_[a-zA-Z0-9_]+)\s*[│|]\s*(-?\d+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PMT-iFAN inference checkpoints."
    )
    parser.add_argument(
        "selectors",
        nargs="*",
        help="Config stem or substring. Omit to validate every config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Directory containing *_inference.pth files.",
    )
    parser.add_argument("--ade20k-root", type=Path, help="ADE20K prepared-data root.")
    parser.add_argument("--coco-root", type=Path, help="COCO 2017 prepared-data root.")
    parser.add_argument(
        "--backbone",
        help=(
            "Hugging Face model ID or local DINOv3 directory. When omitted, "
            "the value in each config is used."
        ),
    )
    parser.add_argument(
        "--gpus", default="0", help="Comma-separated GPU IDs (default: 0)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra Lightning CLI argument; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover(args: argparse.Namespace):
    configs = sorted((PROJECT_ROOT / "configs").glob("*_inference.yaml"))
    if args.selectors:
        configs = [
            config
            for config in configs
            if any(selector in config.stem for selector in args.selectors)
        ]
    if not configs:
        raise SystemExit("No matching config was found.")

    jobs = []
    for config in configs:
        name = config.name.removesuffix("_inference.yaml")
        checkpoint = args.checkpoint_dir / f"{name}_inference.pth"
        is_ade20k = name.startswith("ade20k_")
        data_root = args.ade20k_root if is_ade20k else args.coco_root
        if not checkpoint.is_file():
            raise SystemExit(f"Missing checkpoint: {checkpoint}")
        if data_root is None and not args.dry_run:
            option = "--ade20k-root" if is_ade20k else "--coco-root"
            raise SystemExit(f"{option} is required for {name}")
        jobs.append((name, config.resolve(), checkpoint.resolve(), data_root))
    return jobs


def parse_metrics(output: str) -> dict[str, float]:
    return {name: float(value) for name, value in METRIC_RE.findall(output)}


def failure_reason(output: str) -> str | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else None


def run_one(job, gpu: str, args: argparse.Namespace):
    name, config, checkpoint, data_root = job
    command = [
        sys.executable,
        "main.py",
        "validate",
        "--config",
        str(config),
        f"--model.ckpt_path={checkpoint}",
        f"--data.path={data_root or '.'}",
    ]
    if args.backbone:
        command.append(f"--model.network.encoder.backbone_name={args.backbone}")
    command.extend(args.extra_arg)
    log_path = args.output_dir / f"{name}.log"
    result = {
        "name": name,
        "gpu": gpu,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "command": command,
        "log": str(log_path),
    }
    if args.dry_run:
        return {**result, "status": "dry-run", "metrics": {}, "elapsed_seconds": 0.0}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT / "image",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(process.stdout, encoding="utf-8")
    result.update(
        status="ok" if process.returncode == 0 else "failed",
        returncode=process.returncode,
        metrics=parse_metrics(process.stdout),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    if process.returncode:
        result["error"] = failure_reason(process.stdout)
    return result


def main() -> int:
    args = parse_args()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = discover(args)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU ID")

    for index, job in enumerate(jobs):
        print(f"[{index + 1}/{len(jobs)}] GPU {gpus[index % len(gpus)]}: {job[0]}")

    def run_gpu_queue(gpu: str, gpu_jobs):
        return [run_one(job, gpu, args) for job in gpu_jobs]

    results = []
    if args.dry_run or len(gpus) == 1:
        for job in jobs:
            result = run_one(job, gpus[0], args)
            results.append(result)
            print(f"{result['name']}: {result['status']} {result['metrics']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(run_gpu_queue, gpu, jobs[index::len(gpus)])
                for index, gpu in enumerate(gpus)
                if jobs[index::len(gpus)]
            ]
            for future in as_completed(futures):
                for result in future.result():
                    results.append(result)
                    print(
                        f"{result['name']}: {result['status']} {result['metrics']}",
                        flush=True,
                    )

    results.sort(key=lambda item: item["name"])
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
