"""Minimal nanoGPT/mHC inference benchmark CLI.

Benchmarks use synthetic tokens and the repo's PyTorch model directly. This is
not a serving benchmark: no KV cache, no vLLM, no TensorRT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from infer import (
    CliError,
    autocast_context,
    cuda_synchronize,
    format_memory,
    load_model_from_checkpoint,
    parameter_count,
    parse_bool,
    peak_memory_mb,
    reset_peak_memory,
)


REPO_DIR = Path(__file__).resolve().parents[2]


def stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise CliError("cannot summarize empty benchmark samples")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def timed_prefill(model, input_ids, torch, *, device, dtype) -> float:
    cuda_synchronize(torch, device)
    start = time.perf_counter()
    with torch.no_grad(), autocast_context(torch, device, dtype):
        model(input_ids)
    cuda_synchronize(torch, device)
    return (time.perf_counter() - start) * 1000.0


def decode_loop(model, input_ids, torch, *, gen_len: int, block_size: int, device, dtype):
    idx = input_ids
    for _ in range(gen_len):
        idx_cond = idx[:, -block_size:]
        with torch.no_grad(), autocast_context(torch, device, dtype):
            logits, _ = model(idx_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


def timed_decode(model, input_ids, torch, *, gen_len: int, block_size: int, device, dtype) -> float:
    cuda_synchronize(torch, device)
    start = time.perf_counter()
    decode_loop(
        model,
        input_ids,
        torch,
        gen_len=gen_len,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )
    cuda_synchronize(torch, device)
    return (time.perf_counter() - start) * 1000.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "mean", "median", "min", "max", "std", "unit"],
        )
        writer.writeheader()
        for metric, stats in payload["metrics"].items():
            unit = "MB" if metric == "peak_vram_mb" else (
                "tokens/sec" if metric == "tokens_per_sec" else "ms"
            )
            row = {"metric": metric, "unit": unit}
            row.update(stats)
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark nanoGPT/mHC inference.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "float32", "bfloat16"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Context crop for generation; defaults to model block_size",
    )
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--compile", type=parse_bool, default=False)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.batch_size <= 0:
            raise CliError("--batch-size must be > 0")
        if args.prompt_len <= 0:
            raise CliError("--prompt-len must be > 0")
        if args.gen_len <= 0:
            raise CliError("--gen-len must be > 0")
        if args.num_warmup < 0:
            raise CliError("--num-warmup must be >= 0")
        if args.num_iters <= 0:
            raise CliError("--num-iters must be > 0")

        torch, model, config, _, device, dtype = load_model_from_checkpoint(
            ckpt_path=args.ckpt,
            config_path=args.config,
            device=args.device,
            dtype=args.dtype,
            compile_model=args.compile,
        )
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        model_block_size = int(config["block_size"])
        block_size = args.block_size or model_block_size
        if block_size > model_block_size:
            raise CliError(
                f"--block-size {block_size} exceeds model block_size {model_block_size}"
            )
        if args.prompt_len > block_size:
            raise CliError("--prompt-len must be <= effective block size")

        input_ids = torch.randint(
            low=0,
            high=int(config["vocab_size"]),
            size=(args.batch_size, args.prompt_len),
            dtype=torch.long,
            device=device,
        )

        for _ in range(args.num_warmup):
            timed_prefill(model, input_ids, torch, device=device, dtype=dtype)
            timed_decode(
                model,
                input_ids,
                torch,
                gen_len=args.gen_len,
                block_size=block_size,
                device=device,
                dtype=dtype,
            )

        prefill_ms: list[float] = []
        decode_total_ms: list[float] = []
        end_to_end_ms: list[float] = []
        tokens_per_sec: list[float] = []
        peak_vram_mb: list[float] = []

        for _ in range(args.num_iters):
            reset_peak_memory(torch, device)
            prefill = timed_prefill(model, input_ids, torch, device=device, dtype=dtype)
            decode = timed_decode(
                model,
                input_ids,
                torch,
                gen_len=args.gen_len,
                block_size=block_size,
                device=device,
                dtype=dtype,
            )
            total = timed_decode(
                model,
                input_ids,
                torch,
                gen_len=args.gen_len,
                block_size=block_size,
                device=device,
                dtype=dtype,
            )
            prefill_ms.append(prefill)
            decode_total_ms.append(decode / args.gen_len)
            end_to_end_ms.append(total)
            tokens = args.batch_size * args.gen_len
            tokens_per_sec.append(tokens / (total / 1000.0) if total > 0 else math.inf)
            current_peak = peak_memory_mb(torch, device)
            if current_peak is not None:
                peak_vram_mb.append(current_peak)

        metrics = {
            "prefill_latency_ms": stat_summary(prefill_ms),
            "decode_latency_per_token_ms": stat_summary(decode_total_ms),
            "end_to_end_latency_ms": stat_summary(end_to_end_ms),
            "tokens_per_sec": stat_summary(tokens_per_sec),
        }
        if peak_vram_mb:
            metrics["peak_vram_mb"] = stat_summary(peak_vram_mb)
        else:
            metrics["peak_vram_mb"] = {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }

        metadata = {
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "device": str(device),
            "dtype": args.dtype,
            "config_path": str(Path(args.config)),
            "checkpoint_path": str(Path(args.ckpt)),
            "model_parameter_count": parameter_count(model),
            "settings": {
                "batch_size": args.batch_size,
                "block_size": block_size,
                "prompt_len": args.prompt_len,
                "gen_len": args.gen_len,
                "num_warmup": args.num_warmup,
                "num_iters": args.num_iters,
                "compile": args.compile,
                "seed": args.seed,
            },
            "benchmark_note": "simple nanoGPT full-context decode; no KV cache",
        }
        payload = {"metadata": metadata, "metrics": metrics}

        if args.output_json is not None:
            write_json(args.output_json, payload)
        if args.output_csv is not None:
            write_csv(args.output_csv, payload)

        print("Benchmark summary:")
        print(f"device: {device}")
        print(f"dtype: {args.dtype}")
        print(f"parameter_count: {parameter_count(model)}")
        print(
            "prefill_latency_ms.mean: "
            f"{metrics['prefill_latency_ms']['mean']:.3f}"
        )
        print(
            "decode_latency_per_token_ms.mean: "
            f"{metrics['decode_latency_per_token_ms']['mean']:.3f}"
        )
        print(
            "end_to_end_latency_ms.mean: "
            f"{metrics['end_to_end_latency_ms']['mean']:.3f}"
        )
        print(f"tokens_per_sec.mean: {metrics['tokens_per_sec']['mean']:.3f}")
        print(f"peak_vram: {format_memory(metrics['peak_vram_mb']['max'])}")
        return 0
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
