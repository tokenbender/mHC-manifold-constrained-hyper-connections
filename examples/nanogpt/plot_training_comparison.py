"""Plot report-ready training and benchmark comparisons for nanoGPT T4 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid run spec {item!r}; expected name=path")
        name, path = item.split("=", 1)
        if not name or not path:
            raise ValueError(f"invalid run spec {item!r}; expected name=path")
        runs.append((name, Path(path)))
    return runs


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_metrics(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = []
    evals = []
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return train, evals
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("event") == "train":
            train.append(payload)
        elif payload.get("event") == "eval":
            evals.append(payload)
    return train, evals


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric_series(rows: list[dict[str, Any]], y_key: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for idx, row in enumerate(rows):
        y = as_float(row.get(y_key))
        if y is None:
            continue
        xs.append(int(row.get("iter", idx)))
        ys.append(y)
    return xs, ys


def save_line_plot(path: Path, title: str, ylabel: str, series: dict[str, tuple[list[int], list[float]]]) -> None:
    plt.figure(figsize=(8, 4.5))
    for name, (xs, ys) in series.items():
        if xs and ys:
            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=name)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_bar_plot(path: Path, title: str, ylabel: str, values: dict[str, float]) -> None:
    plt.figure(figsize=(6.5, 4.0))
    names = list(values)
    vals = [values[name] for name in names]
    plt.bar(names, vals, color=["#4C78A8", "#F58518", "#54A24B"][: len(names)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def final_val_loss(summary: dict[str, Any], eval_rows: list[dict[str, Any]]) -> float | None:
    last_eval = summary.get("last_eval")
    if isinstance(last_eval, dict):
        value = as_float(last_eval.get("val"))
        if value is not None:
            return value
    for row in reversed(eval_rows):
        value = as_float(row.get("val_loss"))
        if value is not None:
            return value
    return None


def read_benchmark_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def benchmark_values(rows: list[dict[str, str]], key: str) -> dict[str, float]:
    values = {}
    for row in rows:
        variant = row.get("variant", "")
        value = as_float(row.get(key))
        if variant and value is not None:
            values[variant] = value
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot nanoGPT training comparisons.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run specs as variant=path")
    parser.add_argument("--benchmark-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        runs = parse_runs(args.runs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_series = {}
    val_series = {}
    tok_series = {}
    vram_series = {}
    best_val = {}
    final_ppl = {}

    for name, run_dir in runs:
        train_rows, eval_rows = read_metrics(run_dir)
        summary = read_json(run_dir / "summary.json")

        train_series[name] = metric_series(train_rows, "train_loss")
        val_series[name] = metric_series(eval_rows, "val_loss")
        tok_series[name] = metric_series(train_rows, "tokens_per_sec")
        vram_series[name] = metric_series(train_rows, "peak_vram_mb")

        best = as_float(summary.get("best_val_loss"))
        if best is None:
            vals = [as_float(row.get("best_val_loss")) for row in eval_rows]
            vals = [v for v in vals if v is not None]
            best = min(vals) if vals else None
        if best is not None:
            best_val[name] = best

        final_val = final_val_loss(summary, eval_rows)
        if final_val is not None:
            final_ppl[name] = math.exp(final_val)

    save_line_plot(args.output_dir / "train_loss_curve.png", "Train Loss", "loss", train_series)
    save_line_plot(args.output_dir / "val_loss_curve.png", "Validation Loss", "loss", val_series)
    save_bar_plot(args.output_dir / "best_val_loss_bar.png", "Best Validation Loss", "loss", best_val)
    save_bar_plot(args.output_dir / "final_val_ppl_bar.png", "Final Validation Perplexity", "perplexity", final_ppl)
    save_line_plot(args.output_dir / "tokens_per_sec_curve.png", "Training Throughput", "tokens/sec", tok_series)
    save_line_plot(args.output_dir / "training_peak_vram_curve.png", "Training Peak VRAM", "MB", vram_series)
    save_line_plot(args.output_dir / "peak_vram_curve.png", "Training Peak VRAM", "MB", vram_series)

    if args.benchmark_summary is not None and args.benchmark_summary.exists():
        rows = read_benchmark_summary(args.benchmark_summary)
        save_bar_plot(
            args.output_dir / "inference_tokens_per_sec_bar.png",
            "Inference Throughput",
            "tokens/sec",
            benchmark_values(rows, "tokens_per_sec_mean"),
        )
        save_bar_plot(
            args.output_dir / "decode_ms_per_token_bar.png",
            "Decode Latency",
            "ms/token",
            benchmark_values(rows, "full_context_generation_latency_per_token_ms_mean"),
        )
        save_bar_plot(
            args.output_dir / "benchmark_peak_vram_bar.png",
            "Benchmark Peak VRAM",
            "MB",
            benchmark_values(rows, "peak_vram_mb_max"),
        )

    for path in sorted(args.output_dir.glob("*.png")):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
