"""Summarize per-variant benchmark JSON files into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "variant",
    "checkpoint_path",
    "config_path",
    "model_parameter_count",
    "dtype",
    "device",
    "gpu_name",
    "prefill_latency_ms_mean",
    "full_context_generation_latency_ms_mean",
    "full_context_generation_latency_per_token_ms_mean",
    "tokens_per_sec_mean",
    "peak_vram_mb_max",
    "peak_vram_mb_mean",
]


def metric_mean(metrics: dict[str, Any], name: str, fallback: str | None = None) -> Any:
    metric = metrics.get(name)
    if metric is None and fallback is not None:
        metric = metrics.get(fallback)
    if not isinstance(metric, dict):
        return ""
    return metric.get("mean", "")


def peak_value(metrics: dict[str, Any], key: str) -> Any:
    metric = metrics.get("peak_vram_mb")
    if not isinstance(metric, dict):
        return ""
    return metric.get(key, "")


def row_from_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    metrics = payload.get("metrics", {})
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        raise ValueError("missing metadata/metrics objects")
    return {
        "variant": path.stem,
        "checkpoint_path": metadata.get("checkpoint_path", ""),
        "config_path": metadata.get("config_path", ""),
        "model_parameter_count": metadata.get("model_parameter_count", ""),
        "dtype": metadata.get("dtype", ""),
        "device": metadata.get("device", ""),
        "gpu_name": metadata.get("gpu_name", ""),
        "prefill_latency_ms_mean": metric_mean(metrics, "prefill_latency_ms"),
        "full_context_generation_latency_ms_mean": metric_mean(
            metrics,
            "full_context_generation_latency_ms",
            fallback="end_to_end_latency_ms",
        ),
        "full_context_generation_latency_per_token_ms_mean": metric_mean(
            metrics,
            "full_context_generation_latency_per_token_ms",
            fallback="decode_latency_per_token_ms",
        ),
        "tokens_per_sec_mean": metric_mean(metrics, "tokens_per_sec"),
        "peak_vram_mb_max": peak_value(metrics, "max"),
        "peak_vram_mb_mean": peak_value(metrics, "mean"),
    }


def collect_rows(benchmark_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    messages = []
    for path in sorted(benchmark_dir.glob("*.json")):
        if path.name in {"summary.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(row_from_payload(path, payload))
        except json.JSONDecodeError:
            messages.append(f"skipped invalid JSON: {path.name}")
        except Exception as exc:
            messages.append(f"skipped invalid benchmark file {path.name}: {exc}")
    return rows, messages


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        "| " + " | ".join(FIELDNAMES) + " |",
        "| " + " | ".join(["---"] * len(FIELDNAMES)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in FIELDNAMES) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark JSON outputs into summary.csv and summary.md."
    )
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args(argv)

    benchmark_dir = args.benchmark_dir
    if not benchmark_dir.exists() or not benchmark_dir.is_dir():
        print(f"error: benchmark directory not found: {benchmark_dir}", file=sys.stderr)
        return 2

    rows, messages = collect_rows(benchmark_dir)
    for message in messages:
        print(message)
    if not rows:
        print("error: no valid benchmark JSON files found", file=sys.stderr)
        return 2

    write_summary_csv(benchmark_dir / "summary.csv", rows)
    write_summary_md(benchmark_dir / "summary.md", rows)
    print(f"wrote {benchmark_dir / 'summary.csv'}")
    print(f"wrote {benchmark_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
