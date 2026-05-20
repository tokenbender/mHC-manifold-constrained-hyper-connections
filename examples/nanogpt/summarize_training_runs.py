"""Summarize nanoGPT training output directories into report-ready files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUNS = [
    ("baseline", "examples/nanogpt/out-t4-baseline"),
    ("hc", "examples/nanogpt/out-t4-hc"),
    ("mhc", "examples/nanogpt/out-t4-mhc"),
]

FIELDNAMES = [
    "variant",
    "run_dir",
    "checkpoint_path",
    "checkpoint_exists",
    "ok",
    "status",
    "best_val_loss",
    "final_train_loss",
    "final_val_loss",
    "final_val_ppl",
    "tokens_seen",
    "iter_num",
    "elapsed_s",
    "device_type",
    "dtype",
    "dataset",
    "max_iters",
    "eval_interval",
    "batch_size",
    "gradient_accumulation_steps",
    "n_layer",
    "n_head",
    "n_embd",
    "block_size",
    "hc_num_streams",
    "hc_disable",
    "mhc",
    "command",
]


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path.name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"invalid JSON object in {path.name}"
    return payload, None


def scan_metrics_jsonl(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, int]:
    last_valid = None
    last_train = None
    last_eval = None
    skipped = 0
    if not path.exists():
        return last_valid, last_train, last_eval, skipped
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(payload, dict):
            skipped += 1
            continue
        last_valid = payload
        if payload.get("event") == "train":
            last_train = payload
        elif payload.get("event") == "eval":
            last_eval = payload
    return last_valid, last_train, last_eval, skipped


def parse_argv_overrides(metadata: dict[str, Any] | None) -> dict[str, str]:
    if not metadata:
        return {}
    argv = metadata.get("argv", [])
    if not isinstance(argv, list):
        return {}
    overrides: dict[str, str] = {}
    for item in argv:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            overrides[key] = value.strip()
    return overrides


def clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return value
    return value


def coerce_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def perplexity(loss: Any) -> float | None:
    number = coerce_float(loss)
    if number is None:
        return None
    try:
        value = math.exp(number)
    except OverflowError:
        return None
    if not math.isfinite(value):
        return None
    return value


def get_config_value(config: dict[str, Any] | None, overrides: dict[str, str], key: str) -> Any:
    if config and key in config:
        return config.get(key)
    return overrides.get(key, "")


def unquote_command_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parts = shlex.split(value)
    except ValueError:
        return value
    return parts[0] if len(parts) == 1 else value


def command_text(run_dir: Path, metadata: dict[str, Any] | None) -> str:
    command_path = run_dir / "command.sh"
    if command_path.exists():
        return " ".join(
            line.strip()
            for line in command_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#") and not line.startswith("set ")
        )
    if metadata and isinstance(metadata.get("argv"), list):
        return shlex.join(str(part) for part in metadata["argv"])
    return ""


def summary_last_eval(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    value = summary.get("last_eval")
    return value if isinstance(value, dict) else {}


def row_for_run(variant: str, run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "ckpt.pt"
    row: dict[str, Any] = {
        "variant": variant,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
    }

    messages: list[str] = []
    if not run_dir.exists():
        row.update({field: "" for field in FIELDNAMES if field not in row})
        row["ok"] = False
        row["status"] = "run directory missing"
        return row

    summary, summary_error = read_json(run_dir / "summary.json")
    config, config_error = read_json(run_dir / "config_effective.json")
    metadata, metadata_error = read_json(run_dir / "run_metadata.json")
    for error in (summary_error, config_error, metadata_error):
        if error:
            messages.append(error)

    last_metric, last_train, last_eval_metric, skipped_jsonl = scan_metrics_jsonl(run_dir / "metrics.jsonl")
    if skipped_jsonl:
        messages.append(f"skipped {skipped_jsonl} invalid metrics.jsonl line(s)")

    last_eval_summary = summary_last_eval(summary)
    overrides = parse_argv_overrides(metadata)

    best_val_loss = (
        summary.get("best_val_loss") if summary and "best_val_loss" in summary else None
    )
    if best_val_loss is None and last_eval_metric:
        best_val_loss = last_eval_metric.get("best_val_loss")

    final_train_loss = last_eval_summary.get("train")
    if final_train_loss is None and last_eval_metric:
        final_train_loss = last_eval_metric.get("train_loss_eval")
    if final_train_loss is None and last_train:
        final_train_loss = last_train.get("train_loss")

    final_val_loss = last_eval_summary.get("val")
    if final_val_loss is None and last_eval_metric:
        final_val_loss = last_eval_metric.get("val_loss")

    iter_num = summary.get("iter_num") if summary and "iter_num" in summary else None
    if iter_num is None and last_metric:
        iter_num = last_metric.get("iter")

    elapsed_s = summary.get("elapsed_s") if summary and "elapsed_s" in summary else None
    if elapsed_s is None and metadata and last_metric:
        ts = coerce_float(metadata.get("ts"))
        metric_ts = coerce_float(last_metric.get("timestamp"))
        if ts is not None and metric_ts is not None and metric_ts >= ts:
            elapsed_s = metric_ts - ts

    device_type = ""
    if metadata:
        device_type = metadata.get("device_type", "")
    if not device_type and summary:
        device = str(summary.get("device", ""))
        device_type = device.split(":", 1)[0] if device else ""

    summary_ok = summary.get("ok") if summary and "ok" in summary else None
    ok = bool(summary_ok) and checkpoint_path.exists() if summary_ok is not None else False
    if not checkpoint_path.exists():
        messages.append("checkpoint missing")
    if not summary:
        messages.append("summary.json missing")

    row.update(
        {
            "ok": ok,
            "status": "; ".join(messages) if messages else "ok",
            "best_val_loss": clean_scalar(best_val_loss),
            "final_train_loss": clean_scalar(final_train_loss),
            "final_val_loss": clean_scalar(final_val_loss),
            "final_val_ppl": clean_scalar(perplexity(final_val_loss)),
            "tokens_seen": clean_scalar(summary.get("tokens_seen") if summary else ""),
            "iter_num": clean_scalar(iter_num),
            "elapsed_s": clean_scalar(elapsed_s),
            "device_type": clean_scalar(device_type),
            "dtype": clean_scalar((summary or {}).get("dtype", "") or get_config_value(config, overrides, "dtype")),
            "dataset": clean_scalar(get_config_value(config, overrides, "dataset")),
            "max_iters": clean_scalar(
                (summary or {}).get("max_iters")
                if summary and "max_iters" in summary
                else get_config_value(config, overrides, "max_iters")
            ),
            "eval_interval": clean_scalar(get_config_value(config, overrides, "eval_interval")),
            "batch_size": clean_scalar(get_config_value(config, overrides, "batch_size")),
            "gradient_accumulation_steps": clean_scalar(
                get_config_value(config, overrides, "gradient_accumulation_steps_total")
                or get_config_value(config, overrides, "gradient_accumulation_steps")
            ),
            "n_layer": clean_scalar(get_config_value(config, overrides, "n_layer")),
            "n_head": clean_scalar(get_config_value(config, overrides, "n_head")),
            "n_embd": clean_scalar(get_config_value(config, overrides, "n_embd")),
            "block_size": clean_scalar(get_config_value(config, overrides, "block_size")),
            "hc_num_streams": clean_scalar(get_config_value(config, overrides, "hc_num_streams")),
            "hc_disable": clean_scalar(get_config_value(config, overrides, "hc_disable")),
            "mhc": clean_scalar(get_config_value(config, overrides, "mhc")),
            "command": command_text(run_dir, metadata),
        }
    )
    row["dtype"] = unquote_command_value(row["dtype"])
    return row


def parse_runs(run_args: list[str] | None) -> list[tuple[str, Path]]:
    values = run_args if run_args else [f"{name}={path}" for name, path in DEFAULT_RUNS]
    runs = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --runs entry {item!r}; expected variant=path")
        variant, path = item.split("=", 1)
        variant = variant.strip()
        path = path.strip()
        if not variant or not path:
            raise ValueError(f"invalid --runs entry {item!r}; expected variant=path")
        runs.append((variant, Path(path)))
    return runs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def md_escape(value: Any) -> str:
    text = str(value) if value is not None else ""
    return text.replace("\n", " ").replace("|", "\\|")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Training Summary",
        "",
        "| " + " | ".join(FIELDNAMES) + " |",
        "| " + " | ".join(["---"] * len(FIELDNAMES)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(row.get(field, "")) for field in FIELDNAMES) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {"runs": rows, "fieldnames": FIELDNAMES}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize nanoGPT training run directories into CSV, Markdown, and JSON."
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        help="Run specs in stable order, as variant=path. Defaults to baseline/hc/mhc T4 dirs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        runs = parse_runs(args.runs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row_for_run(variant, run_dir) for variant, run_dir in runs]

    write_csv(args.output_dir / "training_summary.csv", rows)
    write_markdown(args.output_dir / "training_summary.md", rows)
    write_json(args.output_dir / "training_summary.json", rows)

    print(f"wrote {args.output_dir / 'training_summary.csv'}")
    print(f"wrote {args.output_dir / 'training_summary.md'}")
    print(f"wrote {args.output_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
