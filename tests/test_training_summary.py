import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(path: Path, *, variant: str, val_loss: float, bad_jsonl: bool = False) -> None:
    path.mkdir(parents=True)
    (path / "ckpt.pt").write_bytes(b"fake checkpoint")
    (path / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd examples/nanogpt\npython train.py config.py\n",
        encoding="utf-8",
    )
    _write_json(
        path / "summary.json",
        {
            "ok": True,
            "best_val_loss": val_loss - 0.1,
            "last_eval": {"train": val_loss + 0.5, "val": val_loss},
            "tokens_seen": 12345,
            "iter_num": 10,
            "elapsed_s": 12.5,
            "device": "cuda",
            "dtype": "float16",
            "max_iters": 20,
        },
    )
    _write_json(
        path / "config_effective.json",
        {
            "dataset": "fineweb10B",
            "max_iters": 20,
            "batch_size": 8,
            "gradient_accumulation_steps_total": 8,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "block_size": 1024,
            "hc_num_streams": 4,
            "hc_disable": variant == "baseline",
            "mhc": variant == "mhc",
        },
    )
    _write_json(
        path / "run_metadata.json",
        {
            "argv": [
                "train.py",
                "config/train_fineweb10B_t4.py",
                "eval_interval=500",
                "dtype='float16'",
            ],
            "device_type": "cuda",
            "ts": 1000,
        },
    )
    lines = [
        json.dumps({"event": "train", "iter": 9, "train_loss": val_loss + 1.0}),
    ]
    if bad_jsonl:
        lines.append("{bad")
    lines.append(
        json.dumps(
            {
                "event": "eval",
                "iter": 10,
                "train_loss_eval": val_loss + 0.5,
                "val_loss": val_loss,
                "best_val_loss": val_loss - 0.1,
                "timestamp": 1010,
            }
        )
    )
    (path / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_summarize_training_runs_writes_report_files(tmp_path: Path):
    repo_dir = Path(__file__).resolve().parents[1]
    baseline = tmp_path / "baseline"
    hc = tmp_path / "hc"
    mhc = tmp_path / "mhc"
    _write_run(baseline, variant="baseline", val_loss=2.0)
    _write_run(hc, variant="hc", val_loss=2.5, bad_jsonl=True)
    _write_run(mhc, variant="mhc", val_loss=3.0)
    out_dir = tmp_path / "reports"

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "examples" / "nanogpt" / "summarize_training_runs.py"),
            "--runs",
            f"baseline={baseline}",
            f"hc={hc}",
            f"mhc={mhc}",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(repo_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    csv_path = out_dir / "training_summary.csv"
    md_path = out_dir / "training_summary.md"
    json_path = out_dir / "training_summary.json"
    assert csv_path.exists()
    assert md_path.exists()
    assert json_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["variant"] for row in rows] == ["baseline", "hc", "mhc"]
    assert math.isclose(float(rows[0]["final_val_ppl"]), math.exp(2.0))
    assert rows[1]["ok"] == "True"
    assert "skipped 1 invalid metrics.jsonl line" in rows[1]["status"]
    assert rows[0]["eval_interval"] == "500"
    assert "baseline" in md_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [row["variant"] for row in payload["runs"]] == ["baseline", "hc", "mhc"]


def test_summarize_training_runs_missing_dir_is_row(tmp_path: Path):
    repo_dir = Path(__file__).resolve().parents[1]
    existing = tmp_path / "baseline"
    missing = tmp_path / "missing"
    _write_run(existing, variant="baseline", val_loss=2.0)
    out_dir = tmp_path / "reports"

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "examples" / "nanogpt" / "summarize_training_runs.py"),
            "--runs",
            f"baseline={existing}",
            f"hc={missing}",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(repo_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    with (out_dir / "training_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["variant"] for row in rows] == ["baseline", "hc"]
    assert rows[1]["ok"] == "False"
    assert rows[1]["status"] == "run directory missing"


def test_run_t4_full_compare_shell_syntax():
    repo_dir = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["bash", "-n", str(repo_dir / "examples" / "nanogpt" / "run_t4_full_compare.sh")],
        cwd=str(repo_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
