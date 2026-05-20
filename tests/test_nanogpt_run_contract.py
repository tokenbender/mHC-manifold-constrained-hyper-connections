import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


FINEWEB_MAGIC = 20240520
FINEWEB_VERSION = 1
FINEWEB_HEADER_SIZE = 256


def _write_fineweb_shard(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert tokens.dtype == np.uint16
    header = np.zeros((FINEWEB_HEADER_SIZE,), dtype=np.int32)
    header[0] = FINEWEB_MAGIC
    header[1] = FINEWEB_VERSION
    header[2] = int(tokens.size)

    with path.open("wb") as f:
        header.tofile(f)
        tokens.tofile(f)


def test_nanogpt_run_contract_smoke(tmp_path: Path):
    repo_dir = Path(__file__).resolve().parents[1]
    nanogpt_dir = repo_dir / "examples" / "nanogpt"

    data_dir = tmp_path / "fineweb10B"
    train_path = data_dir / "fineweb_train_000000.bin"
    val_path = data_dir / "fineweb_val_000000.bin"

    # Ensure there is enough data for block_size + 1.
    rng = np.random.default_rng(0)
    _write_fineweb_shard(train_path, rng.integers(0, 1000, size=4096, dtype=np.uint16))
    _write_fineweb_shard(val_path, rng.integers(0, 1000, size=2048, dtype=np.uint16))

    out_dir = tmp_path / "out" / "run-contract-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = out_dir / "stdout.log"

    cmd = [
        sys.executable,
        "train.py",
        "config/train_fineweb10B.py",
        f"out_dir={out_dir}",
        f"data_dir={data_dir}",
        "wandb_log=False",
        "compile_model=False",
        "dtype='float32'",
        "device='cpu'",
        "max_iters=0",
        "eval_interval=1",
        "eval_iters=1",
        "log_interval=1",
        "batch_size=2",
        "block_size=32",
        "n_layer=1",
        "n_head=1",
        "n_embd=32",
        "dropout=0.0",
        "gradient_accumulation_steps=1",
        "hc_disable=True",
        "mhc=False",
        "v_residual=False",
    ]

    env = os.environ.copy()
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    env.pop("WORLD_SIZE", None)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    with stdout_log.open("wb") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(nanogpt_dir),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )

    assert proc.returncode == 0, stdout_log.read_text(errors="replace")
    assert stdout_log.exists() and stdout_log.stat().st_size > 0

    required = [
        "command.sh",
        "run_metadata.json",
        "config_effective.json",
        "dataset_manifest.json",
        "metrics.jsonl",
        "ckpt.pt",
        "summary.json",
    ]
    for name in required:
        assert (out_dir / name).exists(), name

    # command.sh
    command_sh = (out_dir / "command.sh")
    assert os.access(command_sh, os.X_OK)
    command_text = command_sh.read_text(encoding="utf-8")
    assert "train.py" in command_text

    # run_metadata.json
    run_metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["run_id"] == out_dir.name
    assert run_metadata["ddp"] is False
    assert run_metadata["world_size"] == 1
    assert run_metadata["device_type"] == "cpu"
    assert run_metadata["wandb"]["enabled"] is False

    # config_effective.json
    cfg = json.loads((out_dir / "config_effective.json").read_text(encoding="utf-8"))
    assert cfg["out_dir"] == str(out_dir)
    assert cfg["dataset"] == "fineweb10B"
    assert cfg["compile_model"] is False
    assert cfg["wandb_log"] is False
    assert cfg["mhc"] is False
    assert cfg["data_loader"] == "memmap"

    # dataset_manifest.json
    manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "fineweb10B"
    assert len(manifest["train"]) == 1
    assert len(manifest["val"]) == 1
    assert Path(manifest["train"][0]["path"]).resolve() == train_path.resolve()
    assert Path(manifest["val"][0]["path"]).resolve() == val_path.resolve()

    # summary.json health checks
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["run_id"] == out_dir.name
    assert summary["ddp"] is False
    assert summary["world_size"] == 1
    assert summary["end_time"] >= summary["start_time"]
    assert summary["elapsed_s"] >= 0
    assert summary["last_eval"] is not None
    assert "train" in summary["last_eval"] and "val" in summary["last_eval"]
    assert math.isfinite(float(summary["last_eval"]["train"]))
    assert math.isfinite(float(summary["last_eval"]["val"]))
    assert "error" not in summary
