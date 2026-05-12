import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _write_tiny_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "block_size = 16",
                "n_layer = 1",
                "n_head = 1",
                "n_embd = 16",
                "dropout = 0.0",
                "bias = False",
                "hc_num_streams = 1",
                "hc_num_fracs = 1",
                "hc_disable = True",
                "mhc = False",
                "v_residual = False",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_tiny_checkpoint(repo_dir: Path, path: Path, config_path: Path) -> None:
    nanogpt_dir = repo_dir / "examples" / "nanogpt"
    sys.path.insert(0, str(nanogpt_dir))
    try:
        from infer import load_config_values
        from model import GPT, GPTConfig

        config = load_config_values(config_path)
        model_config = GPTConfig(**config)
        model = GPT(model_config)
        torch.save(
            {
                "model": model.state_dict(),
                "config": model_config.__dict__,
                "iter_num": 0,
                "best_val_loss": 0.0,
            },
            path,
        )
    finally:
        sys.path.remove(str(nanogpt_dir))


def test_benchmark_cli_writes_machine_readable_outputs(tmp_path: Path):
    repo_dir = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "tiny_config.py"
    ckpt_path = tmp_path / "tiny_ckpt.pt"
    json_path = tmp_path / "bench.json"
    csv_path = tmp_path / "bench.csv"

    _write_tiny_config(config_path)
    _write_tiny_checkpoint(repo_dir, ckpt_path, config_path)

    cmd = [
        sys.executable,
        str(repo_dir / "examples" / "nanogpt" / "benchmark_inference.py"),
        "--ckpt",
        str(ckpt_path),
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--batch-size",
        "1",
        "--prompt-len",
        "4",
        "--gen-len",
        "2",
        "--num-warmup",
        "0",
        "--num-iters",
        "1",
        "--compile",
        "false",
        "--output-json",
        str(json_path),
        "--output-csv",
        str(csv_path),
        "--seed",
        "123",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["device"] == "cpu"
    assert payload["metadata"]["dtype"] == "float32"
    assert payload["metadata"]["settings"]["gen_len"] == 2
    assert "prefill_latency_ms" in payload["metrics"]
    assert "decode_latency_per_token_ms" in payload["metrics"]
    assert "end_to_end_latency_ms" in payload["metrics"]
    assert "tokens_per_sec" in payload["metrics"]

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {row["metric"] for row in rows} >= {
        "prefill_latency_ms",
        "decode_latency_per_token_ms",
        "end_to_end_latency_ms",
        "tokens_per_sec",
    }


def test_infer_cli_generates_text_with_tiktoken(tmp_path: Path):
    pytest.importorskip("tiktoken")

    repo_dir = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "tiny_config.py"
    ckpt_path = tmp_path / "tiny_ckpt.pt"

    _write_tiny_config(config_path)
    _write_tiny_checkpoint(repo_dir, ckpt_path, config_path)

    cmd = [
        sys.executable,
        str(repo_dir / "examples" / "nanogpt" / "infer.py"),
        "--ckpt",
        str(ckpt_path),
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--compile",
        "false",
        "--prompt",
        "hello",
        "--max-new-tokens",
        "2",
        "--temperature",
        "1.0",
        "--top-k",
        "10",
        "--seed",
        "123",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "Generated text:" in proc.stdout
    assert "device: cpu" in proc.stdout
    assert "dtype: float32" in proc.stdout
    assert "generated_tokens: 2" in proc.stdout
