import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


TINY_CONFIGS = {
    "baseline": "train_fineweb10B_tiny_t4.py",
    "hc": "train_fineweb10B_hc_tiny_t4.py",
    "mhc": "train_fineweb10B_mhc_tiny_t4.py",
}


def _load_config(path: Path) -> dict:
    values: dict = {}
    exec(path.read_text(encoding="utf-8"), values)
    return values


def test_tiny_config_files_exist_and_instantiate_gpt():
    torch = pytest.importorskip("torch")
    repo_dir = Path(__file__).resolve().parents[1]
    nanogpt_dir = repo_dir / "examples" / "nanogpt"
    sys.path.insert(0, str(nanogpt_dir))
    try:
        from model import GPT, GPTConfig
    finally:
        sys.path.pop(0)

    for filename in TINY_CONFIGS.values():
        cfg = _load_config(nanogpt_dir / "config" / filename)
        model = GPT(
            GPTConfig(
                block_size=cfg["block_size"],
                vocab_size=50304,
                n_layer=cfg["n_layer"],
                n_head=cfg["n_head"],
                n_embd=cfg["n_embd"],
                dropout=cfg["dropout"],
                bias=cfg["bias"],
                hc_num_streams=cfg["hc_num_streams"],
                hc_num_fracs=cfg["hc_num_fracs"],
                hc_disable=cfg["hc_disable"],
                mhc=cfg["mhc"],
                sinkhorn_iters=cfg.get("sinkhorn_iters", 10),
                sinkhorn_tau=cfg.get("sinkhorn_tau", 0.05),
                mhc_h_res_proj=cfg.get("mhc_h_res_proj", "sinkhorn"),
                ns_steps=cfg.get("ns_steps", 5),
                ns_eps=cfg.get("ns_eps", 1e-7),
                ns_coeffs=cfg.get("ns_coeffs", (3.0, -3.2, 1.2)),
                mhc_residual_identity_mix=cfg.get("mhc_residual_identity_mix", False),
                mhc_residual_alpha=cfg.get("mhc_residual_alpha", 0.01),
            )
        )
        x = torch.randint(0, 100, (1, 8), dtype=torch.long)
        logits, _ = model(x)
        assert logits.shape[:2] == (1, 8)


def test_tiny_variants_share_architecture_and_training_budget():
    repo_dir = Path(__file__).resolve().parents[1]
    config_dir = repo_dir / "examples" / "nanogpt" / "config"
    configs = {name: _load_config(config_dir / file) for name, file in TINY_CONFIGS.items()}

    shared_keys = [
        "dataset",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "dropout",
        "bias",
        "batch_size",
        "gradient_accumulation_steps",
        "max_iters",
        "eval_interval",
        "eval_iters",
        "learning_rate",
        "weight_decay",
        "dtype",
        "compile_model",
        "wandb_log",
        "data_loader",
    ]
    baseline = configs["baseline"]
    for cfg in configs.values():
        for key in shared_keys:
            assert cfg[key] == baseline[key], key

    assert configs["baseline"]["hc_disable"] is True
    assert configs["baseline"]["mhc"] is False
    assert configs["hc"]["hc_disable"] is False
    assert configs["hc"]["mhc"] is False
    assert configs["mhc"]["hc_disable"] is False
    assert configs["mhc"]["mhc"] is True


def test_tiny_shell_scripts_have_valid_syntax():
    repo_dir = Path(__file__).resolve().parents[1]
    scripts = [
        "run_t4_tiny_compare.sh",
        "run_t4_tiny_full_compare.sh",
        "run_t4_mini_compare.sh",
        "run_t4_mini_full_compare.sh",
    ]
    for script in scripts:
        proc = subprocess.run(
            ["bash", "-n", str(repo_dir / "examples" / "nanogpt" / script)],
            cwd=str(repo_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout


def _write_metrics(path: Path, train_losses: list[float], val_losses: list[float]) -> None:
    path.mkdir(parents=True)
    lines = []
    for idx, loss in enumerate(train_losses):
        lines.append(
            json.dumps(
                {
                    "event": "train",
                    "iter": idx,
                    "train_loss": loss,
                    "tokens_per_sec": 1000 + idx,
                    "peak_vram_mb": 512 + idx,
                }
            )
        )
    for idx, loss in enumerate(val_losses):
        lines.append(
            json.dumps(
                {
                    "event": "eval",
                    "iter": idx * 2,
                    "val_loss": loss,
                    "best_val_loss": min(val_losses[: idx + 1]),
                }
            )
        )
    (path / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path / "summary.json").write_text(
        json.dumps({"best_val_loss": min(val_losses), "last_eval": {"val": val_losses[-1]}}),
        encoding="utf-8",
    )


def test_plot_training_comparison_writes_pngs(tmp_path: Path):
    pytest.importorskip("matplotlib")
    repo_dir = Path(__file__).resolve().parents[1]
    baseline = tmp_path / "baseline"
    hc = tmp_path / "hc"
    mhc = tmp_path / "mhc"
    _write_metrics(baseline, [3.0, 2.9], [3.1, 2.8])
    _write_metrics(hc, [3.2, 3.0], [3.3, 2.9])
    _write_metrics(mhc, [3.1, 2.7], [3.0, 2.6])

    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    with (bench_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "tokens_per_sec_mean",
                "full_context_generation_latency_per_token_ms_mean",
                "peak_vram_mb_max",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "baseline",
                "tokens_per_sec_mean": 100,
                "full_context_generation_latency_per_token_ms_mean": 2,
                "peak_vram_mb_max": 500,
            }
        )

    out_dir = tmp_path / "figures"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "examples" / "nanogpt" / "plot_training_comparison.py"),
            "--runs",
            f"baseline={baseline}",
            f"hc={hc}",
            f"mhc={mhc}",
            "--benchmark-summary",
            str(bench_dir / "summary.csv"),
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
    expected = [
        "train_loss_curve.png",
        "val_loss_curve.png",
        "best_val_loss_bar.png",
        "final_val_ppl_bar.png",
        "tokens_per_sec_curve.png",
        "training_peak_vram_curve.png",
        "peak_vram_curve.png",
        "inference_tokens_per_sec_bar.png",
        "decode_ms_per_token_bar.png",
        "benchmark_peak_vram_bar.png",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), name
        assert path.stat().st_size > 0


def test_colab_notebook_is_mini_first():
    repo_dir = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo_dir / "notebooks" / "mhc_colab_t4_train_compare.json").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "Mini Colab T4 Training and Benchmark" in text
    assert 'EXPERIMENT_PRESET = "mini"' in text
    assert 'RUN_NAME = f"t4-{EXPERIMENT_PRESET}-fineweb{DATA_SHARDS}shards-{MAX_ITERS}iters"' in text
    assert '"out_dir": "out-t4-mini-baseline"' in text
    assert '"out_dir": "out-t4-mini-hc"' in text
    assert '"out_dir": "out-t4-mini-mhc"' in text
    assert "Smoke test — not for reporting" in text
    assert "Optional Full T4 Experiment — not recommended for Colab Free" in text
    assert "not full paper reproduction" in text
