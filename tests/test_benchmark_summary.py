import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_benchmark(path: Path, variant: str) -> None:
    payload = {
        "metadata": {
            "checkpoint_path": f"ckpts/{variant}.pt",
            "config_path": f"config/{variant}.py",
            "model_parameter_count": 1234,
            "dtype": "float16",
            "device": "cuda",
            "gpu_name": "NVIDIA T4",
        },
        "metrics": {
            "prefill_latency_ms": {"mean": 1.0},
            "full_context_generation_latency_ms": {"mean": 10.0},
            "full_context_generation_latency_per_token_ms": {"mean": 0.5},
            "tokens_per_sec": {"mean": 64.0},
            "peak_vram_mb": {"mean": 100.0, "max": 120.0},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_benchmarks_writes_csv_and_markdown(tmp_path: Path):
    repo_dir = Path(__file__).resolve().parents[1]
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    _write_benchmark(bench_dir / "baseline.json", "baseline")
    _write_benchmark(bench_dir / "mhc.json", "mhc")
    (bench_dir / "bad.json").write_text("{bad", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "examples" / "nanogpt" / "summarize_benchmarks.py"),
            str(bench_dir),
        ],
        cwd=str(repo_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "skipped invalid JSON: bad.json" in proc.stdout
    csv_path = bench_dir / "summary.csv"
    md_path = bench_dir / "summary.md"
    assert csv_path.exists()
    assert md_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["variant"] for row in rows] == ["baseline", "mhc"]
    assert rows[0]["prefill_latency_ms_mean"] == "1.0"
    assert "full_context_generation_latency_ms_mean" in rows[0]
    assert "baseline" in md_path.read_text(encoding="utf-8")
