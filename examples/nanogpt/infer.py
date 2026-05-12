"""Minimal nanoGPT/mHC inference CLI.

This is intentionally simple: no KV cache, no serving framework, no training code
imports. It loads checkpoints written by examples/nanogpt/train.py.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any


NANOGPT_DIR = Path(__file__).resolve().parent
REPO_DIR = NANOGPT_DIR.parents[1]

DEFAULT_CONFIG: dict[str, Any] = {
    "block_size": 256,
    "vocab_size": 50304,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.0,
    "bias": False,
    "hc_num_streams": 1,
    "hc_num_fracs": 1,
    "hc_disable": True,
    "mhc": False,
    "sinkhorn_iters": 10,
    "sinkhorn_tau": 0.05,
    "mhc_h_res_proj": "sinkhorn",
    "ns_steps": 5,
    "ns_eps": 1e-7,
    "ns_coeffs": (3.0, -3.2, 1.2),
    "mhc_residual_identity_mix": False,
    "mhc_residual_alpha": 0.01,
    "v_residual": False,
    "v_residual_constrained": False,
    "v_residual_lamb_lr": 1e-2,
}

MODEL_CONFIG_KEYS = set(DEFAULT_CONFIG)


class CliError(RuntimeError):
    """User-facing CLI failure."""


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise CliError(
            "PyTorch is required. Install project dependencies, e.g. `pip install -e .`."
        ) from exc
    return torch


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    if base is not None:
        base_candidate = base / candidate
        if base_candidate.exists():
            return base_candidate
    return cwd_candidate


def load_config_values(config_path: str | Path | None) -> dict[str, Any]:
    values = dict(DEFAULT_CONFIG)
    if config_path is None:
        return values

    path = resolve_path(config_path, base=NANOGPT_DIR)
    if not path.exists():
        raise CliError(f"config file not found: {path}")

    namespace: dict[str, Any] = dict(values)
    namespace["__file__"] = str(path)
    namespace["__name__"] = "__nanogpt_config__"
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    except Exception as exc:
        raise CliError(f"failed to load config {path}: {exc}") from exc

    for key in MODEL_CONFIG_KEYS:
        if key in namespace:
            values[key] = namespace[key]
    return values


def load_checkpoint(ckpt_path: str | Path, torch, *, map_location: str = "cpu") -> dict[str, Any]:
    path = resolve_path(ckpt_path, base=NANOGPT_DIR)
    if not path.exists():
        raise CliError(f"checkpoint not found: {path}")
    try:
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)
    except Exception as exc:
        raise CliError(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise CliError(
            "incompatible checkpoint: expected dict with key 'model' from train.py"
        )
    if not isinstance(checkpoint["model"], dict):
        raise CliError("incompatible checkpoint: checkpoint['model'] is not a state dict")
    return checkpoint


def merged_config(config_path: str | Path | None, checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = load_config_values(config_path)
    ckpt_config = checkpoint.get("config")
    if ckpt_config is not None:
        if not isinstance(ckpt_config, dict):
            raise CliError("incompatible checkpoint: checkpoint['config'] is not a dict")
        for key, value in ckpt_config.items():
            if key in MODEL_CONFIG_KEYS:
                config[key] = value
    return config


def validate_runtime(torch, *, device: str, dtype: str):
    if device not in {"cpu", "cuda"} and not device.startswith("cuda:"):
        raise CliError(f"unsupported device {device!r}; use 'cpu' or 'cuda'")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CliError("CUDA requested but torch.cuda.is_available() is false")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise CliError(f"unsupported dtype {dtype!r}; use float32, float16, or bfloat16")

    if device == "cpu" and dtype != "float32":
        raise CliError("CPU inference supports dtype=float32 only in this CLI")
    if device.startswith("cuda") and dtype == "bfloat16":
        supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not supported:
            raise CliError(
                "bfloat16 requested but this CUDA device does not report BF16 support; "
                "use float16 on NVIDIA T4"
            )
    return torch.device(device), dtype_map[dtype]


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("_orig_mod.", "module."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


def build_model(config: dict[str, Any], torch, *, device, dtype):
    sys.path.insert(0, str(NANOGPT_DIR))
    try:
        from model import GPT, GPTConfig
    finally:
        try:
            sys.path.remove(str(NANOGPT_DIR))
        except ValueError:
            pass

    model_config = GPTConfig(**config)
    model = GPT(model_config)
    model.to(device=device)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    model.eval()
    return model


def load_model_from_checkpoint(
    *,
    ckpt_path: str | Path,
    config_path: str | Path | None,
    device: str,
    dtype: str,
    compile_model: bool,
):
    torch = import_torch()
    torch_device, torch_dtype = validate_runtime(torch, device=device, dtype=dtype)
    checkpoint = load_checkpoint(ckpt_path, torch)
    config = merged_config(config_path, checkpoint)
    model = build_model(
        config,
        torch,
        device=torch_device,
        dtype=torch_dtype,
    )
    state_dict = normalize_state_dict(checkpoint["model"])
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise CliError(f"checkpoint/model state_dict mismatch: {exc}") from exc
    if compile_model:
        if not hasattr(torch, "compile"):
            raise CliError("torch.compile requested but this PyTorch build lacks compile")
        model = torch.compile(model)
    model.eval()
    return torch, model, config, checkpoint, torch_device, torch_dtype


def parameter_count(model) -> int:
    raw_model = getattr(model, "_orig_mod", model)
    return sum(p.numel() for p in raw_model.parameters())


def autocast_context(torch, device, dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def cuda_synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(torch, device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def gpt2_tokenizer():
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise CliError(
            "tiktoken is required for text prompts. Install dependencies with "
            "`pip install -e .`."
        ) from exc
    return tiktoken.get_encoding("gpt2")


def sample_next_token(logits, torch, *, temperature: float, top_k: int | None):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < values[:, [-1]], -torch.inf, logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_tokens(
    model,
    idx,
    torch,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    block_size: int,
    device,
    dtype,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        with autocast_context(torch, device, dtype):
            logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        idx_next = sample_next_token(
            next_token_logits, torch, temperature=temperature, top_k=top_k
        )
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def format_memory(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} MB"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run nanoGPT/mHC text inference.")
    parser.add_argument("--ckpt", required=True, help="Path to train.py checkpoint")
    parser.add_argument("--config", required=True, help="Path to nanoGPT config")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "float32", "bfloat16"),
        help="Inference dtype",
    )
    parser.add_argument("--compile", type=parse_bool, default=False)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.max_new_tokens < 0:
            raise CliError("--max-new-tokens must be >= 0")
        if args.temperature < 0:
            raise CliError("--temperature must be >= 0")
        top_k = args.top_k if args.top_k > 0 else None

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

        enc = gpt2_tokenizer()
        prompt_tokens = enc.encode(args.prompt)
        if not prompt_tokens:
            raise CliError("prompt encoded to zero tokens")
        if len(prompt_tokens) > int(config["block_size"]):
            raise CliError(
                f"prompt length {len(prompt_tokens)} exceeds model block_size "
                f"{config['block_size']}"
            )

        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        reset_peak_memory(torch, device)
        cuda_synchronize(torch, device)
        start = time.perf_counter()
        with torch.no_grad():
            out = generate_tokens(
                model,
                idx,
                torch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                block_size=int(config["block_size"]),
                device=device,
                dtype=dtype,
            )
        cuda_synchronize(torch, device)
        elapsed = time.perf_counter() - start
        generated_tokens = out.shape[1] - len(prompt_tokens)
        tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else math.inf
        text = enc.decode(out[0].tolist())

        print("Generated text:")
        print(text)
        print("")
        print("Metrics:")
        print(f"device: {device}")
        print(f"dtype: {args.dtype}")
        print(f"parameter_count: {parameter_count(model)}")
        print(f"prompt_length: {len(prompt_tokens)}")
        print(f"generated_tokens: {generated_tokens}")
        print(f"elapsed_seconds: {elapsed:.6f}")
        print(f"tokens_per_sec: {tokens_per_sec:.3f}")
        print(f"peak_cuda_memory: {format_memory(peak_memory_mb(torch, device))}")
        return 0
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
