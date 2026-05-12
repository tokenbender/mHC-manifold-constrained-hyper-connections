"""
Train nanoGPT with HyperConnections.

Usage:
    python train.py config/train_fineweb10B.py
    torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B.py
"""

import glob
import json
import math
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from hyper_connections import HyperConnections
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# default config values (can be overridden by config file)

out_dir = "out"
eval_interval = 200
log_interval = 10
eval_iters = 200
max_iters = 2000

batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 200
lr_decay_iters = 2000
min_lr = 6e-5

gradient_accumulation_steps = 1

seed = 1337

# dataset: "fineweb10B"
dataset = "fineweb10B"

# Optional override (useful for tests and external mounts). If not set, defaults
# to `examples/nanogpt/data/<dataset>`.
data_dir = None

NS_COEFFS = (
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
)

NS_STEPS = len(NS_COEFFS)

# hyper-connections config
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True
mhc = False
sinkhorn_iters = 10
sinkhorn_tau = 0.05
mhc_h_res_proj = "sinkhorn"
ns_steps = 5
ns_eps = 1e-7
ns_coeffs = (3.0, -3.2, 1.2)
mhc_residual_identity_mix = False
mhc_residual_alpha = 0.01

# value residual config
v_residual = False
v_residual_constrained = False
v_residual_lamb_lr = 1e-2

# dtype: "float32", "bfloat16", "float16"
dtype = "bfloat16"

# Optional override. If unset, will auto-select (cuda > mps > cpu).
device = None

# torch.compile (requires PyTorch 2.0+)
compile_model = False

# wandb logging
wandb_log = True
wandb_project = "mhc-nanogpt"
wandb_run_name = "baseline"
wandb_group = None
wandb_log_layer_stats = True
wandb_log_layer_cosine = True

# DDP backend: "nccl", "gloo", etc.
# If NCCL fails, set NCCL_IB_DISABLE=1 or use backend="gloo"
backend = "nccl"

# -----------------------------------------------------------------------------
# load config file if provided
exec(open(os.path.join(os.path.dirname(__file__), "configurator.py")).read())


def get_wandb_variant():
    if v_residual:
        return "vres"
    if mhc:
        return "mhc"
    if not hc_disable:
        return "hc"
    return "baseline"


wandb_variant = get_wandb_variant()
if wandb_group is None:
    wandb_group = f"{dataset}-L{n_layer}-D{n_embd}-H{n_head}"
if wandb_run_name == "baseline":
    wandb_run_name = f"{dataset}-{wandb_variant}-L{n_layer}-D{n_embd}-H{n_head}-s{seed}"
wandb_job_type = wandb_variant
wandb_tags = [
    dataset,
    wandb_variant,
    f"L{n_layer}",
    f"D{n_embd}",
    f"H{n_head}",
    f"streams={hc_num_streams}",
    f"fracs={hc_num_fracs}",
    f"block={block_size}",
    f"dtype={dtype}",
    f"lr={learning_rate:g}",
    f"wd={weight_decay:g}",
    f"seed={seed}",
]

if mhc:
    wandb_tags.extend(
        [
            f"sinkhorn_iters={sinkhorn_iters}",
            f"sinkhorn_tau={sinkhorn_tau:g}",
            f"mhc_res_proj={mhc_h_res_proj}",
            f"ns_steps={ns_steps}",
        ]
    )

if v_residual:
    wandb_tags.append("v_residual")
    if v_residual_constrained:
        wandb_tags.append("v_residual_constrained")

# -----------------------------------------------------------------------------
# DDP setup

ddp = int(os.environ.get("RANK", -1)) != -1

# Keep the user-provided value for reproducibility. In DDP we divide
# `gradient_accumulation_steps` by world size (below).
gradient_accumulation_steps_total = gradient_accumulation_steps

if ddp:
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend=backend, device_id=device)
    dist.barrier()
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = (
    device.type
    if isinstance(device, torch.device)
    else (
        "cuda"
        if "cuda" in device
        else ("mps" if "mps" in device else "cpu")
    )
)

# -----------------------------------------------------------------------------
# Minimal run artifact contract

RUN_CONTRACT_VERSION = 1


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    return str(value)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _try_git_info(repo_dir: Path) -> Optional[dict[str, str]]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "--quiet"],
            text=True,
            check=False,
        ).returncode
        return {"commit": commit, "branch": branch, "dirty": "true" if dirty != 0 else "false"}
    except Exception:
        return None


def _write_run_contract_init(*, out_dir_path: Path, repo_dir: Path, run_id: str) -> None:
    # command.sh should be reproducible and must not contain secrets.
    if ddp:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ddp_world_size}",
            "train.py",
            *sys.argv[1:],
        ]
    else:
        cmd = [sys.executable, "train.py", *sys.argv[1:]]

    command_sh = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(Path(__file__).resolve().parent))}",
            shlex.join(cmd),
            "",
        ]
    )
    _atomic_write_text(out_dir_path / "command.sh", command_sh)
    os.chmod(out_dir_path / "command.sh", 0o755)

    git_info = _try_git_info(repo_dir)
    payload = {
        "contract_version": RUN_CONTRACT_VERSION,
        "run_id": run_id,
        "run_kind": "nanogpt_train",
        "out_dir": str(out_dir_path),
        "argv": list(sys.argv),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ddp": ddp,
        "rank": ddp_rank if ddp else 0,
        "world_size": ddp_world_size,
        "device": str(device),
        "device_type": device_type,
        "wandb": {
            "enabled": bool(wandb_log),
            "project": wandb_project,
            "group": wandb_group,
            "run_name": wandb_run_name,
            "job_type": wandb_job_type,
            "variant": wandb_variant,
        },
        "git": git_info,
        "ts": int(time.time()),
    }
    _atomic_write_json(out_dir_path / "run_metadata.json", payload)


def _write_config_effective(*, out_dir_path: Path) -> None:
    keys = [
        # output + run
        "out_dir",
        "dataset",
        "seed",
        "dtype",
        "compile_model",
        # model
        "n_layer",
        "n_head",
        "n_embd",
        "dropout",
        "bias",
        # optimization
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "warmup_iters",
        "lr_decay_iters",
        "min_lr",
        "max_iters",
        "batch_size",
        "block_size",
        # hyper-connections
        "hc_num_streams",
        "hc_num_fracs",
        "hc_disable",
        "mhc",
        "sinkhorn_iters",
        "sinkhorn_tau",
        "mhc_h_res_proj",
        "ns_steps",
        "ns_eps",
        "ns_coeffs",
        "mhc_residual_identity_mix",
        "mhc_residual_alpha",
        # v-residual
        "v_residual",
        "v_residual_constrained",
        "v_residual_lamb_lr",
        # logging
        "wandb_log",
        "wandb_project",
        "wandb_run_name",
        "wandb_group",
        "wandb_job_type",
        "wandb_tags",
        "wandb_log_layer_stats",
        "wandb_log_layer_cosine",
        # DDP
        "backend",
    ]
    effective = {k: _json_safe(globals().get(k)) for k in keys}
    effective.update(
        {
            "contract_version": RUN_CONTRACT_VERSION,
            "wandb_variant": wandb_variant,
            "ddp": ddp,
            "world_size": ddp_world_size,
            "gradient_accumulation_steps_total": gradient_accumulation_steps_total,
            "gradient_accumulation_steps_per_rank": gradient_accumulation_steps,
        }
    )
    _atomic_write_json(out_dir_path / "config_effective.json", effective)


def _write_dataset_manifest(*, out_dir_path: Path, dataset_name: str, data_dir_path: Path, train_files, val_files) -> None:
    def file_info(path: str) -> dict[str, object]:
        p = Path(path)
        stat = p.stat()
        return {
            "path": str(p),
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }

    payload = {
        "contract_version": RUN_CONTRACT_VERSION,
        "dataset": dataset_name,
        "data_dir": str(data_dir_path),
        "train": [file_info(p) for p in train_files],
        "val": [file_info(p) for p in val_files],
    }
    _atomic_write_json(out_dir_path / "dataset_manifest.json", payload)


def _append_metrics_jsonl(*, out_dir_path: Path, payload: dict) -> None:
    path = out_dir_path / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")


def _peak_vram_mb() -> Optional[float]:
    if device_type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)


out_dir_path = Path(out_dir)
run_id = out_dir_path.resolve().name
repo_dir = Path(__file__).resolve().parents[2]

if master_process:
    out_dir_path.mkdir(parents=True, exist_ok=True)
    _write_run_contract_init(out_dir_path=out_dir_path, repo_dir=repo_dir, run_id=run_id)
    _write_config_effective(out_dir_path=out_dir_path)

if ddp:
    dist.barrier()

# -----------------------------------------------------------------------------
# AMP setup

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

if device_type == "cpu":
    ctx = nullcontext()
    scaler = None
else:
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    # GradScaler only needed for float16 (not bf16)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))

# -----------------------------------------------------------------------------
# Data loading

if data_dir is None:
    data_dir = os.path.join(os.path.dirname(__file__), "data", dataset)

if dataset != "fineweb10B":
    raise ValueError(f"unknown dataset: {dataset}")

# FineWeb10B: pretokenized GPT-2 shards
# Format: 256 x int32 header, then uint16 tokens
# Header: [0]=magic(20240520), [1]=version(1), [2]=num_tokens

FINEWEB_MAGIC = 20240520
FINEWEB_VERSION = 1
HEADER_SIZE = 256  # int32 count


def load_fineweb_shard(path):
    """Load a FineWeb shard, validate header, return tokens as int64 tensor."""
    header = torch.from_file(
        str(path), shared=False, size=HEADER_SIZE, dtype=torch.int32
    )
    assert header[0].item() == FINEWEB_MAGIC, f"bad magic in {path}"
    assert header[1].item() == FINEWEB_VERSION, f"bad version in {path}"
    num_tokens = int(header[2].item())

    # read tokens (uint16 -> convert to int64 for embedding lookup)
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE * 4)  # skip header (256 * 4 bytes)
        buf = np.frombuffer(f.read(num_tokens * 2), dtype=np.uint16)
        tokens = torch.from_numpy(buf.astype(np.int64))

    return tokens

# find shards
train_shards = sorted(glob.glob(os.path.join(data_dir, "fineweb_train_*.bin")))
val_shards = sorted(glob.glob(os.path.join(data_dir, "fineweb_val_*.bin")))

assert len(train_shards) > 0, f"no train shards found in {data_dir}"
assert len(val_shards) > 0, f"no val shards found in {data_dir}"

if master_process:
    print(f"Found {len(train_shards)} train shards, {len(val_shards)} val shards")
    _write_dataset_manifest(
        out_dir_path=out_dir_path,
        dataset_name=dataset,
        data_dir_path=Path(data_dir),
        train_files=train_shards,
        val_files=val_shards,
    )

# load all shards into memory (for simplicity; ~200MB per shard)
# for large-scale, would stream shards instead
train_data = torch.cat([load_fineweb_shard(s) for s in train_shards])
val_data = torch.cat([load_fineweb_shard(s) for s in val_shards])

if master_process:
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

vocab_size = 50304  # GPT-2 vocab size rounded up for efficiency

# -----------------------------------------------------------------------------
# Batch sampling (simple random contiguous windows)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


# -----------------------------------------------------------------------------
# Model setup

model_config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    hc_num_streams=hc_num_streams,
    hc_num_fracs=hc_num_fracs,
    hc_disable=hc_disable,
    mhc=mhc,
    sinkhorn_iters=sinkhorn_iters,
    sinkhorn_tau=sinkhorn_tau,
    mhc_h_res_proj=mhc_h_res_proj,
    ns_steps=ns_steps,
    ns_eps=ns_eps,
    ns_coeffs=ns_coeffs,
    mhc_residual_identity_mix=mhc_residual_identity_mix,
    mhc_residual_alpha=mhc_residual_alpha,
    v_residual=v_residual,
    v_residual_constrained=v_residual_constrained,
    v_residual_lamb_lr=v_residual_lamb_lr,
)

model = GPT(model_config)
model.to(device)

if compile_model:
    print("Compiling model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=mhc)

raw_model = model.module if ddp else model

if wandb_log and wandb_log_layer_stats:
    for block in raw_model.transformer.h:
        for hc in (block.hc_attn, block.hc_mlp):
            if isinstance(hc, HyperConnections):
                hc.collect_stats = True

optimizer = raw_model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device_type,
)

# -----------------------------------------------------------------------------
# Learning rate schedule


def get_lr(it):
    # linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def collect_hc_layer_stats():
    layer_count = len(raw_model.transformer.h) * 2
    layer_stats = {}
    for block_idx, block in enumerate(raw_model.transformer.h):
        for sub_idx, hc in enumerate((block.hc_attn, block.hc_mlp)):
            if not hasattr(hc, "last_stats"):
                continue
            layer_index = block_idx * 2 + sub_idx
            for key, value in hc.last_stats.items():
                layer_stats.setdefault(key, [None] * layer_count)
                layer_stats[key][layer_index] = value.item()
    return layer_stats


def build_layer_table(layer_stats):
    if not layer_stats:
        return None
    keys = sorted(layer_stats.keys())
    layer_count = max(len(v) for v in layer_stats.values())
    table = wandb.Table(columns=["layer"] + keys)
    for i in range(layer_count):
        row_vals = []
        for key in keys:
            values = layer_stats[key]
            val = values[i] if i < len(values) else None
            row_vals.append(val)
        if all(v is None for v in row_vals):
            continue
        table.add_data(i, *row_vals)
    return table


def forward_with_layer_cosine(x, y):
    sims = []
    prev = [None]
    handles = []

    def hook(_, __, output):
        out = output.detach()
        if prev[0] is not None:
            prev_flat = prev[0].reshape(-1, prev[0].shape[-1])
            out_flat = out.reshape(-1, out.shape[-1])
            sim = F.cosine_similarity(prev_flat, out_flat, dim=-1).mean()
            sims.append(sim)
        prev[0] = out

    for block in raw_model.transformer.h:
        handles.append(block.register_forward_hook(hook))

    with ctx:
        _, loss = model(x, y)

    for handle in handles:
        handle.remove()

    sims = [s.item() for s in sims]
    return loss, sims


# -----------------------------------------------------------------------------
# Evaluation


@torch.no_grad()
def estimate_loss():
    out = {}
    layer_cosine = None
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            if (
                layer_cosine is None
                and wandb_log
                and wandb_log_layer_cosine
                and split == "train"
                and k == 0
            ):
                loss, layer_cosine = forward_with_layer_cosine(x, y)
            else:
                with ctx:
                    _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out, layer_cosine


# -----------------------------------------------------------------------------
# Training loop

iter_num = 0
best_val_loss = 1e9
last_eval_losses = None
last_eval_iter = None

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"Training on {device}, dtype={dtype}, DDP={ddp}")
    print(f"  tokens per iteration: {tokens_per_iter:,}")
    if ddp:
        print(
            f"  world_size={ddp_world_size}, grad_accum_steps={gradient_accumulation_steps}"
        )
    print(f"  model params: {sum(p.numel() for p in raw_model.parameters()):,}")
    print()

if wandb_log and master_process:
    import wandb as wandb_lib

    wandb = wandb_lib
else:
    wandb = None

start_time = time.time()
run_success = False
run_error = None

try:
    if wandb is not None:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            job_type=wandb_job_type,
            tags=wandb_tags,
            config={
                "dataset": dataset,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "batch_size": batch_size,
                "block_size": block_size,
                "learning_rate": learning_rate,
                "max_iters": max_iters,
                "hc_num_streams": hc_num_streams,
                "hc_num_fracs": hc_num_fracs,
                "hc_disable": hc_disable,
                "mhc": mhc,
                "sinkhorn_iters": sinkhorn_iters,
                "sinkhorn_tau": sinkhorn_tau,
                "mhc_h_res_proj": mhc_h_res_proj,
                "ns_steps": ns_steps,
                "ns_eps": ns_eps,
                "ns_coeffs": ns_coeffs,
                "mhc_residual_identity_mix": mhc_residual_identity_mix,
                "mhc_residual_alpha": mhc_residual_alpha,
                "v_residual": v_residual,
                "v_residual_constrained": v_residual_constrained,
                "v_residual_lamb_lr": v_residual_lamb_lr,
                "dtype": dtype,
                "world_size": ddp_world_size,
                "tokens_per_iter": tokens_per_iter,
                "wandb_log_layer_stats": wandb_log_layer_stats,
                "wandb_log_layer_cosine": wandb_log_layer_cosine,
            },
        )

    while iter_num <= max_iters:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            lr_scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = lr * lr_scale

        # evaluation
        if iter_num % eval_interval == 0 and master_process:
            losses, layer_cosine = estimate_loss()
            last_eval_losses = losses
            last_eval_iter = iter_num
            print(
                f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if wandb is not None:
                eval_log = {
                    "val/loss": losses["val"],
                    "train/loss_eval": losses["train"],
                    "perf/elapsed_s": time.time() - start_time,
                    "tokens/seen": iter_num * tokens_per_iter,
                }
                wandb.log(eval_log, step=iter_num)
                if wandb_log_layer_cosine and layer_cosine is not None:
                    layer_table = wandb.Table(columns=["layer", "cosine"])
                    for idx, value in enumerate(layer_cosine):
                        layer_table.add_data(idx, value)
                    wandb.log({"hc/layer_cosine": layer_table}, step=iter_num)
                if wandb_log_layer_stats:
                    layer_stats = collect_hc_layer_stats()
                    layer_stats_table = build_layer_table(layer_stats)
                    if layer_stats_table is not None:
                        wandb.log({"hc/layer_stats": layer_stats_table}, step=iter_num)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                os.makedirs(out_dir, exist_ok=True)
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": model_config.__dict__,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            _append_metrics_jsonl(
                out_dir_path=out_dir_path,
                payload={
                    "event": "eval",
                    "iter": iter_num,
                    "train_loss_eval": losses["train"],
                    "val_loss": losses["val"],
                    "best_val_loss": best_val_loss,
                    "lr": lr,
                    "timestamp": time.time(),
                    "peak_vram_mb": _peak_vram_mb(),
                },
            )

        t0 = time.time()

        # training step with gradient accumulation
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # only sync gradients on the last micro step
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )

            x, y = get_batch("train")

            with ctx:
                _, loss = model(x, y)
                loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # gradient clipping
        grad_norm = None
        if grad_clip != 0.0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), grad_clip
            )

        # optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        dt = time.time() - t0
        tokens_per_sec = tokens_per_iter / dt

        if iter_num % log_interval == 0 and master_process:
            loss_item = loss.item() * gradient_accumulation_steps
            grad_norm_item = grad_norm.item() if grad_norm is not None else None
            print(
                f"iter {iter_num}: loss {loss_item:.4f}, lr {lr:.2e}, "
                f"time {dt * 1000:.0f}ms, tok/s {tokens_per_sec:.0f}"
            )
            _append_metrics_jsonl(
                out_dir_path=out_dir_path,
                payload={
                    "event": "train",
                    "iter": iter_num,
                    "train_loss": loss_item,
                    "lr": lr,
                    "iteration_time_ms": dt * 1000,
                    "tokens_per_sec": tokens_per_sec,
                    "grad_norm": grad_norm_item,
                    "timestamp": time.time(),
                    "peak_vram_mb": _peak_vram_mb(),
                },
            )
            if wandb is not None:
                log_dict = {
                    "train/loss": loss_item,
                    "train/lr": lr,
                    "perf/tok_per_sec": tokens_per_sec,
                    "perf/iter_time_ms": dt * 1000,
                    "perf/elapsed_s": time.time() - start_time,
                    "tokens/seen": iter_num * tokens_per_iter,
                }
                if grad_norm_item is not None:
                    log_dict["train/grad_norm"] = grad_norm_item
                if device_type == "cuda":
                    log_dict["perf/max_mem_allocated_mb"] = (
                        torch.cuda.max_memory_allocated() / 1e6
                    )
                    log_dict["perf/max_mem_reserved_mb"] = (
                        torch.cuda.max_memory_reserved() / 1e6
                    )
                wandb.log(log_dict, step=iter_num)
            if device_type == "cuda":
                torch.cuda.reset_peak_memory_stats()

        iter_num += 1

    run_success = True
except Exception:
    run_error = traceback.format_exc()
    raise
finally:
    if master_process:
        end_time = time.time()
        summary = {
            "contract_version": RUN_CONTRACT_VERSION,
            "run_id": run_id,
            "run_kind": "nanogpt_train",
            "ok": bool(run_success),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "elapsed_s": float(end_time - start_time),
            "iter_num": int(iter_num),
            "max_iters": int(max_iters),
            "tokens_per_iter": int(tokens_per_iter),
            "tokens_seen": int((iter_num - 1) * tokens_per_iter) if iter_num > 0 else 0,
            "best_val_loss": float(best_val_loss),
            "last_eval_iter": int(last_eval_iter) if last_eval_iter is not None else None,
            "last_eval": _json_safe(last_eval_losses) if last_eval_losses is not None else None,
            "ddp": bool(ddp),
            "world_size": int(ddp_world_size),
            "device": str(device),
            "dtype": str(dtype),
            "wandb": {
                "enabled": wandb is not None,
                "project": wandb_project,
                "group": wandb_group,
                "run_name": wandb_run_name,
                "job_type": wandb_job_type,
                "variant": wandb_variant,
                "tags": _json_safe(wandb_tags),
            },
        }
        if run_error:
            summary["error"] = run_error
        _atomic_write_json(out_dir_path / "summary.json", summary)

    if wandb is not None and master_process:
        try:
            wandb.finish()
        except Exception:
            pass

    if ddp:
        dist.destroy_process_group()
