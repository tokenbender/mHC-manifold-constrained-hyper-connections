"""
Train nanoGPT with HyperConnections.

Usage:
    python train.py config/train_fineweb10B.py
    torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B.py
"""

import glob
import math
import os
import time
from contextlib import nullcontext

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
ns_steps = NS_STEPS
ns_eps = 1e-7
ns_coeffs = NS_COEFFS

# value residual config
v_residual = False
v_residual_lamb_lr = 1e-2

# dtype: "float32", "bfloat16", "float16"
dtype = "bfloat16"

# torch.compile (requires PyTorch 2.0+)
compile_model = False

# wandb logging
wandb_log = True
wandb_project = "mhc-nanogpt"
wandb_run_name = "baseline"
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

# -----------------------------------------------------------------------------
# DDP setup

ddp = int(os.environ.get("RANK", -1)) != -1

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = (
    device.type
    if isinstance(device, torch.device)
    else ("cuda" if "cuda" in device else "cpu")
)

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
    v_residual=v_residual,
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
    import wandb

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
            "v_residual": v_residual,
            "v_residual_lamb_lr": v_residual_lamb_lr,
            "dtype": dtype,
            "world_size": ddp_world_size,
            "tokens_per_iter": tokens_per_iter,
            "wandb_log_layer_stats": wandb_log_layer_stats,
            "wandb_log_layer_cosine": wandb_log_layer_cosine,
        },
    )

start_time = time.time()

while iter_num <= max_iters:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        lr_scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * lr_scale

    # evaluation
    if iter_num % eval_interval == 0 and master_process:
        losses, layer_cosine = estimate_loss()
        print(
            f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
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
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

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
        print(
            f"iter {iter_num}: loss {loss_item:.4f}, lr {lr:.2e}, "
            f"time {dt * 1000:.0f}ms, tok/s {tokens_per_sec:.0f}"
        )
        if wandb_log:
            log_dict = {
                "train/loss": loss_item,
                "train/lr": lr,
                "perf/tok_per_sec": tokens_per_sec,
                "perf/iter_time_ms": dt * 1000,
                "perf/elapsed_s": time.time() - start_time,
                "tokens/seen": iter_num * tokens_per_iter,
            }
            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm.item()
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

# -----------------------------------------------------------------------------
# Cleanup

if wandb_log and master_process:
    wandb.finish()

if ddp:
    dist.destroy_process_group()
