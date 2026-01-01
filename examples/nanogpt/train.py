"""
Training script for nanoGPT with HyperConnections.

Supports:
- Single GPU or multi-GPU via torchrun
- bf16/fp16 AMP
- Shakespeare char-level (legacy) or FineWeb token-level data

Usage:
    # Single GPU, Shakespeare
    python train.py config/train_shakespeare_char.py

    # Single GPU, FineWeb
    python train.py config/train_fineweb10B.py

    # Multi-GPU (4x), FineWeb
    torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B.py

    # If NCCL fails (no InfiniBand), try:
    NCCL_IB_DISABLE=1 torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B.py
"""

import glob
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

# dataset: "shakespeare_char" or "fineweb10B"
dataset = "shakespeare_char"

# hyper-connections config
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True

# dtype: "float32", "bfloat16", "float16"
dtype = "bfloat16"

# torch.compile (requires PyTorch 2.0+)
compile_model = False

# wandb logging
wandb_log = True
wandb_project = "mhc-nanogpt"
wandb_run_name = "baseline"

# DDP backend: "nccl", "gloo", etc.
# If NCCL fails, set NCCL_IB_DISABLE=1 or use backend="gloo"
backend = "nccl"

# -----------------------------------------------------------------------------
# load config file if provided
exec(open(os.path.join(os.path.dirname(__file__), "configurator.py")).read())

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

if dataset == "fineweb10B":
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

else:
    # Shakespeare char-level (legacy)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    meta_path = os.path.join(data_dir, "meta.json")

    train_data = torch.load(train_path, weights_only=True)
    val_data = torch.load(val_path, weights_only=True)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]

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
)

model = GPT(model_config)
model.to(device)

if compile_model:
    print("Compiling model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

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


# -----------------------------------------------------------------------------
# Evaluation


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


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
            "hc_disable": hc_disable,
            "dtype": dtype,
            "world_size": ddp_world_size,
            "tokens_per_iter": tokens_per_iter,
        },
    )

while iter_num <= max_iters:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluation
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {"val/loss": losses["val"], "train/loss_eval": losses["train"]},
                step=iter_num,
            )
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
    if grad_clip != 0.0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

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
            wandb.log(
                {
                    "train/loss": loss_item,
                    "train/lr": lr,
                    "perf/tok_per_sec": tokens_per_sec,
                    "perf/iter_time_ms": dt * 1000,
                },
                step=iter_num,
            )

    iter_num += 1

# -----------------------------------------------------------------------------
# Cleanup

if wandb_log and master_process:
    wandb.finish()

if ddp:
    dist.destroy_process_group()
