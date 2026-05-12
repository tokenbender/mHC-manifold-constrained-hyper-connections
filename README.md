## mHC (Manifold-Constrained Hyper-Connections)

Research implementation of **mHC** (DeepSeek; https://arxiv.org/abs/2512.24880) as a drop-in variant of **Hyper-Connections** (https://arxiv.org/abs/2409.19606).

### What we're building

A runnable PyTorch implementation of the mHC layer update

`x_{l+1} = H_l^{res} x_l + H_l^{post,T} F(H_l^{pre} x_l, W_l)`

with the key constraints:

- `H_res`: **doubly stochastic** (Birkhoff polytope; entries ≥ 0, rows sum to 1, cols sum to 1), via **Sinkhorn-Knopp**.
- `H_pre`, `H_post`: **non-negative** mixing maps.

### Implementation direction

Static per-layer matrices:
- learn `H_res_logits ∈ R^{s×s}` and project to `H_res` with Sinkhorn
- learn `H_pre_logits`, `H_post_logits` and map to non-negative weights (e.g. softmax)

This is a research prototype aimed at correctness + clarity, not the paper's systems optimizations.

### Running (nanoGPT on FineWeb10B)

Run from `examples/nanogpt/`. Adjust `--nproc_per_node` to match your GPU count.

**Recommended T4 comparison (single GPU, presentation workflow):**
```bash
MAX_ITERS=5000 EVAL_INTERVAL=500 EVAL_ITERS=50 \
BATCH_SIZE=8 GRAD_ACCUM=8 DEVICE=cuda DTYPE=float16 WANDB_LOG=False \
bash examples/nanogpt/run_t4_train_compare.sh
```

This trains the main comparison variants:
- baseline Transformer residual: `config/train_fineweb10B_t4.py`
- traditional HC: `config/train_fineweb10B_hc_t4.py`
- mHC: `config/train_fineweb10B_mhc_t4.py`

Each run writes local `metrics.jsonl` and `ckpt.pt` under its output directory.
Use `docs/T4_INFERENCE_BENCHMARK.md` for inference, benchmark, and summary
commands. This is a small-scale nanoGPT/FineWeb10B comparison, not a full paper
reproduction; do not use random or smoke checkpoints as model-quality evidence.
Paper benchmarks such as BBH/MMLU/GSM8K are not reproduced here.

**6-layer configs (~20M params):**
```bash
python train.py config/train_fineweb10B.py
python train.py config/train_fineweb10B_hc.py
python train.py config/train_fineweb10B_mhc.py
python train.py config/train_fineweb10B_vres.py
python train.py config/train_fineweb10B_vres_mhc.py
python train.py config/train_fineweb10B_cvres_mhc.py
```

**48-layer configs (~20M params):**
```bash
python train.py config/train_fineweb10B_48l.py
python train.py config/train_fineweb10B_hc_48l.py
python train.py config/train_fineweb10B_mhc_48l.py
python train.py config/train_fineweb10B_vres_48l.py
python train.py config/train_fineweb10B_vres_mhc_48l.py
python train.py config/train_fineweb10B_cvres_mhc_48l.py
```

**Multi-GPU example:**
```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_mhc_48l.py
```

#### Orthostochastic mHC option
mHC supports an orthostochastic H_res projection via Newton-Schulz. Set `mhc_h_res_proj = "orthostochastic"` in your config.

By default, configs use fixed Newton-Schulz coefficients (`ns_steps=5`, `ns_coeffs=(3.0, -3.2, 1.2)`). For research, `ns_coeffs` can also be a per-step schedule (tuple of `(a, b, c)` triplets); set `ns_steps = len(ns_coeffs)`.

#### Residual identity-mix (optional)
For an ablation that keeps residual routing close to identity, enable:
- `mhc_residual_identity_mix = True`
- `mhc_residual_alpha = 0.01`

This applies `H_res = (1-α) * I + α * S` where `S` is the projected matrix (Sinkhorn or orthostochastic) and `α` is learned.

#### Value residual (vRes) notes
- `train_fineweb10B_vres*.py` enables value residual only.
- `train_fineweb10B_vres_mhc*.py` combines vRes + mHC.
- `train_fineweb10B_cvres_mhc*.py` combines vRes + mHC with `v_residual_constrained=True` (convex mixing via softmax).

### Implemented research
- Value residual ablations with baseline/HC/mHC
- Combined vRes + mHC configs (unconstrained + constrained)
- H^res = `(1−α)*I + α*S` instead of full doubly stochastic
- Orthostochastic H_res projection (Newton-Schulz) as alternative to Sinkhorn-Knopp
- Opt-in Newton-Schulz coefficient schedule for orthostochastic projection


### Acknowledgements

Built using code snippets from `nanogpt`, `lucidrains/hyper-connections` and my own mHC implementation.

### License

Apache 2.0
