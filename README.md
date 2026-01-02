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

**Baseline** (no hyper-connections):

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B.py
```

**HC** (vanilla Hyper-Connections):

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_hc.py
```

**mHC** (Manifold-Constrained Hyper-Connections):

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_mhc.py
```

Run from `examples/nanogpt/`. Adjust `--nproc_per_node` to match your GPU count.

### Next steps planned
- [x] Value residual ablations with baseline/HC/mHC
- [ ] AltUP ablation
- [ ] H^res = `(1−α)*I + α*S` instead of full doubly stochastic
- [ ] Replace sinkhorn-knopp w/ Muon's orthogonalization op
- [ ] U-net-based variants + value embeddings


### Acknowledgements

Built using code snippets from `nanogpt`, `lucidrains/hyper-connections` and my own mHC implementation.

### License

Apache 2.0