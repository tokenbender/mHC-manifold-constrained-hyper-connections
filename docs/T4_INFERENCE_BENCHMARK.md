# T4 Inference and Benchmark Guide

This guide turns the nanoGPT/mHC research code into a minimal inference and
benchmark workflow for an NVIDIA T4 GPU.

## What mHC Changes

Standard residual connections pass one residual stream through each layer:

```text
x_{l+1} = x_l + F(x_l)
```

Hyper-Connections (HC) widen the residual stream into multiple streams and learn
how to read, mix, and write those streams. This increases routing flexibility but
can weaken the identity-mapping behavior that makes residual networks stable.

Manifold-Constrained Hyper-Connections (mHC) keep the widened HC idea, but
constrain the residual mixing matrix. In this repo, `H_res` is projected with
Sinkhorn to be doubly stochastic, while `H_pre` and `H_post` use non-negative
softmax weights. The practical goal is to keep residual-stream routing better
conditioned while still comparing baseline, HC, mHC, vRes, and vRes+mHC variants.

## Implemented Inference Workflow

New files:

- `examples/nanogpt/infer.py`: text generation from a `train.py` checkpoint.
- `examples/nanogpt/benchmark_inference.py`: synthetic-token inference benchmark.
- `examples/nanogpt/run_t4_benchmarks.sh`: multi-variant T4 helper.

The implementation is intentionally direct PyTorch. It does not add KV cache,
vLLM, TensorRT, custom CUDA kernels, or training changes.

## Run Inference

Use a trained checkpoint from `examples/nanogpt/train.py`:

```bash
python examples/nanogpt/infer.py \
  --ckpt examples/nanogpt/out-fineweb10B/ckpt.pt \
  --config examples/nanogpt/config/train_fineweb10B.py \
  --device cuda \
  --dtype float16 \
  --compile false \
  --prompt "The future of machine learning is" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 200 \
  --seed 1337
```

T4 recommendation: use `--dtype float16`. Do not default to `bfloat16` on T4.

The CLI prints generated text plus device, dtype, parameter count, prompt length,
generated token count, elapsed time, tokens/sec, and peak CUDA memory.

## Run One Benchmark

```bash
python examples/nanogpt/benchmark_inference.py \
  --ckpt examples/nanogpt/out-fineweb10B/ckpt.pt \
  --config examples/nanogpt/config/train_fineweb10B.py \
  --device cuda \
  --dtype float16 \
  --batch-size 1 \
  --prompt-len 128 \
  --gen-len 32 \
  --num-warmup 5 \
  --num-iters 20 \
  --compile false \
  --output-json benchmarks/baseline.json \
  --output-csv benchmarks/baseline.csv \
  --seed 1337
```

Benchmark input is synthetic token IDs. No dataset download is needed.

## Compare Variants

Set checkpoint paths for the variants you have. Missing checkpoints are skipped.

```bash
CKPT_BASELINE=examples/nanogpt/out-fineweb10B/ckpt.pt \
CKPT_HC=examples/nanogpt/out-fineweb10B-hc/ckpt.pt \
CKPT_MHC=examples/nanogpt/out-fineweb10B-mhc/ckpt.pt \
CKPT_VRES=examples/nanogpt/out-fineweb10B-vres/ckpt.pt \
CKPT_VRES_MHC=examples/nanogpt/out-fineweb10B-vres-mhc/ckpt.pt \
DEVICE=cuda DTYPE=float16 BATCH_SIZE=1 PROMPT_LEN=128 GEN_LEN=32 NUM_ITERS=20 COMPILE=false \
bash examples/nanogpt/run_t4_benchmarks.sh
```

Outputs are written to `benchmarks/t4-<timestamp>/` as JSON and CSV.

## Metrics

- `prefill_latency_ms`: time for one forward pass over the prompt tokens.
- `decode_latency_per_token_ms`: average time to produce one new token in the
  simple full-context nanoGPT loop.
- `end_to_end_latency_ms`: time to generate `gen_len` tokens from the prompt.
- `tokens_per_sec`: generated tokens per second, including batch size.
- `peak_vram_mb`: peak CUDA memory allocated during timed benchmark work.

The decode loop has no KV cache, so it reprocesses context each step. Treat this
as a research-code comparison, not a production serving benchmark.

## Presentation Interpretation

For a university presentation, separate three claims:

1. Correctness of pipeline: CLI loads checkpoint, runs generation, records metrics.
2. Performance comparison: baseline vs HC vs mHC latency, throughput, and VRAM.
3. Model quality: only claim quality if checkpoint was truly trained and evaluated.

If using random weights or a very small smoke checkpoint, label every output as:

```text
pipeline smoke test only; no model-quality claim
```

Good comparison table columns:

```text
variant | checkpoint | params | prefill ms | decode ms/token | tokens/sec | peak VRAM MB
```

## Known Limitations

- Research implementation, optimized for clarity more than serving speed.
- No KV cache; decode is slower than production LLM serving.
- Not vLLM/TensorRT optimized.
- Benchmark results depend on model size, checkpoint, prompt length, generation
  length, PyTorch version, and GPU state.
- T4 is FP16-friendly; BF16 should not be assumed.
- Speed benchmarks can use any checkpoint, but quality claims require a real
  trained checkpoint.

## Suggested Slide Outline

1. Problem: residual connections vs wider Hyper-Connections.
2. mHC idea: constrain residual stream mixing with doubly stochastic `H_res`.
3. Project setup: nanoGPT variants, T4 GPU, FP16 inference.
4. Inference CLI: checkpoint loading, prompt generation, metrics.
5. Benchmark method: warmups, timed iterations, CUDA synchronization, JSON/CSV.
6. Results table: baseline vs HC vs mHC vs vRes vs vRes+mHC.
7. Interpretation: latency/throughput/VRAM tradeoffs and honesty boundary.
8. Limitations and future work: KV cache, TensorRT/vLLM, trained checkpoints.
