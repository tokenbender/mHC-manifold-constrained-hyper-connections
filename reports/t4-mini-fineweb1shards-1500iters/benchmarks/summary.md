# Benchmark Summary

| variant | checkpoint_path | config_path | model_parameter_count | dtype | device | gpu_name | prefill_latency_ms_mean | full_context_generation_latency_ms_mean | full_context_generation_latency_per_token_ms_mean | tokens_per_sec_mean | peak_vram_mb_max | peak_vram_mb_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | examples/nanogpt/out-t4-mini-baseline/ckpt.pt | examples/nanogpt/config/train_fineweb10B_mini_t4.py | 11478720 | float16 | cuda | Tesla T4 | 3.4053852499710047 | 103.75285244995212 | 3.2422766390610036 | 317.3386710849443 | 61.4765625 | 61.4765625 |
| hc | examples/nanogpt/out-t4-mini-hc/ckpt.pt | examples/nanogpt/config/train_fineweb10B_hc_mini_t4.py | 11489680 | float16 | cuda | Tesla T4 | 10.193662850042529 | 333.78190914995685 | 10.430684660936151 | 97.53349122947444 | 61.515625 | 61.515625 |
| mhc | examples/nanogpt/out-t4-mini-mhc/ckpt.pt | examples/nanogpt/config/train_fineweb10B_mhc_mini_t4.py | 11489872 | float16 | cuda | Tesla T4 | 26.056962600023326 | 827.1162536000702 | 25.847382925002194 | 39.24325161541602 | 61.52734375 | 61.52734375 |
