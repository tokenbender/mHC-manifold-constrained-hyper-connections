# FineWeb10B baseline config for one NVIDIA T4 (16GB).
#
# T4 supports FP16 well; BF16 should not be assumed. This config keeps the same
# model width/depth as the existing small FineWeb10B configs, but lowers the
# per-step batch for single-GPU training.

out_dir = "out-t4-baseline"
wandb_run_name = "t4-baseline"
wandb_log = False

dataset = "fineweb10B"

# model: comparable across baseline / HC / mHC T4 configs
block_size = 1024
n_layer = 6
n_head = 6
n_embd = 288
dropout = 0.0
bias = False

# training: safe defaults for a 16GB T4; override from helper script as needed
batch_size = 8
gradient_accumulation_steps = 8
max_iters = 5000
eval_interval = 500
log_interval = 10
eval_iters = 50

learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

dtype = "float16"
compile_model = False

# baseline residual path
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True
mhc = False
