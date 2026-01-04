# FineWeb10B with vRes + mHC (4 streams, 48 layers)
# ~20M param GPT-2 style model
#
# Combines:
#   - Value Residual (unconstrained lamb1/lamb2)
#   - mHC (manifold-constrained residual stream routing)
#
# Usage:
#   python train.py config/train_fineweb10B_vres_mhc_48l.py
#   torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_vres_mhc_48l.py

out_dir = "out-fineweb10B-vres-mhc-48l"
wandb_run_name = "vres-mhc-48l"
wandb_project = "mhc-nanogpt-48"

dataset = "fineweb10B"

# model
block_size = 1024
n_layer = 48
n_head = 6
n_embd = 150
dropout = 0.0
bias = False

batch_size = 24
gradient_accumulation_steps = 4
max_iters = 6666
eval_interval = 500
log_interval = 10
eval_iters = 100

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
warmup_iters = 200
lr_decay_iters = 6666
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# hyper-connections: mHC enabled (4 streams)
hc_num_streams = 4
hc_num_fracs = 1
hc_disable = False
mhc = True
sinkhorn_iters = 10
sinkhorn_tau = 0.05
mhc_h_res_proj = "sinkhorn"
ns_steps = 5
ns_eps = 1e-7
ns_coeffs = (3.0, -3.2, 1.2)

# value residual (unconstrained)
v_residual = True
v_residual_lamb_lr = 1e-2
