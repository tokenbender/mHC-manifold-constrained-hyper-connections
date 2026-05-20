# FineWeb10B mini manifold-constrained Hyper-Connections config for Colab T4.

out_dir = "out-t4-mini-mhc"
wandb_run_name = "t4-mini-mhc"
wandb_log = False

dataset = "fineweb10B"
data_loader = "memmap"

block_size = 256
n_layer = 4
n_head = 4
n_embd = 192
dropout = 0.0
bias = False

batch_size = 2
gradient_accumulation_steps = 16
max_iters = 1500
eval_interval = 75
log_interval = 5
eval_iters = 20

learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 75
lr_decay_iters = 1500
min_lr = 6e-5

dtype = "float16"
compile_model = False

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
mhc_residual_identity_mix = False
mhc_residual_alpha = 0.01
