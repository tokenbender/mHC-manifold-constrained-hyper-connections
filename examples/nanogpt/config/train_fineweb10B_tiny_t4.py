# FineWeb10B tiny baseline config for quick controlled Colab T4 experiments.

out_dir = "out-t4-tiny-baseline"
wandb_run_name = "t4-tiny-baseline"
wandb_log = False

dataset = "fineweb10B"
data_loader = "memmap"

block_size = 128
n_layer = 2
n_head = 2
n_embd = 128
dropout = 0.0
bias = False

batch_size = 4
gradient_accumulation_steps = 8
max_iters = 1000
eval_interval = 50
log_interval = 5
eval_iters = 20

learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 50
lr_decay_iters = 1000
min_lr = 6e-5

dtype = "float16"
compile_model = False

hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True
mhc = False
