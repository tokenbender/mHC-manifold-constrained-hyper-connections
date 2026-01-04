import math

import torch
from torch import nn
import torch.nn.functional as F

from hyper_connections import get_init_and_expand_reduce_stream_functions
from value_residual import ValueResidualState


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


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.v_residual = config.v_residual
        self.v_residual_constrained = config.v_residual_constrained
        if self.v_residual:
            if self.v_residual_constrained:
                self.lamb_logits = nn.Parameter(torch.tensor([0.0, -8.0]))
            else:
                self.lamb1 = nn.Parameter(torch.tensor(0.5))
                self.lamb2 = nn.Parameter(torch.tensor(0.5))

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            bias = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer(
                "bias", bias.view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x, vrl_state=None):
        b, t, c = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(b, t, self.n_head, self.head_dim)
        k = k.view(b, t, self.n_head, self.head_dim)
        v = v.view(b, t, self.n_head, self.head_dim)

        if self.v_residual:
            if vrl_state is None:
                raise ValueError("v_residual requires vrl_state")
            if self.v_residual_constrained:
                lamb = F.softmax(self.lamb_logits, dim=-1)
                v = vrl_state.mix(v, lamb[0], lamb[1])
            else:
                v = vrl_state.mix(v, self.lamb1, self.lamb2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class AttnBranch(nn.Module):
    def __init__(self, norm, attn):
        super().__init__()
        self.norm = norm
        self.attn = attn

    def forward(self, x, vrl_state=None):
        x = self.norm(x)
        return self.attn(x, vrl_state=vrl_state)


class Block(nn.Module):
    def __init__(self, config, layer_idx, init_hc):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.attn_branch = AttnBranch(self.ln_1, self.attn)

        hc_kwargs = dict(
            mhc=config.mhc,
            sinkhorn_iters=config.sinkhorn_iters,
            sinkhorn_tau=config.sinkhorn_tau,
            mhc_h_res_proj=config.mhc_h_res_proj,
            ns_steps=config.ns_steps,
            ns_eps=config.ns_eps,
            ns_coeffs=config.ns_coeffs,
        )

        self.hc_attn = init_hc(
            dim=config.n_embd,
            branch=self.attn_branch,
            layer_index=layer_idx * 2,
            **hc_kwargs,
        )

        self.hc_mlp = init_hc(
            dim=config.n_embd,
            branch=nn.Sequential(self.ln_2, self.mlp),
            layer_index=layer_idx * 2 + 1,
            **hc_kwargs,
        )

    def forward(self, x, vrl_state=None):
        x = self.hc_attn(x, vrl_state=vrl_state)
        x = self.hc_mlp(x)
        return x


class GPTConfig:
    def __init__(self, **kwargs):
        self.block_size = kwargs.pop("block_size", 1024)
        self.vocab_size = kwargs.pop("vocab_size", 50304)
        self.n_layer = kwargs.pop("n_layer", 12)
        self.n_head = kwargs.pop("n_head", 12)
        self.n_embd = kwargs.pop("n_embd", 768)
        self.dropout = kwargs.pop("dropout", 0.0)
        self.bias = kwargs.pop("bias", True)

        self.hc_num_streams = kwargs.pop("hc_num_streams", 1)
        self.hc_num_fracs = kwargs.pop("hc_num_fracs", 1)
        self.hc_disable = kwargs.pop("hc_disable", False)
        self.mhc = kwargs.pop("mhc", False)
        self.sinkhorn_iters = kwargs.pop("sinkhorn_iters", 10)
        self.sinkhorn_tau = kwargs.pop("sinkhorn_tau", 0.05)
        self.mhc_h_res_proj = kwargs.pop("mhc_h_res_proj", "sinkhorn")
        self.ns_steps = kwargs.pop("ns_steps", NS_STEPS)
        self.ns_eps = kwargs.pop("ns_eps", 1e-7)
        self.ns_coeffs = kwargs.pop("ns_coeffs", NS_COEFFS)
        self.v_residual = kwargs.pop("v_residual", False)
        self.v_residual_constrained = kwargs.pop("v_residual_constrained", False)
        self.v_residual_lamb_lr = kwargs.pop("v_residual_lamb_lr", 1e-2)

        for key, value in kwargs.items():
            setattr(self, key, value)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.vrl_state = ValueResidualState() if config.v_residual else None

        init_hc, expand_stream, reduce_stream = (
            get_init_and_expand_reduce_stream_functions(
                config.hc_num_streams,
                num_fracs=config.hc_num_fracs,
                disable=config.hc_disable,
            )
        )

        self.expand_stream = expand_stream
        self.reduce_stream = reduce_stream

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config, i, init_hc) for i in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)
        x = self.expand_stream(x)

        vrl_state = self.vrl_state
        if vrl_state is not None:
            vrl_state.reset()

        for block in self.transformer.h:
            x = block(x, vrl_state=vrl_state)

        x = self.transformer.ln_f(x)
        x = self.reduce_stream(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name

                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, nn.Linear):
                    decay.add(full_name)
                else:
                    no_decay.add(full_name)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}
        lamb_params = {pn for pn in param_dict if "lamb" in pn}

        decay -= lamb_params
        no_decay -= lamb_params

        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay | lamb_params)) == 0

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if lamb_params:
            lamb_lr = self.config.v_residual_lamb_lr
            optim_groups.append(
                {
                    "params": [param_dict[pn] for pn in sorted(lamb_params)],
                    "weight_decay": 0.0,
                    "lr": lamb_lr,
                    "lr_scale": lamb_lr / learning_rate,
                }
            )

        use_fused = (
            device_type == "cuda"
        ) and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )

        return optimizer
