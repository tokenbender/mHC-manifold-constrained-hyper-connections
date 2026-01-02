from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange
import math

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# helper functions


def exists(v):
    return v is not None


def divisible_by(num, den):
    return (num % den) == 0


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def add(x, y):
    return x + y


def sinkhorn_log(logits, num_iters=10, tau=0.05):
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.full(
        (n,), -math.log(n), device=logits.device, dtype=logits.dtype
    )

    u = torch.zeros(n, device=Z.device, dtype=Z.dtype)
    v = torch.zeros(n, device=Z.device, dtype=Z.dtype)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(1), dim=0)

    return torch.exp(Z + u.unsqueeze(1) + v.unsqueeze(0)) * n


# main functions


def get_expand_reduce_stream_functions(
    num_streams, add_stream_embed=False, dim=None, disable=False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), (
            "`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added"
        )

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams=True)
    else:
        expand_fn = Reduce(
            pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams
        )

    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)

    return expand_fn, reduce_fn


def get_init_and_expand_reduce_stream_functions(
    num_streams, num_fracs=1, dim=None, add_stream_embed=False, disable=None
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs=num_fracs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# norms


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


# main classes

# residual base class


class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs,
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,
    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


# hyper connection residual streams


class HyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index=None,
        tanh=True,
        channel_first=False,
        dropout=0.0,
        residual_transform: Module
        | None = None,  # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual=True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views=1,  # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn=add,
        num_fracs=1,  # https://arxiv.org/abs/2503.14125
        mhc=False,
        sinkhorn_iters=10,
        sinkhorn_tau=0.05,
        mhc_residual_identity_mix=False,
        mhc_residual_alpha=0.01,
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch

        self.act = nn.Tanh() if tanh else nn.Identity()

        # frac-connections paper - num_fracs > 1 will be the `m` in their paper https://arxiv.org/abs/2503.14125

        assert num_fracs >= 1

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")

        assert divisible_by(dim, num_fracs), (
            f"feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})"
        )

        dim //= num_fracs  # effective dim handled in dimension is feature dimension divided by num fractions

        # they used layernorm in paper, but rmsnorm is fine given what we know now

        self.norm = RMSNorm(dim)

        assert num_residual_streams > 0, "`num_residual_streams` must be greater than 0"

        self.num_residual_streams = num_residual_streams
        init_residual_index = (
            default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        )  # just choose one random residual stream if layer index not given

        # handle the parameter dimensions, which may require (num_residuals x num_fractions) - generalizing hyper + frac connections

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        # width num residual streams

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # width connection

        init_alpha0 = torch.zeros((num_residual_streams_fracs, num_input_views_fracs))
        init_alpha0[init_residual_index, :] = 1.0

        self.static_alpha = nn.Parameter(
            cat((init_alpha0, torch.eye(num_residual_streams_fracs)), dim=1)
        )

        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(dim, num_residual_streams_fracs + num_input_views_fracs)
        )
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # depth connection related (beta)

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            self.static_beta = nn.Parameter(torch.ones(num_residual_streams_fracs))

            dynamic_beta_shape = (
                (dim,) if num_fracs == 1 else (dim, num_fracs)
            )  # preserve backwards compat
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))

            self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # dropouts

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM

        self.depth_residual_fn = depth_residual_fn

        self.mhc = mhc
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.mhc_residual_identity_mix = mhc_residual_identity_mix

        if mhc:
            assert num_fracs == 1, "mhc currently requires num_fracs = 1"
            assert num_input_views == 1, "mhc currently requires num_input_views = 1"

            H_res_init = torch.full((num_residual_streams, num_residual_streams), -8.0)
            H_res_init.fill_diagonal_(0.0)
            self.H_res_logits = nn.Parameter(H_res_init)

            H_pre_init = torch.full((num_residual_streams,), -8.0)
            H_pre_init[init_residual_index] = 0.0
            self.H_pre_logits = nn.Parameter(H_pre_init)

            if add_branch_out_to_residual:
                self.H_post_logits = nn.Parameter(torch.zeros(num_residual_streams))

            if mhc_residual_identity_mix:
                alpha_clamped = max(1e-4, min(1 - 1e-4, mhc_residual_alpha))
                alpha_logit_init = math.log(alpha_clamped / (1 - alpha_clamped))
                self.H_res_alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init))

    def width_connection(self, residuals):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)

        # width connection

        # handle channel first

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        # split out fractions

        residuals = self.split_fracs(residuals)

        # split out streams

        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

        if self.mhc:
            residuals_mixed_source = maybe_transformed_residuals

            if self.channel_first:
                residuals_mixed_source = rearrange(
                    residuals_mixed_source, "b d ... -> b ... d"
                )

            residuals_mixed_source = self.split_fracs(residuals_mixed_source)
            residuals_mixed_source = rearrange(
                residuals_mixed_source, "(b s) ... d -> b ... s d", s=streams
            )

            S = sinkhorn_log(self.H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau)

            if self.mhc_residual_identity_mix:
                alpha = torch.sigmoid(self.H_res_alpha_logit)
                I = torch.eye(streams, device=S.device, dtype=S.dtype)
                H_res = (1 - alpha) * I + alpha * S
            else:
                H_res = S

            H_pre = F.softmax(self.H_pre_logits, dim=-1)

            H_post = None
            if self.add_branch_out_to_residual:
                H_post = F.softmax(self.H_post_logits, dim=-1)

            residuals_mixed = einsum(
                H_res, residuals_mixed_source, "s t, ... s d -> ... t d"
            )
            branch_input = einsum(H_pre, residuals, "s, ... s d -> ... d")

            if getattr(self, "collect_stats", False):
                with torch.no_grad():
                    stats = dict(
                        h_res_min=H_res.min(),
                        h_res_row_sum=H_res.sum(dim=-1).mean(),
                        h_res_col_sum=H_res.sum(dim=-2).mean(),
                        h_pre_min=H_pre.min(),
                    )
                    if H_post is not None:
                        stats["h_post_min"] = H_post.min()
                    if self.mhc_residual_identity_mix:
                        stats["h_res_alpha"] = torch.sigmoid(self.H_res_alpha_logit)
                    self.last_stats = {k: v.detach() for k, v in stats.items()}

            if self.channel_first:
                branch_input = rearrange(branch_input, "b ... d -> b d ...")

            branch_input = self.merge_fracs(branch_input)

            return (
                branch_input,
                maybe_transformed_residuals,
                dict(beta=H_post, residuals_mixed=residuals_mixed),
            )

        # norm

        normed = self.norm(residuals)

        # alpha for weighted sum of residuals going into branch

        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale

        static_alpha = rearrange(self.static_alpha, "(f s) d -> f s d", s=streams)

        alpha = dynamic_alpha + static_alpha

        alpha = self.split_fracs(
            alpha
        )  # (batch, seq, fracs1, streams, fracs2, input + residual streams)

        # beta for weights from branch output back to residual streams

        beta = None

        if self.add_branch_out_to_residual:
            dc_weight = self.act(normed @ self.dynamic_beta_fn)

            if not self.has_fracs:
                dc_weight = rearrange(dc_weight, "... -> ... 1")

            dynamic_beta = dc_weight * self.dynamic_beta_scale

            static_beta = rearrange(self.static_beta, "... (s f) -> ... s f", s=streams)

            beta = dynamic_beta + static_beta

        if getattr(self, "collect_stats", False):
            with torch.no_grad():
                num_input_views_fracs = self.num_input_views * self.num_fracs
                alpha_branch = alpha[..., :num_input_views_fracs]
                alpha_residual = alpha[..., num_input_views_fracs:]
                alpha_branch_abs_mean = alpha_branch.abs().mean()
                alpha_residual_abs_mean = alpha_residual.abs().mean()
                stats = dict(
                    alpha_branch_mean=alpha_branch.mean(),
                    alpha_branch_abs_mean=alpha_branch_abs_mean,
                    alpha_residual_mean=alpha_residual.mean(),
                    alpha_residual_abs_mean=alpha_residual_abs_mean,
                    alpha_branch_residual_ratio=alpha_branch_abs_mean
                    / (alpha_residual_abs_mean + 1e-8),
                )
                if beta is not None:
                    stats.update(
                        beta_mean=beta.mean(),
                        beta_abs_mean=beta.abs().mean(),
                        beta_min=beta.min(),
                        beta_max=beta.max(),
                    )
                self.last_stats = {k: v.detach() for k, v in stats.items()}

        mix_h = einsum(alpha, residuals, "... f1 s f2 t, ... f1 s d -> ... f2 t d")

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = (
                mix_h[..., : self.num_input_views, :],
                mix_h[..., self.num_input_views :, :],
            )
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        # maybe merge fractions back

        branch_input = self.merge_fracs(branch_input)

        return branch_input, maybe_transformed_residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta, residuals_mixed=None):
        assert self.add_branch_out_to_residual

        # maybe split fractions

        branch_output = self.split_fracs(branch_output)

        # 'depth' connection

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        if self.mhc:
            assert residuals_mixed is not None
            assert beta is not None

            branch_to_streams = einsum(branch_output, beta, "b ... d, s -> b ... s d")
            output = residuals_mixed + branch_to_streams
            output = rearrange(output, "b ... s d -> (b s) ... d")

            output = self.merge_fracs(output)

            if self.channel_first:
                output = rearrange(output, "b ... d -> b d ...")

            return self.dropout(output)

        output = einsum(
            branch_output, beta, "b ... f1 d, b ... f1 s f2 -> b ... f2 s d"
        )

        output = rearrange(output, "b ... s d -> (b s) ... d")

        # merge merge back fractions

        output = self.merge_fracs(output)

        # channel first

        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


HyperConnections.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)

# stream embed


class StreamEmbed(Module):
    def __init__(self, num_streams, dim, channel_first=False, expand_to_streams=False):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):
        if self.expand_to_streams:
            residuals = repeat(residuals, "b ... -> (b s) ...", s=self.num_streams)

        if self.channel_first:
            residuals = rearrange(
                residuals, "(b s) d ... -> b ... s d", s=self.num_streams
            )
        else:
            residuals = rearrange(
                residuals, "(b s) ... d -> b ... s d", s=self.num_streams
            )

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(
                residuals, "b ... s d -> (b s) d ...", s=self.num_streams
            )
        else:
            residuals = rearrange(
                residuals, "b ... s d -> (b s) ... d", s=self.num_streams
            )

        return residuals


# attention pool - taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else


class AttentionPoolReduceStream(Module):
    def __init__(self, num_streams, dim, channel_first=False):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias=False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):
        if self.channel_first:
            residuals = rearrange(
                residuals, "(b s) d ... -> b ... s d", s=self.num_streams
            )
        else:
            residuals = rearrange(
                residuals, "(b s) ... d -> b ... s d", s=self.num_streams
            )

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim=-2)

        residuals = reduce(residuals * attn, "b ... s d -> b ... d", "sum")

        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")

        return residuals
